import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
import numpy as np


class GraphNeuralNetwork(nn.Module):
    """
    Highly optimized neural network that preserves exact graph topology.
    Uses vectorized scatter/gather operations instead of loops.
    """
    def __init__(self, graph_network, input_neurons, output_neurons, 
                 activation='relu', device='cpu', prune_dead=True,
                 use_mixed_precision=False):
        super(GraphNeuralNetwork, self).__init__()
        
        self.graph = graph_network
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        
        # Create pruned copies
        self._adj = {n: set(children) for n, children in self.graph.adjacency.items()}
        self._topo_order = list(self.graph.topo_order)
        self._position = {n: i for i, n in enumerate(self._topo_order)}

        if prune_dead:
            self._prune_dead_neurons()
        
        self._validate_structure()
        
        # Build vectorized sparse structures
        self._build_vectorized_structures()
        
        self.activation_fn = self._get_activation(activation)
        self.to(self.device)
        
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler()
    
    def _prune_dead_neurons(self):
        """Prune neurons not reachable from inputs or not reaching outputs."""
        name_to_node = {n.name: n for n in self._adj.keys()}

        input_nodes = [name_to_node[name] for name in self.input_neurons]
        output_nodes = [name_to_node[name] for name in self.output_neurons]

        rev = {n: set() for n in self._adj.keys()}
        for p, childs in self._adj.items():
            for c in childs:
                if c in rev:
                    rev[c].add(p)

        reachable_from_inputs = set()
        stack = list(input_nodes)
        while stack:
            cur = stack.pop()
            if cur in reachable_from_inputs:
                continue
            reachable_from_inputs.add(cur)
            for ch in self._adj.get(cur, ()):
                if ch not in reachable_from_inputs:
                    stack.append(ch)

        can_reach_outputs = set()
        stack = list(output_nodes)
        while stack:
            cur = stack.pop()
            if cur in can_reach_outputs:
                continue
            can_reach_outputs.add(cur)
            for p in rev.get(cur, ()):
                if p not in can_reach_outputs:
                    stack.append(p)

        keep = reachable_from_inputs & can_reach_outputs
        keep.update(input_nodes)
        keep.update(output_nodes)

        self._adj = {n: {c for c in childs if c in keep} 
                     for n, childs in self._adj.items() if n in keep}
        self._topo_order = [n for n in self._topo_order if n in keep]
        self._position = {n: i for i, n in enumerate(self._topo_order)}
    
    def _build_vectorized_structures(self):
        """Build fully vectorized structures using scatter/gather operations."""
        # Map neurons to indices
        self.neuron_to_idx = {node.name: i for i, node in enumerate(self._topo_order)}
        self.idx_to_neuron = {i: name for name, i in self.neuron_to_idx.items()}
        self.num_neurons = len(self.neuron_to_idx)
        
        # Compute layers
        self.layers = self._compute_layers()
        
        # Build vectorized layer processing structures
        self.layer_params = nn.ModuleList()
        
        for layer_idx in range(len(self.layers) - 1):
            prev_layer = self.layers[layer_idx]
            curr_layer = self.layers[layer_idx + 1]
            
            # Collect all edges for this layer transition
            edge_list = []
            for curr_local_idx, curr_neuron_idx in enumerate(curr_layer):
                for prev_local_idx, prev_neuron_idx in enumerate(prev_layer):
                    prev_node = self._topo_order[prev_neuron_idx]
                    curr_node = self._topo_order[curr_neuron_idx]
                    
                    if curr_node in self._adj.get(prev_node, set()):
                        edge_list.append((curr_local_idx, prev_local_idx))
            
            # Create layer parameters
            layer_module = nn.Module()
            
            if edge_list:
                num_edges = len(edge_list)
                edge_tensor = torch.tensor(edge_list, dtype=torch.long).t()
                
                # Register as buffer (non-trainable)
                layer_module.register_buffer('edge_index', edge_tensor)
                
                # Weights for each edge
                weights = nn.Parameter(torch.randn(num_edges) * np.sqrt(2.0 / (len(prev_layer) + len(curr_layer))))
                layer_module.register_parameter('weights', weights)
            else:
                # No edges (shouldn't happen)
                layer_module.register_buffer('edge_index', torch.zeros(2, 0, dtype=torch.long))
                layer_module.register_parameter('weights', nn.Parameter(torch.zeros(0)))
            
            # Bias for each neuron in current layer
            layer_module.register_parameter('bias', nn.Parameter(torch.zeros(len(curr_layer))))
            
            self.layer_params.append(layer_module)
        
        # Store layer sizes
        self.layer_sizes = [len(layer) for layer in self.layers]
        
        # Input/output indices
        self.input_indices = [self.neuron_to_idx[name] for name in self.input_neurons]
        self.output_indices = [self.neuron_to_idx[name] for name in self.output_neurons]
    
    def _compute_layers(self):
        """Group neurons into layers that can be computed in parallel."""
        layers = []
        processed = set()
        
        # Build parent lookup
        parents_per_neuron = [[] for _ in range(self.num_neurons)]
        for parent_node, children in self._adj.items():
            parent_idx = self.neuron_to_idx[parent_node.name]
            for child_node in children:
                child_idx = self.neuron_to_idx[child_node.name]
                parents_per_neuron[child_idx].append(parent_idx)
        
        # Start with input neurons
        current_layer = [self.neuron_to_idx[name] for name in self.input_neurons]
        layers.append(current_layer)
        processed.update(current_layer)
        
        # Build subsequent layers
        while len(processed) < self.num_neurons:
            next_layer = []
            for node_idx in range(self.num_neurons):
                if node_idx in processed:
                    continue
                if all(p in processed for p in parents_per_neuron[node_idx]):
                    next_layer.append(node_idx)
            
            if not next_layer:
                break
            
            layers.append(next_layer)
            processed.update(next_layer)
        
        return layers
    
    def _get_activation(self, activation):
        """Get PyTorch activation function."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _validate_structure(self):
        """Validate network structure."""
        adj = self._adj
        
        for name in self.input_neurons + self.output_neurons:
            if name not in [node.name for node in adj.keys()]:
                raise ValueError(f"Neuron '{name}' not found in graph")
        
        for name in self.input_neurons:
            node = next((n for n in adj.keys() if n.name == name), None)
            if node:
                parents = [p for p, children in adj.items() if node in children]
                if parents:
                    raise ValueError(f"Input neuron '{name}' has incoming edges")
        
        for name in self.output_neurons:
            node = next((n for n in adj.keys() if n.name == name), None)
            if node and len(adj[node]) > 0:
                raise ValueError(f"Output neuron '{name}' has outgoing edges")
    
    def forward_batch(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized forward pass using scatter_add for aggregation.
        Preserves exact graph topology while being GPU-efficient.
        
        :param input_tensor: [batch_size, num_inputs]
        :return: [batch_size, num_outputs]
        """
        batch_size = input_tensor.shape[0]
        layer_output = input_tensor
        
        for layer_idx, layer_module in enumerate(self.layer_params):
            curr_layer_size = self.layer_sizes[layer_idx + 1]
            
            # Initialize with bias
            next_output = layer_module.bias.unsqueeze(0).expand(batch_size, -1).clone()
            
            # Get edge information
            edge_index = layer_module.edge_index
            weights = layer_module.weights
            
            if edge_index.shape[1] > 0:
                # Vectorized edge processing using advanced indexing
                # edge_index[0]: target neuron indices in current layer
                # edge_index[1]: source neuron indices in previous layer
                
                # Get all source activations: [batch_size, num_edges]
                source_activations = layer_output[:, edge_index[1]]
                
                # Multiply by weights: [batch_size, num_edges]
                weighted_activations = source_activations * weights.unsqueeze(0)
                
                # Aggregate to target neurons using scatter_add
                # This sums all incoming edges for each neuron
                next_output.scatter_add_(
                    1,
                    edge_index[0].unsqueeze(0).expand(batch_size, -1),
                    weighted_activations
                )
            
            # Apply activation
            layer_output = self.activation_fn(next_output)
        
        return layer_output
    
    def forward(self, input_dict: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Single sample forward pass (legacy compatibility)."""
        input_tensor = torch.zeros(1, len(self.input_neurons), device=self.device)
        for i, name in enumerate(self.input_neurons):
            input_tensor[0, i] = input_dict.get(name, 0.0)
        
        # Get all layer activations
        layer_activations = [input_tensor]
        layer_output = input_tensor
        
        for layer_module in self.layer_params:
            curr_layer_size = layer_module.bias.shape[0]
            next_output = layer_module.bias.unsqueeze(0).clone()
            
            edge_index = layer_module.edge_index
            weights = layer_module.weights
            
            if edge_index.shape[1] > 0:
                source_activations = layer_output[:, edge_index[1]]
                weighted_activations = source_activations * weights.unsqueeze(0)
                next_output.scatter_add_(1, edge_index[0].unsqueeze(0), weighted_activations)
            
            layer_output = self.activation_fn(next_output)
            layer_activations.append(layer_output)
        
        # Map back to neuron names
        neuron_values = {}
        for layer_idx, layer_indices in enumerate(self.layers):
            if layer_idx < len(layer_activations):
                for local_idx, global_idx in enumerate(layer_indices):
                    neuron_name = self.idx_to_neuron[global_idx]
                    neuron_values[neuron_name] = layer_activations[layer_idx][0, local_idx].unsqueeze(0)
        
        return neuron_values
    
    def train_step_batch(self, input_batch: torch.Tensor, target_batch: torch.Tensor,
                        optimizer, criterion=nn.MSELoss()):
        """Optimized batch training step."""
        optimizer.zero_grad()
        
        if self.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                outputs = self.forward_batch(input_batch)
                loss = criterion(outputs, target_batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = self.forward_batch(input_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def train_step(self, inputs: Dict[str, float], targets: Dict[str, float],
                   optimizer, criterion=nn.MSELoss()):
        """Single sample training (legacy compatibility)."""
        input_tensor = torch.zeros(1, len(self.input_neurons), device=self.device)
        target_tensor = torch.zeros(1, len(self.output_neurons), device=self.device)
        
        for i, name in enumerate(self.input_neurons):
            input_tensor[0, i] = inputs.get(name, 0.0)
        for i, name in enumerate(self.output_neurons):
            target_tensor[0, i] = targets.get(name, 0.0)
        
        return self.train_step_batch(input_tensor, target_tensor, optimizer, criterion)
    
    def predict(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Make prediction without gradients."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            return {name: value.item() for name, value in outputs.items()}
    
    def get_output_neurons(self) -> List[str]:
        return self.output_neurons
    
    def get_input_neurons(self) -> List[str]:
        return self.input_neurons
    
    def get_num_edges(self) -> int:
        """Return total number of edges in the graph."""
        return sum(module.edge_index.shape[1] for module in self.layer_params)


# Example usage
# if __name__ == "__main__":
#     from .network import Node, GraphNetwork
#     import time
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
    
#     from torchvision import datasets, transforms
#     from torch.utils.data import DataLoader

#     input_size = 28 * 28
#     hidden_size = 128
#     num_classes = 10
#     batch_size = 512
#     epochs = 5

#     print("Building network...")
#     g = GraphNetwork()
#     input_nodes = [Node(f"x{i}") for i in range(input_size)]
#     hidden_nodes = [Node(f"h{i}") for i in range(hidden_size)]
#     output_nodes = [Node(f"y{i}") for i in range(num_classes)]

#     edges = []
#     for inp in input_nodes:
#         for h in hidden_nodes:
#             edges.append((inp, h))
#     for h in hidden_nodes:
#         for out in output_nodes:
#             edges.append((h, out))

#     g.add_edges(edges)

#     print("Initializing model...")
#     nn_model = GraphNeuralNetwork(
#         g,
#         input_neurons=[n.name for n in input_nodes],
#         output_neurons=[n.name for n in output_nodes],
#         activation='relu',
#         device=device,
#         prune_dead=True,
#         use_mixed_precision=(device == 'cuda')
#     )

#     print(f"\nModel Stats:")
#     print(f"Total parameters: {sum(p.numel() for p in nn_model.parameters())}")
#     print(f"Number of layers: {len(nn_model.layers)}")
#     print(f"Number of edges (preserved from graph): {nn_model.get_num_edges()}")
#     print(f"Mixed precision: {nn_model.use_mixed_precision}")

#     print("\nLoading data...")
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
#                              num_workers=4, pin_memory=(device=='cuda'))
#     test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False,
#                             num_workers=2, pin_memory=(device=='cuda'))

#     optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
#     criterion = nn.CrossEntropyLoss()

#     print("\nTraining (vectorized + topology-preserving)...")
#     for epoch in range(epochs):
#         nn_model.train()
#         total_loss = 0.0
#         num_batches = 0
#         epoch_start = time.time()
        
#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images = images.view(images.size(0), -1).to(device, non_blocking=True)
            
#             targets = torch.zeros(images.size(0), num_classes, device=device)
#             targets.scatter_(1, labels.unsqueeze(1).to(device), 1.0)
            
#             loss = nn_model.train_step_batch(images, targets, optimizer, criterion)
#             total_loss += loss
#             num_batches += 1
            
#             if batch_idx % 50 == 0:
#                 print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss:.4f}", end='\r')

#         epoch_time = time.time() - epoch_start
#         scheduler.step()
#         print(f"\nEpoch {epoch+1}/{epochs}, avg loss: {total_loss/num_batches:.6f}, time: {epoch_time:.2f}s")

#     print("\nEvaluating...")
#     nn_model.eval()
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.view(images.size(0), -1).to(device, non_blocking=True)
#             outputs = nn_model.forward_batch(images)
#             predictions = outputs.argmax(dim=1)
#             correct += (predictions == labels.to(device)).sum().item()
#             total += labels.size(0)

#     print(f"Test accuracy: {correct}/{total} = {correct/total:.4f}")
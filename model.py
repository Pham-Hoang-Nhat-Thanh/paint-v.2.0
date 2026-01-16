import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple


class GraphNeuralNetwork(nn.Module):
    """
    Neural network built from GraphNetwork structure with backprop and GPU support.
    Each node is an individual learnable neuron.
    """
    def __init__(self, graph_network, input_neurons, output_neurons, 
                 activation='relu', device='cpu'):
        """
        Convert GraphNetwork to PyTorch Neural Network.
        
        :param graph_network: GraphNetwork instance
        :param input_neurons: list of input neuron names
        :param output_neurons: list of output neuron names
        :param activation: activation function ('relu', 'sigmoid', 'tanh', 'linear')
        :param device: 'cpu' or 'cuda' for GPU acceleration
        """
        super(GraphNeuralNetwork, self).__init__()
        
        self.graph = graph_network
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Store input/output neuron specifications
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        
        # Validate network structure
        self._validate_structure()
        
        # Create parameter dictionaries for weights and biases
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        
        # Store neuron names and activation
        self.neuron_names = [node.name for node in graph_network.adjacency.keys()]
        self.activation_fn = self._get_activation(activation)
        
        # Initialize parameters for each edge and neuron
        self._initialize_parameters()
        
        # Move to device
        self.to(self.device)
        
    def _get_activation(self, activation):
        """Get PyTorch activation function."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _validate_structure(self):
        """Validate that input/output neurons match graph structure."""
        # Check that all specified neurons exist
        for name in self.input_neurons:
            if name not in [node.name for node in self.graph.adjacency.keys()]:
                raise ValueError(f"Input neuron '{name}' not found in graph")
        
        for name in self.output_neurons:
            if name not in [node.name for node in self.graph.adjacency.keys()]:
                raise ValueError(f"Output neuron '{name}' not found in graph")
        
        # Check input neurons have no parents
        for name in self.input_neurons:
            node = next((n for n in self.graph.adjacency.keys() if n.name == name), None)
            if node:
                parents = [p for p, children in self.graph.adjacency.items() 
                          if node in children]
                if parents:
                    raise ValueError(
                        f"Input neuron '{name}' has incoming edges from {[p.name for p in parents]}. "
                        f"Input neurons cannot have parents."
                    )
        
        # Check output neurons have no children
        for name in self.output_neurons:
            node = next((n for n in self.graph.adjacency.keys() if n.name == name), None)
            if node and len(self.graph.adjacency[node]) > 0:
                children = [c.name for c in self.graph.adjacency[node]]
                raise ValueError(
                    f"Output neuron '{name}' has outgoing edges to {children}. "
                    f"Output neurons cannot have children."
                )
        
        # Warn about neurons that are neither input nor output nor connected
        all_specified = set(self.input_neurons + self.output_neurons)
        all_neurons = set(node.name for node in self.graph.adjacency.keys())
        
        # Find disconnected neurons (no parents and no children, but not specified as input/output)
        for node in self.graph.adjacency.keys():
            if node.name not in all_specified:
                parents = [p for p, children in self.graph.adjacency.items() 
                          if node in children]
                children = self.graph.adjacency[node]
                if not parents and not children:
                    raise ValueError(
                        f"Neuron '{node.name}' is disconnected (no parents or children) "
                        f"and not specified as input or output. This creates an ambiguous network."
                    )
    
    def _initialize_parameters(self):
        """Initialize weights and biases for all neurons."""
        # Initialize bias for each neuron
        for node in self.graph.adjacency.keys():
            self.biases[node.name] = nn.Parameter(torch.randn(1) * 0.1)
        
        # Initialize weights for each edge
        for parent_node, children in self.graph.adjacency.items():
            for child_node in children:
                edge_key = f"{parent_node.name}->{child_node.name}"
                # Xavier initialization
                self.weights[edge_key] = nn.Parameter(
                    torch.randn(1) * (2.0 / (1 + 1)) ** 0.5
                )
    
    def forward(self, input_dict: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Forward propagation through the network.
        
        :param input_dict: dict mapping input neuron names -> values
        :return: dict mapping all neuron names -> tensor values
        """
        # Store neuron values
        neuron_values = {}
        
        # Convert inputs to tensors and set input neuron values
        for name, value in input_dict.items():
            if name in self.neuron_names:
                neuron_values[name] = torch.tensor([value], 
                                                   dtype=torch.float32, 
                                                   device=self.device)
        
        # Process neurons in topological order
        for node in self.graph.topo_order:
            # Find parents of this node
            parents = [p for p, children in self.graph.adjacency.items() 
                      if node in children]
            
            if not parents:
                # Input neuron - already set
                if node.name not in neuron_values:
                    neuron_values[node.name] = torch.zeros(1, device=self.device)
                continue
            
            # Compute weighted sum of parent activations
            weighted_sum = self.biases[node.name]
            
            for parent in parents:
                edge_key = f"{parent.name}->{node.name}"
                parent_value = neuron_values[parent.name]
                weighted_sum = weighted_sum + self.weights[edge_key] * parent_value
            
            # Apply activation function
            neuron_values[node.name] = self.activation_fn(weighted_sum)
        
        return neuron_values
    
    def get_output_neurons(self) -> List[str]:
        """Get names of output neurons."""
        return self.output_neurons
    
    def get_input_neurons(self) -> List[str]:
        """Get names of input neurons."""
        return self.input_neurons
    
    def train_step(self, inputs: Dict[str, float], targets: Dict[str, float], 
                   optimizer, criterion=nn.MSELoss()):
        """
        Single training step with backpropagation.
        
        :param inputs: dict mapping input neuron names -> values
        :param targets: dict mapping output neuron names -> target values
        :param optimizer: PyTorch optimizer
        :param criterion: loss function
        :return: loss value
        """
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Compute loss for output neurons
        loss = torch.tensor(0.0, device=self.device)
        for name, target_value in targets.items():
            if name in outputs:
                target = torch.tensor([target_value], 
                                    dtype=torch.float32, 
                                    device=self.device)
                loss = loss + criterion(outputs[name], target)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def predict(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Make prediction (forward pass without gradients).
        
        :param inputs: dict mapping input neuron names -> values
        :return: dict mapping all neuron names -> predicted values
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            return {name: value.item() for name, value in outputs.items()}


# Example usage with training
if __name__ == "__main__":
    from network import Node, GraphNetwork
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create graph network
    g = GraphNetwork()
    
    # Create nodes for XOR problem
    x1 = Node("x1")
    x2 = Node("x2")
    h1 = Node("h1")
    h2 = Node("h2")
    h3 = Node("h3")
    y = Node("y")
    
    # Build network: x1, x2 -> h1, h2 -> y
    g.add_edges([
        (x1, h1), (x1, h2),
        (x2, h1), (x2, h2), (x2, h3),
        (h1, y), (h2, y)
    ])
    
    # Create neural network with GPU support
    nn_model = GraphNeuralNetwork(
        g, 
        input_neurons=['x1', 'x2'],
        output_neurons=['y'],
        activation='sigmoid', 
        device=device
    )
    
    print(f"\n{nn_model}")
    print(f"Input neurons: {nn_model.get_input_neurons()}")
    print(f"Output neurons: {nn_model.get_output_neurons()}")
    print(f"Total parameters: {sum(p.numel() for p in nn_model.parameters())}")
    
    # Training setup
    optimizer = optim.Adam(nn_model.parameters(), lr=0.1)
    
    # XOR training data
    training_data = [
        ({'x1': 0.0, 'x2': 0.0}, {'y': 0.0}),
        ({'x1': 0.0, 'x2': 1.0}, {'y': 1.0}),
        ({'x1': 1.0, 'x2': 0.0}, {'y': 1.0}),
        ({'x1': 1.0, 'x2': 1.0}, {'y': 0.0}),
    ]
    
    # Training loop
    print("\nTraining XOR problem...")
    nn_model.train()
    for epoch in range(5000):
        total_loss = 0
        for inputs, targets in training_data:
            loss = nn_model.train_step(inputs, targets, optimizer)
            total_loss += loss
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/4:.6f}")
    
    # Test predictions
    print("\nPredictions after training:")
    for inputs, targets in training_data:
        predictions = nn_model.predict(inputs)
        print(f"Input: {inputs} -> Output: {predictions['y']:.4f} (Target: {targets['y']})")
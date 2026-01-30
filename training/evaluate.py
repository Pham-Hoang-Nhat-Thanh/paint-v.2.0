import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
from typing import Dict
from env.model import GraphNeuralNetwork
import numpy as np


class ArchitectureEvaluator:
    """
    Evaluates NAS architectures by training them on real tasks.
    
    This is the missing piece that connects MCTS to actual performance.
    """
    
    def __init__(self, 
                 task: str = 'mnist',
                 train_subset_size: int = 5000,
                 val_subset_size: int = 1000,
                 train_epochs: int = 5,
                 batch_size: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 cache_evaluations: bool = True):
        """
        Args:
            task: 'mnist', 'cifar10', etc.
            train_subset_size: Use subset for faster evaluation
            val_subset_size: Validation set size
            train_epochs: Epochs to train each architecture
            batch_size: Training batch size
            device: 'cuda' or 'cpu'
            cache_evaluations: Cache results for identical architectures
        """
        self.task = task
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Cache to avoid re-training identical architectures
        self.cache_evaluations = cache_evaluations
        self.eval_cache = {}
        
        # Load dataset once
        self._load_dataset()
        
        print(f"ArchitectureEvaluator initialized:")
        print(f"  Task: {task}")
        print(f"  Train subset: {train_subset_size}")
        print(f"  Val subset: {val_subset_size}")
        print(f"  Device: {device}")
    
    def _load_dataset(self):
        """Load and prepare dataset."""
        if self.task == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            val_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            
            self.input_size = 28 * 28
            self.num_classes = 10
            
        elif self.task == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            
            self.input_size = 32 * 32 * 3
            self.num_classes = 10
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Use subsets for faster evaluation
        train_indices = torch.randperm(len(train_ds))[:self.train_subset_size]
        val_indices = torch.randperm(len(val_ds))[:self.val_subset_size]
        
        train_subset = Subset(train_ds, train_indices)
        val_subset = Subset(val_ds, val_indices)
        
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=(self.device == 'cuda')
        )
        
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=(self.device == 'cuda')
        )
    
    def _graph_to_network(self, nas_graph) -> GraphNeuralNetwork:
        """
        Convert NASGraph to trainable GraphNeuralNetwork.
        
        This is the key conversion step!
        """
        # Your GraphNeuralNetwork expects:
        # - graph_network: The graph structure
        # - input_neurons: List of input node identifiers
        # - output_neurons: List of output node identifiers
        
        # Map NASGraph structure to input/output neurons
        input_neurons = [nas_graph.nodes[i] for i in range(nas_graph.n_input)]
        output_neurons = [nas_graph.nodes[i] for i in range(
            nas_graph.n_input + nas_graph.n_hidden,
            nas_graph.n_nodes
        )]
        # Ensure graph outputs match dataset classes to avoid label/logit mismatch
        if nas_graph.n_output != self.num_classes:
            raise ValueError(
                f"NASGraph.n_output={nas_graph.n_output} does not match evaluator num_classes={self.num_classes}. "
                "Create NASGraph with matching n_output or adjust the evaluator."
            )
        
        # Create your GraphNeuralNetwork
        model = GraphNeuralNetwork(
            graph_network=nas_graph,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            activation='relu',
            device=self.device,
            prune_dead=True,
            use_mixed_precision=(self.device == 'cuda')
        )
        
        return model
    
    def evaluate(self, nas_graph) -> Dict[str, float]:
        """
        Evaluate architecture by training on task.
        
        Args:
            nas_graph: NASGraph from MCTS
        
        Returns:
            Dictionary with:
            - accuracy: Validation accuracy (primary reward)
            - loss: Final validation loss
            - train_time: Training time in seconds
            - num_params: Number of trainable parameters
            - num_edges: Number of edges in architecture
        """
        # Check cache
        graph_hash = nas_graph.get_hash()
        if self.cache_evaluations and graph_hash in self.eval_cache:
            print(f"  [CACHE HIT] Using cached evaluation")
            return self.eval_cache[graph_hash]
        
        print(f"  [TRAINING] Evaluating architecture (hash: {graph_hash})")
        start_time = time.time()
        
        try:
            # Convert to trainable network
            model = self._graph_to_network(nas_graph)

            # Quick sanity check: ensure model output size matches evaluator classes
            try:
                input_dim = len(model.input_indices)
            except Exception:
                input_dim = None

            if input_dim is not None:
                dummy_in = torch.zeros(1, input_dim, device=model.device)
                try:
                    out = model.forward_batch(dummy_in)
                    n_out = out.size(1)
                except Exception as e:
                    raise RuntimeError(f"Failed to run forward_batch sanity check: {e}")

                if n_out != self.num_classes:
                    raise ValueError(
                        f"Model output size ({n_out}) does not match evaluator num_classes ({self.num_classes}). "
                        "This usually means the NASGraph output nodes do not align with dataset classes."
                    )
            
            # Setup optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Train
            model.train()
            for epoch in range(self.train_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for images, labels in self.train_loader:
                    # Prepare data
                    images = images.view(images.size(0), -1).to(self.device)
                    labels = labels.to(self.device)

                    # Train step (CrossEntropyLoss expects class indices)
                    loss = model.train_step_batch(images, labels, optimizer, criterion)
                    epoch_loss += loss
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                print(f"    Epoch {epoch+1}/{self.train_epochs}, Loss: {avg_loss:.4f}")
            
            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.view(images.size(0), -1).to(self.device)
                    outputs = model.forward_batch(images)
                    
                    # Accuracy
                    labels = labels.to(self.device)
                    predictions = outputs.argmax(dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                    # Loss (CrossEntropyLoss expects class indices)
                    val_loss += criterion(outputs, labels).item()
            
            accuracy = correct / total
            avg_val_loss = val_loss / len(self.val_loader)
            train_time = time.time() - start_time
            
            # Collect metrics
            results = {
                'accuracy': accuracy,
                'loss': avg_val_loss,
                'train_time': train_time,
                'num_params': sum(p.numel() for p in model.parameters()),
                'num_edges': nas_graph.get_num_edges()
            }
            
            print(f"  [RESULT] Accuracy: {accuracy:.4f}, Time: {train_time:.1f}s")
            
            # Cache result
            if self.cache_evaluations:
                self.eval_cache[graph_hash] = results
            
            return results
            
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            print(f"  [ERROR] Evaluation failed: type={type(e).__name__}, repr={repr(e)}")
            # Return poor performance on failure
            return {
                'accuracy': 0.0,
                'loss': 100.0,
                'train_time': 0.0,
                'num_params': 0,
                'num_edges': nas_graph.get_num_edges()
            }


def reward_function(evaluator: ArchitectureEvaluator, accuracy_threshold: float = 0.8):
    """
    Create a reward function that applies complexity penalties only when
    the architecture reaches `accuracy_threshold` accuracy.
    """
    def reward_function(graph) -> float:
        """
        Reward based on actual architecture performance.

        Primary objective: accuracy.
        Complexity penalties (params, edges) are applied only when
        accuracy >= accuracy_threshold.
        """
        # Evaluate architecture
        results = evaluator.evaluate(graph)

        accuracy = results['accuracy']
        num_params = results['num_params']
        num_edges = results['num_edges']

        # Base reward is accuracy
        reward = accuracy

        # Apply complexity penalties only when accuracy meets threshold
        if accuracy >= accuracy_threshold:
            if num_params > 0:
                reward -= 0.01 * np.log(num_params + 1)  # Small parameter penalty

            if num_edges > 0:
                reward -= 0.001 * np.log(num_edges + 1)  # Small edge penalty

        print(f"  [REWARD] Acc: {accuracy:.4f}, Params: {num_params}, Edges: {num_edges} → Reward: {reward:.4f}")

        return reward

    return reward_function


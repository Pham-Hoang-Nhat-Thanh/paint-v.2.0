import torch
from agent.policy_value_net import PolicyValueNet
from training.selfplay import SelfPlayEngine
from training.replay_buffer import ReplayBuffer
from training.trainer import AlphaZeroTrainer
from training.loss import AlphaZeroLoss
from mcts.main import NASGraph, MultiHeadMCTS
from training.evaluate import reward_function, ArchitectureEvaluator


def example_main():    
    # Configuration
    # Use MNIST-sized inputs and reasonable hidden size
    n_input = 28 * 28  # 784
    n_hidden = 128
    n_output = 10  # MNIST classes
    n_nodes = n_input + n_hidden + n_output

    # Define sensible node subsets for heads: split hidden nodes into chunks
    n_heads = 4
    chunk = n_hidden // n_heads
    node_subsets = []
    for i in range(n_heads):
        start = n_input + i * chunk
        end = n_input + (i + 1) * chunk if i < n_heads - 1 else n_input + n_hidden
        node_subsets.append(list(range(start, end)))
    
    # Create network
    network = PolicyValueNet(
        node_subsets=node_subsets,
        n_nodes=n_nodes,
        node_embed_dim=128,
        head_embed_dim=128
    )
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Create MCTS with batched evaluator
    mcts = MultiHeadMCTS(
        node_subsets=node_subsets,
        n_nodes=n_nodes,
        num_threads=4,
        cache_size=10000
    )
    
    # Set batched evaluator (OPTIMIZED)
    mcts.set_evaluator(
        policy_fn=lambda states, heads: network.predict_batch(states, heads)[0],
        value_fn=lambda states: network.predict_batch(states, [0]*len(states))[1],
        batched=True
    )

    evaluator = ArchitectureEvaluator(
        task='mnist',
        train_subset_size=5000,   # Fast evaluation
        val_subset_size=1000,
        train_epochs=3,           # Quick training
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache_evaluations=True    # Cache to avoid retraining
    )
    
    reward_fn = reward_function(evaluator=evaluator)

    # Self-play
    selfplay = SelfPlayEngine(
        mcts_coordinator=mcts,
        initial_graph_fn=lambda: NASGraph(n_input, n_hidden, n_output),
        reward_fn=reward_fn,
        max_steps=1000,
        num_simulations=1000
    )
    
    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=10000, min_size=100)
    
    # Trainer with optimized batching
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = AlphaZeroLoss()
    
    trainer = AlphaZeroTrainer(
        network=network,
        mcts_coordinator=mcts,
        selfplay_engine=selfplay,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        loss_fn=loss_fn
    )
    
    # Training loop
    print("Starting training...")
    trainer.train(
        num_iterations=100,
        episodes_per_iteration=5,
        train_steps_per_iteration=50,
        batch_size=16
    )


if __name__ == '__main__':
    example_main()

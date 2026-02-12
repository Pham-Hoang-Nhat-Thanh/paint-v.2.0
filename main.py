import torch
from agent.policy_value_net import PolicyValueNet
from training.selfplay import SelfPlayEngine
from training.replay_buffer import ReplayBuffer
from training.trainer import AlphaZeroTrainer
from training.loss import AlphaZeroLoss
from mcts.main import MultiHeadMCTS
from training.evaluate import reward_function, ArchitectureEvaluator
# from env.network import NASGraph
from env.fast_network import NASGraph
from env.partition import build_node_subsets


def main():
    n_input = 784
    n_hidden = 128
    n_output = 10
    n_nodes = n_input + n_hidden + n_output

    # Build node subsets for partitioned MCTS
    node_subsets = build_node_subsets(
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output,
        subset_size=32,
        seed=42)
    
    print(f"Built {len(node_subsets)} node subsets for partitioned MCTS.")
    
    # Create network
    network = PolicyValueNet(
        node_subsets=node_subsets,
        n_nodes=n_nodes,
        node_embed_dim=1024,
        head_embed_dim=1024,
    )

    network.compile_for_inference()
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Create MCTS with batched evaluator
    mcts = MultiHeadMCTS(
        node_subsets=node_subsets,
        n_nodes=n_nodes,
        num_threads=32,
        cache_size=10000
    )
    
    # Set network evaluator (OPTIMIZED - single predict_batch call)
    mcts.set_evaluator(network=network)

    evaluator = ArchitectureEvaluator(
        task='mnist',
        train_subset_size=-1,   # Full dataset
        val_subset_size=-1,     # Full dataset
        train_epochs=5,           # Quick training
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache_evaluations=False # Architecture is different enough each time that caching is not beneficial
    )
    
    reward_fn = reward_function(evaluator=evaluator)

    # Self-play with batched MCTS (4x more simulations with batching)
    selfplay = SelfPlayEngine(
        mcts_coordinator=mcts,
        initial_graph_fn=lambda: NASGraph(n_input, n_hidden, n_output),
        reward_fn=reward_fn,
        max_steps=100,
        num_simulations=100, 
        mcts_batch_size=128
    )
    
    # Store evaluator reference for cache clearing
    selfplay.evaluator = evaluator

    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=10000, min_size=1000)

    # Loss function - NO anti-overfitting measures (they caused catastrophic forgetting)
    loss_fn = AlphaZeroLoss(
        value_loss_weight=0.15,
        entropy_weight=0.1,
        l2_weight=1e-4,
        value_label_smoothing=0.0,  # DISABLED: Confuses learning signal
        use_huber_loss=False
    )

    # Trainer - NO value weight decay (it weakened the value head too much)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2)

    trainer = AlphaZeroTrainer(
        network=network,
        mcts_coordinator=mcts,
        selfplay_engine=selfplay,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        loss_fn=loss_fn,
        value_loss_decay=1.0,  # DISABLED: No decay (catastrophic forgetting)
        min_value_loss_weight=0.15
    )
    
    # Training loop
    print("Starting training...")
    trainer.train(
        num_iterations=1000,
        episodes_per_iteration=10,
        train_steps_per_iteration=50,
        batch_size=32
    )


if __name__ == '__main__':
    main()

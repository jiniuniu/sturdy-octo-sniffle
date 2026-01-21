"""
Step 1: Base Diffusion Verification

This experiment validates the basic diffusion mechanics:
- share_probability = 1.0 (everyone shares)
- adopt_probability = 1.0 (everyone adopts when aware)
- Used to verify network structure and diffusion speed
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt

from diffusion_sim import (
    NetworkConfig,
    NetworkVisualizer,
    SimpleDiffusionEngine,
    SimulationConfig,
    SmallWorldNetworkBuilder,
    TimeSeriesVisualizer,
)


def run_base_experiment(
    n: int = 100,
    k: int = 6,
    p: float = 0.1,
    max_steps: int = 50,
    initial_adopters: int = 1,
    seed: int = 42,
):
    """
    Run a baseline diffusion experiment.

    Args:
        n: Number of nodes
        k: Average degree (must be even)
        p: Rewiring probability
        max_steps: Maximum simulation steps
        initial_adopters: Number of initial seed adopters
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("BASE DIFFUSION EXPERIMENT (Step 1)")
    print("=" * 60)
    print(f"Network: n={n}, k={k}, p={p}")
    print(f"Simulation: max_steps={max_steps}, initial_adopters={initial_adopters}")
    print(f"Seed: {seed}")
    print()

    # Configure network
    network_config = NetworkConfig(network_type="small_world", n=n, k=k, p=p, seed=seed)

    # Configure simulation
    sim_config = SimulationConfig(
        max_steps=max_steps,
        seed=seed,
        initial_adopters=initial_adopters,
        share_probability=1.0,  # Everyone shares
        adopt_probability=1.0,  # Everyone adopts when aware
    )

    # Build network
    print("Building network...")
    builder = SmallWorldNetworkBuilder()
    graph = builder.build(network_config)

    # Print network stats
    stats = builder.get_network_stats(graph)
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Run simulation
    print("\nRunning simulation...")
    engine = SimpleDiffusionEngine(graph, sim_config, network_config)
    result = engine.run()

    print(f"\nSimulation completed in {result.total_steps} steps")
    print(f"Final awareness rate: {result.final_awareness_rate:.1%}")
    print(f"Final adoption rate: {result.final_adoption_rate:.1%}")

    # Visualize results
    print("\nGenerating visualizations...")

    # Create summary plot
    output_dir = Path(__file__).parent.parent / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = TimeSeriesVisualizer.create_summary_plot(result)
    save_path = output_dir / f"base_diffusion_n{n}_k{k}_p{p:.2f}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Summary plot saved to: {save_path}")

    # Create network snapshots at key points
    fig2, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Compute layout once
    visualizer = NetworkVisualizer(graph)

    # Snapshots at different time points
    snapshot_steps = [
        0,
        result.total_steps // 3,
        2 * result.total_steps // 3,
        result.total_steps,
    ]

    for i, step in enumerate(snapshot_steps):
        # Reset graph to snapshot state
        snapshot = result.snapshots[step]

        # Update graph states to match snapshot
        from diffusion_sim import BaseAgent

        for node_id in graph.nodes():
            state = BaseAgent.get_state(graph, node_id)
            state.aware = node_id in snapshot.aware_agents
            state.adopted = node_id in snapshot.adopted_agents

        visualizer.draw_snapshot(step, ax=axes[i])

    plt.tight_layout()
    save_path2 = output_dir / f"base_diffusion_snapshots_n{n}_k{k}_p{p:.2f}.png"
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    print(f"Network snapshots saved to: {save_path2}")

    print("\nExperiment complete!")
    print("=" * 60)

    return result


def compare_network_types():
    """
    Compare diffusion speed across different network configurations.
    """
    print("\n" + "=" * 60)
    print("NETWORK COMPARISON EXPERIMENT")
    print("=" * 60)

    configs = [
        {"p": 0.0, "label": "Regular lattice (p=0)"},
        {"p": 0.1, "label": "Small-world (p=0.1)"},
        {"p": 1.0, "label": "Random (p=1.0)"},
    ]

    results = []
    labels = []

    for config in configs:
        print(f"\nRunning: {config['label']}")
        result = run_base_experiment(n=100, k=6, p=config["p"], max_steps=30, seed=42)
        results.append(result)
        labels.append(config["label"])

    # Compare results
    fig, ax = plt.subplots(figsize=(12, 6))
    TimeSeriesVisualizer.plot_comparison(results, labels, metric="adopted", ax=ax)

    output_dir = Path(__file__).parent.parent / "data" / "outputs"
    save_path = output_dir / "network_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {save_path}")


if __name__ == "__main__":
    # Run single experiment
    result = run_base_experiment(n=100, k=6, p=0.1, max_steps=30, initial_adopters=1)

    # Uncomment to run comparison
    # compare_network_types()

    plt.show()

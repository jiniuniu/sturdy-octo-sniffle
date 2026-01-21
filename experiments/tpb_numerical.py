"""
Step 2: TPB Numerical Experiment

This experiment validates the TPB decision model with numerical traits:
- Heterogeneous agents with different TPB traits
- Population stratified by Rogers' categories
- Decision-making based on attitude, social norm, and PBC
- Compare diffusion patterns across different configurations
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
from diffusion_sim import (
    NetworkConfig,
    SimulationConfig,
    SmallWorldNetworkBuilder,
    TPBDiffusionEngine,
    TPBDecisionModel,
    PopulationGenerator,
    NetworkVisualizer,
    TimeSeriesVisualizer,
)
from diffusion_sim.decisions.components import FrictionModel


def run_tpb_experiment(
    n: int = 100,
    k: int = 6,
    p: float = 0.1,
    max_steps: int = 100,
    initial_adopters: int = 1,
    seed: int = 42,
    w_attitude: float = 0.4,
    w_social_norm: float = 0.35,
    w_pbc: float = 0.25,
    friction_level: float = 0.0,
    initial_strategy: str = "innovators",
):
    """
    Run a TPB-based diffusion experiment.

    Args:
        n: Number of nodes
        k: Average degree
        p: Rewiring probability
        max_steps: Maximum simulation steps
        initial_adopters: Number of initial seed adopters
        seed: Random seed
        w_attitude: Weight for attitude in TPB
        w_social_norm: Weight for social norm in TPB
        w_pbc: Weight for PBC in TPB
        friction_level: Base friction level (0-1)
        initial_strategy: Strategy for selecting initial adopters
    """
    print("=" * 60)
    print("TPB NUMERICAL EXPERIMENT (Step 2)")
    print("=" * 60)
    print(f"Network: n={n}, k={k}, p={p}")
    print(f"Simulation: max_steps={max_steps}, initial_adopters={initial_adopters}")
    print(f"TPB weights: A={w_attitude}, SN={w_social_norm}, PBC={w_pbc}")
    print(f"Friction: {friction_level}")
    print(f"Initial strategy: {initial_strategy}")
    print(f"Seed: {seed}")
    print()

    # Configure network
    network_config = NetworkConfig(
        network_type="small_world", n=n, k=k, p=p, seed=seed
    )

    # Configure simulation
    sim_config = SimulationConfig(
        max_steps=max_steps,
        seed=seed,
        initial_adopters=initial_adopters,
    )

    # Build network
    print("Building network...")
    builder = SmallWorldNetworkBuilder()
    graph = builder.build(network_config)

    # Generate heterogeneous population
    print("Generating population with TPB traits...")
    pop_generator = PopulationGenerator(seed=seed)
    pop_generator.generate_population(graph)

    # Print population distribution
    category_dist = pop_generator.get_category_distribution(graph)
    print("\nPopulation distribution:")
    for category, count in category_dist.items():
        ratio = count / n
        print(f"  {category}: {count} ({ratio:.1%})")

    # Select initial adopters
    initial_ids = pop_generator.select_initial_adopters(
        graph, initial_adopters, strategy=initial_strategy
    )
    print(f"\nSelected {len(initial_ids)} initial adopters using '{initial_strategy}' strategy")

    # Create TPB decision model
    friction = FrictionModel(base_friction=friction_level, mode="fixed")
    decision_model = TPBDecisionModel(
        w_attitude=w_attitude,
        w_social_norm=w_social_norm,
        w_pbc=w_pbc,
        sn_strategy="mean",
        intention_mode="sigmoid",
        friction=friction,
        share_threshold=0.5,
    )

    # Run simulation
    print("\nRunning TPB simulation...")
    engine = TPBDiffusionEngine(graph, sim_config, network_config, decision_model)
    engine.initialize(initial_adopter_ids=initial_ids)
    result = engine.run()

    print(f"\nSimulation completed in {result.total_steps} steps")
    print(f"Final awareness rate: {result.final_awareness_rate:.1%}")
    print(f"Final adoption rate: {result.final_adoption_rate:.1%}")

    # Visualize results
    print("\nGenerating visualizations...")

    output_dir = Path(__file__).parent.parent / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary plot
    fig = TimeSeriesVisualizer.create_summary_plot(result)
    save_path = output_dir / f"tpb_numerical_n{n}_f{friction_level:.2f}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Summary plot saved to: {save_path}")

    plt.close(fig)

    print("\nExperiment complete!")
    print("=" * 60)

    return result


def compare_tpb_weights():
    """
    Compare diffusion patterns with different TPB weight configurations.
    """
    print("\n" + "=" * 60)
    print("TPB WEIGHTS COMPARISON")
    print("=" * 60)

    configs = [
        {"w_a": 0.6, "w_sn": 0.2, "w_pbc": 0.2, "label": "Attitude-driven (A=0.6)"},
        {"w_a": 0.2, "w_sn": 0.6, "w_pbc": 0.2, "label": "Social Norm-driven (SN=0.6)"},
        {"w_a": 0.33, "w_sn": 0.34, "w_pbc": 0.33, "label": "Balanced (A=SN=PBC)"},
    ]

    results = []
    labels = []

    for config in configs:
        print(f"\nRunning: {config['label']}")
        result = run_tpb_experiment(
            n=100,
            k=6,
            p=0.1,
            max_steps=80,
            initial_adopters=2,
            seed=42,
            w_attitude=config["w_a"],
            w_social_norm=config["w_sn"],
            w_pbc=config["w_pbc"],
        )
        results.append(result)
        labels.append(config["label"])

    # Compare results
    fig, ax = plt.subplots(figsize=(12, 6))
    TimeSeriesVisualizer.plot_comparison(results, labels, metric="adopted", ax=ax)

    output_dir = Path(__file__).parent.parent / "data" / "outputs"
    save_path = output_dir / "tpb_weights_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {save_path}")

    plt.close(fig)


def compare_friction_levels():
    """
    Compare diffusion patterns with different friction levels.
    """
    print("\n" + "=" * 60)
    print("FRICTION LEVELS COMPARISON")
    print("=" * 60)

    friction_levels = [0.0, 0.2, 0.4]
    results = []
    labels = []

    for friction in friction_levels:
        print(f"\nRunning: friction={friction}")
        result = run_tpb_experiment(
            n=100,
            k=6,
            p=0.1,
            max_steps=80,
            initial_adopters=2,
            seed=42,
            friction_level=friction,
        )
        results.append(result)
        labels.append(f"Friction = {friction}")

    # Compare results
    fig, ax = plt.subplots(figsize=(12, 6))
    TimeSeriesVisualizer.plot_comparison(results, labels, metric="adopted", ax=ax)

    output_dir = Path(__file__).parent.parent / "data" / "outputs"
    save_path = output_dir / "tpb_friction_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {save_path}")

    plt.close(fig)


def compare_initial_strategies():
    """
    Compare different strategies for selecting initial adopters.
    """
    print("\n" + "=" * 60)
    print("INITIAL ADOPTER STRATEGY COMPARISON")
    print("=" * 60)

    strategies = ["innovators", "random", "high_degree"]
    results = []
    labels = []

    for strategy in strategies:
        print(f"\nRunning: strategy={strategy}")
        result = run_tpb_experiment(
            n=100,
            k=6,
            p=0.1,
            max_steps=80,
            initial_adopters=2,
            seed=42,
            initial_strategy=strategy,
        )
        results.append(result)
        labels.append(f"{strategy.replace('_', ' ').title()}")

    # Compare results
    fig, ax = plt.subplots(figsize=(12, 6))
    TimeSeriesVisualizer.plot_comparison(results, labels, metric="adopted", ax=ax)

    output_dir = Path(__file__).parent.parent / "data" / "outputs"
    save_path = output_dir / "tpb_initial_strategy_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    # Run single experiment
    result = run_tpb_experiment(
        n=100, k=6, p=0.1, max_steps=80, initial_adopters=2, seed=42
    )

    # Uncomment to run comparisons
    # compare_tpb_weights()
    # compare_friction_levels()
    # compare_initial_strategies()

    plt.show()

"""
Test script for sentiment diffusion simulation.

Tests the brand crisis sentiment diffusion implementation with:
- SBM network (community structure)
- 4 population types (strong_critics, conformists, silent_observers, amplifiers)
- Directional social norm (negativity bias)
- Interpretation and expression decisions
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffusion_sim.networks.sbm import SBMNetworkBuilder
from diffusion_sim.population.sentiment_sampling import SentimentPopulationGenerator
from diffusion_sim.decisions.sentiment import SimpleSentimentModel
from diffusion_sim.simulation.sentiment_engine import SentimentDiffusionEngine
from diffusion_sim.core.schemas import SimulationConfig, NetworkConfig
from diffusion_sim.visualization.sentiment_viz import (
    plot_sentiment_diffusion_curves,
    plot_polarity_over_time,
    plot_polarity_distribution,
    plot_sentiment_summary,
)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Note: matplotlib not available, skipping plots")


def run_sentiment_simulation(
    n: int = 200,
    num_blocks: int = 4,
    p_within: float = 0.3,
    p_between: float = 0.01,
    seed: int = 42,
    initial_aware: int = 5,
    initial_strategy: str = "critics",
    w_attitude: float = 0.4,
    w_social_norm: float = 0.35,
    w_moral: float = 0.25,
    alpha: float = 0.3,
    beta: float = 0.7,
    max_steps: int = 100,
):
    """
    Run a complete sentiment diffusion simulation.

    Args:
        n: Number of agents
        num_blocks: Number of community blocks in SBM network
        p_within: Edge probability within blocks
        p_between: Edge probability between blocks
        seed: Random seed
        initial_aware: Number of initial aware agents
        initial_strategy: Strategy for selecting initial aware agents
        w_attitude: Weight for attitude component
        w_social_norm: Weight for social norm component
        w_moral: Weight for moral sensitivity component
        alpha: Weight for positive sentiment (social norm)
        beta: Weight for negative sentiment (social norm, > alpha)
        max_steps: Maximum simulation steps

    Returns:
        Tuple of (result, graph, pop_generator)
    """
    print("=" * 80)
    print("Sentiment Diffusion Simulation - Brand Crisis Scenario")
    print("=" * 80)

    # Step 1: Create SBM network
    print(f"\n[1/5] Creating SBM network (n={n}, blocks={num_blocks})...")
    graph = SBMNetworkBuilder.build(
        n=n,
        num_blocks=num_blocks,
        p_within=p_within,
        p_between=p_between,
        seed=seed
    )
    network_stats = SBMNetworkBuilder.get_network_stats(graph)
    print(f"  Network created: {network_stats['num_edges']} edges")
    print(f"  Modularity: {network_stats['modularity']:.3f}")
    print(f"  Average degree: {network_stats['avg_degree']:.2f}")

    # Step 2: Generate population
    print(f"\n[2/5] Generating sentiment population...")
    pop_generator = SentimentPopulationGenerator(seed=seed)
    pop_generator.generate_population(graph)

    category_dist = pop_generator.get_category_distribution(graph)
    print("  Population distribution:")
    for cat, count in category_dist.items():
        percentage = count / n * 100
        print(f"    {cat.replace('_', ' ').title():<20}: {count:>3} ({percentage:>5.1f}%)")

    # Step 3: Create decision model
    print(f"\n[3/5] Creating sentiment decision model...")
    print(f"  Weights: A={w_attitude}, SN={w_social_norm}, Moral={w_moral}")
    print(f"  Directional SN: α={alpha}, β={beta} (negativity bias: β > α)")
    decision_model = SimpleSentimentModel(
        w_attitude=w_attitude,
        w_social_norm=w_social_norm,
        w_moral=w_moral,
        alpha=alpha,
        beta=beta,
    )

    # Step 4: Create simulation engine
    print(f"\n[4/5] Initializing simulation engine...")
    sim_config = SimulationConfig(
        max_steps=max_steps,
        seed=seed,
        initial_adopters=initial_aware,
    )
    network_config = NetworkConfig(
        network_type="sbm",
        n=n,
        seed=seed,
    )

    engine = SentimentDiffusionEngine(
        graph=graph,
        config=sim_config,
        network_config=network_config,
        decision_model=decision_model,
    )

    # Select initial aware agents
    initial_aware_ids = pop_generator.select_initial_aware(
        graph, initial_aware, strategy=initial_strategy
    )
    print(f"  Selected {len(initial_aware_ids)} initial aware agents ({initial_strategy} strategy)")
    engine.initialize(initial_aware_ids)

    # Step 5: Run simulation
    print(f"\n[5/5] Running simulation (max {max_steps} steps)...")
    result = engine.run()

    print(f"\n{'=' * 80}")
    print("Simulation Complete")
    print("=" * 80)
    print(f"Total steps: {result.total_steps}")
    print(f"Final awareness rate: {result.final_awareness_rate:.1%}")
    print(f"Final expression rate: {result.final_adoption_rate:.1%}")

    # Get final snapshot
    final_snapshot = result.snapshots[-1]
    print(f"\nFinal state:")
    print(f"  Aware agents: {final_snapshot.aware_count}")
    print(f"  Interpreted agents: {final_snapshot.interpreted_count}")
    print(f"  Expressed agents: {final_snapshot.expressed_count}")
    print(f"  Positive sentiment: {final_snapshot.positive_count}")
    print(f"  Negative sentiment: {final_snapshot.negative_count}")
    print(f"  Mean polarity: {final_snapshot.mean_polarity:.3f}")

    # Polarity distribution
    polarity_dist = engine.get_polarity_distribution()
    print(f"\nPolarity distribution:")
    for category, count in polarity_dist.items():
        print(f"  {category.capitalize():<15}: {count:>3}")

    return result, graph, pop_generator


def main():
    """Run test simulation and generate visualizations."""
    # Run simulation
    result, graph, pop_generator = run_sentiment_simulation(
        n=200,
        num_blocks=4,
        p_within=0.3,
        p_between=0.01,
        seed=42,
        initial_aware=5,
        initial_strategy="critics",
        w_attitude=0.4,
        w_social_norm=0.35,
        w_moral=0.25,
        alpha=0.3,
        beta=0.7,
        max_steps=100,
    )

    # Generate visualizations
    if PLOT_AVAILABLE:
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        # Create output directory
        output_dir = Path(__file__).parent.parent / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Diffusion curves
        print("\n[1/4] Generating diffusion curves...")
        fig1 = plot_sentiment_diffusion_curves(result.snapshots)
        fig1.savefig(output_dir / "sentiment_diffusion_curves.png", dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_dir / 'sentiment_diffusion_curves.png'}")

        # Plot 2: Polarity over time
        print("[2/4] Generating polarity over time...")
        fig2 = plot_polarity_over_time(result.snapshots)
        fig2.savefig(output_dir / "sentiment_polarity_time.png", dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_dir / 'sentiment_polarity_time.png'}")

        # Plot 3: Polarity distribution
        print("[3/4] Generating polarity distribution...")
        fig3 = plot_polarity_distribution(graph)
        fig3.savefig(output_dir / "sentiment_polarity_dist.png", dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_dir / 'sentiment_polarity_dist.png'}")

        # Plot 4: Summary
        print("[4/4] Generating summary visualization...")
        # Compute category statistics for visualization
        from diffusion_sim.agents.sentiment_agent import SentimentAgent
        category_stats = {}
        for cat in pop_generator.distribution.keys():
            interpreted_count = 0
            expressed_count = 0
            for node_id in graph.nodes():
                if SentimentAgent.get_category(graph, node_id) == cat:
                    state = SentimentAgent.get_state(graph, node_id)
                    if state.interpreted:
                        interpreted_count += 1
                    if state.expressed:
                        expressed_count += 1
            category_stats[cat] = {
                'interpreted_count': interpreted_count,
                'expressed_count': expressed_count,
            }

        fig4 = plot_sentiment_summary(result.snapshots, graph, category_stats)
        fig4.savefig(output_dir / "sentiment_summary.png", dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_dir / 'sentiment_summary.png'}")

        print("\n✓ All visualizations saved successfully")
        plt.show()
    else:
        print("\n(Matplotlib not available, skipping visualizations)")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

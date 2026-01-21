"""
Example: How to use the tracked simulation for diffusion sequence analysis.

This script demonstrates the complete workflow for analyzing diffusion sequences
and validating Rogers' theory predictions.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
from diffusion_sim.utils import (
    run_tracked_simulation,
    analyze_diffusion_sequence,
    check_rogers_sequence,
    print_diffusion_report,
)
from diffusion_sim.visualization import (
    visualize_diffusion_sequence,
    plot_category_comparison,
)


def main():
    """Run a complete tracked simulation analysis."""

    print("="*80)
    print("Diffusion Sequence Tracking Experiment")
    print("="*80)
    print("\nRunning tracked simulation...")

    # 1. Run tracked simulation
    result, tracking_data, graph = run_tracked_simulation(
        n=200,              # 200 nodes for better statistics
        k=6,                # Average degree
        p=0.1,              # Small-world network
        seed=42,            # Reproducibility
        initial_adopters=5, # 5 seed adopters (2.5%)
        initial_strategy="early_adopters",  # Select early adopters as seeds
        w_attitude=0.4,     # TPB weights
        w_social_norm=0.35,
        w_pbc=0.25,
        friction=0.0,       # No friction for clear signal
        max_steps=100,
    )

    print(f"Simulation complete: {result.total_steps} steps")
    print(f"Final adoption rate: {result.final_adoption_rate:.1%}")
    print(f"Final awareness rate: {result.final_awareness_rate:.1%}")

    # 2. Analyze diffusion sequence
    print("\nAnalyzing diffusion sequence...")
    stats = analyze_diffusion_sequence(tracking_data)

    # 3. Validate against Rogers' theory
    print("\nValidating Rogers' theory...")
    validation = check_rogers_sequence(stats)

    # 4. Print detailed report
    print_diffusion_report(stats, validation)

    # 5. Visualize results
    print("\nGenerating visualizations...")

    # Main visualization with 6 subplots
    fig1 = visualize_diffusion_sequence(tracking_data, result)
    plt.savefig(
        Path(__file__).parent.parent / "data" / "outputs" / "diffusion_sequence_analysis.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("✓ Detailed analysis plot saved")

    # Comparison plot
    fig2 = plot_category_comparison(tracking_data, result)
    plt.savefig(
        Path(__file__).parent.parent / "data" / "outputs" / "category_comparison.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("✓ Category comparison plot saved")

    plt.show()

    # 6. Key findings
    print("\n" + "="*80)
    print("Key Findings")
    print("="*80)

    # Check if early adopters are first
    first_cat = validation['aware_sequence'][0] if validation['aware_sequence'] else None
    if first_cat == 'early_adopters':
        print("✅ Early Adopters became aware first, as expected")
    else:
        print(f"⚠️  {first_cat.replace('_', ' ').title()} became aware first, parameters may need adjustment")

    # Check adoption rate trend
    adoption_rates = [stats[cat]['adoption_rate'] for cat in ['early_adopters', 'early_majority', 'late_majority', 'laggards'] if cat in stats]
    if len(adoption_rates) >= 2 and adoption_rates[0] >= adoption_rates[-1]:
        print("✅ Adoption rates show decreasing trend, as expected")
    else:
        print("⚠️  Adoption rate trend is unexpected")

    # Check delays
    delays = [stats[cat].get('mean_delay', 0) or 0 for cat in ['early_adopters', 'early_majority', 'late_majority', 'laggards'] if cat in stats and stats[cat].get('mean_delay')]
    if len(delays) >= 2 and delays[-1] >= delays[0]:
        print("✅ Decision delays increase, Laggards more hesitant, as expected")
    else:
        print("⚠️  Decision delay trend needs attention")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()

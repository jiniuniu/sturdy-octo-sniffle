"""
Quick test of Gaussian parameter configuration.

Verifies that the new trait sampling produces clear category differentiation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffusion_sim.utils import run_tracked_simulation, analyze_diffusion_sequence, print_diffusion_report

# Try to import matplotlib, but continue if not available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Note: matplotlib not available, skipping plots")


def main():
    print("="*80)
    print("Testing Gaussian Parameter Configuration")
    print("="*80)
    print("\nRunning simulation with new parameters...")
    print("- Gaussian sampling: mean ± std")
    print("- Category means: EA=0.8, EM=0.6, LM=0.4, Laggards=0.2")
    print("- Fixed: conformity=1.0, risk_aversion=0.0")
    print("- TPB weights: A=0.4, SN=0.35, PBC=0.25")
    print("- Friction=0.0")

    # Run tracked simulation
    result, tracking_data, graph = run_tracked_simulation(
        n=200,
        k=6,
        p=0.1,
        seed=42,
        initial_adopters=5,
        initial_strategy="early_adopters",
        w_attitude=0.4,
        w_social_norm=0.35,
        w_pbc=0.25,
        friction=0.0,
        max_steps=100,
    )

    print(f"\nSimulation complete: {result.total_steps} steps")
    print(f"Final adoption rate: {result.final_adoption_rate:.1%}")
    print(f"Final awareness rate: {result.final_awareness_rate:.1%}")

    # Analyze diffusion sequence
    stats = analyze_diffusion_sequence(tracking_data)

    # Print detailed report
    print_diffusion_report(stats)

    # Check category differentiation
    print("\n" + "="*80)
    print("Category Differentiation Analysis")
    print("="*80)

    categories = ['early_adopters', 'early_majority', 'late_majority', 'laggards']

    # Adoption rates
    adoption_rates = [stats[cat]['adoption_rate'] * 100 for cat in categories]
    print("\nAdoption Rates:")
    for cat, rate in zip(categories, adoption_rates):
        print(f"  {cat.replace('_', ' ').title():<20}: {rate:>6.1f}%")

    # Check if decreasing trend
    is_decreasing = all(adoption_rates[i] >= adoption_rates[i+1] for i in range(len(adoption_rates)-1))
    if is_decreasing:
        print("\n✅ Adoption rates show clear decreasing trend (EA > EM > LM > Laggards)")
    else:
        print("\n⚠️  Adoption rates do NOT show expected decreasing trend")

    # Mean aware time
    print("\nMean Awareness Time:")
    for cat in categories:
        mean_aware = stats[cat]['mean_aware']
        if mean_aware is not None:
            print(f"  {cat.replace('_', ' ').title():<20}: {mean_aware:>6.1f} steps")

    # Mean adoption delay
    print("\nMean Adoption Delay (Aware → Adopt):")
    delays = []
    for cat in categories:
        delay = stats[cat]['mean_delay']
        delays.append(delay if delay is not None else 0)
        delay_str = f"{delay:.1f}" if delay is not None else "N/A"
        print(f"  {cat.replace('_', ' ').title():<20}: {delay_str:>6} steps")

    # Check if delays increase
    valid_delays = [d for d in delays if d > 0]
    if len(valid_delays) >= 2:
        is_increasing = all(valid_delays[i] <= valid_delays[i+1] for i in range(len(valid_delays)-1))
        if is_increasing:
            print("\n✅ Decision delays increase (Laggards more hesitant)")
        else:
            print("\n⚠️  Decision delays do NOT show expected increasing trend")

    # Visualize adoption rates (if matplotlib available)
    if PLOT_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Adoption rates by category
        colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
        x_pos = np.arange(len(categories))
        axes[0].bar(x_pos, adoption_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=15, ha='right')
        axes[0].set_ylabel('Adoption Rate (%)', fontsize=11)
        axes[0].set_title('Final Adoption Rate by Category\n(Verify Decreasing Trend)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate(adoption_rates):
            axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Right: Mean delays by category
        valid_categories = [cat for cat, d in zip(categories, delays) if d > 0]
        valid_delay_vals = [d for d in delays if d > 0]
        if valid_delay_vals:
            x_pos2 = np.arange(len(valid_categories))
            colors2 = [colors[categories.index(c)] for c in valid_categories]
            axes[1].bar(x_pos2, valid_delay_vals, color=colors2, alpha=0.7, edgecolor='black', linewidth=1.5)
            axes[1].set_xticks(x_pos2)
            axes[1].set_xticklabels([c.replace('_', ' ').title() for c in valid_categories], rotation=15, ha='right')
            axes[1].set_ylabel('Mean Delay (steps)', fontsize=11)
            axes[1].set_title('Aware → Adopt Delay by Category\n(Verify Increasing Trend)', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')

            # Add value labels
            for i, v in enumerate(valid_delay_vals):
                axes[1].text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Save plot
        output_dir = Path(__file__).parent.parent / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "gaussian_params_test.png", dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {output_dir / 'gaussian_params_test.png'}")

        plt.show()
    else:
        print("\n(Matplotlib not available, skipping visualization)")

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)


if __name__ == "__main__":
    main()

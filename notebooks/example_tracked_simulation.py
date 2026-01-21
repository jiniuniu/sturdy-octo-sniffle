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
    print("创新扩散时序追踪实验")
    print("="*80)
    print("\n运行追踪仿真...")

    # 1. Run tracked simulation
    result, tracking_data, graph = run_tracked_simulation(
        n=200,              # 200 nodes for better statistics
        k=6,                # Average degree
        p=0.1,              # Small-world network
        seed=42,            # Reproducibility
        initial_adopters=5, # 5 seed adopters (2.5%)
        initial_strategy="innovators",  # Select innovators as seeds
        w_attitude=0.4,     # TPB weights
        w_social_norm=0.35,
        w_pbc=0.25,
        friction=0.0,       # No friction for clear signal
        max_steps=100,
    )

    print(f"仿真完成: {result.total_steps} 步")
    print(f"最终采纳率: {result.final_adoption_rate:.1%}")
    print(f"最终知晓率: {result.final_awareness_rate:.1%}")

    # 2. Analyze diffusion sequence
    print("\n分析扩散时序...")
    stats = analyze_diffusion_sequence(tracking_data)

    # 3. Validate against Rogers' theory
    print("\n验证Rogers理论...")
    validation = check_rogers_sequence(stats)

    # 4. Print detailed report
    print_diffusion_report(stats, validation)

    # 5. Visualize results
    print("\n生成可视化...")

    # Main visualization with 6 subplots
    fig1 = visualize_diffusion_sequence(tracking_data, result)
    plt.savefig(
        Path(__file__).parent.parent / "data" / "outputs" / "diffusion_sequence_analysis.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("✓ 详细分析图已保存")

    # Comparison plot
    fig2 = plot_category_comparison(tracking_data, result)
    plt.savefig(
        Path(__file__).parent.parent / "data" / "outputs" / "category_comparison.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("✓ 分类对比图已保存")

    plt.show()

    # 6. Key findings
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)

    # Check if innovators are first
    first_cat = validation['aware_sequence'][0] if validation['aware_sequence'] else None
    if first_cat == 'innovator':
        print("✅ Innovators首先知晓，符合预期")
    else:
        print(f"⚠️  {first_cat.replace('_', ' ').title()}首先知晓，可能需要调整参数")

    # Check adoption rate trend
    adoption_rates = [stats[cat]['adoption_rate'] for cat in ['innovator', 'early_adopter', 'early_majority', 'late_majority', 'laggard'] if cat in stats]
    if len(adoption_rates) >= 2 and adoption_rates[0] >= adoption_rates[-1]:
        print("✅ 采纳率呈递减趋势，符合预期")
    else:
        print("⚠️  采纳率趋势异常")

    # Check delays
    delays = [stats[cat].get('mean_delay', 0) or 0 for cat in ['innovator', 'early_adopter', 'early_majority', 'late_majority', 'laggard'] if cat in stats and stats[cat].get('mean_delay')]
    if len(delays) >= 2 and delays[-1] >= delays[0]:
        print("✅ 决策延迟递增，Laggards更犹豫，符合预期")
    else:
        print("⚠️  决策延迟趋势需要关注")

    print("\n实验完成！")


if __name__ == "__main__":
    main()

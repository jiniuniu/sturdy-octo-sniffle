"""
Visualization functions for diffusion sequence analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from ..core.schemas import SimulationResult


def visualize_diffusion_sequence(tracking_data: Dict, result: SimulationResult, figsize=(18, 10)):
    """
    Create comprehensive visualization of diffusion sequence.

    Generates 6 subplots showing:
    1. Awareness time distribution by category (violin plot)
    2. First awareness time by category (bar chart)
    3. Cumulative awareness curves by category
    4. Adoption time distribution by category (violin plot)
    5. Awareness vs Adoption rates by category
    6. Aware→Adopt delay by category (box plot)

    Args:
        tracking_data: Tracking data from run_tracked_simulation
        result: Simulation result object
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    agent_info = tracking_data['agent_info']
    categories = ['early_adopters', 'early_majority', 'late_majority', 'laggards']
    colors = {
        'early_adopters': '#e74c3c',
        'early_majority': '#2ecc71',
        'late_majority': '#3498db',
        'laggards': '#9b59b6'
    }

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. Awareness time distribution (violin plot)
    aware_data = {}
    for cat in categories:
        aware_times = [
            info['aware_at'] for aid, info in agent_info.items()
            if info['category'] == cat and info['aware_at'] is not None
        ]
        aware_data[cat] = aware_times

    # Only plot if we have data
    plot_data = [aware_data[cat] for cat in categories if aware_data[cat]]
    plot_labels = [cat for cat in categories if aware_data[cat]]

    if plot_data:
        parts = axes[0, 0].violinplot(plot_data, positions=range(len(plot_labels)), showmedians=True)
        axes[0, 0].set_xticks(range(len(plot_labels)))
        axes[0, 0].set_xticklabels([c.replace('_', ' ').title() for c in plot_labels], fontsize=9)
        axes[0, 0].set_ylabel('Awareness Time (step)', fontsize=10)
        axes[0, 0].set_title('Awareness Time Distribution\n(Verify Temporal Sequence)', fontsize=11, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. First aware time comparison
    first_aware = []
    valid_cats = []
    for cat in categories:
        times = [
            info['aware_at'] for aid, info in agent_info.items()
            if info['category'] == cat and info['aware_at'] is not None
        ]
        if times:
            first_aware.append(min(times))
            valid_cats.append(cat)

    if first_aware:
        axes[0, 1].bar(
            range(len(valid_cats)),
            first_aware,
            color=[colors[c] for c in valid_cats],
            alpha=0.7,
            edgecolor='black',
            linewidth=1.5
        )
        axes[0, 1].set_xticks(range(len(valid_cats)))
        axes[0, 1].set_xticklabels([c.replace('_', ' ').title() for c in valid_cats], fontsize=9)
        axes[0, 1].set_ylabel('First Awareness (step)', fontsize=10)
        axes[0, 1].set_title('First Awareness Time by Category\n(Expected: EA → EM → LM → Laggards)', fontsize=11, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate(first_aware):
            axes[0, 1].text(i, v + 0.5, f'{int(v)}', ha='center', va='bottom', fontweight='bold')

    # 3. Cumulative awareness curves (percentage)
    for cat in categories:
        # Get total agents in this category
        total_in_cat = sum(1 for aid, info in agent_info.items() if info['category'] == cat)

        aware_times = sorted([
            info['aware_at'] for aid, info in agent_info.items()
            if info['category'] == cat and info['aware_at'] is not None
        ])
        if aware_times and total_in_cat > 0:
            # Calculate cumulative percentage
            cumulative_pct = [(i + 1) / total_in_cat * 100 for i in range(len(aware_times))]
            axes[0, 2].plot(
                aware_times, cumulative_pct,
                label=cat.replace('_', ' ').title(),
                linewidth=2.5,
                color=colors[cat]
            )

    axes[0, 2].set_xlabel('Time Step', fontsize=10)
    axes[0, 2].set_ylabel('Cumulative Aware (%)', fontsize=10)
    axes[0, 2].set_title('Cumulative Awareness Curves\n(Verify Diffusion Order)', fontsize=11, fontweight='bold')
    axes[0, 2].legend(fontsize=8, loc='best')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Adoption time distribution
    adopt_data = {}
    for cat in categories:
        adopt_times = [
            info['adopted_at'] for aid, info in agent_info.items()
            if info['category'] == cat and info['adopted_at'] is not None
        ]
        adopt_data[cat] = adopt_times

    plot_adopt_data = [adopt_data[cat] for cat in categories if adopt_data[cat]]
    plot_adopt_labels = [cat for cat in categories if adopt_data[cat]]

    if plot_adopt_data:
        parts = axes[1, 0].violinplot(plot_adopt_data, positions=range(len(plot_adopt_labels)), showmedians=True)
        axes[1, 0].set_xticks(range(len(plot_adopt_labels)))
        axes[1, 0].set_xticklabels([c.replace('_', ' ').title() for c in plot_adopt_labels], fontsize=9)
        axes[1, 0].set_ylabel('Adoption Time (step)', fontsize=10)
        axes[1, 0].set_title('Adoption Time Distribution', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 5. Aware vs Adopted rates
    aware_rates = []
    adopt_rates = []
    rate_cats = []

    for cat in categories:
        total = len([1 for aid, info in agent_info.items() if info['category'] == cat])
        if total == 0:
            continue

        aware = len([1 for aid, info in agent_info.items()
                    if info['category'] == cat and info['aware_at'] is not None])
        adopted = len([1 for aid, info in agent_info.items()
                      if info['category'] == cat and info['adopted_at'] is not None])

        aware_rates.append(aware / total * 100)
        adopt_rates.append(adopted / total * 100)
        rate_cats.append(cat)

    if rate_cats:
        x = np.arange(len(rate_cats))
        width = 0.35
        axes[1, 1].bar(x - width/2, aware_rates, width, label='Aware', alpha=0.8, color='#3498db')
        axes[1, 1].bar(x + width/2, adopt_rates, width, label='Adopted', alpha=0.8, color='#2ecc71')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([c.replace('_', ' ').title() for c in rate_cats], fontsize=9)
        axes[1, 1].set_ylabel('Rate (%)', fontsize=10)
        axes[1, 1].set_title('Awareness vs Adoption Rate\n(Expected: Decreasing)', fontsize=11, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim(0, 110)

    # 6. Aware to Adopt delay (box plot)
    delays = {}
    for cat in categories:
        cat_delays = []
        for aid, info in agent_info.items():
            if (info['category'] == cat and
                info['aware_at'] is not None and
                info['adopted_at'] is not None):
                delay = info['adopted_at'] - info['aware_at']
                cat_delays.append(delay)
        delays[cat] = cat_delays

    plot_delays = [delays[cat] for cat in categories if delays[cat]]
    delay_labels = [cat for cat in categories if delays[cat]]

    if plot_delays:
        bp = axes[1, 2].boxplot(
            plot_delays,
            labels=[c.replace('_', ' ').title() for c in delay_labels],
            patch_artist=True
        )

        # Color the boxes
        for patch, cat in zip(bp['boxes'], delay_labels):
            patch.set_facecolor(colors[cat])
            patch.set_alpha(0.7)

        axes[1, 2].set_ylabel('Delay (steps)', fontsize=10)
        axes[1, 2].set_title('Aware → Adopt Delay\n(Decision Speed, Expected: Increasing)', fontsize=11, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_category_comparison(tracking_data: Dict, result: SimulationResult, figsize=(12, 5)):
    """
    Create a focused comparison plot showing key metrics by category.

    Args:
        tracking_data: Tracking data from simulation
        result: Simulation result object
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    agent_info = tracking_data['agent_info']
    categories = ['early_adopters', 'early_majority', 'late_majority', 'laggards']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Calculate metrics
    metrics = []
    valid_cats = []

    for cat in categories:
        agents_in_cat = [aid for aid, info in agent_info.items() if info['category'] == cat]
        if not agents_in_cat:
            continue

        aware_times = [
            info['aware_at'] for aid, info in agent_info.items()
            if info['category'] == cat and info['aware_at'] is not None
        ]

        if not aware_times:
            continue

        metrics.append({
            'category': cat,
            'first_aware': min(aware_times),
            'median_aware': np.median(aware_times),
            'adoption_rate': len([1 for aid, info in agent_info.items()
                                 if info['category'] == cat and info['adopted_at'] is not None]) / len(agents_in_cat) * 100
        })
        valid_cats.append(cat)

    if not metrics:
        return fig

    # Plot 1: First and Median awareness time
    x = np.arange(len(metrics))
    width = 0.35

    first_times = [m['first_aware'] for m in metrics]
    median_times = [m['median_aware'] for m in metrics]

    axes[0].bar(x - width/2, first_times, width, label='First Aware', alpha=0.8, color='#3498db')
    axes[0].bar(x + width/2, median_times, width, label='Median Aware', alpha=0.8, color='#e67e22')

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m['category'].replace('_', ' ').title() for m in metrics], fontsize=9)
    axes[0].set_ylabel('Time Step', fontsize=10)
    axes[0].set_title('Awareness Timing by Category', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Final adoption rate
    adoption_rates = [m['adoption_rate'] for m in metrics]
    axes[1].bar(x, adoption_rates, color=colors[:len(metrics)], alpha=0.7, edgecolor='black', linewidth=1.5)

    for i, v in enumerate(adoption_rates):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m['category'].replace('_', ' ').title() for m in metrics], fontsize=9)
    axes[1].set_ylabel('Adoption Rate (%)', fontsize=10)
    axes[1].set_title('Final Adoption Rate by Category', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 110)

    plt.tight_layout()
    return fig

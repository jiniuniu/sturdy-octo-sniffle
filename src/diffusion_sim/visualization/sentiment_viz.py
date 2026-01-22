"""
Visualization tools for sentiment diffusion simulations.

Provides basic charts for analyzing sentiment spread, polarity distribution,
and category-wise behavior in brand crisis scenarios.
"""
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_sentiment_diffusion_curves(
    snapshots: List,
    title: str = "Sentiment Diffusion Over Time",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot time series of sentiment diffusion metrics.

    Shows three curves:
    - Aware count
    - Interpreted count (formed sentiment)
    - Expressed count (public expression)

    Args:
        snapshots: List of SentimentSnapshot objects
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    steps = [s.step for s in snapshots]
    aware_counts = [s.aware_count for s in snapshots]
    interpreted_counts = [getattr(s, 'interpreted_count', 0) for s in snapshots]
    expressed_counts = [getattr(s, 'expressed_count', 0) for s in snapshots]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(steps, aware_counts, label='Aware', marker='o', linewidth=2, color='#3498db')
    ax.plot(steps, interpreted_counts, label='Interpreted', marker='s', linewidth=2, color='#f39c12')
    ax.plot(steps, expressed_counts, label='Expressed', marker='^', linewidth=2, color='#e74c3c')

    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Number of Agents', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_polarity_over_time(
    snapshots: List,
    title: str = "Sentiment Polarity Over Time",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot sentiment polarity evolution (positive vs negative).

    Args:
        snapshots: List of SentimentSnapshot objects
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    steps = [s.step for s in snapshots]
    positive_counts = [getattr(s, 'positive_count', 0) for s in snapshots]
    negative_counts = [getattr(s, 'negative_count', 0) for s in snapshots]
    mean_polarities = [getattr(s, 'mean_polarity', 0.0) for s in snapshots]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: Positive vs Negative counts
    ax1.plot(steps, positive_counts, label='Positive', marker='o', linewidth=2, color='#2ecc71')
    ax1.plot(steps, negative_counts, label='Negative', marker='o', linewidth=2, color='#e74c3c')
    ax1.set_ylabel('Number of Agents', fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom: Mean polarity
    ax2.plot(steps, mean_polarities, linewidth=2, color='#9b59b6', marker='s')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Mean Polarity', fontsize=11)
    ax2.set_title('Mean Sentiment Polarity', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    return fig


def plot_category_expression_rates(
    category_stats: Dict[str, Dict[str, float]],
    title: str = "Expression Rates by Category",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot expression rates across different population categories.

    Args:
        category_stats: Dictionary mapping category to statistics
            Expected keys: 'expressed_count', 'interpreted_count'
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    categories = list(category_stats.keys())
    expression_rates = []

    for cat in categories:
        stats = category_stats[cat]
        interpreted = stats.get('interpreted_count', 0)
        expressed = stats.get('expressed_count', 0)
        rate = (expressed / interpreted * 100) if interpreted > 0 else 0
        expression_rates.append(rate)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#e74c3c', '#3498db', '#95a5a6', '#f39c12']
    x_pos = np.arange(len(categories))

    bars = ax.bar(x_pos, expression_rates, color=colors[:len(categories)],
                  alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in categories],
                       rotation=15, ha='right')
    ax.set_ylabel('Expression Rate (%)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(expression_rates):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_polarity_distribution(
    graph,
    title: str = "Sentiment Polarity Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot histogram of sentiment polarity values.

    Args:
        graph: Network graph with sentiment agents
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from ..agents.sentiment_agent import SentimentAgent

    polarities = []
    for node_id in graph.nodes():
        polarity = SentimentAgent.get_polarity(graph, node_id)
        if polarity is not None:
            polarities.append(polarity)

    if len(polarities) == 0:
        # No interpreted agents yet
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No interpreted agents yet',
               ha='center', va='center', fontsize=14)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram with color coding
    bins = np.linspace(-1, 1, 21)
    counts, bin_edges, patches = ax.hist(polarities, bins=bins, edgecolor='black',
                                         linewidth=1.2, alpha=0.7)

    # Color bars based on polarity
    for i, patch in enumerate(patches):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        if bin_center < -0.1:
            patch.set_facecolor('#e74c3c')  # Red for negative
        elif bin_center > 0.1:
            patch.set_facecolor('#2ecc71')  # Green for positive
        else:
            patch.set_facecolor('#95a5a6')  # Gray for neutral

    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Polarity', fontsize=11)
    ax.set_ylabel('Number of Agents', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    mean_polarity = np.mean(polarities)
    negative_count = sum(1 for p in polarities if p < -0.1)
    positive_count = sum(1 for p in polarities if p > 0.1)
    neutral_count = len(polarities) - negative_count - positive_count

    stats_text = (f"Mean: {mean_polarity:.3f}\n"
                 f"Negative: {negative_count}\n"
                 f"Neutral: {neutral_count}\n"
                 f"Positive: {positive_count}")
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_sentiment_summary(
    snapshots: List,
    graph,
    category_stats: Optional[Dict[str, Dict[str, float]]] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create comprehensive summary visualization with multiple panels.

    Args:
        snapshots: List of SentimentSnapshot objects
        graph: Network graph with sentiment agents
        category_stats: Optional category statistics
        figsize: Figure size

    Returns:
        Matplotlib figure with subplots
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: Diffusion curves
    ax1 = fig.add_subplot(gs[0, 0])
    steps = [s.step for s in snapshots]
    aware_counts = [s.aware_count for s in snapshots]
    interpreted_counts = [getattr(s, 'interpreted_count', 0) for s in snapshots]
    expressed_counts = [getattr(s, 'expressed_count', 0) for s in snapshots]

    ax1.plot(steps, aware_counts, label='Aware', marker='o', linewidth=2, color='#3498db')
    ax1.plot(steps, interpreted_counts, label='Interpreted', marker='s', linewidth=2, color='#f39c12')
    ax1.plot(steps, expressed_counts, label='Expressed', marker='^', linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Time Step', fontsize=10)
    ax1.set_ylabel('Number of Agents', fontsize=10)
    ax1.set_title('Sentiment Diffusion Over Time', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Top-right: Polarity over time
    ax2 = fig.add_subplot(gs[0, 1])
    positive_counts = [getattr(s, 'positive_count', 0) for s in snapshots]
    negative_counts = [getattr(s, 'negative_count', 0) for s in snapshots]
    ax2.plot(steps, positive_counts, label='Positive', marker='o', linewidth=2, color='#2ecc71')
    ax2.plot(steps, negative_counts, label='Negative', marker='o', linewidth=2, color='#e74c3c')
    ax2.set_xlabel('Time Step', fontsize=10)
    ax2.set_ylabel('Number of Agents', fontsize=10)
    ax2.set_title('Positive vs Negative Sentiment', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Polarity distribution
    ax3 = fig.add_subplot(gs[1, 0])
    from ..agents.sentiment_agent import SentimentAgent
    polarities = []
    for node_id in graph.nodes():
        polarity = SentimentAgent.get_polarity(graph, node_id)
        if polarity is not None:
            polarities.append(polarity)

    if len(polarities) > 0:
        bins = np.linspace(-1, 1, 21)
        counts, bin_edges, patches = ax3.hist(polarities, bins=bins, edgecolor='black',
                                              linewidth=1, alpha=0.7)
        for i, patch in enumerate(patches):
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            if bin_center < -0.1:
                patch.set_facecolor('#e74c3c')
            elif bin_center > 0.1:
                patch.set_facecolor('#2ecc71')
            else:
                patch.set_facecolor('#95a5a6')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax3.set_xlabel('Polarity', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Polarity Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Bottom-right: Category expression rates (if provided)
    ax4 = fig.add_subplot(gs[1, 1])
    if category_stats:
        categories = list(category_stats.keys())
        expression_rates = []
        for cat in categories:
            stats = category_stats[cat]
            interpreted = stats.get('interpreted_count', 0)
            expressed = stats.get('expressed_count', 0)
            rate = (expressed / interpreted * 100) if interpreted > 0 else 0
            expression_rates.append(rate)

        colors = ['#e74c3c', '#3498db', '#95a5a6', '#f39c12']
        x_pos = np.arange(len(categories))
        ax4.bar(x_pos, expression_rates, color=colors[:len(categories)],
               alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([c.replace('_', ' ').title() for c in categories],
                           rotation=15, ha='right', fontsize=9)
        ax4.set_ylabel('Expression Rate (%)', fontsize=10)
        ax4.set_title('Expression Rates by Category', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(expression_rates):
            ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'Category statistics not provided',
                ha='center', va='center', fontsize=11)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

    plt.suptitle('Sentiment Diffusion Summary', fontsize=14, fontweight='bold', y=0.995)
    return fig

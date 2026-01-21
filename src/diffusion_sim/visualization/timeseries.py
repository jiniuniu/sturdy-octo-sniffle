"""
Time series visualization for diffusion metrics.
"""

from typing import List, Optional

import matplotlib.pyplot as plt

from ..core.schemas import SimulationResult


class TimeSeriesVisualizer:
    """Visualizer for diffusion time series data."""

    @staticmethod
    def plot_diffusion_curves(
        result: SimulationResult,
        ax: Optional[plt.Axes] = None,
        show_share: bool = False,
    ) -> plt.Axes:
        """
        Plot awareness and adoption curves over time.

        Args:
            result: Simulation result object
            ax: Matplotlib axes (if None, creates new)
            show_share: Whether to show share count

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        steps = [s.step for s in result.snapshots]
        aware_counts = result.get_timeseries("aware")
        adopted_counts = result.get_timeseries("adopted")

        # Plot curves
        ax.plot(steps, aware_counts, label="Aware", color="orange", linewidth=2)
        ax.plot(steps, adopted_counts, label="Adopted", color="green", linewidth=2)

        if show_share:
            shared_counts = result.get_timeseries("shared")
            ax.plot(
                steps, shared_counts, label="Shares (per step)", color="blue", alpha=0.6
            )

        # Formatting
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Diffusion Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Add final stats as text
        total_nodes = result.network_config.n
        final_text = (
            f"Final: {result.final_adoption_rate:.1%} adopted "
            f"({result.snapshots[-1].adopted_count}/{total_nodes})"
        )
        ax.text(
            0.02,
            0.98,
            final_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        return ax

    @staticmethod
    def plot_adoption_rate(
        result: SimulationResult, ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot adoption rate (as percentage) over time.

        Args:
            result: Simulation result object
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        steps = [s.step for s in result.snapshots]
        total_nodes = result.network_config.n
        adoption_rates = [
            (s.adopted_count / total_nodes) * 100 for s in result.snapshots
        ]

        ax.plot(steps, adoption_rates, color="green", linewidth=2, marker="o")
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Adoption Rate (%)", fontsize=12)
        ax.set_title("Adoption Rate Over Time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        return ax

    @staticmethod
    def plot_comparison(
        results: List[SimulationResult],
        labels: List[str],
        metric: str = "adopted",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Compare multiple simulation runs.

        Args:
            results: List of simulation results
            labels: Labels for each result
            metric: Metric to compare ("aware", "adopted", "shared")
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(range(len(results)))

        for i, (result, label) in enumerate(zip(results, labels)):
            steps = [s.step for s in result.snapshots]
            values = result.get_timeseries(metric)
            ax.plot(steps, values, label=label, color=colors[i], linewidth=2)

        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel(f"{metric.capitalize()} Count", fontsize=12)
        ax.set_title(
            f"Comparison: {metric.capitalize()}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def create_summary_plot(result: SimulationResult, save_path: Optional[str] = None):
        """
        Create a comprehensive summary plot with multiple subplots.

        Args:
            result: Simulation result object
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Plot 1: Diffusion curves
        ax1 = fig.add_subplot(gs[0, :])
        TimeSeriesVisualizer.plot_diffusion_curves(result, ax=ax1, show_share=True)

        # Plot 2: Adoption rate
        ax2 = fig.add_subplot(gs[1, 0])
        TimeSeriesVisualizer.plot_adoption_rate(result, ax=ax2)

        # Plot 3: Network stats summary
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")

        # Summary statistics
        stats_text = f"""
        Network Configuration:
        - Nodes: {result.network_config.n}
        - Avg Degree: {result.network_config.k}
        - Rewire Prob: {result.network_config.p}

        Simulation Results:
        - Total Steps: {result.total_steps}
        - Initial Adopters: {result.config.initial_adopters}
        - Final Awareness: {result.final_awareness_rate:.1%}
        - Final Adoption: {result.final_adoption_rate:.1%}

        Diffusion Speed:
        - Steps to 50%: {TimeSeriesVisualizer._steps_to_threshold(result, 0.5)}
        - Steps to 90%: {TimeSeriesVisualizer._steps_to_threshold(result, 0.9)}
        """

        ax3.text(
            0.1,
            0.9,
            stats_text,
            transform=ax3.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=10,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def _steps_to_threshold(result: SimulationResult, threshold: float) -> str:
        """Helper to find steps to reach adoption threshold."""
        total = result.network_config.n
        for snapshot in result.snapshots:
            if snapshot.adopted_count / total >= threshold:
                return str(snapshot.step)
        return "N/A"

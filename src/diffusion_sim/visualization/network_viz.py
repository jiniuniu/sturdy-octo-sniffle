"""
Network visualization for diffusion simulation.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx

from ..agents.base import BaseAgent


class NetworkVisualizer:
    """Visualizer for network graphs with agent states."""

    def __init__(self, graph: nx.Graph, layout: Optional[Dict] = None):
        """
        Initialize network visualizer.

        Args:
            graph: Network graph
            layout: Pre-computed layout (if None, will compute spring layout)
        """
        self.graph = graph
        self.layout = layout or nx.spring_layout(graph, seed=42)

    def draw(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Network State",
        show_labels: bool = False,
    ) -> plt.Axes:
        """
        Draw network with nodes colored by state.

        Color scheme:
        - Gray: unaware
        - Yellow: aware but not adopted
        - Green: adopted

        Args:
            ax: Matplotlib axes (if None, creates new)
            title: Plot title
            show_labels: Whether to show node labels

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Categorize nodes by state
        unaware = []
        aware_only = []
        adopted = []

        for node_id in self.graph.nodes():
            state = BaseAgent.get_state(self.graph, node_id)
            if state.adopted:
                adopted.append(node_id)
            elif state.aware:
                aware_only.append(node_id)
            else:
                unaware.append(node_id)

        # Draw edges first
        nx.draw_networkx_edges(
            self.graph, self.layout, ax=ax, alpha=0.2, edge_color="gray"
        )

        # Draw nodes by category
        node_size = 100 if len(self.graph.nodes()) > 100 else 300

        if unaware:
            nx.draw_networkx_nodes(
                self.graph,
                self.layout,
                nodelist=unaware,
                node_color="lightgray",
                node_size=node_size,
                ax=ax,
                label="Unaware",
            )

        if aware_only:
            nx.draw_networkx_nodes(
                self.graph,
                self.layout,
                nodelist=aware_only,
                node_color="yellow",
                node_size=node_size,
                ax=ax,
                label="Aware",
            )

        if adopted:
            nx.draw_networkx_nodes(
                self.graph,
                self.layout,
                nodelist=adopted,
                node_color="limegreen",
                node_size=node_size,
                ax=ax,
                label="Adopted",
            )

        # Optional labels
        if show_labels and len(self.graph.nodes()) <= 50:
            nx.draw_networkx_labels(self.graph, self.layout, ax=ax, font_size=8)

        ax.set_title(title)
        ax.axis("off")
        ax.legend(loc="upper right")

        return ax

    def draw_snapshot(self, step: int, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Draw network at a specific simulation step.

        Args:
            step: Step number
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        return self.draw(ax=ax, title=f"Network State - Step {step}")

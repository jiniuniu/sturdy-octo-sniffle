"""
Social Norm (SN) computation strategies.

In TPB, social norm captures the perceived social pressure to perform a behavior.
In diffusion context, it's calculated from neighbors' adoption states.
"""
from typing import List
import networkx as nx
from ...agents.base import BaseAgent


class SocialNormAggregator:
    """
    Aggregator for computing Social Norm from network neighbors.

    Supports multiple aggregation strategies:
    - mean: Average adoption rate among neighbors
    - weighted: Weighted by neighbor influence (future extension)
    - threshold: Binary based on threshold
    """

    def __init__(self, strategy: str = "mean"):
        """
        Initialize aggregator.

        Args:
            strategy: Aggregation strategy ("mean", "threshold")
        """
        self.strategy = strategy

    def compute(
        self,
        graph: nx.Graph,
        agent_id: int,
        threshold: float = 0.5
    ) -> float:
        """
        Compute social norm score for an agent.

        Args:
            graph: Network graph
            agent_id: ID of the agent
            threshold: Threshold for binary strategy (default 0.5)

        Returns:
            Social norm score (0-1)
        """
        neighbors = BaseAgent.get_neighbors(graph, agent_id)

        if len(neighbors) == 0:
            return 0.0

        # Count adopted neighbors
        adopted_count = 0
        for neighbor_id in neighbors:
            neighbor_state = BaseAgent.get_state(graph, neighbor_id)
            if neighbor_state.adopted:
                adopted_count += 1

        adoption_ratio = adopted_count / len(neighbors)

        if self.strategy == "mean":
            # Simple average
            return adoption_ratio

        elif self.strategy == "threshold":
            # Binary: 1 if above threshold, else 0
            return 1.0 if adoption_ratio >= threshold else 0.0

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def compute_weighted(
        self,
        graph: nx.Graph,
        agent_id: int,
        neighbor_weights: dict
    ) -> float:
        """
        Compute weighted social norm (for future use with influence weights).

        Args:
            graph: Network graph
            agent_id: ID of the agent
            neighbor_weights: Dict mapping neighbor_id -> weight

        Returns:
            Weighted social norm score (0-1)
        """
        neighbors = BaseAgent.get_neighbors(graph, agent_id)

        if len(neighbors) == 0:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for neighbor_id in neighbors:
            weight = neighbor_weights.get(neighbor_id, 1.0)
            neighbor_state = BaseAgent.get_state(graph, neighbor_id)

            total_weight += weight
            if neighbor_state.adopted:
                weighted_sum += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

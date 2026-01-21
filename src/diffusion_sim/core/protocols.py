"""
Protocol definitions for pluggable components.
"""
from typing import Protocol, runtime_checkable
import networkx as nx
from .schemas import NetworkConfig, AgentState


@runtime_checkable
class NetworkBuilder(Protocol):
    """Protocol for network generation strategies."""

    def build(self, config: NetworkConfig) -> nx.Graph:
        """
        Build a network graph based on configuration.

        Args:
            config: Network configuration parameters

        Returns:
            NetworkX graph with initialized node attributes
        """
        ...


@runtime_checkable
class DecisionModel(Protocol):
    """Protocol for agent decision-making models."""

    def compute_intention(
        self, agent_state: AgentState, graph: nx.Graph, agent_id: int
    ) -> float:
        """
        Compute the intention score for an agent.

        Args:
            agent_state: Current state of the agent
            graph: The network graph
            agent_id: ID of the agent

        Returns:
            Intention score (0-1)
        """
        ...

    def decide_adopt(
        self, agent_state: AgentState, graph: nx.Graph, agent_id: int
    ) -> bool:
        """
        Decide whether the agent should adopt.

        Args:
            agent_state: Current state of the agent
            graph: The network graph
            agent_id: ID of the agent

        Returns:
            True if agent decides to adopt
        """
        ...

    def decide_share(
        self,
        agent_state: AgentState,
        graph: nx.Graph,
        agent_id: int,
        neighbor_id: int,
    ) -> bool:
        """
        Decide whether to share with a specific neighbor.

        Args:
            agent_state: Current state of the agent
            graph: The network graph
            agent_id: ID of the agent
            neighbor_id: ID of the neighbor

        Returns:
            True if agent decides to share
        """
        ...

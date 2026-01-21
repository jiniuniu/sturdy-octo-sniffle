"""
Base agent implementation for diffusion simulation.
"""
from typing import Optional
import networkx as nx
from ..core.schemas import AgentState
from ..core.exceptions import AgentError


class BaseAgent:
    """
    Base agent class managing state transitions in diffusion process.

    This class provides basic state management for awareness, adoption,
    and sharing behavior. Decision logic is delegated to DecisionModel.
    """

    @staticmethod
    def get_state(graph: nx.Graph, agent_id: int) -> AgentState:
        """
        Retrieve agent state from graph.

        Args:
            graph: Network graph
            agent_id: ID of the agent

        Returns:
            Agent state object

        Raises:
            AgentError: If agent state not found
        """
        if agent_id not in graph.nodes:
            raise AgentError(f"Agent {agent_id} not found in graph")

        agent_state = graph.nodes[agent_id].get("agent_state")
        if agent_state is None:
            raise AgentError(f"Agent {agent_id} has no agent_state attribute")

        return agent_state

    @staticmethod
    def set_state(graph: nx.Graph, agent_id: int, state: AgentState) -> None:
        """
        Update agent state in graph.

        Args:
            graph: Network graph
            agent_id: ID of the agent
            state: New agent state
        """
        if agent_id not in graph.nodes:
            raise AgentError(f"Agent {agent_id} not found in graph")

        graph.nodes[agent_id]["agent_state"] = state

    @staticmethod
    def make_aware(graph: nx.Graph, agent_id: int) -> None:
        """
        Mark agent as aware of the innovation.

        Args:
            graph: Network graph
            agent_id: ID of the agent
        """
        state = BaseAgent.get_state(graph, agent_id)
        state.aware = True
        BaseAgent.set_state(graph, agent_id, state)

    @staticmethod
    def make_adopted(graph: nx.Graph, agent_id: int) -> None:
        """
        Mark agent as having adopted the innovation.

        Args:
            graph: Network graph
            agent_id: ID of the agent
        """
        state = BaseAgent.get_state(graph, agent_id)
        if not state.aware:
            raise AgentError(
                f"Agent {agent_id} cannot adopt without being aware first"
            )
        state.adopted = True
        BaseAgent.set_state(graph, agent_id, state)

    @staticmethod
    def mark_shared_with(graph: nx.Graph, agent_id: int, neighbor_id: int) -> None:
        """
        Record that agent has shared with a neighbor.

        Args:
            graph: Network graph
            agent_id: ID of the agent
            neighbor_id: ID of the neighbor
        """
        state = BaseAgent.get_state(graph, agent_id)
        state.shared_with.add(neighbor_id)
        BaseAgent.set_state(graph, agent_id, state)

    @staticmethod
    def has_shared_with(graph: nx.Graph, agent_id: int, neighbor_id: int) -> bool:
        """
        Check if agent has already shared with a neighbor.

        Args:
            graph: Network graph
            agent_id: ID of the agent
            neighbor_id: ID of the neighbor

        Returns:
            True if already shared with this neighbor
        """
        state = BaseAgent.get_state(graph, agent_id)
        return neighbor_id in state.shared_with

    @staticmethod
    def get_neighbors(graph: nx.Graph, agent_id: int) -> list:
        """
        Get list of neighbor IDs.

        Args:
            graph: Network graph
            agent_id: ID of the agent

        Returns:
            List of neighbor IDs
        """
        return list(graph.neighbors(agent_id))

    @staticmethod
    def count_adopted_neighbors(graph: nx.Graph, agent_id: int) -> int:
        """
        Count how many neighbors have adopted.

        Args:
            graph: Network graph
            agent_id: ID of the agent

        Returns:
            Number of adopted neighbors
        """
        neighbors = BaseAgent.get_neighbors(graph, agent_id)
        count = 0
        for neighbor_id in neighbors:
            neighbor_state = BaseAgent.get_state(graph, neighbor_id)
            if neighbor_state.adopted:
                count += 1
        return count

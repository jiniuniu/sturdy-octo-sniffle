"""
TPB-based agent implementation.

Extends BaseAgent with TPB-specific traits and behaviors.
"""
from typing import Dict, Optional
import networkx as nx
from ..core.schemas import AgentState
from .base import BaseAgent


class TPBAgent(BaseAgent):
    """
    Agent with TPB (Theory of Planned Behavior) traits.

    Traits:
    - attitude: Personal evaluation of the innovation (0-1)
    - pbc: Perceived Behavioral Control - ease of adoption (0-1)
    - conformity: Sensitivity to social norms (0-1)
    - risk_aversion: Aversion to risk/uncertainty (0-1)
    - share_propensity: Willingness to share with others (0-1)
    - innovativeness: Early adopter tendency (0-1, optional)
    """

    @staticmethod
    def initialize_agent(
        graph: nx.Graph,
        agent_id: int,
        traits: Dict[str, float]
    ) -> None:
        """
        Initialize agent with TPB traits.

        Args:
            graph: Network graph
            agent_id: Agent ID
            traits: Dictionary of trait values
        """
        # Validate required traits
        required = ["attitude", "pbc", "conformity", "risk_aversion", "share_propensity"]
        for trait in required:
            if trait not in traits:
                raise ValueError(f"Missing required trait: {trait}")
            if not 0 <= traits[trait] <= 1:
                raise ValueError(f"Trait {trait} must be in [0, 1], got {traits[trait]}")

        # Create agent state with traits
        agent_state = AgentState(agent_id=agent_id, traits=traits)
        graph.nodes[agent_id]["agent_state"] = agent_state

    @staticmethod
    def get_trait(graph: nx.Graph, agent_id: int, trait_name: str) -> float:
        """
        Get specific trait value for an agent.

        Args:
            graph: Network graph
            agent_id: Agent ID
            trait_name: Name of the trait

        Returns:
            Trait value
        """
        state = BaseAgent.get_state(graph, agent_id)
        if state.traits is None:
            raise ValueError(f"Agent {agent_id} has no traits")

        if trait_name not in state.traits:
            raise KeyError(f"Agent {agent_id} does not have trait '{trait_name}'")

        return state.traits[trait_name]

    @staticmethod
    def update_trait(
        graph: nx.Graph,
        agent_id: int,
        trait_name: str,
        new_value: float
    ) -> None:
        """
        Update a specific trait value (for dynamic traits).

        Args:
            graph: Network graph
            agent_id: Agent ID
            trait_name: Name of the trait
            new_value: New value (0-1)
        """
        if not 0 <= new_value <= 1:
            raise ValueError(f"Trait value must be in [0, 1], got {new_value}")

        state = BaseAgent.get_state(graph, agent_id)
        if state.traits is None:
            raise ValueError(f"Agent {agent_id} has no traits")

        state.traits[trait_name] = new_value
        BaseAgent.set_state(graph, agent_id, state)

    @staticmethod
    def get_all_traits(graph: nx.Graph, agent_id: int) -> Dict[str, float]:
        """
        Get all traits for an agent.

        Args:
            graph: Network graph
            agent_id: Agent ID

        Returns:
            Dictionary of all traits
        """
        state = BaseAgent.get_state(graph, agent_id)
        if state.traits is None:
            raise ValueError(f"Agent {agent_id} has no traits")

        return state.traits.copy()

    @staticmethod
    def compute_attitude_from_traits(traits: Dict[str, float]) -> float:
        """
        Compute attitude score from agent traits.

        This is a helper for agents whose attitude might be derived
        from multiple underlying factors.

        Args:
            traits: Agent traits dictionary

        Returns:
            Computed attitude score (0-1)
        """
        # For now, just return the attitude trait
        # In future, could combine innovativeness, benefit perception, etc.
        return traits.get("attitude", 0.5)

    @staticmethod
    def categorize_agent(traits: Dict[str, float]) -> str:
        """
        Categorize agent by Rogers' diffusion categories (4 categories).

        Based on innovativeness and risk aversion traits.

        Args:
            traits: Agent traits dictionary

        Returns:
            Category: "early_adopters", "early_majority",
                     "late_majority", or "laggards"
        """
        innovativeness = traits.get("innovativeness", 0.5)
        risk_aversion = traits.get("risk_aversion", 0.5)

        # Combine scores (high innovativeness, low risk aversion = earlier adopter)
        score = (innovativeness + (1 - risk_aversion)) / 2

        if score >= 0.65:
            return "early_adopters"
        elif score >= 0.5:
            return "early_majority"
        elif score >= 0.35:
            return "late_majority"
        else:
            return "laggards"

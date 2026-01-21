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
        traits: Dict[str, float],
        category: Optional[str] = None
    ) -> None:
        """
        Initialize agent with TPB traits and Rogers category.

        Args:
            graph: Network graph
            agent_id: Agent ID
            traits: Dictionary of trait values
            category: Rogers' category (early_adopters, early_majority, late_majority, laggards)
        """
        # Validate required traits
        required = ["attitude", "pbc", "conformity", "risk_aversion", "share_propensity"]
        for trait in required:
            if trait not in traits:
                raise ValueError(f"Missing required trait: {trait}")
            if not 0 <= traits[trait] <= 1:
                raise ValueError(f"Trait {trait} must be in [0, 1], got {traits[trait]}")

        # Create agent state with traits and category
        agent_state = AgentState(agent_id=agent_id, traits=traits, category=category)
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
    def get_category(graph: nx.Graph, agent_id: int) -> Optional[str]:
        """
        Get Rogers' category for an agent.

        Returns the category assigned during population sampling.
        If not available, returns None (legacy agents).

        Args:
            graph: Network graph
            agent_id: Agent ID

        Returns:
            Category string or None
        """
        state = BaseAgent.get_state(graph, agent_id)
        return state.category

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

        NOTE: This is a fallback method for legacy agents. New agents should
        have their category assigned during population sampling and stored in
        AgentState.category. Use TPBAgent.get_category() to retrieve it.

        This method infers category from innovativeness trait.

        Thresholds are set to align with Gaussian mean values:
        - EA: mean=0.85, std=0.10 → threshold >= 0.70
        - EM: mean=0.55, std=0.10 → threshold >= 0.45
        - LM: mean=0.35, std=0.10 → threshold >= 0.25
        - Laggards: mean=0.15, std=0.10 → threshold < 0.25

        Args:
            traits: Agent traits dictionary

        Returns:
            Category: "early_adopters", "early_majority",
                     "late_majority", or "laggards"
        """
        innovativeness = traits.get("innovativeness", 0.5)

        # Categorize based on innovativeness thresholds
        if innovativeness >= 0.70:
            return "early_adopters"
        elif innovativeness >= 0.45:
            return "early_majority"
        elif innovativeness >= 0.25:
            return "late_majority"
        else:
            return "laggards"

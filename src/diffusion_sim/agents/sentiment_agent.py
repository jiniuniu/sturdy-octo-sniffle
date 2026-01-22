"""
Sentiment agent implementation for brand crisis diffusion.

Extends BaseAgent with sentiment-specific traits and behaviors.
"""
from typing import Dict, Optional, Tuple
import networkx as nx
from ..core.schemas import AgentState
from .base import BaseAgent


class SentimentAgent(BaseAgent):
    """
    Agent with sentiment diffusion traits.

    Traits:
    - attitude_baseline: Base attitude toward brand before crisis (0-1)
    - moral_sensitivity: Sensitivity to moral/ethical issues (0-1)
    - conformity_sensitivity: Sensitivity to social influence (0-1)
    - expression_threshold: Threshold for public expression (0-1)
    - social_risk_aversion: Aversion to expressing controversial views (0-1)
    - negativity_bias: Tendency to focus on negative information (0-1)
    - share_propensity: Willingness to share content (0-1)
    """

    @staticmethod
    def initialize_agent(
        graph: nx.Graph,
        agent_id: int,
        traits: Dict[str, float],
        category: Optional[str] = None
    ) -> None:
        """
        Initialize agent with sentiment traits and population category.

        Args:
            graph: Network graph
            agent_id: Agent ID
            traits: Dictionary of trait values
            category: Population type (strong_critics, conformists,
                     silent_observers, amplifiers)
        """
        # Validate required traits
        required = [
            "attitude_baseline",
            "moral_sensitivity",
            "conformity_sensitivity",
            "expression_threshold",
            "social_risk_aversion",
            "negativity_bias",
            "share_propensity"
        ]
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
        Get population category for an agent.

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
    def make_interpreted(graph: nx.Graph, agent_id: int, polarity: float) -> None:
        """
        Mark agent as having interpreted the crisis and formed sentiment.

        Args:
            graph: Network graph
            agent_id: Agent ID
            polarity: Sentiment polarity (-1 to 1, negative to positive)
        """
        if not -1 <= polarity <= 1:
            raise ValueError(f"Polarity must be in [-1, 1], got {polarity}")

        state = BaseAgent.get_state(graph, agent_id)
        if not state.aware:
            raise ValueError(
                f"Agent {agent_id} cannot interpret without being aware first"
            )
        state.interpreted = True
        state.polarity = polarity
        BaseAgent.set_state(graph, agent_id, state)

    @staticmethod
    def make_expressed(graph: nx.Graph, agent_id: int) -> None:
        """
        Mark agent as having publicly expressed their sentiment.

        Args:
            graph: Network graph
            agent_id: Agent ID
        """
        state = BaseAgent.get_state(graph, agent_id)
        if not state.interpreted:
            raise ValueError(
                f"Agent {agent_id} cannot express without interpreting first"
            )
        state.expressed = True
        BaseAgent.set_state(graph, agent_id, state)

    @staticmethod
    def get_polarity(graph: nx.Graph, agent_id: int) -> Optional[float]:
        """
        Get sentiment polarity for an agent.

        Args:
            graph: Network graph
            agent_id: Agent ID

        Returns:
            Polarity value (-1 to 1) or None if not interpreted
        """
        state = BaseAgent.get_state(graph, agent_id)
        return state.polarity

    @staticmethod
    def count_neighbors_by_sentiment(
        graph: nx.Graph,
        agent_id: int
    ) -> Tuple[int, int, int]:
        """
        Count neighbors by sentiment polarity (expressed agents only).

        Args:
            graph: Network graph
            agent_id: Agent ID

        Returns:
            Tuple of (positive_count, negative_count, unexpressed_count)
            - positive_count: Number of neighbors with polarity > 0 who expressed
            - negative_count: Number of neighbors with polarity < 0 who expressed
            - unexpressed_count: Number of neighbors who haven't expressed
        """
        neighbors = BaseAgent.get_neighbors(graph, agent_id)
        positive_count = 0
        negative_count = 0
        unexpressed_count = 0

        for neighbor_id in neighbors:
            neighbor_state = BaseAgent.get_state(graph, neighbor_id)
            if neighbor_state.expressed and neighbor_state.polarity is not None:
                if neighbor_state.polarity > 0:
                    positive_count += 1
                elif neighbor_state.polarity < 0:
                    negative_count += 1
                # polarity == 0 is neutral, counted in neither
            else:
                unexpressed_count += 1

        return positive_count, negative_count, unexpressed_count

    @staticmethod
    def compute_directional_social_norm(
        graph: nx.Graph,
        agent_id: int,
        alpha: float = 0.3,
        beta: float = 0.7
    ) -> float:
        """
        Compute directional social norm from neighborhood sentiment.

        Implements asymmetric weighting where negative sentiment has
        stronger influence (negativity bias in social perception).

        SN_directional = (α * N_pos - β * N_neg) / N_total

        Args:
            graph: Network graph
            agent_id: Agent ID
            alpha: Weight for positive sentiment (default 0.3)
            beta: Weight for negative sentiment (default 0.7, β > α)

        Returns:
            Social norm score (-1 to 1)
            - Positive: Net positive social pressure
            - Negative: Net negative social pressure
            - Zero: Balanced or no expressed neighbors
        """
        positive_count, negative_count, unexpressed_count = \
            SentimentAgent.count_neighbors_by_sentiment(graph, agent_id)

        total_expressed = positive_count + negative_count

        if total_expressed == 0:
            return 0.0  # No social norm signal

        # Directional social norm with asymmetric weighting
        numerator = alpha * positive_count - beta * negative_count
        sn_directional = numerator / total_expressed

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sn_directional))

    @staticmethod
    def categorize_agent(traits: Dict[str, float]) -> str:
        """
        Categorize agent by population type (4 categories).

        NOTE: This is a fallback method for legacy agents. New agents should
        have their category assigned during population sampling and stored in
        AgentState.category. Use SentimentAgent.get_category() to retrieve it.

        This method infers category from expression_threshold and moral_sensitivity.

        Categories:
        - strong_critics: High moral sensitivity, low expression threshold
        - conformists: High conformity sensitivity, moderate expression threshold
        - silent_observers: High expression threshold (unlikely to express)
        - amplifiers: High share propensity, low social risk aversion

        Args:
            traits: Agent traits dictionary

        Returns:
            Category: "strong_critics", "conformists",
                     "silent_observers", or "amplifiers"
        """
        moral_sensitivity = traits.get("moral_sensitivity", 0.5)
        expression_threshold = traits.get("expression_threshold", 0.5)
        conformity_sensitivity = traits.get("conformity_sensitivity", 0.5)
        share_propensity = traits.get("share_propensity", 0.5)
        social_risk_aversion = traits.get("social_risk_aversion", 0.5)

        # Strong critics: High moral sensitivity + low expression threshold
        if moral_sensitivity >= 0.7 and expression_threshold <= 0.3:
            return "strong_critics"

        # Amplifiers: High share propensity + low social risk aversion
        if share_propensity >= 0.7 and social_risk_aversion <= 0.3:
            return "amplifiers"

        # Silent observers: High expression threshold
        if expression_threshold >= 0.7:
            return "silent_observers"

        # Conformists: Default category (moderate on all dimensions)
        return "conformists"

"""
Sentiment-based decision model for brand crisis diffusion.

Implements TPB framework adapted for sentiment diffusion:
1. Interpretation: aware → interpreted (form attitude based on traits and social context)
2. Polarity: Determine sentiment direction (negative/positive)
3. Expression: interpreted → expressed (public expression decision)
4. Sharing: expressed → share with neighbors

Key differences from innovation diffusion:
- Directional Social Norm: Positive and negative sentiment weighted differently (β > α)
- Expression Threshold: Barrier to public expression based on social risk
- Polarity Determination: Continuous [-1, 1] based on attitude and social influence
"""
import random
import math
from typing import Optional
import networkx as nx
from ..core.schemas import AgentState
from ..agents.sentiment_agent import SentimentAgent


class SentimentDecisionModel:
    """
    Complete sentiment-based decision model for brand crisis diffusion.

    State transitions:
    1. aware → interpreted (with polarity determination)
    2. interpreted → expressed (public expression)
    3. expressed → share with neighbors
    """

    def __init__(
        self,
        w_attitude: float = 0.4,
        w_social_norm: float = 0.35,
        w_moral: float = 0.25,
        alpha: float = 0.3,
        beta: float = 0.7,
        expression_mode: str = "threshold",
        share_threshold: float = 0.7,
    ):
        """
        Initialize sentiment decision model.

        Args:
            w_attitude: Weight for attitude baseline component
            w_social_norm: Weight for social norm component
            w_moral: Weight for moral sensitivity component
            alpha: Weight for positive sentiment in social norm (< beta)
            beta: Weight for negative sentiment in social norm (> alpha, negativity bias)
            expression_mode: Expression decision mode ("threshold", "probabilistic")
            share_threshold: Minimum sentiment intensity to share (0-1)
        """
        # Validate weights sum to 1
        total = w_attitude + w_social_norm + w_moral
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

        self.w_attitude = w_attitude
        self.w_social_norm = w_social_norm
        self.w_moral = w_moral

        # Directional social norm weights
        if beta <= alpha:
            raise ValueError(f"Negativity bias requires β > α, got α={alpha}, β={beta}")
        self.alpha = alpha
        self.beta = beta

        self.expression_mode = expression_mode
        self.share_threshold = share_threshold

    def compute_interpretation_score(
        self,
        agent_state: AgentState,
        graph: nx.Graph,
        agent_id: int
    ) -> float:
        """
        Compute interpretation/sentiment formation score.

        Combines:
        - Attitude baseline (personal predisposition)
        - Moral sensitivity (reaction to crisis severity)
        - Directional social norm (neighborhood sentiment)

        Args:
            agent_state: Current agent state
            graph: Network graph
            agent_id: Agent ID

        Returns:
            Interpretation score (0-1, higher = more likely to interpret)
        """
        if agent_state.traits is None:
            raise ValueError(f"Agent {agent_id} has no traits defined")

        traits = agent_state.traits

        # Get components
        attitude_baseline = traits.get("attitude_baseline", 0.5)
        moral_sensitivity = traits.get("moral_sensitivity", 0.5)

        # Compute directional social norm
        sn_directional = SentimentAgent.compute_directional_social_norm(
            graph, agent_id, self.alpha, self.beta
        )

        # Convert directional SN [-1, 1] to interpretation likelihood [0, 1]
        # More extreme sentiment in neighborhood → higher interpretation likelihood
        sn_magnitude = abs(sn_directional)

        # Weighted combination
        score = (
            self.w_attitude * attitude_baseline +
            self.w_social_norm * sn_magnitude +
            self.w_moral * moral_sensitivity
        )

        return score

    def decide_polarity(
        self,
        agent_state: AgentState,
        graph: nx.Graph,
        agent_id: int
    ) -> float:
        """
        Determine sentiment polarity (direction).

        Polarity is influenced by:
        - Attitude baseline (positive = pro-brand, negative = critical)
        - Directional social norm (neighborhood sentiment)
        - Negativity bias (stronger weight on negative neighbors)

        Args:
            agent_state: Current agent state
            graph: Network graph
            agent_id: Agent ID

        Returns:
            Polarity value (-1 to 1)
            - Negative: Critical/negative sentiment
            - Positive: Supportive/positive sentiment
        """
        if agent_state.traits is None:
            raise ValueError(f"Agent {agent_id} has no traits defined")

        traits = agent_state.traits

        # Attitude baseline: convert [0, 1] to [-1, 1]
        # 0 = very negative, 0.5 = neutral, 1 = very positive
        attitude_baseline = traits.get("attitude_baseline", 0.5)
        attitude_polarity = (attitude_baseline - 0.5) * 2  # Map to [-1, 1]

        # Directional social norm (already in [-1, 1])
        sn_directional = SentimentAgent.compute_directional_social_norm(
            graph, agent_id, self.alpha, self.beta
        )

        # Conformity sensitivity
        conformity_sensitivity = traits.get("conformity_sensitivity", 0.5)

        # Negativity bias (individual trait)
        negativity_bias = traits.get("negativity_bias", 0.5)

        # Combine attitude and social norm
        # Higher conformity → more weight on social norm
        w_personal = 1.0 - conformity_sensitivity
        w_social = conformity_sensitivity

        base_polarity = w_personal * attitude_polarity + w_social * sn_directional

        # Apply negativity bias: amplify negative sentiment
        if base_polarity < 0:
            # Amplify negative sentiment based on negativity_bias trait
            # negativity_bias = 0.5 → no change
            # negativity_bias = 1.0 → amplify by 1.5x
            amplification = 1.0 + negativity_bias
            polarity = base_polarity * amplification
        else:
            # Positive sentiment unchanged
            polarity = base_polarity

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, polarity))

    def decide_interpret(
        self,
        agent_state: AgentState,
        graph: nx.Graph,
        agent_id: int
    ) -> bool:
        """
        Decide whether agent should interpret the crisis and form sentiment.

        Args:
            agent_state: Current agent state
            graph: Network graph
            agent_id: Agent ID

        Returns:
            True if agent decides to interpret
        """
        # Only decide if aware but not yet interpreted
        if not agent_state.aware or agent_state.interpreted:
            return False

        # Compute interpretation score
        interpretation_score = self.compute_interpretation_score(
            agent_state, graph, agent_id
        )

        # Stochastic decision based on interpretation score
        return random.random() < interpretation_score

    def decide_express(
        self,
        agent_state: AgentState,
        graph: nx.Graph,
        agent_id: int
    ) -> bool:
        """
        Decide whether agent should publicly express their sentiment.

        Expression is modulated by:
        - Expression threshold (barrier to expression)
        - Social risk aversion (fear of social consequences)
        - Sentiment intensity (stronger sentiment → more likely to express)

        Args:
            agent_state: Current agent state
            graph: Network graph
            agent_id: Agent ID

        Returns:
            True if agent decides to express
        """
        # Only decide if interpreted but not yet expressed
        if not agent_state.interpreted or agent_state.expressed:
            return False

        if agent_state.traits is None:
            raise ValueError(f"Agent {agent_id} has no traits defined")

        traits = agent_state.traits
        polarity = agent_state.polarity
        if polarity is None:
            raise ValueError(f"Agent {agent_id} has no polarity")

        # Get traits
        expression_threshold = traits.get("expression_threshold", 0.5)
        social_risk_aversion = traits.get("social_risk_aversion", 0.5)

        # Sentiment intensity (how strong is the sentiment)
        sentiment_intensity = abs(polarity)

        # Expression likelihood decreases with:
        # - Higher expression threshold
        # - Higher social risk aversion
        # Expression likelihood increases with:
        # - Higher sentiment intensity

        if self.expression_mode == "threshold":
            # Simple threshold: express if intensity exceeds threshold
            adjusted_threshold = expression_threshold * (1.0 + social_risk_aversion)
            return sentiment_intensity >= adjusted_threshold

        elif self.expression_mode == "probabilistic":
            # Probabilistic: compute expression probability
            # Higher intensity → higher probability
            # Higher barriers → lower probability
            expression_prob = sentiment_intensity / (1.0 + expression_threshold + social_risk_aversion)
            return random.random() < expression_prob

        else:
            raise ValueError(f"Unknown expression_mode: {self.expression_mode}")

    def decide_share(
        self,
        agent_state: AgentState,
        graph: nx.Graph,
        agent_id: int,
        neighbor_id: int,
    ) -> bool:
        """
        Decide whether to share sentiment with a specific neighbor.

        Args:
            agent_state: Current agent state
            graph: Network graph
            agent_id: Agent ID
            neighbor_id: Neighbor ID

        Returns:
            True if agent decides to share
        """
        # Only share if expressed
        if not agent_state.expressed:
            return False

        # Check if already shared with this neighbor
        if neighbor_id in agent_state.shared_with:
            return False

        # Get share propensity from traits
        share_propensity = agent_state.traits.get("share_propensity", 0.8) \
            if agent_state.traits else 0.8

        # Get sentiment intensity
        polarity = agent_state.polarity
        if polarity is None:
            return False

        sentiment_intensity = abs(polarity)

        # Share if sentiment is strong enough and propensity check
        if sentiment_intensity >= self.share_threshold:
            return random.random() < share_propensity

        return False


class SimpleSentimentModel(SentimentDecisionModel):
    """
    Simplified sentiment model with standard parameters.

    Useful for initial testing and comparison.
    """

    def __init__(
        self,
        w_attitude: float = 0.4,
        w_social_norm: float = 0.35,
        w_moral: float = 0.25,
        alpha: float = 0.3,
        beta: float = 0.7,
    ):
        super().__init__(
            w_attitude=w_attitude,
            w_social_norm=w_social_norm,
            w_moral=w_moral,
            alpha=alpha,
            beta=beta,
            expression_mode="probabilistic",
            share_threshold=0.5,
        )

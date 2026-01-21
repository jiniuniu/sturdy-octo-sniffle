"""
Theory of Planned Behavior (TPB) decision model.

Implements the complete TPB framework:
1. Attitude (A): Personal evaluation of the innovation
2. Social Norm (SN): Perceived social pressure from network
3. Perceived Behavioral Control (PBC): Perceived ease/difficulty
4. Intention (I): Weighted combination of A, SN, PBC
5. Behavior: Stochastic decision based on intention (with friction)

Reference: Ajzen, I. (1991). The theory of planned behavior.
"""

import random
from typing import Optional

import networkx as nx

from ..core.protocols import DecisionModel
from ..core.schemas import AgentState
from .components import FrictionModel, IntentionCalculator, SocialNormAggregator


class TPBDecisionModel:
    """
    Complete TPB-based decision model for innovation diffusion.

    This model implements the DecisionModel protocol and uses TPB
    to compute adoption and sharing decisions.
    """

    def __init__(
        self,
        w_attitude: float = 0.4,
        w_social_norm: float = 0.35,
        w_pbc: float = 0.25,
        sn_strategy: str = "mean",
        intention_mode: str = "sigmoid",
        friction: Optional[FrictionModel] = None,
        share_threshold: float = 0.7,
    ):
        """
        Initialize TPB decision model.

        Args:
            w_attitude: Weight for attitude component
            w_social_norm: Weight for social norm component
            w_pbc: Weight for PBC component
            sn_strategy: Social norm aggregation strategy
            intention_mode: Intention computation mode
            friction: Friction model (default: no friction)
            share_threshold: Minimum intention to share (0-1)
        """
        self.sn_aggregator = SocialNormAggregator(strategy=sn_strategy)
        self.intention_calc = IntentionCalculator(
            w_attitude=w_attitude,
            w_social_norm=w_social_norm,
            w_pbc=w_pbc,
            mode=intention_mode,
        )
        self.friction = (
            friction if friction is not None else FrictionModel(base_friction=0.0)
        )
        self.share_threshold = share_threshold

    def compute_intention(
        self, agent_state: AgentState, graph: nx.Graph, agent_id: int
    ) -> float:
        """
        Compute behavioral intention for an agent.

        Args:
            agent_state: Current agent state
            graph: Network graph
            agent_id: Agent ID

        Returns:
            Intention score (0-1)
        """
        if agent_state.traits is None:
            raise ValueError(f"Agent {agent_id} has no traits defined")

        traits = agent_state.traits

        # Get TPB components from traits
        attitude = traits.get("attitude", 0.5)
        pbc = traits.get("pbc", 0.5)

        # Compute social norm from network
        social_norm = self.sn_aggregator.compute(graph, agent_id)

        # Get conformity sensitivity
        conformity = traits.get("conformity", 1.0)

        # Compute intention
        intention = self.intention_calc.compute_with_traits(
            attitude=attitude,
            social_norm=social_norm,
            pbc=pbc,
            conformity_sensitivity=conformity,
        )

        return intention

    def decide_adopt(
        self, agent_state: AgentState, graph: nx.Graph, agent_id: int
    ) -> bool:
        """
        Decide whether agent should adopt the innovation.

        Args:
            agent_state: Current agent state
            graph: Network graph
            agent_id: Agent ID

        Returns:
            True if agent decides to adopt
        """
        # Only decide if aware but not yet adopted
        if not agent_state.aware or agent_state.adopted:
            return False

        # Compute intention
        intention = self.compute_intention(agent_state, graph, agent_id)

        # Apply friction based on risk aversion
        risk_aversion = (
            agent_state.traits.get("risk_aversion", 0.0) if agent_state.traits else 0.0
        )

        # Make stochastic decision
        return self.friction.decide_adopt(intention, risk_aversion)

    def decide_share(
        self,
        agent_state: AgentState,
        graph: nx.Graph,
        agent_id: int,
        neighbor_id: int,
    ) -> bool:
        """
        Decide whether to share innovation with a specific neighbor.

        Args:
            agent_state: Current agent state
            graph: Network graph
            agent_id: Agent ID
            neighbor_id: Neighbor ID

        Returns:
            True if agent decides to share
        """
        # Only share if adopted
        if not agent_state.adopted:
            return False

        # Check if already shared with this neighbor
        if neighbor_id in agent_state.shared_with:
            return False

        # Get share propensity from traits
        share_propensity = (
            agent_state.traits.get("share_propensity", 0.8)
            if agent_state.traits
            else 0.8
        )

        # Compute own intention (as proxy for enthusiasm)
        intention = self.compute_intention(agent_state, graph, agent_id)

        # Share if intention above threshold and propensity check
        if intention >= self.share_threshold:
            return random.random() < share_propensity

        return False

    def step(self):
        """Advance model by one time step (for time-dependent components)."""
        self.friction.step()

    def reset(self):
        """Reset model state."""
        self.friction.reset()


class SimpleTPBModel(TPBDecisionModel):
    """
    Simplified TPB model with no friction and high share rate.

    Useful for debugging and comparison with baseline.
    """

    def __init__(
        self,
        w_attitude: float = 0.4,
        w_social_norm: float = 0.35,
        w_pbc: float = 0.25,
    ):
        super().__init__(
            w_attitude=w_attitude,
            w_social_norm=w_social_norm,
            w_pbc=w_pbc,
            sn_strategy="mean",
            intention_mode="sigmoid",
            friction=FrictionModel(base_friction=0.0),
            share_threshold=0.5,
        )

"""
Sentiment-based simulation engine for brand crisis diffusion.

Implements sentiment diffusion with state transitions:
1. aware → interpreted (with polarity determination)
2. interpreted → expressed
3. expressed → share with neighbors
"""
import random
from typing import List, Optional, Dict
import networkx as nx
from ..core.schemas import (
    SimulationConfig,
    NetworkConfig,
    StepSnapshot,
    SimulationResult,
)
from ..agents.base import BaseAgent
from ..agents.sentiment_agent import SentimentAgent
from ..decisions.sentiment import SentimentDecisionModel
from ..core.exceptions import SimulationError


class SentimentSnapshot(StepSnapshot):
    """
    Extended snapshot for sentiment diffusion tracking.

    Adds sentiment-specific metrics beyond basic awareness/adoption.
    """
    interpreted_count: int = 0
    expressed_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    mean_polarity: float = 0.0


class SentimentDiffusionEngine:
    """
    Simulation engine for sentiment diffusion in brand crisis scenarios.

    State machine:
    - aware: Agent becomes aware of the crisis event
    - interpreted: Agent forms internal sentiment (polarity)
    - expressed: Agent publicly expresses their sentiment
    - shared: Agent shares sentiment with neighbors

    Unlike innovation diffusion (aware → adopted), sentiment diffusion
    separates interpretation (internal) from expression (public).
    """

    def __init__(
        self,
        graph: nx.Graph,
        config: SimulationConfig,
        network_config: NetworkConfig,
        decision_model: SentimentDecisionModel,
    ):
        """
        Initialize sentiment diffusion engine.

        Args:
            graph: Network graph with sentiment agents
            config: Simulation configuration
            network_config: Network configuration
            decision_model: Sentiment decision model
        """
        self.graph = graph
        self.config = config
        self.network_config = network_config
        self.decision_model = decision_model
        self.snapshots: List[SentimentSnapshot] = []

        # Set random seed
        if config.seed is not None:
            random.seed(config.seed)

        # Validate that all agents have traits
        self._validate_agents()

    def _validate_agents(self):
        """Validate that all agents have sentiment traits."""
        for node_id in self.graph.nodes():
            state = BaseAgent.get_state(self.graph, node_id)
            if state.traits is None:
                raise SimulationError(
                    f"Agent {node_id} has no traits. "
                    "Use SentimentPopulationGenerator to initialize agents."
                )

    def initialize(self, initial_aware_ids: Optional[List[int]] = None) -> None:
        """
        Initialize simulation with seed aware agents (crisis exposure).

        Args:
            initial_aware_ids: Specific node IDs to initialize as aware
                If None, will randomly select based on config.initial_adopters
        """
        if initial_aware_ids is None:
            # Random selection
            all_nodes = list(self.graph.nodes())
            if self.config.initial_adopters > len(all_nodes):
                raise SimulationError(
                    f"initial_adopters ({self.config.initial_adopters}) "
                    f"exceeds number of nodes ({len(all_nodes)})"
                )
            initial_aware_ids = random.sample(all_nodes, self.config.initial_adopters)

        # Initialize seed aware agents
        for node_id in initial_aware_ids:
            BaseAgent.make_aware(self.graph, node_id)

    def step(self) -> SentimentSnapshot:
        """
        Execute one simulation step for sentiment diffusion.

        Process:
        1. Aware agents decide to interpret (form sentiment with polarity)
        2. Interpreted agents decide to express (public expression)
        3. Expressed agents decide to share with neighbors
        4. Shared sentiment makes neighbors aware
        5. Record statistics

        Returns:
            Snapshot of state after this step
        """
        newly_aware = []
        newly_interpreted = []
        newly_expressed = []
        share_count = 0

        # Phase 1: Interpretation (aware → interpreted)
        aware_not_interpreted = [
            node_id
            for node_id in self.graph.nodes()
            if (
                BaseAgent.get_state(self.graph, node_id).aware
                and not BaseAgent.get_state(self.graph, node_id).interpreted
            )
        ]

        for agent_id in aware_not_interpreted:
            agent_state = BaseAgent.get_state(self.graph, agent_id)

            # Decide to interpret using decision model
            if self.decision_model.decide_interpret(agent_state, self.graph, agent_id):
                # Determine polarity
                polarity = self.decision_model.decide_polarity(
                    agent_state, self.graph, agent_id
                )

                # Mark as interpreted with polarity
                SentimentAgent.make_interpreted(self.graph, agent_id, polarity)
                newly_interpreted.append(agent_id)

        # Phase 2: Expression (interpreted → expressed)
        interpreted_not_expressed = [
            node_id
            for node_id in self.graph.nodes()
            if (
                BaseAgent.get_state(self.graph, node_id).interpreted
                and not BaseAgent.get_state(self.graph, node_id).expressed
            )
        ]

        for agent_id in interpreted_not_expressed:
            agent_state = BaseAgent.get_state(self.graph, agent_id)

            # Decide to express using decision model
            if self.decision_model.decide_express(agent_state, self.graph, agent_id):
                SentimentAgent.make_expressed(self.graph, agent_id)
                newly_expressed.append(agent_id)

        # Phase 3: Sharing (expressed → share with neighbors)
        expressed_agents = [
            node_id
            for node_id in self.graph.nodes()
            if BaseAgent.get_state(self.graph, node_id).expressed
        ]

        for agent_id in expressed_agents:
            agent_state = BaseAgent.get_state(self.graph, agent_id)
            neighbors = BaseAgent.get_neighbors(self.graph, agent_id)

            for neighbor_id in neighbors:
                # Skip if already shared
                if BaseAgent.has_shared_with(self.graph, agent_id, neighbor_id):
                    continue

                neighbor_state = BaseAgent.get_state(self.graph, neighbor_id)

                # Decide to share using decision model
                if self.decision_model.decide_share(
                    agent_state, self.graph, agent_id, neighbor_id
                ):
                    # Mark as shared
                    BaseAgent.mark_shared_with(self.graph, agent_id, neighbor_id)
                    share_count += 1

                    # Make neighbor aware if not already
                    if not neighbor_state.aware:
                        BaseAgent.make_aware(self.graph, neighbor_id)
                        newly_aware.append(neighbor_id)

        # Collect statistics
        snapshot = self._create_snapshot(len(self.snapshots), share_count)
        return snapshot

    def run(self) -> SimulationResult:
        """
        Run complete sentiment diffusion simulation.

        Returns:
            Complete simulation results with all snapshots
        """
        # Record initial state (step 0)
        initial_snapshot = self._create_snapshot(0, 0)
        self.snapshots.append(initial_snapshot)

        # Run simulation steps
        for step_num in range(1, self.config.max_steps + 1):
            snapshot = self.step()
            self.snapshots.append(snapshot)

            # Stop early if no more changes (everyone aware and interpreted)
            # Note: Unlike innovation diffusion, we don't expect 100% expression
            if snapshot.aware_count == self.graph.number_of_nodes() and \
               snapshot.interpreted_count == snapshot.aware_count:
                # Check if any agents can still express
                can_express = any(
                    BaseAgent.get_state(self.graph, node_id).interpreted and
                    not BaseAgent.get_state(self.graph, node_id).expressed
                    for node_id in self.graph.nodes()
                )
                if not can_express:
                    break

        # Create result
        total_nodes = self.graph.number_of_nodes()
        final_snapshot = self.snapshots[-1]

        # For sentiment diffusion, we use "expressed" as the analog to "adopted"
        result = SimulationResult(
            config=self.config,
            network_config=self.network_config,
            total_steps=len(self.snapshots) - 1,
            snapshots=self.snapshots,
            final_adoption_rate=final_snapshot.expressed_count / total_nodes,
            final_awareness_rate=final_snapshot.aware_count / total_nodes,
        )

        return result

    def _create_snapshot(self, step: int, share_count: int) -> SentimentSnapshot:
        """
        Create a snapshot of current sentiment diffusion state.

        Args:
            step: Current step number
            share_count: Number of shares in this step

        Returns:
            SentimentSnapshot object
        """
        aware_agents = []
        interpreted_agents = []
        expressed_agents = []
        polarities = []
        positive_count = 0
        negative_count = 0

        for node_id in self.graph.nodes():
            state = BaseAgent.get_state(self.graph, node_id)

            if state.aware:
                aware_agents.append(node_id)

            if state.interpreted:
                interpreted_agents.append(node_id)
                if state.polarity is not None:
                    polarities.append(state.polarity)
                    if state.polarity > 0:
                        positive_count += 1
                    elif state.polarity < 0:
                        negative_count += 1

            if state.expressed:
                expressed_agents.append(node_id)

        # Compute mean polarity
        mean_polarity = sum(polarities) / len(polarities) if len(polarities) > 0 else 0.0

        # Note: StepSnapshot expects 'adopted_agents' field
        # For sentiment, we use expressed agents as the analog
        snapshot = SentimentSnapshot(
            step=step,
            aware_count=len(aware_agents),
            adopted_count=len(expressed_agents),  # Use expressed as "adopted"
            shared_count=share_count,
            aware_agents=aware_agents,
            adopted_agents=expressed_agents,  # Use expressed as "adopted"
            interpreted_count=len(interpreted_agents),
            expressed_count=len(expressed_agents),
            positive_count=positive_count,
            negative_count=negative_count,
            mean_polarity=mean_polarity,
        )

        return snapshot

    def get_polarity_distribution(self) -> Dict[str, int]:
        """
        Get distribution of sentiment polarity in current state.

        Returns:
            Dictionary with counts for positive, negative, neutral, unexpressed
        """
        positive = 0
        negative = 0
        neutral = 0
        unexpressed = 0

        for node_id in self.graph.nodes():
            state = BaseAgent.get_state(self.graph, node_id)
            if state.expressed and state.polarity is not None:
                if state.polarity > 0.1:
                    positive += 1
                elif state.polarity < -0.1:
                    negative += 1
                else:
                    neutral += 1
            else:
                unexpressed += 1

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "unexpressed": unexpressed,
        }

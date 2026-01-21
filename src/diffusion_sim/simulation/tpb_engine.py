"""
TPB-based simulation engine.

Extends the base simulation engine with TPB decision-making.
"""

import random
from typing import List, Optional

import networkx as nx

from ..agents.base import BaseAgent
from ..agents.tpb_agent import TPBAgent
from ..core.exceptions import SimulationError
from ..core.schemas import (
    NetworkConfig,
    SimulationConfig,
    SimulationResult,
    StepSnapshot,
)
from ..decisions.tpb import TPBDecisionModel


class TPBDiffusionEngine:
    """
    Simulation engine with TPB-based decision making.

    Unlike SimpleDiffusionEngine which uses fixed probabilities,
    this engine uses TPB model to compute agent-specific adoption
    and sharing decisions based on their traits and network context.
    """

    def __init__(
        self,
        graph: nx.Graph,
        config: SimulationConfig,
        network_config: NetworkConfig,
        decision_model: TPBDecisionModel,
    ):
        """
        Initialize TPB simulation engine.

        Args:
            graph: Network graph with TPB agents
            config: Simulation configuration
            network_config: Network configuration
            decision_model: TPB decision model
        """
        self.graph = graph
        self.config = config
        self.network_config = network_config
        self.decision_model = decision_model
        self.snapshots: List[StepSnapshot] = []

        # Set random seed
        if config.seed is not None:
            random.seed(config.seed)

        # Validate that all agents have traits
        self._validate_agents()

    def _validate_agents(self):
        """Validate that all agents have TPB traits."""
        for node_id in self.graph.nodes():
            state = BaseAgent.get_state(self.graph, node_id)
            if state.traits is None:
                raise SimulationError(
                    f"Agent {node_id} has no traits. "
                    "Use PopulationGenerator to initialize agents."
                )

    def initialize(self, initial_adopter_ids: Optional[List[int]] = None) -> None:
        """
        Initialize simulation with seed adopters.

        Args:
            initial_adopter_ids: Specific node IDs to initialize as adopters
                If None, will randomly select based on config.initial_adopters
        """
        if initial_adopter_ids is None:
            # Random selection
            all_nodes = list(self.graph.nodes())
            if self.config.initial_adopters > len(all_nodes):
                raise SimulationError(
                    f"initial_adopters ({self.config.initial_adopters}) "
                    f"exceeds number of nodes ({len(all_nodes)})"
                )
            initial_adopter_ids = random.sample(all_nodes, self.config.initial_adopters)

        # Initialize seed adopters
        for node_id in initial_adopter_ids:
            BaseAgent.make_aware(self.graph, node_id)
            BaseAgent.make_adopted(self.graph, node_id)

    def step(self) -> StepSnapshot:
        """
        Execute one simulation step with TPB decision making.

        Process:
        1. Identify all adopted agents
        2. Each adopted agent decides whether to share with each neighbor (TPB)
        3. Newly aware agents decide whether to adopt (TPB)
        4. Record statistics

        Returns:
            Snapshot of state after this step
        """
        newly_aware = []
        share_count = 0

        # Get all currently adopted agents
        adopted_agents = [
            node_id
            for node_id in self.graph.nodes()
            if BaseAgent.get_state(self.graph, node_id).adopted
        ]

        # Phase 1: Sharing
        for agent_id in adopted_agents:
            agent_state = BaseAgent.get_state(self.graph, agent_id)
            neighbors = BaseAgent.get_neighbors(self.graph, agent_id)

            for neighbor_id in neighbors:
                # Skip if already shared
                if BaseAgent.has_shared_with(self.graph, agent_id, neighbor_id):
                    continue

                neighbor_state = BaseAgent.get_state(self.graph, neighbor_id)

                # Decide to share using TPB model
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

        # Phase 2: Adoption
        newly_adopted = []
        for agent_id in newly_aware:
            agent_state = BaseAgent.get_state(self.graph, agent_id)

            # Decide to adopt using TPB model
            if self.decision_model.decide_adopt(agent_state, self.graph, agent_id):
                BaseAgent.make_adopted(self.graph, agent_id)
                newly_adopted.append(agent_id)

        # Also check previously aware agents who haven't adopted yet
        aware_not_adopted = [
            node_id
            for node_id in self.graph.nodes()
            if (
                BaseAgent.get_state(self.graph, node_id).aware
                and not BaseAgent.get_state(self.graph, node_id).adopted
                and node_id not in newly_aware
            )
        ]

        for agent_id in aware_not_adopted:
            agent_state = BaseAgent.get_state(self.graph, agent_id)
            if self.decision_model.decide_adopt(agent_state, self.graph, agent_id):
                BaseAgent.make_adopted(self.graph, agent_id)
                newly_adopted.append(agent_id)

        # Advance decision model time step
        self.decision_model.step()

        # Collect statistics
        snapshot = self._create_snapshot(len(self.snapshots), share_count)
        return snapshot

    def run(self) -> SimulationResult:
        """
        Run complete simulation.

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

            # Stop early if everyone has adopted
            if snapshot.adopted_count == self.graph.number_of_nodes():
                break

        # Create result
        total_nodes = self.graph.number_of_nodes()
        final_snapshot = self.snapshots[-1]

        result = SimulationResult(
            config=self.config,
            network_config=self.network_config,
            total_steps=len(self.snapshots) - 1,
            snapshots=self.snapshots,
            final_adoption_rate=final_snapshot.adopted_count / total_nodes,
            final_awareness_rate=final_snapshot.aware_count / total_nodes,
        )

        return result

    def _create_snapshot(self, step: int, share_count: int) -> StepSnapshot:
        """
        Create a snapshot of current simulation state.

        Args:
            step: Current step number
            share_count: Number of shares in this step

        Returns:
            Snapshot object
        """
        aware_agents = []
        adopted_agents = []

        for node_id in self.graph.nodes():
            state = BaseAgent.get_state(self.graph, node_id)
            if state.aware:
                aware_agents.append(node_id)
            if state.adopted:
                adopted_agents.append(node_id)

        return StepSnapshot(
            step=step,
            aware_count=len(aware_agents),
            adopted_count=len(adopted_agents),
            shared_count=share_count,
            aware_agents=aware_agents,
            adopted_agents=adopted_agents,
        )

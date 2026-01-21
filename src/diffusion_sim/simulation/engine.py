"""
Core simulation engine for diffusion process.
"""

import random
from typing import List

import networkx as nx

from ..agents.base import BaseAgent
from ..core.exceptions import SimulationError
from ..core.schemas import (
    NetworkConfig,
    SimulationConfig,
    SimulationResult,
    StepSnapshot,
)


class SimpleDiffusionEngine:
    """
    Simple diffusion engine for baseline experiments.

    In this mode (Step 1):
    - share_probability = 1.0: aware agents always share
    - adopt_probability = 1.0: aware agents always adopt
    - Used to verify network structure and basic diffusion mechanics
    """

    def __init__(
        self,
        graph: nx.Graph,
        config: SimulationConfig,
        network_config: NetworkConfig,
    ):
        """
        Initialize simulation engine.

        Args:
            graph: Network graph with initialized agents
            config: Simulation configuration
            network_config: Network configuration (for result tracking)
        """
        self.graph = graph
        self.config = config
        self.network_config = network_config
        self.snapshots: List[StepSnapshot] = []

        # Set random seed
        if config.seed is not None:
            random.seed(config.seed)

    def initialize(self) -> None:
        """
        Initialize simulation with seed adopters.

        Randomly select initial adopters and mark them as aware and adopted.
        """
        all_nodes = list(self.graph.nodes())
        if self.config.initial_adopters > len(all_nodes):
            raise SimulationError(
                f"initial_adopters ({self.config.initial_adopters}) "
                f"exceeds number of nodes ({len(all_nodes)})"
            )

        # Randomly select initial adopters
        initial_nodes = random.sample(all_nodes, self.config.initial_adopters)

        # Initialize them as aware and adopted
        for node_id in initial_nodes:
            BaseAgent.make_aware(self.graph, node_id)
            BaseAgent.make_adopted(self.graph, node_id)

    def step(self) -> StepSnapshot:
        """
        Execute one simulation step.

        Process:
        1. Identify all aware & adopted agents
        2. For each, share with all neighbors they haven't shared with
        3. Mark newly aware agents
        4. Newly aware agents immediately adopt (for baseline)
        5. Record statistics

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

        # Each adopted agent shares with neighbors
        for agent_id in adopted_agents:
            neighbors = BaseAgent.get_neighbors(self.graph, agent_id)

            for neighbor_id in neighbors:
                # Skip if already shared with this neighbor
                if BaseAgent.has_shared_with(self.graph, agent_id, neighbor_id):
                    continue

                neighbor_state = BaseAgent.get_state(self.graph, neighbor_id)

                # Share with probability
                if random.random() <= self.config.share_probability:
                    # Mark as shared
                    BaseAgent.mark_shared_with(self.graph, agent_id, neighbor_id)
                    share_count += 1

                    # Make neighbor aware if not already
                    if not neighbor_state.aware:
                        BaseAgent.make_aware(self.graph, neighbor_id)
                        newly_aware.append(neighbor_id)

        # Newly aware agents decide to adopt
        for agent_id in newly_aware:
            if random.random() <= self.config.adopt_probability:
                BaseAgent.make_adopted(self.graph, agent_id)

        # Collect statistics
        snapshot = self._create_snapshot(len(self.snapshots), share_count)
        return snapshot

    def run(self) -> SimulationResult:
        """
        Run complete simulation.

        Returns:
            Complete simulation results with all snapshots
        """
        # Initialize
        self.initialize()

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
            total_steps=len(self.snapshots) - 1,  # Excluding initial step
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

"""
Enhanced TPB simulation engine with detailed tracking for diffusion analysis.

This module extends the basic TPB engine to track awareness propagation,
adoption timing, and influence paths for validating Rogers' diffusion theory.
"""
import random
from typing import List, Dict, Optional, Tuple
import networkx as nx
from ..core.schemas import (
    SimulationConfig,
    NetworkConfig,
    StepSnapshot,
    SimulationResult,
)
from ..agents.base import BaseAgent
from ..agents.tpb_agent import TPBAgent
from ..decisions.tpb import TPBDecisionModel
from ..core.exceptions import SimulationError


class TrackedTPBDiffusionEngine:
    """
    TPB simulation engine with detailed tracking of diffusion process.

    Tracks for each agent:
    - aware_at: timestep when agent became aware
    - adopted_at: timestep when agent adopted
    - informed_by: which agent informed this agent
    - influence_path: chain of influence from seed to this agent
    """

    def __init__(
        self,
        graph: nx.Graph,
        config: SimulationConfig,
        network_config: NetworkConfig,
        decision_model: TPBDecisionModel,
    ):
        self.graph = graph
        self.config = config
        self.network_config = network_config
        self.decision_model = decision_model
        self.snapshots: List[StepSnapshot] = []

        # Tracking dictionary: agent_id -> tracking info
        self.tracking_info: Dict[int, Dict] = {}

        if config.seed is not None:
            random.seed(config.seed)

        self._validate_agents()
        self._initialize_tracking()

    def _validate_agents(self):
        """Validate that all agents have TPB traits."""
        for node_id in self.graph.nodes():
            state = BaseAgent.get_state(self.graph, node_id)
            if state.traits is None:
                raise SimulationError(
                    f"Agent {node_id} has no traits. "
                    "Use PopulationGenerator to initialize agents."
                )

    def _initialize_tracking(self):
        """Initialize tracking info for all agents."""
        for node_id in self.graph.nodes():
            traits = TPBAgent.get_all_traits(self.graph, node_id)
            # Use category assigned during sampling (not computed from traits)
            category = TPBAgent.get_category(self.graph, node_id)

            # Fallback to trait-based categorization for legacy agents
            if category is None:
                category = TPBAgent.categorize_agent(traits)

            self.tracking_info[node_id] = {
                'category': category,
                'traits': traits.copy(),
                'aware_at': None,
                'adopted_at': None,
                'informed_by': None,
                'influence_depth': None,  # Distance from seed adopters
            }

    def initialize(self, initial_adopter_ids: Optional[List[int]] = None) -> None:
        """
        Initialize simulation with seed adopters and track them.

        Args:
            initial_adopter_ids: Specific node IDs to initialize as adopters
        """
        if initial_adopter_ids is None:
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

            # Track seed adopters
            self.tracking_info[node_id]['aware_at'] = 0
            self.tracking_info[node_id]['adopted_at'] = 0
            self.tracking_info[node_id]['informed_by'] = 'seed'
            self.tracking_info[node_id]['influence_depth'] = 0

    def step(self) -> Tuple[StepSnapshot, Dict]:
        """
        Execute one simulation step with detailed tracking.

        Returns:
            Tuple of (snapshot, step_events)
            step_events contains lists of newly aware/adopted agents
        """
        current_step = len(self.snapshots)
        newly_aware = []
        newly_adopted = []
        share_count = 0

        # Track who informed whom in this step
        influence_events = []

        # Get all currently adopted agents
        adopted_agents = [
            node_id
            for node_id in self.graph.nodes()
            if BaseAgent.get_state(self.graph, node_id).adopted
        ]

        # Phase 1: Sharing and awareness propagation
        for agent_id in adopted_agents:
            agent_state = BaseAgent.get_state(self.graph, agent_id)
            neighbors = BaseAgent.get_neighbors(self.graph, agent_id)

            for neighbor_id in neighbors:
                if BaseAgent.has_shared_with(self.graph, agent_id, neighbor_id):
                    continue

                neighbor_state = BaseAgent.get_state(self.graph, neighbor_id)

                # Decide to share using TPB model
                if self.decision_model.decide_share(
                    agent_state, self.graph, agent_id, neighbor_id
                ):
                    BaseAgent.mark_shared_with(self.graph, agent_id, neighbor_id)
                    share_count += 1

                    # Make neighbor aware if not already
                    if not neighbor_state.aware:
                        BaseAgent.make_aware(self.graph, neighbor_id)
                        newly_aware.append(neighbor_id)

                        # Track awareness propagation
                        self.tracking_info[neighbor_id]['aware_at'] = current_step
                        self.tracking_info[neighbor_id]['informed_by'] = agent_id

                        # Calculate influence depth
                        source_depth = self.tracking_info[agent_id]['influence_depth']
                        if source_depth is not None:
                            self.tracking_info[neighbor_id]['influence_depth'] = source_depth + 1

                        # Record influence event
                        influence_events.append({
                            'step': current_step,
                            'from': agent_id,
                            'to': neighbor_id,
                            'event_type': 'awareness'
                        })

        # Phase 2: Adoption decisions
        # Check newly aware agents
        for agent_id in newly_aware:
            agent_state = BaseAgent.get_state(self.graph, agent_id)

            if self.decision_model.decide_adopt(agent_state, self.graph, agent_id):
                BaseAgent.make_adopted(self.graph, agent_id)
                newly_adopted.append(agent_id)
                self.tracking_info[agent_id]['adopted_at'] = current_step

        # Check previously aware but not adopted agents
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
                self.tracking_info[agent_id]['adopted_at'] = current_step

        # Advance decision model time step
        self.decision_model.step()

        # Create snapshot
        snapshot = self._create_snapshot(current_step, share_count)

        # Create step events summary
        step_events = {
            'newly_aware': newly_aware,
            'newly_adopted': newly_adopted,
            'influence_events': influence_events,
        }

        return snapshot, step_events

    def run(self) -> Tuple[SimulationResult, Dict]:
        """
        Run complete simulation with tracking.

        Returns:
            Tuple of (SimulationResult, tracking_info)
        """
        # Record initial state (step 0)
        initial_snapshot = self._create_snapshot(0, 0)
        self.snapshots.append(initial_snapshot)

        all_influence_events = []

        # Run simulation steps
        for step_num in range(1, self.config.max_steps + 1):
            snapshot, step_events = self.step()
            self.snapshots.append(snapshot)
            all_influence_events.extend(step_events['influence_events'])

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

        # Add influence events to tracking info
        tracking_data = {
            'agent_info': self.tracking_info,
            'influence_events': all_influence_events,
        }

        return result, tracking_data

    def _create_snapshot(self, step: int, share_count: int) -> StepSnapshot:
        """Create a snapshot of current simulation state."""
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

    def get_influence_chain(self, agent_id: int) -> List[int]:
        """
        Get the chain of influence from seed to a specific agent.

        Args:
            agent_id: Target agent ID

        Returns:
            List of agent IDs representing the influence path
        """
        chain = []
        current = agent_id

        while current is not None:
            chain.append(current)
            informed_by = self.tracking_info[current]['informed_by']

            if informed_by == 'seed' or informed_by is None:
                break

            current = informed_by

        return list(reversed(chain))  # Return from seed to target

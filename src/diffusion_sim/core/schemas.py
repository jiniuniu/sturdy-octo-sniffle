"""
Core data schemas using Pydantic for type safety and validation.
"""
from typing import Set, Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class AgentState(BaseModel):
    """State of an agent in the diffusion process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: int
    aware: bool = False  # Has the agent become aware of the innovation?
    adopted: bool = False  # Has the agent adopted the innovation?
    shared_with: Set[int] = Field(default_factory=set)  # IDs of neighbors shared with

    # For TPB implementation
    traits: Optional[Dict[str, float]] = None
    category: Optional[str] = None  # Rogers' category assigned during sampling


class NetworkConfig(BaseModel):
    """Configuration for network generation."""

    network_type: str = "small_world"  # Type of network to generate
    n: int = Field(default=100, ge=2)  # Number of nodes
    k: int = Field(default=4, ge=1)  # Average degree (for small world)
    p: float = Field(default=0.1, ge=0, le=1)  # Rewiring probability (for small world)
    seed: Optional[int] = None  # Random seed for reproducibility


class SimulationConfig(BaseModel):
    """Configuration for simulation execution."""

    max_steps: int = Field(default=50, ge=1)  # Maximum simulation steps
    seed: Optional[int] = None  # Random seed
    initial_adopters: int = Field(default=1, ge=1)  # Number of initial adopters
    share_probability: float = Field(default=1.0, ge=0, le=1)  # Probability of sharing
    adopt_probability: float = Field(default=1.0, ge=0, le=1)  # Probability of adoption when aware


class StepSnapshot(BaseModel):
    """Snapshot of simulation state at a specific step."""

    step: int
    aware_count: int
    adopted_count: int
    shared_count: int  # Number of shares that occurred this step
    aware_agents: List[int] = Field(default_factory=list)
    adopted_agents: List[int] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Complete results from a simulation run."""

    config: SimulationConfig
    network_config: NetworkConfig
    total_steps: int
    snapshots: List[StepSnapshot]
    final_adoption_rate: float
    final_awareness_rate: float

    def get_timeseries(self, metric: str) -> List[int]:
        """Extract a specific metric as a time series."""
        if metric == "aware":
            return [s.aware_count for s in self.snapshots]
        elif metric == "adopted":
            return [s.adopted_count for s in self.snapshots]
        elif metric == "shared":
            return [s.shared_count for s in self.snapshots]
        else:
            raise ValueError(f"Unknown metric: {metric}")

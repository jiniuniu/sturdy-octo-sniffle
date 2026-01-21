"""
Core abstractions and data models for the diffusion simulation framework.
"""

from .exceptions import (
    AgentError,
    ConfigurationError,
    DiffusionSimError,
    NetworkBuildError,
    SimulationError,
)
from .protocols import DecisionModel, NetworkBuilder
from .schemas import (
    AgentState,
    NetworkConfig,
    SimulationConfig,
    SimulationResult,
    StepSnapshot,
)

__all__ = [
    # Schemas
    "AgentState",
    "NetworkConfig",
    "SimulationConfig",
    "StepSnapshot",
    "SimulationResult",
    # Protocols
    "NetworkBuilder",
    "DecisionModel",
    # Exceptions
    "DiffusionSimError",
    "NetworkBuildError",
    "SimulationError",
    "ConfigurationError",
    "AgentError",
]

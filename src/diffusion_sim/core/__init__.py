"""
Core abstractions and data models for the diffusion simulation framework.
"""
from .schemas import (
    AgentState,
    NetworkConfig,
    SimulationConfig,
    StepSnapshot,
    SimulationResult,
)
from .protocols import NetworkBuilder, DecisionModel
from .exceptions import (
    DiffusionSimError,
    NetworkBuildError,
    SimulationError,
    ConfigurationError,
    AgentError,
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

"""
Innovation Diffusion Simulation Framework.

A modular framework for simulating innovation diffusion using agent-based models,
Theory of Planned Behavior (TPB), and LLM-generated agent heterogeneity.
"""

__version__ = "0.1.0"

from .core import (
    AgentState,
    NetworkConfig,
    SimulationConfig,
    SimulationResult,
    NetworkBuilder,
    DecisionModel,
)
from .networks import SmallWorldNetworkBuilder
from .agents import BaseAgent
from .simulation import SimpleDiffusionEngine
from .visualization import NetworkVisualizer, TimeSeriesVisualizer

__all__ = [
    "__version__",
    # Core
    "AgentState",
    "NetworkConfig",
    "SimulationConfig",
    "SimulationResult",
    "NetworkBuilder",
    "DecisionModel",
    # Networks
    "SmallWorldNetworkBuilder",
    # Agents
    "BaseAgent",
    # Simulation
    "SimpleDiffusionEngine",
    # Visualization
    "NetworkVisualizer",
    "TimeSeriesVisualizer",
]

"""
Innovation Diffusion Simulation Framework.

A modular framework for simulating innovation diffusion using agent-based models,
Theory of Planned Behavior (TPB), and LLM-generated agent heterogeneity.
"""

__version__ = "0.1.0"

from .agents import BaseAgent, TPBAgent
from .core import (
    AgentState,
    DecisionModel,
    NetworkBuilder,
    NetworkConfig,
    SimulationConfig,
    SimulationResult,
)
from .decisions import SimpleTPBModel, TPBDecisionModel
from .networks import SmallWorldNetworkBuilder
from .population import PopulationGenerator, TraitSampler
from .simulation import SimpleDiffusionEngine, TPBDiffusionEngine
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
    "TPBAgent",
    # Simulation
    "SimpleDiffusionEngine",
    "TPBDiffusionEngine",
    # Decisions
    "TPBDecisionModel",
    "SimpleTPBModel",
    # Population
    "PopulationGenerator",
    "TraitSampler",
    # Visualization
    "NetworkVisualizer",
    "TimeSeriesVisualizer",
]

"""
Custom exceptions for the diffusion simulation framework.
"""


class DiffusionSimError(Exception):
    """Base exception for diffusion simulation errors."""

    pass


class NetworkBuildError(DiffusionSimError):
    """Raised when network generation fails."""

    pass


class SimulationError(DiffusionSimError):
    """Raised when simulation execution fails."""

    pass


class ConfigurationError(DiffusionSimError):
    """Raised when configuration is invalid."""

    pass


class AgentError(DiffusionSimError):
    """Raised when agent operations fail."""

    pass

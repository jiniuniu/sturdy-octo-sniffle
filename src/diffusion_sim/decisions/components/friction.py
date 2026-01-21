"""
Friction modeling for innovation adoption.

Friction represents barriers that prevent intention from converting to action:
- Cost/price barriers
- Complexity/effort required
- Risk perception
- Switching costs

In TPB, this is partially captured by PBC, but can be modeled separately
as a friction coefficient that reduces adoption probability.
"""
import random
from typing import Optional


class FrictionModel:
    """
    Model for adoption friction/barriers.

    Friction reduces the probability of adoption even when intention is high.
    P(adopt | intention) = intention * (1 - friction)
    """

    def __init__(
        self,
        base_friction: float = 0.0,
        mode: str = "fixed"
    ):
        """
        Initialize friction model.

        Args:
            base_friction: Base friction level (0-1)
            mode: Friction mode ("fixed", "random", "decreasing")
        """
        if not 0 <= base_friction <= 1:
            raise ValueError(f"base_friction must be in [0, 1], got {base_friction}")

        self.base_friction = base_friction
        self.mode = mode
        self.current_step = 0

    def apply(
        self,
        intention: float,
        agent_risk_aversion: Optional[float] = None
    ) -> float:
        """
        Apply friction to intention to get adoption probability.

        Args:
            intention: Base intention score (0-1)
            agent_risk_aversion: Agent-specific risk aversion (0-1)

        Returns:
            Adjusted adoption probability (0-1)
        """
        if not 0 <= intention <= 1:
            raise ValueError(f"intention must be in [0, 1], got {intention}")

        # Get effective friction
        friction = self.get_friction(agent_risk_aversion)

        # Apply friction: reduces probability
        # P(adopt) = intention * (1 - friction)
        return intention * (1.0 - friction)

    def get_friction(self, agent_risk_aversion: Optional[float] = None) -> float:
        """
        Get current friction level.

        Args:
            agent_risk_aversion: Agent-specific risk aversion (0-1)
                Modulates base friction

        Returns:
            Effective friction level (0-1)
        """
        if self.mode == "fixed":
            # Static friction
            base = self.base_friction

        elif self.mode == "random":
            # Random friction each time (environmental uncertainty)
            base = random.uniform(
                max(0, self.base_friction - 0.1),
                min(1, self.base_friction + 0.1)
            )

        elif self.mode == "decreasing":
            # Friction decreases over time (learning/normalization)
            # Exponential decay: f(t) = f_0 * exp(-t/tau)
            # Simplified: f(t) = f_0 * (1 - t / T)
            tau = 50  # Time constant
            decay = max(0, 1.0 - self.current_step / tau)
            base = self.base_friction * decay

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Modulate by agent risk aversion if provided
        if agent_risk_aversion is not None:
            # Higher risk aversion -> higher effective friction
            # friction' = friction + (1 - friction) * risk_aversion * 0.5
            base = base + (1.0 - base) * agent_risk_aversion * 0.5

        return min(1.0, base)

    def decide_adopt(self, intention: float, agent_risk_aversion: Optional[float] = None) -> bool:
        """
        Make binary adoption decision based on intention and friction.

        Args:
            intention: Intention score (0-1)
            agent_risk_aversion: Agent-specific risk aversion

        Returns:
            True if adopt, False otherwise
        """
        prob = self.apply(intention, agent_risk_aversion)
        return random.random() < prob

    def step(self):
        """Advance time step (for time-dependent friction modes)."""
        self.current_step += 1

    def reset(self):
        """Reset friction model state."""
        self.current_step = 0


class NoFriction(FrictionModel):
    """Convenience class for no-friction scenario."""

    def __init__(self):
        super().__init__(base_friction=0.0, mode="fixed")

    def apply(self, intention: float, agent_risk_aversion: Optional[float] = None) -> float:
        """No friction applied."""
        return intention

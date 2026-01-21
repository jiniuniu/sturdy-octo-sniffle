"""
Intention computation for Theory of Planned Behavior.

Intention (I) = f(Attitude, Social Norm, Perceived Behavioral Control)

Following Ajzen's TPB:
I = w_A * A + w_SN * SN + w_PBC * PBC

where weights sum to 1 and can be calibrated.
"""

import math
from typing import Optional


class IntentionCalculator:
    """
    Calculator for behavioral intention in TPB framework.

    Supports multiple computation modes:
    - linear: Weighted sum (standard TPB)
    - sigmoid: Non-linear transformation for bounded output
    - threshold: Binary decision based on threshold
    """

    def __init__(
        self,
        w_attitude: float = 0.4,
        w_social_norm: float = 0.35,
        w_pbc: float = 0.25,
        mode: str = "sigmoid",
    ):
        """
        Initialize intention calculator.

        Args:
            w_attitude: Weight for attitude component
            w_social_norm: Weight for social norm component
            w_pbc: Weight for perceived behavioral control
            mode: Computation mode ("linear", "sigmoid", "threshold")
        """
        # Validate weights
        total_weight = w_attitude + w_social_norm + w_pbc
        if not math.isclose(total_weight, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight:.3f}")

        self.w_attitude = w_attitude
        self.w_social_norm = w_social_norm
        self.w_pbc = w_pbc
        self.mode = mode

    def compute(
        self, attitude: float, social_norm: float, pbc: float, threshold: float = 0.5
    ) -> float:
        """
        Compute behavioral intention.

        Args:
            attitude: Attitude score (0-1)
            social_norm: Social norm score (0-1)
            pbc: Perceived behavioral control score (0-1)
            threshold: Threshold for binary mode (default 0.5)

        Returns:
            Intention score (0-1)
        """
        # Validate inputs
        for name, value in [
            ("attitude", attitude),
            ("social_norm", social_norm),
            ("pbc", pbc),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value:.3f}")

        # Weighted sum
        linear_score = (
            self.w_attitude * attitude
            + self.w_social_norm * social_norm
            + self.w_pbc * pbc
        )

        if self.mode == "linear":
            return linear_score

        elif self.mode == "sigmoid":
            # Apply sigmoid for smooth non-linearity
            # Map [0, 1] to more sensitive range
            # sigmoid(x) where x = (linear_score - 0.5) * 6
            # This makes 0.5 the inflection point
            x = (linear_score - 0.5) * 6
            return 1.0 / (1.0 + math.exp(-x))

        elif self.mode == "threshold":
            # Binary decision
            return 1.0 if linear_score >= threshold else 0.0

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def compute_with_traits(
        self,
        attitude: float,
        social_norm: float,
        pbc: float,
        conformity_sensitivity: Optional[float] = None,
    ) -> float:
        """
        Compute intention with agent-specific trait modulation.

        Args:
            attitude: Base attitude score
            social_norm: Social norm score
            pbc: PBC score
            conformity_sensitivity: Agent's sensitivity to social norm (0-1)
                If provided, modulates the social_norm component

        Returns:
            Intention score (0-1)
        """
        # Modulate social norm by conformity if provided
        if conformity_sensitivity is not None:
            effective_sn = social_norm * conformity_sensitivity
        else:
            effective_sn = social_norm

        return self.compute(attitude, effective_sn, pbc)

    def set_weights(self, w_attitude: float, w_social_norm: float, w_pbc: float):
        """
        Update TPB weights.

        Args:
            w_attitude: New weight for attitude
            w_social_norm: New weight for social norm
            w_pbc: New weight for PBC
        """
        total = w_attitude + w_social_norm + w_pbc
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

        self.w_attitude = w_attitude
        self.w_social_norm = w_social_norm
        self.w_pbc = w_pbc


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function.

    Args:
        x: Input value

    Returns:
        Sigmoid output (0-1)
    """
    return 1.0 / (1.0 + math.exp(-x))

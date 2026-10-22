"""Explicit calibration gate for comparing historical swap-leg USD prices.

A recorded difference is evidence, not permission to assign acquisition basis.
A price-disagreement tolerance is intentionally absent until validated against
real executed swaps with independently verified economic meaning.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class DivergenceDecision(StrEnum):
    """Classify two priced legs without inventing an approval tolerance."""

    UNCALIBRATED = "uncalibrated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class PriceDivergencePolicy:
    """Gate volatile valuations behind an explicitly documented calibration.

    :param max_relative_spread: Empirically justified tolerance, or None pending research.
    :param calibration_reference: Nonblank reference to calibration evidence when enabled.
    :raises TypeError: If inputs have incorrect types.
    :raises ValueError: If threshold or evidence reference is invalid.
    """

    max_relative_spread: Decimal | None = None
    calibration_reference: str | None = None

    def __post_init__(self) -> None:
        """Forbid silent heuristics and unaudited tolerances.

        :raises TypeError: If provided values use unsupported types.
        :raises ValueError: If the selected configuration is inconsistent.
        """
        if self.max_relative_spread is None:
            if self.calibration_reference is not None:
                raise ValueError("calibration_reference requires a calibrated threshold")
            return
        if not isinstance(self.max_relative_spread, Decimal):
            raise TypeError("max_relative_spread must be Decimal or None")
        if not self.max_relative_spread.is_finite() or not Decimal(
            0
        ) <= self.max_relative_spread <= Decimal(1):
            raise ValueError("max_relative_spread must be finite and between 0 and 1")
        if not isinstance(self.calibration_reference, str):
            raise TypeError("calibration_reference must be a string when calibrated")
        normalized = self.calibration_reference.strip()
        if not normalized:
            raise ValueError("a calibrated tolerance requires an evidence reference")
        object.__setattr__(self, "calibration_reference", normalized)

    def evaluate(self, relative_spread: Decimal) -> DivergenceDecision:
        """Evaluate an observed disagreement without modifying its evidence.

        :param relative_spread: Exact fractional difference between two positive USD values.
        :return: Uncalibrated, accepted or rejected decision.
        :raises TypeError: If the spread is not a Decimal.
        :raises ValueError: If the spread is nonfinite or negative.
        """
        if not isinstance(relative_spread, Decimal):
            raise TypeError("relative_spread must be Decimal")
        if not relative_spread.is_finite() or relative_spread < 0:
            raise ValueError("relative_spread must be finite and nonnegative")
        if self.max_relative_spread is None:
            return DivergenceDecision.UNCALIBRATED
        if relative_spread > self.max_relative_spread:
            return DivergenceDecision.REJECTED
        return DivergenceDecision.ACCEPTED

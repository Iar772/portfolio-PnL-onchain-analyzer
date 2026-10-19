"""USD financial values without implicit conversion or presentation rounding."""

from dataclasses import dataclass
from decimal import Decimal


def _require_finite_decimal(value: Decimal, *, owner: str) -> None:
    """Validate the numeric representation of a financial value.

    :param value: Candidate value to validate without coercion.
    :param owner: Name of the value object used in validation errors.
    :raises TypeError: If the value is not a Decimal.
    :raises ValueError: If the value is NaN or infinite.
    """
    if not isinstance(value, Decimal):
        raise TypeError(f"{owner}.value must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{owner}.value must be finite")


@dataclass(frozen=True, slots=True)
class UsdAmount:
    """Represent a signed USD amount without rounding or float conversion.

    :param value: Finite Decimal representing basis, proceeds, value, or PnL.
    """

    value: Decimal

    def __post_init__(self) -> None:
        """Enforce the USD amount's numeric invariant at construction time.

        :raises TypeError: If value is not a Decimal.
        :raises ValueError: If value is not finite.
        """
        _require_finite_decimal(self.value, owner="UsdAmount")


@dataclass(frozen=True, slots=True)
class UsdPrice:
    """Represent a known, non-negative USD price per asset unit.

    Zero is valid only when it is an explicitly known price, not missing data.

    :param value: Finite non-negative Decimal price per one asset unit.
    """

    value: Decimal

    def __post_init__(self) -> None:
        """Enforce the USD price's numeric and sign invariants.

        :raises TypeError: If value is not a Decimal.
        :raises ValueError: If value is not finite or is negative.
        """
        _require_finite_decimal(self.value, owner="UsdPrice")

        if self.value < 0:
            raise ValueError("UsdPrice value must not be negative")

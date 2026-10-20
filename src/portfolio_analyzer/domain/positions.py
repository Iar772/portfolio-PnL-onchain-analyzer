"""Deterministic per-asset inventory coverage from remaining acquisition lots."""

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.lots import Lot
from portfolio_analyzer.domain.money import UsdAmount, UsdPrice

# Ratios such as 1/3 cannot be represented exactly as finite decimals.
# Keep ratio math reproducible; UI rounding belongs elsewhere.
_RATIO_CONTEXT = Context(prec=50, rounding=ROUND_HALF_EVEN)


def _sum_exact(values: Iterable[Decimal]) -> Decimal:
    """Sum finite Decimal values exactly regardless of the ambient context.

    Convert each number to an integer coefficient at a shared decimal
    exponent, then construct the result without Decimal arithmetic rounding.

    :param values: Finite Decimal values to add.
    :return: Exact sum, or Decimal zero when the iterable is empty.
    """
    terms: list[tuple[int, int]] = []
    for value in values:
        parts = value.as_tuple()
        exponent = parts.exponent
        if not isinstance(exponent, int):
            raise ValueError("_sum_exact requires finite Decimal values")
        coefficient = int("".join(str(digit) for digit in parts.digits))
        terms.append((-coefficient if parts.sign else coefficient, exponent))

    if not terms:
        return Decimal("0")

    common_exponent = min(exponent for _, exponent in terms)
    total_coefficient = sum(
        coefficient * 10 ** (exponent - common_exponent) for coefficient, exponent in terms
    )
    return Decimal(f"{total_coefficient}e{common_exponent}")


@dataclass(frozen=True, slots=True)
class Position:
    """Represent one asset's current replayed inventory and cost-basis coverage.

    Quantities are computed only from remaining lot inventory. A known-basis
    lot with zero remaining quantity contributes nothing. A basis of None is
    unknown, NOT a dollar value of zero.

    :param asset_id: Canonical identity of the asset whose lots are grouped.
    :param lots: Immutable acquisition lot states for that same asset.
    :raises TypeError: If identity or lots have the wrong types.
    :raises ValueError: If any lot belongs to a different asset.
    """

    asset_id: AssetId
    lots: tuple[Lot, ...]

    def __post_init__(self) -> None:
        """Ensure the position contains only lots for its own asset.

        :raises TypeError: If asset_id or lots have an invalid type.
        :raises ValueError: If any lot has another asset identity.
        """
        if not isinstance(self.asset_id, AssetId):
            raise TypeError("Position.asset_id must be an AssetId")
        if not isinstance(self.lots, tuple):
            raise TypeError("Position.lots must be a tuple")
        for lot in self.lots:
            if not isinstance(lot, Lot):
                raise TypeError("Position.lots must contain Lot instances")
            if lot.asset_id != self.asset_id:
                raise ValueError("Position.lots must belong to the same asset")

    @property
    def total_quantity(self) -> Decimal:
        """Add all remaining inventory, regardless of cost-basis knowledge.

        :return: Exact total token quantity in the position.
        """
        return _sum_exact(lot.remaining_quantity for lot in self.lots)

    @property
    def known_quantity(self) -> Decimal:
        """Add the remaining inventory of lots with known acquisition basis.

        :return: Exact token quantity whose USD cost basis is known.
        """
        return _sum_exact(lot.remaining_quantity for lot in self.lots if lot.is_basis_known)

    @property
    def unknown_quantity(self) -> Decimal:
        """Add the remaining inventory of lots with unknown acquisition basis.

        :return: Exact token quantity without a known USD cost basis.
        """
        return _sum_exact(lot.remaining_quantity for lot in self.lots if not lot.is_basis_known)

    @property
    def known_cost_basis(self) -> UsdAmount | None:
        """Sum remaining USD basis ONLY for known, nonexhausted lots.

        No known inventory means None, not a manufactured known zero.
        An actual known zero-basis lot with remaining inventory yields $0.

        :return: USD basis for the known portion, or None if none is known.
        """
        active_known_lots = tuple(
            lot for lot in self.lots if lot.is_basis_known and lot.remaining_quantity > 0
        )
        if not active_known_lots:
            return None

        return UsdAmount(
            _sum_exact(
                lot.remaining_cost_basis.value
                for lot in active_known_lots
                if lot.remaining_cost_basis is not None
            )
        )

    @property
    def known_basis_coverage(self) -> Decimal | None:
        """Calculate the known-basis fraction of current token inventory.

        Coverage is between 0 and 1 (0.8 means 80%), not a confidence score.
        An empty position has undefined coverage, represented by None.

        :return: Known quantity divided by total quantity, or None if empty.
        """
        with localcontext(_RATIO_CONTEXT):
            total = self.total_quantity
            if total == 0:
                return None
            return self.known_quantity / total

    @property
    def average_entry(self) -> UsdPrice | None:
        """Calculate USD entry price per token for the known portion only.

        Unknown-basis lots never contribute to the numerator or denominator.
        The ratio is not rounded for presentation inside the domain.

        :return: Known-portion USD entry price, or None with no known inventory.
        """
        known_quantity = self.known_quantity
        if known_quantity == 0:
            return None

        known_basis = self.known_cost_basis
        assert known_basis is not None

        with localcontext(_RATIO_CONTEXT):
            return UsdPrice(known_basis.value / known_quantity)

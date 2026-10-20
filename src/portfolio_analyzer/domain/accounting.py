"""Deterministic, in-memory FIFO inventory consumption without PnL valuation."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from portfolio_analyzer.domain.assets import AssetAmount
from portfolio_analyzer.domain.errors import InsufficientQuantityError
from portfolio_analyzer.domain.lots import Lot
from portfolio_analyzer.domain.money import UsdAmount


@dataclass(frozen=True, slots=True)
class LotConsumption:
    """Record how much of one acquisition lot was consumed by a disposal.

    :param source_lot: Lot state immediately before this consumption.
    :param quantity: Positive quantity taken from the source lot.
    :param cost_basis: Consumed USD basis, or None if the source basis is unknown.
    """

    source_lot: Lot
    quantity: Decimal
    cost_basis: UsdAmount | None


@dataclass(frozen=True, slots=True)
class FifoConsumptionResult:
    """Return the updated inventory and a per-lot disposal audit trail.

    Exhausted lots are retained with zero inventory in ``updated_lots``.
    No realized PnL is calculated at this stage.

    :param disposal: Exact amount successfully consumed.
    :param updated_lots: Immutable lot states in FIFO order, including exhausted lots.
    :param consumptions: Ordered record of each lot contributing to the disposal.
    """

    disposal: AssetAmount
    updated_lots: tuple[Lot, ...]
    consumptions: tuple[LotConsumption, ...]


def _working_precision(lots: Sequence[Lot], disposal: AssetAmount) -> int:
    """Choose a reproducible Decimal precision independent of ambient settings.

    The guard digits cover typical on-chain quantities, large coefficients, and
    high-scale decimals. Division may still repeat, so its fractional share
    is approximate; the remaining lot retains the complementary residual.

    :param lots: Candidate lots for the requested asset.
    :param disposal: Requested whole-token quantity to consume.
    :return: Significant-digit precision for accounting arithmetic.
    """
    values = [disposal.quantity]
    for lot in lots:
        values.append(lot.remaining_quantity)
        if lot.remaining_cost_basis is not None:
            values.append(lot.remaining_cost_basis.value)
    largest_coefficient = max(len(value.as_tuple().digits) for value in values)
    largest_exponent = max(abs(value.adjusted()) for value in values)
    return max(80, 2 * largest_coefficient + largest_exponent + 16)


def _split_known_basis(lot: Lot, consumed_quantity: Decimal) -> tuple[UsdAmount, UsdAmount]:
    """Split the remaining known USD basis for a partial lot consumption.

    The caller guarantees a known-basis lot and a strictly partial amount.
    Return the allocated USD basis AND its complementary remainder so the
    original basis is conserved rather than rounded to display cents.

    :param lot: Known-basis lot before partial disposal.
    :param consumed_quantity: Quantity consumed, strictly below remaining quantity.
    :return: Pair of consumed USD basis and the new remaining USD basis.
    """
    remaining_basis = lot.remaining_cost_basis
    if remaining_basis is None:
        raise ValueError("A known cost basis is required for proportional allocation")

    consumed_value = remaining_basis.value * consumed_quantity / lot.remaining_quantity
    remaining_value = remaining_basis.value - consumed_value

    return UsdAmount(consumed_value), UsdAmount(remaining_value)


def consume_fifo(lots: Sequence[Lot], disposal: AssetAmount) -> FifoConsumptionResult:
    """Consume an asset's lots oldest-first without mutating caller state.

    Lots are sorted by acquisition timestamp. Python's stable sort preserves
    the input order for ties, which must already reflect chain replay order.
    Returns updated lots including exhausted inventory for future audit.

    :param lots: Acquisition lots belonging to the asset being disposed.
    :param disposal: Strictly positive quantity to remove from this asset.
    :return: Updated lots and per-lot basis information.
    :raises TypeError: If disposal or one of the lots has an invalid type.
    :raises ValueError: If the disposal is nonpositive or assets do not match.
    :raises InsufficientQuantityError: If available inventory is insufficient.
    """
    if not isinstance(disposal, AssetAmount):
        raise TypeError("disposal must be an AssetAmount")
    if disposal.quantity <= 0:
        raise ValueError("disposal.quantity must be positive")

    for lot in lots:
        if not isinstance(lot, Lot):
            raise TypeError("lots must contain Lot instances")
        if lot.asset_id != disposal.asset_id:
            raise ValueError("All lots must belong to the disposal asset")

    ordered_lots = sorted(lots, key=lambda lot: lot.acquired_at)
    updated_lots: list[Lot] = []
    consumptions: list[LotConsumption] = []

    context = Context(prec=_working_precision(ordered_lots, disposal), rounding=ROUND_HALF_EVEN)
    with localcontext(context):
        remaining_to_consume = disposal.quantity

        for lot in ordered_lots:
            if remaining_to_consume == 0 or lot.remaining_quantity == 0:
                updated_lots.append(lot)
                continue

            taken = min(lot.remaining_quantity, remaining_to_consume)
            new_quantity = lot.remaining_quantity - taken

            if lot.remaining_cost_basis is None:
                consumed_basis = None
                new_basis = None
            elif taken == lot.remaining_quantity:
                consumed_basis = lot.remaining_cost_basis
                new_basis = UsdAmount(Decimal("0"))
            else:
                consumed_basis, new_basis = _split_known_basis(lot, taken)

            updated_lots.append(
                replace(lot, remaining_quantity=new_quantity, remaining_cost_basis=new_basis)
            )
            consumptions.append(
                LotConsumption(source_lot=lot, quantity=taken, cost_basis=consumed_basis)
            )
            remaining_to_consume -= taken

        if remaining_to_consume > 0:
            raise InsufficientQuantityError(
                f"Insufficient quantity: missing {remaining_to_consume} "
                f"for asset {disposal.asset_id}"
            )

    return FifoConsumptionResult(
        disposal=disposal,
        updated_lots=tuple(updated_lots),
        consumptions=tuple(consumptions),
    )

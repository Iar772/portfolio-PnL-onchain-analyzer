"""Deterministic unit tests for atomic FIFO lot consumption and USD basis."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Context, Decimal, localcontext
from typing import cast

import pytest

from portfolio_analyzer.domain.accounting import consume_fifo
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.errors import DomainError, InsufficientQuantityError
from portfolio_analyzer.domain.lots import Lot
from portfolio_analyzer.domain.money import UsdAmount

ETH = AssetId(chain_id=1, contract_address=None)
LINK = AssetId(chain_id=1, contract_address="0x" + "12" * 20)
OTHER_TOKEN = AssetId(chain_id=1, contract_address="0x" + "34" * 20)
START = datetime(2026, 1, 1, tzinfo=UTC)


def make_lot(
    quantity: str,
    basis: str | None,
    *,
    asset_id: AssetId = LINK,
    days: int = 0,
) -> Lot:
    """Construct a known- or unknown-basis acquisition lot for FIFO tests.

    :param quantity: Whole-token acquisition amount as a decimal string.
    :param basis: Known USD acquisition cost, or None for unknown basis.
    :param asset_id: Identity of the asset acquired.
    :param days: Offset from the common acquisition date.
    :return: A new, completely unconsumed Lot.
    """
    return Lot.open(
        AssetAmount(asset_id, Decimal(quantity)),
        cost_basis=None if basis is None else UsdAmount(Decimal(basis)),
        acquired_at=START + timedelta(days=days),
    )


def disposal(quantity: str, *, asset_id: AssetId = LINK) -> AssetAmount:
    """Build a positive or invalid disposal for testing.

    :param quantity: Whole-token disposal amount as a string.
    :param asset_id: Asset to consume.
    :return: The requested AssetAmount.
    """
    return AssetAmount(asset_id, Decimal(quantity))


def test_full_disposal_consumes_exact_basis_and_preserves_acquisition() -> None:
    """Fully consume one known lot without modifying its original facts."""
    original = make_lot("10", "100")

    result = consume_fifo([original], disposal("10"))

    assert result.disposal == disposal("10")
    assert len(result.consumptions) == 1
    assert result.consumptions[0].source_lot is original
    assert result.consumptions[0].quantity == Decimal("10")
    assert result.consumptions[0].cost_basis == UsdAmount(Decimal("100"))
    assert result.updated_lots[0].remaining_quantity == Decimal("0")
    assert result.updated_lots[0].remaining_cost_basis == UsdAmount(Decimal("0"))
    assert result.updated_lots[0].original_quantity == Decimal("10")
    assert result.updated_lots[0].original_cost_basis == UsdAmount(Decimal("100"))
    assert original.remaining_quantity == Decimal("10")


def test_partial_disposal_allocates_only_proportional_basis() -> None:
    """Take three quarters of one known lot while preserving the remainder."""
    original = make_lot("8", "200")

    result = consume_fifo([original], disposal("6"))

    assert result.consumptions[0].quantity == Decimal("6")
    assert result.consumptions[0].cost_basis == UsdAmount(Decimal("150"))
    assert result.updated_lots[0].remaining_quantity == Decimal("2")
    assert result.updated_lots[0].remaining_cost_basis == UsdAmount(Decimal("50"))


def test_multiple_lots_are_consumed_oldest_first() -> None:
    """Consume a full old lot and then part of a newer acquisition."""
    oldest = make_lot("10", "100", days=1)
    newer = make_lot("20", "300", days=2)

    result = consume_fifo([newer, oldest], disposal("12"))

    assert [entry.source_lot for entry in result.consumptions] == [oldest, newer]
    assert [entry.quantity for entry in result.consumptions] == [Decimal("10"), Decimal("2")]
    assert [entry.cost_basis for entry in result.consumptions] == [
        UsdAmount(Decimal("100")),
        UsdAmount(Decimal("30")),
    ]
    assert [lot.remaining_quantity for lot in result.updated_lots] == [
        Decimal("0"),
        Decimal("18"),
    ]
    assert result.updated_lots[1].remaining_cost_basis == UsdAmount(Decimal("270"))


def test_full_disposal_across_three_lots() -> None:
    """Consume all inventory of three lots in chronological order."""
    lots = [
        make_lot("2", "20", days=1),
        make_lot("3", "60", days=2),
        make_lot("4", "120", days=3),
    ]

    result = consume_fifo(lots, disposal("9"))

    assert len(result.consumptions) == 3
    assert [entry.quantity for entry in result.consumptions] == list(map(Decimal, ["2", "3", "4"]))
    assert all(lot.remaining_quantity == 0 for lot in result.updated_lots)
    assert [entry.cost_basis for entry in result.consumptions] == [
        UsdAmount(Decimal("20")),
        UsdAmount(Decimal("60")),
        UsdAmount(Decimal("120")),
    ]


def test_same_timestamp_keeps_caller_replay_order() -> None:
    """Keep a stable ordering for lots with the same acquisition timestamp."""
    first = make_lot("2", "20")
    second = make_lot("2", "40")

    result = consume_fifo([second, first], disposal("3"))

    assert [entry.source_lot for entry in result.consumptions] == [second, first]
    assert [entry.cost_basis for entry in result.consumptions] == [
        UsdAmount(Decimal("40")),
        UsdAmount(Decimal("10")),
    ]


def test_unknown_partial_disposal_preserves_unknown_basis() -> None:
    """Consume some unknown-basis inventory without manufacturing a zero."""
    original = make_lot("5", None)

    result = consume_fifo([original], disposal("2"))

    assert result.consumptions[0].cost_basis is None
    assert result.updated_lots[0].remaining_cost_basis is None
    assert result.updated_lots[0].remaining_quantity == Decimal("3")
    assert result.updated_lots[0].is_basis_known is False


def test_unknown_full_disposal_still_has_unknown_basis() -> None:
    """Keep historical uncertainty after exhausting unknown-basis inventory."""
    result = consume_fifo([make_lot("1", None)], disposal("1"))

    assert result.consumptions[0].cost_basis is None
    assert result.updated_lots[0].remaining_cost_basis is None
    assert result.updated_lots[0].remaining_quantity == 0


def test_mixed_known_unknown_lots_report_each_cost_separately() -> None:
    """Keep each source lot's cost-basis knownness in a mixed disposal."""
    lots = [
        make_lot("2", "20", days=0),
        make_lot("3", None, days=1),
        make_lot("4", "80", days=2),
    ]

    result = consume_fifo(lots, disposal("6"))

    assert [entry.quantity for entry in result.consumptions] == [
        Decimal("2"),
        Decimal("3"),
        Decimal("1"),
    ]
    assert [entry.cost_basis for entry in result.consumptions] == [
        UsdAmount(Decimal("20")),
        None,
        UsdAmount(Decimal("20")),
    ]
    assert result.updated_lots[2].remaining_quantity == Decimal("3")
    assert result.updated_lots[2].remaining_cost_basis == UsdAmount(Decimal("60"))


def test_explicit_known_zero_is_not_unknown() -> None:
    """Preserve known $0 basis as a real value in a disposal."""
    result = consume_fifo([make_lot("2", "0")], disposal("1"))

    assert result.consumptions[0].cost_basis == UsdAmount(Decimal("0"))
    assert result.updated_lots[0].remaining_cost_basis == UsdAmount(Decimal("0"))


def test_skip_previously_exhausted_lot_without_a_consumption_entry() -> None:
    """Skip zero-remaining older lots while retaining them in updated states."""
    exhausted = make_lot("1", "10", days=0)
    exhausted = replace(
        exhausted,
        remaining_quantity=Decimal("0"),
        remaining_cost_basis=UsdAmount(Decimal("0")),
    )
    active = make_lot("3", "30", days=1)

    result = consume_fifo([active, exhausted], disposal("2"))

    assert result.updated_lots[0] is exhausted
    assert len(result.consumptions) == 1
    assert result.consumptions[0].source_lot is active
    assert result.updated_lots[1].remaining_quantity == Decimal("1")


def test_insufficient_quantity_raises_domain_error_without_mutation() -> None:
    """Fail atomically rather than returning partially depleted lot states."""
    original = make_lot("4", "40")

    with pytest.raises(InsufficientQuantityError, match="Insufficient quantity"):
        consume_fifo([original], disposal("5"))

    assert original.remaining_quantity == Decimal("4")
    assert original.remaining_cost_basis == UsdAmount(Decimal("40"))
    assert issubclass(InsufficientQuantityError, DomainError)


def test_empty_lots_cannot_fulfill_disposal() -> None:
    """Raise a domain error if disposal has no available lots."""
    with pytest.raises(InsufficientQuantityError, match="Insufficient quantity"):
        consume_fifo([], disposal("1"))


@pytest.mark.parametrize("quantity", ["0", "-1", "-0.00001"])
def test_nonpositive_disposal_is_invalid(quantity: str) -> None:
    """Reject a request that has no valid strictly positive magnitude.

    :param quantity: A nonpositive amount represented as a decimal string.
    """
    with pytest.raises(ValueError, match="disposal.quantity must be positive"):
        consume_fifo([make_lot("5", "50")], disposal(quantity))


def test_disposal_requires_asset_amount() -> None:
    """Reject invalid runtime argument types at the accounting boundary."""
    with pytest.raises(TypeError, match="disposal must be an AssetAmount"):
        consume_fifo([], cast(AssetAmount, "5"))


def test_foreign_asset_lot_is_rejected_even_if_quantity_is_sufficient() -> None:
    """Prevent ETH or another ERC-20's inventory from being sold as LINK."""
    with pytest.raises(ValueError, match="disposal asset"):
        consume_fifo([make_lot("3", "30", asset_id=ETH)], disposal("2"))


def test_other_contract_with_same_chain_is_rejected() -> None:
    """Identify token inventory by contract address, not a ticker or chain alone."""
    with pytest.raises(ValueError, match="disposal asset"):
        consume_fifo([make_lot("3", "30", asset_id=OTHER_TOKEN)], disposal("2"))


def test_invalid_lot_type_is_rejected() -> None:
    """Ensure all inventory entries are real Lot instances."""
    with pytest.raises(TypeError, match="lots must contain Lot instances"):
        consume_fifo([cast(Lot, "fake")], disposal("1"))


def test_partial_basis_retains_fractional_residual() -> None:
    """Preserve the complementary remainder for a recurring 1/3 share."""
    original = make_lot("3", "100")

    result = consume_fifo([original], disposal("1"))

    used = result.consumptions[0].cost_basis
    left = result.updated_lots[0].remaining_cost_basis
    assert used is not None and left is not None
    with localcontext(Context(prec=80)):
        assert used.value + left.value == Decimal("100")
    assert left.value > used.value


def test_full_final_disposal_consume_exact_residual_after_partial() -> None:
    """Give the last disposal the exact remaining cost basis with no dust."""
    original = make_lot("3", "100")
    first = consume_fifo([original], disposal("1"))

    second = consume_fifo(first.updated_lots, disposal("2"))

    assert second.consumptions[0].cost_basis == first.updated_lots[0].remaining_cost_basis
    assert second.updated_lots[0].remaining_cost_basis == UsdAmount(Decimal("0"))
    assert second.updated_lots[0].remaining_quantity == Decimal("0")


def test_arithmetic_is_independent_of_callers_decimal_context() -> None:
    """Ignore an ambient low-precision context during basis allocation."""
    original = make_lot("7", "12345678901234567890.123456789")

    with localcontext(Context(prec=6)):
        result = consume_fifo([original], disposal("2"))

    consumed = result.consumptions[0].cost_basis
    remaining = result.updated_lots[0].remaining_cost_basis
    assert consumed is not None and remaining is not None
    original_basis = original.remaining_cost_basis
    assert original_basis is not None
    with localcontext(Context(prec=80)):
        assert consumed.value + remaining.value == original_basis.value
    assert result.updated_lots[0].remaining_quantity == Decimal("5")


def test_original_sequence_is_never_mutated() -> None:
    """Return new lots and leave the input collection and its lots unchanged."""
    first = make_lot("5", "50", days=0)
    second = make_lot("2", "20", days=1)
    inventory = [second, first]

    result = consume_fifo(inventory, disposal("1"))

    assert inventory == [second, first]
    assert first.remaining_quantity == Decimal("5")
    assert second.remaining_quantity == Decimal("2")
    assert result.updated_lots[0] is not first
    assert result.updated_lots[1] is second

"""Deterministic tests for per-asset known and unknown cost-basis coverage."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from typing import cast

import pytest

from portfolio_analyzer.domain.accounting import consume_fifo
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.lots import Lot
from portfolio_analyzer.domain.money import UsdAmount
from portfolio_analyzer.domain.positions import Position, _sum_exact

LINK = AssetId(1, "0x" + "12" * 20)
OTHER_LINK = AssetId(1, "0x" + "34" * 20)
ACQUIRED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def make_lot(quantity: str, basis: str | None, *, days: int = 0) -> Lot:
    """Create an initial lot with known or unknown USD acquisition cost.

    :param quantity: Positive asset quantity expressed as a decimal string.
    :param basis: Known cost as a decimal string, or None for unknown.
    :param days: Days after the initial acquisition timestamp.
    :return: A newly opened immutable LINK lot.
    """
    return Lot.open(
        AssetAmount(LINK, Decimal(quantity)),
        cost_basis=UsdAmount(Decimal(basis)) if basis is not None else None,
        acquired_at=ACQUIRED_AT + timedelta(days=days),
    )


def test_eighty_known_twenty_unknown_gives_eighty_percent_coverage() -> None:
    """Count unknown inventory without pretending it has a USD cost basis."""
    position = Position(LINK, (make_lot("80", "800"), make_lot("20", None)))

    assert position.total_quantity == Decimal("100")
    assert position.known_quantity == Decimal("80")
    assert position.unknown_quantity == Decimal("20")
    assert position.known_basis_coverage == Decimal("0.8")
    assert position.known_cost_basis == UsdAmount(Decimal("800"))


def test_all_unknown_lots_have_zero_coverage_but_no_known_dollar_basis() -> None:
    """Keep 0% coverage separate from a zero-dollar known purchase."""
    position = Position(LINK, (make_lot("7", None), make_lot("3", None)))

    assert position.total_quantity == Decimal("10")
    assert position.known_quantity == Decimal("0")
    assert position.unknown_quantity == Decimal("10")
    assert position.known_basis_coverage == Decimal("0")
    assert position.known_cost_basis is None


def test_all_known_lots_have_complete_coverage() -> None:
    """Report all inventory and all remaining known basis for known lots."""
    position = Position(LINK, (make_lot("7", "70"), make_lot("3", "60")))

    assert position.total_quantity == Decimal("10")
    assert position.known_quantity == Decimal("10")
    assert position.unknown_quantity == Decimal("0")
    assert position.known_basis_coverage == Decimal("1")
    assert position.known_cost_basis == UsdAmount(Decimal("130"))


def test_known_zero_cost_basis_is_not_unknown() -> None:
    """Include a genuinely known zero-cost lot in coverage and known basis."""
    position = Position(LINK, (make_lot("8", "0"), make_lot("2", None)))

    assert position.known_quantity == Decimal("8")
    assert position.unknown_quantity == Decimal("2")
    assert position.known_basis_coverage == Decimal("0.8")
    assert position.known_cost_basis == UsdAmount(Decimal("0"))


def test_multiple_known_and_unknown_lots_are_aggregated_separately() -> None:
    """Aggregate all lots while excluding unknown basis from USD summation."""
    position = Position(
        LINK,
        (make_lot("5", "20"), make_lot("2", None), make_lot("15", "75"), make_lot("3", None)),
    )

    assert position.total_quantity == Decimal("25")
    assert position.known_quantity == Decimal("20")
    assert position.unknown_quantity == Decimal("5")
    assert position.known_basis_coverage == Decimal("0.8")
    assert position.known_cost_basis == UsdAmount(Decimal("95"))


def test_exhausted_lots_contribute_no_current_quantity_or_basis() -> None:
    """Exclude fully consumed history from the current position totals."""
    historical_known = make_lot("5", "50")
    historical_unknown = make_lot("6", None)
    exhausted_known = replace(
        historical_known,
        remaining_quantity=Decimal("0"),
        remaining_cost_basis=UsdAmount(Decimal("0")),
    )
    exhausted_unknown = replace(historical_unknown, remaining_quantity=Decimal("0"))
    position = Position(LINK, (exhausted_known, exhausted_unknown, make_lot("2", None)))

    assert position.total_quantity == Decimal("2")
    assert position.known_quantity == Decimal("0")
    assert position.unknown_quantity == Decimal("2")
    assert position.known_basis_coverage == Decimal("0")
    assert position.known_cost_basis is None


def test_empty_position_has_undefined_coverage_and_no_basis() -> None:
    """Avoid reporting a fabricated percentage for an empty position."""
    position = Position(LINK, ())

    assert position.total_quantity == Decimal("0")
    assert position.known_quantity == Decimal("0")
    assert position.unknown_quantity == Decimal("0")
    assert position.known_basis_coverage is None
    assert position.known_cost_basis is None


def test_fully_depleted_position_has_undefined_coverage() -> None:
    """Treat a position with only exhausted lots as having no inventory."""
    empty_lot = replace(
        make_lot("2", "10"),
        remaining_quantity=Decimal("0"),
        remaining_cost_basis=UsdAmount(Decimal("0")),
    )
    position = Position(LINK, (empty_lot,))

    assert position.total_quantity == Decimal("0")
    assert position.known_cost_basis is None
    assert position.known_basis_coverage is None


def test_position_reflects_remaining_lots_after_fifo_consumption() -> None:
    """Aggregate updated lots after FIFO without restoring consumed inventory."""
    original = [make_lot("10", "100"), make_lot("20", None, days=1)]
    result = consume_fifo(original, AssetAmount(LINK, Decimal("5")))
    position = Position(LINK, result.updated_lots)

    assert position.total_quantity == Decimal("25")
    assert position.known_quantity == Decimal("5")
    assert position.unknown_quantity == Decimal("20")
    assert position.known_basis_coverage == Decimal("0.2")
    assert position.known_cost_basis == UsdAmount(Decimal("50"))
    assert original[0].remaining_quantity == Decimal("10")


def test_fractional_quantities_remain_exact_even_under_low_decimal_precision() -> None:
    """Do not lose token dust when global Decimal precision is small."""
    with localcontext() as context:
        context.prec = 6
        position = Position(
            LINK,
            (
                make_lot("12345678901234567890", "100"),
                make_lot("0.000000000000000001", None),
            ),
        )
        assert position.total_quantity == Decimal("12345678901234567890.000000000000000001")
        assert position.known_quantity == Decimal("12345678901234567890")
        assert position.unknown_quantity == Decimal("0.000000000000000001")
        assert position.known_cost_basis == UsdAmount(Decimal("100"))


def test_coverage_is_independent_of_ambient_decimal_context() -> None:
    """Use the domain ratio context rather than the caller's global precision."""
    position = Position(LINK, (make_lot("1", "2"), make_lot("2", None)))
    with localcontext() as context:
        context.prec = 4
        result_low_precision = position.known_basis_coverage
    with localcontext() as context:
        context.prec = 30
        result_high_precision = position.known_basis_coverage

    assert result_low_precision == result_high_precision
    assert result_low_precision is not None
    assert str(result_low_precision).startswith("0.33333333333333333333")


def test_exact_sum_preserves_small_quantity_without_rounding() -> None:
    """Accumulate widely different exponents exactly with the integer helper."""
    with localcontext() as context:
        context.prec = 3
        actual = _sum_exact([Decimal("100000000000000000000"), Decimal("0.000000000000000001")])

    assert actual == Decimal("100000000000000000000.000000000000000001")


def test_exact_sum_empty_input_is_zero() -> None:
    """Return an exact zero for a collection without any values."""
    assert _sum_exact(()) == Decimal("0")


def test_position_rejects_lots_of_another_asset() -> None:
    """Reject a mixed-asset position even when symbols would match."""
    foreign_lot = replace(make_lot("1", None), asset_id=OTHER_LINK)

    with pytest.raises(ValueError, match="same asset"):
        Position(LINK, (foreign_lot,))


@pytest.mark.parametrize("invalid", [None, [], {}, "lots", 42])
def test_position_requires_immutable_lot_tuple(invalid: object) -> None:
    """Validate the tuple boundary before reading lot identities.

    :param invalid: Malformed collection passed deliberately at runtime.
    """
    with pytest.raises(TypeError, match="must be a tuple"):
        Position(LINK, cast(tuple[Lot, ...], invalid))


@pytest.mark.parametrize("invalid", [None, "not a lot", 42, True])
def test_position_rejects_invalid_lot_members(invalid: object) -> None:
    """Reject malformed lot entries even if they arrive inside a tuple.

    :param invalid: Non-Lot item inserted for runtime boundary checking.
    """
    with pytest.raises(TypeError, match="must contain Lot"):
        Position(LINK, (cast(Lot, invalid),))


def test_position_requires_asset_id() -> None:
    """Reject an invalid asset identity before inspecting lots."""
    with pytest.raises(TypeError, match="must be an AssetId"):
        Position(cast(AssetId, "LINK"), ())


def test_position_is_immutable() -> None:
    """Keep position state immutable to support deterministic accounting."""
    position = Position(LINK, (make_lot("1", "10"),))
    with pytest.raises(FrozenInstanceError):
        position.asset_id = OTHER_LINK

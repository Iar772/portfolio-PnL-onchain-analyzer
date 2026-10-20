"""Unit tests for remaining known-portion USD average entry prices."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, localcontext

from portfolio_analyzer.domain.accounting import consume_fifo
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.lots import Lot
from portfolio_analyzer.domain.money import UsdAmount, UsdPrice
from portfolio_analyzer.domain.positions import Position

LINK = AssetId(1, "0x" + "12" * 20)
ACQUIRED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def make_lot(quantity: str, basis: str | None) -> Lot:
    """Build a LINK acquisition lot with known or unknown cost basis.

    :param quantity: Token quantity as a decimal string.
    :param basis: Acquisition cost in USD, or None for unknown.
    :return: Open lot with all initial inventory available.
    """
    return Lot.open(
        AssetAmount(LINK, Decimal(quantity)),
        cost_basis=UsdAmount(Decimal(basis)) if basis is not None else None,
        acquired_at=ACQUIRED_AT,
    )


def test_average_entry_for_fully_known_position() -> None:
    """Calculate a weighted price from all remaining known lots."""
    position = Position(LINK, (make_lot("10", "100"), make_lot("20", "300")))

    assert position.known_basis_coverage == Decimal("1")
    assert position.known_quantity == Decimal("30")
    price = position.average_entry
    assert isinstance(price, UsdPrice)
    assert str(price.value).startswith("13.333333333333333333333333333333")


def test_average_entry_excludes_unknown_quantity_from_denominator() -> None:
    """Do not dilute known entry price with unknown incoming tokens."""
    position = Position(
        LINK,
        (make_lot("50", "500"), make_lot("30", "450"), make_lot("20", None)),
    )

    assert position.known_basis_coverage == Decimal("0.8")
    assert position.total_quantity == Decimal("100")
    assert position.known_quantity == Decimal("80")
    assert position.known_cost_basis == UsdAmount(Decimal("950"))
    assert position.average_entry == UsdPrice(Decimal("11.875"))


def test_average_entry_is_none_when_basis_is_entirely_unknown() -> None:
    """Distinguish unknown entry from a known zero-dollar entry."""
    position = Position(LINK, (make_lot("8", None), make_lot("2", None)))

    assert position.known_basis_coverage == Decimal("0")
    assert position.average_entry is None


def test_average_entry_is_none_for_empty_position() -> None:
    """Avoid dividing by zero when the position has no inventory."""
    position = Position(LINK, ())

    assert position.known_basis_coverage is None
    assert position.average_entry is None


def test_exhausted_known_lots_do_not_create_an_average_entry() -> None:
    """Ignore exhausted known lots even if their original basis was known."""
    exhausted = replace(
        make_lot("10", "100"),
        remaining_quantity=Decimal("0"),
        remaining_cost_basis=UsdAmount(Decimal("0")),
    )
    position = Position(LINK, (exhausted, make_lot("5", None)))

    assert position.known_quantity == Decimal("0")
    assert position.average_entry is None


def test_explicit_known_zero_basis_produces_known_zero_average() -> None:
    """Keep a known zero-dollar acquisition separate from missing basis."""
    position = Position(LINK, (make_lot("8", "0"), make_lot("2", None)))

    assert position.known_basis_coverage == Decimal("0.8")
    assert position.average_entry == UsdPrice(Decimal("0"))


def test_average_entry_uses_remaining_fifo_basis_not_original_basis() -> None:
    """Recompute entry from inventory left after a partial FIFO disposal."""
    original = (make_lot("10", "100"), make_lot("20", None))
    consumed = consume_fifo(original, AssetAmount(LINK, Decimal("5")))
    position = Position(LINK, consumed.updated_lots)

    assert position.known_cost_basis == UsdAmount(Decimal("50"))
    assert position.known_quantity == Decimal("5")
    assert position.average_entry == UsdPrice(Decimal("10"))
    assert position.known_basis_coverage == Decimal("0.2")


def test_average_entry_does_not_round_to_usd_cents() -> None:
    """Preserve sub-cent prices in the domain without UI quantization."""
    position = Position(LINK, (make_lot("8", "0.00000008"),))

    assert position.average_entry == UsdPrice(Decimal("0.00000001"))


def test_nonterminating_average_is_independent_of_ambient_precision() -> None:
    """Use a stable domain context without leaking or inheriting caller settings."""
    position = Position(LINK, (make_lot("3", "1"),))
    with localcontext() as context:
        context.prec = 4
        low_precision = position.average_entry
        assert context.prec == 4
    with localcontext() as context:
        context.prec = 30
        high_precision = position.average_entry
        assert context.prec == 30

    assert low_precision == high_precision
    assert low_precision is not None
    assert str(low_precision.value).startswith("0.333333333333333333333333333")

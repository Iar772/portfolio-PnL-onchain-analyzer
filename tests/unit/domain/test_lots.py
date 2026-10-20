"""Deterministic unit tests for FIFO lot facts and known/unknown basis."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from typing import cast

import pytest

from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.lots import Lot
from portfolio_analyzer.domain.money import UsdAmount

ETH = AssetId(chain_id=1, contract_address=None)
ACQUIRED_AT = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


def open_known_lot(quantity: str = "10", basis: str = "100") -> Lot:
    """Build a known-basis lot for validation and mutation tests.

    :param quantity: Exact token quantity as a decimal string.
    :param basis: Exact USD acquisition cost as a decimal string.
    :return: An open known-basis acquisition lot.
    """
    return Lot.open(
        AssetAmount(asset_id=ETH, quantity=Decimal(quantity)),
        cost_basis=UsdAmount(Decimal(basis)),
        acquired_at=ACQUIRED_AT,
    )


def test_open_known_lot_preserves_full_original_inventory() -> None:
    """Store matching original and remaining inventory for a new acquisition."""
    lot = open_known_lot()

    assert lot.asset_id == ETH
    assert lot.original_quantity == Decimal("10")
    assert lot.remaining_quantity == Decimal("10")
    assert lot.original_cost_basis == UsdAmount(Decimal("100"))
    assert lot.remaining_cost_basis == UsdAmount(Decimal("100"))
    assert lot.is_basis_known is True


def test_open_unknown_lot_does_not_invent_zero_cost_basis() -> None:
    """Preserve unknown acquisition cost using None, not zero dollars."""
    lot = Lot.open(AssetAmount(ETH, Decimal("2.25")), cost_basis=None, acquired_at=ACQUIRED_AT)

    assert lot.remaining_quantity == Decimal("2.25")
    assert lot.original_cost_basis is None
    assert lot.remaining_cost_basis is None
    assert lot.is_basis_known is False


def test_explicit_zero_cost_basis_is_known() -> None:
    """Keep an explicitly known zero distinct from a missing cost basis."""
    lot = open_known_lot(basis="0")

    assert lot.is_basis_known is True
    assert lot.original_cost_basis == UsdAmount(Decimal("0"))
    assert lot.remaining_cost_basis == UsdAmount(Decimal("0"))


@pytest.mark.parametrize("quantity", ["0", "-0.1"])
def test_open_rejects_nonpositive_acquired_quantity(quantity: str) -> None:
    """Reject acquisitions with no received inventory or negative inventory.

    :param quantity: Invalid acquisition quantity as a decimal string.
    """
    with pytest.raises(ValueError, match="acquired quantity must be positive"):
        Lot.open(AssetAmount(ETH, Decimal(quantity)), cost_basis=None, acquired_at=ACQUIRED_AT)


@pytest.mark.parametrize("candidate", [None, 10, "asset", True])
def test_open_rejects_non_asset_amount(candidate: object) -> None:
    """Validate the runtime boundary of the acquisition factory.

    :param candidate: Value that is intentionally not an AssetAmount.
    """
    with pytest.raises(TypeError, match="must be an AssetAmount"):
        Lot.open(cast(AssetAmount, candidate), cost_basis=None, acquired_at=ACQUIRED_AT)


@pytest.mark.parametrize("field_name", ["original_quantity", "remaining_quantity"])
@pytest.mark.parametrize("candidate", [1, 1.25, "1", None, True])
def test_quantities_require_decimal(field_name: str, candidate: object) -> None:
    """Reject non-Decimal quantities even when static typing is bypassed.

    :param field_name: Quantity field being validated.
    :param candidate: Invalid runtime quantity.
    """
    with pytest.raises(TypeError, match="must be a Decimal"):
        replace(open_known_lot(), **{field_name: cast(Decimal, candidate)})


@pytest.mark.parametrize("field_name", ["original_quantity", "remaining_quantity"])
@pytest.mark.parametrize("candidate", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_quantities_must_be_finite(field_name: str, candidate: str) -> None:
    """Prevent non-finite quantities from entering accounting state.

    :param field_name: Quantity field being validated.
    :param candidate: Non-finite Decimal string.
    """
    with pytest.raises(ValueError, match="must be finite"):
        replace(open_known_lot(), **{field_name: Decimal(candidate)})


def test_remaining_quantity_cannot_be_negative() -> None:
    """Prevent a lot from holding negative unconsumed inventory."""
    with pytest.raises(ValueError, match="remaining_quantity must not be negative"):
        replace(open_known_lot(), remaining_quantity=Decimal("-1"))


def test_original_quantity_must_be_positive() -> None:
    """Ensure the initial acquisition quantity is strictly positive."""
    with pytest.raises(ValueError, match="original_quantity must be positive"):
        replace(open_known_lot(), original_quantity=Decimal("0"))


def test_remaining_quantity_cannot_exceed_original_quantity() -> None:
    """Prevent creation of inventory not present in the acquisition."""
    with pytest.raises(ValueError, match="remaining_quantity cannot exceed original_quantity"):
        replace(open_known_lot(), remaining_quantity=Decimal("11"))


@pytest.mark.parametrize("field_name", ["original_cost_basis", "remaining_cost_basis"])
@pytest.mark.parametrize("candidate", [1, 1.5, "1", True])
def test_basis_rejects_invalid_runtime_types(field_name: str, candidate: object) -> None:
    """Reject malformed USD basis values at the domain boundary.

    :param field_name: Cost-basis field being validated.
    :param candidate: Runtime value under test.
    """
    with pytest.raises(TypeError, match="must be a UsdAmount or None"):
        replace(open_known_lot(), **{field_name: cast(UsdAmount, candidate)})


@pytest.mark.parametrize("field_name", ["original_cost_basis", "remaining_cost_basis"])
def test_negative_cost_basis_is_rejected(field_name: str) -> None:
    """Reject negative USD acquisition costs even though USD PnL may be negative.

    :param field_name: Cost-basis field being validated.
    """
    with pytest.raises(ValueError, match="must not be negative"):
        replace(open_known_lot(), **{field_name: UsdAmount(Decimal("-1"))})


@pytest.mark.parametrize("field_name", ["original_cost_basis", "remaining_cost_basis"])
def test_mixed_known_and_unknown_basis_is_rejected(field_name: str) -> None:
    """Require both original and remaining basis fields to share knownness.

    :param field_name: Basis field intentionally made unknown.
    """
    with pytest.raises(ValueError, match="cost basis must be both known or both unknown"):
        replace(open_known_lot(), **{field_name: None})


def test_remaining_basis_cannot_exceed_original_basis() -> None:
    """Reject creating basis that did not exist at acquisition."""
    with pytest.raises(ValueError, match="remaining_cost_basis cannot exceed original_cost_basis"):
        replace(open_known_lot(), remaining_cost_basis=UsdAmount(Decimal("101")))


def test_exhausted_known_lot_has_no_remaining_usd_basis() -> None:
    """Require all known USD basis to be consumed with remaining inventory."""
    with pytest.raises(ValueError, match="exhausted lot must have zero remaining_cost_basis"):
        replace(open_known_lot(), remaining_quantity=Decimal("0"))


def test_exhausted_known_lot_can_have_zero_basis() -> None:
    """Allow a fully consumed lot whose remaining known cost is exactly zero."""
    lot = replace(
        open_known_lot(),
        remaining_quantity=Decimal("0"),
        remaining_cost_basis=UsdAmount(Decimal("0")),
    )

    assert lot.remaining_quantity == Decimal("0")
    assert lot.is_basis_known is True


def test_exhausted_unknown_lot_remains_unknown() -> None:
    """Do not fabricate basis when an unknown lot has been fully consumed."""
    lot = Lot.open(AssetAmount(ETH, Decimal("3")), cost_basis=None, acquired_at=ACQUIRED_AT)
    exhausted = replace(lot, remaining_quantity=Decimal("0"))

    assert exhausted.original_cost_basis is None
    assert exhausted.remaining_cost_basis is None
    assert exhausted.is_basis_known is False


def test_partially_consumed_lot_can_have_reduced_inventory_and_basis() -> None:
    """Represent an intermediate lot state without performing FIFO consumption."""
    lot = replace(
        open_known_lot(),
        remaining_quantity=Decimal("4"),
        remaining_cost_basis=UsdAmount(Decimal("40")),
    )

    assert lot.remaining_quantity == Decimal("4")
    assert lot.remaining_cost_basis == UsdAmount(Decimal("40"))
    assert lot.original_quantity == Decimal("10")


def test_timestamp_is_normalized_to_utc() -> None:
    """Normalize a timezone-aware acquisition time without changing its instant."""
    local_time = datetime(2026, 1, 2, 6, 4, tzinfo=timezone(timedelta(hours=3)))
    lot = Lot.open(AssetAmount(ETH, Decimal("1")), cost_basis=None, acquired_at=local_time)

    assert lot.acquired_at == ACQUIRED_AT
    assert lot.acquired_at.tzinfo is UTC


@pytest.mark.parametrize("candidate", ["2026-01-02", None, 123])
def test_acquired_at_must_be_datetime(candidate: object) -> None:
    """Require acquisition timestamps to be datetime values.

    :param candidate: Invalid timestamp input.
    """
    with pytest.raises(TypeError, match="must be a datetime"):
        replace(open_known_lot(), acquired_at=cast(datetime, candidate))


def test_acquired_at_must_be_timezone_aware() -> None:
    """Reject naive timestamps that could lead to unstable FIFO ordering."""
    with pytest.raises(ValueError, match="must be timezone-aware"):
        replace(open_known_lot(), acquired_at=datetime(2026, 1, 2, 3, 4))


def test_asset_id_must_be_asset_identity() -> None:
    """Prevent token symbols or other values from becoming acquisition identity."""
    with pytest.raises(TypeError, match="Lot.asset_id must be an AssetId"):
        replace(open_known_lot(), asset_id=cast(AssetId, "ETH"))


def test_open_retains_precision_under_low_decimal_context() -> None:
    """Preserve all supplied digits regardless of the active arithmetic context."""
    amount = AssetAmount(ETH, Decimal("0.000000000000000001"))
    basis = UsdAmount(Decimal("12345678901234567890.123456789"))

    with localcontext() as context:
        context.prec = 6
        lot = Lot.open(amount, cost_basis=basis, acquired_at=ACQUIRED_AT)

    assert lot.original_quantity == amount.quantity
    assert lot.remaining_quantity == amount.quantity
    assert lot.original_cost_basis == basis
    assert lot.remaining_cost_basis == basis


def test_lot_is_immutable() -> None:
    """Prevent later code from mutating an acquired lot's original facts."""
    lot = open_known_lot()

    with pytest.raises(FrozenInstanceError):
        lot.original_quantity = Decimal("100")

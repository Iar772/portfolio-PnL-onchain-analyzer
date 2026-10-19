"""Deterministic contract tests for USD domain value objects."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from typing import cast

import pytest

from portfolio_analyzer.domain.money import UsdAmount, UsdPrice


def test_usd_amount_keeps_decimal_value_without_rounding() -> None:
    """Verify exact Decimal storage without quantization."""
    value = Decimal("12345678901234567890.123456789012345678")

    amount = UsdAmount(value)

    assert amount.value == value
    assert str(amount.value) == "12345678901234567890.123456789012345678"


def test_usd_amount_can_be_negative_for_pnl() -> None:
    """Verify that a signed amount can represent a loss."""
    amount = UsdAmount(Decimal("-15.25"))

    assert amount.value == Decimal("-15.25")


def test_usd_price_preserves_small_value() -> None:
    """Verify that very small known prices retain their precision."""
    price = UsdPrice(Decimal("0.000000000000000001"))

    assert price.value == Decimal("0.000000000000000001")


def test_usd_price_rejects_negative_value() -> None:
    """Verify that negative prices are invalid."""
    with pytest.raises(ValueError, match="must not be negative"):
        UsdPrice(Decimal("-0.01"))


def test_usd_price_allows_explicit_known_zero() -> None:
    """Verify that an explicitly known zero price is accepted."""
    assert UsdPrice(Decimal("0")).value == Decimal("0")


@pytest.mark.parametrize(
    "value",
    [Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"), Decimal("-Infinity")],
)
def test_non_finite_usd_amount_is_rejected(value: Decimal) -> None:
    """Verify that invalid non-finite amount values are rejected."""
    with pytest.raises(ValueError, match="finite"):
        UsdAmount(value)


@pytest.mark.parametrize(
    "value",
    [Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"), Decimal("-Infinity")],
)
def test_non_finite_usd_price_is_rejected(value: Decimal) -> None:
    """Verify that invalid non-finite price values are rejected."""
    with pytest.raises(ValueError, match="finite"):
        UsdPrice(value)


@pytest.mark.parametrize("value", [1, 1.2, "1.2", None, True])
def test_usd_amount_rejects_non_decimal_inputs(value: object) -> None:
    """Verify runtime validation even for statically mistyped inputs."""
    # cast intentionally bypasses mypy to test the runtime validation boundary.
    with pytest.raises(TypeError, match="must be a Decimal"):
        UsdAmount(cast(Decimal, value))


def test_usd_amount_is_immutable() -> None:
    """Verify that monetary value objects cannot be mutated."""
    amount = UsdAmount(Decimal("1"))

    with pytest.raises(FrozenInstanceError):
        amount.value = Decimal("2")


def test_usd_amount_keeps_small_negative_fraction() -> None:
    """Verify that very small negative amounts retain their original sign."""
    amount = UsdAmount(Decimal("-0.000000000000000001"))

    assert amount.value == Decimal("-0.000000000000000001")
    assert amount.value.is_signed()
    assert amount.value.as_tuple() == Decimal("-0.000000000000000001").as_tuple()


@pytest.mark.parametrize(
    "value",
    [5, 5.5, None, True, "5"],
)
def test_usd_price_rejects_types_except_decimal(value: object) -> None:
    """Verify that non-Decimal price inputs raise TypeError."""
    with pytest.raises(TypeError, match="must be a Decimal"):
        UsdPrice(value)

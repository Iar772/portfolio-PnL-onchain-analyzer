"""Offline contract tests for Ethereum-first asset value objects."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from typing import cast

import pytest

from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata


@pytest.fixture
def native_eth() -> AssetId:
    """Provide the canonical Ethereum Mainnet native-asset identity.

    :return: AssetId with chain 1 and no ERC-20 contract address.
    """
    return AssetId(chain_id=1, contract_address=None)


def test_native_eth_identity(native_eth: AssetId) -> None:
    """Verify that a missing contract address denotes the native asset.

    :param native_eth: Fixture holding Ethereum's native identity.
    """
    assert native_eth.is_native
    assert native_eth.chain_id == 1
    assert native_eth.contract_address is None


def test_contract_address_is_canonicalized_to_lowercase() -> None:
    """Verify that ERC-20 contract casing does not change identity."""
    mixed_case = AssetId(1, "0x" + "Ab" * 20)
    lowercase = AssetId(1, "0x" + "ab" * 20)

    assert mixed_case == lowercase
    assert hash(mixed_case) == hash(lowercase)
    assert mixed_case.contract_address == "0x" + "ab" * 20
    assert not mixed_case.is_native


def test_same_symbol_does_not_define_asset_identity() -> None:
    """Verify that distinct contracts remain distinct despite identical tickers."""
    first_contract = AssetId(1, "0x" + "11" * 20)
    second_contract = AssetId(1, "0x" + "22" * 20)
    metadata = AssetMetadata(symbol="USDC", name="Untrusted label", decimals=6)

    assert metadata.symbol == "USDC"
    assert first_contract != second_contract


@pytest.mark.parametrize("bad_chain_id", [0, -1])
def test_chain_id_must_be_positive(bad_chain_id: int) -> None:
    """Reject non-positive chain identifiers.

    :param bad_chain_id: Invalid chain ID supplied by the test.
    """
    with pytest.raises(ValueError, match="must be positive"):
        AssetId(bad_chain_id, None)


@pytest.mark.parametrize("bad_chain_id", [True, "1", 1.0, None])
def test_chain_id_must_be_a_real_integer(bad_chain_id: object) -> None:
    """Reject bool and other values that are not integer chain IDs.

    :param bad_chain_id: Invalid runtime value to validate.
    """
    with pytest.raises(TypeError, match="must be an integer"):
        AssetId(cast(int, bad_chain_id), None)


@pytest.mark.parametrize(
    "bad_address",
    ["0x" + "a" * 39, "0x" + "a" * 41, "0x" + "g" * 40, "a" * 40, ""],
)
def test_invalid_erc20_address_is_rejected(bad_address: str) -> None:
    """Reject malformed ERC-20 contract addresses.

    :param bad_address: An invalid address representation.
    """
    with pytest.raises(ValueError, match="20-byte"):
        AssetId(1, bad_address)


def test_contract_address_must_be_a_string_or_none() -> None:
    """Reject numeric contract addresses at the runtime boundary."""
    with pytest.raises(TypeError, match="string or None"):
        AssetId(1, cast(str, 123))


@pytest.mark.parametrize("field_name", ["symbol", "name"])
def test_metadata_strings_must_be_nonblank(field_name: str) -> None:
    """Reject blank ERC-20 display metadata.

    :param field_name: Metadata field to invalidate.
    """
    symbol = "   " if field_name == "symbol" else "TKN"
    name = "   " if field_name == "name" else "Example token"
    with pytest.raises(ValueError, match="must not be blank"):
        AssetMetadata(symbol=symbol, name=name, decimals=18)


@pytest.mark.parametrize("decimals", [-1, 256])
def test_metadata_decimal_range(decimals: int) -> None:
    """Enforce the 0..255 decimal range exposed by ERC-20 metadata.

    :param decimals: Out-of-range decimal precision.
    """
    with pytest.raises(ValueError, match="between 0 and 255"):
        AssetMetadata("TKN", "Example token", decimals)


@pytest.mark.parametrize("decimals", [True, 18.0, "18", None])
def test_metadata_decimals_must_be_integer(decimals: object) -> None:
    """Reject runtime values that are not integer decimal precision.

    :param decimals: Incorrect precision type to validate.
    """
    with pytest.raises(TypeError, match="must be an integer"):
        AssetMetadata("TKN", "Example token", cast(int, decimals))


def test_asset_amount_can_be_negative(native_eth: AssetId) -> None:
    """Allow negative wallet deltas for transfers out.

    :param native_eth: Canonical ETH identity.
    """
    amount = AssetAmount(native_eth, Decimal("-0.125"))
    assert amount.quantity == Decimal("-0.125")


def test_asset_amount_preserves_extreme_precision(native_eth: AssetId) -> None:
    """Keep precise token quantities unchanged in the domain.

    :param native_eth: Canonical ETH identity.
    """
    quantity = Decimal("123456789012345678901234567890.123456789012345678")
    assert AssetAmount(native_eth, quantity).quantity == quantity


@pytest.mark.parametrize("quantity", [Decimal("NaN"), Decimal("Infinity")])
def test_asset_amount_rejects_nonfinite_quantity(native_eth: AssetId, quantity: Decimal) -> None:
    """Reject invalid Decimal token amounts.

    :param native_eth: Canonical ETH identity.
    :param quantity: Nonfinite candidate token quantity.
    """
    with pytest.raises(ValueError, match="must be finite"):
        AssetAmount(native_eth, quantity)


def test_asset_amount_rejects_float(native_eth: AssetId) -> None:
    """Ensure token quantities never pass through float conversion.

    :param native_eth: Canonical ETH identity.
    """
    with pytest.raises(TypeError, match="must be a Decimal"):
        AssetAmount(native_eth, cast(Decimal, 0.1))


def test_asset_identity_and_amount_are_immutable(native_eth: AssetId) -> None:
    """Verify runtime immutability of frozen domain value objects.

    :param native_eth: Canonical ETH identity.
    """
    amount = AssetAmount(native_eth, Decimal("1"))
    with pytest.raises(FrozenInstanceError):
        amount.quantity = Decimal("2")


@pytest.mark.parametrize(
    ("raw_units", "decimals", "expected"),
    [
        (1, 18, Decimal("0.000000000000000001")),
        (1_234_567, 6, Decimal("1.234567")),
        (-25, 1, Decimal("-2.5")),
        (10**40 + 1, 18, Decimal("10000000000000000000000.000000000000000001")),
    ],
)
def test_from_raw_units_is_exact(
    native_eth: AssetId, raw_units: int, decimals: int, expected: Decimal
) -> None:
    """Require exact base-unit conversion without Decimal-context rounding.

    :param native_eth: Canonical asset identity.
    :param raw_units: Signed raw token quantity.
    :param decimals: Token metadata precision.
    :param expected: Exact human-readable Decimal quantity.
    """
    metadata = AssetMetadata("TKN", "Example", decimals)
    amount = AssetAmount.from_raw_units(native_eth, raw_units, metadata)
    assert amount.quantity == expected
    assert amount.asset_id == native_eth


@pytest.mark.parametrize("raw_units", [True, 1.0, "100"])
def test_from_raw_units_rejects_noninteger_inputs(native_eth: AssetId, raw_units: object) -> None:
    """Reject non-integer raw token units at the runtime boundary.

    :param native_eth: Canonical Ethereum native asset identity.
    :param raw_units: Invalid token quantity to reject.
    """
    metadata = AssetMetadata("ETH", "Ether", 18)

    with pytest.raises(TypeError, match="raw_units must be an integer"):
        AssetAmount.from_raw_units(native_eth, cast(int, raw_units), metadata)

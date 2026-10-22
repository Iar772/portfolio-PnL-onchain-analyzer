"""Unit tests for chain-specific, immutable USD stablecoin allowlists."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.stablecoins import StablecoinRegistry, UsdStablecoin

USDC = AssetId(1, "0x" + "ab" * 20)
USDT = AssetId(1, "0x" + "cd" * 20)
SAME_TICKER = AssetId(1, "0x" + "ef" * 20)


def test_lookup_is_by_identity_not_symbol() -> None:
    """Two tokens sharing a display symbol are separate assets."""
    registry = StablecoinRegistry([UsdStablecoin(USDC, "USDC")])
    assert registry.is_trusted(USDC)
    assert registry.find(SAME_TICKER) is None
    assert not registry.is_trusted(SAME_TICKER)


def test_lookup_is_chain_qualified() -> None:
    """The same EVM address on another chain is not automatically trusted."""
    registry = StablecoinRegistry([UsdStablecoin(USDC, "USDC")])
    assert registry.find(AssetId(8453, USDC.contract_address)) is None
    assert registry.find(AssetId(1, USDC.contract_address.upper().replace("0X", "0x"))) is not None


def test_peg_target_is_typed_and_not_an_observed_quote() -> None:
    """A declared USD peg target is not an observed timestamped price."""
    entry = UsdStablecoin(USDC, "USDC")
    assert entry.peg_target_usd == Decimal("1")
    assert not hasattr(entry.peg_target_usd, "observed_at")
    assert not hasattr(entry.peg_target_usd, "source")


def test_symbols_are_display_only_and_whitespace_is_removed() -> None:
    """Symbol normalization must not influence registry identity."""
    entry = UsdStablecoin(USDC, "  FAKE-TICKER  ")
    registry = StablecoinRegistry([entry])
    assert registry.find(USDC) == entry
    found = registry.find(USDC)
    assert found is not None
    assert found.symbol == "FAKE-TICKER"


def test_custom_registry_can_add_another_evm_network() -> None:
    """Other EVM chains may be added without changing the lookup contract."""
    base_usdc = UsdStablecoin(USDC, "USDC")
    another_network = UsdStablecoin(AssetId(8453, "0x" + "12" * 20), "USDC")
    registry = StablecoinRegistry([base_usdc, another_network])
    assert registry.entries() == (base_usdc, another_network)
    assert registry.find(another_network.asset_id) == another_network


def test_registry_input_defensively_copied() -> None:
    """Mutating the input list cannot mutate an existing trusted registry."""
    original = [UsdStablecoin(USDC, "USDC")]
    registry = StablecoinRegistry(original)
    original.append(UsdStablecoin(USDT, "USDT"))
    original.clear()
    assert registry.entries() == (UsdStablecoin(USDC, "USDC"),)


def test_entry_is_immutable() -> None:
    """A definition cannot silently change its identity or label."""
    entry = UsdStablecoin(USDC, "USDC")
    with pytest.raises(FrozenInstanceError):
        entry.symbol = "OTHER"  # type: ignore[misc]


def test_repeated_id_even_with_distinct_symbols_is_rejected() -> None:
    """No registry entry is allowed to replace a verified contract."""
    with pytest.raises(ValueError, match="duplicate"):
        StablecoinRegistry([UsdStablecoin(USDC, "USDC"), UsdStablecoin(USDC, "USD-C")])


def test_repeated_symbol_at_different_addresses_is_allowed() -> None:
    """Display labels never determine whether a token is trusted."""
    registry = StablecoinRegistry([UsdStablecoin(USDC, "USDC"), UsdStablecoin(USDT, "USDC")])
    assert len(registry.entries()) == 2


@pytest.mark.parametrize("invalid", [None, object(), "USDC", AssetId(1, None)])
def test_invalid_asset_id_is_not_accepted_as_definition(invalid: object) -> None:
    """A registry item must refer to a real token contract."""
    with pytest.raises((TypeError, ValueError)):
        UsdStablecoin(invalid, "USDC")  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [None, 7, "", "  "])
def test_invalid_symbol_is_rejected(invalid: object) -> None:
    """Missing or non-text display labels cannot be registered."""
    with pytest.raises((TypeError, ValueError)):
        UsdStablecoin(USDC, invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [None, "USDC", AssetId(1, None)])
def test_lookup_rejects_wrong_model_types(invalid: object) -> None:
    """Do not accidentally look up a token using a ticker or null value."""
    registry = StablecoinRegistry([UsdStablecoin(USDC, "USDC")])
    if isinstance(invalid, AssetId):
        assert registry.find(invalid) is None
    else:
        with pytest.raises(TypeError):
            registry.find(invalid)  # type: ignore[arg-type]


def test_non_definition_entry_is_rejected() -> None:
    """The registry cannot silently admit raw strings or provider DTOs."""
    with pytest.raises(TypeError):
        StablecoinRegistry(["USDC"])  # type: ignore[list-item]


def test_empty_registry_is_valid_and_conservative() -> None:
    """An empty allowlist trusts no asset by default."""
    registry = StablecoinRegistry([])
    assert registry.entries() == ()
    assert registry.find(USDC) is None
    assert not registry.is_trusted(USDC)


def test_registry_itself_is_frozen() -> None:
    """An allowlist must not be replaced by reassigning its internal mapping."""
    registry = StablecoinRegistry([UsdStablecoin(USDC, "USDC")])
    with pytest.raises(FrozenInstanceError):
        registry._by_asset_id = {}  # type: ignore[misc]

"""Ethereum Mainnet token address allowlist and valuation boundary tests."""

from decimal import Decimal

from portfolio_analyzer.application.valuation import value_current_balances
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.ethereum.stablecoins import (
    MAINNET_USD_STABLECOINS,
    MAINNET_USDC,
    MAINNET_USDT,
)


def test_mainnet_registry_contains_exactly_usdc_and_usdt() -> None:
    """Initial registry is minimal and uses verified native issuer contracts."""
    assert {x.asset_id for x in MAINNET_USD_STABLECOINS.entries()} == {
        MAINNET_USDC,
        MAINNET_USDT,
    }
    assert MAINNET_USDC == AssetId(1, "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")
    assert MAINNET_USDT == AssetId(1, "0xdac17f958d2ee523a2206206994597c13d831ec7")


def test_mainnet_allows_different_case_same_contract() -> None:
    """Case variants of the same ERC-20 identity match one allowlist record."""
    assert MAINNET_USD_STABLECOINS.is_trusted(
        AssetId(1, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    )
    assert MAINNET_USD_STABLECOINS.is_trusted(
        AssetId(1, "0xdAC17F958D2ee523a2206206994597C13D831ec7")
    )


def test_spoofed_usdt_or_usdc_is_not_allowlisted() -> None:
    """Matching token symbol never qualifies an unknown contract as trusted."""
    spoofed = AssetId(1, "0x" + "11" * 20)
    assert not MAINNET_USD_STABLECOINS.is_trusted(spoofed)
    assert MAINNET_USD_STABLECOINS.find(spoofed) is None


def test_same_contract_on_other_chain_is_not_allowlisted() -> None:
    """Ethereum issuer contract must not leak trust into other EVM chains."""
    assert not MAINNET_USD_STABLECOINS.is_trusted(AssetId(137, MAINNET_USDC.contract_address))
    assert not MAINNET_USD_STABLECOINS.is_trusted(AssetId(8453, MAINNET_USDT.contract_address))


def test_native_eth_is_not_stablecoin() -> None:
    """Ethereum native ETH is not a pegged ERC-20 token."""
    assert MAINNET_USD_STABLECOINS.find(AssetId(1, None)) is None


def test_peg_target_can_be_used_as_reference_for_later_valuation() -> None:
    """Trusted records expose a USD reference but no fabricated market quote."""
    for stablecoin in MAINNET_USD_STABLECOINS.entries():
        assert stablecoin.peg_target_usd == Decimal("1")


def test_missing_quote_stays_unpriced_even_for_allowlisted_usdc() -> None:
    """Allowlist membership must not silently inject a 1 USD current quote."""
    assert MAINNET_USD_STABLECOINS.is_trusted(MAINNET_USDC)
    result = value_current_balances([AssetAmount(MAINNET_USDC, Decimal("100"))], {})
    assert not result.is_complete
    assert result.known_total_usd.value == Decimal(0)
    assert tuple(item.balance.asset_id for item in result.unpriced) == (MAINNET_USDC,)

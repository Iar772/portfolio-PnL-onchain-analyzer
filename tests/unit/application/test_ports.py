"""Offline tests of provider-neutral contracts and replacement fakes."""

import ast
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from portfolio_analyzer.application.ports import (
    BalanceProvider,
    HistoryProvider,
    PriceProvider,
    RpcProvider,
    TokenMetadataProvider,
)
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata
from portfolio_analyzer.domain.money import UsdPrice
from portfolio_analyzer.ethereum.models import EthereumTransaction

_ETH = AssetId(chain_id=1, contract_address=None)
_TOKEN = AssetId(chain_id=1, contract_address="0x" + "11" * 20)
_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class FakeRpc:
    """Provide chain data without implementing a concrete RPC adapter."""

    async def get_chain_id(self) -> int:
        """Return the fake chain ID.

        :return: Ethereum Mainnet chain ID.
        """
        return 1

    async def get_block_number(self) -> int:
        """Return a stable observed block number.

        :return: A deterministic test block height.
        """
        return 100


class FakeHistory:
    """Return provider-neutral transactions from memory."""

    async def get_history(self, address: str) -> list[EthereumTransaction]:
        """Return a deterministic transaction history.

        :param address: Wallet address ignored by this single-wallet fake.
        :return: One internal transaction envelope.
        """
        return [EthereumTransaction("0x" + "ab" * 32, 99, _AT)]


class FakeBalances:
    """Return a static native balance for tests."""

    async def get_balances(self, address: str) -> list[AssetAmount]:
        """Return one current on-chain quantity without API access.

        :param address: Wallet address ignored by this fake.
        :return: Fixed ETH balance.
        """
        return [AssetAmount(_ETH, Decimal("2"))]


class FakeMetadata:
    """Resolve ETH metadata from a built-in mapping."""

    async def get_metadata(self, asset_id: AssetId) -> AssetMetadata | None:
        """Return only metadata known to the fake.

        :param asset_id: Requested canonical asset ID.
        :return: ETH metadata or None for an unknown token.
        """
        return AssetMetadata("ETH", "Ether", 18) if asset_id == _ETH else None


class FakePrices:
    """Resolve known ETH prices without a vendor or network."""

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Return current ETH price when present.

        :param asset_id: Requested asset.
        :return: Known ETH price or None.
        """
        return PriceQuote(UsdPrice(Decimal("2000")), _AT, "fake") if asset_id == _ETH else None

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Return a known historical observation without inventing a timestamp.

        :param asset_id: Requested asset.
        :param at: Target transaction timestamp.
        :return: An observation on or before target, or None if unknown.
        """
        if asset_id == _ETH and at >= _AT:
            return PriceQuote(UsdPrice(Decimal("1800")), _AT, "fake")
        return None


async def _consume_fake_providers(
    rpc: RpcProvider,
    history: HistoryProvider,
    balances: BalanceProvider,
    metadata: TokenMetadataProvider,
    prices: PriceProvider,
) -> tuple[int, Decimal, UsdPrice]:
    """Use abstract providers without referencing a concrete integration.

    :param rpc: Chain status provider.
    :param history: Transaction history provider.
    :param balances: Current balance provider.
    :param metadata: Token display metadata provider.
    :param prices: Historical and current USD price provider.
    :return: Transaction count, ETH balance, and known historical USD price.
    """
    assert await rpc.get_chain_id() == 1
    assert await rpc.get_block_number() >= 0
    transactions = await history.get_history("0x" + "22" * 20)
    holdings = await balances.get_balances("0x" + "22" * 20)
    assert await metadata.get_metadata(_ETH) == AssetMetadata("ETH", "Ether", 18)
    assert await prices.get_current_price(_TOKEN) is None
    historical = await prices.get_historical_price(_ETH, _AT)
    assert historical is not None
    return len(transactions), holdings[0].quantity, historical.price_usd


@pytest.mark.asyncio
async def test_all_five_ports_accept_pure_python_fake_providers() -> None:
    """Exercise the same consumer through fakes implementing the five Protocols."""
    result = await _consume_fake_providers(
        FakeRpc(), FakeHistory(), FakeBalances(), FakeMetadata(), FakePrices()
    )
    assert result == (1, Decimal("2"), UsdPrice(Decimal("1800")))


@pytest.mark.asyncio
async def test_missing_metadata_and_prices_remain_unknown() -> None:
    """Return None for missing facts rather than manufacturing zero values."""
    metadata: TokenMetadataProvider = FakeMetadata()
    prices: PriceProvider = FakePrices()

    assert await metadata.get_metadata(_TOKEN) is None
    assert await prices.get_current_price(_TOKEN) is None
    assert await prices.get_historical_price(_ETH, _AT - timedelta(days=1)) is None


def test_price_quote_preserves_known_zero_price() -> None:
    """Keep an explicitly known zero price distinct from an unknown quote."""
    quote = PriceQuote(UsdPrice(Decimal("0")), _AT, "fake")
    assert quote.price_usd.value == Decimal("0")


def test_price_quote_utc_normalization_and_source_trimming() -> None:
    """Normalize aware observations into UTC and trim provenance identifiers."""
    eastern = timezone(timedelta(hours=3))
    quote = PriceQuote(UsdPrice(Decimal("42")), _AT.astimezone(eastern), "  fake  ")
    assert quote.observed_at == _AT
    assert quote.observed_at.tzinfo is UTC
    assert quote.source == "fake"


@pytest.mark.parametrize("bad_value", [None, 12, "2026-01-01", True])
def test_price_quote_rejects_invalid_timestamp_types(bad_value: object) -> None:
    """Reject observation timestamps that are not datetime instances.

    :param bad_value: Invalid runtime timestamp value.
    """
    with pytest.raises(TypeError, match="observed_at"):
        PriceQuote(UsdPrice(Decimal("1")), cast(datetime, bad_value), "fake")


def test_price_quote_rejects_naive_timestamp() -> None:
    """Reject timestamps lacking a timezone rather than assuming UTC."""
    with pytest.raises(ValueError, match="timezone-aware"):
        PriceQuote(UsdPrice(Decimal("1")), datetime(2026, 1, 1), "fake")


@pytest.mark.parametrize("bad_value", [None, 1, True])
def test_price_quote_rejects_non_string_source(bad_value: object) -> None:
    """Reject non-string price sources.

    :param bad_value: Invalid runtime provenance value.
    """
    with pytest.raises(TypeError, match="source"):
        PriceQuote(UsdPrice(Decimal("1")), _AT, cast(str, bad_value))


@pytest.mark.parametrize("blank", ["", "  ", "\t"])
def test_price_quote_rejects_blank_source(blank: str) -> None:
    """Reject missing price-source provenance.

    :param blank: Whitespace-only or empty price-source identifier.
    """
    with pytest.raises(ValueError, match="source"):
        PriceQuote(UsdPrice(Decimal("1")), _AT, blank)


def test_price_quote_requires_usd_price_object() -> None:
    """Reject untyped Decimal values where a UsdPrice is required."""
    with pytest.raises(TypeError, match="price_usd"):
        PriceQuote(cast(UsdPrice, Decimal("42")), _AT, "fake")


def test_price_quote_is_immutable() -> None:
    """Keep price observations immutable once constructed."""
    quote = PriceQuote(UsdPrice(Decimal("42")), _AT, "fake")
    with pytest.raises(FrozenInstanceError):
        quote.source = "other"


def test_domain_and_application_do_not_import_external_provider_sdks() -> None:
    """Guard inward dependency direction with an offline source inspection."""
    source_root = Path(__file__).resolve().parents[3] / "src" / "portfolio_analyzer"
    prohibited = ("alchemy", "httpx", "web3", "requests", "aiohttp")
    for package in ("domain", "ethereum", "application"):
        for path in (source_root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
            for name in names:
                assert not any(part in prohibited for part in name.split(".")), (
                    f"Forbidden provider import in {path}: {name}"
                )

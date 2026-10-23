"""Offline tests for vendor-independent bounded historical quote caching."""

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from portfolio_analyzer.application.cached_prices import CachedPriceProvider
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.run_guardrails import AnalysisRunStopped, RunGuardrails
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.money import UsdPrice

_ETH = AssetId(1, None)
_TOKEN = AssetId(1, "0x" + "ab" * 20)
_AT = datetime(2024, 5, 3, 12, 0, tzinfo=UTC)
_QUOTE = PriceQuote(
    price_usd=UsdPrice(Decimal("1500")),
    observed_at=_AT,
    source="fake_prices",
)


class FakePrices:
    """Count network-equivalent delegated lookups while returning configured data."""

    def __init__(self, result: PriceQuote | None = _QUOTE) -> None:
        """Choose returned quote without any network traffic."""
        self.result = result
        self.historical_calls: list[tuple[AssetId, datetime]] = []
        self.current_calls = 0
        self.error: Exception | None = None
        self.block: asyncio.Event | None = None

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Return one current quote without caching it."""
        self.current_calls += 1
        return self.result

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Record historical calls and optionally block or fail."""
        self.historical_calls.append((asset_id, at))
        if self.block is not None:
            await self.block.wait()
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_cache_hit_uses_same_quote_and_counts_only_one_lookup() -> None:
    """The exact same historical timestamp does not cost two provider calls."""
    fake = FakePrices()
    guardrails = RunGuardrails(max_http_attempts=3)
    cache = CachedPriceProvider(fake, guardrails=guardrails)
    assert await cache.get_historical_price(_ETH, _AT) == _QUOTE
    assert await cache.get_historical_price(_ETH, _AT) == _QUOTE
    assert len(fake.historical_calls) == 1
    stats = guardrails.snapshot()
    assert stats.historical_cache_hits == 1
    assert stats.historical_cache_misses == 1
    assert stats.http_attempts == 0  # Only the transport reserves HTTP attempts.


@pytest.mark.asyncio
async def test_cache_normalizes_timezone_for_identity() -> None:
    """Equivalent instants in different timezones map to one historical key."""
    fake = FakePrices()
    cache = CachedPriceProvider(fake)
    assert await cache.get_historical_price(_TOKEN, _AT) == _QUOTE
    assert (
        await cache.get_historical_price(_TOKEN, _AT.astimezone(timezone(timedelta(hours=2))))
        == _QUOTE
    )
    assert len(fake.historical_calls) == 1


@pytest.mark.asyncio
async def test_cache_does_not_merge_distinct_assets_or_instants() -> None:
    """Same ticker and adjacent instants are never assumed to have equal quotes."""
    fake = FakePrices()
    cache = CachedPriceProvider(fake)
    await cache.get_historical_price(_ETH, _AT)
    await cache.get_historical_price(_TOKEN, _AT)
    await cache.get_historical_price(_ETH, _AT + timedelta(seconds=1))
    assert len(fake.historical_calls) == 3


@pytest.mark.asyncio
async def test_missing_price_is_cacheable_but_never_turned_into_zero() -> None:
    """Explicit missing historical observations may be reused as None."""
    fake = FakePrices(None)
    cache = CachedPriceProvider(fake)
    assert await cache.get_historical_price(_ETH, _AT) is None
    assert await cache.get_historical_price(_ETH, _AT) is None
    assert len(fake.historical_calls) == 1


@pytest.mark.asyncio
async def test_failures_are_not_cached() -> None:
    """Transient provider failures remain retryable on a later independent lookup."""
    fake = FakePrices()
    fake.error = RuntimeError("temporary fake failure")
    cache = CachedPriceProvider(fake)
    with pytest.raises(RuntimeError):
        await cache.get_historical_price(_ETH, _AT)
    fake.error = None
    assert await cache.get_historical_price(_ETH, _AT) == _QUOTE
    assert len(fake.historical_calls) == 2


@pytest.mark.asyncio
async def test_lru_capacity_eviction_is_bounded() -> None:
    """Oldest lookup is evicted, while most recently used lookups survive."""
    fake = FakePrices()
    cache = CachedPriceProvider(fake, max_entries=2)
    await cache.get_historical_price(_ETH, _AT)
    await cache.get_historical_price(_TOKEN, _AT)
    await cache.get_historical_price(_ETH, _AT)
    await cache.get_historical_price(_ETH, _AT + timedelta(seconds=1))
    assert cache.entry_count == 2
    await cache.get_historical_price(_TOKEN, _AT)
    assert len(fake.historical_calls) == 4


@pytest.mark.asyncio
async def test_current_prices_are_not_cached() -> None:
    """Current prices still obey their existing freshness policy."""
    fake = FakePrices()
    cache = CachedPriceProvider(fake)
    await cache.get_current_price(_ETH)
    await cache.get_current_price(_ETH)
    assert fake.current_calls == 2
    assert cache.entry_count == 0


@pytest.mark.asyncio
async def test_concurrent_identical_price_queries_singleflight() -> None:
    """Duplicate concurrent lookups share one provider call without spawning tasks."""
    fake = FakePrices()
    guardrails = RunGuardrails(max_http_attempts=10)
    cache = CachedPriceProvider(fake, guardrails=guardrails)
    results = await asyncio.gather(*(cache.get_historical_price(_ETH, _AT) for _ in range(10)))
    assert results == [_QUOTE] * 10
    assert len(fake.historical_calls) == 1
    assert guardrails.snapshot().historical_cache_hits == 9


@pytest.mark.asyncio
async def test_stop_rejects_cache_hits_and_prevents_new_queries() -> None:
    """A cancelled analysis cannot return cached data as though it completed."""
    fake = FakePrices()
    guardrails = RunGuardrails(max_http_attempts=2)
    cache = CachedPriceProvider(fake, guardrails=guardrails)
    await cache.get_historical_price(_ETH, _AT)
    guardrails.request_stop()
    with pytest.raises(AnalysisRunStopped) as exc:
        await cache.get_historical_price(_ETH, _AT)
    assert exc.value.snapshot.historical_cache_misses == 1
    assert len(fake.historical_calls) == 1


@pytest.mark.asyncio
async def test_cancelled_lookup_is_not_cached() -> None:
    """An externally cancelled lookup leaves the cache available for clean retry."""
    fake = FakePrices()
    block = asyncio.Event()
    fake.block = block
    cache = CachedPriceProvider(fake)
    task = asyncio.create_task(cache.get_historical_price(_ETH, _AT))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cache.entry_count == 0
    block.set()
    assert await cache.get_historical_price(_ETH, _AT) == _QUOTE
    assert len(fake.historical_calls) == 2


@pytest.mark.parametrize("max_entries", [0, -1, True, 1.5])
def test_invalid_capacity_is_rejected(max_entries: object) -> None:
    """The in-memory cache must have a finite positive configured size."""
    with pytest.raises(ValueError):
        CachedPriceProvider(FakePrices(), max_entries=max_entries)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_cache_rejects_naive_timestamps_and_invalid_assets() -> None:
    """Invalid inputs do not count as provider calls or cache misses."""
    fake = FakePrices()
    cache = CachedPriceProvider(fake)
    with pytest.raises(ValueError):
        await cache.get_historical_price(_ETH, _AT.replace(tzinfo=None))
    with pytest.raises(TypeError):
        await cache.get_historical_price("ETH", _AT)  # type: ignore[arg-type]
    assert fake.historical_calls == []

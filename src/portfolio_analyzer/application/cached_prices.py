"""Run-local, bounded historical price caching independent of price vendors."""

import asyncio
from collections import OrderedDict
from datetime import UTC, datetime

from portfolio_analyzer.application.ports import PriceProvider
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.run_guardrails import RunGuardrails
from portfolio_analyzer.domain.assets import AssetId


class CachedPriceProvider:
    """Wrap any PriceProvider with a bounded per-run historical quote cache.

    Only historical prices are cached: current quotes must remain fresh.
    Cache keys include asset identity and the exact UTC transaction timestamp.
    Missing historical prices (None) are cached too, but provider errors are not.
    The lock serializes historical misses and suppresses duplicate in-flight
    lookups; it never creates unbounded asyncio tasks.

    :param provider: Vendor-neutral PriceProvider to delegate to.
    :param max_entries: Maximum historical lookup results retained in memory.
    :param guardrails: Optional shared per-analysis usage metrics and stop state.
    :raises ValueError: If the cache capacity is not positive.
    """

    def __init__(
        self,
        provider: PriceProvider,
        *,
        max_entries: int = 512,
        guardrails: RunGuardrails | None = None,
    ) -> None:
        """Initialize a bounded LRU cache without network activity.

        :param provider: Provider implementing current and historical pricing.
        :param max_entries: Maximum distinct historical lookup keys.
        :param guardrails: Shared optional usage state for this analysis.
        """
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if guardrails is not None and not isinstance(guardrails, RunGuardrails):
            raise TypeError("guardrails must be RunGuardrails or None")
        self._provider = provider
        self._guardrails = guardrails
        self._max_entries = max_entries
        self._cache: OrderedDict[tuple[AssetId, datetime], PriceQuote | None] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Delegate current prices without caching potentially stale quotes.

        :param asset_id: Identity of the asset to quote.
        :return: Fresh quote, or None when unavailable.
        :raises AnalysisRunStopped: If the analysis was already stopped.
        """
        if self._guardrails is not None:
            self._guardrails.ensure_active()
        return await self._provider.get_current_price(asset_id)

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Reuse exact historical lookups while preserving missing price evidence.

        :param asset_id: Exact chain-and-contract asset identity.
        :param at: Timezone-aware transaction timestamp.
        :return: Cached or freshly requested quote, possibly None.
        :raises ValueError: For naive timestamps.
        :raises TypeError: For wrong input types.
        :raises AnalysisRunStopped: If the analysis was stopped.
        """
        if not isinstance(asset_id, AssetId):
            raise TypeError("asset_id must be AssetId")
        if not isinstance(at, datetime):
            raise TypeError("at must be datetime")
        if at.utcoffset() is None:
            raise ValueError("at must be timezone-aware")
        key = (asset_id, at.astimezone(UTC))
        if self._guardrails is not None:
            self._guardrails.ensure_active()
        async with self._lock:
            if self._guardrails is not None:
                self._guardrails.ensure_active()
            if key in self._cache:
                self._cache.move_to_end(key)
                if self._guardrails is not None:
                    self._guardrails.record_historical_cache_access(hit=True)
                return self._cache[key]
            if self._guardrails is not None:
                self._guardrails.record_historical_cache_access(hit=False)
            quote = await self._provider.get_historical_price(asset_id, key[1])
            if self._guardrails is not None:
                self._guardrails.ensure_active()
            self._cache[key] = quote
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
            return quote

    @property
    def entry_count(self) -> int:
        """Return the number of cached historical lookup keys.

        :return: Bounded number of price cache entries.
        """
        return len(self._cache)

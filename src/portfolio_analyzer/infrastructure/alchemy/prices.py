"""Current USD price adapter for Ethereum Mainnet using Alchemy Prices API."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

from portfolio_analyzer.application.provider_errors import ProviderMalformedResponseError
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.money import UsdPrice
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider, _mapping

_OPERATION = "get_current_price"
_LOGGER = logging.getLogger(__name__)


class AlchemyCurrentPriceProvider:
    """Fetch trusted Mainnet USD quotes without guessing unsupported prices.

    ERC-20 tokens are identified by chain+contract, never by ticker. Native ETH
    is the sole explicitly allowlisted symbol lookup. Stale/missing quotes are
    None; malformed provider data and transport failures remain explicit errors.

    :param rpc: Shared Alchemy transport with bounded retries and HTTP lifecycle.
    :param max_age: Maximum age of a quote permitted for current valuation.
    :param now: Injectable current UTC time for deterministic freshness checks.
    :raises TypeError: If dependencies have wrong types.
    :raises ValueError: If max_age is invalid.
    """

    def __init__(
        self,
        rpc: AlchemyRpcProvider,
        *,
        max_age: timedelta = timedelta(hours=1),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Use the existing RPC transport and validate price freshness policy.

        :param rpc: Shared Alchemy transport adapter.
        :param max_age: Positive price maximum age.
        :param now: Optional clock override, UTC when omitted.
        :raises TypeError: For invalid dependency types.
        :raises ValueError: For invalid max_age values.
        """
        if not isinstance(rpc, AlchemyRpcProvider):
            raise TypeError("rpc must be an AlchemyRpcProvider")
        if not isinstance(max_age, timedelta):
            raise TypeError("max_age must be timedelta")
        if not timedelta(0) < max_age <= timedelta(days=1):
            raise ValueError("max_age must be positive and at most one day")
        if now is not None and not callable(now):
            raise TypeError("now must be callable")
        self._rpc = rpc
        self._max_age = max_age
        self._now = now if now is not None else _utc_now

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Fetch one current USD price for a native ETH or ERC-20 identity.

        :param asset_id: Ethereum Mainnet asset, never a symbol.
        :return: PriceQuote or None if missing, unavailable or stale.
        :raises ValueError: If asset belongs to another chain.
        :raises ProviderMalformedResponseError: For corrupted response data.
        """
        if not isinstance(asset_id, AssetId):
            raise TypeError("asset_id must be an AssetId")
        if asset_id.chain_id != 1:
            raise ValueError("Alchemy Prices supports Ethereum Mainnet only")
        raw = await self._rpc.request_alchemy_prices(contract_address=asset_id.contract_address)
        data = _mapping(raw, operation=_OPERATION)
        entries = data.get("data")
        if not isinstance(entries, list):
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        if not entries:
            self._log_unpriced()
            return None
        if len(entries) != 1:
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        entry = _mapping(entries[0], operation=_OPERATION)
        if asset_id.is_native:
            if entry.get("symbol") != "ETH":
                raise ProviderMalformedResponseError("alchemy", _OPERATION)
        elif (
            entry.get("network") != "eth-mainnet"
            or not isinstance(entry.get("address"), str)
            or entry["address"].lower() != asset_id.contract_address
        ):
            raise ProviderMalformedResponseError("alchemy", _OPERATION)

        error = entry.get("error")
        if error is not None:
            if not isinstance(error, str):
                raise ProviderMalformedResponseError("alchemy", _OPERATION)
            self._log_unpriced()
            return None
        prices = entry.get("prices")
        if not isinstance(prices, list):
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        usd_rows: list[dict[str, object]] = []
        for row in prices:
            price_row = _mapping(row, operation=_OPERATION)
            currency = price_row.get("currency")
            if not isinstance(currency, str):
                raise ProviderMalformedResponseError("alchemy", _OPERATION)
            if currency.lower() == "usd":
                usd_rows.append(price_row)
        if not usd_rows:
            self._log_unpriced()
            return None
        if len(usd_rows) != 1:
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        row = usd_rows[0]
        value = row.get("value")
        timestamp = row.get("lastUpdatedAt")
        if not isinstance(value, str) or len(value) > 128:
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        try:
            price = UsdPrice(Decimal(value))
        except (InvalidOperation, ValueError, TypeError):
            raise ProviderMalformedResponseError("alchemy", _OPERATION) from None
        if abs(price.value.adjusted()) > 1000:
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        if not isinstance(timestamp, str) or len(timestamp) > 64:
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        try:
            observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            raise ProviderMalformedResponseError("alchemy", _OPERATION) from None
        if observed.utcoffset() is None:
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        observed = observed.astimezone(UTC)
        current = self._now()
        if not isinstance(current, datetime) or current.utcoffset() is None:
            raise ValueError("Price clock must return timezone-aware datetime")
        age = current.astimezone(UTC) - observed
        if age > self._max_age or age < -timedelta(minutes=5):
            self._log_unpriced()
            return None
        quote = PriceQuote(
            price_usd=price,
            observed_at=observed,
            source="alchemy_prices_by_symbol"
            if asset_id.is_native
            else "alchemy_prices_by_address",
        )
        _LOGGER.debug(
            "current_price_completed",
            extra={"provider": "alchemy", "operation": _OPERATION},
        )
        return quote

    @staticmethod
    def _log_unpriced() -> None:
        """Report a missing quote without exposing wallet or token identity."""
        _LOGGER.info(
            "current_price_unavailable", extra={"provider": "alchemy", "operation": _OPERATION}
        )


def _utc_now() -> datetime:
    """Return the current UTC time for production price freshness checks.

    :return: Current timezone-aware UTC datetime.
    """
    return datetime.now(UTC)

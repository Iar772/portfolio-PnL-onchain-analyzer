"""Historical USD price observations near Ethereum transaction timestamps."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

from portfolio_analyzer.application.provider_errors import (
    ProviderError,
    ProviderMalformedResponseError,
)
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.money import UsdPrice
from portfolio_analyzer.infrastructure.alchemy.prices import AlchemyCurrentPriceProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider, _mapping

_OPERATION = "get_historical_price"
_LOGGER = logging.getLogger(__name__)


class AlchemyPriceProvider(AlchemyCurrentPriceProvider):
    """Provide both current and timestamped historical USD observations.

    Both kinds of request use the same Alchemy transport/client, concurrency
    limit, bounded retry policy and vendor-neutral PriceQuote representation.
    A nearby price point is an observed market reference, not an execution price.

    :param rpc: Existing shared Alchemy RPC and REST transport.
    :param max_distance: Maximum acceptable time difference from a transaction.
    :param max_age: Maximum current-price age, forwarded to the current adapter.
    :param now: Optional clock for current-price freshness checking.
    :raises TypeError: If max_distance is not a timedelta.
    :raises ValueError: If max_distance is outside the supported lookup range.
    """

    def __init__(
        self,
        rpc: AlchemyRpcProvider,
        *,
        max_distance: timedelta = timedelta(minutes=15),
        max_age: timedelta = timedelta(hours=1),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Configure a bounded lookup window while reusing current pricing.

        :param rpc: Caller-owned shared HTTP transport.
        :param max_distance: Positive maximum distance to a price observation.
        :param max_age: Current-quote freshness policy.
        :param now: Optional deterministic current-quote clock.
        :raises TypeError: If max_distance is not a timedelta.
        :raises ValueError: If the requested tolerance is outside the policy.
        """
        super().__init__(rpc, max_age=max_age, now=now)
        if not isinstance(max_distance, timedelta):
            raise TypeError("max_distance must be timedelta")
        if not timedelta(0) < max_distance <= timedelta(hours=1):
            raise ValueError("max_distance must be positive and at most one hour")
        self._max_distance = max_distance

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Look up the closest USD observation around a transaction timestamp.

        Preference is given to the earlier price when two points are equally
        close. No price is fabricated, interpolated, or assigned a tx timestamp.

        :param asset_id: Mainnet native ETH or ERC-20 identity.
        :param at: Timezone-aware transaction timestamp.
        :return: Closest valid PriceQuote, or None if no suitable point exists.
        :raises TypeError: If an argument has the wrong type.
        :raises ValueError: If chain, timestamp or date range is invalid.
        :raises ProviderError: For malformed responses and transport failures.
        """
        if not isinstance(asset_id, AssetId):
            raise TypeError("asset_id must be an AssetId")
        if asset_id.chain_id != 1:
            raise ValueError("Alchemy Prices supports Ethereum Mainnet only")
        if not isinstance(at, datetime):
            raise TypeError("transaction timestamp must be datetime")
        if at.utcoffset() is None:
            raise ValueError("transaction timestamp must be timezone-aware")
        requested_at = at.astimezone(UTC)
        try:
            start_time = requested_at - self._max_distance
            end_time = requested_at + self._max_distance
        except OverflowError:
            raise ValueError("transaction timestamp exceeds representable lookup window") from None
        raw = await self._rpc.request_alchemy_historical_prices(
            contract_address=asset_id.contract_address,
            start_time=start_time,
            end_time=end_time,
        )
        if raw is None:
            self._log_missing()
            return None
        data = _mapping(raw, operation=_OPERATION)
        if data.get("error") is not None:
            raise ProviderError("alchemy", _OPERATION)
        if asset_id.is_native:
            if data.get("symbol") != "ETH":
                raise ProviderMalformedResponseError("alchemy", _OPERATION)
        elif (
            data.get("network") != "eth-mainnet"
            or not isinstance(data.get("address"), str)
            or data["address"].lower() != asset_id.contract_address
        ):
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        currency = data.get("currency")
        if not isinstance(currency, str) or currency.lower() != "usd":
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        rows = data.get("data")
        if not isinstance(rows, list):
            raise ProviderMalformedResponseError("alchemy", _OPERATION)

        points: dict[datetime, UsdPrice] = {}
        for raw_row in rows:
            row = _mapping(raw_row, operation=_OPERATION)
            observed_at = _parse_observation_time(row.get("timestamp"))
            price = _parse_usd_price(row.get("value"))
            previous = points.get(observed_at)
            if previous is not None and previous != price:
                raise ProviderMalformedResponseError("alchemy", _OPERATION)
            points[observed_at] = price

        candidates = [
            (observed_at, price)
            for observed_at, price in points.items()
            if abs(observed_at - requested_at) <= self._max_distance
        ]
        if not candidates:
            self._log_missing()
            return None
        best_time, best_price = min(
            candidates,
            key=lambda item: (
                abs(item[0] - requested_at),
                item[0] > requested_at,
                item[0],
            ),
        )
        quote = PriceQuote(
            price_usd=best_price,
            observed_at=best_time,
            source=(
                "alchemy_historical_eth_symbol_5m"
                if asset_id.is_native
                else "alchemy_historical_token_address_5m"
            ),
        )
        _LOGGER.debug(
            "historical_price_completed",
            extra={"provider": "alchemy", "operation": _OPERATION},
        )
        return quote

    @staticmethod
    def _log_missing() -> None:
        """Emit a safe event without token identifiers or transaction hashes."""
        _LOGGER.info(
            "historical_price_unavailable",
            extra={"provider": "alchemy", "operation": _OPERATION},
        )


def _parse_observation_time(value: object) -> datetime:
    """Parse a vendor ISO-8601 timestamp and preserve actual observation time.

    :param value: Vendor timestamp value.
    :return: Timezone-aware UTC timestamp.
    :raises ProviderMalformedResponseError: For absent or naive timestamps.
    """
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() is None:
            raise ValueError("naive timestamp")
        return result.astimezone(UTC)
    except (ValueError, OverflowError):
        raise ProviderMalformedResponseError("alchemy", _OPERATION) from None


def _parse_usd_price(value: object) -> UsdPrice:
    """Parse an exact nonnegative price without numeric coercion or floats.

    :param value: Vendor decimal price string.
    :return: Finite nonnegative USD price.
    :raises ProviderMalformedResponseError: For malformed or extreme values.
    """
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    try:
        price = UsdPrice(Decimal(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ProviderMalformedResponseError("alchemy", _OPERATION) from None
    if abs(price.value.adjusted()) > 1000:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    return price

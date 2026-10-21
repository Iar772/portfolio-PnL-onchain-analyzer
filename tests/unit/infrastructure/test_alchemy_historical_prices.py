"""Offline integration-contract tests for historical Alchemy price observations."""

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO

import httpx
import pytest

from portfolio_analyzer.application.ports import PriceProvider
from portfolio_analyzer.application.provider_errors import (
    ProviderError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.infrastructure.alchemy.historical_prices import AlchemyPriceProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider
from portfolio_analyzer.logging_config import configure_logging

_ETH = AssetId(chain_id=1, contract_address=None)
_TOKEN = AssetId(chain_id=1, contract_address="0x" + "ab" * 20)
_OTHER = AssetId(chain_id=1, contract_address="0x" + "cd" * 20)
_AT = datetime(2024, 7, 1, 12, 0, tzinfo=UTC)


def _point(*, value: object = "1.01", stamp: object = "2024-07-01T11:59:00Z") -> dict[str, object]:
    """Return one sample from Alchemy historical endpoint."""
    return {"value": value, "timestamp": stamp}


def _payload(*, token: bool = True, data: object = None, **overrides: object) -> dict[str, object]:
    """Return a syntactically valid historical Prices API response."""
    value: dict[str, object] = {
        "currency": "usd",
        "data": [_point()] if data is None else data,
    }
    if token:
        value.update(network="eth-mainnet", address=_TOKEN.contract_address)
    else:
        value["symbol"] = "ETH"
    value.update(overrides)
    return value


def _json_transport(payload: object) -> httpx.MockTransport:
    """Return an offline successful historical REST response."""
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))


def _provider(
    client: httpx.AsyncClient,
    *,
    max_distance: timedelta = timedelta(minutes=15),
    max_attempts: int = 3,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> AlchemyPriceProvider:
    """Construct the full price port using the existing bounded transport."""
    if sleep is None:
        rpc = AlchemyRpcProvider(client, "SECRET_ALCHEMY_KEY", max_attempts=max_attempts)
    else:
        rpc = AlchemyRpcProvider(
            client, "SECRET_ALCHEMY_KEY", max_attempts=max_attempts, sleep=sleep
        )
    return AlchemyPriceProvider(rpc, max_distance=max_distance)


@pytest.mark.asyncio
async def test_combined_price_provider_supports_current_and_historical() -> None:
    """One provider satisfies the full PriceProvider port and shares HTTP client."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/by-address"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "network": "eth-mainnet",
                            "address": _TOKEN.contract_address,
                            "prices": [
                                {
                                    "currency": "usd",
                                    "value": "3",
                                    "lastUpdatedAt": "2024-07-01T12:00:00Z",
                                }
                            ],
                            "error": None,
                        }
                    ]
                },
            )
        return httpx.Response(200, json=_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "SECRET_ALCHEMY_KEY")
        provider: PriceProvider = AlchemyPriceProvider(rpc, now=lambda: _AT)
        current = await provider.get_current_price(_TOKEN)
        historical = await provider.get_historical_price(_TOKEN, _AT)
    assert current is not None and historical is not None
    assert current.price_usd.value == Decimal("3")
    assert historical.price_usd.value == Decimal("1.01")
    assert seen[0].endswith("/by-address")
    assert seen[1].endswith("/historical")


@pytest.mark.asyncio
async def test_erc20_price_by_address_and_nearest_point() -> None:
    """Request address-identified 5m data and select nearest real observation."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_payload(
                data=[
                    _point(value="2", stamp="2024-07-01T11:53:00Z"),
                    _point(value="1.234567890123456789", stamp="2024-07-01T11:59:00Z"),
                    _point(value="3", stamp="2024-07-01T12:08:00Z"),
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = _provider(client)
        price_port: PriceProvider = provider
        assert price_port is provider
        quote = await provider.get_historical_price(_TOKEN, _AT)
    assert quote is not None
    assert quote.price_usd.value == Decimal("1.234567890123456789")
    assert quote.observed_at == _AT - timedelta(minutes=1)
    assert quote.source == "alchemy_historical_token_address_5m"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/prices/v1/SECRET_ALCHEMY_KEY/tokens/historical"
    assert json.loads(requests[0].content) == {
        "network": "eth-mainnet",
        "address": _TOKEN.contract_address,
        "startTime": "2024-07-01T11:45:00Z",
        "endTime": "2024-07-01T12:15:00Z",
        "interval": "5m",
    }


@pytest.mark.asyncio
async def test_native_eth_by_explicit_allowlisted_symbol() -> None:
    """ETH lookup uses only ETH, not arbitrary metadata symbol."""
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_payload(token=False, data=[_point(value="3300")]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        quote = await _provider(client).get_historical_price(_ETH, _AT)
    assert quote is not None
    assert quote.price_usd.value == Decimal("3300")
    assert quote.source == "alchemy_historical_eth_symbol_5m"
    assert seen[0]["symbol"] == "ETH"
    assert "address" not in seen[0]
    assert "network" not in seen[0]


@pytest.mark.asyncio
async def test_timezone_aware_input_normalized_to_utc() -> None:
    """Transaction timestamps with offsets produce an equivalent UTC window."""
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_payload())

    offset = timezone(timedelta(hours=3))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        quote = await _provider(client).get_historical_price(
            _TOKEN, datetime(2024, 7, 1, 15, 0, tzinfo=offset)
        )
    assert quote is not None
    assert quote.observed_at == _AT - timedelta(minutes=1)
    assert seen[0]["startTime"] == "2024-07-01T11:45:00Z"


@pytest.mark.asyncio
async def test_equidistant_quotes_prefer_prior_observation() -> None:
    """Tie-breaking is stable and avoids future price lookahead."""
    data = [
        _point(stamp="2024-07-01T12:03:00Z", value="3"),
        _point(stamp="2024-07-01T11:57:00Z", value="2"),
    ]
    async with httpx.AsyncClient(transport=_json_transport(_payload(data=data))) as client:
        quote = await _provider(client).get_historical_price(_TOKEN, _AT)
    assert quote is not None
    assert quote.price_usd.value == Decimal("2")
    assert quote.observed_at < _AT


@pytest.mark.asyncio
async def test_nearest_after_timestamp_when_no_closer_prior() -> None:
    """Actual future sample time remains visible to accounting callers."""
    data = [
        _point(stamp="2024-07-01T11:52:00Z", value="2"),
        _point(stamp="2024-07-01T12:02:00Z", value="3"),
    ]
    async with httpx.AsyncClient(transport=_json_transport(_payload(data=data))) as client:
        quote = await _provider(client).get_historical_price(_TOKEN, _AT)
    assert quote is not None
    assert quote.observed_at == _AT + timedelta(minutes=2)
    assert quote.price_usd.value == Decimal("3")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        _payload(data=[]),
        _payload(data=[_point(stamp="2024-07-01T11:44:59Z")]),
        _payload(data=[_point(stamp="2024-07-01T12:15:01Z")]),
    ],
)
async def test_missing_historical_price_returns_none(payload: object) -> None:
    """No nearby quote is unknown; never substitute a zero price."""
    async with httpx.AsyncClient(transport=_json_transport(payload)) as client:
        assert await _provider(client).get_historical_price(_TOKEN, _AT) is None


@pytest.mark.asyncio
async def test_http_404_is_explicit_missing_historical_price() -> None:
    """HTTP 404 means not found only for historical price endpoint."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"message": "not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await _provider(client).get_historical_price(_TOKEN, _AT) is None
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"data": []},
        _payload(currency="eur"),
        _payload(currency=5),
        _payload(data={}),
        _payload(data="incorrect"),
        _payload(data=[None]),
        _payload(data=[_point(value="NaN")]),
        _payload(data=[_point(value="Infinity")]),
        _payload(data=[_point(value="-1")]),
        _payload(data=[_point(value=3.14)]),
        _payload(data=[_point(value=True)]),
        _payload(data=[_point(value="1e1001")]),
        _payload(data=[_point(value="1" * 129)]),
        _payload(data=[_point(stamp="invalid")]),
        _payload(data=[_point(stamp="2024-07-01T12:00:00")]),
        _payload(data=[_point(stamp=1234)]),
        _payload(data=[_point(stamp="x" * 65)]),
        _payload(network="base-mainnet"),
        _payload(address=_OTHER.contract_address),
        _payload(address=123),
        {"symbol": "USDC", "currency": "usd", "data": [_point()]},
        _payload(
            data=[
                _point(stamp="2024-07-01T11:59:00Z", value="1"),
                _point(stamp="2024-07-01T11:59:00Z", value="2"),
            ]
        ),
    ],
)
async def test_malformed_historical_data_is_not_silently_missing(payload: object) -> None:
    """Unexpected API schemas raise normalized errors rather than unknown price."""
    async with httpx.AsyncClient(transport=_json_transport(payload)) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await _provider(client, max_attempts=1).get_historical_price(_TOKEN, _AT)


@pytest.mark.asyncio
async def test_same_timestamp_same_price_dedup_and_zero_valid() -> None:
    """Duplicated identical points are harmless; reported zero is known zero."""
    data = [
        _point(stamp="2024-07-01T12:00:00Z", value="0"),
        _point(stamp="2024-07-01T12:00:00+00:00", value="0.0"),
    ]
    async with httpx.AsyncClient(transport=_json_transport(_payload(data=data))) as client:
        quote = await _provider(client).get_historical_price(_TOKEN, _AT)
    assert quote is not None
    assert quote.price_usd.value == Decimal(0)


@pytest.mark.asyncio
async def test_vendor_200_explicit_error_is_not_missing() -> None:
    """Vendor errors delivered as HTTP 200 are never treated as no price."""
    async with httpx.AsyncClient(
        transport=_json_transport(_payload(error={"message": "oops"}))
    ) as client:
        with pytest.raises(ProviderError):
            await _provider(client, max_attempts=1).get_historical_price(_TOKEN, _AT)


@pytest.mark.asyncio
async def test_api_rate_limit_retried_under_existing_policy() -> None:
    """Historical Prices uses exactly the RPC shared bounded retry policy."""
    calls = 0
    waits: list[float] = []

    async def fake_sleep(duration: float) -> None:
        """Capture backoff without real waiting."""
        waits.append(duration)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        quote = await _provider(client, sleep=fake_sleep).get_historical_price(_TOKEN, _AT)
        assert quote is not None
    assert calls == 2
    assert waits == [0.5]


@pytest.mark.asyncio
async def test_permanent_rate_limit_exhausts_exact_attempt_limit() -> None:
    """No infinite retries on persistent 429s."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429)

    async def no_sleep(duration: float) -> None:
        """Skip backoff delay."""
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderRateLimitError):
            await _provider(client, sleep=no_sleep).get_historical_price(_TOKEN, _AT)
    assert calls == 3


@pytest.mark.asyncio
async def test_timeout_keeps_existing_error_semantics() -> None:
    """Timeout remains explicit instead of being misclassified as missing."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("SECRET_ALCHEMY_KEY wallet 0xdeadbeef")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderTimeoutError):
            await _provider(client, max_attempts=1).get_historical_price(_TOKEN, _AT)


@pytest.mark.asyncio
async def test_generic_http_400_is_not_misclassified_as_missing() -> None:
    """Only HTTP 404 receives the explicit historical missing policy."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(400, json={"bad": "request"}))
    ) as client:
        with pytest.raises(ProviderError):
            await _provider(client, max_attempts=1).get_historical_price(_TOKEN, _AT)


@pytest.mark.asyncio
async def test_invalid_json_response_is_malformed() -> None:
    """Malformed HTTP body is distinct from a missing price."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"{broken"))
    ) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await _provider(client, max_attempts=1).get_historical_price(_TOKEN, _AT)


def test_invalid_lookup_args_before_http() -> None:
    """Reject naive dates, wrong chains and invalid configuration eagerly."""
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    )
    try:
        rpc = AlchemyRpcProvider(async_client, "SECRET_ALCHEMY_KEY")
        with pytest.raises(TypeError):
            AlchemyPriceProvider(rpc, max_distance=10)  # type: ignore[arg-type]
        for duration in (timedelta(0), timedelta(hours=2)):
            with pytest.raises(ValueError):
                AlchemyPriceProvider(rpc, max_distance=duration)
    finally:
        # This synchronous validation test never sends a request.
        import asyncio

        asyncio.run(async_client.aclose())


@pytest.mark.asyncio
async def test_bad_asset_or_timestamp_rejected() -> None:
    """Only Mainnet assets and timezone-aware datetimes can be priced."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    ) as client:
        provider = _provider(client)
        for asset, moment, expected in (
            (AssetId(10, _TOKEN.contract_address), _AT, ValueError),
            (_TOKEN, datetime(2024, 7, 1, 12), ValueError),
            (_TOKEN, "bad", TypeError),
            ("bad", _AT, TypeError),
        ):
            with pytest.raises(expected):
                await provider.get_historical_price(asset, moment)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_http_404_is_not_enabled_for_regular_rpc() -> None:
    """Historical-only 404 mapping does not change established RPC errors."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(404))
    ) as client:
        with pytest.raises(ProviderError):
            await AlchemyRpcProvider(client, "SECRET_ALCHEMY_KEY", max_attempts=1).get_chain_id()


@pytest.mark.asyncio
async def test_safe_structured_logs_for_historical_prices() -> None:
    """Historical price logs omit keys, token identity and vendor response text."""
    stream = StringIO()
    configure_logging(AppConfig(log_level="DEBUG"), stream=stream)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, json={"sensitive": "WALLET_ADDRESS"})
        )
    ) as client:
        assert await _provider(client).get_historical_price(_TOKEN, _AT) is None
    contents = stream.getvalue()
    assert "historical_price_unavailable" in contents
    assert "SECRET_ALCHEMY_KEY" not in contents
    assert _TOKEN.contract_address not in contents
    assert "WALLET_ADDRESS" not in contents
    for line in contents.splitlines():
        payload = json.loads(line)
        assert payload["provider"] == "alchemy"
        assert payload["operation"] == "get_historical_price"


@pytest.mark.asyncio
async def test_ancient_timestamp_outside_supported_datetime_range() -> None:
    """Prevent interval overflow before constructing the outbound request."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    ) as client:
        with pytest.raises(ValueError, match="representable"):
            await _provider(client).get_historical_price(_ETH, datetime.min.replace(tzinfo=UTC))


@pytest.mark.asyncio
async def test_native_response_identity_cannot_change_to_other_symbol() -> None:
    """A native ETH quote never uses a substituted vendor ticker."""
    async with httpx.AsyncClient(
        transport=_json_transport(_payload(token=False, symbol="WETH"))
    ) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await _provider(client).get_historical_price(_ETH, _AT)

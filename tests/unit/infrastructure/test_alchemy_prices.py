"""Offline API-contract tests for Alchemy current USD price observations."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO

import httpx
import pytest

from portfolio_analyzer.application.provider_errors import (
    ProviderError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.infrastructure.alchemy.prices import AlchemyCurrentPriceProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider
from portfolio_analyzer.logging_config import configure_logging

_ETH = AssetId(1, None)
_USDC = AssetId(1, "0x" + "ab" * 20)
_OTHER = AssetId(1, "0x" + "cd" * 20)
_NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
_STAMP = "2026-10-21T11:59:00Z"


def _price(
    value: object = "1.001", *, stamp: object = _STAMP, currency: object = "usd"
) -> dict[str, object]:
    """Return one synthetic Alchemy price row."""
    return {"currency": currency, "value": value, "lastUpdatedAt": stamp}


def _token_entry(
    *,
    price_rows: list[object] | None = None,
    error: object = None,
    address: str = _USDC.contract_address or "",
    network: str = "eth-mainnet",
) -> dict[str, object]:
    """Return vendor price response for an ERC-20 token."""
    return {
        "network": network,
        "address": address,
        "prices": price_rows if price_rows is not None else [_price()],
        "error": error,
    }


def _native_entry(*, prices: list[object] | None = None, symbol: str = "ETH") -> dict[str, object]:
    """Return vendor price response for the single native allowlist symbol."""
    return {
        "symbol": symbol,
        "prices": prices if prices is not None else [_price("3000")],
        "error": None,
    }


def _provider(client: httpx.AsyncClient) -> AlchemyCurrentPriceProvider:
    """Inject a fixed clock and real shared Alchemy RPC HTTP policies."""
    return AlchemyCurrentPriceProvider(
        AlchemyRpcProvider(client, "TEST_SECRET_TOKEN"), now=lambda: _NOW
    )


@pytest.mark.asyncio
async def test_erc20_price_by_address_and_network_not_symbol() -> None:
    """Only chain and contract address may determine an ERC-20 quote."""
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, json={"data": [_token_entry()]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        quote = await _provider(client).get_current_price(_USDC)
    assert quote is not None
    assert quote.price_usd.value == Decimal("1.001")
    assert quote.observed_at == datetime(2026, 10, 21, 11, 59, tzinfo=UTC)
    assert quote.source == "alchemy_prices_by_address"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/prices/v1/TEST_SECRET_TOKEN/tokens/by-address"
    assert json.loads(requests[0].content) == {
        "addresses": [{"network": "eth-mainnet", "address": _USDC.contract_address}]
    }


@pytest.mark.asyncio
async def test_native_eth_uses_only_fixed_allowlisted_symbol() -> None:
    """A native ETH query cannot inherit a spoofed ERC-20 symbol."""
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, json={"data": [_native_entry()]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        quote = await _provider(client).get_current_price(_ETH)
    assert quote is not None
    assert quote.price_usd.value == Decimal("3000")
    assert quote.source == "alchemy_prices_by_symbol"
    assert requests[0].method == "GET"
    assert requests[0].url.path.endswith("/by-symbol")
    assert requests[0].url.params["symbols"] == "ETH"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"data": [_token_entry(price_rows=[])]},
        {"data": [_token_entry(price_rows=[_price(currency="eur")])]},
        {"data": [_token_entry(error="not supported")]},
        {"data": [_token_entry(price_rows=[_price(stamp="2026-10-19T11:00:00Z")])]},
        {"data": [_token_entry(price_rows=[_price(stamp="2026-10-21T12:06:00Z")])]},
    ],
)
async def test_missing_unsupported_or_stale_returns_unpriced(payload: object) -> None:
    """An unavailable quote never defaults to a fabricated price."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    ) as client:
        assert await _provider(client).get_current_price(_USDC) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": {}},
        {"data": "wrong"},
        {"data": [1]},
        {"data": [_token_entry(network="base-mainnet")]},
        {"data": [_token_entry(address="0x" + "ef" * 20)]},
        {"data": [_token_entry(), _token_entry()]},
        {"data": [_token_entry(price_rows="invalid")]},
        {"data": [_token_entry(price_rows=[_price(), _price()])]},
        {"data": [_token_entry(price_rows=[_price(value="NaN")])]},
        {"data": [_token_entry(price_rows=[_price(value="Infinity")])]},
        {"data": [_token_entry(price_rows=[_price(value="-1")])]},
        {"data": [_token_entry(price_rows=[_price(value=3.14)])]},
        {"data": [_token_entry(price_rows=[_price(stamp="invalid")])]},
        {"data": [_token_entry(price_rows=[_price(stamp="2026-10-21T12:00:00")])]},
        {"data": [_token_entry(price_rows=[_price(currency=None)])]},
        {"data": [_token_entry(error={"message": "not supported"})]},
        {"data": [_token_entry(price_rows=[_price(value="1e99999999")])]},
    ],
)
async def test_malformed_data_is_not_silent_missing_price(payload: object) -> None:
    """Corrupt or mismatched vendor responses are explicit provider failures."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await _provider(client).get_current_price(_USDC)


@pytest.mark.asyncio
async def test_known_zero_price_remains_a_known_quote() -> None:
    """Distinguish the well-defined zero quote from missing observations."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"data": [_token_entry(price_rows=[_price("0")])]})
        )
    ) as client:
        quote = await _provider(client).get_current_price(_USDC)
    assert quote is not None and quote.price_usd.value == Decimal(0)


@pytest.mark.asyncio
async def test_exact_decimal_large_high_precision_price() -> None:
    """Never convert the API's decimal string through float."""
    value = "0.000000000000000000000000123456789123456789"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"data": [_token_entry(price_rows=[_price(value)])]}
            )
        )
    ) as client:
        quote = await _provider(client).get_current_price(_USDC)
    assert quote is not None and quote.price_usd.value == Decimal(value)


@pytest.mark.asyncio
async def test_wrong_chain_and_asset_type_rejected_before_network() -> None:
    """Ethereum Mainnet only; never use an unsupported chain implicitly."""
    requests = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = _provider(client)
        with pytest.raises(ValueError):
            await provider.get_current_price(AssetId(137, _USDC.contract_address))
        with pytest.raises(TypeError):
            await provider.get_current_price("USDC")  # type: ignore[arg-type]
    assert requests == 0


@pytest.mark.asyncio
async def test_retries_http_429_once_with_shared_policy() -> None:
    """Prices REST shares the same bounded retry budget and semaphore as RPC."""
    attempts = 0
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"data": [_token_entry()]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "TEST_SECRET_TOKEN", sleep=sleep)
        provider = AlchemyCurrentPriceProvider(rpc, now=lambda: _NOW)
        assert await provider.get_current_price(_USDC) is not None
        assert not client.is_closed
    assert client.is_closed
    assert attempts == 2 and delays == [0.5]


@pytest.mark.asyncio
async def test_rate_limit_retry_budget_exhaustion() -> None:
    """A rate limit raises a typed exception after at most three attempts."""
    attempts = 0

    async def sleep(seconds: float) -> None:
        return None

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AlchemyCurrentPriceProvider(
            AlchemyRpcProvider(client, "TEST_SECRET_TOKEN", sleep=sleep), now=lambda: _NOW
        )
        with pytest.raises(ProviderRateLimitError):
            await provider.get_current_price(_USDC)
    assert attempts == 3


@pytest.mark.asyncio
async def test_timeout_is_not_misreported_as_unpriced() -> None:
    """Temporary request failures propagate, not treated as missing prices."""

    async def sleep(seconds: float) -> None:
        return None

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private text", request=req)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "TEST_SECRET_TOKEN", sleep=sleep)
        with pytest.raises(ProviderTimeoutError):
            await AlchemyCurrentPriceProvider(rpc, now=lambda: _NOW).get_current_price(_USDC)


@pytest.mark.asyncio
async def test_authorization_error_is_not_retried() -> None:
    """Credential failures do not trigger repeated requests."""
    attempts = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError):
            await _provider(client).get_current_price(_USDC)
    assert attempts == 1


@pytest.mark.asyncio
async def test_json_logs_do_not_expose_secrets_or_token_addresses() -> None:
    """Only allowlisted static event attributes survive configured formatter."""
    output = StringIO()
    configure_logging(AppConfig(log_level="DEBUG"), stream=output)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [_token_entry(error="PRIVATE_VENDOR_TRACE")]})

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await _provider(client).get_current_price(_USDC)
        logs = output.getvalue()
        assert '"event":"current_price_unavailable"' in logs
        assert "PRIVATE_VENDOR_TRACE" not in logs
        assert "TEST_SECRET_TOKEN" not in logs
        assert (_USDC.contract_address or "") not in logs
    finally:
        import logging

        logging.getLogger("portfolio_analyzer").handlers.clear()


@pytest.mark.asyncio
async def test_unexpected_native_symbol_rejected() -> None:
    """Native ETH must not accept other symbol responses."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"data": [_native_entry(symbol="WETH")]})
        )
    ) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await _provider(client).get_current_price(_ETH)


@pytest.mark.parametrize("age", [timedelta(0), timedelta(days=2)])
@pytest.mark.asyncio
async def test_invalid_freshness_setting_is_rejected(age: timedelta) -> None:
    """Prevent configuration that accepts absent or arbitrary-old prices."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(500))
    ) as client:
        with pytest.raises(ValueError):
            AlchemyCurrentPriceProvider(AlchemyRpcProvider(client, "test"), max_age=age)


def test_invalid_provider_constructor_arguments() -> None:
    """Reject configuration bugs before issuing any HTTP requests."""
    with pytest.raises(TypeError):
        AlchemyCurrentPriceProvider("rpc")  # type: ignore[arg-type]
    client = httpx.AsyncClient()
    rpc = AlchemyRpcProvider(client, "test")
    with pytest.raises(TypeError):
        AlchemyCurrentPriceProvider(rpc, max_age=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AlchemyCurrentPriceProvider(rpc, now="clock")  # type: ignore[arg-type]
    import asyncio

    asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_invalid_timestamp_type_or_clock_is_explicit() -> None:
    """Reject untrusted timestamps or broken clock rather than guessing prices."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"data": [_token_entry(price_rows=[_price(stamp=123)])]}
            )
        )
    ) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await _provider(client).get_current_price(_USDC)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"data": [_token_entry()]})
        )
    ) as client:
        rpc = AlchemyRpcProvider(client, "TEST_SECRET_TOKEN")
        with pytest.raises(ValueError, match="clock"):
            await AlchemyCurrentPriceProvider(
                rpc, now=lambda: datetime(2026, 10, 21)
            ).get_current_price(_USDC)


@pytest.mark.asyncio
async def test_default_clock_writes_current_utc_time() -> None:
    """Production default time source returns an aware UTC observation."""
    from portfolio_analyzer.infrastructure.alchemy.prices import _utc_now

    observed = _utc_now()
    assert observed.tzinfo is UTC


@pytest.mark.asyncio
async def test_rest_and_rpc_share_semaphore() -> None:
    """Different Alchemy endpoint types honor one bounded HTTP concurrency cap."""
    import asyncio

    active = 0
    peak = 0

    async def handler(req: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        if req.url.path.endswith("/by-address"):
            return httpx.Response(200, json={"data": [_token_entry()]})
        body = json.loads(req.content)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "TEST_SECRET_TOKEN", max_concurrency=1)
        price = AlchemyCurrentPriceProvider(rpc, now=lambda: _NOW)
        results = await asyncio.gather(rpc.get_chain_id(), price.get_current_price(_USDC))
    assert results[0] == 1 and results[1] is not None
    assert peak == 1


@pytest.mark.asyncio
async def test_price_adapter_rejects_invalid_data_from_transport_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail safely even if a future transport implementation violates its contract."""

    async def broken_prices(*, contract_address: str | None = None) -> object:
        """Pretend the transport returned a malformed REST data envelope."""
        return {"data": "wrong"}

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(500))
    ) as client:
        rpc = AlchemyRpcProvider(client, "TEST_SECRET_TOKEN")
        monkeypatch.setattr(rpc, "request_alchemy_prices", broken_prices)
        provider = AlchemyCurrentPriceProvider(rpc, now=lambda: _NOW)
        with pytest.raises(ProviderMalformedResponseError):
            await provider.get_current_price(_USDC)

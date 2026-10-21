"""Authoritative native ETH balance contract tests without live Alchemy calls."""

import json
from decimal import Decimal
from typing import Any

import httpx
import pytest

from portfolio_analyzer.application.ports import RpcProvider
from portfolio_analyzer.application.provider_errors import (
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider

_WALLET = "0x" + "AB" * 20
_CANONICAL_WALLET = _WALLET.lower()
_NATIVE_ETH = AssetId(chain_id=1, contract_address=None)


def _rpc_reply(request: httpx.Request, result: object) -> httpx.Response:
    """Build a valid JSON-RPC response matching the outbound request ID.

    :param request: Outgoing request with a JSON-RPC envelope.
    :param result: Mocked RPC result value.
    :return: Correlated successful JSON-RPC response.
    """
    envelope: dict[str, Any] = json.loads(request.content)
    return httpx.Response(
        200,
        json={"jsonrpc": "2.0", "id": envelope["id"], "result": result},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_wei", "expected_eth"),
    [
        (0, Decimal("0")),
        (1, Decimal("0.000000000000000001")),
        (10**18, Decimal("1")),
        (10**18 + 1, Decimal("1.000000000000000001")),
        (
            123456789012345678901234567890123456789,
            Decimal("123456789012345678901.234567890123456789"),
        ),
    ],
)
async def test_native_balance_preserves_exact_wei(raw_wei: int, expected_eth: Decimal) -> None:
    """Convert wei to a native ETH domain value without floating point.

    :param raw_wei: Exact nonnegative ETH amount in wei.
    :param expected_eth: Expected whole-ETH Decimal representation.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the configured balance as a valid RPC quantity.

        :param request: JSON-RPC request.
        :return: Result containing the native balance in hex wei.
        """
        return _rpc_reply(request, hex(raw_wei))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc: RpcProvider = AlchemyRpcProvider(client, "test_key")
        result = await rpc.get_balance(_WALLET)

    assert isinstance(result, AssetAmount)
    assert result.asset_id == _NATIVE_ETH
    assert result.quantity == expected_eth


@pytest.mark.asyncio
async def test_native_balance_supports_latest_and_pinned_block() -> None:
    """Query the requested state snapshot rather than silently switching blocks."""
    observed: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record requested block selectors for verification.

        :param request: Outbound JSON-RPC request.
        :return: One wei balance for both block selectors.
        """
        body: dict[str, Any] = json.loads(request.content)
        observed.append(body)
        return _rpc_reply(request, "0x1")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key")
        await rpc.get_balance(_WALLET)
        await rpc.get_balance(_WALLET, at_block=20_000_000)

    assert [entry["method"] for entry in observed] == ["eth_getBalance", "eth_getBalance"]
    assert [entry["params"] for entry in observed] == [
        [_CANONICAL_WALLET, "latest"],
        [_CANONICAL_WALLET, hex(20_000_000)],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [None, "0x", "0x00", "0x01", "1", -1, 1.0])
async def test_native_balance_rejects_malformed_rpc_quantity(malformed: object) -> None:
    """Reject invalid raw quantities instead of inventing a usable ETH balance.

    :param malformed: Noncanonical or non-hex RPC result to reject.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an invalid JSON-RPC result for a negative-path test.

        :param request: Outgoing RPC request.
        :return: Response with malformed balance payload.
        """
        return _rpc_reply(request, malformed)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key", max_attempts=1)
        with pytest.raises(ProviderMalformedResponseError):
            await rpc.get_balance(_WALLET)


@pytest.mark.asyncio
async def test_native_balance_preserves_rate_limit_error() -> None:
    """Expose the typed rate-limit failure rather than a fabricated zero balance."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Simulate Alchemy responding with HTTP 429.

        :param request: Outgoing RPC request.
        :return: Rate-limited HTTP response.
        """
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key", max_attempts=1)
        with pytest.raises(ProviderRateLimitError):
            await rpc.get_balance(_WALLET)


@pytest.mark.asyncio
async def test_native_balance_preserves_timeout_error() -> None:
    """Propagate a typed timeout instead of reporting an incorrect balance."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Raise a read timeout from the mocked transport.

        :param request: Outgoing RPC request.
        :raises httpx.ReadTimeout: Simulated network timeout.
        """
        raise httpx.ReadTimeout("mock timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key", max_attempts=1)
        with pytest.raises(ProviderTimeoutError):
            await rpc.get_balance(_WALLET)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_block", [-1, True, "100"])
async def test_native_balance_rejects_invalid_snapshot(invalid_block: object) -> None:
    """Reject unsupported block selectors before an HTTP request.

    :param invalid_block: Negative, boolean or string block selector.
    """
    requests_made = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Record unexpected network requests during input validation.

        :param request: Unexpected outbound request.
        :return: Zero balance response when called.
        """
        nonlocal requests_made
        requests_made += 1
        return _rpc_reply(request, "0x0")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key")
        with pytest.raises((TypeError, ValueError)):
            await rpc.get_balance(_WALLET, at_block=invalid_block)  # type: ignore[arg-type]

    assert requests_made == 0

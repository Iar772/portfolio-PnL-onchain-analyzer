"""Offline Alchemy RPC contract tests using an injected HTTPX transport."""

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest

from portfolio_analyzer.application.ports import RpcProvider
from portfolio_analyzer.application.provider_errors import (
    ProviderError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider

_ADDRESS = "0x" + "ab" * 20
_TARGET = "0x" + "cd" * 20
_TX_HASH = "0x" + "ef" * 32
_TOPIC = "0x" + "11" * 32


def _reply(request: httpx.Request, result: object) -> httpx.Response:
    """Build a correlated JSON-RPC success response.

    :param request: Outbound RPC request.
    :param result: JSON-RPC result payload.
    :return: Mock HTTP success response.
    """
    body: dict[str, Any] = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


@pytest.mark.asyncio
async def test_native_eth_balance_is_exact_and_block_scoped() -> None:
    """Preserve very small wei quantities without floating point arithmetic."""
    observed: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Inspect eth_getBalance params without invoking the network."""
        body = json.loads(request.content)
        observed.append(body)
        return _reply(request, "0x1")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key")
        port: RpcProvider = rpc
        balance = await port.get_balance(_ADDRESS, at_block=10)
        assert balance.asset_id.chain_id == 1
        assert balance.asset_id.contract_address is None
        assert balance.quantity == Decimal("0.000000000000000001")
        assert observed[0]["method"] == "eth_getBalance"
        assert observed[0]["params"] == [_ADDRESS, "0xa"]


@pytest.mark.asyncio
async def test_chain_id_and_latest_block() -> None:
    """Convert RPC quantities to plain Python integers."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return deterministic chain status for both RPC methods."""
        body = json.loads(request.content)
        return _reply(request, "0x1" if body["method"] == "eth_chainId" else "0xabcdef")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key")
        assert await rpc.get_chain_id() == 1
        assert await rpc.get_block_number() == 0xABCDEF


@pytest.mark.asyncio
async def test_pending_transaction_has_no_mined_block_and_no_receipt() -> None:
    """Keep pending transaction state separate from confirmed history."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a pending RPC transaction or missing receipt."""
        body = json.loads(request.content)
        if body["method"] == "eth_getTransactionReceipt":
            return _reply(request, None)
        return _reply(
            request,
            {
                "hash": _TX_HASH,
                "from": _ADDRESS,
                "to": _TARGET,
                "blockNumber": None,
                "value": "0x0",
                "input": "0x1234",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key")
        tx = await rpc.get_transaction_by_hash(_TX_HASH)
        assert tx is not None
        assert tx.block_number is None
        assert tx.input_data == bytes.fromhex("1234")
        assert await rpc.get_transaction_receipt(_TX_HASH) is None


@pytest.mark.asyncio
async def test_confirmed_receipt_logs_and_block_are_internal_models() -> None:
    """Parse receipt logs and UTC block facts without vendor dictionaries escaping."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Mock the two RPC endpoints needed for receipt and block facts."""
        body = json.loads(request.content)
        if body["method"] == "eth_getBlockByNumber":
            return _reply(request, {"number": "0xa", "timestamp": "0x64"})
        return _reply(
            request,
            {
                "transactionHash": _TX_HASH,
                "blockNumber": "0xa",
                "status": "0x1",
                "gasUsed": "0x5208",
                "effectiveGasPrice": "0x3b9aca00",
                "logs": [{"address": _TARGET, "topics": [_TOPIC], "data": "0x1234"}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key")
        receipt = await rpc.get_transaction_receipt(_TX_HASH)
        assert receipt is not None
        assert receipt.success is True
        assert receipt.gas_used == 21000
        assert receipt.effective_gas_price_wei == 10**9
        assert receipt.logs[0].topics == (bytes.fromhex("11" * 32),)
        assert receipt.logs[0].data == bytes.fromhex("1234")
        block = await rpc.get_block(10)
        assert block is not None
        assert block.timestamp == datetime.fromtimestamp(100, tz=UTC)


@pytest.mark.asyncio
@pytest.mark.parametrize("status, expected", [("0x1", True), ("0x0", False)])
async def test_receipt_execution_status(status: str, expected: bool) -> None:
    """Map canonical EVM receipt status without decoding strings in application."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Respond with a receipt carrying the parameterized status."""
        return _reply(
            request,
            {
                "transactionHash": _TX_HASH,
                "blockNumber": "0x1",
                "status": status,
                "gasUsed": "0x0",
                "logs": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        receipt = await AlchemyRpcProvider(client, "test_key").get_transaction_receipt(_TX_HASH)
        assert receipt is not None
        assert receipt.success is expected
        assert receipt.effective_gas_price_wei is None


@pytest.mark.asyncio
async def test_pre_byzantium_receipt_keeps_unknown_success() -> None:
    """Avoid inventing execution status for old root-based receipts."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return legacy receipt with root instead of status."""
        return _reply(
            request,
            {
                "transactionHash": _TX_HASH,
                "blockNumber": "0x1",
                "root": "0x" + "aa" * 32,
                "gasUsed": "0x1",
                "logs": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        receipt = await AlchemyRpcProvider(client, "test_key").get_transaction_receipt(_TX_HASH)
        assert receipt is not None and receipt.success is None


@pytest.mark.asyncio
async def test_eth_call_returns_raw_bytes_without_token_decoder() -> None:
    """Forward raw calldata and return binary output from a read-only call."""
    params: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture eth_call parameters and return raw EVM bytes."""
        body = json.loads(request.content)
        params.extend(body["params"])
        return _reply(request, "0x0001")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        output = await AlchemyRpcProvider(client, "test_key").eth_call(
            _TARGET, bytes.fromhex("70a08231"), at_block=100
        )
        assert output == b"\x00\x01"
        assert params == [{"to": _TARGET, "data": "0x70a08231"}, "0x64"]


@pytest.mark.asyncio
async def test_429_retries_without_real_sleep() -> None:
    """Retry 429 once and stop immediately after successful response."""
    count = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        """Record retry delay without actually sleeping."""
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one rate limit followed by a successful response."""
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429)
        return _reply(request, "0x1")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key", sleep=fake_sleep)
        assert await rpc.get_chain_id() == 1
        assert count == 2
        assert delays == [0.5]


@pytest.mark.asyncio
async def test_timeout_retries_then_exhausts_budget_without_url_leak() -> None:
    """Bound transport timeout retries and omit private API keys from errors."""
    count = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        """Record exponential delays."""
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        """Raise a synthetic timeout on every request."""
        nonlocal count
        count += 1
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "VERY_PRIVATE_KEY", sleep=fake_sleep)
        with pytest.raises(ProviderTimeoutError) as captured:
            await rpc.get_chain_id()
        assert count == 3
        assert delays == [0.5, 1.0]
        assert "VERY_PRIVATE_KEY" not in str(captured.value)


@pytest.mark.asyncio
async def test_bad_schema_not_retried() -> None:
    """Reject bad JSON-RPC response schema without another HTTP attempt."""
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a wrong response ID to trigger schema validation."""
        nonlocal count
        count += 1
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": -4, "result": "0x1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await AlchemyRpcProvider(client, "test_key").get_chain_id()
        assert count == 1


@pytest.mark.asyncio
async def test_json_rpc_rate_limit_error_is_retryable() -> None:
    """Treat Alchemy JSON-RPC rate-limit code as typed retryable failure."""
    count = 0

    async def fake_sleep(delay: float) -> None:
        """Avoid real backoff delays."""
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        """Return JSON-RPC rate limit even though HTTP status is 200."""
        nonlocal count
        count += 1
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "error": {"code": -32005, "message": "rate limited"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderRateLimitError):
            await AlchemyRpcProvider(client, "test_key", sleep=fake_sleep).get_chain_id()
        assert count == 3


@pytest.mark.asyncio
async def test_unauthorized_failure_is_nonretryable() -> None:
    """Reject unauthorized credentials immediately without exposing secrets."""
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an authentication failure."""
        nonlocal count
        count += 1
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError) as err:
            await AlchemyRpcProvider(client, "private_key").get_chain_id()
        assert count == 1
        assert "private_key" not in str(err.value)


@pytest.mark.asyncio
async def test_http_503_retries_then_unavailable() -> None:
    """Turn provider outage responses into retryable typed errors."""
    count = 0

    async def fake_sleep(delay: float) -> None:
        """Avoid real sleep in retry test."""
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        """Always respond with a transient unavailable status."""
        nonlocal count
        count += 1
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderUnavailableError):
            await AlchemyRpcProvider(client, "test_key", sleep=fake_sleep).get_chain_id()
        assert count == 3


@pytest.mark.asyncio
async def test_shared_client_and_bounded_concurrency() -> None:
    """Limit parallel outbound requests while borrowing a shared AsyncClient."""
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        """Track active HTTP requests with deterministic scheduling."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return _reply(request, "0x1")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key", max_concurrency=2)
        values = await asyncio.gather(*(rpc.get_chain_id() for _ in range(6)))
        assert values == [1] * 6
        assert peak <= 2
        assert not client.is_closed
    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["0x00", "10", 123, None, "0xg", "0x01"])
async def test_noncanonical_quantity_is_malformed(response: object) -> None:
    """Reject hex quantities whose encoding is missing or noncanonical."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an invalid eth_chainId result."""
        return _reply(request, response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await AlchemyRpcProvider(client, "test_key").get_chain_id()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["0x2", "0x01", "0x", None])
async def test_invalid_receipt_status_is_rejected(status: object) -> None:
    """Prevent unsupported or malformed execution status from becoming false."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return invalid status with otherwise valid receipt data."""
        return _reply(
            request,
            {
                "transactionHash": _TX_HASH,
                "blockNumber": "0x1",
                "status": status,
                "gasUsed": "0x1",
                "logs": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await AlchemyRpcProvider(client, "test_key").get_transaction_receipt(_TX_HASH)


@pytest.mark.asyncio
async def test_missing_receipt_status_and_root_is_rejected() -> None:
    """Reject an incomplete receipt instead of fabricating execution success."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a receipt missing both recognized execution indicators."""
        return _reply(
            request,
            {
                "transactionHash": _TX_HASH,
                "blockNumber": "0x1",
                "gasUsed": "0x1",
                "logs": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await AlchemyRpcProvider(client, "test_key").get_transaction_receipt(_TX_HASH)

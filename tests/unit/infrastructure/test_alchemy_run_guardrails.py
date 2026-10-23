"""Offline HTTP and cross-adapter integration tests for API request guardrails."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from portfolio_analyzer.application.cached_prices import CachedPriceProvider
from portfolio_analyzer.application.run_guardrails import AnalysisRunStopped, RunGuardrails
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.infrastructure.alchemy.historical_prices import AlchemyPriceProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider

_AT = datetime(2024, 7, 1, 12, 0, tzinfo=UTC)
_ETH = AssetId(1, None)


def _rpc_reply(request: httpx.Request, result: object) -> httpx.Response:
    """Return correctly matched JSON-RPC payload for a mock request."""
    payload = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})


@pytest.mark.asyncio
async def test_guardrails_limit_requests_across_multiple_rpc_operations() -> None:
    """Different RPC methods consume one shared budget on the same transport."""
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        payload = json.loads(request.content)
        return _rpc_reply(request, "0x1" if payload["method"] == "eth_chainId" else "0x2")

    guardrails = RunGuardrails(max_http_attempts=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "FAKE_KEY", guardrails=guardrails)
        assert await rpc.get_chain_id() == 1
        assert await rpc.get_block_number() == 2
        with pytest.raises(AnalysisRunStopped) as exc:
            await rpc.get_block_number()
    assert seen == 2
    assert exc.value.snapshot.http_attempts == 2
    assert [(x.operation, x.attempts) for x in exc.value.snapshot.per_operation] == [
        ("get_block_number", 1),
        ("get_chain_id", 1),
    ]


@pytest.mark.asyncio
async def test_failed_retry_consumes_budget_before_second_request() -> None:
    """Retry and first attempt are both charged against hard cap."""
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        return httpx.Response(429)

    async def instant_sleep(seconds: float) -> None:
        """Avoid sleeping during deterministic retry tests."""

    guardrails = RunGuardrails(max_http_attempts=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(
            client, "FAKE_KEY", guardrails=guardrails, max_attempts=4, sleep=instant_sleep
        )
        with pytest.raises(AnalysisRunStopped) as exc:
            await rpc.get_block_number()
    assert seen == 2
    stats = exc.value.snapshot
    assert stats.http_attempts == 2
    assert stats.retry_attempts == 1
    assert stats.failed_attempts == 2
    assert stats.stop_reason == "request_budget_exhausted"


@pytest.mark.asyncio
async def test_cross_rest_rpc_and_cached_historical_prices_share_budget() -> None:
    """Cache hits do not use HTTP budget, while REST+JSON-RPC both do."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/historical"):
            return httpx.Response(
                200,
                json={
                    "symbol": "ETH",
                    "currency": "usd",
                    "data": [{"timestamp": "2024-07-01T11:59:00Z", "value": "3100"}],
                },
            )
        return _rpc_reply(request, "0x1")

    guardrails = RunGuardrails(max_http_attempts=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "FAKE_KEY", guardrails=guardrails)
        prices = CachedPriceProvider(AlchemyPriceProvider(rpc), guardrails=guardrails)
        assert await rpc.get_chain_id() == 1
        first = await prices.get_historical_price(_ETH, _AT)
        assert first is not None
        assert await prices.get_historical_price(_ETH, _AT) is first
        with pytest.raises(AnalysisRunStopped):
            await prices.get_historical_price(_ETH, _AT.replace(hour=13))
    assert len(seen) == 2
    stats = guardrails.snapshot()
    assert stats.historical_cache_hits == 1
    assert stats.historical_cache_misses == 2
    assert stats.successful_attempts == 2
    assert stats.http_attempts == 2


@pytest.mark.asyncio
async def test_cooperative_stop_does_not_hide_partial_paginated_history() -> None:
    """Stopping during history paging raises a typed stop, never a completed list."""
    from portfolio_analyzer.infrastructure.alchemy.history import AlchemyHistoryProvider

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload["method"])
        if payload["method"] == "eth_blockNumber":
            return _rpc_reply(request, "0xa")
        if payload["method"] == "alchemy_getAssetTransfers":
            return _rpc_reply(request, {"transfers": [], "pageKey": None})
        raise AssertionError("Unexpected method")

    guardrails = RunGuardrails(max_http_attempts=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "FAKE_KEY", guardrails=guardrails)
        history = AlchemyHistoryProvider(rpc)
        with pytest.raises(AnalysisRunStopped):
            await history.get_history("0x" + "cd" * 20)
    assert seen == ["eth_blockNumber", "alchemy_getAssetTransfers"]
    assert guardrails.snapshot().stop_reason == "request_budget_exhausted"
    assert [(row.stage, row.pages_completed) for row in guardrails.snapshot().progress] == [
        ("history_outgoing", 1)
    ]


@pytest.mark.asyncio
async def test_external_task_cancel_retains_usage_snapshot() -> None:
    """Cancellation propagates, but the reserved request remains counted."""
    entered = asyncio.Event()
    forever = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await forever.wait()
        return _rpc_reply(request, "0x1")

    guardrails = RunGuardrails(max_http_attempts=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "FAKE_KEY", guardrails=guardrails)
        task = asyncio.create_task(rpc.get_chain_id())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    stats = guardrails.snapshot()
    assert stats.http_attempts == 1
    assert stats.failed_attempts == 1
    assert stats.in_flight_attempts == 0


@pytest.mark.asyncio
async def test_request_stop_prohibits_subsequent_rpc_call() -> None:
    """An explicit stop is distinguishable from a 429 and not retried."""
    guardrails = RunGuardrails(max_http_attempts=5)
    guardrails.request_stop()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    ) as client:
        rpc = AlchemyRpcProvider(client, "FAKE_KEY", guardrails=guardrails)
        with pytest.raises(AnalysisRunStopped) as exc:
            await rpc.get_chain_id()
    assert exc.value.snapshot.stop_reason == "requested"
    assert exc.value.snapshot.http_attempts == 0

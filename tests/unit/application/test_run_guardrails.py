"""Offline tests for shared per-analysis HTTP attempt budgets and safe stopping."""

import asyncio
import json

import pytest

from portfolio_analyzer.application.run_guardrails import (
    AnalysisRunStopped,
    RunGuardrails,
)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4", None])
def test_request_budget_rejects_invalid_configuration(value: object) -> None:
    """Invalid budgets never silently become unbounded request policies."""
    with pytest.raises((TypeError, ValueError)):
        RunGuardrails(max_http_attempts=value)  # type: ignore[arg-type]


def test_budget_counters_are_immutable_and_not_live() -> None:
    """Snapshots keep their original state after later operations."""
    guardrails = RunGuardrails(max_http_attempts=3)
    old = guardrails.snapshot()
    guardrails.record_historical_cache_access(hit=True)
    guardrails.record_historical_cache_access(hit=False)
    current = guardrails.snapshot()
    assert old.historical_cache_hits == 0
    assert current.historical_cache_hits == 1
    assert current.historical_cache_misses == 1
    with pytest.raises(AttributeError):
        current.http_attempts = 55  # type: ignore[misc]


@pytest.mark.asyncio
async def test_budget_stops_at_hard_limit_and_retains_diagnostics() -> None:
    """The next attempt is rejected before another network call is allowed."""
    guardrails = RunGuardrails(max_http_attempts=2)
    await guardrails.reserve_http_attempt(operation="get_history", is_retry=False)
    guardrails.record_http_completion(success=True)
    await guardrails.reserve_http_attempt(operation="get_history", is_retry=True)
    guardrails.record_http_completion(success=False)
    with pytest.raises(AnalysisRunStopped) as exc:
        await guardrails.reserve_http_attempt(operation="get_historical_price", is_retry=False)
    stats = exc.value.snapshot
    assert stats.http_attempts == 2
    assert stats.retry_attempts == 1
    assert stats.successful_attempts == 1
    assert stats.failed_attempts == 1
    assert stats.in_flight_attempts == 0
    assert stats.stop_reason == "request_budget_exhausted"
    assert [(x.operation, x.attempts) for x in stats.per_operation] == [("get_history", 2)]
    assert guardrails.snapshot() == stats


@pytest.mark.asyncio
async def test_concurrent_reservations_never_overshoot_budget() -> None:
    """An asyncio lock serializes atomic reservations across concurrent providers."""
    guardrails = RunGuardrails(max_http_attempts=7)

    async def reserve_once() -> bool:
        try:
            await guardrails.reserve_http_attempt(operation="get_balance", is_retry=False)
        except AnalysisRunStopped:
            return False
        guardrails.record_http_completion(success=True)
        return True

    results = await asyncio.gather(*(reserve_once() for _ in range(30)))
    assert sum(results) == 7
    assert guardrails.snapshot().http_attempts == 7
    assert guardrails.snapshot().successful_attempts == 7


def test_cooperative_stop_is_idempotent_and_preserves_snapshot() -> None:
    """User-requested stopping has a distinct reason from quota exhaustion."""
    guardrails = RunGuardrails(max_http_attempts=5)
    guardrails.request_stop()
    guardrails.request_stop()
    with pytest.raises(AnalysisRunStopped) as exc:
        guardrails.ensure_active()
    assert exc.value.snapshot.stop_reason == "requested"
    assert exc.value.snapshot.http_attempts == 0
    assert "requested" in str(exc.value)


@pytest.mark.parametrize("operation", ["", "eth_getBalance", "get_balance:secret", " get_balance"])
@pytest.mark.asyncio
async def test_only_safe_operation_identifiers_are_tracked(operation: str) -> None:
    """No raw URL, address or API-key-bearing text may enter snapshots."""
    guardrails = RunGuardrails(max_http_attempts=2)
    with pytest.raises(ValueError):
        await guardrails.reserve_http_attempt(operation=operation, is_retry=False)
    assert guardrails.snapshot().http_attempts == 0


@pytest.mark.parametrize("method", ["completion", "cache_access"])
def test_metrics_reject_non_bool_counters(method: str) -> None:
    """Usage metrics reject truthy integers instead of misclassifying them."""
    guardrails = RunGuardrails(max_http_attempts=1)
    with pytest.raises(TypeError):
        if method == "completion":
            guardrails.record_http_completion(success=1)  # type: ignore[arg-type]
        else:
            guardrails.record_historical_cache_access(hit=1)  # type: ignore[arg-type]


def test_progress_checkpoint_and_safe_json_round_trip() -> None:
    """Validated page counts survive stop and can be saved without identifiers."""
    guardrails = RunGuardrails(max_http_attempts=9)
    guardrails.record_progress(stage="history_outgoing", pages_completed=1, items_seen=10)
    guardrails.record_progress(stage="history_outgoing", pages_completed=2, items_seen=15)
    guardrails.request_stop()
    stats = guardrails.snapshot()
    assert stats.progress[0].pages_completed == 2
    assert stats.progress[0].items_seen == 15
    payload = json.loads(stats.to_json())
    assert payload["stop_reason"] == "requested"
    assert payload["progress"] == [
        {"stage": "history_outgoing", "pages_completed": 2, "items_seen": 15}
    ]
    assert "0x" not in stats.to_json()
    assert "api_key" not in stats.to_json()


@pytest.mark.parametrize(
    "stage,pages,items",
    [("not:safe", 1, 1), ("history_outgoing", -1, 1), ("history_outgoing", 1, -1)],
)
def test_invalid_progress_is_rejected(stage: str, pages: int, items: int) -> None:
    """Invalid diagnostics cannot overwrite verified progress."""
    guardrails = RunGuardrails(max_http_attempts=3)
    with pytest.raises(ValueError):
        guardrails.record_progress(stage=stage, pages_completed=pages, items_seen=items)
    assert guardrails.snapshot().progress == ()


def test_progress_does_not_regress() -> None:
    """Later partial page cannot erase earlier verified pages."""
    guardrails = RunGuardrails(max_http_attempts=5)
    guardrails.record_progress(stage="history_outgoing", pages_completed=2, items_seen=20)
    with pytest.raises(ValueError):
        guardrails.record_progress(stage="history_outgoing", pages_completed=1, items_seen=10)


@pytest.mark.asyncio
async def test_separate_prices_budget_prevents_exceeding_pricing_cap() -> None:
    """An explicit Prices API cap is enforced independently of RPC capacity."""
    guardrails = RunGuardrails(max_http_attempts=8, max_price_api_attempts=2)
    await guardrails.reserve_http_attempt(operation="get_chain_id", is_retry=False)
    guardrails.record_http_completion(success=True)
    await guardrails.reserve_http_attempt(operation="get_current_price", is_retry=False)
    guardrails.record_http_completion(success=True)
    await guardrails.reserve_http_attempt(operation="get_historical_price", is_retry=False)
    guardrails.record_http_completion(success=True)
    with pytest.raises(AnalysisRunStopped) as exc:
        await guardrails.reserve_http_attempt(operation="get_historical_price", is_retry=True)
    assert exc.value.snapshot.stop_reason == "price_api_budget_exhausted"
    assert exc.value.snapshot.http_attempts == 3
    assert exc.value.snapshot.price_api_attempts == 2
    assert exc.value.snapshot.retry_attempts == 0


@pytest.mark.parametrize("price_limit", [0, -1, True, "3", 11])
def test_invalid_prices_budget_rejected(price_limit: object) -> None:
    """A Prices API cap cannot be absent accidentally through invalid configuration."""
    with pytest.raises((TypeError, ValueError)):
        RunGuardrails(max_http_attempts=10, max_price_api_attempts=price_limit)  # type: ignore[arg-type]

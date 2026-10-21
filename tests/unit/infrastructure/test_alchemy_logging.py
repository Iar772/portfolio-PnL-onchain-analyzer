"""Privacy-safe structured logging tests for the existing Alchemy adapters."""

import json
import logging
from collections.abc import Iterator
from io import StringIO
from typing import Any, cast

import httpx
import pytest

from portfolio_analyzer.application.provider_errors import (
    ProviderMalformedResponseError,
    ProviderPartialResultError,
    ProviderTimeoutError,
)
from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.infrastructure.alchemy.history import AlchemyHistoryProvider
from portfolio_analyzer.infrastructure.alchemy.metadata import AlchemyTokenMetadataProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider
from portfolio_analyzer.logging_config import JsonEventFormatter, configure_logging

_WALLET = "0x" + "ab" * 20
_TOKEN = AssetId(1, "0x" + "cd" * 20)
_API_KEY = "DO_NOT_LOG_THIS_API_KEY"


@pytest.fixture
def log_stream() -> Iterator[StringIO]:
    """Configure isolated debug logging and restore package state afterward.

    :return: Captured structured logging output.
    """
    package_logger = logging.getLogger("portfolio_analyzer")
    previous_handlers = tuple(package_logger.handlers)
    previous_level = package_logger.level
    previous_propagate = package_logger.propagate
    for handler in previous_handlers:
        package_logger.removeHandler(handler)
    stream = StringIO()
    configure_logging(AppConfig(log_level="DEBUG"), stream=stream)
    try:
        yield stream
    finally:
        for handler in tuple(package_logger.handlers):
            package_logger.removeHandler(handler)
            handler.close()
        for handler in previous_handlers:
            package_logger.addHandler(handler)
        package_logger.setLevel(previous_level)
        package_logger.propagate = previous_propagate


def _events(stream: StringIO) -> list[dict[str, Any]]:
    """Read structured events from the in-memory stream.

    :param stream: JSON log capture fixture.
    :return: Parsed event payloads.
    """
    return cast(
        list[dict[str, Any]],
        [json.loads(line) for line in stream.getvalue().splitlines()],
    )


def _reply(request: httpx.Request, value: object) -> httpx.Response:
    """Return a properly correlated JSON-RPC success response.

    :param request: Outbound fake request.
    :param value: Controlled JSON-RPC result.
    :return: HTTP response without network activity.
    """
    request_body: dict[str, Any] = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": request_body["id"], "result": value})


def _assert_private_fields_absent(stream: StringIO) -> None:
    """Assert no raw wallet, token, secret, or Alchemy URL leaked.

    :param stream: In-memory formatted logs.
    :raises AssertionError: If private fields appear in output.
    """
    output = stream.getvalue().lower()
    for secret in (_API_KEY, _WALLET, _TOKEN.contract_address, "eth-mainnet.g.alchemy.com"):
        if secret is not None:
            assert secret.lower() not in output


@pytest.mark.asyncio
async def test_rpc_success_records_duration_and_attempt_without_secret(
    log_stream: StringIO,
) -> None:
    """Record successful request duration only using allowlisted scalar fields."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: _reply(request, "0x1"))
    ) as client:
        result = await AlchemyRpcProvider(client, _API_KEY).get_chain_id()
    assert result == 1
    events = _events(log_stream)
    assert [entry["event"] for entry in events] == [
        "provider_request_started",
        "provider_request_completed",
    ]
    completed = events[1]
    assert completed["provider"] == "alchemy"
    assert completed["operation"] == "get_chain_id"
    assert completed["attempt"] == 1
    assert type(completed["duration_ms"]) is int and completed["duration_ms"] >= 0
    assert set(completed) == {
        "timestamp",
        "level",
        "logger",
        "event",
        "provider",
        "operation",
        "attempt",
        "duration_ms",
    }
    _assert_private_fields_absent(log_stream)


@pytest.mark.asyncio
async def test_rate_limit_warning_classifies_retry_without_vendor_details(
    log_stream: StringIO,
) -> None:
    """Expose typed retry cause, never raw response or private URL."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return 429 then a valid chain id."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text=f"secret {_API_KEY}")
        return _reply(request, "0x1")

    async def no_sleep(_: float) -> None:
        """Skip real backoff in unit tests."""

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await AlchemyRpcProvider(client, _API_KEY, sleep=no_sleep).get_chain_id() == 1

    assert calls == 2
    events = _events(log_stream)
    retries = [item for item in events if item["event"] == "provider_request_retry"]
    assert len(retries) == 1
    assert retries[0]["failure_kind"] == "rate_limit"
    assert retries[0]["attempt"] == 1
    assert retries[0]["level"] == "WARNING"
    assert not any(item["event"] == "provider_request_failed" for item in events)
    _assert_private_fields_absent(log_stream)


@pytest.mark.asyncio
async def test_timeout_logs_bounded_retries_and_final_failure(log_stream: StringIO) -> None:
    """Describe timeout retries without logging exception messages or URLs."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Raise synthetic timeout that embeds the secret in its message."""
        raise httpx.ReadTimeout(f"timeout request={_API_KEY}", request=request)

    async def no_sleep(_: float) -> None:
        """Prevent actual retries from sleeping."""

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderTimeoutError):
            await AlchemyRpcProvider(client, _API_KEY, sleep=no_sleep).get_block_number()

    retry_and_failed = [
        item
        for item in _events(log_stream)
        if item["event"] in ("provider_request_retry", "provider_request_failed")
    ]
    assert [(entry["event"], entry["attempt"]) for entry in retry_and_failed] == [
        ("provider_request_retry", 1),
        ("provider_request_retry", 2),
        ("provider_request_failed", 3),
    ]
    assert all(entry["failure_kind"] == "timeout" for entry in retry_and_failed)
    _assert_private_fields_absent(log_stream)


@pytest.mark.asyncio
async def test_malformed_response_is_a_final_error_not_a_retry(log_stream: StringIO) -> None:
    """Keep parsing failures distinguishable from temporary transport failures."""

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="not-json"))
    ) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await AlchemyRpcProvider(client, _API_KEY).get_chain_id()

    events = _events(log_stream)
    failed = [item for item in events if item["event"] == "provider_request_failed"]
    assert len(failed) == 1
    assert failed[0]["failure_kind"] == "malformed_response"
    assert not any(item["event"] == "provider_request_retry" for item in events)
    _assert_private_fields_absent(log_stream)


@pytest.mark.asyncio
async def test_history_reports_completed_directions_and_unique_total(log_stream: StringIO) -> None:
    """Count paginated movements without recording wallet addresses or cursors."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Respond to frozen block and both complete empty transfer directions."""
        body: dict[str, Any] = json.loads(request.content)
        result = "0x20" if body["method"] == "eth_blockNumber" else {"transfers": []}
        return _reply(request, result)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        history = await AlchemyHistoryProvider(AlchemyRpcProvider(client, _API_KEY)).get_history(
            _WALLET
        )
    assert history == []
    events = _events(log_stream)
    completed = [item for item in events if item["event"] == "history_direction_completed"]
    assert [entry["direction"] for entry in completed] == ["outgoing", "incoming"]
    assert [(entry["page_count"], entry["movement_count"]) for entry in completed] == [
        (1, 0),
        (1, 0),
    ]
    total = [item for item in events if item["event"] == "history_completed"]
    assert len(total) == 1 and total[0]["movement_count"] == 0
    _assert_private_fields_absent(log_stream)


@pytest.mark.asyncio
async def test_history_partial_result_never_logs_history_completed(log_stream: StringIO) -> None:
    """Distinguish incomplete history from successful analysis even with a page key."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an inconsistent empty page with a continuation cursor."""
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "eth_blockNumber":
            return _reply(request, "0x20")
        return _reply(request, {"transfers": [], "pageKey": f"secret-cursor-{_API_KEY}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderPartialResultError):
            await AlchemyHistoryProvider(AlchemyRpcProvider(client, _API_KEY)).get_history(_WALLET)

    events = _events(log_stream)
    incomplete = [item for item in events if item["event"] == "history_incomplete"]
    assert len(incomplete) == 1 and incomplete[0]["direction"] == "outgoing"
    assert not any(item["event"] == "history_completed" for item in events)
    _assert_private_fields_absent(log_stream)


@pytest.mark.asyncio
async def test_metadata_absence_is_observable_without_changing_identity(
    log_stream: StringIO,
) -> None:
    """Log missing display fields without recording token address or inferred decimals."""

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: _reply(request, {"decimals": None}))
    ) as client:
        metadata = await AlchemyTokenMetadataProvider(
            AlchemyRpcProvider(client, _API_KEY)
        ).get_metadata(_TOKEN)
    assert metadata is None
    events = _events(log_stream)
    assert len([entry for entry in events if entry["event"] == "token_metadata_unavailable"]) == 1
    _assert_private_fields_absent(log_stream)


@pytest.mark.asyncio
async def test_invalid_metadata_emits_warning_and_raises(log_stream: StringIO) -> None:
    """Log invalid metadata separately from unavailable metadata."""

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: _reply(request, {"name": "Token", "symbol": "TKN", "decimals": -1})
        )
    ) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await AlchemyTokenMetadataProvider(AlchemyRpcProvider(client, _API_KEY)).get_metadata(
                _TOKEN
            )

    events = _events(log_stream)
    assert [item["event"] for item in events if item["level"] == "WARNING"] == [
        "token_metadata_invalid"
    ]
    _assert_private_fields_absent(log_stream)


def test_formatter_sanitizes_new_telemetry_dimensions() -> None:
    """Preserve only documented fields with safe types and identifiers."""
    safe = logging.LogRecord(
        "portfolio_analyzer", logging.INFO, __file__, 1, "history_completed", (), None
    )
    safe.__dict__.update(
        direction="outgoing",
        failure_kind="rate_limit",
        page_count=2,
        movement_count=0,
        wallet_address=_WALLET,
        api_key=_API_KEY,
        page_key="secret-cursor",
    )
    payload = json.loads(JsonEventFormatter().format(safe))
    assert payload["direction"] == "outgoing"
    assert payload["failure_kind"] == "rate_limit"
    assert payload["page_count"] == 2
    assert payload["movement_count"] == 0
    assert all(name not in payload for name in ("wallet_address", "api_key", "page_key"))

    unsafe = logging.LogRecord(
        "portfolio_analyzer", logging.INFO, __file__, 1, "history_incomplete", (), None
    )
    unsafe.__dict__.update(
        direction=f"outgoing {_API_KEY}",
        failure_kind="RATE_LIMIT",
        page_count=True,
        movement_count=-10,
    )
    filtered = json.loads(JsonEventFormatter().format(unsafe))
    assert all(
        name not in filtered
        for name in ("direction", "failure_kind", "page_count", "movement_count")
    )


@pytest.mark.asyncio
async def test_info_level_filters_successful_rpc_debug_events(log_stream: StringIO) -> None:
    """Avoid noisy per-request success messages under production INFO level."""
    configure_logging(AppConfig(log_level="INFO"), stream=log_stream)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: _reply(request, "0x1"))
    ) as client:
        assert await AlchemyRpcProvider(client, _API_KEY).get_chain_id() == 1
    assert log_stream.getvalue() == ""

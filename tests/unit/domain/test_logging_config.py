"""Deterministic tests for privacy-conscious JSON event logging."""

import json
import logging
from collections.abc import Iterator
from io import StringIO
from typing import cast

import pytest

from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.logging_config import JsonEventFormatter, configure_logging


@pytest.fixture
def package_logger() -> Iterator[logging.Logger]:
    """Isolate application logger state from other tests.

    :return: The package logger restored to its original state afterward.
    """
    logger = logging.getLogger("portfolio_analyzer")
    original_handlers = tuple(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    for handler in original_handlers:
        logger.removeHandler(handler)

    try:
        yield logger
    finally:
        for handler in tuple(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def _make_record(event: str, **extra: object) -> logging.LogRecord:
    """Create a deterministic record without touching the logging system.

    :param event: Candidate structured event name.
    :param extra: Extra record attributes supplied by application callers.
    :return: A record suitable for directly testing the formatter.
    """
    record = logging.LogRecord("portfolio_analyzer", logging.INFO, __file__, 1, event, (), None)
    record.created = 1_700_000_000.125
    record.__dict__.update(extra)
    return record


def _format_json(record: logging.LogRecord) -> dict[str, str | int]:
    """Parse a structured event for assertions.

    :param record: Input record for the formatter.
    :return: Parsed single-line JSON object.
    """
    return cast(dict[str, str | int], json.loads(JsonEventFormatter().format(record)))


def test_formatter_emits_utc_timestamp_and_fixed_keys() -> None:
    """Use UTC and stable record properties in a single-line JSON log."""
    payload = _format_json(_make_record("wallet_analysis_started"))

    assert payload == {
        "timestamp": "2023-11-14T22:13:20.125Z",
        "level": "INFO",
        "logger": "portfolio_analyzer",
        "event": "wallet_analysis_started",
    }


def test_formatter_includes_allowlisted_provider_fields() -> None:
    """Preserve safe provider identifiers for future request diagnostics."""
    payload = _format_json(
        _make_record(
            "provider_request_completed", provider="alchemy", operation="get_asset_transfers"
        )
    )
    assert payload["provider"] == "alchemy"
    assert payload["operation"] == "get_asset_transfers"


def test_formatter_includes_non_negative_integer_metrics() -> None:
    """Preserve numeric operation metrics without losing zero values."""
    payload = _format_json(_make_record("provider_request_completed", duration_ms=0, attempt=2))
    assert payload["duration_ms"] == 0
    assert payload["attempt"] == 2


@pytest.mark.parametrize("invalid", [True, -1, 1.2, "20", None])
def test_formatter_omits_invalid_numeric_metrics(invalid: object) -> None:
    """Avoid corrupt metrics from incorrectly typed logging context.

    :param invalid: Invalid duration and retry-attempt field value.
    """
    payload = _format_json(
        _make_record("provider_request_completed", duration_ms=invalid, attempt=invalid)
    )
    assert "duration_ms" not in payload
    assert "attempt" not in payload


def test_formatter_does_not_serialize_arbitrary_sensitive_extra() -> None:
    """Drop non-allowlisted attributes such as secrets and wallet identifiers."""
    raw_log = JsonEventFormatter().format(
        _make_record(
            "provider_request_completed",
            api_key="secret-api-key",
            authorization="Bearer secret-token",
            wallet_address="0x" + "a" * 40,
            provider="alchemy",
        )
    )
    assert "secret-api-key" not in raw_log
    assert "secret-token" not in raw_log
    assert "wallet_address" not in raw_log
    assert json.loads(raw_log)["provider"] == "alchemy"


def test_formatter_masks_free_text_event_names() -> None:
    """Do not copy arbitrary message text into event-name fields."""
    payload = _format_json(_make_record("request failed: api_key=my-secret"))
    assert payload["event"] == "invalid_event_name"


def test_formatter_omits_unsafe_provider_context() -> None:
    """Reject context values that do not look like safe identifiers."""
    payload = _format_json(
        _make_record("provider_request_completed", provider="account 0x123", operation="fetch")
    )
    assert "provider" not in payload
    assert payload["operation"] == "fetch"


def test_configure_logging_is_idempotent(package_logger: logging.Logger) -> None:
    """Prevent duplicate handlers from duplicating structured events.

    :param package_logger: Isolated application logger fixture.
    """
    stream = StringIO()
    first = configure_logging(AppConfig(log_level="DEBUG"), stream=stream)
    first.info("analysis_started")

    second = configure_logging(AppConfig(log_level="INFO"), stream=stream)
    second.info("analysis_completed")

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["event"] for line in lines] == [
        "analysis_started",
        "analysis_completed",
    ]
    assert package_logger is second
    assert len(package_logger.handlers) == 1


def test_configure_logging_does_not_change_root_logger(package_logger: logging.Logger) -> None:
    """Configure only our package logger, leaving third-party logging alone.

    :param package_logger: Isolated application logger fixture.
    """
    root = logging.getLogger()
    root_level_before = root.level
    root_handlers_before = tuple(root.handlers)

    configure_logging(AppConfig(log_level="WARNING"), stream=StringIO())

    assert root.level == root_level_before
    assert tuple(root.handlers) == root_handlers_before
    assert package_logger.level == logging.WARNING
    assert not package_logger.propagate


def test_configure_logging_rejects_invalid_config() -> None:
    """Reject unvalidated configuration objects at the logging boundary."""
    with pytest.raises(TypeError, match="AppConfig"):
        configure_logging(None)  # type: ignore[arg-type]

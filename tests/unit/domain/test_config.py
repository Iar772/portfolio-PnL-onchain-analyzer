"""Offline tests for the centralized application configuration."""

from typing import cast

import pytest

from portfolio_analyzer.config import AppConfig, load_config


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("debug", "DEBUG"),
        (" INFO ", "INFO"),
        ("Warning", "WARNING"),
        ("ERROR", "ERROR"),
        ("critical", "CRITICAL"),
    ],
)
def test_config_normalizes_valid_log_levels(requested: str, expected: str) -> None:
    """Normalize supported log levels without changing the original input.

    :param requested: User-provided log level spelling.
    :param expected: Canonical uppercase logging level.
    """
    assert AppConfig(log_level=requested).log_level == expected


@pytest.mark.parametrize("requested", ["", " ", "TRACE", "INFOO", "OFF"])
def test_config_rejects_unsupported_log_levels(requested: str) -> None:
    """Reject missing and unknown logging levels.

    :param requested: Unsupported LOG_LEVEL value.
    """
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        AppConfig(log_level=requested)


@pytest.mark.parametrize("requested", [None, 10, True, 1.5])
def test_config_rejects_non_string_log_levels(requested: object) -> None:
    """Prevent unexpected config types from entering the logger.

    :param requested: Non-string LOG_LEVEL value.
    """
    with pytest.raises(TypeError, match="LOG_LEVEL"):
        AppConfig(log_level=cast(str, requested))


def test_load_config_uses_default_when_level_missing() -> None:
    """Use INFO without requiring any external environment setup."""
    assert load_config({}) == AppConfig(log_level="INFO")


def test_load_config_uses_explicit_mapping() -> None:
    """Read only the supplied environment mapping for deterministic tests."""
    assert load_config({"LOG_LEVEL": " debug "}).log_level == "DEBUG"


def test_load_config_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read process environment only at the centralized config boundary.

    :param monkeypatch: pytest fixture for restoring environment state.
    """
    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert load_config().log_level == "WARNING"

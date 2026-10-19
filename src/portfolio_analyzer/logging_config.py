"""Minimal privacy-conscious JSON event logging for the Engine MVP."""

import json
import logging
import re
from datetime import UTC, datetime
from typing import TextIO

from portfolio_analyzer.config import AppConfig

_EVENT_NAME_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


class JsonEventFormatter(logging.Formatter):
    """Serialize controlled event names and allowlisted fields as JSON.

    Callers must use static event identifiers rather than free-text messages.
    Unsupported event names are replaced instead of being copied into logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a single log record without arbitrary extra attributes.

        :param record: Python logging record emitted by application code.
        :return: A compact, single-line JSON object with a UTC timestamp.
        """
        event_name = record.getMessage()
        if _EVENT_NAME_RE.fullmatch(event_name) is None:
            event_name = "invalid_event_name"

        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        payload: dict[str, str | int] = {
            "timestamp": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": event_name,
        }

        for field_name in ("provider", "operation"):
            field_value = getattr(record, field_name, None)
            if isinstance(field_value, str) and _EVENT_NAME_RE.fullmatch(field_value):
                payload[field_name] = field_value

        for field_name in ("duration_ms", "attempt"):
            field_value = getattr(record, field_name, None)
            if type(field_value) is not int or field_value is None or field_value < 0:
                continue
            payload[field_name] = field_value

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def configure_logging(config: AppConfig, *, stream: TextIO | None = None) -> logging.Logger:
    """Configure only the application's logger, without altering root logging.

    Repeated calls replace handlers owned by the package logger so records
    are not duplicated. This is called from the future composition root.

    :param config: Validated application configuration.
    :param stream: Optional output stream, useful for deterministic tests.
    :return: The configured portfolio_analyzer package logger.
    :raises TypeError: If config has an unexpected type.
    """
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")

    logger = logging.getLogger("portfolio_analyzer")
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)
    logger.setLevel(config.log_level)
    logger.propagate = False
    return logger

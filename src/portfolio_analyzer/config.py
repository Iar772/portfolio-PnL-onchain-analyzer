"""Centralized environment configuration for the Engine MVP."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

_ALLOWED_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Hold the application settings needed by the current engine stage.

    :param log_level: Case-insensitive standard Python logging level.
    :raises TypeError: If the log level is not a string.
    :raises ValueError: If the log level is unsupported.
    """

    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Normalize and validate the application logging level.

        :raises TypeError: If log_level is not a string.
        :raises ValueError: If log_level is empty or unsupported.
        """
        if not isinstance(self.log_level, str):
            raise TypeError("LOG_LEVEL must be a string")
        normalized_log_level = self.log_level.upper().strip()

        if normalized_log_level not in _ALLOWED_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(sorted(_ALLOWED_LOG_LEVELS))}")

        object.__setattr__(self, "log_level", normalized_log_level)


def load_config(environ: Mapping[str, str] | None = None) -> AppConfig:
    """Read supported settings from one environment boundary.

    Only LOG_LEVEL is read in E1.5. Alchemy keys are introduced when the
    Alchemy adapter is implemented; .env files are not parsed automatically.

    :param environ: Optional environment mapping for deterministic tests.
    :return: Validated immutable application configuration.
    :raises TypeError: If a configuration value has an invalid type.
    :raises ValueError: If a configuration value is unsupported.
    """
    source = os.environ if environ is None else environ
    return AppConfig(log_level=source.get("LOG_LEVEL", "INFO"))

"""Vendor-neutral provider failures for application orchestration."""

import re

# Only controlled identifiers appear in public error messages; never embed vendor payloads.
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


class ProviderError(Exception):
    """Represent an external provider failure without exposing vendor details.

    :param provider: Safe internal provider identifier, such as ``alchemy``.
    :param operation: Safe internal operation name, such as ``get_history``.
    :raises TypeError: If an identifier is not a string.
    :raises ValueError: If an identifier is not a safe, nonempty name.
    """

    def __init__(self, provider: str, operation: str) -> None:
        """Store structured context and a deterministic diagnostic message.

        :param provider: Configured provider name, not a URL or API key.
        :param operation: Internal method name, not a raw request body.
        :raises TypeError: If either identifier is not a string.
        :raises ValueError: If either identifier is unsafe or blank.
        """
        for field_name, value in (("provider", provider), ("operation", operation)):
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if _IDENTIFIER_RE.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a safe identifier")

        self.provider = provider
        self.operation = operation
        super().__init__(f"{type(self).__name__}: {provider}.{operation}")


class ProviderTimeoutError(ProviderError):
    """Report that a provider request exceeded its configured time limit."""


class ProviderRateLimitError(ProviderError):
    """Report that a provider rejected a request due to rate limiting."""


class ProviderUnavailableError(ProviderError):
    """Report that a provider is temporarily unreachable or unavailable."""


class ProviderMalformedResponseError(ProviderError):
    """Report a response that does not satisfy the expected data contract."""


class ProviderPartialResultError(ProviderError):
    """Reject incomplete provider data rather than treating it as full history."""


class ProviderNotFoundError(ProviderError):
    """Represent a resource that was not found by the external provider."""


class PriceNotFoundError(ProviderNotFoundError):
    """Represent a price lookup that explicitly reported no available price."""


class AssetMetadataNotFoundError(ProviderNotFoundError):
    """Represent a metadata lookup that explicitly reported no asset metadata."""


def should_retry_provider_error(
    error: ProviderError, *, attempts_made: int, max_attempts: int
) -> bool:
    """Decide whether a provider operation is eligible for another attempt.

    This classifies failures only. Delays, backoff, and retries themselves
    are the responsibility of the infrastructure adapter in Epic 4.

    :param error: Normalized provider failure.
    :param attempts_made: Number of attempts already completed, starting at 1.
    :param max_attempts: Maximum total attempts allowed, including the first.
    :return: Whether another attempt is allowed for this failure and budget.
    :raises TypeError: If the error or attempt counters have invalid types.
    :raises ValueError: If the attempt counters are outside the allowed range.
    """
    if not isinstance(error, ProviderError):
        raise TypeError("error must be a ProviderError")
    if type(attempts_made) is not int or type(max_attempts) is not int:
        raise TypeError("attempt counters must be integers")
    if max_attempts < 1 or attempts_made < 1:
        raise ValueError("attempt counters must be positive")

    return attempts_made < max_attempts and isinstance(
        error,
        (
            ProviderTimeoutError,
            ProviderRateLimitError,
            ProviderUnavailableError,
        ),
    )

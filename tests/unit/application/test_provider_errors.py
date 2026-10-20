"""Offline contract tests for normalized provider failures and retry eligibility."""

from typing import cast

import pytest

from portfolio_analyzer.application.provider_errors import (
    AssetMetadataNotFoundError,
    PriceNotFoundError,
    ProviderError,
    ProviderMalformedResponseError,
    ProviderNotFoundError,
    ProviderPartialResultError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    should_retry_provider_error,
)


@pytest.mark.parametrize(
    "error_type",
    [
        ProviderTimeoutError,
        ProviderRateLimitError,
        ProviderUnavailableError,
        ProviderMalformedResponseError,
        ProviderPartialResultError,
        ProviderNotFoundError,
        PriceNotFoundError,
        AssetMetadataNotFoundError,
    ],
)
def test_all_provider_errors_have_structured_context(error_type: type[ProviderError]) -> None:
    """Make every provider error catchable without parsing external messages.

    :param error_type: Provider error subclass to construct.
    """
    error = error_type("alchemy", "get_history")

    assert isinstance(error, ProviderError)
    assert error.provider == "alchemy"
    assert error.operation == "get_history"
    assert str(error) == f"{error_type.__name__}: alchemy.get_history"


def test_not_found_errors_share_a_specific_parent() -> None:
    """Allow the application to handle missing prices and metadata together."""
    assert isinstance(PriceNotFoundError("alchemy", "get_price"), ProviderNotFoundError)
    assert isinstance(AssetMetadataNotFoundError("alchemy", "get_metadata"), ProviderNotFoundError)


@pytest.mark.parametrize("field", ["provider", "operation"])
@pytest.mark.parametrize("bad_value", ["", " ", "ALCHEMY", "https://example.com/key", "a-b"])
def test_provider_context_rejects_unsafe_identifiers(field: str, bad_value: str) -> None:
    """Prevent URLs, secrets and arbitrary vendor text from entering error messages.

    :param field: Structured error field under test.
    :param bad_value: Unsafe identifier to reject.
    """
    provider = bad_value if field == "provider" else "alchemy"
    operation = bad_value if field == "operation" else "get_history"

    with pytest.raises(ValueError, match="safe identifier"):
        ProviderUnavailableError(provider, operation)


@pytest.mark.parametrize("bad_value", [None, 123, True])
def test_provider_context_rejects_non_string_values(bad_value: object) -> None:
    """Reject invalid provider identifiers at the application boundary.

    :param bad_value: Wrong-typed identifier to reject.
    """
    with pytest.raises(TypeError, match="must be a string"):
        ProviderTimeoutError(cast(str, bad_value), "get_history")


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        (ProviderTimeoutError, True),
        (ProviderRateLimitError, True),
        (ProviderUnavailableError, True),
        (ProviderMalformedResponseError, False),
        (ProviderPartialResultError, False),
        (ProviderNotFoundError, False),
        (PriceNotFoundError, False),
        (AssetMetadataNotFoundError, False),
    ],
)
def test_retry_classification(error_type: type[ProviderError], expected: bool) -> None:
    """Classify transient failures without relying on a vendor response string.

    :param error_type: Failure category to classify.
    :param expected: Whether one more attempt is appropriate.
    """
    assert (
        should_retry_provider_error(
            error_type("alchemy", "get_history"), attempts_made=1, max_attempts=3
        )
        is expected
    )


@pytest.mark.parametrize("attempts_made", [3, 4])
def test_retry_stops_when_budget_is_exhausted(attempts_made: int) -> None:
    """Reject further attempts once the configured attempt budget is spent.

    :param attempts_made: Number of completed tries.
    """
    error = ProviderRateLimitError("alchemy", "get_history")

    assert not should_retry_provider_error(error, attempts_made=attempts_made, max_attempts=3)


def test_retry_allows_a_second_attempt_with_two_total_attempts() -> None:
    """Treat max_attempts as the total number of calls, not retries."""
    error = ProviderTimeoutError("alchemy", "get_history")

    assert should_retry_provider_error(error, attempts_made=1, max_attempts=2)
    assert not should_retry_provider_error(error, attempts_made=2, max_attempts=2)


@pytest.mark.parametrize("bad_value", [True, 1.1, "2", None])
def test_retry_rejects_invalid_attempt_types(bad_value: object) -> None:
    """Reject booleans and non-integers in retry counters.

    :param bad_value: Incorrect attempt-counter value.
    """
    error = ProviderUnavailableError("alchemy", "get_history")
    with pytest.raises(TypeError, match="attempt counters must be integers"):
        should_retry_provider_error(error, attempts_made=cast(int, bad_value), max_attempts=3)


@pytest.mark.parametrize(("attempts_made", "max_attempts"), [(0, 3), (1, 0), (-1, 3)])
def test_retry_rejects_nonpositive_attempt_limits(attempts_made: int, max_attempts: int) -> None:
    """Enforce a positive, one-based retry budget.

    :param attempts_made: Completed attempts, which must be positive.
    :param max_attempts: Total allowed calls, which must be positive.
    """
    error = ProviderTimeoutError("alchemy", "get_history")
    with pytest.raises(ValueError, match="attempt counters must be positive"):
        should_retry_provider_error(error, attempts_made=attempts_made, max_attempts=max_attempts)


def test_retry_rejects_non_provider_exception() -> None:
    """Avoid classifying transport exceptions outside the normalized boundary."""
    with pytest.raises(TypeError, match="error must be a ProviderError"):
        should_retry_provider_error(
            cast(ProviderError, ValueError("vendor error")), attempts_made=1, max_attempts=3
        )

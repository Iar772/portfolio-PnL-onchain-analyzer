"""Per-analysis request budgets, safe stopping, and in-memory usage diagnostics.

One guardrails instance belongs to one analysis run, not to a vendor account.
Request counts are local observations, not Alchemy Compute Units or dollars.
"""

import asyncio
import json
import re
from dataclasses import dataclass

_SAFE_OPERATION = re.compile(r"[a-z][a-z0-9_]*\Z")
_PRICE_OPERATIONS = frozenset({"get_current_price", "get_historical_price"})


@dataclass(frozen=True, slots=True)
class OperationUsage:
    """Summarize attempted network requests for one safe operation name.

    :param operation: Internal, privacy-safe operation identifier.
    :param attempts: Attempts sent or reserved for this operation.
    """

    operation: str
    attempts: int


@dataclass(frozen=True, slots=True)
class RunProgress:
    """Retain partial but validated processing progress without wallet data.

    :param stage: Controlled stage identifier such as history_outgoing.
    :param pages_completed: Number of validated API pages so far.
    :param items_seen: Number of validated items in those pages.
    """

    stage: str
    pages_completed: int
    items_seen: int


@dataclass(frozen=True, slots=True)
class RunUsageSnapshot:
    """Hold an immutable diagnostic snapshot for an analysis attempt.

    Counts never contain wallet addresses, API keys, request bodies or token IDs.
    An attempt is reserved immediately before sending and includes retries.

    :param max_http_attempts: Configured maximum outgoing HTTP attempts.
    :param max_price_api_attempts: Optional dedicated Prices API attempt budget.
    :param price_api_attempts: Reserved Prices API attempts within the run.
    :param http_attempts: Number of reserved attempts across all adapters.
    :param successful_attempts: Attempts returning a valid HTTP/provider response.
    :param failed_attempts: Attempts that failed or were cancelled after reservation.
    :param retry_attempts: Reserved attempts beyond an operation's initial try.
    :param historical_cache_hits: Hits for historical quote lookups (including cached None).
    :param historical_cache_misses: Historical quote fetches attempted by the cache.
    :param per_operation: Immutable and sorted safe per-operation attempt counts.
    :param stop_reason: Explicit reason, or None while the run is active.
    :param progress: Immutable validated checkpoints for partially completed stages.
    """

    max_http_attempts: int
    max_price_api_attempts: int | None
    http_attempts: int
    price_api_attempts: int
    successful_attempts: int
    failed_attempts: int
    retry_attempts: int
    historical_cache_hits: int
    historical_cache_misses: int
    per_operation: tuple[OperationUsage, ...]
    stop_reason: str | None
    progress: tuple[RunProgress, ...]

    def to_json(self) -> str:
        """Encode safe diagnostic counters for later CLI/report persistence.

        :return: JSON containing only allowlisted usage and progress fields.
        """
        return json.dumps(
            {
                "max_http_attempts": self.max_http_attempts,
                "max_price_api_attempts": self.max_price_api_attempts,
                "price_api_attempts": self.price_api_attempts,
                "http_attempts": self.http_attempts,
                "successful_attempts": self.successful_attempts,
                "failed_attempts": self.failed_attempts,
                "retry_attempts": self.retry_attempts,
                "historical_cache_hits": self.historical_cache_hits,
                "historical_cache_misses": self.historical_cache_misses,
                "stop_reason": self.stop_reason,
                "per_operation": [
                    {"operation": row.operation, "attempts": row.attempts}
                    for row in self.per_operation
                ],
                "progress": [
                    {
                        "stage": row.stage,
                        "pages_completed": row.pages_completed,
                        "items_seen": row.items_seen,
                    }
                    for row in self.progress
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def in_flight_attempts(self) -> int:
        """Return HTTP attempts reserved but not yet completed.

        :return: Nonnegative count of pending HTTP attempts.
        """
        return self.http_attempts - self.successful_attempts - self.failed_attempts


class AnalysisRunStopped(Exception):
    """Stop safely with diagnostics instead of returning an incomplete analysis.

    :param snapshot: Immutable usage statistics captured at the stop decision.
    """

    def __init__(self, snapshot: RunUsageSnapshot) -> None:
        """Expose the stop reason and safe diagnostic snapshot.

        :param snapshot: Captured counters without wallet identifiers.
        """
        self.snapshot = snapshot
        super().__init__(f"Analysis stopped: {snapshot.stop_reason}")


class RunGuardrails:
    """Apply a shared request cap and cooperative stop to one analysis.

    The cap covers HTTP attempts, not logical calls. Reserving happens under
    an asyncio lock, preventing concurrent requests from overshooting the cap.

    :param max_http_attempts: Maximum total attempts for this analysis, including retries.
    :param max_price_api_attempts: Optional, smaller cap for current/historical Prices API calls.
    :raises TypeError: If the budget is not an integer.
    :raises ValueError: If the budget is not positive.
    """

    def __init__(
        self, *, max_http_attempts: int, max_price_api_attempts: int | None = None
    ) -> None:
        """Initialize counters scoped to one run.

        :param max_http_attempts: Positive outbound HTTP attempt budget.
        :param max_price_api_attempts: Optional cap for Prices API requests.
        """
        if type(max_http_attempts) is not int:
            raise TypeError("max_http_attempts must be an integer")
        if max_http_attempts < 1:
            raise ValueError("max_http_attempts must be positive")
        if max_price_api_attempts is not None:
            if type(max_price_api_attempts) is not int:
                raise TypeError("max_price_api_attempts must be an integer")
            if max_price_api_attempts < 1 or max_price_api_attempts > max_http_attempts:
                raise ValueError("max_price_api_attempts must be between 1 and max_http_attempts")
        self._max_http_attempts = max_http_attempts
        self._max_price_api_attempts = max_price_api_attempts
        self._price_api_attempts = 0
        self._lock = asyncio.Lock()
        self._http_attempts = 0
        self._successful_attempts = 0
        self._failed_attempts = 0
        self._retry_attempts = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._per_operation: dict[str, int] = {}
        self._progress: dict[str, RunProgress] = {}
        self._stop_reason: str | None = None

    def snapshot(self) -> RunUsageSnapshot:
        """Read a stable copy of the currently observable counters.

        :return: Immutable diagnostics useful after stop, failure or success.
        """
        return RunUsageSnapshot(
            max_http_attempts=self._max_http_attempts,
            max_price_api_attempts=self._max_price_api_attempts,
            price_api_attempts=self._price_api_attempts,
            http_attempts=self._http_attempts,
            successful_attempts=self._successful_attempts,
            failed_attempts=self._failed_attempts,
            retry_attempts=self._retry_attempts,
            historical_cache_hits=self._cache_hits,
            historical_cache_misses=self._cache_misses,
            per_operation=tuple(
                OperationUsage(operation=operation, attempts=attempts)
                for operation, attempts in sorted(self._per_operation.items())
            ),
            stop_reason=self._stop_reason,
            progress=tuple(self._progress[k] for k in sorted(self._progress)),
        )

    def request_stop(self) -> None:
        """Ask the current analysis to stop before its next outbound attempt.

        An in-flight HTTP attempt may finish; the next cache lookup or HTTP
        reservation rejects further work without returning partial results.
        """
        if self._stop_reason is None:
            self._stop_reason = "requested"

    def ensure_active(self) -> None:
        """Reject a run after cooperative stop, including cached lookups.

        :raises AnalysisRunStopped: When the run has been stopped.
        """
        if self._stop_reason is not None:
            raise AnalysisRunStopped(self.snapshot())

    async def reserve_http_attempt(self, *, operation: str, is_retry: bool) -> None:
        """Atomically reserve exactly one outgoing attempt against the budget.

        :param operation: Safe logical operation identifier, not an endpoint or wallet.
        :param is_retry: Whether this HTTP attempt follows a failed attempt.
        :raises ValueError: For unsafe operation names.
        :raises AnalysisRunStopped: When stopped or the request cap is reached.
        """
        if not isinstance(operation, str) or _SAFE_OPERATION.fullmatch(operation) is None:
            raise ValueError("operation must be a safe identifier")
        if type(is_retry) is not bool:
            raise TypeError("is_retry must be bool")
        async with self._lock:
            self.ensure_active()
            if self._http_attempts >= self._max_http_attempts:
                self._stop_reason = "request_budget_exhausted"
                raise AnalysisRunStopped(self.snapshot())
            if (
                operation in _PRICE_OPERATIONS
                and self._max_price_api_attempts is not None
                and self._price_api_attempts >= self._max_price_api_attempts
            ):
                self._stop_reason = "price_api_budget_exhausted"
                raise AnalysisRunStopped(self.snapshot())
            self._http_attempts += 1
            if operation in _PRICE_OPERATIONS:
                self._price_api_attempts += 1
            self._per_operation[operation] = self._per_operation.get(operation, 0) + 1
            if is_retry:
                self._retry_attempts += 1

    def record_http_completion(self, *, success: bool) -> None:
        """Count the outcome of a previously reserved HTTP attempt.

        :param success: Whether the provider response was accepted.
        :raises TypeError: If success is not bool.
        """
        if type(success) is not bool:
            raise TypeError("success must be bool")
        if success:
            self._successful_attempts += 1
        else:
            self._failed_attempts += 1

    def record_historical_cache_access(self, *, hit: bool) -> None:
        """Measure historical price cache efficiency without logging identities.

        :param hit: True for a cached result, False for a provider lookup.
        :raises TypeError: If hit is not bool.
        """
        if type(hit) is not bool:
            raise TypeError("hit must be bool")
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def record_progress(self, *, stage: str, pages_completed: int, items_seen: int) -> None:
        """Record a monotonic checkpoint of verified, not speculative, progress.

        :param stage: Static processing stage name, never wallet or tx identifiers.
        :param pages_completed: Number of completely validated pages.
        :param items_seen: Validated movements obtained from those pages.
        :raises ValueError: For unsafe names or regressing progress counters.
        :raises TypeError: For noninteger page or item counters.
        """
        if not isinstance(stage, str) or _SAFE_OPERATION.fullmatch(stage) is None:
            raise ValueError("stage must be a safe identifier")
        if type(pages_completed) is not int or type(items_seen) is not int:
            raise TypeError("progress counters must be integers")
        if pages_completed < 0 or items_seen < 0:
            raise ValueError("progress counters must be nonnegative")
        previous = self._progress.get(stage)
        if previous is not None and (
            pages_completed < previous.pages_completed or items_seen < previous.items_seen
        ):
            raise ValueError("progress counters cannot regress")
        self._progress[stage] = RunProgress(stage, pages_completed, items_seen)

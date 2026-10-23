"""Alchemy Ethereum JSON-RPC adapter with bounded, shared-client async I/O."""

import asyncio
import itertools
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from portfolio_analyzer.application.provider_errors import (
    ProviderError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    should_retry_provider_error,
)
from portfolio_analyzer.application.run_guardrails import AnalysisRunStopped, RunGuardrails
from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata
from portfolio_analyzer.ethereum.rpc_models import RpcBlock, RpcLog, RpcReceipt, RpcTransaction

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}\Z")
_HASH_RE = re.compile(r"0x[0-9a-fA-F]{64}\Z")
_QUANTITY_RE = re.compile(r"0x(?:0|[1-9a-fA-F][0-9a-fA-F]*)\Z")
_DATA_RE = re.compile(r"0x(?:[0-9a-fA-F]{2})*\Z")
_ETH = AssetId(chain_id=1, contract_address=None)
_ETH_METADATA = AssetMetadata(symbol="ETH", name="Ether", decimals=18)
_LOGGER = logging.getLogger(__name__)


def _failure_kind(error: ProviderError) -> str:
    """Classify a provider error without exposing vendor text or HTTP payloads.

    :param error: Normalized provider exception.
    :return: Static, privacy-safe error category for structured logging.
    """
    if isinstance(error, ProviderRateLimitError):
        return "rate_limit"
    if isinstance(error, ProviderTimeoutError):
        return "timeout"
    if isinstance(error, ProviderUnavailableError):
        return "unavailable"
    if isinstance(error, ProviderMalformedResponseError):
        return "malformed_response"
    return "provider_error"


def _address(value: object) -> str:
    """Validate and canonicalize a user or RPC Ethereum address.

    :param value: Candidate 20-byte 0x-prefixed address.
    :return: Lowercase canonical address.
    :raises ValueError: If address is malformed.
    """
    if not isinstance(value, str) or _ADDRESS_RE.fullmatch(value) is None:
        raise ValueError("Invalid Ethereum address")
    return value.lower()


def _hash(value: object) -> str:
    """Validate and canonicalize a transaction hash.

    :param value: Candidate 32-byte hash.
    :return: Lowercase canonical hash.
    :raises ValueError: If the hash is malformed.
    """
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError("Invalid Ethereum transaction hash")
    return value.lower()


def _block_param(at_block: int | None) -> str:
    """Encode a block number without accepting bool or negative numbers.

    :param at_block: Optional block height; None means latest.
    :return: JSON-RPC block selector.
    :raises TypeError: If a height is not an integer.
    :raises ValueError: If a height is negative.
    """
    if at_block is None:
        return "latest"
    if type(at_block) is not int:
        raise TypeError("block number must be an integer")
    if at_block < 0:
        raise ValueError("block number must not be negative")
    return hex(at_block)


def _quantity(value: object, *, operation: str) -> int:
    """Parse one nonnegative JSON-RPC hex quantity without coercion.

    :param value: JSON value returned by the RPC node.
    :param operation: Safe logical operation name for errors.
    :return: Nonnegative integer value.
    :raises ProviderMalformedResponseError: If the encoding is invalid.
    """
    if not isinstance(value, str) or _QUANTITY_RE.fullmatch(value) is None:
        raise ProviderMalformedResponseError("alchemy", operation)
    return int(value, 16)


def _data(value: object, *, operation: str, size: int | None = None) -> bytes:
    """Parse even-length RPC hex data into bytes.

    :param value: RPC 0x-prefixed hex data.
    :param operation: Safe logical operation name for errors.
    :param size: Optional required byte length.
    :return: Decoded bytes, possibly empty.
    :raises ProviderMalformedResponseError: If data has the wrong format.
    """
    if not isinstance(value, str) or _DATA_RE.fullmatch(value) is None:
        raise ProviderMalformedResponseError("alchemy", operation)
    result = bytes.fromhex(value[2:])
    if size is not None and len(result) != size:
        raise ProviderMalformedResponseError("alchemy", operation)
    return result


def _mapping(value: object, *, operation: str) -> dict[str, Any]:
    """Require a JSON object before extracting required properties.

    :param value: Candidate response value.
    :param operation: Safe logical operation name for errors.
    :return: A string-keyed mapping of JSON fields.
    :raises ProviderMalformedResponseError: If input is not a JSON object.
    """
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ProviderMalformedResponseError("alchemy", operation)
    return value


def _field_address(value: object, *, operation: str) -> str:
    """Translate a malformed vendor address into an internal error.

    :param value: RPC address value.
    :param operation: Safe method name.
    :return: Canonical address.
    :raises ProviderMalformedResponseError: If the address is malformed.
    """
    try:
        return _address(value)
    except ValueError:
        raise ProviderMalformedResponseError("alchemy", operation) from None


def _field_hash(value: object, *, operation: str) -> str:
    """Translate a malformed vendor hash into an internal error.

    :param value: RPC hash value.
    :param operation: Safe method name.
    :return: Canonical hash.
    :raises ProviderMalformedResponseError: If the hash is malformed.
    """
    try:
        return _hash(value)
    except ValueError:
        raise ProviderMalformedResponseError("alchemy", operation) from None


def _receipt_success(raw: dict[str, Any]) -> bool | None:
    """Classify receipt status while preserving pre-Byzantium uncertainty.

    :param raw: JSON-RPC receipt data validated as an object.
    :return: True for 0x1, False for 0x0, None for legacy root-only receipts.
    :raises ProviderMalformedResponseError: For missing or invalid status data.
    """
    operation = "get_transaction_receipt"

    if "status" in raw:
        status = raw["status"]
        if status == "0x1":
            return True
        if status == "0x0":
            return False
        raise ProviderMalformedResponseError("alchemy", operation)

    if "root" in raw:
        _data(raw["root"], operation=operation, size=32)
        return None

    raise ProviderMalformedResponseError("alchemy", operation)


class AlchemyRpcProvider:
    """Implement RpcProvider through a caller-owned shared HTTPX client.

    The caller must create/close the AsyncClient. Requests use one bounded
    semaphore; attempts and backoff are finite and happen outside the slot.

    :param client: Shared AsyncClient managed by the composition root.
    :param api_key: Private Alchemy API key; never included in errors/logs.
    :param request_timeout: Per-request timeout in seconds.
    :param max_concurrency: Maximum simultaneous outbound HTTP requests.
    :param max_attempts: Total attempts, including the first request.
    :param base_backoff: Initial retry backoff in seconds.
    :param max_backoff: Maximum retry backoff in seconds.
    :param sleep: Injectable async sleep for deterministic unit tests.
    :param guardrails: Optional per-analysis shared request cap and usage diagnostics.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        *,
        request_timeout: float = 15.0,
        max_concurrency: int = 5,
        max_attempts: int = 3,
        base_backoff: float = 0.5,
        max_backoff: float = 4.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        guardrails: RunGuardrails | None = None,
    ) -> None:
        """Bind a shared client and validate bounded I/O settings.

        :param client: Externally owned HTTP session.
        :param api_key: Private authentication token.
        :param request_timeout: Finite positive per-request timeout.
        :param max_concurrency: Positive in-flight HTTP request cap.
        :param max_attempts: Positive count of total HTTP attempts.
        :param base_backoff: Nonnegative initial backoff.
        :param max_backoff: Nonnegative maximum backoff.
        :param sleep: Async sleep dependency.
        :param guardrails: Shared optional per-run request guardrails.
        :raises ValueError: If configuration is invalid.
        :raises TypeError: If client has the wrong type.
        """
        if not isinstance(client, httpx.AsyncClient):
            raise TypeError("client must be an httpx.AsyncClient")
        if not isinstance(api_key, str) or re.fullmatch(r"[A-Za-z0-9_-]+", api_key) is None:
            raise ValueError("Alchemy API key must be a nonempty token")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < minimum
            for number, minimum in (
                (request_timeout, 0.0000001),
                (base_backoff, 0),
                (max_backoff, 0),
            )
        ):
            raise ValueError("timeout/backoff must be finite and nonnegative")
        if guardrails is not None and not isinstance(guardrails, RunGuardrails):
            raise TypeError("guardrails must be RunGuardrails or None")
        self._guardrails = guardrails
        self._client = client
        self._url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
        self._prices_url = f"https://api.g.alchemy.com/prices/v1/{api_key}/tokens"
        self._timeout = float(request_timeout)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts
        self._base_backoff = float(base_backoff)
        self._max_backoff = float(max_backoff)
        self._sleep = sleep
        self._request_ids = itertools.count(1)

    @property
    def run_guardrails(self) -> RunGuardrails | None:
        """Expose shared per-analysis metrics to sibling infrastructure adapters.

        :return: Configured guardrails, or None for legacy/test callers.
        """
        return self._guardrails

    async def _execute_http(
        self,
        *,
        operation: str,
        send: Callable[[], Awaitable[httpx.Response]],
        decode: Callable[[httpx.Response], object],
        not_found_returns_none: bool = False,
    ) -> object:
        """Apply one bounded request/retry policy to RPC and Prices APIs.

        :param operation: Safe internal operation name.
        :param send: One HTTP attempt, borrowing the shared client.
        :param decode: Validate a successful response without exposing payloads.
        :param not_found_returns_none: Treat an HTTP 404 as missing only for opted-in lookups.
        :return: Validated vendor result for the infrastructure caller.
        :raises ProviderError: For exhausted attempts or invalid responses.
        """
        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            context = {"provider": "alchemy", "operation": operation, "attempt": attempt}
            _LOGGER.debug("provider_request_started", extra=context)
            reserved = False
            successful = False
            try:
                async with self._semaphore:
                    # Reserve only after obtaining a concurrency slot. Both
                    # retries and the first attempt consume the shared budget.
                    if self._guardrails is not None:
                        await self._guardrails.reserve_http_attempt(
                            operation=operation, is_retry=attempt > 1
                        )
                        reserved = True
                    response = await asyncio.wait_for(send(), timeout=self._timeout)
                if response.status_code == 404 and not_found_returns_none:
                    successful = True
                    _LOGGER.debug(
                        "provider_request_completed",
                        extra={
                            **context,
                            "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
                        },
                    )
                    return None
                if response.status_code == 429:
                    raise ProviderRateLimitError("alchemy", operation)
                if response.status_code in (408, 504):
                    raise ProviderTimeoutError("alchemy", operation)
                if response.status_code in (500, 502, 503):
                    raise ProviderUnavailableError("alchemy", operation)
                if response.status_code >= 400:
                    raise ProviderError("alchemy", operation)
                result = decode(response)
                successful = True
                _LOGGER.debug(
                    "provider_request_completed",
                    extra={
                        **context,
                        "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
                    },
                )
                return result
            except AnalysisRunStopped as exc:
                _LOGGER.warning(
                    "analysis_run_stopped",
                    extra={
                        **context,
                        "http_attempts": exc.snapshot.http_attempts,
                        "cache_hit_count": exc.snapshot.historical_cache_hits,
                    },
                )
                raise
            except (httpx.TimeoutException, TimeoutError):
                failure: ProviderError = ProviderTimeoutError("alchemy", operation)
            except httpx.TransportError:
                failure = ProviderUnavailableError("alchemy", operation)
            except ProviderError as exc:
                failure = exc
            finally:
                # Also count cancellation of an already reserved attempt.
                if reserved and self._guardrails is not None:
                    self._guardrails.record_http_completion(success=successful)
            failure_context = {
                **context,
                "failure_kind": _failure_kind(failure),
                "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
            }
            if not should_retry_provider_error(
                failure, attempts_made=attempt, max_attempts=self._max_attempts
            ):
                _LOGGER.error("provider_request_failed", extra=failure_context)
                raise failure from None
            _LOGGER.warning("provider_request_retry", extra=failure_context)
            delay = min(self._max_backoff, self._base_backoff * 2 ** (attempt - 1))
            await self._sleep(delay)
        raise AssertionError("Unreachable: bounded HTTP retry loop exhausted")

    async def _request(self, method: str, params: list[object], *, operation: str) -> object:
        """Send a JSON-RPC request through shared, bounded HTTP handling.

        :param method: Controlled JSON-RPC method name.
        :param params: JSON-serializable RPC method parameters.
        :param operation: Safe operation identifier for errors.
        :return: Validated JSON-RPC result field.
        :raises ProviderError: If transport, JSON or RPC validation fails.
        """
        request_id = 0

        async def send() -> httpx.Response:
            """Perform one authenticated JSON-RPC HTTP attempt."""
            nonlocal request_id
            request_id = next(self._request_ids)
            return await self._client.post(
                self._url,
                json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                timeout=self._timeout,
            )

        def decode(response: httpx.Response) -> object:
            """Validate the RPC envelope and return its result."""
            try:
                payload: object = response.json()
            except (json.JSONDecodeError, ValueError):
                raise ProviderMalformedResponseError("alchemy", operation) from None
            data = _mapping(payload, operation=operation)
            if data.get("jsonrpc") != "2.0" or type(data.get("id")) is not int:
                raise ProviderMalformedResponseError("alchemy", operation)
            if data["id"] != request_id:
                raise ProviderMalformedResponseError("alchemy", operation)
            if "error" in data:
                rpc_error = _mapping(data["error"], operation=operation)
                if rpc_error.get("code") == -32005:
                    raise ProviderRateLimitError("alchemy", operation)
                raise ProviderError("alchemy", operation)
            if "result" not in data:
                raise ProviderMalformedResponseError("alchemy", operation)
            return data["result"]

        return await self._execute_http(operation=operation, send=send, decode=decode)

    async def request_alchemy_prices(self, *, contract_address: str | None = None) -> object:
        """Read one Mainnet USD price via the Alchemy Prices REST API.

        Native ETH uses a fixed ETH symbol, never token metadata. ERC-20 prices
        are looked up by network and contract address, not by ticker.

        :param contract_address: Canonical token address, or None for native ETH.
        :return: Validated top-level Prices API object (infrastructure only).
        :raises ProviderError: For transport or response failures.
        :raises ValueError: If contract address is invalid.
        """
        address = None if contract_address is None else _address(contract_address)
        operation = "get_current_price"

        async def send() -> httpx.Response:
            """Use the same owned session, timeout and semaphore as RPC."""
            if address is None:
                return await self._client.get(
                    f"{self._prices_url}/by-symbol",
                    params={"symbols": "ETH"},
                    timeout=self._timeout,
                )
            return await self._client.post(
                f"{self._prices_url}/by-address",
                json={"addresses": [{"network": "eth-mainnet", "address": address}]},
                timeout=self._timeout,
            )

        def decode(response: httpx.Response) -> object:
            """Validate the REST envelope before asset-specific price parsing."""
            try:
                payload: object = response.json()
            except (json.JSONDecodeError, ValueError):
                raise ProviderMalformedResponseError("alchemy", operation) from None
            data = _mapping(payload, operation=operation)
            if not isinstance(data.get("data"), list):
                raise ProviderMalformedResponseError("alchemy", operation)
            return data

        return await self._execute_http(operation=operation, send=send, decode=decode)

    async def request_alchemy_historical_prices(
        self,
        *,
        contract_address: str | None,
        start_time: datetime,
        end_time: datetime,
    ) -> object | None:
        """Fetch timestamped USD price points using the shared Prices REST transport.

        Native ETH uses the fixed ETH symbol; ERC-20 uses Mainnet+address.
        HTTP 404 means unsupported historical prices only for this endpoint.

        :param contract_address: Token contract or None for native ETH.
        :param start_time: Inclusive timezone-aware start of lookup window.
        :param end_time: Inclusive timezone-aware end of lookup window.
        :return: Top-level REST payload or None on HTTP 404.
        :raises TypeError: If timestamps have incorrect types.
        :raises ValueError: If window or contract address is invalid.
        :raises ProviderError: For transport or malformed JSON failures.
        """
        address = None if contract_address is None else _address(contract_address)
        if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
            raise TypeError("historical price timestamps must be datetimes")
        if start_time.utcoffset() is None or end_time.utcoffset() is None:
            raise ValueError("historical price timestamps must be timezone-aware")
        start_utc = start_time.astimezone(UTC)
        end_utc = end_time.astimezone(UTC)
        if start_utc >= end_utc:
            raise ValueError("historical price window must be increasing")
        operation = "get_historical_price"
        payload: dict[str, object] = {
            "startTime": start_utc.isoformat().replace("+00:00", "Z"),
            "endTime": end_utc.isoformat().replace("+00:00", "Z"),
            "interval": "5m",
        }
        if address is None:
            payload["symbol"] = "ETH"
        else:
            payload["network"] = "eth-mainnet"
            payload["address"] = address

        async def send() -> httpx.Response:
            """Issue one bounded historical-pricing request."""
            return await self._client.post(
                f"{self._prices_url}/historical",
                json=payload,
                timeout=self._timeout,
            )

        def decode(response: httpx.Response) -> object:
            """Reject malformed REST JSON before price-specific validation."""
            try:
                data: object = response.json()
            except (json.JSONDecodeError, ValueError):
                raise ProviderMalformedResponseError("alchemy", operation) from None
            return _mapping(data, operation=operation)

        return await self._execute_http(
            operation=operation,
            send=send,
            decode=decode,
            not_found_returns_none=True,
        )

    async def request_alchemy_method(
        self, method: str, params: list[object], *, operation: str
    ) -> object:
        """Share bounded HTTP transport with sibling Alchemy adapters.

        This is infrastructure-only; it is not part of the RpcProvider port.

        :param method: Vendor JSON-RPC method name.
        :param params: JSON-RPC parameters.
        :param operation: Safe internal operation name for errors.
        :return: Decoded JSON-RPC result payload.
        :raises ProviderError: When request, retry, or response validation fails.
        """
        return await self._request(method, params, operation=operation)

    async def get_chain_id(self) -> int:
        """Return provider chain ID; the composition root must require Mainnet.

        :return: Nonnegative EVM chain ID.
        """
        chain_id = _quantity(
            await self._request("eth_chainId", [], operation="get_chain_id"),
            operation="get_chain_id",
        )
        if chain_id == 0:
            raise ProviderMalformedResponseError("alchemy", "get_chain_id")
        return chain_id

    async def get_block_number(self) -> int:
        """Return latest canonical block number.

        :return: Nonnegative block number.
        """
        result = await self._request("eth_blockNumber", [], operation="get_block_number")
        return _quantity(result, operation="get_block_number")

    async def get_balance(self, address: str, *, at_block: int | None = None) -> AssetAmount:
        """Return exact native ETH balance for a wallet.

        :param address: Ethereum address.
        :param at_block: Optional historical height; None for latest.
        :return: Exact native ETH quantity.
        """
        result = await self._request(
            "eth_getBalance", [_address(address), _block_param(at_block)], operation="get_balance"
        )
        wei = _quantity(result, operation="get_balance")
        return AssetAmount.from_raw_units(_ETH, wei, _ETH_METADATA)

    async def get_transaction_by_hash(self, tx_hash: str) -> RpcTransaction | None:
        """Read a transaction, including pending transactions.

        :param tx_hash: Transaction identifier.
        :return: Own transaction model or None when absent.
        """
        requested_hash = _hash(tx_hash)
        result = await self._request(
            "eth_getTransactionByHash", [requested_hash], operation="get_transaction_by_hash"
        )
        if result is None:
            return None
        operation = "get_transaction_by_hash"
        raw = _mapping(result, operation=operation)
        if _field_hash(raw.get("hash"), operation=operation) != requested_hash:
            raise ProviderMalformedResponseError("alchemy", operation)
        from_address = _field_address(raw.get("from"), operation=operation)
        to = raw.get("to")
        to_address = None if to is None else _field_address(to, operation=operation)
        raw_block = raw.get("blockNumber")
        block = None if raw_block is None else _quantity(raw_block, operation=operation)
        value = _quantity(raw.get("value"), operation=operation)
        calldata = _data(raw.get("input"), operation=operation)
        return RpcTransaction(requested_hash, from_address, to_address, block, value, calldata)

    async def get_transaction_receipt(self, tx_hash: str) -> RpcReceipt | None:
        """Read a mined transaction receipt including ERC-20 event logs.

        :param tx_hash: Transaction identifier.
        :return: Parsed receipt or None while absent or pending.
        """
        requested_hash = _hash(tx_hash)
        result = await self._request(
            "eth_getTransactionReceipt", [requested_hash], operation="get_transaction_receipt"
        )
        if result is None:
            return None
        operation = "get_transaction_receipt"
        raw = _mapping(result, operation=operation)
        if _field_hash(raw.get("transactionHash"), operation=operation) != requested_hash:
            raise ProviderMalformedResponseError("alchemy", operation)
        block = _quantity(raw.get("blockNumber"), operation=operation)
        gas_used = _quantity(raw.get("gasUsed"), operation=operation)
        raw_price = raw.get("effectiveGasPrice")
        gas_price = None if raw_price is None else _quantity(raw_price, operation=operation)
        raw_logs = raw.get("logs")
        if not isinstance(raw_logs, list):
            raise ProviderMalformedResponseError("alchemy", operation)
        parsed_logs: list[RpcLog] = []
        for entry in raw_logs:
            log = _mapping(entry, operation=operation)
            raw_topics = log.get("topics")
            if not isinstance(raw_topics, list):
                raise ProviderMalformedResponseError("alchemy", operation)
            parsed_logs.append(
                RpcLog(
                    address=_field_address(log.get("address"), operation=operation),
                    topics=tuple(
                        _data(topic, operation=operation, size=32) for topic in raw_topics
                    ),
                    data=_data(log.get("data"), operation=operation),
                )
            )
        return RpcReceipt(
            tx_hash=requested_hash,
            block_number=block,
            success=_receipt_success(raw),
            gas_used=gas_used,
            effective_gas_price_wei=gas_price,
            logs=tuple(parsed_logs),
        )

    async def get_block(self, block_number: int) -> RpcBlock | None:
        """Retrieve timestamp of a numbered block as UTC datetime.

        :param block_number: Nonnegative block height.
        :return: Block facts or None when absent.
        """
        selector = _block_param(block_number)
        result = await self._request(
            "eth_getBlockByNumber", [selector, False], operation="get_block"
        )
        if result is None:
            return None
        operation = "get_block"
        raw = _mapping(result, operation=operation)
        number = _quantity(raw.get("number"), operation=operation)
        if number != block_number:
            raise ProviderMalformedResponseError("alchemy", operation)
        seconds = _quantity(raw.get("timestamp"), operation=operation)
        try:
            timestamp = datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            raise ProviderMalformedResponseError("alchemy", operation) from None
        return RpcBlock(number, timestamp)

    async def eth_call(self, to_address: str, data: bytes, *, at_block: int | None = None) -> bytes:
        """Call a contract without ABI decoding and return raw response bytes.

        :param to_address: Destination contract address.
        :param data: ABI-encoded calldata bytes.
        :param at_block: Historical height if desired.
        :return: Raw EVM return data.
        :raises TypeError: If calldata is not bytes.
        """
        if not isinstance(data, bytes):
            raise TypeError("eth_call data must be bytes")
        result = await self._request(
            "eth_call",
            [{"to": _address(to_address), "data": "0x" + data.hex()}, _block_param(at_block)],
            operation="eth_call",
        )
        return _data(result, operation="eth_call")

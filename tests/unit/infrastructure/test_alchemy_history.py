"""Offline Alchemy history tests using bounded shared RPC transport."""

import json
from collections.abc import Callable
from datetime import UTC
from typing import Any

import httpx
import pytest

from portfolio_analyzer.application.ports import HistoryProvider
from portfolio_analyzer.application.provider_errors import (
    ProviderMalformedResponseError,
    ProviderPartialResultError,
)
from portfolio_analyzer.ethereum.movements import MovementCategory
from portfolio_analyzer.infrastructure.alchemy.history import AlchemyHistoryProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider

WALLET = "0x" + "aa" * 20
OTHER = "0x" + "bb" * 20
TOKEN = "0x" + "cc" * 20
HASH = "0x" + "11" * 32


def transfer(
    *,
    unique_id: str = "first",
    from_address: str = OTHER,
    to_address: str = WALLET,
    category: str = "erc20",
    raw_value: str = "0x1",
    block: str = "0x10",
    contract: str | None = TOKEN,
) -> dict[str, Any]:
    """Create a representative vendor transfer containing lossless raw units.

    :param unique_id: Vendor-provided event provenance identity.
    :param from_address: Transfer sender.
    :param to_address: Transfer recipient.
    :param category: Source movement kind.
    :param raw_value: Exact on-chain integer amount.
    :param block: Hex block number.
    :param contract: Token contract, or None for native ETH.
    :return: Transfer JSON dictionary.
    """
    return {
        "uniqueId": unique_id,
        "hash": HASH,
        "blockNum": block,
        "from": from_address,
        "to": to_address,
        "category": category,
        "asset": "ANY_FAKE_SYMBOL",
        "value": 0.00000000000000001,
        "rawContract": {"address": contract, "value": raw_value, "decimal": None},
        "metadata": {"blockTimestamp": "2026-01-01T12:00:00.000Z"},
    }


def make_transport(
    callback: Callable[[str, dict[str, Any]], tuple[int, object]],
) -> httpx.MockTransport:
    """Create a validating JSON-RPC transport stub.

    :param callback: Fake service taking method and first request parameter.
    :return: Reusable offline HTTP transport.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Reply with a fixture using the request's JSON-RPC identifier.

        :param request: Incoming offline HTTP request.
        :return: JSON-RPC HTTP response.
        """
        body = json.loads(request.content)
        method = body["method"]
        params = body["params"][0] if body["params"] else {}
        status, result = callback(method, params)
        return httpx.Response(
            status,
            json={"jsonrpc": "2.0", "id": body["id"], "result": result},
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_history_pages_both_directions_and_converts_exact_raw_amount() -> None:
    """Fetch every page, pin a block snapshot, and retain exact token amounts."""
    seen: list[tuple[str, dict[str, Any]]] = []
    large_value = 10**50 + 1

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Serve native and ERC-20 movements split across pages.

        :param method: Requested RPC method.
        :param params: Transfer query parameters.
        :return: HTTP status and payload.
        """
        seen.append((method, dict(params)))
        if method == "eth_blockNumber":
            return 200, "0x20"
        if "fromAddress" in params:
            if "pageKey" not in params:
                return 200, {
                    "transfers": [
                        transfer(
                            unique_id="out",
                            from_address=WALLET,
                            to_address=OTHER,
                            category="external",
                            raw_value="0xde0b6b3a7640000",
                            contract=None,
                        )
                    ],
                    "pageKey": "p2",
                }
            assert params["pageKey"] == "p2"
            return 200, {
                "transfers": [
                    transfer(
                        unique_id="big",
                        from_address=WALLET,
                        to_address=OTHER,
                        raw_value=hex(large_value),
                    )
                ]
            }
        return 200, {"transfers": [transfer(unique_id="incoming", raw_value="0x1")]}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        rpc = AlchemyRpcProvider(client, "test-key", base_backoff=0)
        provider = AlchemyHistoryProvider(rpc, page_size=2)
        checked_port: HistoryProvider = provider
        assert hasattr(checked_port, "get_history")
        result = await provider.get_history(WALLET.upper().replace("0X", "0x"))

    assert len(result) == 3
    assert result[0].asset_id.is_native
    assert result[0].raw_amount == 10**18
    assert result[0].category is MovementCategory.EXTERNAL
    assert result[1].asset_id.contract_address == TOKEN
    assert result[1].raw_amount == large_value
    assert result[2].raw_amount == 1
    assert result[0].timestamp.tzinfo is UTC
    calls = [(method, params) for method, params in seen if method == "alchemy_getAssetTransfers"]
    assert len(calls) == 3
    assert all(params["toBlock"] == "0x20" for _, params in calls)
    assert all(params["withMetadata"] is True for _, params in calls)
    assert all(params["category"] == ["external", "internal", "erc20"] for _, params in calls)
    assert all(params["maxCount"] == "0x2" for _, params in calls)
    assert "fromAddress" in calls[0][1]
    assert "toAddress" in calls[-1][1]


@pytest.mark.asyncio
async def test_history_deduplicates_self_transfer_across_directions() -> None:
    """Do not count the same wallet-to-self movement twice."""
    item = transfer(from_address=WALLET, to_address=WALLET, unique_id="self")

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Return a self-transfer once per requested direction.

        :param method: RPC method.
        :param params: Query filters.
        :return: Static fake response.
        """
        return (200, "0x20") if method == "eth_blockNumber" else (200, {"transfers": [item]})

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        result = await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)
    assert len(result) == 1
    assert result[0].from_address == result[0].to_address == WALLET


@pytest.mark.asyncio
async def test_conflicting_duplicate_identity_raises_partial_error() -> None:
    """Reject inconsistent provenance across incoming and outgoing pages."""
    first = transfer(from_address=WALLET, to_address=WALLET, unique_id="same")
    second = transfer(from_address=WALLET, to_address=WALLET, unique_id="same", raw_value="0x5")

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Provide contradictory data for one transfer identity.

        :param method: Requested method.
        :param params: Query filters.
        :return: Contradictory fixture.
        """
        if method == "eth_blockNumber":
            return 200, "0x20"
        return 200, {"transfers": [first if "fromAddress" in params else second]}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        with pytest.raises(ProviderPartialResultError):
            await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)


@pytest.mark.asyncio
async def test_second_page_failure_never_returns_first_page_as_complete() -> None:
    """A failure after page one must fail the entire history operation."""

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Reply with one page and then exhaust rate-limit retries.

        :param method: Requested method.
        :param params: Query filters.
        :return: HTTP response status and result.
        """
        if method == "eth_blockNumber":
            return 200, "0x20"
        if "pageKey" in params:
            return 429, {}
        return 200, {"transfers": [transfer(from_address=WALLET)], "pageKey": "next"}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        rpc = AlchemyRpcProvider(client, "abc", max_attempts=2, base_backoff=0)
        with pytest.raises(ProviderPartialResultError):
            await AlchemyHistoryProvider(rpc).get_history(WALLET)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_case", ["repeat_cursor", "empty_page", "max_pages", "invalid_page"])
async def test_incomplete_pagination_rejected(bad_case: str) -> None:
    """Reject looped cursors, pagination limits and invalid result shapes.

    :param bad_case: Incomplete pagination fixture category.
    """

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Simulate a broken cursor chain.

        :param method: RPC method.
        :param params: Request filters.
        :return: Broken pagination fixture.
        """
        if method == "eth_blockNumber":
            return 200, "0x20"
        if bad_case == "invalid_page":
            return 200, {"not_transfers": []}
        if bad_case == "empty_page":
            return 200, {"transfers": [], "pageKey": "next"}
        return 200, {"transfers": [transfer(from_address=WALLET)], "pageKey": "cursor"}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        rpc = AlchemyRpcProvider(client, "abc")
        max_pages = 1 if bad_case == "max_pages" else 3
        provider = AlchemyHistoryProvider(rpc, max_pages_per_direction=max_pages)
        expected_error = (
            ProviderMalformedResponseError
            if bad_case == "invalid_page"
            else ProviderPartialResultError
        )
        with pytest.raises(expected_error):
            await provider.get_history(WALLET)


@pytest.mark.asyncio
async def test_empty_history_is_complete_when_both_directions_terminate() -> None:
    """Permit an empty history only when both query directions terminate."""

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Return two legitimately empty completed direction results.

        :param method: RPC method.
        :param params: Filter options.
        :return: Empty success fixture.
        """
        return (200, "0x20") if method == "eth_blockNumber" else (200, {"transfers": []})

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        result = await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)
    assert result == []


@pytest.mark.asyncio
async def test_missing_raw_contract_value_fails_instead_of_using_float_value() -> None:
    """Do not use approximate vendor value if exact raw quantity is absent."""
    item = transfer(from_address=WALLET)
    item["rawContract"]["value"] = None

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Return a movement without exact integer amount.

        :param method: RPC method.
        :param params: Transfer filters.
        :return: Malformed transfer fixture.
        """
        return (200, "0x20") if method == "eth_blockNumber" else (200, {"transfers": [item]})

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)


@pytest.mark.asyncio
async def test_unsupported_response_category_fails_loudly() -> None:
    """Treat an unexpected NFT category as invalid for this ETH/ERC-20 adapter."""
    item = transfer(from_address=WALLET, category="erc721")

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Return unsupported movement type.

        :param method: RPC method.
        :param params: Filters.
        :return: Movement fixture.
        """
        return (200, "0x20") if method == "eth_blockNumber" else (200, {"transfers": [item]})

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)


@pytest.mark.asyncio
async def test_unavailable_incoming_direction_is_not_a_complete_history() -> None:
    """Failure on second direction rejects already loaded outgoing movements."""

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Return outgoing facts then an incomplete incoming query.

        :param method: RPC method.
        :param params: Filters.
        :return: Fake RPC response.
        """
        if method == "eth_blockNumber":
            return 200, "0x20"
        if "toAddress" in params:
            return 200, {"transfers": [], "pageKey": "loop"}
        return 200, {"transfers": [transfer(from_address=WALLET, to_address=OTHER)]}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        with pytest.raises(ProviderPartialResultError):
            await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)


def test_reject_unsupported_page_size() -> None:
    """Validate provider page limits rather than silently truncating history."""
    async_client = httpx.AsyncClient()
    try:
        rpc = AlchemyRpcProvider(async_client, "abc")
        with pytest.raises(ValueError):
            AlchemyHistoryProvider(rpc, page_size=0)
        with pytest.raises(ValueError):
            AlchemyHistoryProvider(rpc, page_size=1001)
    finally:
        import asyncio

        asyncio.run(async_client.aclose())


@pytest.mark.asyncio
async def test_raw_contract_zero_padded_hex_remains_exact() -> None:
    """Preserve ERC-20 event amounts encoded as padded 32-byte hex values."""
    item = transfer(raw_value="0x" + "00" * 31 + "0f")

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Return an incoming transfer with padded raw units.

        :param method: JSON-RPC method.
        :param params: Transfer filters.
        :return: Static fake response.
        """
        if method == "eth_blockNumber":
            return 200, "0x20"
        if "toAddress" in params:
            return 200, {"transfers": [item]}
        return 200, {"transfers": []}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        result = await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)
    assert result[0].raw_amount == 15


@pytest.mark.asyncio
async def test_rate_limit_retry_succeeds_without_duplicate_pages() -> None:
    """Retry temporary 429 once, then preserve only the final page data."""
    failures = 0

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Throttle first outgoing request then provide a valid page.

        :param method: RPC method.
        :param params: Query parameters.
        :return: HTTP code and response.
        """
        nonlocal failures
        if method == "eth_blockNumber":
            return 200, "0x20"
        if "fromAddress" in params and failures == 0:
            failures += 1
            return 429, {}
        if "fromAddress" in params:
            return 200, {"transfers": [transfer(from_address=WALLET, to_address=OTHER)]}
        return 200, {"transfers": []}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        result = await AlchemyHistoryProvider(
            AlchemyRpcProvider(client, "abc", max_attempts=2, base_backoff=0)
        ).get_history(WALLET)
    assert failures == 1
    assert len(result) == 1


@pytest.mark.asyncio
async def test_internal_native_movement_uses_eth_identity() -> None:
    """Treat internal native value transfer as ETH, not a fungible token."""
    item = transfer(category="internal", contract=None, raw_value="0x10")

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Provide one internal incoming movement.

        :param method: Requested method.
        :param params: Transfer query.
        :return: One native transfer fixture.
        """
        if method == "eth_blockNumber":
            return 200, "0x20"
        return 200, {"transfers": [item] if "toAddress" in params else []}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        result = await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)
    assert result[0].asset_id.is_native
    assert result[0].category is MovementCategory.INTERNAL
    assert result[0].raw_amount == 16


@pytest.mark.asyncio
async def test_merged_history_is_sorted_by_block_number() -> None:
    """Sort across outgoing/incoming streams without assuming log-level ordering."""
    newest = transfer(unique_id="newer", from_address=WALLET, to_address=OTHER, block="0x20")
    oldest = transfer(unique_id="older", from_address=OTHER, to_address=WALLET, block="0x10")

    def callback(method: str, params: dict[str, Any]) -> tuple[int, object]:
        """Return incoming older than outgoing despite query order.

        :param method: JSON-RPC method.
        :param params: Query filters.
        :return: Ordered-by-direction mocked response.
        """
        if method == "eth_blockNumber":
            return 200, "0x30"
        return 200, {"transfers": [newest if "fromAddress" in params else oldest]}

    async with httpx.AsyncClient(transport=make_transport(callback)) as client:
        result = await AlchemyHistoryProvider(AlchemyRpcProvider(client, "abc")).get_history(WALLET)
    assert [item.transfer_id for item in result] == ["older", "newer"]

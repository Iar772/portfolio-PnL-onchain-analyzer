"""Offline tests for Alchemy ERC-20 balances and conservative coverage."""

import json
import logging
from decimal import Decimal
from io import StringIO
from typing import Any

import httpx
import pytest

from portfolio_analyzer.application.provider_errors import (
    ProviderMalformedResponseError,
    ProviderPartialResultError,
    ProviderTimeoutError,
)
from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata
from portfolio_analyzer.ethereum.balance_models import Erc20BalanceSnapshot, UnresolvedTokenBalance
from portfolio_analyzer.infrastructure.alchemy.balances import AlchemyErc20BalanceProvider
from portfolio_analyzer.infrastructure.alchemy.metadata import AlchemyTokenMetadataProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider
from portfolio_analyzer.logging_config import configure_logging

_WALLET = "0x" + "ab" * 20
_USDC = "0x" + "a0" * 20
_WETH = "0x" + "b0" * 20
_DAI = "0x" + "c0" * 20
_METADATA = {"name": "USD Coin", "symbol": "USDC", "decimals": 6}


def _reply(request: httpx.Request, result: object) -> httpx.Response:
    """Build matching JSON-RPC response for a fake request.

    :param request: Recorded request.
    :param result: Alchemy result value.
    :return: Valid JSON-RPC response.
    """
    body: dict[str, Any] = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


def _row(contract: str, balance: str | None, *, error: object = None) -> dict[str, object]:
    """Make one vendor balance record.

    :param contract: Token contract address.
    :param balance: Encoded token balance.
    :param error: Per-token provider error, if any.
    :return: Synthetic vendor row.
    """
    return {"contractAddress": contract, "tokenBalance": balance, "error": error}


def _page(*rows: object, cursor: object = None, wallet: str = _WALLET) -> dict[str, object]:
    """Build a paginated Token API response.

    :param rows: Token balance entries.
    :param cursor: Optional next-page key.
    :param wallet: Queried address echoed by Alchemy.
    :return: Mock result page.
    """
    result: dict[str, object] = {"address": wallet, "tokenBalances": list(rows)}
    if cursor is not None:
        result["pageKey"] = cursor
    return result


def _provider(client: httpx.AsyncClient, **settings: int) -> AlchemyErc20BalanceProvider:
    """Construct providers with the exact shared client.

    :param client: Caller-owned async HTTPX client.
    :param settings: Optional pagination settings.
    :return: Fully configured ERC-20 adapter.
    """
    rpc = AlchemyRpcProvider(client, "secret_api_key")
    metadata = AlchemyTokenMetadataProvider(rpc)
    return AlchemyErc20BalanceProvider(rpc, metadata, **settings)


@pytest.mark.asyncio
async def test_exact_units_zero_filter_and_shared_transport() -> None:
    """Keep only nonzero ERC-20 amounts and use true metadata decimals."""
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Mock Token and Metadata APIs from the same transport."""
        body: dict[str, Any] = json.loads(request.content)
        requests.append(body)
        if body["method"] == "alchemy_getTokenBalances":
            return _reply(
                request,
                _page(
                    _row(_USDC.upper().replace("0X", "0x"), "0x000000000016e360"),
                    _row(_WETH, "0x0000"),
                ),
            )
        assert body["method"] == "alchemy_getTokenMetadata"
        assert body["params"] == [_USDC]
        return _reply(request, _METADATA)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _provider(client).get_erc20_balances(_WALLET.upper().replace("0X", "0x"))

    assert result == Erc20BalanceSnapshot(
        balances=(AssetAmount.from_raw_units(AssetId(1, _USDC), 1500000, _metadata_model()),),
        unresolved=(),
    )
    assert str(result.balances[0].quantity) == "1.500000"
    assert result.is_complete is True
    assert len(requests) == 2  # Never lookup metadata for 0 balance
    assert requests[0]["params"] == [_WALLET, "erc20", {"maxCount": 100}]


def _metadata_model() -> AssetMetadata:
    """Provide the exact metadata instance used in amount checks.

    :return: Test token metadata with six decimal places.
    """
    return AssetMetadata("USDC", "USD Coin", 6)


@pytest.mark.asyncio
async def test_multi_page_ignores_zeros_and_preserves_order() -> None:
    """Traverse all cursors before normalizing balance data."""
    keys: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Supply two balance pages and distinct metadata."""
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenBalances":
            keys.append(body["params"][2].get("pageKey"))
            if len(keys) == 1:
                assert body["params"][2]["maxCount"] == 2
                return _reply(request, _page(_row(_USDC, hex(1500000)), cursor="next"))
            return _reply(request, _page(_row(_WETH, "0x0"), _row(_DAI, hex(10**18))))
        if body["params"] == [_USDC]:
            return _reply(request, _METADATA)
        return _reply(request, {"name": "Dai", "symbol": "DAI", "decimals": 18})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _provider(client, page_size=2).get_erc20_balances(_WALLET)
    assert keys == [None, "next"]
    assert [item.asset_id.contract_address for item in result.balances] == [_USDC, _DAI]
    assert [str(item.quantity) for item in result.balances] == ["1.500000", "1.000000000000000000"]
    assert result.unresolved == ()


@pytest.mark.asyncio
async def test_missing_metadata_preserves_unknown_not_zero() -> None:
    """Never infer 18 decimals or silently erase an owned token."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Report a nonzero token without metadata."""
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenBalances":
            return _reply(request, _page(_row(_USDC, "0x3e8")))
        return _reply(request, None)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await _provider(client).get_erc20_balances(_WALLET)
    assert snapshot.balances == ()
    assert snapshot.unresolved == (
        UnresolvedTokenBalance(AssetId(1, _USDC), 1000, "metadata_unavailable"),
    )
    assert not snapshot.is_complete


@pytest.mark.asyncio
async def test_per_token_error_is_unresolved_not_zero() -> None:
    """Distinguish an inaccessible balance from an actual zero."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return Alchemy per-token failure."""
        body: dict[str, Any] = json.loads(request.content)
        assert body["method"] == "alchemy_getTokenBalances"
        return _reply(request, _page(_row(_DAI, None, error="secret vendor failure")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await _provider(client).get_erc20_balances(_WALLET)
    assert snapshot == Erc20BalanceSnapshot(
        (), (UnresolvedTokenBalance(AssetId(1, _DAI), None, "balance_unavailable"),)
    )
    assert not snapshot.is_complete


@pytest.mark.asyncio
async def test_same_contract_symbol_different_address_are_not_merged() -> None:
    """Account for contract identity only, even when tickers match."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return duplicate tickers from distinct contracts."""
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenBalances":
            return _reply(request, _page(_row(_USDC, "0x1"), _row(_WETH, "0x2")))
        return _reply(request, _METADATA)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await _provider(client).get_erc20_balances(_WALLET)
    assert len(snapshot.balances) == 2
    assert snapshot.balances[0].asset_id != snapshot.balances[1].asset_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        "bad",
        {},
        {"address": _WALLET},
        {"address": _WALLET, "tokenBalances": {}},
        _page(_row(_USDC, "10")),
        _page(_row(_USDC, "0x")),
        _page(_row(_USDC, "0x-1")),
        _page(_row(_USDC, "0xgg")),
        _page(_row(_USDC, None)),
        _page(_row("broken", "0x1")),
        _page(_row(_USDC, "0x" + "f" * 65)),
        _page(_row(_USDC, "0x1"), wallet=_WETH),
        _page(_row(_USDC, "0x1"), cursor=7),
        _page(_row(_USDC, "0x1"), cursor=""),
    ],
)
async def test_malformed_initial_page_is_explicit(raw: object) -> None:
    """Malformed first result is not reported as an empty wallet."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the corrupt first page."""
        return _reply(request, raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderMalformedResponseError):
            await _provider(client).get_erc20_balances(_WALLET)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,second",
    [
        (_page(_row(_USDC, "0x1"), cursor="next"), {"bad": "page"}),
        (_page(_row(_USDC, "0x1"), cursor="next"), _page(_row(_USDC, "0x2"))),
        (_page(_row(_USDC, "0x0"), cursor="next"), _page(_row(_USDC, "0x1"))),
        (_page(_row(_USDC, "0x1"), cursor="next"), _page(_row(_USDC, "0x1"), cursor="next")),
    ],
)
async def test_partial_pages_and_conflicts_raise(first: object, second: object) -> None:
    """Never silently accept truncated or internally contradictory pages."""
    counter = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Generate chosen first and second pages."""
        nonlocal counter
        counter += 1
        return _reply(request, first if counter == 1 else second)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderPartialResultError):
            await _provider(client).get_erc20_balances(_WALLET)


@pytest.mark.asyncio
async def test_identical_duplicate_across_pages_is_deduplicated() -> None:
    """Keep one copy of identical data without counting it twice."""
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the same token twice with consistent units."""
        nonlocal seen
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenMetadata":
            return _reply(request, _METADATA)
        seen += 1
        page = _page(_row(_USDC, "0x1"), cursor="next") if seen == 1 else _page(_row(_USDC, "0x1"))
        return _reply(request, page)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _provider(client).get_erc20_balances(_WALLET)
    assert len(result.balances) == 1


@pytest.mark.asyncio
async def test_enforce_page_cap() -> None:
    """Exceeding page budget is an explicit partial result."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Always return another cursor."""
        return _reply(request, _page(cursor="never_end"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderPartialResultError):
            await _provider(client, max_pages=1).get_erc20_balances(_WALLET)


@pytest.mark.asyncio
async def test_http_rate_limit_is_retried_without_losing_balance() -> None:
    """Use shared bounded retry policy for token balance requests."""
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Rate-limit first call, then return one real ERC-20 balance."""
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429)
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenBalances":
            return _reply(request, _page(_row(_USDC, "0x1")))
        return _reply(request, _METADATA)

    async def immediate_sleep(seconds: float) -> None:
        """Skip real retry waits in unit tests."""

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "secret_api_key", sleep=immediate_sleep)
        adapter = AlchemyErc20BalanceProvider(rpc, AlchemyTokenMetadataProvider(rpc))
        result = await adapter.get_erc20_balances(_WALLET)
    assert len(result.balances) == 1
    assert count == 3


@pytest.mark.asyncio
async def test_failure_on_second_page_is_partial_after_retries() -> None:
    """Do not convert upstream timeout into a complete truncated inventory."""
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return page one and time out every later attempt."""
        nonlocal count
        count += 1
        if count == 1:
            return _reply(request, _page(_row(_USDC, "0x1"), cursor="second"))
        raise httpx.ReadTimeout("secret_internal_error")

    async def immediate_sleep(seconds: float) -> None:
        """Skip waiting for backoff in tests."""

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "secret_api_key", sleep=immediate_sleep)
        with pytest.raises(ProviderPartialResultError):
            adapter = AlchemyErc20BalanceProvider(rpc, AlchemyTokenMetadataProvider(rpc))
            await adapter.get_erc20_balances(_WALLET)
    assert count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("size,cap", [(0, 100), (101, 100), (True, 10), (5, 0), (5, False)])
async def test_invalid_pagination_settings_rejected(size: object, cap: object) -> None:
    """Require bounded pagination with strict integer arguments."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        rpc = AlchemyRpcProvider(client, "secret_key")
        with pytest.raises(ValueError):
            AlchemyErc20BalanceProvider(
                rpc, AlchemyTokenMetadataProvider(rpc), page_size=size, max_pages=cap
            )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"asset_id": AssetId(1, None), "raw_units": 10, "reason": "metadata_unavailable"},
        {"asset_id": AssetId(1, _USDC), "raw_units": 0, "reason": "metadata_unavailable"},
        {"asset_id": AssetId(1, _USDC), "raw_units": None, "reason": "metadata_unavailable"},
        {"asset_id": AssetId(1, _USDC), "raw_units": 1, "reason": "balance_unavailable"},
        {"asset_id": AssetId(1, _USDC), "raw_units": 1, "reason": "unexpected"},
    ],
)
def test_unresolved_model_validation(kwargs: dict[str, Any]) -> None:
    """Reject contradictory or ambiguous unresolved balances."""
    with pytest.raises(ValueError):
        UnresolvedTokenBalance(**kwargs)


def test_snapshot_rejects_duplicate_assets() -> None:
    """Prevent contradictory balance and unresolved reporting."""
    amount = AssetAmount.from_raw_units(AssetId(1, _USDC), 1, _metadata_model())
    with pytest.raises(ValueError):
        Erc20BalanceSnapshot((amount, amount), ())
    native = AssetAmount.from_raw_units(AssetId(1, None), 1, AssetMetadata("ETH", "Ether", 18))
    with pytest.raises(ValueError):
        Erc20BalanceSnapshot((native,), ())
    with pytest.raises(ValueError):
        Erc20BalanceSnapshot(
            (amount,), (UnresolvedTokenBalance(AssetId(1, _USDC), 1, "metadata_unavailable"),)
        )


@pytest.mark.asyncio
async def test_logging_does_not_leak_wallet_secret_or_raw_payload() -> None:
    """Emit safe success metrics without wallet or API key exposure."""
    root = logging.getLogger("portfolio_analyzer")
    saved = (tuple(root.handlers), root.level, root.propagate)
    for h in saved[0]:
        root.removeHandler(h)
    stream = StringIO()
    configure_logging(AppConfig(log_level="DEBUG"), stream=stream)

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a zero and one nonzero ERC-20 balance."""
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenBalances":
            return _reply(request, _page(_row(_USDC, "0x3e8")))
        return _reply(request, None)

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            snapshot = await _provider(client).get_erc20_balances(_WALLET)
        assert snapshot.unresolved
        output = stream.getvalue()
        assert '"event":"erc20_balances_completed"' in output
        assert '"unresolved_count":1' in output
        assert '"token_count":0' in output
        assert _WALLET not in output
        assert _USDC not in output
        assert "secret_api_key" not in output
    finally:
        for h in tuple(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])
        root.propagate = saved[2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_units,decimals",
    [
        (1, 0),
        (1, 6),
        (1, 18),
        (10**45 + 12345, 18),
        (1, 255),
    ],
)
async def test_normalization_preserves_full_decimal_precision(
    raw_units: int, decimals: int
) -> None:
    """Use exact raw units even beyond default Decimal context precision.

    :param raw_units: On-chain integer quantity.
    :param decimals: Token precision specified by metadata.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one balance and declared token decimal precision.

        :param request: Test request.
        :return: Full Token API or Metadata API response.
        """
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenBalances":
            return _reply(request, _page(_row(_USDC, hex(raw_units))))
        return _reply(request, {"symbol": "TOK", "name": "Token", "decimals": decimals})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await _provider(client).get_erc20_balances(_WALLET)
    expected = Decimal((0, tuple(int(digit) for digit in str(raw_units)), -decimals))
    assert snapshot.balances[0].quantity.as_tuple() == expected.as_tuple()
    assert snapshot.is_complete


@pytest.mark.asyncio
async def test_empty_wallet_returns_empty_complete_snapshot() -> None:
    """Differentiate an empty result from an incomplete result."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an empty complete Token API page.

        :param request: Recorded request.
        :return: Empty page without pagination cursor.
        """
        return _reply(request, _page())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await _provider(client).get_erc20_balances(_WALLET)
    assert snapshot.is_complete
    assert snapshot.balances == ()
    assert snapshot.unresolved == ()


@pytest.mark.asyncio
async def test_metadata_transport_failure_is_not_mislabeled_as_missing() -> None:
    """Retry exhaustion during metadata retrieval stays a typed provider failure."""
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return balance then repeatedly time out on metadata.

        :param request: Fake provider request.
        :return: Balance response or raise a transport timeout.
        """
        nonlocal count
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenBalances":
            return _reply(request, _page(_row(_USDC, "0x1")))
        count += 1
        raise httpx.ReadTimeout("private vendor error")

    async def immediate_sleep(seconds: float) -> None:
        """Avoid real delays in retries.

        :param seconds: Requested backoff duration.
        """

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "secret_api_key", sleep=immediate_sleep)
        with pytest.raises(ProviderTimeoutError):
            await AlchemyErc20BalanceProvider(
                rpc, AlchemyTokenMetadataProvider(rpc)
            ).get_erc20_balances(_WALLET)
    assert count == 3


@pytest.mark.asyncio
async def test_reject_non_alchemy_adapter_dependencies() -> None:
    """Require existing shared Alchemy adapters rather than arbitrary objects."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        rpc = AlchemyRpcProvider(client, "secret_key")
        metadata = AlchemyTokenMetadataProvider(rpc)
        with pytest.raises(TypeError):
            AlchemyErc20BalanceProvider(None, metadata)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            AlchemyErc20BalanceProvider(rpc, None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_unavailable_metadata_does_not_block_other_valid_tokens() -> None:
    """Keep a known ERC-20 amount when another token's metadata is missing."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Provide one known and one unresolved nonzero position.

        :param request: Mock JSON-RPC request.
        :return: Balance or token-specific metadata response.
        """
        body: dict[str, Any] = json.loads(request.content)
        if body["method"] == "alchemy_getTokenBalances":
            return _reply(request, _page(_row(_USDC, "0xf4240"), _row(_WETH, "0x1")))
        if body["params"] == [_USDC]:
            return _reply(request, _METADATA)
        return _reply(request, {"symbol": None, "name": "Missing", "decimals": None})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await _provider(client).get_erc20_balances(_WALLET)

    assert len(snapshot.balances) == 1
    assert snapshot.balances[0].asset_id == AssetId(1, _USDC)
    assert snapshot.balances[0].quantity == Decimal("1.000000")
    assert snapshot.unresolved == (
        UnresolvedTokenBalance(AssetId(1, _WETH), 1, "metadata_unavailable"),
    )
    assert not snapshot.is_complete

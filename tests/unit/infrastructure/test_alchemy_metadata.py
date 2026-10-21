"""Offline tests of token metadata parsing, provenance, and failure policy."""

import json
from typing import Any, cast

import httpx
import pytest

from portfolio_analyzer.application.ports import TokenMetadataProvider
from portfolio_analyzer.application.provider_errors import (
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.infrastructure.alchemy.metadata import AlchemyTokenMetadataProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider

_USDC = AssetId(1, "0x" + "a0" * 20)
_FAKE_USDC = AssetId(1, "0x" + "b0" * 20)
_ETH = AssetId(1, None)
_VALID: dict[str, object] = {
    "name": "USD Coin",
    "symbol": "USDC",
    "decimals": 6,
    "logo": "https://example.invalid/icon.png",
}


def _rpc_reply(request: httpx.Request, result: object) -> httpx.Response:
    """Return a JSON-RPC result matching the request ID.

    :param request: Outbound RPC request.
    :param result: The desired result object.
    :return: Correlated JSON response.
    """
    body: dict[str, Any] = json.loads(request.content)
    return httpx.Response(
        200,
        json={"jsonrpc": "2.0", "id": body["id"], "result": result},
    )


@pytest.mark.asyncio
async def test_metadata_uses_exact_contract_and_returns_domain_model() -> None:
    """Look up by contract address and never leak vendor fields into the domain."""
    observed: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture request details for deterministic assertions.

        :param request: Mocked HTTP request.
        :return: Complete USDC metadata response.
        """
        observed.append(json.loads(request.content))
        return _rpc_reply(request, _VALID)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key")
        provider: TokenMetadataProvider = AlchemyTokenMetadataProvider(rpc)
        metadata = await provider.get_metadata(_USDC)

    assert metadata == AssetMetadata("USDC", "USD Coin", 6)
    assert isinstance(metadata, AssetMetadata)
    assert observed[0]["method"] == "alchemy_getTokenMetadata"
    assert observed[0]["params"] == [_USDC.contract_address]
    assert not hasattr(metadata, "logo")
    assert not hasattr(metadata, "contract_address")


@pytest.mark.asyncio
async def test_identical_symbols_do_not_collapse_contract_identities() -> None:
    """Treat same-symbol contracts as different assets rather than one position."""
    observed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the same ticker for unrelated token addresses.

        :param request: Mocked metadata request.
        :return: Same display name and ticker in both responses.
        """
        body: dict[str, Any] = json.loads(request.content)
        observed.append(body["params"][0])
        return _rpc_reply(request, _VALID)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AlchemyTokenMetadataProvider(AlchemyRpcProvider(client, "test_key"))
        metadata_1 = await provider.get_metadata(_USDC)
        metadata_2 = await provider.get_metadata(_FAKE_USDC)

    assert metadata_1 == metadata_2
    assert _USDC != _FAKE_USDC
    assert _USDC.contract_address != _FAKE_USDC.contract_address
    assert observed == [_USDC.contract_address, _FAKE_USDC.contract_address]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_metadata",
    [
        None,
        {},
        {"symbol": "USDC", "decimals": 6},
        {"name": "USD Coin", "decimals": 6},
        {"name": "USD Coin", "symbol": "USDC"},
        {"name": None, "symbol": "USDC", "decimals": 6},
        {"name": "USD Coin", "symbol": None, "decimals": 6},
        {"name": "USD Coin", "symbol": "USDC", "decimals": None},
        {"name": "  ", "symbol": "USDC", "decimals": 6},
        {"name": "USD Coin", "symbol": " ", "decimals": 6},
    ],
)
async def test_unavailable_metadata_returns_none(raw_metadata: object) -> None:
    """Model incomplete provider metadata as missing, never with made-up decimals.

    :param raw_metadata: Absent, null, or incomplete vendor fields.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the incomplete response without raising transport errors.

        :param request: Mocked JSON-RPC request.
        :return: Response containing the parameterized metadata.
        """
        return _rpc_reply(request, raw_metadata)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AlchemyTokenMetadataProvider(AlchemyRpcProvider(client, "test_key"))
        assert await provider.get_metadata(_USDC) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_metadata",
    [
        [],
        "not a mapping",
        5,
        {"name": 1, "symbol": "USDC", "decimals": 6},
        {"name": "USD Coin", "symbol": [], "decimals": 6},
        {"name": "USD Coin", "symbol": "USDC", "decimals": True},
        {"name": "USD Coin", "symbol": "USDC", "decimals": "6"},
        {"name": "USD Coin", "symbol": "USDC", "decimals": -1},
        {"name": "USD Coin", "symbol": "USDC", "decimals": 256},
        {"name": "USD Coin", "symbol": "USDC", "decimals": 6.0},
    ],
)
async def test_malformed_metadata_raises_provider_error(raw_metadata: object) -> None:
    """Reject corrupt present fields rather than translating them to fake values.

    :param raw_metadata: Vendor payload with invalid types or range.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve corrupt response data from the mock transport.

        :param request: Mocked JSON-RPC request.
        :return: Malformed metadata response.
        """
        return _rpc_reply(request, raw_metadata)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AlchemyTokenMetadataProvider(AlchemyRpcProvider(client, "test_key"))
        with pytest.raises(ProviderMalformedResponseError) as error:
            await provider.get_metadata(_USDC)
    assert error.value.provider == "alchemy"
    assert error.value.operation == "get_token_metadata"


@pytest.mark.asyncio
@pytest.mark.parametrize("decimals", [0, 6, 18, 255])
async def test_metadata_accepts_supported_uint8_decimals(decimals: int) -> None:
    """Accept valid ERC-20 precision including boundary values.

    :param decimals: Boundary or typical allowed precision.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Respond with the requested uint8 decimals.

        :param request: Mocked JSON-RPC request.
        :return: Well-formed metadata.
        """
        return _rpc_reply(request, {**_VALID, "decimals": decimals})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AlchemyTokenMetadataProvider(AlchemyRpcProvider(client, "test_key"))
        result = await provider.get_metadata(_USDC)
        assert result is not None
        assert result.decimals == decimals


@pytest.mark.asyncio
async def test_metadata_strips_display_text_without_modifying_identity() -> None:
    """Normalize whitespace around a display label without touching AssetId."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return whitespace-padded display fields.

        :param request: Mocked JSON-RPC request.
        :return: Token name and ticker with padding.
        """
        return _rpc_reply(request, {**_VALID, "name": " USD Coin ", "symbol": " USDC "})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AlchemyTokenMetadataProvider(AlchemyRpcProvider(client, "test_key"))
        assert await provider.get_metadata(_USDC) == AssetMetadata("USDC", "USD Coin", 6)
    assert _USDC.contract_address == "0x" + "a0" * 20


@pytest.mark.asyncio
async def test_native_eth_metadata_is_local_and_has_no_network_call() -> None:
    """Return ETH/Ether/18 for the native identity without a provider request."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count unexpected outbound calls.

        :param request: Mocked HTTP request.
        :return: Error response indicating a stray request.
        """
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AlchemyTokenMetadataProvider(AlchemyRpcProvider(client, "test_key"))
        assert await provider.get_metadata(_ETH) == AssetMetadata("ETH", "Ether", 18)
    assert calls == 0


@pytest.mark.asyncio
async def test_only_mainnet_ids_supported() -> None:
    """Reject cross-chain lookups instead of silently querying Mainnet."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Prevent network calls for invalid chain identities.

        :param request: Mocked HTTP request.
        :return: Unexpected service response.
        """
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AlchemyTokenMetadataProvider(AlchemyRpcProvider(client, "test_key"))
        with pytest.raises(ValueError, match="Mainnet only"):
            await provider.get_metadata(AssetId(137, _USDC.contract_address))
        with pytest.raises(TypeError, match="AssetId"):
            await provider.get_metadata(cast(AssetId, "bad"))


@pytest.mark.asyncio
async def test_rate_limit_propagates_as_a_typed_error() -> None:
    """Do not pretend rate-limiting means missing metadata."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Simulate HTTP 429 from Alchemy.

        :param request: Mocked HTTP request.
        :return: Rate-limited response.
        """
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key", max_attempts=1)
        provider = AlchemyTokenMetadataProvider(rpc)
        with pytest.raises(ProviderRateLimitError):
            await provider.get_metadata(_USDC)


@pytest.mark.asyncio
async def test_transient_rate_limit_is_retried_with_existing_transport() -> None:
    """Reuse bounded retry logic rather than creating another HTTP client."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail once with HTTP 429, then return full metadata.

        :param request: Mocked HTTP request.
        :return: Rate limit or successful metadata.
        """
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429)
        return _rpc_reply(request, _VALID)

    async def no_sleep(seconds: float) -> None:
        """Avoid real retry delays in the unit test.

        :param seconds: Backoff seconds, ignored in test.
        """

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key", max_attempts=2, sleep=no_sleep)
        provider = AlchemyTokenMetadataProvider(rpc)
        assert await provider.get_metadata(_USDC) == AssetMetadata("USDC", "USD Coin", 6)
    assert attempts == 2


@pytest.mark.asyncio
async def test_timeout_is_not_silently_returned_as_no_metadata() -> None:
    """Keep network timeouts separate from a known missing metadata result."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Simulate transport-level read timeout.

        :param request: Mocked HTTP request.
        :return: Never returns a response due to the simulated timeout.
        :raises httpx.ReadTimeout: For the emulated request failure.
        """
        raise httpx.ReadTimeout("simulated read timeout")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rpc = AlchemyRpcProvider(client, "test_key", max_attempts=1)
        provider = AlchemyTokenMetadataProvider(rpc)
        with pytest.raises(ProviderTimeoutError):
            await provider.get_metadata(_USDC)

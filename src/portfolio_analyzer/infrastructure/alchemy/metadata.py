"""Alchemy ERC-20 metadata adapter, independent of asset identity."""

import logging

from portfolio_analyzer.application.provider_errors import ProviderMalformedResponseError
from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.infrastructure.alchemy.rpc import AlchemyRpcProvider, _mapping

_OPERATION = "get_token_metadata"
_NATIVE_ETH_METADATA = AssetMetadata(symbol="ETH", name="Ether", decimals=18)
_LOGGER = logging.getLogger(__name__)


def _parse_token_metadata(raw: object) -> AssetMetadata | None:
    """Convert Alchemy token fields to domain metadata without guessing missing data.

    Missing or blank required fields mean metadata is unavailable, not zero.
    Structurally invalid non-null fields are explicit provider errors.

    :param raw: JSON-RPC result for alchemy_getTokenMetadata.
    :return: Valid token metadata, or None if required fields are unavailable.
    :raises ProviderMalformedResponseError: If present fields have invalid types or ranges.
    """
    if raw is None:
        return None

    data = _mapping(raw, operation=_OPERATION)
    name = data.get("name")
    symbol = data.get("symbol")
    decimals = data.get("decimals")

    if name is None or symbol is None or decimals is None:
        return None

    if not isinstance(name, str) or not isinstance(symbol, str):
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    if type(decimals) is not int or not 0 <= decimals <= 255:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)

    normalized_name = name.strip()
    normalized_symbol = symbol.strip()
    if not normalized_name or not normalized_symbol:
        return None

    return AssetMetadata(
        symbol=normalized_symbol,
        name=normalized_name,
        decimals=decimals,
    )


class AlchemyTokenMetadataProvider:
    """Resolve Ethereum Mainnet token metadata through a shared RPC transport.

    The adapter never changes AssetId, and missing vendor metadata returns None.
    Transport failures are propagated as typed provider errors, not masked as absence.

    :param rpc: Shared Alchemy adapter that owns bounded HTTP retries and concurrency.
    :raises TypeError: If rpc is not the expected transport adapter.
    """

    def __init__(self, rpc: AlchemyRpcProvider) -> None:
        """Reuse the existing JSON-RPC transport without owning its lifecycle.

        :param rpc: Already configured AlchemyRpcProvider.
        :raises TypeError: If rpc has an invalid type.
        """
        if not isinstance(rpc, AlchemyRpcProvider):
            raise TypeError("rpc must be an AlchemyRpcProvider")
        self._rpc = rpc

    async def get_metadata(self, asset_id: AssetId) -> AssetMetadata | None:
        """Get token display fields by contract identity, never by ticker.

        :param asset_id: Canonical Mainnet asset identity.
        :return: Valid ERC-20 metadata, native ETH metadata, or None if absent.
        :raises TypeError: If asset_id is not an AssetId.
        :raises ValueError: If asset_id refers to a different network.
        :raises ProviderMalformedResponseError: For an invalid vendor payload.
        """
        if not isinstance(asset_id, AssetId):
            raise TypeError("asset_id must be an AssetId")
        if asset_id.chain_id != 1:
            raise ValueError("Alchemy metadata provider supports Ethereum Mainnet only")
        if asset_id.is_native:
            return _NATIVE_ETH_METADATA

        # AssetId guarantees that ERC-20 identities have canonical contract addresses.
        assert asset_id.contract_address is not None
        raw = await self._rpc.request_alchemy_method(
            "alchemy_getTokenMetadata",
            [asset_id.contract_address],
            operation=_OPERATION,
        )
        try:
            metadata = _parse_token_metadata(raw)
        except ProviderMalformedResponseError:
            _LOGGER.warning(
                "token_metadata_invalid",
                extra={"provider": "alchemy", "operation": _OPERATION},
            )
            raise
        if metadata is None:
            _LOGGER.info(
                "token_metadata_unavailable",
                extra={"provider": "alchemy", "operation": _OPERATION},
            )
        else:
            _LOGGER.debug(
                "token_metadata_completed",
                extra={"provider": "alchemy", "operation": _OPERATION},
            )
        return metadata

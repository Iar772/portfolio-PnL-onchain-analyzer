"""Async provider contracts owned by the application, not by Alchemy."""

from datetime import datetime
from typing import Protocol

from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata
from portfolio_analyzer.ethereum.models import EthereumTransaction


class RpcProvider(Protocol):
    """Expose chain status without leaking JSON-RPC transport details."""

    async def get_chain_id(self) -> int:
        """Return the chain identifier of the connected network.

        :return: EVM chain ID; Ethereum Mainnet is 1.
        """
        ...

    async def get_block_number(self) -> int:
        """Return the most recently observed block number.

        :return: Nonnegative latest block number.
        """
        ...


class HistoryProvider(Protocol):
    """Retrieve Ethereum transaction history as provider-neutral models."""

    async def get_history(self, address: str) -> list[EthereumTransaction]:
        """Load transactions related to an Ethereum wallet.

        :param address: Ethereum wallet address to analyze.
        :return: Transactions mapped to our own Ethereum models.
        """
        ...


class BalanceProvider(Protocol):
    """Retrieve the current native ETH and ERC-20 balances of a wallet."""

    async def get_balances(self, address: str) -> list[AssetAmount]:
        """Return current on-chain token quantities.

        :param address: Ethereum wallet address to analyze.
        :return: Nonnegative balances identified by AssetId, never symbol.
        """
        ...


class TokenMetadataProvider(Protocol):
    """Retrieve display metadata separately from token identity."""

    async def get_metadata(self, asset_id: AssetId) -> AssetMetadata | None:
        """Look up symbol, name, and decimal precision for an asset.

        :param asset_id: Canonical asset identity.
        :return: Valid metadata, or None when unavailable.
        """
        ...


class PriceProvider(Protocol):
    """Provide timestamped USD prices without assuming a particular vendor."""

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Get a current trusted USD price observation.

        :param asset_id: Canonical asset identity.
        :return: Price observation, or None when unavailable.
        """
        ...

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Get a USD price observation relevant to a transaction timestamp.

        :param asset_id: Canonical asset identity.
        :param at: Timezone-aware UTC-compatible transaction timestamp.
        :return: Price observation with its actual observed_at, or None if unknown.
        """
        ...

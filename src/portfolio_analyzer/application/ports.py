"""Async provider contracts owned by the application, not by Alchemy."""

from datetime import datetime
from typing import Protocol

from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata
from portfolio_analyzer.ethereum.movements import AssetMovement
from portfolio_analyzer.ethereum.rpc_models import RpcBlock, RpcReceipt, RpcTransaction


class RpcProvider(Protocol):
    """Expose Ethereum read-only RPC facts without vendor transport details."""

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

    async def get_balance(self, address: str, *, at_block: int | None = None) -> AssetAmount:
        """Retrieve native ETH balance exactly, without float conversion.

        :param address: Ethereum wallet address.
        :param at_block: Optional historical block; None means latest.
        :return: Native ETH balance in whole-ETH units.
        """
        ...

    async def get_transaction_by_hash(self, tx_hash: str) -> RpcTransaction | None:
        """Retrieve a transaction, including pending transactions.

        :param tx_hash: Ethereum transaction hash.
        :return: Transaction facts or None when not found.
        """
        ...

    async def get_transaction_receipt(self, tx_hash: str) -> RpcReceipt | None:
        """Retrieve an execution receipt, absent while transaction is pending.

        :param tx_hash: Ethereum transaction hash.
        :return: Execution receipt or None when not yet available.
        """
        ...

    async def get_block(self, block_number: int) -> RpcBlock | None:
        """Retrieve block timestamp facts by a nonnegative block number.

        :param block_number: Ethereum block height.
        :return: Block facts or None if missing.
        """
        ...

    async def eth_call(self, to_address: str, data: bytes, *, at_block: int | None = None) -> bytes:
        """Execute a read-only contract call without decoding ABI data.

        :param to_address: Contract address.
        :param data: ABI-encoded call data.
        :param at_block: Optional historical block, default latest.
        :return: Raw return bytes for the caller to decode.
        """
        ...


class HistoryProvider(Protocol):
    """Retrieve complete Ethereum asset movements using our internal models."""

    async def get_history(self, address: str) -> list[AssetMovement]:
        """Load native ETH and ERC-20 movements for an Ethereum wallet.

        :param address: Ethereum wallet address to analyze.
        :return: Fully paginated movements with raw on-chain quantities.
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


class CurrentPriceProvider(Protocol):
    """Provide current price observations without requiring historical lookup."""

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Get a timestamped USD quote for one asset.

        :param asset_id: Canonical asset identity.
        :return: Known quote, or None if current price is unavailable.
        """
        ...


class PriceProvider(CurrentPriceProvider, Protocol):
    """Extend current-price access with historical price lookup when implemented."""

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Get a USD price observation relevant to a transaction timestamp.

        :param asset_id: Canonical asset identity.
        :param at: Timezone-aware UTC-compatible transaction timestamp.
        :return: Historical observation, or None when unknown.
        """
        ...

"""Deterministic wallet-relative net asset changes from mined movements."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata
from portfolio_analyzer.ethereum.models import EthereumTransaction
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
)

_WALLET_RE = re.compile(r"0x[0-9a-fA-F]{40}\Z")


@dataclass(frozen=True, slots=True)
class RawAssetDelta:
    """Record a nonzero signed net asset movement in exact base units.

    The conversion to whole tokens is postponed until trustworthy decimals
    are available. Negative values mean the wallet spent the asset.

    :param asset_id: Immutable contract-based asset identity.
    :param raw_units: Signed net quantity in smallest asset units.
    :raises TypeError: For invalid field types.
    :raises ValueError: For zero net changes.
    """

    asset_id: AssetId
    raw_units: int

    def __post_init__(self) -> None:
        """Validate the exact signed quantity and asset identity.

        :raises TypeError: For invalid field types.
        :raises ValueError: If there is no net change.
        """
        if not isinstance(self.asset_id, AssetId):
            raise TypeError("asset_id must be an AssetId")
        if type(self.raw_units) is not int:
            raise TypeError("raw_units must be an integer")
        if self.raw_units == 0:
            raise ValueError("zero net changes must be omitted")

    def to_asset_amount(self, metadata: AssetMetadata) -> AssetAmount:
        """Convert the net to exact whole-token units when decimals are known.

        :param metadata: Trusted metadata for this asset (18 for native ETH).
        :return: Signed normalized AssetAmount; never uses provider decimal hints.
        :raises TypeError: If metadata is invalid.
        """
        return AssetAmount.from_raw_units(self.asset_id, self.raw_units, metadata)


@dataclass(frozen=True, slots=True)
class WalletAssetDelta:
    """Represent a transaction's wallet-relative transfer legs, excluding gas.

    This is not yet an economic trade, even if it has a positive and a
    negative asset leg. Transaction execution fees are separate receipt
    facts and must never be added to these legs.

    :param transaction: Canonical transaction identity and block timestamp.
    :param wallet_address: Canonical lower-case Ethereum wallet address.
    :param changes: Ordered, distinct, nonzero asset changes (possibly empty).
    :param transaction_index: Verified in-block transaction index, if known.
    :raises TypeError: For incompatible field types.
    :raises EthereumNormalizationError: For duplicate or cross-chain assets.
    """

    transaction: EthereumTransaction
    wallet_address: str
    changes: tuple[RawAssetDelta, ...]
    transaction_index: int | None = None

    def __post_init__(self) -> None:
        """Protect internal wallet delta invariants.

        :raises TypeError: For unsupported field types.
        :raises EthereumNormalizationError: For inconsistent change data.
        """
        if not isinstance(self.transaction, EthereumTransaction):
            raise TypeError("transaction must be an EthereumTransaction")
        if not isinstance(self.wallet_address, str) or not _WALLET_RE.fullmatch(
            self.wallet_address
        ):
            raise EthereumNormalizationError("wallet_address must be 20-byte hex")
        object.__setattr__(self, "wallet_address", self.wallet_address.lower())
        if not isinstance(self.changes, tuple):
            raise TypeError("changes must be a tuple")
        if self.transaction_index is not None and (
            type(self.transaction_index) is not int or self.transaction_index < 0
        ):
            raise EthereumNormalizationError("transaction_index must be nonnegative")

        seen: set[AssetId] = set()
        for item in self.changes:
            if not isinstance(item, RawAssetDelta):
                raise TypeError("changes must contain RawAssetDelta values")
            if item.asset_id.chain_id != 1:
                raise EthereumNormalizationError("wallet delta supports Ethereum Mainnet only")
            if item.asset_id in seen:
                raise EthereumNormalizationError("duplicate asset in wallet net changes")
            seen.add(item.asset_id)


def calculate_wallet_asset_delta(
    group: TransactionMovementGroup,
    wallet_address: str,
) -> WalletAssetDelta:
    """Aggregate native/ERC-20 transfers relative to one wallet, not gas.

    For a movement, sender contributes -raw_amount and recipient contributes
    +raw_amount. Self-transfers cancel. Unrelated contract-to-contract
    transfers do not affect the wallet. Output order follows first
    appearance of each asset in the group's ordered movement facts.

    Fees are not present in AssetMovement and are NOT inferred from ETH
    transfers: gas must be read separately from the transaction receipt.

    :param group: One validated, mined transaction and its ordered movements.
    :param wallet_address: An Ethereum address, case-insensitive.
    :return: Exact, immutable, zero-free wallet-relative transaction changes.
    :raises TypeError: If the group or wallet address has an invalid type.
    :raises EthereumNormalizationError: For malformed wallet or other chain.
    """
    if not isinstance(group, TransactionMovementGroup):
        raise TypeError("group must be a TransactionMovementGroup")
    if not isinstance(wallet_address, str) or not _WALLET_RE.fullmatch(wallet_address):
        raise EthereumNormalizationError("wallet_address must be 20-byte hex")
    canonical_address = wallet_address.lower()
    net_by_asset: dict[AssetId, int] = {}

    for movement in group.movements:
        if movement.asset_id.chain_id != 1:
            raise EthereumNormalizationError("wallet delta supports Ethereum Mainnet only")
        signed = 0
        if movement.from_address == canonical_address:
            signed -= movement.raw_amount
        if movement.to_address == canonical_address:
            signed += movement.raw_amount
        if signed:
            net_by_asset[movement.asset_id] = net_by_asset.get(movement.asset_id, 0) + signed

    return WalletAssetDelta(
        transaction=group.transaction,
        wallet_address=canonical_address,
        changes=tuple(
            RawAssetDelta(asset_id=asset_id, raw_units=amount)
            for asset_id, amount in net_by_asset.items()
            if amount != 0
        ),
        transaction_index=group.transaction_index,
    )


def calculate_wallet_asset_deltas(
    groups: Iterable[TransactionMovementGroup],
    wallet_address: str,
) -> tuple[WalletAssetDelta, ...]:
    """Project transaction groups to wallet deltas without reordering them.

    IMPORTANT: When grouping has unknown same-block indices, the order
    remains observed order, not necessarily canonical EVM execution order.

    :param groups: Groups in the order established by the grouping stage.
    :param wallet_address: Ethereum wallet under analysis.
    :return: One WalletAssetDelta per supplied transaction, including empties.
    :raises EthereumNormalizationError: For malformed wallet addresses.
    """
    # Validate even when the iterable is empty.
    if not isinstance(wallet_address, str) or not _WALLET_RE.fullmatch(wallet_address):
        raise EthereumNormalizationError("wallet_address must be 20-byte hex")
    return tuple(calculate_wallet_asset_delta(group, wallet_address) for group in groups)

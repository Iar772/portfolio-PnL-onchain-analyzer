"""Ethereum asset-movement facts independent of Alchemy response schemas."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from portfolio_analyzer.domain.assets import AssetId

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}\Z")
_HASH_RE = re.compile(r"0x[0-9a-fA-F]{64}\Z")


class MovementCategory(StrEnum):
    """Distinguish native value movements from ERC-20 event movements."""

    EXTERNAL = "external"
    INTERNAL = "internal"
    ERC20 = "erc20"


@dataclass(frozen=True, slots=True)
class AssetMovement:
    """Store one on-chain movement without assuming its economic meaning.

    Amount is deliberately kept in integer smallest units until reliable
    token decimals are available. This is not a classified TransferIn or Swap.

    :param transfer_id: Opaque unique transfer provenance identifier.
    :param tx_hash: Ethereum transaction hash.
    :param block_number: Containing block height.
    :param timestamp: Block timestamp normalized to UTC.
    :param from_address: On-chain sender address.
    :param to_address: On-chain recipient address.
    :param asset_id: Native ETH or ERC-20 contract identity.
    :param raw_amount: Nonnegative integer amount in minimal on-chain units.
    :param category: Native external/internal movement or ERC-20 transfer.
    :param decimals_hint: Optional nonauthoritative provider decimal hint.
    """

    transfer_id: str
    tx_hash: str
    block_number: int
    timestamp: datetime
    from_address: str
    to_address: str
    asset_id: AssetId
    raw_amount: int
    category: MovementCategory
    decimals_hint: int | None = None

    def __post_init__(self) -> None:
        """Validate identity and ordering facts without vendor JSON.

        :raises TypeError: For invalid field types.
        :raises ValueError: For malformed on-chain facts.
        """
        if not isinstance(self.transfer_id, str) or not self.transfer_id.strip():
            raise ValueError("transfer_id must be a nonempty string")
        if not isinstance(self.tx_hash, str) or not _HASH_RE.fullmatch(self.tx_hash):
            raise ValueError("tx_hash must be 32-byte hex")
        object.__setattr__(self, "tx_hash", self.tx_hash.lower())
        for name in ("from_address", "to_address"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ADDRESS_RE.fullmatch(value):
                raise ValueError(f"{name} must be 20-byte hex")
            object.__setattr__(self, name, value.lower())
        if type(self.block_number) is not int or self.block_number < 0:
            raise ValueError("block_number must be a nonnegative integer")
        if not isinstance(self.timestamp, datetime) or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))
        if not isinstance(self.asset_id, AssetId):
            raise TypeError("asset_id must be an AssetId")
        if type(self.raw_amount) is not int or self.raw_amount < 0:
            raise ValueError("raw_amount must be a nonnegative integer")
        if not isinstance(self.category, MovementCategory):
            raise TypeError("category must be a MovementCategory")
        if self.asset_id.is_native != (self.category != MovementCategory.ERC20):
            raise ValueError("category conflicts with asset identity")
        if self.decimals_hint is not None and (
            type(self.decimals_hint) is not int or not 0 <= self.decimals_hint <= 255
        ):
            raise ValueError("decimals_hint must be between 0 and 255")

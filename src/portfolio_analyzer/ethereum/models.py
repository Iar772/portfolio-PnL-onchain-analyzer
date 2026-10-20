"""Minimal Ethereum chain fact models used by provider contracts.

Transaction movements, logs, and gas facts will be added when the Ethereum
normalization pipeline needs them. This is not an Alchemy response model.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

_TX_HASH_RE = re.compile(r"0x[0-9a-fA-F]{64}\Z")


@dataclass(frozen=True, slots=True)
class EthereumTransaction:
    """Identify an on-chain transaction and its block timestamp.

    :param tx_hash: 0x-prefixed 32-byte Ethereum transaction hash.
    :param block_number: Nonnegative block number.
    :param timestamp: Timezone-aware block timestamp, normalized to UTC.
    :raises TypeError: If fields have unexpected runtime types.
    :raises ValueError: If transaction identity or timestamp is invalid.
    """

    tx_hash: str
    block_number: int
    timestamp: datetime

    def __post_init__(self) -> None:
        """Reject malformed chain facts and normalize the timestamp.

        :raises TypeError: For incorrect field types.
        :raises ValueError: For malformed identifiers or timestamps.
        """
        if not isinstance(self.tx_hash, str):
            raise TypeError("EthereumTransaction.tx_hash must be a string")
        if _TX_HASH_RE.fullmatch(self.tx_hash) is None:
            raise ValueError("EthereumTransaction.tx_hash must be a 32-byte hex hash")
        object.__setattr__(self, "tx_hash", self.tx_hash.lower())

        if isinstance(self.block_number, bool) or not isinstance(self.block_number, int):
            raise TypeError("EthereumTransaction.block_number must be an integer")
        if self.block_number < 0:
            raise ValueError("EthereumTransaction.block_number must not be negative")

        if not isinstance(self.timestamp, datetime):
            raise TypeError("EthereumTransaction.timestamp must be a datetime")
        if self.timestamp.utcoffset() is None:
            raise ValueError("EthereumTransaction.timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))

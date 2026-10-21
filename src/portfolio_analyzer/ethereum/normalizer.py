"""Pure Ethereum movement grouping before economic event classification."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from portfolio_analyzer.ethereum.models import EthereumTransaction
from portfolio_analyzer.ethereum.movements import AssetMovement


class EthereumNormalizationError(ValueError):
    """Report inconsistent transaction provenance without hiding data loss."""


@dataclass(frozen=True, slots=True)
class TransactionMovementGroup:
    """Collect movements belonging to one mined Ethereum transaction.

    The original per-transaction movement order is preserved. A transaction
    index is optional because Alchemy's Transfers API does not expose a
    reliable canonical index in our current provider-neutral movement model.

    :param transaction: Canonical transaction identity, block and UTC timestamp.
    :param movements: Nonempty tuple of movements in their observed order.
    :param transaction_index: Verified zero-based index within the block, if available.
    :raises TypeError: If an input field has an incompatible type.
    :raises EthereumNormalizationError: If provenance fields contradict one another.
    """

    transaction: EthereumTransaction
    movements: tuple[AssetMovement, ...]
    transaction_index: int | None = None

    def __post_init__(self) -> None:
        """Reject empty groups, mixed transactions and invalid chain positions.

        :raises TypeError: If field types do not match the contract.
        :raises EthereumNormalizationError: If one group's facts disagree.
        """
        if not isinstance(self.transaction, EthereumTransaction):
            raise TypeError("transaction must be an EthereumTransaction")
        if not isinstance(self.movements, tuple):
            raise TypeError("movements must be a tuple")
        if not self.movements:
            raise EthereumNormalizationError("a transaction group cannot be empty")
        if self.transaction_index is not None and (
            type(self.transaction_index) is not int or self.transaction_index < 0
        ):
            raise EthereumNormalizationError("transaction_index must be nonnegative")

        seen_ids: set[str] = set()
        for movement in self.movements:
            if not isinstance(movement, AssetMovement):
                raise TypeError("movements must contain AssetMovement objects")
            if (
                movement.tx_hash != self.transaction.tx_hash
                or movement.block_number != self.transaction.block_number
                or movement.timestamp != self.transaction.timestamp
            ):
                raise EthereumNormalizationError("inconsistent transaction provenance")
            if movement.transfer_id in seen_ids:
                raise EthereumNormalizationError("duplicate transfer_id within transaction")
            seen_ids.add(movement.transfer_id)


def group_movements_by_transaction(
    movements: Iterable[AssetMovement],
    *,
    transaction_indices: Mapping[str, int] | None = None,
) -> tuple[TransactionMovementGroup, ...]:
    """Group by tx hash and order transactions without inventing chain positions.

    Results are ordered by block height. Inside the same block, the first
    observed transaction position is stable unless a complete mapping of
    verified on-chain transaction indices is explicitly supplied. Individual
    movements retain their original relative order within each transaction.

    IMPORTANT: Without transaction_indices, order between distinct transactions
    in the same block is *observed*, not provably the EVM execution order.
    A caller must obtain real transaction indices before relying on that order
    for FIFO replay in same-block multi-transaction cases.

    :param movements: Provider-neutral movements, possibly interleaved by tx hash.
    :param transaction_indices: Optional complete mapping of tx hashes to verified
        zero-based in-block transaction indices; keys must use canonical hashes.
    :return: Immutable groups in increasing block/transaction order.
    :raises TypeError: If movements or transaction index mapping has wrong types.
    :raises EthereumNormalizationError: For inconsistent, duplicate or missing facts.
    """
    if transaction_indices is not None and not isinstance(transaction_indices, Mapping):
        raise TypeError("transaction_indices must be a mapping or None")

    buckets: dict[str, list[AssetMovement]] = {}
    metadata: dict[str, EthereumTransaction] = {}
    first_seen: dict[str, int] = {}
    seen_transfers: set[str] = set()
    block_timestamps: dict[int, datetime] = {}

    for observed_index, movement in enumerate(movements):
        if not isinstance(movement, AssetMovement):
            raise TypeError("movements must contain AssetMovement objects")
        if movement.transfer_id in seen_transfers:
            raise EthereumNormalizationError("duplicate transfer_id in movement history")
        seen_transfers.add(movement.transfer_id)

        tx_hash = movement.tx_hash
        current = metadata.get(tx_hash)
        if current is None:
            metadata[tx_hash] = EthereumTransaction(
                tx_hash=tx_hash,
                block_number=movement.block_number,
                timestamp=movement.timestamp,
            )
            first_seen[tx_hash] = observed_index
            buckets[tx_hash] = []
        elif (
            current.block_number != movement.block_number or current.timestamp != movement.timestamp
        ):
            raise EthereumNormalizationError("inconsistent block or timestamp for tx hash")
        prior_timestamp = block_timestamps.setdefault(movement.block_number, movement.timestamp)
        if prior_timestamp != movement.timestamp:
            raise EthereumNormalizationError("inconsistent timestamp for block")
        buckets[tx_hash].append(movement)

    if transaction_indices is not None:
        if set(transaction_indices) != set(buckets):
            raise EthereumNormalizationError(
                "transaction_indices must cover exactly the grouped hashes"
            )
        used_positions: set[tuple[int, int]] = set()
        for tx_hash, index in transaction_indices.items():
            if type(index) is not int or index < 0:
                raise EthereumNormalizationError("transaction index must be nonnegative integer")
            position = (metadata[tx_hash].block_number, index)
            if position in used_positions:
                raise EthereumNormalizationError("same block transaction index used twice")
            used_positions.add(position)

    groups = tuple(
        TransactionMovementGroup(
            transaction=metadata[tx_hash],
            movements=tuple(bucket),
            transaction_index=(
                transaction_indices[tx_hash] if transaction_indices is not None else None
            ),
        )
        for tx_hash, bucket in buckets.items()
    )
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                group.transaction.block_number,
                group.transaction_index
                if group.transaction_index is not None
                else first_seen[group.transaction.tx_hash],
            ),
        )
    )

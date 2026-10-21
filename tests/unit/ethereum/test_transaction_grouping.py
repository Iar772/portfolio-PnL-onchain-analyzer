"""Tests of deterministic transaction-level Ethereum movement grouping."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.ethereum.models import EthereumTransaction
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
    group_movements_by_transaction,
)

WALLET = "0x" + "aa" * 20
OTHER = "0x" + "bb" * 20
TOKEN = "0x" + "cc" * 20
T0 = datetime(2026, 1, 1, 12, tzinfo=UTC)


def movement(
    transfer_id: str,
    *,
    tx_number: int = 1,
    block: int = 10,
    timestamp: datetime = T0,
    category: MovementCategory = MovementCategory.ERC20,
    raw_amount: int = 10,
) -> AssetMovement:
    """Create independent, valid provider-neutral movement facts.

    :param transfer_id: Movement provenance ID.
    :param tx_number: Integer used to synthesize a unique tx hash.
    :param block: Block number.
    :param timestamp: Block timestamp.
    :param category: Native or token movement category.
    :param raw_amount: Exact base-unit quantity.
    :return: A typed immutable movement fixture.
    """
    return AssetMovement(
        transfer_id=transfer_id,
        tx_hash=f"0x{tx_number:064x}",
        block_number=block,
        timestamp=timestamp,
        from_address=WALLET,
        to_address=OTHER,
        asset_id=AssetId(1, TOKEN if category is MovementCategory.ERC20 else None),
        raw_amount=raw_amount,
        category=category,
    )


def test_empty_history_returns_empty_immutable_result() -> None:
    """Empty history needs no synthetic transaction group."""
    assert group_movements_by_transaction([]) == ()
    assert group_movements_by_transaction(iter(()), transaction_indices={}) == ()


def test_interleaved_movements_are_grouped_without_losing_order() -> None:
    """Interleaved movements in the same block remain ordered inside each tx."""
    first_out = movement("a-out", tx_number=11, raw_amount=50)
    unrelated = movement("b-in", tx_number=22, raw_amount=20)
    first_in = movement("a-in", tx_number=11, raw_amount=40)
    first_native = movement("a-gas-leg", tx_number=11, category=MovementCategory.EXTERNAL)

    grouped = group_movements_by_transaction([first_out, unrelated, first_in, first_native])
    assert len(grouped) == 2
    assert grouped[0].transaction.tx_hash == first_out.tx_hash
    assert grouped[0].movements == (first_out, first_in, first_native)
    assert grouped[1].movements == (unrelated,)
    assert all(group.transaction_index is None for group in grouped)
    assert all(isinstance(group.movements, tuple) for group in grouped)


def test_different_blocks_sort_by_chain_height_not_timestamp() -> None:
    """Block height determines cross-block ordering even for unordered inputs."""
    newer = movement("new", tx_number=2, block=15, timestamp=T0 + timedelta(minutes=1))
    older = movement("old", tx_number=1, block=10, timestamp=T0)
    groups = group_movements_by_transaction(item for item in (newer, older))
    assert tuple(group.transaction.block_number for group in groups) == (10, 15)
    assert tuple(group.movements[0] for group in groups) == (older, newer)


def test_same_block_order_is_stable_first_seen_when_indices_absent() -> None:
    """Stable observed order does not imply canonical chain order."""
    second = movement("second", tx_number=2)
    first = movement("first", tx_number=1)
    tx_hashes = [
        group.transaction.tx_hash for group in group_movements_by_transaction([second, first])
    ]
    assert tx_hashes == [second.tx_hash, first.tx_hash]


def test_verified_indices_override_observed_same_block_order() -> None:
    """Explicit real EVM transaction indices provide canonical in-block ordering."""
    second = movement("second", tx_number=2, block=100)
    first = movement("first", tx_number=1, block=100)
    grouped = group_movements_by_transaction(
        [second, first],
        transaction_indices={second.tx_hash: 5, first.tx_hash: 2},
    )
    assert [group.transaction.tx_hash for group in grouped] == [first.tx_hash, second.tx_hash]
    assert [group.transaction_index for group in grouped] == [2, 5]


def test_indices_can_repeat_in_different_blocks() -> None:
    """TransactionIndex is scoped to its containing block."""
    first = movement("first", tx_number=1, block=1)
    second = movement("second", tx_number=2, block=2)
    groups = group_movements_by_transaction(
        [second, first], transaction_indices={first.tx_hash: 0, second.tx_hash: 0}
    )
    assert [group.transaction.block_number for group in groups] == [1, 2]


def test_one_transaction_with_multiple_native_and_erc20_legs() -> None:
    """A mixed swap-like transaction remains one unclassified group."""
    eth_leg = movement("native", category=MovementCategory.EXTERNAL)
    token_leg = movement("erc20")
    groups = group_movements_by_transaction([eth_leg, token_leg])
    assert len(groups) == 1
    assert groups[0].movements == (eth_leg, token_leg)
    assert isinstance(groups[0].transaction, EthereumTransaction)
    assert groups[0].transaction.timestamp == T0


def test_grouping_does_not_modify_input_or_quantities() -> None:
    """Input is not mutated and minimal units stay exact."""
    huge = 10**90 + 7
    a = movement("a", raw_amount=huge)
    b = movement("b", tx_number=2)
    history = [a, b]
    result = group_movements_by_transaction(history)
    assert history == [a, b]
    assert result[0].movements[0].raw_amount == huge


def test_mixed_timezone_representation_matches_utc_instants() -> None:
    """Two timestamps representing the same instant belong to one transaction."""
    plus_three = timezone(timedelta(hours=3))
    same_time = T0.astimezone(plus_three)
    a = movement("a", timestamp=T0)
    b = movement("b", timestamp=same_time)
    assert len(group_movements_by_transaction([a, b])) == 1
    assert group_movements_by_transaction([a, b])[0].transaction.timestamp is not None
    assert group_movements_by_transaction([a, b])[0].transaction.timestamp.tzinfo is UTC


@pytest.mark.parametrize("duplicate_identical", [True, False])
def test_duplicate_provenance_rejected_instead_of_counted_twice(duplicate_identical: bool) -> None:
    """Even repeated identical transfer IDs violate the complete-history contract.

    :param duplicate_identical: Whether the second movement equals the first one.
    """
    a = movement("dup")
    b = a if duplicate_identical else movement("dup", tx_number=2)
    with pytest.raises(EthereumNormalizationError, match="duplicate transfer_id"):
        group_movements_by_transaction([a, b])


@pytest.mark.parametrize("inconsistent_field", ["block", "timestamp"])
def test_same_hash_with_conflicting_provenance_is_rejected(inconsistent_field: str) -> None:
    """Do not merge a tx hash observed in incompatible blocks or time instants.

    :param inconsistent_field: Provenance attribute made contradictory.
    """
    first = movement("a")
    second = movement(
        "b",
        block=11 if inconsistent_field == "block" else 10,
        timestamp=T0 + timedelta(seconds=1) if inconsistent_field == "timestamp" else T0,
    )
    with pytest.raises(EthereumNormalizationError, match="inconsistent block or timestamp"):
        group_movements_by_transaction([first, second])


@pytest.mark.parametrize("bad_value", [None, object(), "not a movement", 5])
def test_nonmovement_items_are_rejected(bad_value: object) -> None:
    """An invalid item must not be silently omitted.

    :param bad_value: Invalid item in the input stream.
    """
    with pytest.raises(TypeError, match="AssetMovement"):
        group_movements_by_transaction([movement("ok"), bad_value])  # type: ignore[list-item]


@pytest.mark.parametrize("bad_index", [-1, True, 1.5, "1", None])
def test_bad_transaction_indices_rejected(bad_index: object) -> None:
    """Indices must be real nonnegative integers rather than bool or floats.

    :param bad_index: Invalid transaction position.
    """
    item = movement("a")
    with pytest.raises(EthereumNormalizationError, match="index must be nonnegative"):
        mapping = {item.tx_hash: bad_index}
        group_movements_by_transaction(
            [item],
            transaction_indices=mapping,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_mapping", [{}, {"0x" + "00" * 32: 0}])
def test_incomplete_or_extra_index_mapping_rejected(bad_mapping: dict[str, int]) -> None:
    """Reject missing or extraneous index provenance.

    :param bad_mapping: Incomplete or unexpected mapping.
    """
    with pytest.raises(EthereumNormalizationError, match="cover exactly"):
        group_movements_by_transaction([movement("a")], transaction_indices=bad_mapping)


def test_duplicate_canonical_transaction_positions_rejected() -> None:
    """Two different transactions cannot share one block and transaction index."""
    a = movement("a", tx_number=1)
    b = movement("b", tx_number=2)
    with pytest.raises(EthereumNormalizationError, match="index used twice"):
        group_movements_by_transaction([a, b], transaction_indices={a.tx_hash: 2, b.tx_hash: 2})


def test_invalid_mapping_type_rejected() -> None:
    """Do not accept a positional sequence as a tx hash mapping."""
    with pytest.raises(TypeError, match="mapping"):
        group_movements_by_transaction([], transaction_indices=[])  # type: ignore[arg-type]


def test_manual_group_construction_checks_all_movements() -> None:
    """The immutable value object independently enforces its invariants."""
    item = movement("a")
    transaction = EthereumTransaction(item.tx_hash, item.block_number, item.timestamp)
    group = TransactionMovementGroup(transaction, (item,))
    assert group.movements == (item,)
    with pytest.raises(EthereumNormalizationError, match="cannot be empty"):
        TransactionMovementGroup(transaction, ())
    with pytest.raises(EthereumNormalizationError, match="inconsistent"):
        TransactionMovementGroup(transaction, (movement("b", tx_number=2),))
    with pytest.raises(EthereumNormalizationError, match="duplicate"):
        TransactionMovementGroup(transaction, (item, item))
    with pytest.raises(TypeError, match="tuple"):
        TransactionMovementGroup(transaction, [item])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EthereumTransaction"):
        TransactionMovementGroup("bad", (item,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AssetMovement"):
        TransactionMovementGroup(transaction, ("bad",))  # type: ignore[arg-type]
    with pytest.raises(EthereumNormalizationError, match="transaction_index"):
        TransactionMovementGroup(transaction, (item,), transaction_index=True)


def test_grouping_never_changes_economic_meaning() -> None:
    """Combining swap legs into a transaction is not yet a Swap classification."""
    a = movement("sold", category=MovementCategory.ERC20, raw_amount=100)
    b = movement("bought", category=MovementCategory.EXTERNAL, raw_amount=200)
    grouped = group_movements_by_transaction([a, b])
    assert len(grouped) == 1
    assert grouped[0].movements == (a, b)
    assert not hasattr(grouped[0], "economic_event")


def test_same_block_with_different_timestamp_is_rejected() -> None:
    """All transactions within one real Ethereum block share its timestamp."""
    a = movement("a", tx_number=1, block=10, timestamp=T0)
    b = movement("b", tx_number=2, block=10, timestamp=T0 + timedelta(seconds=1))
    with pytest.raises(EthereumNormalizationError, match="timestamp for block"):
        group_movements_by_transaction([a, b])


def test_groups_real_direction_merge_without_double_counting() -> None:
    """Group the E4.2 merge output while retaining unique self-transfers."""
    from portfolio_analyzer.infrastructure.alchemy.history import _merge_directions

    sent = movement("sent", tx_number=3, category=MovementCategory.EXTERNAL)
    received = movement("received", tx_number=3, raw_amount=123)
    self_transfer = movement("self", tx_number=4)
    merged = _merge_directions([sent, self_transfer], [received, self_transfer])
    grouped = group_movements_by_transaction(merged)
    assert len(grouped) == 2
    assert [len(group.movements) for group in grouped] == [2, 1]
    assert {m.transfer_id for m in grouped[0].movements} == {"sent", "received"}
    assert grouped[1].movements == (self_transfer,)

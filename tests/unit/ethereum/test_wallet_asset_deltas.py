"""Tests for wallet-relative asset delta calculation, without guessing gas."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
    group_movements_by_transaction,
)
from portfolio_analyzer.ethereum.wallet_deltas import (
    RawAssetDelta,
    WalletAssetDelta,
    calculate_wallet_asset_delta,
    calculate_wallet_asset_deltas,
)

WALLET = "0x" + "aa" * 20
OTHER = "0x" + "bb" * 20
THIRD = "0x" + "dd" * 20
TOKEN_A = AssetId(1, "0x" + "cc" * 20)
TOKEN_B = AssetId(1, "0x" + "ee" * 20)
ETH = AssetId(1, None)
T0 = datetime(2026, 10, 21, 12, tzinfo=UTC)


def move(
    transfer_id: str,
    *,
    asset: AssetId = TOKEN_A,
    raw: int = 1,
    sender: str = WALLET,
    receiver: str = OTHER,
    tx_no: int = 1,
    block: int = 10,
    category: MovementCategory | None = None,
) -> AssetMovement:
    """Build a wallet transfer for one mined Ethereum transaction.

    :param transfer_id: Unique movement identifier.
    :param asset: ERC-20 or native ETH identity.
    :param raw: Exact positive base-unit quantity.
    :param sender: Ethereum sender address.
    :param receiver: Ethereum recipient address.
    :param tx_no: Number used for a deterministic fake tx hash.
    :param block: Block number of the movement.
    :param category: Movement category, inferred when omitted.
    :return: Provider-neutral immutable movement.
    """
    return AssetMovement(
        transfer_id=transfer_id,
        tx_hash=f"0x{tx_no:064x}",
        block_number=block,
        timestamp=T0 + timedelta(seconds=block - 10),
        from_address=sender,
        to_address=receiver,
        asset_id=asset,
        raw_amount=raw,
        category=(
            category
            if category is not None
            else MovementCategory.EXTERNAL
            if asset.is_native
            else MovementCategory.ERC20
        ),
    )


def group(*movements: AssetMovement) -> TransactionMovementGroup:
    """Produce a validated single-transaction group from movements.

    :param movements: Movement facts belonging to the same transaction.
    :return: Immutable grouping result.
    """
    groups = group_movements_by_transaction(movements)
    assert len(groups) == 1
    return groups[0]


def delta(*movements: AssetMovement, wallet: str = WALLET) -> WalletAssetDelta:
    """Calculate wallet deltas for a single transaction in the fixture.

    :param movements: Transaction transfers.
    :param wallet: Wallet to analyze.
    :return: WalletAssetDelta for the supplied transaction.
    """
    return calculate_wallet_asset_delta(group(*movements), wallet)


def values(result: WalletAssetDelta) -> dict[AssetId, int]:
    """Map exact signed raw changes for concise test assertions.

    :param result: Calculated net changes.
    :return: Asset identities and signed raw units.
    """
    return {item.asset_id: item.raw_units for item in result.changes}


def test_eth_out_and_link_in_is_not_a_classified_swap() -> None:
    """Mixed-asset transaction aggregates only transfers, not economic intent."""
    result = delta(
        move("native", asset=ETH, raw=5 * 10**17),
        move("link", asset=TOKEN_A, raw=50 * 10**18, sender=OTHER, receiver=WALLET),
    )
    assert values(result) == {ETH: -(5 * 10**17), TOKEN_A: 50 * 10**18}
    assert result.transaction.tx_hash == f"0x{1:064x}"
    assert result.wallet_address == WALLET
    assert isinstance(result.changes[0], RawAssetDelta)
    eth_amount = result.changes[0].to_asset_amount(AssetMetadata("ETH", "Ether", 18))
    link_amount = result.changes[1].to_asset_amount(AssetMetadata("LINK", "Chainlink", 18))
    assert eth_amount.quantity == Decimal("-0.5")
    assert link_amount.quantity == Decimal("50")


@pytest.mark.parametrize("category", [MovementCategory.EXTERNAL, MovementCategory.INTERNAL])
def test_native_incoming_and_outgoing_are_aggregated(category: MovementCategory) -> None:
    """Both external value and internal native transfers are valid wallet legs.

    :param category: Native transfer category.
    """
    result = delta(
        move("out", asset=ETH, raw=300, category=category),
        move("in", asset=ETH, raw=100, sender=OTHER, receiver=WALLET, category=category),
    )
    assert values(result) == {ETH: -200}


def test_same_asset_multiple_debits_and_credits_are_net_aggregated() -> None:
    """Multiple transfers in one transaction produce one net asset change."""
    result = delta(
        move("out1", raw=90),
        move("in1", raw=20, sender=OTHER, receiver=WALLET),
        move("out2", raw=5),
        move("in2", raw=3, sender=OTHER, receiver=WALLET),
    )
    assert values(result) == {TOKEN_A: -72}
    assert len(result.changes) == 1


def test_self_transfer_cancels_without_double_count() -> None:
    """A wallet sending to itself changes no token balance."""
    result = delta(move("self", raw=10**40, receiver=WALLET))
    assert result.changes == ()


def test_self_transfer_does_not_cancel_unrelated_incoming() -> None:
    """Self-transfer is ignored while other legs are included."""
    result = delta(
        move("self", raw=300, receiver=WALLET),
        move("received", raw=120, sender=OTHER, receiver=WALLET),
    )
    assert values(result) == {TOKEN_A: 120}


def test_unrelated_third_party_movements_do_not_affect_wallet() -> None:
    """Other participants in the same transaction cannot change wallet delta."""
    result = delta(
        move("unrelated", raw=500, sender=OTHER, receiver=THIRD),
        move("ours", asset=TOKEN_B, raw=4, sender=WALLET, receiver=THIRD),
    )
    assert values(result) == {TOKEN_B: -4}


def test_group_with_no_wallet_leg_yields_empty_result() -> None:
    """A transaction is still represented even if wallet sees no net change."""
    result = delta(move("a", raw=100, sender=OTHER, receiver=THIRD))
    assert result.changes == ()


def test_zero_movements_do_not_produce_zero_leg() -> None:
    """Zero-value facts cannot appear as trade legs."""
    result = delta(move("zero", raw=0), move("other", raw=3, sender=OTHER, receiver=WALLET))
    assert values(result) == {TOKEN_A: 3}


def test_balanced_incoming_and_outgoing_are_eliminated() -> None:
    """Exact net-zero legs disappear after aggregation."""
    result = delta(
        move("a", raw=50),
        move("b", raw=50, sender=OTHER, receiver=WALLET),
    )
    assert result.changes == ()


def test_first_observed_asset_order_is_preserved() -> None:
    """The output asset ordering is deterministic and independent of metadata."""
    result = delta(
        move("b", asset=TOKEN_B, raw=3),
        move("eth", asset=ETH, raw=4),
        move("a", asset=TOKEN_A, raw=7),
        move("b2", asset=TOKEN_B, raw=1),
    )
    assert tuple(item.asset_id for item in result.changes) == (TOKEN_B, ETH, TOKEN_A)


def test_first_asset_with_net_zero_not_in_output() -> None:
    """A canceled initial asset is omitted without disturbing later ordering."""
    result = delta(
        move("out", raw=1),
        move("eth", raw=5, asset=ETH),
        move("in", raw=1, sender=OTHER, receiver=WALLET),
    )
    assert values(result) == {ETH: -5}


def test_gas_is_not_subtracted_from_eth_trade_leg() -> None:
    """Even on fee-paying wallet txs the transfer delta excludes the receipt fee."""
    hypothetical_receipt_gas_used = 21000
    hypothetical_effective_gas_price_wei = 20 * 10**9
    result = delta(move("native", asset=ETH, raw=5 * 10**17))
    assert values(result) == {ETH: -(5 * 10**17)}
    assert hypothetical_receipt_gas_used * hypothetical_effective_gas_price_wei != 0
    # Transaction fees are not represented by AssetMovement; they belong to
    # receipt-based gas accounting, implemented in a separate stage.
    assert not hasattr(result, "gas_fee_wei")


def test_unrelated_contract_gas_does_not_inject_eth_leg() -> None:
    """A pure ERC-20 transaction can have gas fees but no ETH transfer leg."""
    result = delta(move("token", raw=70))
    assert values(result) == {TOKEN_A: -70}
    assert ETH not in values(result)


def test_huge_integers_preserved_beyond_decimal_default_precision() -> None:
    """Integer base-unit arithmetic has no floating or Decimal context rounding."""
    huge = 10**120 + 3
    result = delta(
        move("big", raw=huge),
        move("small", raw=2, sender=OTHER, receiver=WALLET),
    )
    assert values(result) == {TOKEN_A: -(huge - 2)}
    amount = result.changes[0].to_asset_amount(AssetMetadata("L", "Large", 18))
    assert amount.quantity == Decimal(f"{-(huge - 2)}e-18")


def test_case_insensitive_wallet_address_is_normalized() -> None:
    """Hex address casing does not affect transfer accounting."""
    result = delta(move("send", raw=7), wallet=WALLET.upper().replace("0X", "0x"))
    assert result.wallet_address == WALLET
    assert values(result) == {TOKEN_A: -7}


def test_erc20_decimals_hint_is_not_used_to_guess_quantity() -> None:
    """Only trusted metadata supplied later can normalize ERC-20 raw units."""
    m = move("raw", raw=1234500)
    result = delta(m)
    assert values(result) == {TOKEN_A: -1234500}
    amount = result.changes[0].to_asset_amount(AssetMetadata("USDC", "USD Coin", 6))
    assert amount.quantity == Decimal("-1.234500")


def test_multiple_transactions_preserve_group_order_and_index() -> None:
    """Batch API keeps E5.1 ordering and any verified transaction indices."""
    tx_2 = move("second", raw=4, tx_no=2)
    tx_1 = move("first", raw=9, tx_no=1)
    groups = group_movements_by_transaction(
        [tx_2, tx_1], transaction_indices={tx_2.tx_hash: 4, tx_1.tx_hash: 1}
    )
    results = calculate_wallet_asset_deltas(groups, WALLET)
    assert [entry.transaction.tx_hash for entry in results] == [tx_1.tx_hash, tx_2.tx_hash]
    assert [entry.transaction_index for entry in results] == [1, 4]
    assert [entry.changes[0].raw_units for entry in results] == [-9, -4]


def test_multiple_transactions_without_indices_keep_observed_order() -> None:
    """Missing indices remain missing, never invented for FIFO."""
    tx_2 = move("second", tx_no=2)
    tx_1 = move("first", tx_no=1)
    result = calculate_wallet_asset_deltas(group_movements_by_transaction([tx_2, tx_1]), WALLET)
    assert [entry.transaction.tx_hash for entry in result] == [tx_2.tx_hash, tx_1.tx_hash]
    assert all(entry.transaction_index is None for entry in result)


def test_batch_empty_history_validates_wallet() -> None:
    """Even an empty batch rejects malformed wallet addresses."""
    assert calculate_wallet_asset_deltas([], WALLET) == ()
    with pytest.raises(EthereumNormalizationError, match="wallet_address"):
        calculate_wallet_asset_deltas([], "wallet")


def test_batch_accepts_iterators() -> None:
    """Batch projection does not materialize all groups before iterating."""
    groups = group_movements_by_transaction([move("a"), move("b", tx_no=2)])
    assert len(calculate_wallet_asset_deltas(iter(groups), WALLET)) == 2


@pytest.mark.parametrize("bad", ["x", "0x", "0x" + "gg" * 20, "0x" + "aa" * 19, None, 23])
def test_invalid_wallet_is_rejected(bad: object) -> None:
    """Malformed wallet addresses must not be silently treated as unrelated.

    :param bad: Invalid address candidate.
    """
    with pytest.raises((EthereumNormalizationError, TypeError)):
        calculate_wallet_asset_delta(group(move("a")), bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("raw", [0, True, 1.0, "1", None])
def test_invalid_raw_delta_is_rejected(raw: object) -> None:
    """Typed model rejects zero, booleans, and non-integer amounts.

    :param raw: Invalid raw amount candidate.
    """
    with pytest.raises((ValueError, TypeError)):
        RawAssetDelta(TOKEN_A, raw)  # type: ignore[arg-type]


def test_invalid_raw_delta_asset_id_rejected() -> None:
    """Typed model requires AssetId, not a ticker or address string."""
    with pytest.raises(TypeError, match="asset_id"):
        RawAssetDelta("LINK", 1)  # type: ignore[arg-type]


def test_wrong_group_type_is_rejected() -> None:
    """Do not accidentally treat raw movements as grouped transactions."""
    with pytest.raises(TypeError, match="group"):
        calculate_wallet_asset_delta(move("a"), WALLET)  # type: ignore[arg-type]


def test_wallet_asset_delta_rejects_duplicate_assets() -> None:
    """The result model cannot report contradictory per-asset totals."""
    base = delta(move("a"))
    with pytest.raises(EthereumNormalizationError, match="duplicate asset"):
        WalletAssetDelta(base.transaction, WALLET, (base.changes[0], base.changes[0]))


def test_wallet_asset_delta_rejects_invalid_wallet_address() -> None:
    """Manually creating a result must not bypass Ethereum address validation."""
    base = delta(move("a"))
    with pytest.raises(EthereumNormalizationError, match="wallet_address"):
        WalletAssetDelta(base.transaction, "not-an-address", base.changes)


def test_wallet_asset_delta_rejects_invalid_transaction_index() -> None:
    """Boolean and negative indices must not masquerade as EVM positions."""
    base = delta(move("a"))
    with pytest.raises(EthereumNormalizationError, match="transaction_index"):
        WalletAssetDelta(base.transaction, WALLET, base.changes, transaction_index=True)
    with pytest.raises(EthereumNormalizationError, match="transaction_index"):
        WalletAssetDelta(base.transaction, WALLET, base.changes, transaction_index=-1)


def test_wallet_asset_delta_rejects_wrong_transaction_or_changes_type() -> None:
    """Model instances cannot hold untyped transactions or lists of changes."""
    base = delta(move("a"))
    with pytest.raises(TypeError, match="transaction"):
        WalletAssetDelta("tx", WALLET, ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="changes"):
        WalletAssetDelta(base.transaction, WALLET, [base.changes[0]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RawAssetDelta"):
        WalletAssetDelta(base.transaction, WALLET, ("token",))  # type: ignore[arg-type]


def test_wallet_asset_delta_rejects_wrong_chain_even_when_zero_net() -> None:
    """Cross-chain movement facts cannot sneak into an Ethereum transaction."""
    foreign = move("foreign", asset=AssetId(728126428, TOKEN_A.contract_address))
    with pytest.raises(EthereumNormalizationError, match="Mainnet"):
        delta(foreign)


def test_wallet_asset_delta_rejects_wrong_chain_result() -> None:
    """A manually constructed result cannot contain another chain's asset."""
    base = delta(move("a"))
    with pytest.raises(EthereumNormalizationError, match="Mainnet"):
        WalletAssetDelta(
            base.transaction,
            WALLET,
            (RawAssetDelta(AssetId(10, TOKEN_A.contract_address), 5),),
        )


def test_raw_delta_requires_valid_metadata_for_normalization() -> None:
    """Never interpret an unknown decimal precision as 18 implicitly."""
    base = delta(move("a"))
    with pytest.raises(TypeError, match="metadata"):
        base.changes[0].to_asset_amount(None)  # type: ignore[arg-type]


def test_inputs_immutable_and_no_mutation() -> None:
    """The calculation is pure and does not change movements or its grouping."""
    item = move("a", raw=50)
    tx_group = group(item)
    result = calculate_wallet_asset_delta(tx_group, WALLET)
    assert tx_group.movements == (item,)
    assert item.raw_amount == 50
    assert isinstance(result.changes, tuple)
    assert values(result) == {TOKEN_A: -50}

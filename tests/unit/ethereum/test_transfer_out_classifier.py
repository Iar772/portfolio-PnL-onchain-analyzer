"""Tests for conservative direct outgoing-transfer classification."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from portfolio_analyzer.domain.accounting import consume_fifo
from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.domain.events import TransferOut
from portfolio_analyzer.domain.lots import Lot
from portfolio_analyzer.domain.money import UsdAmount
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
    group_movements_by_transaction,
)
from portfolio_analyzer.ethereum.rpc_models import RpcLog, RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.transfer_out import classify_transfer_out
from portfolio_analyzer.ethereum.wallet_deltas import (
    WalletAssetDelta,
    calculate_wallet_asset_delta,
)

WALLET = "0x" + "aa" * 20
RECIPIENT = "0x" + "bb" * 20
OUTSIDER = "0x" + "cc" * 20
TOKEN = AssetId(1, "0x" + "dd" * 20)
OTHER_TOKEN = AssetId(1, "0x" + "ee" * 20)
ETH = AssetId(1, None)
TX_HASH = "0x" + "f0" * 32
AT = datetime(2026, 10, 21, 14, 30, tzinfo=UTC)
METADATA = {
    TOKEN: AssetMetadata("LINK", "Chainlink", 18),
    OTHER_TOKEN: AssetMetadata("USD", "Dollar", 6),
}
type Facts = tuple[TransactionMovementGroup, WalletAssetDelta, RpcTransaction, RpcReceipt]


def movement(
    name: str = "out",
    *,
    asset: AssetId = TOKEN,
    amount: int = 50 * 10**18,
    sender: str = WALLET,
    recipient: str = RECIPIENT,
    category: MovementCategory | None = None,
) -> AssetMovement:
    """Create a canonical movement to or from an Ethereum wallet.

    :param name: Unique movement identifier.
    :param asset: Contract identity or native ETH.
    :param amount: Transfer amount in integer base units.
    :param sender: On-chain transfer sender.
    :param recipient: On-chain transfer recipient.
    :param category: Optional Ethereum movement category.
    :return: Immutable validated movement.
    """
    return AssetMovement(
        transfer_id=name,
        tx_hash=TX_HASH,
        block_number=123,
        timestamp=AT,
        from_address=sender,
        to_address=recipient,
        asset_id=asset,
        raw_amount=amount,
        category=category
        if category is not None
        else MovementCategory.EXTERNAL
        if asset.is_native
        else MovementCategory.ERC20,
    )


def calldata(recipient: str = RECIPIENT, amount: int = 50 * 10**18) -> bytes:
    """Encode the standard ERC-20 transfer(address,uint256) function.

    :param recipient: Transfer recipient.
    :param amount: Exact integer ERC-20 raw amount.
    :return: ABI selector followed by padded receiver and amount.
    """
    return (
        bytes.fromhex("a9059cbb")
        + bytes(12)
        + bytes.fromhex(recipient[2:])
        + amount.to_bytes(32, "big")
    )


def scenario(
    *movements: AssetMovement,
    wallet: str = WALLET,
    sender: str = WALLET,
    to: str | None = None,
    value_wei: int | None = None,
    input_data: bytes | None = None,
    success: bool | None = True,
    transaction_index: int | None = None,
) -> Facts:
    """Create consistent grouped movements, wallet deltas and mined RPC facts.

    :param movements: Movements belonging to one mined transaction.
    :param wallet: Target wallet address.
    :param sender: Ethereum transaction originator.
    :param to: RPC transaction recipient, optionally inferred.
    :param value_wei: RPC ETH value, optionally inferred.
    :param input_data: Transaction calldata, optionally inferred.
    :param success: Receipt execution status.
    :param transaction_index: Known canonical in-block index.
    :return: All four consistent classifier inputs.
    """
    if not movements:
        movements = (movement(),)
    first = movements[0]
    group = group_movements_by_transaction(
        movements,
        transaction_indices={TX_HASH: transaction_index} if transaction_index is not None else None,
    )[0]
    delta = calculate_wallet_asset_delta(group, wallet)
    if to is None:
        to = first.to_address if first.asset_id.is_native else first.asset_id.contract_address
    if value_wei is None:
        value_wei = first.raw_amount if first.asset_id.is_native else 0
    if input_data is None:
        input_data = (
            b"" if first.asset_id.is_native else calldata(first.to_address, first.raw_amount)
        )
    tx = RpcTransaction(TX_HASH, sender, to, 123, value_wei, input_data)
    receipt = RpcReceipt(TX_HASH, 123, success, 21000, 10**9, ())
    return group, delta, tx, receipt


def classify(
    facts: Facts,
    *,
    metadata: dict[AssetId, AssetMetadata] | None = None,
) -> TransferOut | None:
    """Call the production classifier with trusted metadata.

    :param facts: Group, deltas, transaction, receipt.
    :param metadata: Optional known token metadata map.
    :return: A safely classified outgoing transfer, or None.
    """
    return classify_transfer_out(*facts, token_metadata=METADATA if metadata is None else metadata)


def test_direct_erc20_outgoing_creates_transfer_out_without_proceeds() -> None:
    """An exact ERC-20 transfer is a disposal, never a synthetic sale."""
    result = classify(scenario(movement()))
    assert isinstance(result, TransferOut)
    assert result.amount.asset_id == TOKEN
    assert result.amount.quantity == Decimal("50")
    assert result.metadata.tx_hash == TX_HASH
    assert result.metadata.block_number == 123
    assert result.metadata.timestamp == AT
    assert not hasattr(result, "proceeds")
    assert not hasattr(result, "proceeds_usd")
    assert not hasattr(result, "cost_basis")


def test_native_outgoing_one_wei_without_fee_in_trade_leg() -> None:
    """A gas-paying wallet sends only the transferred ETH, excluding gas."""
    result = classify(scenario(movement(asset=ETH, amount=1)))
    assert isinstance(result, TransferOut)
    assert result.amount.asset_id == ETH
    assert result.amount.quantity == Decimal("0.000000000000000001")


@pytest.mark.parametrize("amount", [1, 10**18, 10**36 + 1])
def test_native_conversion_preserves_exact_wei(amount: int) -> None:
    """Whole-token conversion is exact for small and huge balances.

    :param amount: Positive raw wei sent.
    """
    result = classify(scenario(movement(asset=ETH, amount=amount)))
    assert isinstance(result, TransferOut)
    assert result.amount.quantity == Decimal(f"{amount}e-18")


@pytest.mark.parametrize("decimals,amount", [(0, 123), (6, 1), (18, 10**36 + 11), (255, 1)])
def test_verified_erc20_decimals_preserve_precision(decimals: int, amount: int) -> None:
    """Decimals from metadata, not symbols or provider hints, determine quantity.

    :param decimals: Verified ERC-20 decimal precision.
    :param amount: Positive raw token amount.
    """
    with localcontext() as context:
        context.prec = 7  # Deliberately hostile ambient precision.
        result = classify(
            scenario(movement(amount=amount)),
            metadata={TOKEN: AssetMetadata("TKN", "Token", decimals)},
        )
    assert isinstance(result, TransferOut)
    assert result.amount.quantity == Decimal(f"{amount}e-{decimals}")


def test_same_symbol_other_contract_never_changes_asset_identity() -> None:
    """Ticker collisions and renames cannot change the consumed asset."""
    a = classify(scenario(movement()), metadata={TOKEN: AssetMetadata("SAME", "A", 6)})
    b = classify(scenario(movement()), metadata={TOKEN: AssetMetadata("DIFFERENT", "B", 6)})
    assert isinstance(a, TransferOut)
    assert isinstance(b, TransferOut)
    assert a.amount.asset_id == b.amount.asset_id == TOKEN


def test_transfer_out_is_compatible_with_later_fifo_inventory_reduction() -> None:
    """FIFO can consume the exact transferred quantity without making proceeds."""
    result = classify(scenario(movement(amount=3 * 10**18)))
    assert isinstance(result, TransferOut)
    acquired = Lot.open(
        replace(result.amount, quantity=Decimal("5")),
        cost_basis=UsdAmount(Decimal("125")),
        acquired_at=AT - timedelta(days=2),
    )
    consumed = consume_fifo([acquired], result.amount)
    assert consumed.disposal.quantity == Decimal("3")
    assert consumed.updated_lots[0].remaining_quantity == Decimal("2")
    assert consumed.consumptions[0].cost_basis == UsdAmount(Decimal("75"))
    assert not hasattr(consumed, "proceeds")


def test_unknown_basis_fifo_transfer_out_stays_unknown() -> None:
    """An outbound movement consumes unknown basis without fabricating a sale."""
    result = classify(scenario(movement(amount=2 * 10**18)))
    assert isinstance(result, TransferOut)
    original = Lot.open(
        replace(result.amount, quantity=Decimal("5")),
        cost_basis=None,
        acquired_at=AT - timedelta(days=1),
    )
    updated = consume_fifo([original], result.amount)
    assert updated.updated_lots[0].remaining_quantity == Decimal("3")
    assert updated.consumptions[0].cost_basis is None


def test_known_transaction_index_is_preserved_by_grouping_not_event() -> None:
    """Classification accepts verified in-block position and retains provenance."""
    facts = scenario(movement(), transaction_index=7)
    assert facts[0].transaction_index == facts[1].transaction_index == 7
    result = classify(facts)
    assert isinstance(result, TransferOut)
    assert result.metadata.timestamp == AT


@pytest.mark.parametrize("success", [False, None])
def test_failed_or_legacy_receipt_not_classified(success: bool | None) -> None:
    """Reverted or status-unknown execution cannot prove a sent transfer.

    :param success: Receipt execution status.
    """
    assert classify(scenario(movement(), success=success)) is None


@pytest.mark.parametrize(
    "fake",
    [
        movement("self", recipient=WALLET),
        movement("incoming", sender=RECIPIENT, recipient=WALLET),
        movement("other", sender=OUTSIDER, recipient=RECIPIENT),
        movement("empty", amount=0),
        movement("internal", asset=ETH, category=MovementCategory.INTERNAL),
    ],
)
def test_other_movements_are_not_simple_outgoing(fake: AssetMovement) -> None:
    """Incoming, unrelated, empty, self and internal transfers do not qualify.

    :param fake: Non-simple movement.
    """
    assert classify(scenario(fake)) is None


def test_native_contract_execution_logs_are_not_assumed_plain_transfer() -> None:
    """A payable contract with side effects needs more than plain value facts."""
    group, delta, tx, receipt = scenario(movement(asset=ETH, amount=10**18))
    receipt = replace(
        receipt,
        logs=(RpcLog(address=RECIPIENT, topics=(), data=b"nonfinancial-event"),),
    )
    assert classify((group, delta, tx, receipt)) is None


def test_two_leg_swap_is_not_classified_as_outgoing_transfer() -> None:
    """Even an outbound transfer plus an inbound one does not prove a transfer."""
    out = movement("out")
    inbound = movement(
        "in", asset=OTHER_TOKEN, amount=25 * 10**6, sender=OUTSIDER, recipient=WALLET
    )
    assert classify(scenario(out, inbound)) is None


def test_two_outgoing_movements_are_not_a_simple_transfer() -> None:
    """Multiple outgoing legs require separate economic classification."""
    assert classify(scenario(movement("one"), movement("two", asset=OTHER_TOKEN))) is None


def test_outbound_to_contract_router_not_classified() -> None:
    """Router calls do not become synthetic direct ERC-20 transfers."""
    assert classify(scenario(movement(), to=OUTSIDER)) is None


@pytest.mark.parametrize(
    "mutated",
    [
        lambda f: (*f[:2], replace(f[2], from_address=OUTSIDER), f[3]),
        lambda f: (*f[:2], replace(f[2], to_address=OUTSIDER), f[3]),
        lambda f: (*f[:2], replace(f[2], value_wei=1), f[3]),
        lambda f: (*f[:2], replace(f[2], input_data=b""), f[3]),
        lambda f: (*f[:2], replace(f[2], input_data=calldata(OUTSIDER)), f[3]),
        lambda f: (*f[:2], replace(f[2], input_data=calldata(RECIPIENT, 12)), f[3]),
        lambda f: (*f[:2], replace(f[2], input_data=b"\x12\x34" + calldata()[2:]), f[3]),
        lambda f: (*f[:2], replace(f[2], input_data=calldata() + b"\x00"), f[3]),
        lambda f: (*f[:2], replace(f[2], input_data=calldata()[:30]), f[3]),
    ],
)
def test_erc20_bad_origin_target_or_calldata_not_classified(
    mutated: Callable[[Facts], Facts],
) -> None:
    """Call ABI must match actual receiver and raw amount exactly.

    :param mutated: Function transforming valid RPC facts.
    """
    facts = mutated(scenario(movement()))
    assert classify(facts) is None


@pytest.mark.parametrize(
    "mutated",
    [
        lambda f: (*f[:2], replace(f[2], to_address=OUTSIDER), f[3]),
        lambda f: (*f[:2], replace(f[2], from_address=OUTSIDER), f[3]),
        lambda f: (*f[:2], replace(f[2], value_wei=19), f[3]),
        lambda f: (*f[:2], replace(f[2], input_data=b"\x01"), f[3]),
    ],
)
def test_native_nonplain_tx_not_classified(mutated: Callable[[Facts], Facts]) -> None:
    """Native ETH transfers must exactly match plain transaction value.

    :param mutated: RPC transformation function.
    """
    assert classify(mutated(scenario(movement(asset=ETH, amount=20)))) is None


def test_unknown_metadata_does_not_invent_token_decimals() -> None:
    """Unknown token metadata must not silently default to 18 decimals."""
    assert classify(scenario(movement()), metadata={}) is None


def test_invalid_metadata_is_an_explicit_source_inconsistency() -> None:
    """Invalid mapping value is rejected, not quietly normalized."""
    with pytest.raises(EthereumNormalizationError, match="metadata"):
        classify_transfer_out(
            *scenario(movement()),
            token_metadata={TOKEN: "LINK"},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize(
    "argument,wrong",
    [
        (0, None),
        (1, None),
        (2, None),
        (3, None),
        (4, None),
    ],
)
def test_wrong_argument_types_are_rejected(argument: int, wrong: object) -> None:
    """Classifier must validate each top-level argument.

    :param argument: Position within five inputs.
    :param wrong: Value violating its expected type.
    """
    args = list(scenario(movement()))
    if argument == 4:
        with pytest.raises(TypeError):
            classify_transfer_out(*args, token_metadata=wrong)  # type: ignore[arg-type]
    else:
        args[argument] = wrong
        with pytest.raises(TypeError):
            classify_transfer_out(*args, token_metadata=METADATA)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value",
    [
        ("tx_hash", "0x" + "fa" * 32),
        ("block_number", 124),
        ("block_number", None),
        ("value_wei", -1),
        ("value_wei", 1.5),
        ("input_data", "0xa9059cbb"),
        ("from_address", "not hex"),
        ("to_address", "not hex"),
    ],
)
def test_corrupt_rpc_transaction_facts_raise(field: str, value: object) -> None:
    """Contradictory or malformed RPC facts fail closed, not classify as trades.

    :param field: RPC fact to mutate.
    :param value: Invalid input.
    """
    facts = scenario(movement())
    new_tx = replace(facts[2], **{field: value})
    with pytest.raises(EthereumNormalizationError):
        classify((facts[0], facts[1], new_tx, facts[3]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("tx_hash", "0x" + "ea" * 32),
        ("block_number", 124),
        ("block_number", None),
        ("success", "true"),
    ],
)
def test_corrupt_rpc_receipt_facts_raise(field: str, value: object) -> None:
    """Provenance or status mismatch must not become a false outgoing event.

    :param field: RPC receipt field to mutate.
    :param value: Malformed value.
    """
    facts = scenario(movement())
    new_receipt = replace(facts[3], **{field: value})
    with pytest.raises(EthereumNormalizationError):
        classify((facts[0], facts[1], facts[2], new_receipt))


def test_another_wallet_perspective_is_not_outgoing() -> None:
    """Incoming from the other wallet is not an outgoing event."""
    facts = scenario(movement())
    wrong_delta = calculate_wallet_asset_delta(facts[0], RECIPIENT)
    assert wrong_delta.wallet_address != facts[1].wallet_address
    # Wallet identity is supplied by the delta. From the recipient's perspective
    # this movement is incoming, so it must not become a TransferOut.
    assert classify((facts[0], wrong_delta, facts[2], facts[3])) is None


def test_incoming_classifier_remains_unchanged_after_shared_validation_refactor() -> None:
    """In and out classifiers share validation but cannot classify the same event."""
    from portfolio_analyzer.ethereum.transfer_in import classify_transfer_in

    facts = scenario(movement())
    assert classify_transfer_in(*facts, token_metadata=METADATA) is None
    assert isinstance(classify(facts), TransferOut)

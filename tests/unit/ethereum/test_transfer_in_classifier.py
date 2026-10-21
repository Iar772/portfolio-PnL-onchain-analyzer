"""Tests for conservative direct incoming-transfer classification."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.domain.events import TransferIn
from portfolio_analyzer.domain.lots import Lot
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
    group_movements_by_transaction,
)
from portfolio_analyzer.ethereum.rpc_models import RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.transfer_in import classify_transfer_in
from portfolio_analyzer.ethereum.wallet_deltas import (
    WalletAssetDelta,
    calculate_wallet_asset_delta,
)

WALLET = "0x" + "aa" * 20
SENDER = "0x" + "bb" * 20
OTHER = "0x" + "cc" * 20
TOKEN = AssetId(1, "0x" + "dd" * 20)
SECOND_TOKEN = AssetId(1, "0x" + "ee" * 20)
ETH = AssetId(1, None)
HASH = "0x" + "f0" * 32
STAMP = datetime(2026, 10, 21, 12, tzinfo=UTC)
TRUSTED = AssetMetadata("LINK", "Chainlink", 18)
TOKEN_DATA = {TOKEN: TRUSTED, SECOND_TOKEN: AssetMetadata("USD", "Dollar", 6)}


def movement(
    name: str = "incoming",
    *,
    asset: AssetId = TOKEN,
    amount: int = 50 * 10**18,
    sender: str = SENDER,
    recipient: str = WALLET,
    category: MovementCategory | None = None,
) -> AssetMovement:
    """Build one immutable movement with canonical Ethereum provenance.

    :param name: Unique movement ID.
    :param asset: Native ETH or an ERC-20 identity.
    :param amount: Base-unit amount.
    :param sender: Transfer sender address.
    :param recipient: Transfer recipient address.
    :param category: Optional explicit transfer category.
    :return: Validated movement for one mined transaction.
    """
    return AssetMovement(
        transfer_id=name,
        tx_hash=HASH,
        block_number=123,
        timestamp=STAMP,
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


def calldata(recipient: str = WALLET, amount: int = 50 * 10**18) -> bytes:
    """Construct an exact standard ERC-20 transfer(address,uint256) call.

    :param recipient: ABI-encoded receiver.
    :param amount: ABI-encoded token base-unit quantity.
    :return: Standard Ethereum transaction calldata.
    """
    return (
        bytes.fromhex("a9059cbb")
        + bytes(12)
        + bytes.fromhex(recipient[2:])
        + amount.to_bytes(32, "big")
    )


def facts(
    *movements: AssetMovement,
    wallet: str = WALLET,
    sender: str = SENDER,
    to: str | None = None,
    value_wei: int | None = None,
    input_data: bytes | None = None,
    success: bool | None = True,
) -> tuple[TransactionMovementGroup, WalletAssetDelta, RpcTransaction, RpcReceipt]:
    """Build E5.1, E5.2 and matching RPC facts for classification.

    :param movements: Transfer facts of one mined transaction.
    :param wallet: Wallet whose asset deltas are calculated.
    :param sender: Ethereum transaction originator.
    :param to: Transaction recipient, or infer from transfer category.
    :param value_wei: Native ETH transaction amount, or infer.
    :param input_data: Calldata, or infer from a direct token transfer.
    :param success: Optional receipt execution status.
    :return: Group, wallet delta, RPC transaction and receipt.
    """
    (group,) = group_movements_by_transaction(movements)
    change = calculate_wallet_asset_delta(group, wallet)
    primary = movements[0]
    if to is None:
        to = primary.to_address if primary.asset_id.is_native else primary.asset_id.contract_address
    if value_wei is None:
        value_wei = primary.raw_amount if primary.asset_id.is_native else 0
    if input_data is None:
        input_data = (
            b"" if primary.asset_id.is_native else calldata(primary.to_address, primary.raw_amount)
        )
    transaction = RpcTransaction(
        tx_hash=HASH,
        from_address=sender,
        to_address=to,
        block_number=123,
        value_wei=value_wei,
        input_data=input_data,
    )
    receipt = RpcReceipt(
        tx_hash=HASH,
        block_number=123,
        success=success,
        gas_used=21000,
        effective_gas_price_wei=10**9,
        logs=(),
    )
    return group, change, transaction, receipt


def classify(
    scenario: tuple[TransactionMovementGroup, WalletAssetDelta, RpcTransaction, RpcReceipt],
    *,
    metadata: dict[AssetId, AssetMetadata] | None = None,
) -> TransferIn | None:
    """Invoke the production classifier with trusted token metadata.

    :param scenario: Related transaction and RPC facts.
    :param metadata: Token metadata, or defaults for supported test tokens.
    :return: TransferIn for a proven simple inbound transfer, otherwise None.
    """
    return classify_transfer_in(
        *scenario,
        token_metadata=TOKEN_DATA if metadata is None else metadata,
    )


def test_plain_incoming_erc20_produces_domain_transfer_in_with_exact_quantity() -> None:
    """One standard ERC-20 transfer is classified without priced assumptions."""
    result = classify(facts(movement()))
    assert isinstance(result, TransferIn)
    assert result.metadata.tx_hash == HASH
    assert result.metadata.block_number == 123
    assert result.metadata.timestamp == STAMP
    assert result.amount.asset_id == TOKEN
    assert result.amount.quantity == Decimal("50")


def test_transfer_in_opens_unknown_cost_basis_lot_not_zero() -> None:
    """Future FIFO lots can preserve unknown acquisition basis explicitly."""
    result = classify(facts(movement()))
    assert isinstance(result, TransferIn)
    assert not hasattr(result, "cost_basis_usd")
    lot = Lot.open(result.amount, cost_basis=None, acquired_at=result.metadata.timestamp)
    assert lot.original_cost_basis is None
    assert lot.remaining_cost_basis is None
    assert lot.is_basis_known is False


def test_native_incoming_plain_eth_transfer_uses_18_decimals() -> None:
    """Native ETH receives preserve wei precision without token metadata."""
    result = classify(facts(movement(asset=ETH, amount=1)))
    assert isinstance(result, TransferIn)
    assert result.amount.asset_id == ETH
    assert result.amount.quantity == Decimal("0.000000000000000001")


@pytest.mark.parametrize("amount", [1, 10**18, 10**36 + 1])
def test_native_value_is_converted_exactly_without_context_rounding(amount: int) -> None:
    """Conversion never uses binary floats or rounds large numbers.

    :param amount: Native wei value used in the transfer.
    """
    result = classify(facts(movement(asset=ETH, amount=amount)))
    assert isinstance(result, TransferIn)
    assert result.amount.quantity == Decimal(f"{amount}e-18")


@pytest.mark.parametrize("decimals,amount", [(0, 123), (6, 1), (18, 10**30), (255, 1)])
def test_metadata_decimals_are_used_exactly(decimals: int, amount: int) -> None:
    """Token quantity respects verified decimals even for extreme precision.

    :param decimals: ERC-20 token precision.
    :param amount: Raw amount to convert.
    """
    result = classify(
        facts(movement(amount=amount)),
        metadata={TOKEN: AssetMetadata("TKN", "Token", decimals)},
    )
    assert isinstance(result, TransferIn)
    assert result.amount.quantity == Decimal(f"{amount}e-{decimals}")


def test_asset_identity_comes_from_chain_and_contract_not_symbol() -> None:
    """Changing ticker metadata cannot change the acquired asset identity."""
    scenario = facts(movement())
    a = classify(scenario, metadata={TOKEN: AssetMetadata("USDT", "Tether", 6)})
    b = classify(scenario, metadata={TOKEN: AssetMetadata("OTHER", "Other Token", 6)})
    assert isinstance(a, TransferIn)
    assert isinstance(b, TransferIn)
    assert a.amount.asset_id == b.amount.asset_id == TOKEN


def test_unknown_token_metadata_means_unclassified_not_zero_quantity() -> None:
    """No decimals leads to no TransferIn rather than manufactured precision."""
    assert classify(facts(movement()), metadata={}) is None


def test_wrong_metadata_type_cannot_be_silently_accepted() -> None:
    """Corrupted metadata results fail loudly rather than assigning quantities."""
    with pytest.raises(EthereumNormalizationError, match="metadata"):
        classify_transfer_in(
            *facts(movement()),
            token_metadata={TOKEN: "18"},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize("success", [False, None])
def test_failed_or_unverified_receipt_is_not_classified(success: bool | None) -> None:
    """Only confirmed successful execution may become TransferIn.

    :param success: On-chain receipt outcome.
    """
    assert classify(facts(movement(), success=success)) is None


@pytest.mark.parametrize(
    "sender,recipient",
    [
        (WALLET, WALLET),
        (WALLET, OTHER),
        (SENDER, OTHER),
    ],
)
def test_self_outbound_and_unrelated_movement_are_not_incoming(sender: str, recipient: str) -> None:
    """Only independent inbound transfers can be recognized.

    :param sender: On-chain movement sender.
    :param recipient: On-chain movement recipient.
    """
    assert classify(facts(movement(sender=sender, recipient=recipient), sender=sender)) is None


def test_external_sender_must_match_the_transaction_originator() -> None:
    """An EOA claim or DEX payout should not appear as a plain transfer."""
    assert classify(facts(movement(), sender=WALLET)) is None
    assert classify(facts(movement(), sender=OTHER)) is None


def test_negative_wallet_delta_is_not_classified_as_transfer_in() -> None:
    """A debit-only transaction must be classified by another handler."""
    assert classify(facts(movement(sender=WALLET, recipient=OTHER), sender=WALLET)) is None


def test_mixed_eth_and_erc20_swap_like_transaction_is_not_transfer_in() -> None:
    """Both inflow and outflow in one transaction are not simple deposits."""
    eth_out = movement("out", asset=ETH, sender=WALLET, recipient=OTHER, amount=10**18)
    token_in = movement("in", asset=TOKEN, amount=50 * 10**18)
    assert classify(facts(eth_out, token_in, sender=WALLET)) is None


def test_multiple_incoming_same_token_stays_for_later_classifier() -> None:
    """A batch of transfers is not automatically a simple inbound event."""
    assert classify(facts(movement("a"), movement("b", amount=1))) is None


def test_multiple_assets_incoming_are_not_simple_transfer() -> None:
    """Multiple asset legs may indicate a contract or distribution operation."""
    assert classify(facts(movement("a"), movement("b", asset=SECOND_TOKEN))) is None


def test_self_transfer_plus_incoming_requires_more_evidence() -> None:
    """Net positive alone is insufficient when extra movements exist."""
    assert (
        classify(
            facts(
                movement("self", sender=WALLET, recipient=WALLET),
                movement("in", sender=SENDER, recipient=WALLET),
            )
        )
        is None
    )


def test_zero_raw_amount_does_not_create_transfer_in() -> None:
    """Zero-value event must not manufacture an acquisition."""
    assert classify(facts(movement(amount=0))) is None


def test_internal_eth_payment_is_not_assumed_to_be_transfer_in() -> None:
    """Contract payouts require different analysis than plain ETH transfers."""
    assert classify(facts(movement(asset=ETH, category=MovementCategory.INTERNAL))) is None


@pytest.mark.parametrize("incorrect_value", [0, 1, 10**18])
def test_native_rpc_amount_must_match_exactly(incorrect_value: int) -> None:
    """Native ETH TransferIn requires exact tx.value matching observed movement.

    :param incorrect_value: RPC value for a 2 ETH transfer.
    """
    assert (
        classify(facts(movement(asset=ETH, amount=2 * 10**18), value_wei=incorrect_value)) is None
    )


def test_native_plain_transfer_with_calldata_not_classified() -> None:
    """A contract call carrying ETH is not necessarily a wallet transfer."""
    assert classify(facts(movement(asset=ETH), input_data=b"\x01")) is None


def test_native_rpc_recipient_must_be_wallet() -> None:
    """Native transaction recipient must match incoming movement target."""
    assert classify(facts(movement(asset=ETH), to=OTHER)) is None


@pytest.mark.parametrize(
    "wrong_input",
    [
        b"",
        b"\x01" * 68,
        calldata(OTHER),
        calldata(WALLET, 10),
        calldata() + b"\x00",
        calldata()[:-1],
        b"\xa9\x05\x9c\xbb" + bytes.fromhex("ff" * 12) + calldata()[16:],
    ],
)
def test_erc20_wrong_calldata_cannot_be_auto_classified(wrong_input: bytes) -> None:
    """Only exact standard transfer(address,uint256) calldata is accepted.

    :param wrong_input: Nonmatching calldata sequence.
    """
    assert classify(facts(movement(), input_data=wrong_input)) is None


def test_erc20_nonzero_eth_value_is_not_plain_token_transfer() -> None:
    """Token transfer carrying ETH requires separate classification."""
    assert classify(facts(movement(), value_wei=1)) is None


def test_erc20_wrong_transaction_recipient_is_not_direct_transfer() -> None:
    """DEX contract execution cannot masquerade as direct ERC-20 transfer."""
    assert classify(facts(movement(), to=OTHER)) is None


def test_erc20_transfer_from_not_classified() -> None:
    """TransferFrom requires more evidence than the direct transfer selector."""
    raw = bytes.fromhex("23b872dd") + bytes(32 * 3)
    assert classify(facts(movement(), input_data=raw)) is None


@pytest.mark.parametrize("mutated", ["tx_hash", "block_number", "delta", "index"])
def test_conflicting_transaction_facts_raise_error(mutated: str) -> None:
    """Conflicting RPC data or delta cannot silently produce a valid event.

    :param mutated: Fact source to invalidate.
    """
    group, delta, tx, receipt = facts(movement())
    if mutated == "tx_hash":
        tx = replace(tx, tx_hash="0x" + "ab" * 32)
    elif mutated == "block_number":
        receipt = replace(receipt, block_number=999)
    elif mutated == "delta":
        delta = replace(delta, transaction=replace(delta.transaction, block_number=456))
    else:
        delta = replace(delta, transaction_index=9)
    with pytest.raises(EthereumNormalizationError):
        classify_transfer_in(group, delta, tx, receipt, token_metadata=TOKEN_DATA)


@pytest.mark.parametrize("bad", [-1, True, 3.0])
def test_invalid_rpc_eth_value_raises_error(bad: object) -> None:
    """Invalid raw transaction value must not be treated as a valid transfer.

    :param bad: Malformed value_wei from an inconsistent RPC implementation.
    """
    group, delta, tx, receipt = facts(movement())
    tx = replace(tx, value_wei=bad)
    with pytest.raises(EthereumNormalizationError, match="value_wei"):
        classify_transfer_in(group, delta, tx, receipt, token_metadata=TOKEN_DATA)


@pytest.mark.parametrize("field", ["from_address", "to_address", "input_data"])
def test_invalid_rpc_sender_recipient_or_input_rejected(field: str) -> None:
    """Malformed RPC objects cannot silently produce classification.

    :param field: Raw transaction field to invalidate.
    """
    group, delta, tx, receipt = facts(movement())
    if field == "from_address":
        tx = replace(tx, from_address="bad")
    elif field == "to_address":
        tx = replace(tx, to_address="bad")
    else:
        tx = replace(tx, input_data="bad")
    with pytest.raises(EthereumNormalizationError):
        classify_transfer_in(group, delta, tx, receipt, token_metadata=TOKEN_DATA)


def test_receipt_tx_hash_mismatch_is_explicit_error() -> None:
    """Inconsistent success receipt cannot be used as evidence."""
    group, delta, tx, receipt = facts(movement())
    receipt = replace(receipt, tx_hash="0x" + "ab" * 32)
    with pytest.raises(EthereumNormalizationError):
        classify_transfer_in(group, delta, tx, receipt, token_metadata=TOKEN_DATA)


def test_metadata_mapping_is_mandatory() -> None:
    """No generic fallback to eighteen decimals for ERC-20 tokens."""
    group, delta, tx, receipt = facts(movement())
    with pytest.raises(TypeError, match="mapping"):
        classify_transfer_in(
            group,
            delta,
            tx,
            receipt,
            token_metadata=None,  # type: ignore[arg-type]
        )


def test_input_objects_are_not_mutated() -> None:
    """Classification is deterministic and does not modify source facts."""
    scenario = facts(movement())
    before = repr(scenario)
    assert isinstance(classify(scenario), TransferIn)
    assert repr(scenario) == before


def test_rpc_address_casing_does_not_change_classification() -> None:
    """RPC address capitalization cannot affect EVM address identity."""
    group, delta, tx, receipt = facts(movement())
    tx = replace(tx, from_address=SENDER.upper().replace("0X", "0x"))
    assert isinstance(classify((group, delta, tx, receipt)), TransferIn)


@pytest.mark.parametrize("component", ["group", "delta", "transaction", "receipt"])
def test_wrong_top_level_type_raises_explicit_error(component: str) -> None:
    """Accidental invocation with wrong component types is rejected.

    :param component: Positional argument to corrupt.
    """
    group, delta, transaction, receipt = facts(movement())
    with pytest.raises(TypeError, match=component):
        if component == "group":
            classify_transfer_in(
                "bad",  # type: ignore[arg-type]
                delta,
                transaction,
                receipt,
                token_metadata=TOKEN_DATA,
            )
        elif component == "delta":
            classify_transfer_in(
                group,
                "bad",  # type: ignore[arg-type]
                transaction,
                receipt,
                token_metadata=TOKEN_DATA,
            )
        elif component == "transaction":
            classify_transfer_in(
                group,
                delta,
                "bad",  # type: ignore[arg-type]
                receipt,
                token_metadata=TOKEN_DATA,
            )
        else:
            classify_transfer_in(
                group,
                delta,
                transaction,
                "bad",  # type: ignore[arg-type]
                token_metadata=TOKEN_DATA,
            )


def test_non_boolean_receipt_success_is_rejected() -> None:
    """An unchecked truthy string cannot be accepted as execution success."""
    group, delta, tx, receipt = facts(movement())
    receipt = replace(receipt, success=1)
    with pytest.raises(EthereumNormalizationError, match="success"):
        classify_transfer_in(group, delta, tx, receipt, token_metadata=TOKEN_DATA)


@pytest.mark.parametrize("component", ["transaction", "receipt"])
@pytest.mark.parametrize("bad_hash", [123, "0x1234", "none"])
def test_malformed_rpc_transaction_hash_is_rejected(component: str, bad_hash: object) -> None:
    """Malformed RPC hashes raise domain normalization errors, not AttributeError.

    :param component: Hash-bearing RPC model to alter.
    :param bad_hash: Invalid hash from a faulty fake or provider.
    """
    group, delta, tx, receipt = facts(movement())
    if component == "transaction":
        tx = replace(tx, tx_hash=bad_hash)
    else:
        receipt = replace(receipt, tx_hash=bad_hash)
    with pytest.raises(EthereumNormalizationError, match="hash"):
        classify_transfer_in(group, delta, tx, receipt, token_metadata=TOKEN_DATA)


@pytest.mark.parametrize("component", ["transaction", "receipt"])
@pytest.mark.parametrize("bad_block", [True, None, 1.0])
def test_malformed_rpc_block_is_rejected(component: str, bad_block: object) -> None:
    """Only mined integer block heights qualify as confirmed chain facts.

    :param component: RPC object whose block height is corrupted.
    :param bad_block: Invalid block representation.
    """
    group, delta, tx, receipt = facts(movement())
    if component == "transaction":
        tx = replace(tx, block_number=bad_block)
    else:
        receipt = replace(receipt, block_number=bad_block)
    with pytest.raises(EthereumNormalizationError, match="block"):
        classify_transfer_in(group, delta, tx, receipt, token_metadata=TOKEN_DATA)

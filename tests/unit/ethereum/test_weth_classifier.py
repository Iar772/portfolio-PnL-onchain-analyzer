"""Exact, conservative Ethereum Mainnet WETH9 wrap and unwrap classification."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, localcontext

import pytest

from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.events import UnknownEvent, Unwrap, Wrap
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
    group_movements_by_transaction,
)
from portfolio_analyzer.ethereum.rpc_models import RpcLog, RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.wallet_deltas import (
    WalletAssetDelta,
    calculate_wallet_asset_delta,
)
from portfolio_analyzer.ethereum.weth import classify_eth_weth

WALLET = "0x" + "aa" * 20
THIRD_PARTY = "0x" + "bb" * 20
WETH_ADDR = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
ETH = AssetId(1, None)
WETH = AssetId(1, WETH_ADDR)
OTHER = AssetId(1, "0x" + "cc" * 20)
TX_HASH = "0x" + "fe" * 32
TIMESTAMP = datetime(2026, 10, 20, 16, 0, tzinfo=UTC)
DEPOSIT_TOPIC = bytes.fromhex("e1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c")
WITHDRAW_TOPIC = bytes.fromhex("7fcf532c15f0a6db0bd6d0e038bea71d30d808c7d98cb3bf7268a95bf5081b65")
DEPOSIT = bytes.fromhex("d0e30db0")
WITHDRAW = bytes.fromhex("2e1a7d4d")


def movement(
    *,
    amount: int = 5 * 10**17,
    direction: str = "wrap",
    asset: AssetId = ETH,
    sender: str | None = None,
    receiver: str | None = None,
    ident: str = "native",
    category: MovementCategory | None = None,
) -> AssetMovement:
    """Build an observed native or token movement for one WETH9 transaction.

    :param amount: Raw minimal-unit amount.
    :param direction: Desired WETH conversion scenario.
    :param asset: On-chain asset identity.
    :param sender: Optional sender override.
    :param receiver: Optional recipient override.
    :param ident: Unique movement identifier.
    :param category: Optional provider-neutral movement category override.
    :return: Immutable on-chain movement.
    """
    return AssetMovement(
        transfer_id=ident,
        tx_hash=TX_HASH,
        block_number=5_000,
        timestamp=TIMESTAMP,
        from_address=sender
        if sender is not None
        else (WALLET if direction == "wrap" else WETH_ADDR),
        to_address=receiver
        if receiver is not None
        else (WETH_ADDR if direction == "wrap" else WALLET),
        asset_id=asset,
        raw_amount=amount,
        category=category
        if category is not None
        else (
            MovementCategory.ERC20
            if not asset.is_native
            else MovementCategory.EXTERNAL
            if direction == "wrap"
            else MovementCategory.INTERNAL
        ),
    )


def log(
    direction: str = "wrap",
    *,
    amount: int = 5 * 10**17,
    beneficiary: str = WALLET,
) -> RpcLog:
    """Build an exact WETH9 receipt log with an indexed wallet.

    :param direction: Wrap or unwrap.
    :param amount: Exact raw wad field.
    :param beneficiary: Account in the indexed event parameter.
    :return: Immutable raw RPC log model.
    """
    return RpcLog(
        address=WETH_ADDR,
        topics=(
            DEPOSIT_TOPIC if direction == "wrap" else WITHDRAW_TOPIC,
            bytes(12) + bytes.fromhex(beneficiary[2:]),
        ),
        data=amount.to_bytes(32, "big"),
    )


def case(
    direction: str = "wrap",
    *,
    amount: int = 5 * 10**17,
    fallback: bool = False,
    movements: tuple[AssetMovement, ...] | None = None,
    logs: tuple[RpcLog, ...] | None = None,
    success: bool | None = True,
    wallet: str = WALLET,
) -> tuple[TransactionMovementGroup, WalletAssetDelta, RpcTransaction, RpcReceipt]:
    """Make consistent transaction, history, delta, and receipt facts.

    :param direction: Conversion direction.
    :param amount: Exact raw wei/wad.
    :param fallback: Use empty calldata for WETH9 payable fallback.
    :param movements: Override source transfer history.
    :param logs: Override emitted receipt logs.
    :param success: Receipt status flag.
    :param wallet: Analyzed wallet address.
    :return: Four provider-neutral classifier inputs.
    """
    if movements is None:
        movements = (movement(direction=direction, amount=amount),)
    group = group_movements_by_transaction(movements)[0]
    delta = calculate_wallet_asset_delta(group, wallet)
    tx = RpcTransaction(
        tx_hash=TX_HASH,
        from_address=WALLET,
        to_address=WETH_ADDR,
        block_number=5_000,
        value_wei=amount if direction == "wrap" else 0,
        input_data=(b"" if fallback else DEPOSIT)
        if direction == "wrap"
        else WITHDRAW + amount.to_bytes(32, "big"),
    )
    receipt = RpcReceipt(
        tx_hash=TX_HASH,
        block_number=5_000,
        success=success,
        gas_used=65_000,
        effective_gas_price_wei=2_000_000_000,
        logs=logs if logs is not None else (log(direction, amount=amount),),
    )
    return group, delta, tx, receipt


def classify(
    facts: tuple[TransactionMovementGroup, WalletAssetDelta, RpcTransaction, RpcReceipt],
) -> Wrap | Unwrap | UnknownEvent | None:
    """Run production classification against independently modeled facts.

    :param facts: Group, raw delta, RPC transaction and receipt.
    :return: One economic event or None.
    """
    return classify_eth_weth(*facts)


@pytest.mark.parametrize("amount", [1, 5 * 10**17, 10**18, 10**37 + 13])
def test_direct_deposit_builds_wrap_with_exact_amount(amount: int) -> None:
    """WETH9 Deposit mints a WETH economic leg without a Transfer log.

    :param amount: Positive exact raw wei value.
    """
    result = classify(case(amount=amount))
    assert isinstance(result, Wrap)
    assert result.amount_out.asset_id == ETH
    assert result.amount_in.asset_id == WETH
    assert result.amount_out.quantity == Decimal(f"{amount}e-18")
    assert result.amount_in.quantity == Decimal(f"{amount}e-18")
    assert result.metadata.tx_hash == TX_HASH
    assert result.metadata.timestamp == TIMESTAMP
    assert not hasattr(result, "cost_basis")
    assert not hasattr(result, "realized_pnl")
    assert not hasattr(result, "proceeds")


@pytest.mark.parametrize("amount", [1, 5 * 10**17, 10**18, 10**37 + 13])
def test_direct_withdraw_builds_unwrap_with_exact_amount(amount: int) -> None:
    """WETH9 Withdrawal restores native ETH and preserves 1:1 quantity.

    :param amount: Positive exact raw wad value.
    """
    result = classify(case("unwrap", amount=amount))
    assert isinstance(result, Unwrap)
    assert result.amount_out.asset_id == WETH
    assert result.amount_in.asset_id == ETH
    assert result.amount_out.quantity == Decimal(f"{amount}e-18")
    assert result.amount_in.quantity == Decimal(f"{amount}e-18")


def test_payable_fallback_deposit_is_supported() -> None:
    """Plain ETH sent to WETH9 invokes its payable fallback deposit."""
    assert isinstance(classify(case(fallback=True)), Wrap)


def test_raw_wallet_delta_only_contains_eth_not_synthetic_weth() -> None:
    """Synthetic WETH comes from receipt log, never duplicated by E5.2."""
    group, delta, tx, receipt = case()
    assert delta.changes[0].asset_id == ETH
    assert delta.changes[0].raw_units == -(5 * 10**17)
    assert len(delta.changes) == 1
    result = classify_eth_weth(group, delta, tx, receipt)
    assert isinstance(result, Wrap)
    assert result.amount_in.asset_id == WETH


def test_gas_is_separate_from_wrap_and_unwrap_legs() -> None:
    """Gas receipt metadata never modifies the amount converted."""
    facts = case()
    group, delta, tx, receipt = facts
    inflated = replace(receipt, gas_used=2_000_000, effective_gas_price_wei=42_000_000_000)
    assert classify((group, delta, tx, inflated)) == classify(facts)
    assert isinstance(classify(case("unwrap")), Unwrap)


def test_decimal_exactness_with_low_ambient_precision() -> None:
    """Conversion from raw units must not round through Decimal context."""
    with localcontext() as context:
        context.prec = 7
        result = classify(case(amount=10**38 + 1))
    assert isinstance(result, Wrap)
    assert result.amount_out.quantity == Decimal(f"{10**38 + 1}e-18")


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
@pytest.mark.parametrize("success", [False, None])
def test_execution_failure_or_unknown_remains_unknown(direction: str, success: bool | None) -> None:
    """Failed or unconfirmed execution never creates a conversion.

    :param direction: Desired conversion direction.
    :param success: Receipt execution state.
    """
    result = classify(case(direction, success=success))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "weth_execution_not_confirmed"


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_missing_log_does_not_invent_token_leg(direction: str) -> None:
    """No WETH9 event means no proven WETH conversion.

    :param direction: Desired conversion direction.
    """
    result = classify(case(direction, logs=()))
    assert isinstance(result, UnknownEvent)
    assert (
        result.reason
        == f"weth_{'deposit' if direction == 'wrap' else 'withdraw'}_evidence_mismatch"
    )


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_duplicate_logs_are_not_interpreted_as_one_conversion(direction: str) -> None:
    """Multiple matching WETH logs cannot be silently collapsed.

    :param direction: Desired conversion direction.
    """
    record = log(direction)
    assert isinstance(classify(case(direction, logs=(record, record))), UnknownEvent)


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_wrong_event_beneficiary_is_unknown(direction: str) -> None:
    """WETH9 event indexed owner must equal analyzed wallet.

    :param direction: Desired conversion direction.
    """
    assert isinstance(
        classify(case(direction, logs=(log(direction, beneficiary=THIRD_PARTY),))), UnknownEvent
    )


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_wrong_log_emitter_is_unknown(direction: str) -> None:
    """Identical event signatures from another contract are not WETH9 facts.

    :param direction: Desired conversion direction.
    """
    emitted = replace(log(direction), address=THIRD_PARTY)
    assert isinstance(classify(case(direction, logs=(emitted,))), UnknownEvent)


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_wrong_log_event_topic_is_unknown(direction: str) -> None:
    """A log for the opposite operation does not establish direction.

    :param direction: Desired conversion direction.
    """
    emitted = replace(
        log(direction),
        topics=(WITHDRAW_TOPIC if direction == "wrap" else DEPOSIT_TOPIC, log(direction).topics[1]),
    )
    assert isinstance(classify(case(direction, logs=(emitted,))), UnknownEvent)


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
@pytest.mark.parametrize(
    "topics", [(), (DEPOSIT_TOPIC,), (DEPOSIT_TOPIC, b"short"), (DEPOSIT_TOPIC, b"bad" * 11)]
)
def test_malformed_topics_cannot_be_accepted(direction: str, topics: tuple[bytes, ...]) -> None:
    """Indexed ABI topics must match exact WETH9 event data.

    :param direction: Desired conversion direction.
    :param topics: Deliberately malformed topic array.
    """
    emitted = replace(log(direction), topics=topics)
    assert isinstance(classify(case(direction, logs=(emitted,))), UnknownEvent)


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
@pytest.mark.parametrize("data", [b"", b"\x00", b"\x00" * 33, b"\x00" * 32])
def test_malformed_or_zero_wad_is_unknown(direction: str, data: bytes) -> None:
    """A missing, malformed or zero wad cannot produce a conversion.

    :param direction: Desired conversion direction.
    :param data: Untrusted ABI log bytes.
    """
    emitted = replace(log(direction), data=data)
    assert isinstance(classify(case(direction, logs=(emitted,))), UnknownEvent)


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_log_value_mismatch_is_unknown(direction: str) -> None:
    """Calldata, ETH movement and log must agree on exact quantity.

    :param direction: Desired conversion direction.
    """
    assert isinstance(classify(case(direction, logs=(log(direction, amount=7),))), UnknownEvent)


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_extra_wallet_movement_is_unknown(direction: str) -> None:
    """Multi-asset/split transactions are outside direct WETH9 v1 scope.

    :param direction: Desired conversion direction.
    """
    extra = movement(
        ident="other",
        asset=OTHER,
        amount=10**18,
        sender=THIRD_PARTY,
        receiver=WALLET,
        direction=direction,
    )
    assert isinstance(
        classify(case(direction, movements=(movement(direction=direction), extra))), UnknownEvent
    )


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_other_address_movement_is_unknown(direction: str) -> None:
    """An unrelated ETH movement in the same transaction is not conversion.

    :param direction: Desired conversion direction.
    """
    assert isinstance(
        classify(
            case(
                direction,
                movements=(movement(direction=direction, sender=THIRD_PARTY, receiver=WALLET),),
            )
        ),
        UnknownEvent,
    )


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_wrong_native_movement_category_is_unknown(direction: str) -> None:
    """Wrap requires external ETH, unwrap requires WETH9 internal ETH.

    :param direction: Desired conversion direction.
    """
    other = MovementCategory.INTERNAL if direction == "wrap" else MovementCategory.EXTERNAL
    altered = replace(movement(direction=direction), category=other)
    assert isinstance(classify(case(direction, movements=(altered,))), UnknownEvent)


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_movement_amount_mismatch_is_unknown(direction: str) -> None:
    """Native movement and event amount must match exactly.

    :param direction: Desired conversion direction.
    """
    assert isinstance(
        classify(case(direction, movements=(movement(direction=direction, amount=17),))),
        UnknownEvent,
    )


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_weth_mint_burn_transfer_fabrication_is_rejected(direction: str) -> None:
    """Do not double-count an invented ERC-20 Transfer movement.

    :param direction: Conversion direction.
    """
    fabricated = movement(
        ident="fabricated-weth",
        direction=direction,
        asset=WETH,
        sender=THIRD_PARTY if direction == "wrap" else WALLET,
        receiver=WALLET if direction == "wrap" else THIRD_PARTY,
    )
    assert isinstance(
        classify(case(direction, movements=(movement(direction=direction), fabricated))),
        UnknownEvent,
    )


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_bad_wallet_delta_detected_as_inconsistent(direction: str) -> None:
    """Caller may not pass delta computed for other movements.

    :param direction: Desired conversion direction.
    """
    group, delta, tx, receipt = case(direction)
    wrong = replace(delta, changes=())
    with pytest.raises(EthereumNormalizationError, match="wallet deltas disagree"):
        classify((group, wrong, tx, receipt))


def test_case_insensitive_rpc_addresses_do_not_affect_weth_identity() -> None:
    """RPC address casing does not change chain-and-contract identity."""
    group, delta, tx, receipt = case()
    assert isinstance(
        classify(
            (group, delta, replace(tx, to_address=WETH_ADDR.upper().replace("0X", "0x")), receipt)
        ),
        Wrap,
    )


def test_wrong_contract_is_not_classified() -> None:
    """Other wrapped tokens or protocols are outside this direct WETH9 classifier."""
    group, delta, tx, receipt = case()
    assert classify((group, delta, replace(tx, to_address=THIRD_PARTY), receipt)) is None


def test_third_party_sender_is_not_classified() -> None:
    """A non-wallet caller is not direct self-wrap."""
    group, delta, tx, receipt = case()
    assert classify((group, delta, replace(tx, from_address=THIRD_PARTY), receipt)) is None


@pytest.mark.parametrize("calldata", [b"\x12\x34\x56\x78", b"\x01", DEPOSIT + b"\x00"])
def test_other_weth_contract_methods_are_not_swaps(calldata: bytes) -> None:
    """Calls other than canonical wrap selectors are unrelated operations.

    :param calldata: Alternate EVM calldata.
    """
    group, delta, tx, receipt = case()
    assert classify((group, delta, replace(tx, input_data=calldata), receipt)) is None


@pytest.mark.parametrize(
    "payload", [WITHDRAW, WITHDRAW + bytes(31), WITHDRAW + bytes(33), WITHDRAW + bytes(32)]
)
def test_invalid_withdraw_calldata_returns_unknown(payload: bytes) -> None:
    """withdraw(uint256) takes exactly one 32-byte positive argument.

    :param payload: Malformed or zero-valued contract calldata.
    """
    group, delta, tx, receipt = case("unwrap")
    assert isinstance(
        classify((group, delta, replace(tx, input_data=payload), receipt)), UnknownEvent
    )


def test_withdraw_cannot_carry_native_eth_value() -> None:
    """An ordinary withdraw() never pays ETH into WETH9."""
    group, delta, tx, receipt = case("unwrap")
    assert isinstance(classify((group, delta, replace(tx, value_wei=1), receipt)), UnknownEvent)


def test_deposit_cannot_have_mismatched_eth_value() -> None:
    """A direct deposit() value must exactly match the emitted wad."""
    group, delta, tx, receipt = case()
    assert isinstance(classify((group, delta, replace(tx, value_wei=1), receipt)), UnknownEvent)


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_conflicting_rpc_hash_is_rejected(direction: str) -> None:
    """Never classify receipt facts from a different transaction.

    :param direction: Conversion direction.
    """
    group, delta, tx, receipt = case(direction)
    with pytest.raises(EthereumNormalizationError, match="identity"):
        classify((group, delta, tx, replace(receipt, tx_hash="0x" + "00" * 32)))


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_block_mismatch_rejected(direction: str) -> None:
    """Movement and receipt blocks must agree before interpretation.

    :param direction: Conversion direction.
    """
    group, delta, tx, receipt = case(direction)
    with pytest.raises(EthereumNormalizationError, match="block"):
        classify((group, delta, replace(tx, block_number=5_001), receipt))


@pytest.mark.parametrize("position,value", [(0, None), (1, None), (2, None), (3, None)])
def test_bad_top_level_types(position: int, value: object) -> None:
    """Top-level classifier arguments must be explicit domain/RPC types.

    :param position: Input being replaced.
    :param value: Invalid replacement.
    """
    args: list[object] = list(case())
    args[position] = value
    with pytest.raises(TypeError):
        classify_eth_weth(*args)  # type: ignore[arg-type]


def test_unknown_event_preserves_signed_native_delta() -> None:
    """Unclassified direct WETH9 operation retains exact known native delta."""
    result = classify(case(logs=()))
    assert isinstance(result, UnknownEvent)
    assert len(result.movements) == 1
    assert result.movements[0].asset_id == ETH
    assert result.movements[0].quantity == Decimal("-0.5")


def test_unknown_with_extra_token_does_not_guess_its_decimals() -> None:
    """Unknown other token stays in raw delta, never guessed as 18 decimals."""
    extra = movement(
        asset=OTHER, ident="extra", direction="wrap", sender=THIRD_PARTY, receiver=WALLET
    )
    group, delta, tx, receipt = case(movements=(movement(), extra))
    result = classify((group, delta, tx, receipt))
    assert isinstance(result, UnknownEvent)
    assert len(result.movements) == 1
    assert result.movements[0].asset_id == ETH
    assert any(change.asset_id == OTHER for change in delta.changes)


def test_weth9_fallback_does_not_turn_into_transfer_out_when_log_missing() -> None:
    """Missing WETH9 event must not make payable fallback a zero-proceeds transfer."""
    from portfolio_analyzer.ethereum.transfer_out import classify_transfer_out

    group, delta, tx, receipt = case(fallback=True, logs=())
    assert isinstance(classify_eth_weth(group, delta, tx, receipt), UnknownEvent)
    assert classify_transfer_out(group, delta, tx, receipt, token_metadata={}) is None


def test_weth9_fallback_does_not_turn_into_transfer_out_when_log_present() -> None:
    """Direct native outgoing transfer classifier must not consume a wrapped ETH leg."""
    from portfolio_analyzer.ethereum.transfer_out import classify_transfer_out

    group, delta, tx, receipt = case(fallback=True)
    assert isinstance(classify_eth_weth(group, delta, tx, receipt), Wrap)
    assert classify_transfer_out(group, delta, tx, receipt, token_metadata={}) is None


def test_simple_swap_classifier_does_not_claim_direct_weth_wrap() -> None:
    """Canonical WETH9 deposit belongs to Wrap even though it is ETH outgoing."""
    from portfolio_analyzer.ethereum.swap import classify_swap_v1

    group, delta, tx, receipt = case()
    assert classify_swap_v1(group, delta, tx, receipt, token_metadata={}) is None
    assert isinstance(classify_eth_weth(group, delta, tx, receipt), Wrap)


def test_transfer_in_classifier_does_not_claim_direct_weth_withdraw() -> None:
    """WETH9 internal ETH repayment must not be assigned unknown acquisition basis."""
    from portfolio_analyzer.ethereum.transfer_in import classify_transfer_in

    group, delta, tx, receipt = case("unwrap")
    assert classify_transfer_in(group, delta, tx, receipt, token_metadata={}) is None
    assert isinstance(classify_eth_weth(group, delta, tx, receipt), Unwrap)


def test_event_with_unrelated_extra_receipt_log_is_unknown() -> None:
    """One conversion event plus another contract event is no longer simple v1."""
    other = RpcLog(address=THIRD_PARTY, topics=(), data=b"")
    assert isinstance(classify(case(logs=(log(), other))), UnknownEvent)


def test_canonical_weth_identity_is_shared_with_other_classifiers() -> None:
    """WETH9 identity comes from one Ethereum registry, never ticker lookup."""
    from portfolio_analyzer.ethereum.known_contracts import (
        MAINNET_WETH9,
        MAINNET_WETH9_METADATA,
    )

    assert MAINNET_WETH9 == WETH
    assert MAINNET_WETH9_METADATA.decimals == 18
    assert MAINNET_WETH9_METADATA.symbol == "WETH"


def test_weth9_withdraw_that_has_multiple_internal_eth_movements_is_unknown() -> None:
    """Cannot collapse multiple observed ETH payouts into one simple unwrap."""
    tx_movements = (
        movement(direction="unwrap", amount=5 * 10**17, ident="first"),
        movement(direction="unwrap", amount=1, ident="second"),
    )
    assert isinstance(classify(case("unwrap", movements=tx_movements)), UnknownEvent)


def test_malformed_rpc_log_object_is_not_accepted() -> None:
    """Do not trust a receipt log if a fake provider violates the model type."""
    from typing import cast

    invalid_log = cast(RpcLog, object())
    assert isinstance(classify(case(logs=(invalid_log,))), UnknownEvent)


def test_token_only_movement_cannot_substitute_native_eth_leg() -> None:
    """Even a known WETH movement does not prove ETH entered/leaved WETH9."""
    fabricated = movement(
        direction="wrap",
        asset=WETH,
        sender=THIRD_PARTY,
        receiver=WALLET,
    )
    assert isinstance(classify(case(movements=(fabricated,))), UnknownEvent)

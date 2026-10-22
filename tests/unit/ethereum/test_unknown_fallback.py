"""End-to-end deterministic classification with UNKNOWN as a first-class outcome."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.domain.events import (
    Swap,
    TransferIn,
    TransferOut,
    UnknownEvent,
    Unwrap,
    Wrap,
)
from portfolio_analyzer.ethereum.approve import NoAccountingAction
from portfolio_analyzer.ethereum.classifier import (
    classify_transaction,
    classify_transaction_without_movements,
)
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
    group_movements_by_transaction,
)
from portfolio_analyzer.ethereum.rpc_models import RpcBlock, RpcLog, RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.wallet_deltas import (
    RawAssetDelta,
    WalletAssetDelta,
    calculate_wallet_asset_delta,
)

WALLET = "0x" + "aa" * 20
PEER = "0x" + "bb" * 20
ROUTER = "0x" + "cc" * 20
POOL = "0x" + "dd" * 20
WETH_ADDRESS = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
ETH = AssetId(1, None)
LINK = AssetId(1, "0x" + "11" * 20)
USDC = AssetId(1, "0x" + "22" * 20)
LDO = AssetId(1, "0x" + "33" * 20)
BONUS = AssetId(1, "0x" + "44" * 20)
WETH = AssetId(1, WETH_ADDRESS)
TX_HASH = "0x" + "fe" * 32
AT = datetime(2026, 10, 21, 12, 30, tzinfo=UTC)
METADATA = {
    LINK: AssetMetadata("LINK", "Chainlink", 18),
    USDC: AssetMetadata("USDC", "USD Coin", 6),
    LDO: AssetMetadata("LDO", "Lido", 18),
    BONUS: AssetMetadata("BONUS", "Bonus", 18),
    WETH: AssetMetadata("WETH", "Wrapped Ether", 18),
}
DEPOSIT_TOPIC = bytes.fromhex("e1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c")
WITHDRAW_TOPIC = bytes.fromhex("7fcf532c15f0a6db0bd6d0e038bea71d30d808c7d98cb3bf7268a95bf5081b65")
APPROVAL_TOPIC = bytes.fromhex("8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925")


def movement(
    ident: str,
    asset: AssetId,
    raw_amount: int,
    sender: str = WALLET,
    recipient: str = ROUTER,
    *,
    category: MovementCategory | None = None,
) -> AssetMovement:
    """Build one valid provider-neutral movement.

    :param ident: Unique transfer identity.
    :param asset: Native or token asset.
    :param raw_amount: Unsigned amount in smallest units.
    :param sender: Ethereum transfer origin.
    :param recipient: Ethereum transfer destination.
    :param category: Optional native external/internal override.
    :return: A validated AssetMovement.
    """
    return AssetMovement(
        transfer_id=ident,
        tx_hash=TX_HASH,
        block_number=17_000,
        timestamp=AT,
        from_address=sender,
        to_address=recipient,
        asset_id=asset,
        raw_amount=raw_amount,
        category=category
        or (MovementCategory.EXTERNAL if asset.is_native else MovementCategory.ERC20),
    )


def setup(
    *items: AssetMovement,
    sender: str = WALLET,
    to: str | None = ROUTER,
    calldata: bytes = bytes.fromhex("12345678") + bytes(96),
    value: int = 0,
    success: bool | None = True,
    logs: tuple[RpcLog, ...] = (),
) -> tuple[TransactionMovementGroup, WalletAssetDelta, RpcTransaction, RpcReceipt]:
    """Construct consistent provider-neutral facts for one mined transaction.

    :param items: Observed movements, at least one.
    :param sender: RPC transaction sender.
    :param to: RPC transaction recipient.
    :param calldata: Raw calldata.
    :param value: Native wei excluding gas.
    :param success: Ethereum receipt execution status.
    :param logs: Ethereum receipt logs.
    :return: One transaction group, wallet delta, transaction and receipt.
    """
    group = group_movements_by_transaction(items)[0]
    delta = calculate_wallet_asset_delta(group, WALLET)
    transaction = RpcTransaction(TX_HASH, sender, to, 17_000, value, calldata)
    receipt = RpcReceipt(TX_HASH, 17_000, success, 21_000, 10**9, logs)
    return group, delta, transaction, receipt


def classify(
    facts: tuple[TransactionMovementGroup, WalletAssetDelta, RpcTransaction, RpcReceipt],
    *,
    metadata: dict[AssetId, AssetMetadata] | None = None,
) -> Swap | TransferIn | TransferOut | Wrap | Unwrap | UnknownEvent:
    """Classify matching evidence through the public dispatcher.

    :param facts: Valid transaction and movement facts.
    :param metadata: Optional trusted metadata override.
    :return: Deterministic economic outcome.
    """
    return classify_transaction(*facts, token_metadata=METADATA if metadata is None else metadata)


def direct_transfer_call(recipient: str, amount: int) -> bytes:
    """Encode a standard ERC-20 transfer(address,uint256) call.

    :param recipient: Destination address.
    :param amount: Exact raw token amount.
    :return: ABI calldata.
    """
    return (
        bytes.fromhex("a9059cbb")
        + bytes(12)
        + bytes.fromhex(recipient[2:])
        + amount.to_bytes(32, "big")
    )


@pytest.mark.parametrize(
    ("asset", "raw_amount"),
    [(USDC, 10**8), (ETH, 5 * 10**17), (LDO, 40 * 10**18)],
)
def test_simple_swaps_keep_single_swap_event(asset: AssetId, raw_amount: int) -> None:
    """USDC/ETH/LDO outbound swaps to LINK share one economic event type."""
    facts = setup(
        movement("out", asset, raw_amount),
        movement("in", LINK, 50 * 10**18, sender=POOL, recipient=WALLET),
        value=raw_amount if asset.is_native else 0,
    )
    event = classify(facts)
    assert isinstance(event, Swap)
    assert event.amount_out.asset_id == asset
    assert event.amount_in.asset_id == LINK
    assert event.amount_in.quantity == Decimal("50")
    assert not hasattr(event, "proceeds")
    assert not hasattr(event, "gas")


def test_direct_incoming_eth_transfer() -> None:
    """Inbound native ETH remains a TransferIn, not UNKNOWN or a buy."""
    amount = 10**18
    facts = setup(
        movement("in", ETH, amount, sender=PEER, recipient=WALLET),
        sender=PEER,
        to=WALLET,
        calldata=b"",
        value=amount,
    )
    event = classify(facts)
    assert isinstance(event, TransferIn)
    assert event.amount.quantity == Decimal("1")
    assert not hasattr(event, "cost_basis")


def test_direct_incoming_erc20_transfer() -> None:
    """Inbound directly transferred ERC-20 obtains a TransferIn with no basis."""
    raw = 1_500_000
    facts = setup(
        movement("in", USDC, raw, sender=PEER, recipient=WALLET),
        sender=PEER,
        to=USDC.contract_address,
        calldata=direct_transfer_call(WALLET, raw),
    )
    event = classify(facts)
    assert isinstance(event, TransferIn)
    assert event.amount.quantity == Decimal("1.500000")


def test_direct_outgoing_eth_transfer_without_gas_in_amount() -> None:
    """Only ETH value is disposed, independent of paid transaction gas."""
    amount = 5 * 10**17
    facts = setup(movement("out", ETH, amount, recipient=PEER), to=PEER, calldata=b"", value=amount)
    event = classify(facts)
    assert isinstance(event, TransferOut)
    assert event.amount.quantity == Decimal("0.5")
    assert not hasattr(event, "proceeds")


def test_direct_outgoing_erc20_transfer() -> None:
    """Direct token transfer remains a non-sale disposal."""
    raw = 30 * 10**18
    facts = setup(
        movement("out", LINK, raw, recipient=PEER),
        to=LINK.contract_address,
        calldata=direct_transfer_call(PEER, raw),
    )
    event = classify(facts)
    assert isinstance(event, TransferOut)
    assert event.amount.quantity == Decimal("30")


@pytest.mark.parametrize("direction", ["wrap", "unwrap"])
def test_weth_wrap_unwrap_has_priority_over_generic_trade(direction: str) -> None:
    """WETH9 direct conversions never become a generic swap or transfer."""
    wei = 5 * 10**17
    is_wrap = direction == "wrap"
    facts = setup(
        movement(
            "native",
            ETH,
            wei,
            WALLET if is_wrap else WETH_ADDRESS,
            WETH_ADDRESS if is_wrap else WALLET,
            category=MovementCategory.EXTERNAL if is_wrap else MovementCategory.INTERNAL,
        ),
        to=WETH_ADDRESS,
        calldata=bytes.fromhex("d0e30db0")
        if is_wrap
        else bytes.fromhex("2e1a7d4d") + wei.to_bytes(32, "big"),
        value=wei if is_wrap else 0,
        logs=(
            RpcLog(
                address=WETH_ADDRESS,
                topics=(
                    (DEPOSIT_TOPIC if is_wrap else WITHDRAW_TOPIC),
                    bytes(12) + bytes.fromhex(WALLET[2:]),
                ),
                data=wei.to_bytes(32, "big"),
            ),
        ),
    )
    event = classify(facts)
    assert isinstance(event, Wrap if is_wrap else Unwrap)
    assert event.amount_out.quantity == Decimal("0.5")
    assert event.amount_in.quantity == Decimal("0.5")


def test_unsupported_weth_keeps_specialist_unknown_reason() -> None:
    """Failed WETH9 evidence must not fall back to generic transfer/swap."""
    wei = 5 * 10**17
    facts = setup(
        movement("native", ETH, wei, recipient=WETH_ADDRESS),
        to=WETH_ADDRESS,
        calldata=b"",
        value=wei,
    )
    event = classify(facts)
    assert isinstance(event, UnknownEvent)
    assert event.reason == "weth_deposit_evidence_mismatch"
    assert event.movements[0].quantity == Decimal("-0.5")


@pytest.mark.parametrize(
    ("assets", "reason"),
    [
        ([(USDC, -100_000_000), (LINK, 50 * 10**18), (BONUS, 10**18)], "multi_asset_transaction"),
        ([(USDC, -100_000_000), (LDO, -(10**18)), (LINK, 50 * 10**18)], "multi_asset_transaction"),
        (
            [(USDC, -100_000_000), (LINK, 50 * 10**18), (BONUS, -(10**18))],
            "multi_asset_transaction",
        ),
    ],
)
def test_multi_asset_returns_diagnostic_unknown(
    assets: list[tuple[AssetId, int]], reason: str
) -> None:
    """Multi-asset transaction is UNKNOWN even with a superficially plausible trade."""
    facts = setup(
        *(
            movement(
                f"part{i}",
                asset,
                abs(raw),
                sender=WALLET if raw < 0 else POOL,
                recipient=ROUTER if raw < 0 else WALLET,
            )
            for i, (asset, raw) in enumerate(assets)
        )
    )
    event = classify(facts)
    assert isinstance(event, UnknownEvent)
    assert event.reason == reason
    assert len(event.movements) == len(assets)
    assert event.metadata.tx_hash == TX_HASH
    assert not hasattr(event, "proceeds")
    assert not hasattr(event, "cost_basis")


def test_single_ambiguous_contract_payout_becomes_unknown() -> None:
    """An internal ETH payout is not automatically a transfer or trade."""
    facts = setup(
        movement(
            "internal",
            ETH,
            10**18,
            sender=ROUTER,
            recipient=WALLET,
            category=MovementCategory.INTERNAL,
        ),
        sender=PEER,
    )
    event = classify(facts)
    assert isinstance(event, UnknownEvent)
    assert event.reason == "unsupported_transaction"
    assert event.movements[0].quantity == Decimal("1")


def test_unknown_self_transfer_with_zero_net_changes() -> None:
    """Self transfer contributes no acquisition or disposal."""
    facts = setup(
        movement("self", LINK, 10**18, sender=WALLET, recipient=WALLET),
        to=LINK.contract_address,
        calldata=direct_transfer_call(WALLET, 10**18),
    )
    event = classify(facts)
    assert isinstance(event, UnknownEvent)
    assert event.reason == "no_wallet_asset_change"
    assert event.movements == ()


def test_missing_token_decimals_is_unknown_not_zero_balance() -> None:
    """Unknown decimals preserve raw delta separately without fake amounts."""
    facts = setup(
        movement("token", USDC, 1_500_000, sender=PEER, recipient=WALLET),
        sender=PEER,
        to=USDC.contract_address,
        calldata=direct_transfer_call(WALLET, 1_500_000),
    )
    event = classify(facts, metadata={})
    assert isinstance(event, UnknownEvent)
    assert event.reason == "missing_asset_metadata"
    assert event.movements == ()
    assert facts[1].changes[0].raw_units == 1_500_000


def test_mixed_missing_metadata_preserves_known_signed_eth() -> None:
    """UNKNOWN holds only trustworthy quantities, never invented decimals."""
    facts = setup(
        movement("eth", ETH, 5 * 10**17),
        movement("usdc", USDC, 1_000_000, sender=POOL, recipient=WALLET),
        value=5 * 10**17,
    )
    event = classify(facts, metadata={})
    assert isinstance(event, UnknownEvent)
    assert len(event.movements) == 1
    assert event.movements[0].asset_id == ETH
    assert event.movements[0].quantity == Decimal("-0.5")


def test_nonmatching_signs_return_specialist_unknown() -> None:
    """Both outgoing legs must not be forced into an asset exchange."""
    facts = setup(movement("out1", USDC, 1_000_000), movement("out2", LINK, 10**18))
    event = classify(facts)
    assert isinstance(event, UnknownEvent)
    assert event.reason == "not_one_in_one_out"


def test_multiple_movements_same_asset_rejected_as_simple_swap() -> None:
    """Split routed movements cannot be inferred to be a one-to-one swap."""
    facts = setup(
        movement("out1", USDC, 1_000_000),
        movement("out2", USDC, 2_000_000),
        movement("in", LINK, 2 * 10**18, sender=POOL, recipient=WALLET),
    )
    event = classify(facts)
    assert isinstance(event, UnknownEvent)
    assert event.reason == "multiple_wallet_movements"


def test_unconfirmed_execution_yields_unknown_not_trade() -> None:
    """Absent legacy success status cannot be assumed successful execution."""
    facts = setup(
        movement("out", USDC, 10**6),
        movement("in", LINK, 10**18, sender=POOL, recipient=WALLET),
        success=None,
    )
    event = classify(facts)
    assert isinstance(event, UnknownEvent)
    assert event.reason == "execution_status_unconfirmed"


def test_reverted_receipt_with_mined_transfers_is_error_not_unknown() -> None:
    """Contradictory execution data is not an unsupported financial operation."""
    facts = setup(movement("in", ETH, 10**18, sender=PEER, recipient=WALLET), success=False)
    with pytest.raises(EthereumNormalizationError, match="reverted"):
        classify(facts)


def test_reverted_zero_amount_transfer_is_still_inconsistent() -> None:
    """Even a zero-value ERC-20 event cannot survive a reverted EVM execution."""
    facts = setup(movement("zero", LINK, 0, sender=WALLET, recipient=PEER), success=False)
    with pytest.raises(EthereumNormalizationError, match="reverted"):
        classify(facts)


def test_rpc_hash_mismatch_is_error() -> None:
    """Conflicting transaction hashes must never be hidden as UNKNOWN."""
    group, delta, transaction, receipt = setup(movement("out", LINK, 10**18))
    bad = replace(transaction, tx_hash="0x" + "01" * 32)
    with pytest.raises(EthereumNormalizationError):
        classify((group, delta, bad, receipt))


def test_rpc_block_mismatch_is_error() -> None:
    """Contradictory block number must never be turned into UNKNOWN."""
    group, delta, transaction, receipt = setup(movement("out", LINK, 10**18))
    with pytest.raises(EthereumNormalizationError):
        classify((group, delta, transaction, replace(receipt, block_number=19)))


def test_stale_wallet_delta_is_error() -> None:
    """A stale derived delta is a data integrity failure, not uncertainty."""
    group, delta, transaction, receipt = setup(movement("out", LINK, 10**18))
    stale = replace(delta, changes=(RawAssetDelta(LINK, -(10**18 + 1)),))
    with pytest.raises(EthereumNormalizationError, match="wallet deltas"):
        classify((group, stale, transaction, receipt))


def test_wrong_wallet_delta_transaction_is_error() -> None:
    """Movement attribution must agree on transaction identity."""
    group, delta, transaction, receipt = setup(movement("out", LINK, 10**18))
    wrong = replace(delta, transaction=replace(delta.transaction, block_number=123))
    with pytest.raises(EthereumNormalizationError):
        classify((group, wrong, transaction, receipt))


def test_malformed_metadata_is_error_not_unknown() -> None:
    """A broken metadata model is not equivalent to legitimately missing decimals."""
    facts = setup(
        movement("in", LINK, 10**18, sender=PEER, recipient=WALLET),
        sender=PEER,
        to=LINK.contract_address,
        calldata=direct_transfer_call(WALLET, 10**18),
    )
    with pytest.raises(EthereumNormalizationError, match="metadata"):
        classify_transaction(*facts, token_metadata={LINK: "not metadata"})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "broken",
    [None, 1, object(), []],
)
def test_wrong_metadata_container_fails_fast(broken: object) -> None:
    """Malformed caller API inputs should not turn into an UNKNOWN event."""
    facts = setup(movement("out", USDC, 1000000))
    with pytest.raises(TypeError, match="token_metadata"):
        classify_transaction(*facts, token_metadata=broken)  # type: ignore[arg-type]


def test_provider_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected classifier exceptions propagate rather than falsifying UNKNOWN."""
    facts = setup(
        movement(
            "in", ETH, 10**18, sender=ROUTER, recipient=WALLET, category=MovementCategory.INTERNAL
        )
    )

    def broken(*args: object, **kwargs: object) -> None:
        """Simulate an underlying failed data dependency."""
        raise RuntimeError("provider failed")

    monkeypatch.setattr("portfolio_analyzer.ethereum.classifier.classify_eth_weth", broken)
    with pytest.raises(RuntimeError, match="provider failed"):
        classify(facts)


def test_no_movement_transaction_approve_becomes_no_accounting_action() -> None:
    """Independently discovered approval does not create a buy or sell."""
    raw = 1_234
    calldata = (
        bytes.fromhex("095ea7b3") + bytes(12) + bytes.fromhex(PEER[2:]) + raw.to_bytes(32, "big")
    )
    receipt = RpcReceipt(
        TX_HASH,
        17_000,
        True,
        55_000,
        10**9,
        (
            RpcLog(
                USDC.contract_address or "",
                (
                    APPROVAL_TOPIC,
                    bytes(12) + bytes.fromhex(WALLET[2:]),
                    bytes(12) + bytes.fromhex(PEER[2:]),
                ),
                raw.to_bytes(32, "big"),
            ),
        ),
    )
    transaction = RpcTransaction(TX_HASH, WALLET, USDC.contract_address, 17_000, 0, calldata)
    result = classify_transaction_without_movements(
        transaction, receipt, RpcBlock(17_000, AT), wallet_address=WALLET, movements_complete=True
    )
    assert isinstance(result, NoAccountingAction)
    assert result.allowance_raw == raw
    assert not hasattr(result, "amount_in")
    assert not hasattr(result, "cost_basis")


def test_no_movement_unrecognized_contract_call_is_unknown() -> None:
    """Movement-free contract call is UNKNOWN rather than silently skipped."""
    transaction = RpcTransaction(TX_HASH, WALLET, ROUTER, 17_000, 0, b"\x01\x02\x03\x04")
    receipt = RpcReceipt(TX_HASH, 17_000, True, 1234, 10**9, ())
    event = classify_transaction_without_movements(
        transaction, receipt, RpcBlock(17_000, AT), wallet_address=WALLET, movements_complete=True
    )
    assert isinstance(event, UnknownEvent)
    assert event.reason == "unsupported_no_transfer_action"
    assert event.movements == ()
    assert event.metadata.timestamp == AT


@pytest.mark.parametrize("verified", [False, None, 1, "true"])
def test_absent_complete_history_evidence_must_not_be_called_no_accounting(
    verified: object,
) -> None:
    """Missing pages of history must not masquerade as no-accounting."""
    transaction = RpcTransaction(TX_HASH, WALLET, ROUTER, 17_000, 0, b"\x01\x02\x03\x04")
    receipt = RpcReceipt(TX_HASH, 17_000, True, 1234, 10**9, ())
    with pytest.raises((TypeError, EthereumNormalizationError)):
        classify_transaction_without_movements(
            transaction,
            receipt,
            RpcBlock(17_000, AT),
            wallet_address=WALLET,
            movements_complete=verified,  # type: ignore[arg-type]
        )


def test_no_movement_rpc_provenance_mismatch_is_error() -> None:
    """An invalid no-transfer receipt must not be classified as UNKNOWN."""
    transaction = RpcTransaction(TX_HASH, WALLET, ROUTER, 17_000, 0, b"\x01\x02\x03\x04")
    receipt = RpcReceipt("0x" + "ff" * 32, 17_000, True, 1234, 10**9, ())
    with pytest.raises(EthereumNormalizationError):
        classify_transaction_without_movements(
            transaction,
            receipt,
            RpcBlock(17_000, AT),
            wallet_address=WALLET,
            movements_complete=True,
        )


def test_no_movement_timestamp_timezone_is_utc() -> None:
    """UnknownEvent timestamp remains a timezone-aware UTC observation."""
    transaction = RpcTransaction(TX_HASH, WALLET, ROUTER, 17_000, 0, b"\x01\x02\x03\x04")
    receipt = RpcReceipt(TX_HASH, 17_000, True, 1234, 10**9, ())
    offset = AT.astimezone(timezone(timedelta(hours=3)))
    event = classify_transaction_without_movements(
        transaction,
        receipt,
        RpcBlock(17_000, offset),
        wallet_address=WALLET,
        movements_complete=True,
    )
    assert isinstance(event, UnknownEvent)
    assert event.metadata.timestamp == AT


@pytest.mark.parametrize("invalid_logs", [None, [], ("invalid",)])
def test_invalid_receipt_logs_not_hidden_as_unknown(invalid_logs: object) -> None:
    """Malformed provider-neutral receipt facts are data errors, not UNKNOWN."""
    group, delta, tx, receipt = setup(
        movement(
            "in", ETH, 10**18, sender=ROUTER, recipient=WALLET, category=MovementCategory.INTERNAL
        )
    )
    with pytest.raises(EthereumNormalizationError, match="receipt logs"):
        classify((group, delta, tx, replace(receipt, logs=invalid_logs)))  # type: ignore[arg-type]


@pytest.mark.parametrize("gas", [-1, True, "10"])
def test_invalid_gas_used_fails_provenance_validation(gas: object) -> None:
    """Malformed fee facts must not silently pass through classification."""
    group, delta, tx, receipt = setup(movement("out", LINK, 10**18))
    with pytest.raises(EthereumNormalizationError, match="gas_used"):
        classify((group, delta, tx, replace(receipt, gas_used=gas)))  # type: ignore[arg-type]


@pytest.mark.parametrize("price", [-1, True, "10"])
def test_invalid_gas_price_fails_provenance_validation(price: object) -> None:
    """Malformed effective gas price is an RPC fact error, not a trade."""
    group, delta, tx, receipt = setup(movement("out", LINK, 10**18))
    with pytest.raises(EthereumNormalizationError, match="gas price"):
        classify((group, delta, tx, replace(receipt, effective_gas_price_wei=price)))  # type: ignore[arg-type]


def test_unverified_two_leg_approval_call_is_unknown_not_swap() -> None:
    """A multi-asset approve-shaped call cannot be assumed to be a swap."""
    facts = setup(
        movement("out", USDC, 100 * 10**6),
        movement("in", LINK, 50 * 10**18, sender=POOL, recipient=WALLET),
        calldata=bytes.fromhex("095ea7b3") + bytes(64),
    )
    event = classify(facts)
    assert isinstance(event, UnknownEvent)
    assert event.reason == "swap_evidence_insufficient"


def test_unknown_is_repeatable_and_does_not_mutate_movements() -> None:
    """Pure classification leaves source facts intact and is deterministic."""
    facts = setup(
        movement("out", USDC, 100 * 10**6),
        movement("in", LINK, 50 * 10**18, sender=POOL, recipient=WALLET),
        movement("bonus", BONUS, 10**18, sender=POOL, recipient=WALLET),
    )
    original = facts[0].movements
    first = classify(facts)
    second = classify(facts)
    assert first == second
    assert isinstance(first, UnknownEvent)
    assert facts[0].movements == original
    assert facts[1].changes == calculate_wallet_asset_delta(facts[0], WALLET).changes


@pytest.mark.parametrize("sender,recipient", [(WALLET, PEER), (PEER, WALLET)])
def test_complete_no_movement_history_cannot_hide_native_value(sender: str, recipient: str) -> None:
    """A value transfer absent from complete indexed history is a data gap."""
    transaction = RpcTransaction(TX_HASH, sender, recipient, 17_000, 10**18, b"")
    receipt = RpcReceipt(TX_HASH, 17_000, True, 21_000, 10**9, ())
    with pytest.raises(EthereumNormalizationError, match="native value"):
        classify_transaction_without_movements(
            transaction,
            receipt,
            RpcBlock(17_000, AT),
            wallet_address=WALLET,
            movements_complete=True,
        )


def test_reverted_value_call_without_movements_is_safe_unknown() -> None:
    """A reverted value-bearing call does not transfer its native value."""
    transaction = RpcTransaction(TX_HASH, WALLET, PEER, 17_000, 10**18, b"")
    receipt = RpcReceipt(TX_HASH, 17_000, False, 21_000, 10**9, ())
    event = classify_transaction_without_movements(
        transaction,
        receipt,
        RpcBlock(17_000, AT),
        wallet_address=WALLET,
        movements_complete=True,
    )
    assert isinstance(event, UnknownEvent)
    assert event.movements == ()


def test_invalid_metadata_used_by_fallback_is_data_error() -> None:
    """Invalid decimals model in fallback must not be silently ignored."""
    facts = setup(movement("internal", LINK, 10**18, sender=ROUTER, recipient=WALLET))
    with pytest.raises(EthereumNormalizationError, match="token metadata"):
        classify_transaction(*facts, token_metadata={LINK: "bad"})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("argument", "error_fragment"),
    [
        (0, "group"),
        (1, "delta"),
        (2, "transaction"),
        (3, "receipt"),
    ],
)
def test_wrong_top_level_types_are_programming_errors(argument: int, error_fragment: str) -> None:
    """Incorrect API arguments are not silently reported as a wallet UNKNOWN."""
    facts = list(setup(movement("single", ETH, 10**18)))
    facts[argument] = None
    with pytest.raises(TypeError, match=error_fragment):
        classify_transaction(*facts, token_metadata=METADATA)  # type: ignore[arg-type]

"""Offline tests for E5.7 standalone ERC-20 approve classification."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import get_args

import pytest

from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.events import (
    EconomicEvent,
    EventMetadata,
    TransferIn,
    TransferOut,
)
from portfolio_analyzer.ethereum.approve import NoAccountingAction, classify_approve
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.ethereum.normalizer import EthereumNormalizationError
from portfolio_analyzer.ethereum.rpc_models import RpcBlock, RpcLog, RpcReceipt, RpcTransaction

WALLET = "0x" + "aa" * 20
SPENDER = "0x" + "bb" * 20
TOKEN = "0x" + "cc" * 20
OTHER = "0x" + "dd" * 20
TX_HASH = "0x" + "ef" * 32
AT = datetime(2026, 10, 21, 18, 0, tzinfo=UTC)
TOPIC = bytes.fromhex("8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925")
TRANSFER_TOPIC = bytes.fromhex("ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")


def calldata(spender: str = SPENDER, amount: int = 100) -> bytes:
    """Encode ERC-20 approve(address,uint256) without depending on an SDK.

    :param spender: Address granted allowance.
    :param amount: UInt256 raw units.
    :return: Exact 68-byte ABI payload.
    """
    return (
        bytes.fromhex("095ea7b3")
        + bytes(12)
        + bytes.fromhex(spender[2:])
        + amount.to_bytes(32, "big")
    )


def approval_log(
    *,
    emitter: str = TOKEN,
    owner: str = WALLET,
    spender: str = SPENDER,
    amount: int = 100,
) -> RpcLog:
    """Encode the indexed Approval(owner,spender,value) event.

    :param emitter: Token contract address.
    :param owner: Wallet which approved the spender.
    :param spender: Approved delegate.
    :param amount: Exact new allowance raw value.
    :return: Receipt log facts.
    """
    return RpcLog(
        emitter,
        (
            TOPIC,
            bytes(12) + bytes.fromhex(owner[2:]),
            bytes(12) + bytes.fromhex(spender[2:]),
        ),
        amount.to_bytes(32, "big"),
    )


def scenario(
    *,
    amount: int = 100,
    data: bytes | None = None,
    logs: tuple[RpcLog, ...] | None = None,
    success: bool | None = True,
    sender: str = WALLET,
    to: str | None = TOKEN,
    value: int = 0,
) -> tuple[RpcTransaction, RpcReceipt, RpcBlock]:
    """Build three matching independently obtained RPC records.

    :param amount: Positive, zero, or infinite raw allowance.
    :param data: Optional calldata override.
    :param logs: Optional receipt logs override.
    :param success: EVM execution status.
    :param sender: Tx sender.
    :param to: Tx contract recipient.
    :param value: Native ETH transferred in tx.
    :return: Transaction, receipt, and block.
    """
    return (
        RpcTransaction(
            TX_HASH, sender, to, 314, value, calldata(amount=amount) if data is None else data
        ),
        RpcReceipt(
            TX_HASH,
            314,
            success,
            46000,
            10**9,
            (approval_log(amount=amount),) if logs is None else logs,
        ),
        RpcBlock(314, AT),
    )


def classify(
    case: tuple[RpcTransaction, RpcReceipt, RpcBlock],
    *,
    wallet: str = WALLET,
    movements: tuple[AssetMovement, ...] = (),
) -> NoAccountingAction | None:
    """Call the production classifier with explicitly declared movements.

    :param case: RPC facts.
    :param wallet: Analyzed wallet.
    :param movements: Full observed movements for tx (empty for approve).
    :return: NoAccountingAction or None.
    """
    return classify_approve(*case, wallet_address=wallet, observed_movements=movements)


def transfer() -> AssetMovement:
    """Create an unrelated real asset transfer for negative tests.

    :return: Transfer history movement in the same transaction.
    """
    return AssetMovement(
        transfer_id="move-1",
        tx_hash=TX_HASH,
        block_number=314,
        timestamp=AT,
        from_address=WALLET,
        to_address=OTHER,
        asset_id=AssetId(1, TOKEN),
        raw_amount=100,
        category=MovementCategory.ERC20,
    )


@pytest.mark.parametrize("amount", [0, 1, 10**6, 2**255, 2**256 - 1])
def test_direct_approve_creates_no_accounting_marker(amount: int) -> None:
    """Zero/revoke, normal, and unlimited allowance are NOT token trades.

    :param amount: Exact raw allowance.
    """
    action = classify(scenario(amount=amount))
    assert isinstance(action, NoAccountingAction)
    assert action.metadata == EventMetadata(TX_HASH, 314, AT)
    assert action.asset_id == AssetId(1, TOKEN)
    assert action.owner == WALLET
    assert action.spender == SPENDER
    assert action.allowance_raw == amount
    for forbidden in ("amount", "amount_in", "amount_out", "proceeds", "cost_basis", "gas"):
        assert not hasattr(action, forbidden)
    assert not isinstance(action, (TransferIn, TransferOut))
    assert not isinstance(action, get_args(EconomicEvent))


def test_same_symbol_not_used_for_asset_identity() -> None:
    """A permission belongs to a contract, not to a ticker or metadata."""
    tx, receipt, block = scenario()
    action = classify(
        (
            replace(tx, to_address=OTHER),
            replace(receipt, logs=(approval_log(emitter=OTHER),)),
            block,
        )
    )
    assert isinstance(action, NoAccountingAction)
    assert action.asset_id == AssetId(1, OTHER)


def test_mixed_case_addresses_are_normalized() -> None:
    """RPC address casing must not affect identity or log matching."""
    tx, receipt, block = scenario(
        sender=WALLET.upper().replace("0X", "0x"),
        to=TOKEN.upper().replace("0X", "0x"),
    )
    log = approval_log(emitter=TOKEN.upper().replace("0X", "0x"))
    action = classify(
        (tx, replace(receipt, logs=(log,)), block),
        wallet=WALLET.upper().replace("0X", "0x"),
    )
    assert isinstance(action, NoAccountingAction)
    assert action.owner == WALLET
    assert action.asset_id == AssetId(1, TOKEN)


@pytest.mark.parametrize("success", [False, None])
def test_failed_or_unknown_receipt_is_not_successful_approval(success: bool | None) -> None:
    """No positive classification without a successful EVM execution.

    :param success: Reverted or unknown EVM status.
    """
    assert classify(scenario(success=success)) is None


@pytest.mark.parametrize(
    "data",
    [
        b"",
        bytes.fromhex("a9059cbb") + calldata()[4:],
        bytes.fromhex("23b872dd") + calldata()[4:],
        calldata()[:-1],
        calldata() + b"\0",
        bytes.fromhex("095ea7b3") + b"\x01" + bytes(11) + calldata()[16:],
        bytes.fromhex("d505accf") + calldata()[4:],
    ],
)
def test_nonstandard_or_malformed_selector_does_not_create_approval(data: bytes) -> None:
    """Only exact standard approve(address,uint256) is supported.

    :param data: Invalid or unrelated method calldata.
    """
    assert classify(scenario(data=data)) is None


@pytest.mark.parametrize("value", [1, 10**18])
def test_payable_approval_like_call_is_rejected(value: int) -> None:
    """A payable call can change native balances and is not standalone approve.

    :param value: ETH transferred with calldata.
    """
    assert classify(scenario(value=value)) is None


def test_unknown_or_external_initiator_is_ignored() -> None:
    """An approval by another wallet must not appear as ours."""
    assert classify(scenario(sender=OTHER)) is None
    assert classify(scenario(), wallet=OTHER) is None
    assert classify(scenario(to=None)) is None


def test_receipt_without_approval_event_is_not_proof() -> None:
    """Some ERC-20 variants do not emit Approval; v1 leaves them unknown."""
    assert classify(scenario(logs=())) is None


def test_spoofed_or_unexpected_logs_are_not_ignored() -> None:
    """Any extra or inconsistent log disqualifies a standalone approval."""
    tx, receipt, block = scenario()
    extra = RpcLog(TOKEN, (TRANSFER_TOPIC,), b"")
    assert classify((tx, replace(receipt, logs=(approval_log(), extra)), block)) is None
    assert classify((tx, replace(receipt, logs=(approval_log(), approval_log())), block)) is None
    assert classify((tx, replace(receipt, logs=(extra,)), block)) is None


@pytest.mark.parametrize(
    "log",
    [
        approval_log(emitter=OTHER),
        approval_log(owner=OTHER),
        approval_log(spender=OTHER),
        approval_log(amount=101),
        RpcLog(TOKEN, (TRANSFER_TOPIC, *approval_log().topics[1:]), approval_log().data),
        RpcLog(TOKEN, approval_log().topics[:2], approval_log().data),
        RpcLog(TOKEN, (*approval_log().topics, bytes(32)), approval_log().data),
        RpcLog(
            TOKEN,
            (TOPIC, bytes.fromhex(WALLET[2:]), approval_log().topics[2]),
            approval_log().data,
        ),
        RpcLog(TOKEN, approval_log().topics, b""),
        RpcLog(TOKEN, approval_log().topics, bytes(31)),
    ],
)
def test_wrong_event_proof_cannot_be_approved(log: RpcLog) -> None:
    """Log issuer, topics, padded addresses, and value must match exactly.

    :param log: Incorrect on-chain event evidence.
    """
    tx, receipt, block = scenario()
    assert classify((tx, replace(receipt, logs=(log,)), block)) is None


def test_actual_asset_transfer_disqualifies_approval_even_if_log_matches() -> None:
    """Never suppress real asset movements just because calldata says approve."""
    assert classify(scenario(), movements=(transfer(),)) is None


def test_native_eth_movement_disqualifies_approval() -> None:
    """Non-token ETH movements also mean the transaction is not a pure no-op."""
    eth = replace(
        transfer(),
        asset_id=AssetId(1, None),
        category=MovementCategory.INTERNAL,
    )
    assert classify(scenario(), movements=(eth,)) is None


def test_nonmatching_movement_provenance_is_rejected() -> None:
    """Do not hide movements attributed to a different block or hash."""
    movement = replace(transfer(), block_number=315)
    with pytest.raises(EthereumNormalizationError, match="movement"):
        classify(scenario(), movements=(movement,))
    movement = replace(transfer(), timestamp=AT + timedelta(seconds=3))
    with pytest.raises(EthereumNormalizationError, match="movement"):
        classify(scenario(), movements=(movement,))


def test_transaction_receipt_or_block_mismatch_fails_closed() -> None:
    """Untrusted provider records must refer to one and the same transaction."""
    tx, receipt, block = scenario()
    with pytest.raises(EthereumNormalizationError, match="hash"):
        classify((tx, replace(receipt, tx_hash="0x" + "12" * 32), block))
    with pytest.raises(EthereumNormalizationError, match="height"):
        classify((tx, replace(receipt, block_number=315), block))
    with pytest.raises(EthereumNormalizationError, match="height"):
        classify((tx, receipt, replace(block, number=315)))
    with pytest.raises(EthereumNormalizationError, match="height"):
        classify((replace(tx, block_number=None), receipt, block))


def test_rpc_validation_rejects_bad_types_and_values() -> None:
    """Incorrect RPC types and corrupt calldata must not be silently accepted."""
    tx, receipt, block = scenario()
    with pytest.raises(EthereumNormalizationError, match="value_wei"):
        classify((replace(tx, value_wei=-1), receipt, block))
    with pytest.raises(EthereumNormalizationError, match="value_wei"):
        classify((replace(tx, value_wei=True), receipt, block))
    with pytest.raises(EthereumNormalizationError, match="calldata"):
        classify((replace(tx, input_data="approve"), receipt, block))  # type: ignore[arg-type]
    with pytest.raises(EthereumNormalizationError, match="sender"):
        classify((replace(tx, from_address="bad"), receipt, block))
    with pytest.raises(EthereumNormalizationError, match="recipient"):
        classify((replace(tx, to_address="bad"), receipt, block))
    with pytest.raises(EthereumNormalizationError, match="status"):
        classify((tx, replace(receipt, success=1), block))  # type: ignore[arg-type]
    with pytest.raises(EthereumNormalizationError, match="logs"):
        classify((tx, replace(receipt, logs=[]), block))  # type: ignore[arg-type]


def test_type_errors_for_wrong_arguments() -> None:
    """Caller mistakes are not converted into unclassified decisions."""
    tx, receipt, block = scenario()
    with pytest.raises(TypeError, match="transaction"):
        classify_approve(None, receipt, block, wallet_address=WALLET, observed_movements=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="receipt"):
        classify_approve(tx, None, block, wallet_address=WALLET, observed_movements=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="block"):
        classify_approve(tx, receipt, None, wallet_address=WALLET, observed_movements=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="observed_movements"):
        classify_approve(tx, receipt, block, wallet_address=WALLET, observed_movements=[])  # type: ignore[arg-type]
    with pytest.raises(EthereumNormalizationError, match="wallet_address"):
        classify_approve(tx, receipt, block, wallet_address="oops", observed_movements=())


def test_marker_is_immutable_and_independently_validated() -> None:
    """Even direct construction cannot produce a bogus accounting marker."""
    from dataclasses import FrozenInstanceError

    action = classify(scenario())
    assert isinstance(action, NoAccountingAction)
    with pytest.raises(FrozenInstanceError):
        action.allowance_raw = 0  # type: ignore[misc]
    with pytest.raises(ValueError, match="uint256"):
        replace(action, allowance_raw=2**256)
    with pytest.raises(ValueError, match="uint256"):
        replace(action, allowance_raw=-1)
    with pytest.raises(TypeError, match="int"):
        replace(action, allowance_raw=True)
    with pytest.raises(ValueError, match="Ethereum Mainnet ERC-20"):
        replace(action, asset_id=AssetId(1, None))
    with pytest.raises(ValueError, match="Ethereum Mainnet ERC-20"):
        replace(action, asset_id=AssetId(137, TOKEN))
    with pytest.raises(ValueError, match="owner"):
        replace(action, owner="invalid")


def test_allowance_is_not_gas_fee_or_token_quantity() -> None:
    """Approval permission is not a wallet amount; gas is a separate fact."""
    tx, receipt, block = scenario(amount=2**256 - 1)
    action = classify((tx, receipt, block))
    assert isinstance(action, NoAccountingAction)
    assert action.allowance_raw == 2**256 - 1
    assert receipt.gas_used == 46000
    assert receipt.effective_gas_price_wei == 10**9
    assert action.asset_id == AssetId(1, TOKEN)
    assert not hasattr(action, "quantity")


def test_corrupt_hash_and_movement_types_raise() -> None:
    """Malformatted RPC transaction identities and movement objects fail closed."""
    tx, receipt, block = scenario()
    with pytest.raises(EthereumNormalizationError, match="RPC transaction hash"):
        classify((replace(tx, tx_hash="bad"), receipt, block))
    with pytest.raises(EthereumNormalizationError, match="receipt transaction hash"):
        classify((tx, replace(receipt, tx_hash="bad"), block))
    with pytest.raises(TypeError, match="AssetMovement"):
        classify_approve(
            tx,
            receipt,
            block,
            wallet_address=WALLET,
            observed_movements=(None,),  # type: ignore[arg-type]
        )


def test_bad_direct_action_construction_fails() -> None:
    """The immutable marker validates metadata and contract identity itself."""
    action = classify(scenario())
    assert isinstance(action, NoAccountingAction)
    with pytest.raises(TypeError, match="EventMetadata"):
        replace(action, metadata=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AssetId"):
        replace(action, asset_id=TOKEN)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spender"):
        replace(action, spender="bad")

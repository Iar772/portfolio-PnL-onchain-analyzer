"""Conservative direct ETH/WETH9 conversion classification on Ethereum Mainnet."""

from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.events import EventMetadata, UnknownEvent, Unwrap, Wrap
from portfolio_analyzer.ethereum.direct_transfer_rules import (
    _NATIVE_METADATA,
    _assert_same_transaction,
)
from portfolio_analyzer.ethereum.known_contracts import (
    MAINNET_WETH9 as _WETH9,
)
from portfolio_analyzer.ethereum.known_contracts import (
    MAINNET_WETH9_METADATA as _WETH9_METADATA,
)
from portfolio_analyzer.ethereum.movements import MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
)
from portfolio_analyzer.ethereum.rpc_models import RpcLog, RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.wallet_deltas import (
    WalletAssetDelta,
    calculate_wallet_asset_delta,
)

_NATIVE_ETH = AssetId(chain_id=1, contract_address=None)
_DEPOSIT_SELECTOR = bytes.fromhex("d0e30db0")
_WITHDRAW_SELECTOR = bytes.fromhex("2e1a7d4d")
# Keccak-256 of the event signatures, not NIST SHA3-256.
_DEPOSIT_TOPIC = bytes.fromhex("e1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c")
_WITHDRAW_TOPIC = bytes.fromhex("7fcf532c15f0a6db0bd6d0e038bea71d30d808c7d98cb3bf7268a95bf5081b65")


def _metadata(group: TransactionMovementGroup) -> EventMetadata:
    """Build event provenance without adding financial assumptions.

    :param group: Confirmed transaction and movement records.
    :return: Immutable domain event metadata.
    """
    tx = group.transaction
    return EventMetadata(
        tx_hash=tx.tx_hash,
        block_number=tx.block_number,
        timestamp=tx.timestamp,
    )


def _unknown(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    reason: str,
) -> UnknownEvent:
    """Preserve trustworthy ETH/WETH deltas when conversion cannot be proven.

    Other assets remain intact in the original WalletAssetDelta; their
    decimals must not be guessed to populate UnknownEvent.movements.

    :param group: Confirmed transaction movement group.
    :param delta: Raw wallet-relative changes without gas.
    :param reason: Human-readable reason classification was rejected.
    :return: Explicitly unclassified economic event.
    """
    amounts = tuple(
        change.to_asset_amount(
            _NATIVE_METADATA if change.asset_id == _NATIVE_ETH else _WETH9_METADATA
        )
        for change in delta.changes
        if change.asset_id in (_NATIVE_ETH, _WETH9)
    )
    return UnknownEvent(metadata=_metadata(group), reason=reason, movements=amounts)


def _weth9_event_amount(receipt: RpcReceipt, wallet_address: str, event_topic: bytes) -> int | None:
    """Decode one exact WETH9 Deposit/Withdrawal log for the analyzed wallet.

    The canonical WETH9 contract emits no ERC-20 Transfer event when
    wrapping or unwrapping. The event's indexed address and unsigned wad
    are authoritative for the WETH-side amount.

    :param receipt: Successful transaction receipt with contract logs.
    :param wallet_address: Canonical lower-case wallet address.
    :param event_topic: Exact Keccak-256 event topic expected for direction.
    :return: Positive raw wad if precisely one valid log exists, else None.
    """
    if not isinstance(receipt.logs, tuple) or len(receipt.logs) != 1:
        return None
    log = receipt.logs[0]
    if not isinstance(log, RpcLog):
        return None
    if not isinstance(log.address, str) or log.address.lower() != _WETH9.contract_address:
        return None
    if (
        not isinstance(log.topics, tuple)
        or len(log.topics) != 2
        or log.topics[0] != event_topic
        or log.topics[1] != bytes(12) + bytes.fromhex(wallet_address[2:])
        or not isinstance(log.data, bytes)
        or len(log.data) != 32
    ):
        return None
    amount = int.from_bytes(log.data, byteorder="big")
    return amount if amount > 0 else None


def _native_leg_matches(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    *,
    direction: str,
    amount_wei: int,
) -> bool:
    """Verify an isolated native ETH movement, excluding unrelated legs/gas.

    :param group: Source movement group from the history provider.
    :param delta: Previously computed wallet-relative changes.
    :param direction: The expected WETH conversion direction.
    :param amount_wei: Exact positive amount established by WETH9 receipt.
    :return: True iff history agrees with that one wallet native transfer.
    """
    if len(group.movements) != 1 or len(delta.changes) != 1:
        return False
    movement = group.movements[0]
    net = delta.changes[0]
    if movement.asset_id != _NATIVE_ETH or net.asset_id != _NATIVE_ETH:
        return False
    if movement.raw_amount != amount_wei:
        return False
    if direction == "wrap":
        return (
            movement.category is MovementCategory.EXTERNAL
            and movement.from_address == delta.wallet_address
            and movement.to_address == _WETH9.contract_address
            and net.raw_units == -amount_wei
        )
    return (
        movement.category is MovementCategory.INTERNAL
        and movement.from_address == _WETH9.contract_address
        and movement.to_address == delta.wallet_address
        and net.raw_units == amount_wei
    )


def classify_eth_weth(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    transaction: RpcTransaction,
    receipt: RpcReceipt,
) -> Wrap | Unwrap | UnknownEvent | None:
    """Recognize **direct** canonical WETH9 wrap/unwrap, not market trades.

    The canonical WETH9 emits Deposit/Withdrawal events (not ERC-20
    Transfer mint/burn logs) and the wallet's native ETH movement is
    separately available through the transfer history. The missing WETH
    transfer is reconstructed solely from verified receipt logs.

    Only a successful, wallet-initiated, direct call to the Ethereum
    Mainnet WETH9 contract, with a single exact native movement and a
    single matching WETH9 event, qualifies. Unexpected/missing signals
    yield UnknownEvent, never an invented Swap or cost basis. Other
    contract methods and third-party calls are not classified here.

    :param group: Provider-neutral grouped transaction movements.
    :param delta: Exact wallet-relative changes excluding gas.
    :param transaction: Mined Ethereum RPC transaction facts.
    :param receipt: Execution status and WETH9 contract logs.
    :return: Wrap, Unwrap, UnknownEvent, or None for unrelated operations.
    :raises TypeError: For invalid top-level input types.
    :raises EthereumNormalizationError: For contradictory transaction facts.
    """
    if not isinstance(group, TransactionMovementGroup):
        raise TypeError("group must be a TransactionMovementGroup")
    if not isinstance(delta, WalletAssetDelta):
        raise TypeError("delta must be a WalletAssetDelta")
    if not isinstance(transaction, RpcTransaction):
        raise TypeError("transaction must be a RpcTransaction")
    if not isinstance(receipt, RpcReceipt):
        raise TypeError("receipt must be a RpcReceipt")

    _assert_same_transaction(group, delta, transaction, receipt)
    if calculate_wallet_asset_delta(group, delta.wallet_address).changes != delta.changes:
        raise EthereumNormalizationError("wallet deltas disagree with transaction movements")

    if (
        transaction.to_address is None
        or transaction.to_address.lower() != _WETH9.contract_address
        or transaction.from_address.lower() != delta.wallet_address
    ):
        return None

    calldata = transaction.input_data
    # deposit() and the payable fallback are both supported by canonical WETH9.
    is_wrap = calldata == _DEPOSIT_SELECTOR or calldata == b""
    is_unwrap = calldata.startswith(_WITHDRAW_SELECTOR)
    if not is_wrap and not is_unwrap:
        return None
    if receipt.success is not True:
        return _unknown(group, delta, "weth_execution_not_confirmed")

    if is_wrap:
        amount = _weth9_event_amount(receipt, delta.wallet_address, _DEPOSIT_TOPIC)
        if amount is None or transaction.value_wei != amount:
            return _unknown(group, delta, "weth_deposit_evidence_mismatch")
        if not _native_leg_matches(group, delta, direction="wrap", amount_wei=amount):
            return _unknown(group, delta, "weth_deposit_wallet_delta_mismatch")
        return Wrap(
            metadata=_metadata(group),
            amount_out=AssetAmount.from_raw_units(_NATIVE_ETH, amount, _NATIVE_METADATA),
            amount_in=AssetAmount.from_raw_units(_WETH9, amount, _WETH9_METADATA),
        )

    # Only the exact ABI-encoded withdraw(uint256) call qualifies.
    if len(calldata) != 36 or transaction.value_wei != 0:
        return _unknown(group, delta, "weth_withdraw_call_malformed")
    amount = int.from_bytes(calldata[4:], byteorder="big")
    log_amount = _weth9_event_amount(receipt, delta.wallet_address, _WITHDRAW_TOPIC)
    if amount <= 0 or log_amount != amount:
        return _unknown(group, delta, "weth_withdraw_evidence_mismatch")
    if not _native_leg_matches(group, delta, direction="unwrap", amount_wei=amount):
        return _unknown(group, delta, "weth_withdraw_wallet_delta_mismatch")
    return Unwrap(
        metadata=_metadata(group),
        amount_out=AssetAmount.from_raw_units(_WETH9, amount, _WETH9_METADATA),
        amount_in=AssetAmount.from_raw_units(_NATIVE_ETH, amount, _NATIVE_METADATA),
    )

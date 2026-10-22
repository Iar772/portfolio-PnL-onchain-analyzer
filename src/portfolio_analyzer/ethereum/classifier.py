"""Deterministic classification with a safe, diagnostic UNKNOWN outcome.

No external I/O and no pricing or FIFO mutations occur here. Provenance
contradictions fail explicitly; unsupported *valid* transactions are UNKNOWN.
"""

from collections.abc import Mapping

from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.domain.events import (
    EconomicEvent,
    EventMetadata,
    UnknownEvent,
)
from portfolio_analyzer.ethereum.approve import NoAccountingAction, classify_approve
from portfolio_analyzer.ethereum.direct_transfer_rules import (
    _NATIVE_METADATA,
    _assert_same_transaction,
)
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
)
from portfolio_analyzer.ethereum.rpc_models import RpcBlock, RpcLog, RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.swap import classify_swap_v1
from portfolio_analyzer.ethereum.transfer_in import classify_transfer_in
from portfolio_analyzer.ethereum.transfer_out import classify_transfer_out
from portfolio_analyzer.ethereum.wallet_deltas import (
    WalletAssetDelta,
    calculate_wallet_asset_delta,
)
from portfolio_analyzer.ethereum.weth import classify_eth_weth


def _unknown_with_deltas(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    token_metadata: Mapping[AssetId, AssetMetadata],
    *,
    reason: str,
) -> UnknownEvent:
    """Keep independently verified normalized legs in an UNKNOWN event.

    Unresolved token quantities remain in the caller's WalletAssetDelta as
    exact signed raw units. Missing decimals are never guessed, and no
    displayed quantity is treated as an acquisition or disposal.

    :param group: Mined group providing transaction provenance.
    :param delta: Wallet-relative raw changes without gas.
    :param token_metadata: Confirmed per-contract token decimals.
    :param reason: Safe, stable, non-vendor diagnostic reason.
    :return: Unknown event with the known normalized changes only.
    :raises EthereumNormalizationError: If supplied metadata is invalid.
    """
    amounts = []
    for change in delta.changes:
        metadata = (
            _NATIVE_METADATA if change.asset_id.is_native else token_metadata.get(change.asset_id)
        )
        if metadata is None:
            continue
        if not isinstance(metadata, AssetMetadata):
            raise EthereumNormalizationError("token metadata must be AssetMetadata")
        amounts.append(change.to_asset_amount(metadata))

    return UnknownEvent(
        metadata=EventMetadata(
            tx_hash=group.transaction.tx_hash,
            block_number=group.transaction.block_number,
            timestamp=group.transaction.timestamp,
        ),
        reason=reason,
        movements=tuple(amounts),
    )


def classify_transaction(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    transaction: RpcTransaction,
    receipt: RpcReceipt,
    *,
    token_metadata: Mapping[AssetId, AssetMetadata],
) -> EconomicEvent:
    """Classify one transfer-bearing transaction, always yielding an event.

    Verified WETH9 conversions are checked before generic transfers/swaps.
    A specialist's UnknownEvent is FINAL: no later, less-specific classifier
    can override that uncertainty. A specialist's ``None`` means only that
    it did not recognize the transaction; UNKNOWN is then the last resort.

    Provider failures, inconsistent RPC provenance, and stale wallet deltas
    are *not* classified as UNKNOWN: the caller must repair missing data.
    This routine assumes the HistoryProvider has finished all pages.

    :param group: Fully loaded and validated transaction movements.
    :param delta: Its wallet-relative net changes excluding gas.
    :param transaction: Mined RPC transaction from the same block.
    :param receipt: Receipt for the same mined transaction.
    :param token_metadata: Known ERC-20 metadata keyed by contract identity.
    :return: TransferIn, TransferOut, Swap, Wrap, Unwrap, or UnknownEvent.
    :raises TypeError: If top-level arguments have incorrect types.
    :raises EthereumNormalizationError: For contradictory source facts.
    """
    if not isinstance(group, TransactionMovementGroup):
        raise TypeError("group must be TransactionMovementGroup")
    if not isinstance(delta, WalletAssetDelta):
        raise TypeError("delta must be WalletAssetDelta")
    if not isinstance(transaction, RpcTransaction):
        raise TypeError("transaction must be RpcTransaction")
    if not isinstance(receipt, RpcReceipt):
        raise TypeError("receipt must be RpcReceipt")
    if not isinstance(token_metadata, Mapping):
        raise TypeError("token_metadata must be a mapping")

    _assert_same_transaction(group, delta, transaction, receipt)
    if calculate_wallet_asset_delta(group, delta.wallet_address).changes != delta.changes:
        raise EthereumNormalizationError("wallet deltas disagree with movements")

    # A reverted transaction emits no successful event, even a zero-value one.
    # A contradiction here is corrupt/incomplete evidence, not an unknown trade.
    if receipt.success is False:
        raise EthereumNormalizationError("reverted receipt contradicts movement history")
    if not isinstance(receipt.logs, tuple) or any(
        not isinstance(log, RpcLog) for log in receipt.logs
    ):
        raise EthereumNormalizationError("receipt logs must be a tuple of RpcLog")
    if type(receipt.gas_used) is not int or receipt.gas_used < 0:
        raise EthereumNormalizationError("invalid receipt gas_used")
    if receipt.effective_gas_price_wei is not None and (
        type(receipt.effective_gas_price_wei) is not int or receipt.effective_gas_price_wei < 0
    ):
        raise EthereumNormalizationError("invalid receipt effective gas price")

    if receipt.success is not True:
        return _unknown_with_deltas(
            group, delta, token_metadata, reason="execution_status_unconfirmed"
        )

    weth = classify_eth_weth(group, delta, transaction, receipt)
    if weth is not None:
        return weth

    received = classify_transfer_in(
        group, delta, transaction, receipt, token_metadata=token_metadata
    )
    if received is not None:
        return received

    sent = classify_transfer_out(group, delta, transaction, receipt, token_metadata=token_metadata)
    if sent is not None:
        return sent

    swap = classify_swap_v1(group, delta, transaction, receipt, token_metadata=token_metadata)
    if swap is not None:
        return swap

    reason = "no_wallet_asset_change" if not delta.changes else "unsupported_transaction"
    if any(
        not change.asset_id.is_native and token_metadata.get(change.asset_id) is None
        for change in delta.changes
    ):
        reason = "missing_asset_metadata"

    return _unknown_with_deltas(group, delta, token_metadata, reason=reason)


def classify_transaction_without_movements(
    transaction: RpcTransaction,
    receipt: RpcReceipt,
    block: RpcBlock,
    *,
    wallet_address: str,
    movements_complete: bool,
) -> NoAccountingAction | UnknownEvent:
    """Classify one independently discovered tx without observed transfers.

    A normal transfers-only history cannot discover standalone approvals.
    The caller must establish separately that the complete movement history
    for this transaction was checked; otherwise a missing transfer is not
    proof of no accounting impact.

    :param transaction: Mined RPC transaction discovered outside transfers.
    :param receipt: Matching mined transaction receipt.
    :param block: Block containing the transaction and authoritative timestamp.
    :param wallet_address: Wallet whose action is being classified.
    :param movements_complete: Explicit confirmation of complete transfer evidence.
    :return: A proven no-accounting approval or an UnknownEvent.
    :raises TypeError: For wrong completeness flag types.
    :raises EthereumNormalizationError: If evidence is incomplete or inconsistent.
    """
    if type(movements_complete) is not bool:
        raise TypeError("movements_complete must be bool")
    if not movements_complete:
        raise EthereumNormalizationError("cannot classify incomplete movement history")

    approval = classify_approve(
        transaction,
        receipt,
        block,
        wallet_address=wallet_address,
        observed_movements=(),
    )
    if approval is not None:
        return approval

    # A successful direct native-value transfer touching this wallet cannot
    # disappear from a supposedly complete set of movement facts.
    if (
        receipt.success is True
        and transaction.value_wei > 0
        and (
            transaction.from_address.lower() == wallet_address.lower()
            or (
                transaction.to_address is not None
                and transaction.to_address.lower() == wallet_address.lower()
            )
        )
    ):
        raise EthereumNormalizationError("native value transfer missing from complete history")

    return UnknownEvent(
        metadata=EventMetadata(
            tx_hash=transaction.tx_hash,
            block_number=block.number,
            timestamp=block.timestamp,
        ),
        reason="unsupported_no_transfer_action",
    )

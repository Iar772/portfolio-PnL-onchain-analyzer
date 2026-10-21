"""Conservative classification of directly received Ethereum transfers."""

from collections.abc import Mapping

from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.domain.events import EventMetadata, TransferIn
from portfolio_analyzer.ethereum.direct_transfer_rules import (
    _NATIVE_METADATA,
    _assert_same_transaction,
    _is_direct_erc20_transfer,
)
from portfolio_analyzer.ethereum.movements import MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
)
from portfolio_analyzer.ethereum.rpc_models import RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.wallet_deltas import WalletAssetDelta


def classify_transfer_in(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    transaction: RpcTransaction,
    receipt: RpcReceipt,
    *,
    token_metadata: Mapping[AssetId, AssetMetadata],
) -> TransferIn | None:
    """Recognize a simple externally initiated, single-leg inbound transfer.

    A positive net delta alone is not sufficient evidence: a swap, claim, or
    contract execution can have only one visible incoming transfer. Therefore,
    native ETH must be a plain value transfer and ERC-20 must be a direct
    ``transfer(address,uint256)`` call whose sender and calldata match the
    single mined movement. More complex cases remain unclassified for future
    classifiers (swaps, claims, bridges, and exchange matching).

    The emitted TransferIn carries no assumed USD acquisition cost. Downstream
    inventory creation must use ``Lot.open(..., cost_basis=None)`` until
    independent provenance can establish a known basis. Price is never inferred.

    :param group: One confirmed transaction with ordered movement facts.
    :param delta: Wallet-relative transfer net from E5.2, excluding gas.
    :param transaction: RPC transaction facts independent of Alchemy DTOs.
    :param receipt: RPC receipt establishing successful execution.
    :param token_metadata: Verified token metadata keyed by canonical AssetId.
    :return: TransferIn with unknown USD basis, or None if not safely recognized.
    :raises TypeError: If any top-level arguments have the wrong types.
    :raises EthereumNormalizationError: For contradictory provenance or metadata types.
    """
    if not isinstance(group, TransactionMovementGroup):
        raise TypeError("group must be a TransactionMovementGroup")
    if not isinstance(delta, WalletAssetDelta):
        raise TypeError("delta must be a WalletAssetDelta")
    if not isinstance(transaction, RpcTransaction):
        raise TypeError("transaction must be a RpcTransaction")
    if not isinstance(receipt, RpcReceipt):
        raise TypeError("receipt must be a RpcReceipt")
    if not isinstance(token_metadata, Mapping):
        raise TypeError("token_metadata must be a mapping")

    _assert_same_transaction(group, delta, transaction, receipt)
    if receipt.success is not True or len(group.movements) != 1 or len(delta.changes) != 1:
        return None

    movement = group.movements[0]
    change = delta.changes[0]
    wallet = delta.wallet_address
    if (
        movement.to_address != wallet
        or movement.from_address == wallet
        or movement.raw_amount <= 0
        or change.asset_id != movement.asset_id
        or change.raw_units != movement.raw_amount
        or transaction.from_address.lower() != movement.from_address
    ):
        return None

    if movement.category is MovementCategory.EXTERNAL:
        if (
            not movement.asset_id.is_native
            or transaction.to_address is None
            or transaction.to_address.lower() != wallet
            or transaction.value_wei != movement.raw_amount
            or transaction.input_data
        ):
            return None
        metadata = _NATIVE_METADATA
    elif movement.category is MovementCategory.ERC20:
        if (
            movement.asset_id.is_native
            or transaction.to_address is None
            or transaction.to_address.lower() != movement.asset_id.contract_address
            or transaction.value_wei != 0
            or not _is_direct_erc20_transfer(
                input_data=transaction.input_data,
                receiver=wallet,
                raw_amount=movement.raw_amount,
            )
        ):
            return None
        metadata = token_metadata.get(movement.asset_id)
        if metadata is None:
            return None
        if not isinstance(metadata, AssetMetadata):
            raise EthereumNormalizationError("token metadata must be AssetMetadata")
    else:
        # Contract-internal ETH receipts may be payouts, refunds, or swaps.
        return None

    return TransferIn(
        metadata=EventMetadata(
            tx_hash=group.transaction.tx_hash,
            block_number=group.transaction.block_number,
            timestamp=group.transaction.timestamp,
        ),
        amount=change.to_asset_amount(metadata),
    )

"""Conservative classification of direct outgoing Ethereum transfers."""

from collections.abc import Mapping

from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata
from portfolio_analyzer.domain.events import EventMetadata, TransferOut
from portfolio_analyzer.ethereum.direct_transfer_rules import (
    _NATIVE_METADATA,
    _assert_same_transaction,
    _is_direct_erc20_transfer,
)
from portfolio_analyzer.ethereum.known_contracts import MAINNET_WETH9
from portfolio_analyzer.ethereum.movements import MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
)
from portfolio_analyzer.ethereum.rpc_models import RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.wallet_deltas import WalletAssetDelta


def classify_transfer_out(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    transaction: RpcTransaction,
    receipt: RpcReceipt,
    *,
    token_metadata: Mapping[AssetId, AssetMetadata],
) -> TransferOut | None:
    """Recognize a successful, simple wallet-originated outgoing transfer.

    Only a direct ETH value transfer (empty calldata) or a direct standard
    ERC-20 transfer(address,uint256) from the analyzed wallet qualifies.
    An outgoing net delta is not by itself proof of a transfer: DEX trades,
    contract calls, bridges and exchange deposits need further evidence.

    TransferOut carries **no proceeds**: it records an inventory disposal,
    not a sale. Later accounting can consume inventory with FIFO without
    manufacturing USD realized PnL. Transaction gas is not part of this event.

    :param group: One mined transaction with its ordered movement facts.
    :param delta: Wallet-relative raw-unit net movements, excluding gas.
    :param transaction: Provider-neutral mined RPC transaction facts.
    :param receipt: Provider-neutral receipt for execution status.
    :param token_metadata: Trusted asset metadata keyed by AssetId.
    :return: TransferOut or None when evidence is insufficient.
    :raises TypeError: For invalid argument types.
    :raises EthereumNormalizationError: For contradictory RPC provenance or metadata.
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
        movement.from_address != wallet
        or movement.to_address == wallet
        or movement.raw_amount <= 0
        or change.asset_id != movement.asset_id
        or change.raw_units != -movement.raw_amount
        or transaction.from_address.lower() != wallet
    ):
        return None

    if movement.category is MovementCategory.EXTERNAL:
        if (
            not movement.asset_id.is_native
            or transaction.to_address is None
            or transaction.to_address.lower() != movement.to_address
            or transaction.value_wei != movement.raw_amount
            or transaction.input_data
            or receipt.logs
            # An ETH payment to canonical WETH9 may execute its deposit fallback.
            # If its receipt logs are unavailable, it is still not provably a gift.
            or transaction.to_address.lower() == MAINNET_WETH9.contract_address
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
                receiver=movement.to_address,
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
        # Internal native payouts are not plain wallet-originated transfers.
        return None

    return TransferOut(
        metadata=EventMetadata(
            tx_hash=group.transaction.tx_hash,
            block_number=group.transaction.block_number,
            timestamp=group.transaction.timestamp,
        ),
        amount=AssetAmount.from_raw_units(movement.asset_id, movement.raw_amount, metadata),
    )

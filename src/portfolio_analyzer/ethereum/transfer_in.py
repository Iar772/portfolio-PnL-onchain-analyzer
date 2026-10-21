"""Conservative classification of directly received Ethereum transfers."""

import re
from collections.abc import Mapping

from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.domain.events import EventMetadata, TransferIn
from portfolio_analyzer.ethereum.movements import MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
)
from portfolio_analyzer.ethereum.rpc_models import RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.wallet_deltas import WalletAssetDelta

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}\Z")
_NATIVE_METADATA = AssetMetadata(symbol="ETH", name="Ether", decimals=18)
_ERC20_TRANSFER_SELECTOR = bytes.fromhex("a9059cbb")


def _assert_same_transaction(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    transaction: RpcTransaction,
    receipt: RpcReceipt,
) -> None:
    """Reject conflicting chain facts before attempting economic classification.

    :param group: Transaction movements to classify.
    :param delta: Wallet-relative net movements for that transaction.
    :param transaction: Independently fetched Ethereum transaction facts.
    :param receipt: Independently fetched Ethereum execution receipt.
    :raises EthereumNormalizationError: If provenance or RPC facts disagree.
    """
    expected = group.transaction
    if delta.transaction != expected or delta.transaction_index != group.transaction_index:
        raise EthereumNormalizationError("wallet delta does not match transaction group")
    for source_hash in (transaction.tx_hash, receipt.tx_hash):
        if not isinstance(source_hash, str) or not re.fullmatch(r"0x[0-9a-fA-F]{64}", source_hash):
            raise EthereumNormalizationError("RPC transaction hash must be 32-byte hex")
    if (
        transaction.tx_hash.lower() != expected.tx_hash
        or receipt.tx_hash.lower() != expected.tx_hash
    ):
        raise EthereumNormalizationError("RPC transaction identity does not match group")
    if type(transaction.block_number) is not int or type(receipt.block_number) is not int:
        raise EthereumNormalizationError("RPC block must be a mined integer")
    if (
        transaction.block_number != expected.block_number
        or receipt.block_number != expected.block_number
    ):
        raise EthereumNormalizationError("RPC block does not match transaction group")
    if type(transaction.value_wei) is not int or transaction.value_wei < 0:
        raise EthereumNormalizationError("transaction value_wei must be nonnegative integer")
    if not isinstance(transaction.input_data, bytes):
        raise EthereumNormalizationError("transaction input_data must be bytes")
    if not isinstance(transaction.from_address, str) or not _ADDRESS_RE.fullmatch(
        transaction.from_address
    ):
        raise EthereumNormalizationError("transaction from_address is invalid")
    if transaction.to_address is not None and (
        not isinstance(transaction.to_address, str)
        or not _ADDRESS_RE.fullmatch(transaction.to_address)
    ):
        raise EthereumNormalizationError("transaction to_address is invalid")
    if receipt.success is not None and type(receipt.success) is not bool:
        raise EthereumNormalizationError("receipt success must be bool or None")


def _is_direct_erc20_transfer(
    *,
    input_data: bytes,
    receiver: str,
    raw_amount: int,
) -> bool:
    """Recognize exactly one standard ERC-20 transfer(address,uint256) call.

    :param input_data: Transaction ABI calldata.
    :param receiver: Validated movement recipient address.
    :param raw_amount: Observed ERC-20 Transfer event quantity in base units.
    :return: Whether the call arguments match the observed incoming transfer.
    """
    if len(input_data) != 68 or not input_data.startswith(_ERC20_TRANSFER_SELECTOR):
        return False
    encoded_receiver = input_data[4:36]
    encoded_amount = input_data[36:68]
    return (
        encoded_receiver == bytes(12) + bytes.fromhex(receiver[2:])
        and int.from_bytes(encoded_amount, "big") == raw_amount
    )


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

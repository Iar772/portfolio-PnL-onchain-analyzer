"""Reusable validation rules for conservative direct transfers."""

import re

from portfolio_analyzer.domain.assets import AssetMetadata
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
    :param raw_amount: Observed ERC-20 Transfer event amount in base units.
    :return: Whether the call arguments match the observed on-chain transfer.
    """
    if len(input_data) != 68 or not input_data.startswith(_ERC20_TRANSFER_SELECTOR):
        return False
    encoded_receiver = input_data[4:36]
    encoded_amount = input_data[36:68]
    return (
        encoded_receiver == bytes(12) + bytes.fromhex(receiver[2:])
        and int.from_bytes(encoded_amount, "big") == raw_amount
    )

"""Conservative classification of standalone ERC-20 allowance approvals.

An allowance change is an on-chain action, NOT an asset transfer or economic
trade. These markers are deliberately not part of domain.EconomicEvent.
"""

import re
from dataclasses import dataclass

from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.events import EventMetadata
from portfolio_analyzer.ethereum.movements import AssetMovement
from portfolio_analyzer.ethereum.normalizer import EthereumNormalizationError
from portfolio_analyzer.ethereum.rpc_models import RpcBlock, RpcLog, RpcReceipt, RpcTransaction

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}\Z")
_HASH_RE = re.compile(r"0x[0-9a-fA-F]{64}\Z")
_APPROVE_SELECTOR = bytes.fromhex("095ea7b3")
_APPROVAL_TOPIC = bytes.fromhex("8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925")


@dataclass(frozen=True, slots=True)
class NoAccountingAction:
    """Record a proven standalone approval without an inventory change.

    This Ethereum-specific observation is **not** an EconomicEvent. Downstream
    accounting must not turn it into a FIFO acquisition or disposal. Approval
    gas remains a separate receipt-level expense for later processing.

    :param metadata: Transaction hash, block and observed time.
    :param asset_id: ERC-20 token contract identity, not its ticker.
    :param owner: Wallet changing the allowance.
    :param spender: Contract/account granted the allowance.
    :param allowance_raw: Exact unsigned uint256 allowance in token raw units.
    :raises TypeError: If the fields have invalid runtime types.
    :raises ValueError: If the asset or amount violates the approval contract.
    """

    metadata: EventMetadata
    asset_id: AssetId
    owner: str
    spender: str
    allowance_raw: int

    def __post_init__(self) -> None:
        """Validate a no-accounting marker independently of its classifier.

        :raises TypeError: For invalid metadata, asset, address, or amount types.
        :raises ValueError: For native assets, other chains, or invalid amounts.
        """
        if not isinstance(self.metadata, EventMetadata):
            raise TypeError("metadata must be EventMetadata")
        if not isinstance(self.asset_id, AssetId):
            raise TypeError("asset_id must be AssetId")
        if self.asset_id.chain_id != 1 or self.asset_id.is_native:
            raise ValueError("approve must target an Ethereum Mainnet ERC-20")
        for field in ("owner", "spender"):
            address = getattr(self, field)
            if not isinstance(address, str) or not _ADDRESS_RE.fullmatch(address):
                raise ValueError(f"{field} must be a 20-byte hex address")
            object.__setattr__(self, field, address.lower())
        if type(self.allowance_raw) is not int:
            raise TypeError("allowance_raw must be int")
        if not 0 <= self.allowance_raw < 2**256:
            raise ValueError("allowance_raw must fit uint256")


def _validate_provenance(
    transaction: RpcTransaction,
    receipt: RpcReceipt,
    block: RpcBlock,
    observed_movements: tuple[AssetMovement, ...],
) -> None:
    """Fail on contradictory RPC and movement identities rather than hiding gaps.

    :param transaction: RPC transaction facts.
    :param receipt: Receipt facts.
    :param block: Block providing the timestamp.
    :param observed_movements: Explicitly supplied transfer facts (may be empty).
    :raises EthereumNormalizationError: If the evidence is inconsistent.
    """
    if not isinstance(transaction.tx_hash, str) or not _HASH_RE.fullmatch(transaction.tx_hash):
        raise EthereumNormalizationError("invalid RPC transaction hash")
    if not isinstance(receipt.tx_hash, str) or not _HASH_RE.fullmatch(receipt.tx_hash):
        raise EthereumNormalizationError("invalid receipt transaction hash")
    if transaction.tx_hash.lower() != receipt.tx_hash.lower():
        raise EthereumNormalizationError("transaction and receipt hash differ")
    if (
        type(transaction.block_number) is not int
        or type(receipt.block_number) is not int
        or type(block.number) is not int
        or transaction.block_number < 0
        or not (transaction.block_number == receipt.block_number == block.number)
    ):
        raise EthereumNormalizationError("transaction, receipt and block height differ")
    if type(transaction.value_wei) is not int or transaction.value_wei < 0:
        raise EthereumNormalizationError("transaction value_wei must be nonnegative integer")
    if not isinstance(transaction.input_data, bytes):
        raise EthereumNormalizationError("transaction calldata must be bytes")
    if not isinstance(transaction.from_address, str) or not _ADDRESS_RE.fullmatch(
        transaction.from_address
    ):
        raise EthereumNormalizationError("invalid transaction sender")
    if transaction.to_address is not None and (
        not isinstance(transaction.to_address, str)
        or not _ADDRESS_RE.fullmatch(transaction.to_address)
    ):
        raise EthereumNormalizationError("invalid transaction recipient")
    if receipt.success is not None and type(receipt.success) is not bool:
        raise EthereumNormalizationError("invalid receipt execution status")
    if not isinstance(receipt.logs, tuple) or any(
        not isinstance(log, RpcLog) for log in receipt.logs
    ):
        raise EthereumNormalizationError("invalid receipt logs")
    for movement in observed_movements:
        if not isinstance(movement, AssetMovement):
            raise TypeError("observed_movements must contain AssetMovement")
        if (
            movement.tx_hash != transaction.tx_hash.lower()
            or movement.block_number != block.number
            or movement.timestamp != block.timestamp
        ):
            raise EthereumNormalizationError("movement does not match transaction block")


def classify_approve(
    transaction: RpcTransaction,
    receipt: RpcReceipt,
    block: RpcBlock,
    *,
    wallet_address: str,
    observed_movements: tuple[AssetMovement, ...],
) -> NoAccountingAction | None:
    """Recognize a standalone ERC-20 approve with no observable asset transfers.

    Require direct ``approve(address,uint256)`` calldata and one exact
    ``Approval(owner,spender,value)`` receipt log from the target contract.
    Other logs or observed movements make the transaction non-standalone.
    A failed transaction, an unknown status, or incomplete proof is not
    positively classified. The result cannot enter the EconomicEvent union.

    Crucially, E5.1 only groups transactions that have movements. For a pure
    approve, callers must obtain the transaction via an independent source;
    no transfer group exists. Do not silently assume that transfer history
    contains every approval transaction.

    :param transaction: Provider-neutral transaction facts.
    :param receipt: Provider-neutral execution status and logs.
    :param block: Timestamp and block provenance.
    :param wallet_address: Wallet being analyzed.
    :param observed_movements: All observed transfers for this transaction.
    :return: NoAccountingAction on exact proof, otherwise None.
    :raises TypeError: For invalid top-level argument types.
    :raises EthereumNormalizationError: For contradictory provenance.
    """
    if not isinstance(transaction, RpcTransaction):
        raise TypeError("transaction must be RpcTransaction")
    if not isinstance(receipt, RpcReceipt):
        raise TypeError("receipt must be RpcReceipt")
    if not isinstance(block, RpcBlock):
        raise TypeError("block must be RpcBlock")
    if not isinstance(wallet_address, str) or not _ADDRESS_RE.fullmatch(wallet_address):
        raise EthereumNormalizationError("wallet_address must be 20-byte hex")
    if not isinstance(observed_movements, tuple):
        raise TypeError("observed_movements must be a tuple")
    _validate_provenance(transaction, receipt, block, observed_movements)

    owner = wallet_address.lower()
    token_contract = transaction.to_address
    calldata = transaction.input_data
    if (
        receipt.success is not True
        or transaction.from_address.lower() != owner
        or token_contract is None
        or transaction.value_wei != 0
        or observed_movements
        or len(calldata) != 68
        or not calldata.startswith(_APPROVE_SELECTOR)
        or calldata[4:16] != bytes(12)
        or len(receipt.logs) != 1
    ):
        return None

    spender = "0x" + calldata[16:36].hex()
    allowance_raw = int.from_bytes(calldata[36:68], "big")
    log = receipt.logs[0]
    if (
        not isinstance(log.address, str)
        or log.address.lower() != token_contract.lower()
        or not isinstance(log.topics, tuple)
        or len(log.topics) != 3
        or log.topics[0] != _APPROVAL_TOPIC
        or log.topics[1] != bytes(12) + bytes.fromhex(owner[2:])
        or log.topics[2] != bytes(12) + bytes.fromhex(spender[2:])
        or not isinstance(log.data, bytes)
        or len(log.data) != 32
        or int.from_bytes(log.data, "big") != allowance_raw
    ):
        return None

    return NoAccountingAction(
        metadata=EventMetadata(
            tx_hash=transaction.tx_hash,
            block_number=block.number,
            timestamp=block.timestamp,
        ),
        asset_id=AssetId(chain_id=1, contract_address=token_contract),
        owner=owner,
        spender=spender,
        allowance_raw=allowance_raw,
    )

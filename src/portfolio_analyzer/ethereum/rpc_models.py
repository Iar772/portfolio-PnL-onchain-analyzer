"""Provider-neutral Ethereum RPC facts, independent of JSON-RPC payloads."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RpcTransaction:
    """Describe a chain transaction without assuming it was mined.

    :param tx_hash: Canonical transaction hash.
    :param from_address: Sender Ethereum address.
    :param to_address: Recipient address or None for contract creation.
    :param block_number: Mined block number, or None while pending.
    :param value_wei: Native ETH value in integer wei.
    :param input_data: Raw transaction calldata.
    """

    tx_hash: str
    from_address: str
    to_address: str | None
    block_number: int | None
    value_wei: int
    input_data: bytes


@dataclass(frozen=True, slots=True)
class RpcLog:
    """Preserve an EVM receipt log without vendor response fields.

    :param address: Emitting contract address.
    :param topics: Ordered 32-byte event topics.
    :param data: Raw ABI-encoded log data.
    """

    address: str
    topics: tuple[bytes, ...]
    data: bytes


@dataclass(frozen=True, slots=True)
class RpcReceipt:
    """Describe execution outcome, fee facts, and emitted logs.

    :param tx_hash: Transaction identifier.
    :param block_number: Block containing the receipt.
    :param success: Execution result, or None for pre-status legacy receipts.
    :param gas_used: Gas units consumed by the transaction.
    :param effective_gas_price_wei: Actual price per gas unit, if present.
    :param logs: Ordered emitted logs.
    """

    tx_hash: str
    block_number: int
    success: bool | None
    gas_used: int
    effective_gas_price_wei: int | None
    logs: tuple[RpcLog, ...]


@dataclass(frozen=True, slots=True)
class RpcBlock:
    """Provide block timestamp for transaction chronological ordering.

    :param number: Mined Ethereum block number.
    :param timestamp: Timestamp of block in UTC.
    """

    number: int
    timestamp: datetime

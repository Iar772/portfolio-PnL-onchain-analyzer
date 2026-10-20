"""Deterministic validation for vendor-neutral Ethereum transaction envelopes."""

from datetime import UTC, datetime, timedelta, timezone
from typing import cast

import pytest

from portfolio_analyzer.ethereum.models import EthereumTransaction


def test_ethereum_transaction_canonicalizes_hash_and_utc_timestamp() -> None:
    """Canonicalize chain transaction metadata without any Alchemy response."""
    tz = timezone(timedelta(hours=2))
    timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=tz)
    transaction = EthereumTransaction("0x" + "AB" * 32, 123, timestamp)
    assert transaction.tx_hash == "0x" + "ab" * 32
    assert transaction.block_number == 123
    assert transaction.timestamp == datetime(2026, 1, 1, 10, tzinfo=UTC)
    assert transaction.timestamp.tzinfo is UTC


@pytest.mark.parametrize("value", ["", "0x" + "a" * 63, "0x" + "z" * 64])
def test_ethereum_transaction_rejects_malformed_hash(value: str) -> None:
    """Reject hashes not matching Ethereum 32-byte identity.

    :param value: Malformed transaction hash.
    """
    with pytest.raises(ValueError, match="32-byte"):
        EthereumTransaction(value, 12, datetime.now(UTC))


def test_ethereum_transaction_rejects_naive_time() -> None:
    """Reject missing timezone rather than inferring one."""
    with pytest.raises(ValueError, match="timezone-aware"):
        EthereumTransaction("0x" + "ab" * 32, 1, datetime(2026, 1, 1))


@pytest.mark.parametrize("value", [-1])
def test_ethereum_transaction_rejects_negative_block_number(value: int) -> None:
    """Reject a negative block height.

    :param value: Out-of-range block number.
    """
    with pytest.raises(ValueError, match="negative"):
        EthereumTransaction("0x" + "ab" * 32, value, datetime.now(UTC))


@pytest.mark.parametrize("value", [1.5, True, None])
def test_ethereum_transaction_rejects_wrong_block_type(value: object) -> None:
    """Reject non-integer block heights at the runtime boundary.

    :param value: Invalid block height.
    """
    with pytest.raises(TypeError, match="block_number"):
        EthereumTransaction("0x" + "ab" * 32, cast(int, value), datetime.now(UTC))

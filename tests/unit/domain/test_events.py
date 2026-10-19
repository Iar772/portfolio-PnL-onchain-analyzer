"""Offline contract tests for unvalued economic events."""

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import cast

import pytest

from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.events import (
    EconomicEvent,
    EventMetadata,
    Swap,
    TransferIn,
    TransferOut,
    UnknownEvent,
    Unwrap,
    Wrap,
)


@pytest.fixture
def metadata() -> EventMetadata:
    """Provide a deterministic UTC source transaction reference.

    :return: Valid event metadata for offline tests.
    """
    return EventMetadata(
        tx_hash="example-transaction",
        block_number=22_000_000,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def eth() -> AssetId:
    """Provide the Ethereum mainnet native-asset identity.

    :return: Canonical ETH asset identity.
    """
    return AssetId(chain_id=1, contract_address=None)


@pytest.fixture
def weth() -> AssetId:
    """Provide a representative wrapped-token contract identity.

    :return: Non-native token identity on Ethereum mainnet.
    """
    return AssetId(chain_id=1, contract_address="0x" + "aa" * 20)


@pytest.fixture
def link() -> AssetId:
    """Provide a second distinct ERC-20 asset identity.

    :return: Non-native token identity on Ethereum mainnet.
    """
    return AssetId(chain_id=1, contract_address="0x" + "bb" * 20)


def test_metadata_converts_offset_timestamp_to_utc() -> None:
    """Normalize an aware source timestamp to the UTC replay clock."""
    timestamp = datetime(2026, 6, 1, 15, 30, tzinfo=timezone(timedelta(hours=3)))
    metadata = EventMetadata("example", 1, timestamp)

    assert metadata.timestamp == datetime(2026, 6, 1, 12, 30, tzinfo=UTC)
    assert metadata.timestamp.tzinfo == UTC


@pytest.mark.parametrize("block", [True, "42", 42.0, None])
def test_metadata_rejects_invalid_block_types(block: object) -> None:
    """Reject runtime non-integer block numbers.

    :param block: Invalid block-number input.
    """
    with pytest.raises(TypeError, match="must be an integer"):
        EventMetadata("tx", cast(int, block), datetime(2026, 1, 1, tzinfo=UTC))


def test_metadata_rejects_negative_block() -> None:
    """Reject negative source block numbers."""
    with pytest.raises(ValueError, match="not be negative"):
        EventMetadata("tx", -1, datetime(2026, 1, 1, tzinfo=UTC))


@pytest.mark.parametrize("tx_hash", ["", "  "])
def test_metadata_rejects_blank_tx_hash(tx_hash: str) -> None:
    """Reject missing provenance identifiers.

    :param tx_hash: An empty or whitespace-only transaction identifier.
    """
    with pytest.raises(ValueError, match="must not be blank"):
        EventMetadata(tx_hash, 1, datetime(2026, 1, 1, tzinfo=UTC))


def test_metadata_rejects_naive_timestamp() -> None:
    """Reject ambiguous timestamps without timezone information."""
    with pytest.raises(ValueError, match="timezone-aware"):
        EventMetadata("tx", 1, datetime(2026, 1, 1))


def test_metadata_rejects_non_datetime() -> None:
    """Reject timestamp inputs that do not contain datetime objects."""
    with pytest.raises(TypeError, match="must be a datetime"):
        EventMetadata("tx", 1, cast(datetime, "2026-01-01T00:00:00Z"))


@pytest.mark.parametrize("quantity", [Decimal("0"), Decimal("-0.1")])
def test_transfer_in_requires_positive_magnitude(
    metadata: EventMetadata, eth: AssetId, quantity: Decimal
) -> None:
    """Reject zero and negative incoming transfer magnitudes.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    :param quantity: Invalid incoming magnitude.
    """
    with pytest.raises(ValueError, match="must be positive"):
        TransferIn(metadata, AssetAmount(eth, quantity))


@pytest.mark.parametrize("quantity", [Decimal("0"), Decimal("-2")])
def test_transfer_out_requires_positive_magnitude(
    metadata: EventMetadata, eth: AssetId, quantity: Decimal
) -> None:
    """Reject zero and negative outgoing transfer magnitudes.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    :param quantity: Invalid outgoing magnitude.
    """
    with pytest.raises(ValueError, match="must be positive"):
        TransferOut(metadata, AssetAmount(eth, quantity))


def test_transfer_out_has_no_sale_or_usd_fields(metadata: EventMetadata, eth: AssetId) -> None:
    """Preserve a token outflow without inventing proceeds or PnL.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    """
    outgoing = TransferOut(metadata, AssetAmount(eth, Decimal("0.5")))

    assert outgoing.amount.quantity == Decimal("0.5")
    assert [field.name for field in fields(outgoing)] == ["metadata", "amount"]


def test_transfer_in_keeps_known_quantity_without_basis(
    metadata: EventMetadata, link: AssetId
) -> None:
    """Preserve incoming quantity without assigning a fabricated cost basis.

    :param metadata: Source transaction reference.
    :param link: ERC-20 asset identity.
    """
    incoming = TransferIn(metadata, AssetAmount(link, Decimal("50")))

    assert incoming.amount.quantity == Decimal("50")
    assert [field.name for field in fields(incoming)] == ["metadata", "amount"]


def test_swap_keeps_both_sides_unvalued(
    metadata: EventMetadata, eth: AssetId, link: AssetId
) -> None:
    """Preserve asset quantities without estimating a USD trade value.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    :param link: Token asset identity.
    """
    event = Swap(
        metadata,
        amount_out=AssetAmount(eth, Decimal("0.5")),
        amount_in=AssetAmount(link, Decimal("50")),
    )

    assert event.amount_out.quantity == Decimal("0.5")
    assert event.amount_in.quantity == Decimal("50")
    assert [field.name for field in fields(event)] == ["metadata", "amount_out", "amount_in"]


def test_swap_rejects_same_asset(metadata: EventMetadata, eth: AssetId) -> None:
    """A swap must exchange distinct asset identities.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    """
    with pytest.raises(ValueError, match="different assets"):
        Swap(metadata, AssetAmount(eth, Decimal("1")), AssetAmount(eth, Decimal("2")))


def test_swap_rejects_cross_chain(metadata: EventMetadata, eth: AssetId) -> None:
    """Reject unsupported cross-chain exchanges at the event boundary.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    """
    other_chain_token = AssetId(10, "0x" + "cc" * 20)

    with pytest.raises(ValueError, match="same chain"):
        Swap(
            metadata,
            AssetAmount(eth, Decimal("1")),
            AssetAmount(other_chain_token, Decimal("50")),
        )


@pytest.mark.parametrize("quantity", [Decimal("0"), Decimal("-3")])
def test_swap_rejects_nonpositive_quantities(
    metadata: EventMetadata, eth: AssetId, link: AssetId, quantity: Decimal
) -> None:
    """Reject invalid magnitudes on either side of the trade.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    :param link: ERC-20 asset identity.
    :param quantity: Invalid test magnitude.
    """
    with pytest.raises(ValueError, match="must be positive"):
        Swap(metadata, AssetAmount(eth, quantity), AssetAmount(link, Decimal("2")))
    with pytest.raises(ValueError, match="must be positive"):
        Swap(metadata, AssetAmount(eth, Decimal("1")), AssetAmount(link, quantity))


def test_wrap_native_to_token_is_one_to_one(
    metadata: EventMetadata, eth: AssetId, weth: AssetId
) -> None:
    """Accept a native-to-wrapped conversion with preserved quantity.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    :param weth: Wrapped-token contract identity.
    """
    event = Wrap(metadata, AssetAmount(eth, Decimal("1.25")), AssetAmount(weth, Decimal("1.25")))
    assert event.amount_out.asset_id.is_native
    assert not event.amount_in.asset_id.is_native


def test_unwrap_token_to_native_is_one_to_one(
    metadata: EventMetadata, eth: AssetId, weth: AssetId
) -> None:
    """Accept a wrapped-to-native conversion with preserved quantity.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    :param weth: Wrapped-token contract identity.
    """
    event = Unwrap(metadata, AssetAmount(weth, Decimal("1")), AssetAmount(eth, Decimal("1")))
    assert event.amount_out.quantity == event.amount_in.quantity


@pytest.mark.parametrize("event_type", [Wrap, Unwrap])
def test_wrapping_rejects_mismatched_amounts(
    metadata: EventMetadata, eth: AssetId, weth: AssetId, event_type: type[Wrap] | type[Unwrap]
) -> None:
    """Reject non-1:1 native/wrapped conversions.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    :param weth: Wrapped-token contract identity.
    :param event_type: Wrap or Unwrap class being tested.
    """
    out_asset, in_asset = (eth, weth) if event_type is Wrap else (weth, eth)
    with pytest.raises(ValueError, match="one-to-one"):
        event_type(
            metadata,
            AssetAmount(out_asset, Decimal("1")),
            AssetAmount(in_asset, Decimal("0.9")),
        )


@pytest.mark.parametrize("event_type", [Wrap, Unwrap])
def test_wrapping_rejects_wrong_direction(
    metadata: EventMetadata, eth: AssetId, weth: AssetId, event_type: type[Wrap] | type[Unwrap]
) -> None:
    """Reject wrapper events that reverse their economic direction.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    :param weth: Token asset identity.
    :param event_type: Event class being tested.
    """
    out_asset, in_asset = (weth, eth) if event_type is Wrap else (eth, weth)
    with pytest.raises(ValueError, match="convert"):
        event_type(
            metadata,
            AssetAmount(out_asset, Decimal("1")),
            AssetAmount(in_asset, Decimal("1")),
        )


def test_wrapping_rejects_cross_chain(metadata: EventMetadata, eth: AssetId) -> None:
    """Reject wrapped conversions mixing unrelated chains.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    """
    token_on_another_chain = AssetId(10, "0x" + "dd" * 20)
    with pytest.raises(ValueError, match="same chain"):
        Wrap(
            metadata,
            AssetAmount(eth, Decimal("1")),
            AssetAmount(token_on_another_chain, Decimal("1")),
        )


def test_unknown_event_requires_reason(metadata: EventMetadata) -> None:
    """Avoid storing uninformative reasons for UNKNOWN classifications.

    :param metadata: Source transaction reference.
    """
    with pytest.raises(ValueError, match="must not be blank"):
        UnknownEvent(metadata, "  ")


def test_unknown_event_requires_immutable_tuple(metadata: EventMetadata, eth: AssetId) -> None:
    """Prevent mutation of known signed wallet deltas through a list.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    """
    with pytest.raises(TypeError, match="must be a tuple"):
        UnknownEvent(
            metadata,
            "unrecognized",
            cast(tuple[AssetAmount, ...], [AssetAmount(eth, Decimal("1"))]),
        )


def test_unknown_event_does_not_allow_bad_movements(metadata: EventMetadata) -> None:
    """Reject movements that do not have the AssetAmount contract.

    :param metadata: Source transaction reference.
    """
    with pytest.raises(TypeError, match="must contain AssetAmount"):
        UnknownEvent(metadata, "unrecognized", cast(tuple[AssetAmount, ...], ("not an amount",)))


def test_economic_events_are_immutable(metadata: EventMetadata, eth: AssetId) -> None:
    """Keep all event data immutable and replayable.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    """
    event = TransferIn(metadata, AssetAmount(eth, Decimal("1")))
    with pytest.raises(FrozenInstanceError):
        event.amount = AssetAmount(eth, Decimal("2"))


def test_event_type_alias_accepts_supported_events(metadata: EventMetadata, eth: AssetId) -> None:
    """Expose a shared annotation for application replay boundaries.

    :param metadata: Source transaction reference.
    :param eth: Native asset identity.
    """
    event: EconomicEvent = TransferOut(metadata, AssetAmount(eth, Decimal("1")))
    assert isinstance(event, TransferOut)


def test_unknown_event_preserves_signed_movements(
    metadata: EventMetadata, eth: AssetId, link: AssetId
) -> None:
    """Preserve observed wallet deltas without assigning financial meaning.

    :param metadata: Source transaction reference.
    :param eth: Native ETH asset identity.
    :param link: ERC-20 LINK asset identity.
    """
    outgoing_eth = AssetAmount(eth, Decimal("-0.2"))
    incoming_link = AssetAmount(link, Decimal("10"))
    movements = (outgoing_eth, incoming_link)

    event = UnknownEvent(
        metadata=metadata,
        reason="unrecognized operation",
        movements=movements,
    )

    assert event.metadata == metadata
    assert event.reason == "unrecognized operation"
    assert event.movements == movements
    assert event.movements[0].asset_id == eth
    assert event.movements[0].quantity == Decimal("-0.2")
    assert event.movements[1].asset_id == link
    assert event.movements[1].quantity == Decimal("10")

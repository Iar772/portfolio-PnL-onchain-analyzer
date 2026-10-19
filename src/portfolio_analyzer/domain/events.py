"""Deterministic economic events, independent of pricing and providers."""

from dataclasses import dataclass
from datetime import UTC, datetime

from portfolio_analyzer.domain.assets import AssetAmount


@dataclass(frozen=True, slots=True)
class EventMetadata:
    """Record the provenance and ordering information of an economic event.

    :param tx_hash: Nonblank source transaction identifier.
    :param block_number: Nonnegative block number containing the transaction.
    :param timestamp: Timezone-aware transaction timestamp, normalized to UTC.
    :raises TypeError: If any field has an invalid runtime type.
    :raises ValueError: If the identifier, block, or timestamp is invalid.
    """

    tx_hash: str
    block_number: int
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate metadata and store the timestamp in UTC.

        :raises TypeError: If metadata fields have invalid types.
        :raises ValueError: If a field violates the provenance contract.
        """
        if not isinstance(self.tx_hash, str):
            raise TypeError("EventMetadata.tx_hash must be a string")
        if not self.tx_hash.strip():
            raise ValueError("EventMetadata.tx_hash must not be blank")
        if isinstance(self.block_number, bool) or not isinstance(self.block_number, int):
            raise TypeError("EventMetadata.block_number must be an integer")
        if self.block_number < 0:
            raise ValueError("EventMetadata.block_number must not be negative")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("EventMetadata.timestamp must be a datetime")
        if self.timestamp.utcoffset() is None:
            raise ValueError("EventMetadata.timestamp must be timezone-aware")

        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))


def _require_metadata(metadata: EventMetadata) -> None:
    """Require a valid event metadata value object.

    :param metadata: Candidate event metadata object.
    :raises TypeError: If metadata is not EventMetadata.
    """
    if not isinstance(metadata, EventMetadata):
        raise TypeError("event metadata must be an EventMetadata")


def _require_positive_amount(amount: AssetAmount, *, field_name: str) -> None:
    """Require a strictly positive transfer or trade magnitude.

    :param amount: Candidate asset quantity.
    :param field_name: Field name used in error messages.
    :raises TypeError: If amount is not an AssetAmount.
    :raises ValueError: If the magnitude is zero or negative.
    """
    if not isinstance(amount, AssetAmount):
        raise TypeError(f"{field_name} must be an AssetAmount")
    if amount.quantity <= 0:
        raise ValueError(f"{field_name}.quantity must be positive")


def _require_same_chain(amount_out: AssetAmount, amount_in: AssetAmount) -> None:
    """Require both sides of a conversion to belong to the same chain.

    :param amount_out: Quantity given up by the wallet.
    :param amount_in: Quantity received by the wallet.
    :raises ValueError: If the two asset identifiers have different chain IDs.
    """
    if amount_out.asset_id.chain_id != amount_in.asset_id.chain_id:
        raise ValueError("Conversion assets must be on the same chain")


@dataclass(frozen=True, slots=True)
class TransferIn:
    """Record an incoming transfer without assuming its acquisition cost.

    :param metadata: Timestamp and transaction provenance.
    :param amount: Strictly positive received asset quantity.
    """

    metadata: EventMetadata
    amount: AssetAmount

    def __post_init__(self) -> None:
        """Enforce the positive received-quantity contract.

        :raises TypeError: If metadata or amount has an invalid type.
        :raises ValueError: If the received quantity is not positive.
        """
        _require_metadata(self.metadata)
        _require_positive_amount(self.amount, field_name="TransferIn.amount")


@dataclass(frozen=True, slots=True)
class TransferOut:
    """Record an outgoing transfer without assuming a taxable sale.

    :param metadata: Timestamp and transaction provenance.
    :param amount: Strictly positive sent asset quantity.
    """

    metadata: EventMetadata
    amount: AssetAmount

    def __post_init__(self) -> None:
        """Enforce the positive sent-quantity contract.

        :raises TypeError: If metadata or amount has an invalid type.
        :raises ValueError: If the sent quantity is not positive.
        """
        _require_metadata(self.metadata)
        _require_positive_amount(self.amount, field_name="TransferOut.amount")


@dataclass(frozen=True, slots=True)
class Swap:
    """Describe a two-sided asset exchange without assigning a USD value.

    :param metadata: Timestamp and transaction provenance.
    :param amount_out: Positive quantity of the asset given up by the wallet.
    :param amount_in: Positive quantity of the asset received by the wallet.
    :raises ValueError: If the asset pair is invalid.
    """

    metadata: EventMetadata
    amount_out: AssetAmount
    amount_in: AssetAmount

    def __post_init__(self) -> None:
        """Validate trade sides without determining proceeds or cost basis.

        :raises TypeError: If metadata or an amount has an invalid type.
        :raises ValueError: If quantities or the trade pair are invalid.
        """
        _require_metadata(self.metadata)
        _require_positive_amount(self.amount_out, field_name="Swap.amount_out")
        _require_positive_amount(self.amount_in, field_name="Swap.amount_in")
        _require_same_chain(self.amount_out, self.amount_in)
        if self.amount_in.asset_id == self.amount_out.asset_id:
            raise ValueError("different assets")


@dataclass(frozen=True, slots=True)
class Wrap:
    """Describe a basis-preserving native-asset to wrapped-token conversion.

    :param metadata: Timestamp and transaction provenance.
    :param amount_out: Native asset quantity given up by the wallet.
    :param amount_in: Wrapped ERC-20 asset quantity received by the wallet.
    """

    metadata: EventMetadata
    amount_out: AssetAmount
    amount_in: AssetAmount

    def __post_init__(self) -> None:
        """Enforce native-to-token wrapping with a one-to-one quantity ratio.

        :raises TypeError: If metadata or an amount has an invalid type.
        :raises ValueError: If the wrap directions, chain, or ratio are invalid.
        """
        _require_metadata(self.metadata)
        _require_positive_amount(self.amount_out, field_name="Wrap.amount_out")
        _require_positive_amount(self.amount_in, field_name="Wrap.amount_in")
        _require_same_chain(self.amount_out, self.amount_in)
        if not self.amount_out.asset_id.is_native or self.amount_in.asset_id.is_native:
            raise ValueError("Wrap must convert a native asset into a token")
        if self.amount_out.quantity != self.amount_in.quantity:
            raise ValueError("Wrap must preserve quantity one-to-one")


@dataclass(frozen=True, slots=True)
class Unwrap:
    """Describe a basis-preserving wrapped-token to native-asset conversion.

    :param metadata: Timestamp and transaction provenance.
    :param amount_out: Wrapped ERC-20 asset quantity given up by the wallet.
    :param amount_in: Native asset quantity received by the wallet.
    """

    metadata: EventMetadata
    amount_out: AssetAmount
    amount_in: AssetAmount

    def __post_init__(self) -> None:
        """Enforce token-to-native unwrapping with a one-to-one quantity ratio.

        :raises TypeError: If metadata or an amount has an invalid type.
        :raises ValueError: If the unwrap directions, chain, or ratio are invalid.
        """
        _require_metadata(self.metadata)
        _require_positive_amount(self.amount_out, field_name="Unwrap.amount_out")
        _require_positive_amount(self.amount_in, field_name="Unwrap.amount_in")
        _require_same_chain(self.amount_out, self.amount_in)
        if self.amount_out.asset_id.is_native or not self.amount_in.asset_id.is_native:
            raise ValueError("Unwrap must convert a token into a native asset")
        if self.amount_out.quantity != self.amount_in.quantity:
            raise ValueError("Unwrap must preserve quantity one-to-one")


@dataclass(frozen=True, slots=True)
class UnknownEvent:
    """Preserve an unclassified event without inventing financial meaning.

    Known signed wallet deltas may be retained without interpreting them as
    purchases, sales, or zero-cost acquisitions.

    :param metadata: Timestamp and transaction provenance.
    :param reason: Nonblank explanation of the classification uncertainty.
    :param movements: Known signed wallet deltas, if available.
    """

    metadata: EventMetadata
    reason: str
    movements: tuple[AssetAmount, ...] = ()

    def __post_init__(self) -> None:
        """Validate uncertainty details without valuing the movements.

        :raises TypeError: If metadata, reason, or movements have invalid types.
        :raises ValueError: If the reason is blank.
        """
        _require_metadata(self.metadata)
        if not isinstance(self.reason, str):
            raise TypeError("UnknownEvent.reason must be a string")
        if not self.reason.strip():
            raise ValueError("UnknownEvent.reason must not be blank")
        if not isinstance(self.movements, tuple):
            raise TypeError("UnknownEvent.movements must be a tuple")
        if any(not isinstance(movement, AssetAmount) for movement in self.movements):
            raise TypeError("UnknownEvent.movements must contain AssetAmount instances")


type EconomicEvent = TransferIn | TransferOut | Swap | Wrap | Unwrap | UnknownEvent

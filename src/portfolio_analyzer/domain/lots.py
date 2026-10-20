"""Immutable acquisition lots with explicit known or unknown USD cost basis."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Self

from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.money import UsdAmount


@dataclass(frozen=True, slots=True)
class Lot:
    """Represent an acquired asset quantity and its remaining FIFO inventory.

    Original fields preserve acquisition facts; remaining fields describe the
    current state during a future accounting replay. ``None`` means the USD
    cost basis is unknown, whereas ``UsdAmount(Decimal("0"))`` is known zero.

    :param asset_id: Canonical asset identity, not a token symbol.
    :param acquired_at: Timezone-aware acquisition timestamp.
    :param original_quantity: Strictly positive amount acquired in token units.
    :param remaining_quantity: Nonnegative amount not yet consumed by FIFO.
    :param original_cost_basis: Known nonnegative USD basis, or None if unknown.
    :param remaining_cost_basis: Remaining known USD basis, or None if unknown.
    :raises TypeError: If a field has the wrong runtime type.
    :raises ValueError: If a quantity, basis, or timestamp is inconsistent.
    """

    asset_id: AssetId
    acquired_at: datetime
    original_quantity: Decimal
    remaining_quantity: Decimal
    original_cost_basis: UsdAmount | None
    remaining_cost_basis: UsdAmount | None

    def __post_init__(self) -> None:
        """Validate a lot without rounding or inferring missing acquisition cost.

        :raises TypeError: If an identity, timestamp, quantity, or basis is invalid.
        :raises ValueError: If a lot's inventory or basis invariants are violated.
        """
        if not isinstance(self.asset_id, AssetId):
            raise TypeError("Lot.asset_id must be an AssetId")

        if not isinstance(self.acquired_at, datetime):
            raise TypeError("Lot.acquired_at must be a datetime")
        if self.acquired_at.utcoffset() is None:
            raise ValueError("Lot.acquired_at must be timezone-aware")
        object.__setattr__(self, "acquired_at", self.acquired_at.astimezone(UTC))

        for field_name in ("original_quantity", "remaining_quantity"):
            quantity = getattr(self, field_name)
            if not isinstance(quantity, Decimal):
                raise TypeError(f"Lot.{field_name} must be a Decimal")
            if not quantity.is_finite():
                raise ValueError(f"Lot.{field_name} must be finite")

        if self.original_quantity <= 0:
            raise ValueError("Lot.original_quantity must be positive")
        if self.remaining_quantity < 0:
            raise ValueError("Lot.remaining_quantity must not be negative")

        if self.remaining_quantity > self.original_quantity:
            raise ValueError("Lot.remaining_quantity cannot exceed original_quantity")

        for field_name in ("original_cost_basis", "remaining_cost_basis"):
            basis = getattr(self, field_name)
            if basis is not None and not isinstance(basis, UsdAmount):
                raise TypeError(f"Lot.{field_name} must be a UsdAmount or None")
            if basis is not None and basis.value < 0:
                raise ValueError(f"Lot.{field_name} must not be negative")

        if self.remaining_quantity > self.original_quantity:
            raise ValueError("remaining_quantity cannot exceed original_quantity")

        original_basis = self.original_cost_basis
        remaining_basis = self.remaining_cost_basis

        if (original_basis is None) != (remaining_basis is None):
            raise ValueError("cost basis must be both known or both unknown")

        if original_basis is not None and remaining_basis is not None:
            if remaining_basis.value > original_basis.value:
                raise ValueError("remaining_cost_basis cannot exceed original_cost_basis")

            if self.remaining_quantity == 0 and remaining_basis.value != 0:
                raise ValueError("exhausted lot must have zero remaining_cost_basis")

    @classmethod
    def open(
        cls,
        acquired: AssetAmount,
        *,
        cost_basis: UsdAmount | None,
        acquired_at: datetime,
    ) -> Self:
        """Open a new FIFO lot with all acquired inventory still available.

        :param acquired: Positive quantity received in an acquisition.
        :param cost_basis: Known USD acquisition cost, or None if unknown.
        :param acquired_at: Timezone-aware acquisition timestamp.
        :return: An immutable lot with original and remaining inventory equal.
        :raises TypeError: If acquired is not an AssetAmount.
        :raises ValueError: If the acquired quantity is not strictly positive.
        """
        if not isinstance(acquired, AssetAmount):
            raise TypeError("Lot.open acquired must be an AssetAmount")
        if acquired.quantity <= 0:
            raise ValueError("Lot.open acquired quantity must be positive")

        return cls(
            asset_id=acquired.asset_id,
            acquired_at=acquired_at,
            original_quantity=acquired.quantity,
            remaining_quantity=acquired.quantity,
            original_cost_basis=cost_basis,
            remaining_cost_basis=cost_basis,
        )

    @property
    def is_basis_known(self) -> bool:
        """Distinguish a known zero-cost acquisition from an unknown one.

        :return: True for a known USD cost basis, including explicit zero.
        """
        return self.original_cost_basis is not None

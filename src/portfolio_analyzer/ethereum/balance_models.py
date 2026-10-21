"""Vendor-independent ERC-20 balance results with explicit unresolved quantities."""

from dataclasses import dataclass
from typing import Literal

from portfolio_analyzer.domain.assets import AssetAmount, AssetId


@dataclass(frozen=True, slots=True)
class UnresolvedTokenBalance:
    """Preserve token identity when on-chain units cannot be safely normalized.

    :param asset_id: Canonical ERC-20 identity, independent of ticker.
    :param raw_units: Exact nonzero base units or None when lookup failed.
    :param reason: Safe classification of the missing information.
    :raises TypeError: If an input has an invalid type.
    :raises ValueError: If raw units or identity are not valid.
    """

    asset_id: AssetId
    raw_units: int | None
    reason: Literal["metadata_unavailable", "balance_unavailable"]

    def __post_init__(self) -> None:
        """Reject ambiguous or untrusted unresolved-balance records."""
        if not isinstance(self.asset_id, AssetId) or self.asset_id.is_native:
            raise ValueError("UnresolvedTokenBalance requires an ERC-20 AssetId")
        if self.raw_units is not None and (type(self.raw_units) is not int or self.raw_units <= 0):
            raise ValueError("raw_units must be positive or None")
        if self.reason not in ("metadata_unavailable", "balance_unavailable"):
            raise ValueError("Unsupported unresolved balance reason")
        if self.reason == "metadata_unavailable" and self.raw_units is None:
            raise ValueError("Missing metadata requires known nonzero raw units")
        if self.reason == "balance_unavailable" and self.raw_units is not None:
            raise ValueError("Unavailable balance cannot have trusted raw units")


@dataclass(frozen=True, slots=True)
class Erc20BalanceSnapshot:
    """Return exactly normalized tokens alongside explicitly unresolved assets.

    :param balances: Nonzero normalized ERC-20 positions with known decimals.
    :param unresolved: ERC-20 identities without trustworthy token quantities.
    :raises ValueError: If assets are duplicated or quantities are invalid.
    """

    balances: tuple[AssetAmount, ...]
    unresolved: tuple[UnresolvedTokenBalance, ...]

    def __post_init__(self) -> None:
        """Require unique assets and positive normalized quantities."""
        assets: set[AssetId] = set()
        for amount in self.balances:
            if not isinstance(amount, AssetAmount) or amount.asset_id.is_native:
                raise ValueError("Snapshot supports ERC-20 AssetAmount entries only")
            if amount.quantity <= 0 or amount.asset_id in assets:
                raise ValueError("Snapshot balances must be unique and positive")
            assets.add(amount.asset_id)
        for entry in self.unresolved:
            if not isinstance(entry, UnresolvedTokenBalance) or entry.asset_id in assets:
                raise ValueError("Snapshot unresolved balances must be unique")
            assets.add(entry.asset_id)

    @property
    def is_complete(self) -> bool:
        """Tell whether all returned token entries have normalized balances.

        :return: False when provider or metadata gaps remain.
        """
        return not self.unresolved

"""Immutable identity-based registry of trusted USD stablecoin candidates.

Registry membership does not imply a market price equal to the USD peg.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from portfolio_analyzer.domain.assets import AssetId


@dataclass(frozen=True, slots=True)
class UsdStablecoin:
    """Describe a verified USD-pegged ERC-20 without supplying a market quote.

    :param asset_id: Chain-specific contract identity, never the ticker symbol.
    :param symbol: Human-readable label for diagnostics and presentation only.
    :raises TypeError: If the identity or label has the wrong type.
    :raises ValueError: If the identity is native or the label is blank.
    """

    asset_id: AssetId
    symbol: str

    def __post_init__(self) -> None:
        """Reject invalid registry entries and normalize presentation text.

        :raises TypeError: If an entry field has the wrong type.
        :raises ValueError: If the entry is not an ERC-20 or has no label.
        """
        if not isinstance(self.asset_id, AssetId):
            raise TypeError("UsdStablecoin.asset_id must be an AssetId")
        if self.asset_id.is_native:
            raise ValueError("UsdStablecoin requires a token contract address")
        if not isinstance(self.symbol, str):
            raise TypeError("UsdStablecoin.symbol must be a string")
        label = self.symbol.strip()
        if not label:
            raise ValueError("UsdStablecoin.symbol must not be blank")
        object.__setattr__(self, "symbol", label)

    @property
    def peg_target_usd(self) -> Decimal:
        """Expose the nominal 1 USD peg target, NOT an observed market price.

        This value has no observation timestamp or provider provenance and must
        never be silently substituted for a missing historical/current quote.

        :return: The nominal USD target, without market-price provenance.
        """
        return Decimal("1")


@dataclass(frozen=True, slots=True, init=False)
class StablecoinRegistry:
    """Look up trusted USD stablecoin identities using chain and contract.

    Registry contents are defensively copied and read-only. Callers can build
    a new registry for other supported EVM networks without modifying this one.

    :param entries: Verified stablecoin descriptions.
    :raises TypeError: If an entry is not a UsdStablecoin.
    :raises ValueError: If a chain-specific asset identity is duplicated.
    """

    _by_asset_id: Mapping[AssetId, UsdStablecoin]

    def __init__(self, entries: Iterable[UsdStablecoin]) -> None:
        """Create an immutable lookup by AssetId, not by token symbol.

        :param entries: Trusted USD stablecoins to register.
        :raises TypeError: If an entry has the wrong type.
        :raises ValueError: If an asset is registered more than once.
        """
        by_asset_id: dict[AssetId, UsdStablecoin] = {}
        for entry in entries:
            if not isinstance(entry, UsdStablecoin):
                raise TypeError("registry entries must be UsdStablecoin instances")
            if entry.asset_id in by_asset_id:
                raise ValueError("duplicate stablecoin asset identity")
            by_asset_id[entry.asset_id] = entry
        object.__setattr__(self, "_by_asset_id", MappingProxyType(by_asset_id))

    def find(self, asset_id: AssetId) -> UsdStablecoin | None:
        """Find a trusted entry by exact chain and canonical contract address.

        :param asset_id: The network-qualified token identity to check.
        :return: The trusted entry, or None if the asset is not allowlisted.
        :raises TypeError: If asset_id is not an AssetId.
        """
        if not isinstance(asset_id, AssetId):
            raise TypeError("asset_id must be an AssetId")
        return self._by_asset_id.get(asset_id)

    def is_trusted(self, asset_id: AssetId) -> bool:
        """Test allowlist membership without consulting token metadata.

        :param asset_id: The network-qualified asset to inspect.
        :return: True only when this precise asset identity is registered.
        :raises TypeError: If asset_id is not an AssetId.
        """
        return self.find(asset_id) is not None

    def entries(self) -> tuple[UsdStablecoin, ...]:
        """Return stablecoin entries in their configured insertion order.

        :return: An immutable tuple containing the registered definitions.
        """
        return tuple(self._by_asset_id.values())

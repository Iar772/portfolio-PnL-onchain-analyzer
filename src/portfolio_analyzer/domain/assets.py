"""Canonical asset identities, token metadata, and asset quantities."""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Self


@dataclass(frozen=True, slots=True)
class AssetId:
    """Identify a native asset or an ERC-20 token independently of its symbol.

    :param chain_id: Positive EVM chain identifier; Ethereum Mainnet uses 1.
    :param contract_address: ERC-20 contract address, or None for the native asset.
    :raises TypeError: If the field types are invalid.
    :raises ValueError: If the chain ID or contract address is invalid.
    """

    chain_id: int
    contract_address: str | None

    def __post_init__(self) -> None:
        """Validate asset identity and canonicalize ERC-20 contract addresses.

        :raises TypeError: If an identity field has the wrong type.
        :raises ValueError: If a chain ID or address does not meet the contract.
        """
        if isinstance(self.chain_id, bool) or not isinstance(self.chain_id, int):
            raise TypeError("AssetId.chain_id must be an integer")
        if self.chain_id <= 0:
            raise ValueError("AssetId.chain_id must be positive")

        if self.contract_address is None:
            return
        if not isinstance(self.contract_address, str):
            raise TypeError("AssetId.contract_address must be a string or None")
        if re.fullmatch(r"0x[0-9a-fA-F]{40}", self.contract_address) is None:
            raise ValueError("AssetId.contract_address must be a 20-byte 0x-prefixed hex address")

        # Address casing is not part of Ethereum token identity.
        object.__setattr__(self, "contract_address", self.contract_address.lower())

    @property
    def is_native(self) -> bool:
        """Report whether the identity belongs to a chain-native asset.

        :return: True for the native asset; otherwise False for an ERC-20.
        """
        return self.contract_address is None


@dataclass(frozen=True, slots=True)
class AssetMetadata:
    """Describe display metadata without influencing asset identity.

    :param symbol: Non-empty display ticker, not a canonical identifier.
    :param name: Non-empty human-readable name.
    :param decimals: Number of decimal places, in the ERC-20 uint8 range.
    :raises TypeError: If fields have the wrong types.
    :raises ValueError: If strings are blank or decimals fall outside 0..255.
    """

    symbol: str
    name: str
    decimals: int

    def __post_init__(self) -> None:
        """Validate basic display metadata without silently modifying it.

        :raises TypeError: If a metadata field has an invalid type.
        :raises ValueError: If a string is blank or decimals are out of range.
        """
        for field_name in ("symbol", "name"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"AssetMetadata.{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"AssetMetadata.{field_name} must not be blank")

        if isinstance(self.decimals, bool) or not isinstance(self.decimals, int):
            raise TypeError("AssetMetadata.decimals must be an integer")
        if not 0 <= self.decimals <= 255:
            raise ValueError("AssetMetadata.decimals must be between 0 and 255")


@dataclass(frozen=True, slots=True)
class AssetAmount:
    """Associate a signed human-readable token quantity with an asset identity.

    Signed quantities allow wallet-relative movements (inflows and outflows).

    :param asset_id: Identity of the asset; never a symbol.
    :param quantity: Finite Decimal quantity in whole-token units.
    :raises TypeError: If an input has an invalid type.
    :raises ValueError: If quantity is NaN or infinite.
    """

    asset_id: AssetId
    quantity: Decimal

    def __post_init__(self) -> None:
        """Validate the asset identity and quantity without rounding.

        :raises TypeError: If the identity or quantity has the wrong type.
        :raises ValueError: If the quantity is not finite.
        """
        if not isinstance(self.asset_id, AssetId):
            raise TypeError("AssetAmount.asset_id must be an AssetId")
        if not isinstance(self.quantity, Decimal):
            raise TypeError("AssetAmount.quantity must be a Decimal")
        if not self.quantity.is_finite():
            raise ValueError("AssetAmount.quantity must be finite")

    @classmethod
    def from_raw_units(cls, asset_id: AssetId, raw_units: int, metadata: AssetMetadata) -> Self:
        """Convert signed integer base units to exact human-readable tokens.

        :param asset_id: Identity of the asset being measured.
        :param raw_units: Signed quantity in the smallest token units.
        :param metadata: Metadata providing the token's decimal precision.
        :return: An AssetAmount with no Decimal-context rounding.
        :raises TypeError: If raw_units is not an integer or metadata is invalid.
        """
        if isinstance(raw_units, bool) or not isinstance(raw_units, int):
            raise TypeError("AssetAmount.raw_units must be an integer")

        if not isinstance(metadata, AssetMetadata):
            raise TypeError("AssetAmount.metadata must be an AssetMetadata")

        quantity = Decimal(f"{raw_units}e-{metadata.decimals}")

        return cls(asset_id=asset_id, quantity=quantity)

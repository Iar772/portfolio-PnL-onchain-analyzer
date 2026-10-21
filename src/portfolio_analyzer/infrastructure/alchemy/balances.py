"""Alchemy Token API ERC-20 balances with complete pagination and exact decimals."""

import logging
import re
from dataclasses import dataclass

from portfolio_analyzer.application.provider_errors import (
    ProviderError,
    ProviderMalformedResponseError,
    ProviderPartialResultError,
)
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.ethereum.balance_models import Erc20BalanceSnapshot, UnresolvedTokenBalance
from portfolio_analyzer.infrastructure.alchemy.metadata import AlchemyTokenMetadataProvider
from portfolio_analyzer.infrastructure.alchemy.rpc import (
    AlchemyRpcProvider,
    _address,
    _field_address,
    _mapping,
)

_OPERATION = "get_erc20_balances"
_RAW_BALANCE_RE = re.compile(r"0x[0-9a-fA-F]+\Z")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RawBalance:
    """Store only trusted token identity and raw integer base units.

    :param asset_id: Canonical token contract identity.
    :param units: Nonzero units or None if a per-token lookup failed.
    """

    asset_id: AssetId
    units: int | None


def _parse_balance(item: object) -> _RawBalance:
    """Validate one vendor row without hiding conflicting zero amounts.

    :param item: Untrusted Alchemy token balance row.
    :return: Exact integer units including zero, or None for an error.
    :raises ProviderMalformedResponseError: If fields are invalid.
    """
    raw = _mapping(item, operation=_OPERATION)
    asset_id = AssetId(1, _field_address(raw.get("contractAddress"), operation=_OPERATION))
    # Alchemy can return an error for an individual token within an HTTP 200.
    # Do not treat that token as having a zero balance.
    if raw.get("error") is not None:
        return _RawBalance(asset_id, None)

    value = raw.get("tokenBalance")
    if not isinstance(value, str) or _RAW_BALANCE_RE.fullmatch(value) is None:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    units = int(value, 16)
    if units >= 2**256:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    return _RawBalance(asset_id, units)


class AlchemyErc20BalanceProvider:
    """Read current ERC-20 balances without losing unknown nonzero positions.

    This is an ERC-20 specific reader; the future full BalanceProvider will
    compose this reader with native ETH from RpcProvider. The Alchemy token API
    does not provide a fixed-block snapshot selector; subsequent reconciliation
    must not assume an atomic snapshot across paginated requests.

    :param rpc: Shared Alchemy transport handling timeout and bounded retries.
    :param metadata: Existing metadata adapter providing token decimals.
    :param page_size: Token API page size from 1 to 100.
    :param max_pages: Strict upper page budget to avoid infinite cursor loops.
    """

    def __init__(
        self,
        rpc: AlchemyRpcProvider,
        metadata: AlchemyTokenMetadataProvider,
        *,
        page_size: int = 100,
        max_pages: int = 10000,
    ) -> None:
        """Validate dependencies without creating another HTTP session.

        :param rpc: Existing shared RPC transport.
        :param metadata: Existing token metadata provider.
        :param page_size: Maximum 100 entries per token-balance API page.
        :param max_pages: Positive maximum number of cursor pages.
        :raises TypeError: For invalid dependency types.
        :raises ValueError: For invalid page settings.
        """
        if not isinstance(rpc, AlchemyRpcProvider):
            raise TypeError("rpc must be AlchemyRpcProvider")
        if not isinstance(metadata, AlchemyTokenMetadataProvider):
            raise TypeError("metadata must be AlchemyTokenMetadataProvider")
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("page_size must be within 1..100")
        if type(max_pages) is not int or max_pages < 1:
            raise ValueError("max_pages must be positive")
        self._rpc = rpc
        self._metadata = metadata
        self._page_size = page_size
        self._max_pages = max_pages

    async def _get_raw_balances(self, address: str) -> dict[AssetId, _RawBalance]:
        """Exhaust the complete Alchemy cursor before using balance data.

        :param address: Canonical wallet address.
        :return: Unique raw ERC-20 balances, including zero or per-token errors.
        :raises ProviderMalformedResponseError: For invalid initial responses.
        :raises ProviderPartialResultError: If a later page is missing/conflicting.
        """
        by_asset: dict[AssetId, _RawBalance] = {}
        seen_keys: set[str] = set()
        page_key: str | None = None
        for page_index in range(self._max_pages):
            options: dict[str, object] = {"maxCount": self._page_size}
            if page_key is not None:
                options["pageKey"] = page_key
            try:
                result = await self._rpc.request_alchemy_method(
                    "alchemy_getTokenBalances",
                    [address, "erc20", options],
                    operation=_OPERATION,
                )
                page = _mapping(result, operation=_OPERATION)
                if _field_address(page.get("address"), operation=_OPERATION) != address:
                    raise ProviderMalformedResponseError("alchemy", _OPERATION)
                rows = page.get("tokenBalances")
                if not isinstance(rows, list):
                    raise ProviderMalformedResponseError("alchemy", _OPERATION)
                for item in rows:
                    entry = _parse_balance(item)
                    existing = by_asset.get(entry.asset_id)
                    if existing is not None and existing != entry:
                        raise ProviderPartialResultError("alchemy", _OPERATION)
                    by_asset[entry.asset_id] = entry
                next_key = page.get("pageKey")
                if next_key is not None:
                    if not isinstance(next_key, str) or not next_key.strip():
                        raise ProviderMalformedResponseError("alchemy", _OPERATION)
                    if next_key in seen_keys:
                        raise ProviderPartialResultError("alchemy", _OPERATION)
                    seen_keys.add(next_key)
            except ProviderError as error:
                if page_index > 0 and not isinstance(error, ProviderPartialResultError):
                    _LOGGER.warning(
                        "erc20_balances_incomplete",
                        extra={
                            "provider": "alchemy",
                            "operation": _OPERATION,
                            "page_count": page_index,
                        },
                    )
                    raise ProviderPartialResultError("alchemy", _OPERATION) from None
                raise
            if next_key is None:
                _LOGGER.info(
                    "erc20_balance_pages_completed",
                    extra={
                        "provider": "alchemy",
                        "operation": _OPERATION,
                        "page_count": page_index + 1,
                    },
                )
                return by_asset
            page_key = next_key
        _LOGGER.warning(
            "erc20_balances_incomplete",
            extra={"provider": "alchemy", "operation": _OPERATION, "page_count": self._max_pages},
        )
        raise ProviderPartialResultError("alchemy", _OPERATION)

    async def get_erc20_balances(self, address: str) -> Erc20BalanceSnapshot:
        """Get normalized nonzero balances and explicitly unresolved token rows.

        Metadata is requested *only* for confirmed nonzero balances. Missing
        metadata preserves exact raw units with a stable AssetId. Missing balance
        data is also represented explicitly, never interpreted as zero.

        :param address: Wallet address to query on Ethereum Mainnet.
        :return: Typed snapshot with normalized and unresolved ERC-20 entries.
        :raises ValueError: If the wallet address is invalid.
        :raises ProviderError: For provider, metadata or pagination failures.
        """
        canonical_address = _address(address)
        raw = await self._get_raw_balances(canonical_address)
        balances: list[AssetAmount] = []
        unresolved: list[UnresolvedTokenBalance] = []

        # Deliberately sequential: no task explosion, and one shared RPC
        # transport still enforces concurrency and retry budgets.
        for asset_id, item in raw.items():
            if item.units == 0:
                continue
            if item.units is None:
                unresolved.append(UnresolvedTokenBalance(asset_id, None, "balance_unavailable"))
                continue
            metadata = await self._metadata.get_metadata(asset_id)
            if metadata is None:
                unresolved.append(
                    UnresolvedTokenBalance(asset_id, item.units, "metadata_unavailable")
                )
                continue
            balances.append(AssetAmount.from_raw_units(asset_id, item.units, metadata))

        snapshot = Erc20BalanceSnapshot(tuple(balances), tuple(unresolved))
        _LOGGER.info(
            "erc20_balances_completed",
            extra={
                "provider": "alchemy",
                "operation": _OPERATION,
                "token_count": len(snapshot.balances),
                "unresolved_count": len(snapshot.unresolved),
            },
        )
        return snapshot

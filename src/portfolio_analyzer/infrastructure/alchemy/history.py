"""Alchemy Transfers API adapter with strict complete-history guarantees."""

import logging
import re
from datetime import datetime
from typing import Any

from portfolio_analyzer.application.provider_errors import (
    ProviderError,
    ProviderMalformedResponseError,
    ProviderPartialResultError,
)
from portfolio_analyzer.application.run_guardrails import AnalysisRunStopped
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.infrastructure.alchemy.rpc import (
    AlchemyRpcProvider,
    _address,
    _field_address,
    _field_hash,
    _mapping,
    _quantity,
)

_OPERATION = "get_history"
_CATEGORIES = ["external", "internal", "erc20"]
_RAW_VALUE_RE = re.compile(r"0x[0-9a-fA-F]+\Z")
_LOGGER = logging.getLogger(__name__)


def _raw_units(value: object) -> int:
    """Decode exact base units, allowing padded ERC-20 hex values.

    Unlike JSON-RPC QUANTITY fields, token raw values can be zero-padded.

    :param value: Hex representation of on-chain base units.
    :return: Nonnegative integer token quantity.
    :raises ProviderMalformedResponseError: If no exact raw value exists.
    """
    if not isinstance(value, str) or _RAW_VALUE_RE.fullmatch(value) is None:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    return int(value, 16)


def _parse_transfer(raw: object) -> AssetMovement:
    """Convert a vendor transfer object into one trusted internal movement.

    :param raw: Transfer JSON from Alchemy's Transfers API.
    :return: Typed Ethereum movement with exact smallest-unit quantity.
    :raises ProviderMalformedResponseError: When required facts are missing.
    """
    item = _mapping(raw, operation=_OPERATION)
    unique_id = item.get("uniqueId")
    if not isinstance(unique_id, str) or not unique_id.strip():
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    category_raw = item.get("category")
    try:
        category = MovementCategory(category_raw)
    except (ValueError, TypeError):
        raise ProviderMalformedResponseError("alchemy", _OPERATION) from None
    contract = _mapping(item.get("rawContract"), operation=_OPERATION)
    if category is MovementCategory.ERC20:
        token_address = _field_address(contract.get("address"), operation=_OPERATION)
        asset_id = AssetId(1, token_address)
    else:
        if contract.get("address") is not None:
            raise ProviderMalformedResponseError("alchemy", _OPERATION)
        asset_id = AssetId(1, None)

    raw_amount = _raw_units(contract.get("value"))
    hint = contract.get("decimal")
    decimals_hint = None if hint is None else _quantity(hint, operation=_OPERATION)
    if decimals_hint is not None and decimals_hint > 255:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    block_number = _quantity(item.get("blockNum"), operation=_OPERATION)
    metadata = _mapping(item.get("metadata"), operation=_OPERATION)
    raw_timestamp = metadata.get("blockTimestamp")
    if not isinstance(raw_timestamp, str):
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    try:
        timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError:
        raise ProviderMalformedResponseError("alchemy", _OPERATION) from None
    if timestamp.utcoffset() is None:
        raise ProviderMalformedResponseError("alchemy", _OPERATION)
    try:
        return AssetMovement(
            transfer_id=unique_id,
            tx_hash=_field_hash(item.get("hash"), operation=_OPERATION),
            block_number=block_number,
            timestamp=timestamp,
            from_address=_field_address(item.get("from"), operation=_OPERATION),
            to_address=_field_address(item.get("to"), operation=_OPERATION),
            asset_id=asset_id,
            raw_amount=raw_amount,
            category=category,
            decimals_hint=decimals_hint,
        )
    except (TypeError, ValueError):
        raise ProviderMalformedResponseError("alchemy", _OPERATION) from None


def _merge_directions(
    outgoing: list[AssetMovement], incoming: list[AssetMovement]
) -> list[AssetMovement]:
    """Merge both directions without losing or double-counting movements.

    Distinct movements must be preserved. A transfer addressed to the same
    wallet may appear twice with the same provenance ID. Incompatible facts
    under that ID are an error, not a reason to arbitrarily choose one.

    :param outgoing: Complete movements where the wallet is sender.
    :param incoming: Complete movements where the wallet is recipient.
    :return: Unique movements in stable ascending block order.
    :raises ProviderPartialResultError: If duplicate identities disagree.
    """
    movements_by_id: dict[str, AssetMovement] = {}

    for movement in (*outgoing, *incoming):
        previous = movements_by_id.get(movement.transfer_id)
        if previous is not None and previous != movement:
            raise ProviderPartialResultError("alchemy", _OPERATION)
        movements_by_id[movement.transfer_id] = movement

    return sorted(movements_by_id.values(), key=lambda movement: movement.block_number)


class AlchemyHistoryProvider:
    """Implement HistoryProvider by exhausting both transfer directions.

    Uses the existing RpcProvider transport for shared timeout/retry/backoff,
    HTTP session and bounded requests. Requests run sequentially so pagination
    cursors cannot expire while queued behind unbounded concurrent work.

    :param rpc: Alchemy RPC adapter with the shared HTTP session.
    :param page_size: Number of transfers requested per page, at most 1000.
    :param max_pages_per_direction: Safety bound; fail instead of truncating.
    """

    def __init__(
        self,
        rpc: AlchemyRpcProvider,
        *,
        page_size: int = 1000,
        max_pages_per_direction: int = 10000,
    ) -> None:
        """Validate page limits and store shared transport.

        :param rpc: Already configured, caller-owned Alchemy RPC adapter.
        :param page_size: Maximum entries per page in [1, 1000].
        :param max_pages_per_direction: Upper bound for pages per direction.
        :raises ValueError: For invalid pagination limits.
        :raises TypeError: For an incompatible RPC adapter.
        """
        if not isinstance(rpc, AlchemyRpcProvider):
            raise TypeError("rpc must be an AlchemyRpcProvider")
        if type(page_size) is not int or not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        if type(max_pages_per_direction) is not int or max_pages_per_direction < 1:
            raise ValueError("max_pages_per_direction must be positive")
        self._rpc = rpc
        self._page_size = page_size
        self._max_pages_per_direction = max_pages_per_direction

    async def _get_direction(
        self, address: str, *, direction: str, through_block: int, prior_completed: bool
    ) -> list[AssetMovement]:
        """Exhaust a single cursor chain, rejecting interrupted pagination.

        :param address: Canonical wallet address.
        :param direction: Either incoming or outgoing query side.
        :param through_block: Frozen inclusive block upper bound.
        :param prior_completed: Whether the opposite direction has completed.
        :return: All validated movements for that direction.
        :raises ProviderPartialResultError: When a page chain is incomplete.
        """
        params: dict[str, Any] = {
            "fromBlock": "0x0",
            "toBlock": hex(through_block),
            "category": _CATEGORIES,
            "excludeZeroValue": True,
            "withMetadata": True,
            "order": "asc",
            "maxCount": hex(self._page_size),
            ("fromAddress" if direction == "outgoing" else "toAddress"): address,
        }
        seen_cursors: set[str] = set()
        results: list[AssetMovement] = []
        for page_num in range(self._max_pages_per_direction):
            try:
                raw = await self._rpc.request_alchemy_method(
                    "alchemy_getAssetTransfers", [params.copy()], operation=_OPERATION
                )
                page = _mapping(raw, operation=_OPERATION)
                transfers = page.get("transfers")
                if not isinstance(transfers, list):
                    raise ProviderMalformedResponseError("alchemy", _OPERATION)
                parsed = [_parse_transfer(item) for item in transfers]
                for movement in parsed:
                    if movement.block_number > through_block:
                        raise ProviderMalformedResponseError("alchemy", _OPERATION)
                    if direction == "outgoing" and movement.from_address != address:
                        raise ProviderMalformedResponseError("alchemy", _OPERATION)
                    if direction == "incoming" and movement.to_address != address:
                        raise ProviderMalformedResponseError("alchemy", _OPERATION)
                cursor = page.get("pageKey")
                if cursor is not None and not isinstance(cursor, str):
                    raise ProviderMalformedResponseError("alchemy", _OPERATION)
                if cursor and not parsed:
                    raise ProviderPartialResultError("alchemy", _OPERATION)
                results.extend(parsed)
                if self._rpc.run_guardrails is not None:
                    self._rpc.run_guardrails.record_progress(
                        stage=f"history_{direction}",
                        pages_completed=page_num + 1,
                        items_seen=len(results),
                    )
                if not cursor:
                    _LOGGER.info(
                        "history_direction_completed",
                        extra={
                            "provider": "alchemy",
                            "operation": _OPERATION,
                            "direction": direction,
                            "page_count": page_num + 1,
                            "movement_count": len(results),
                        },
                    )
                    return results
                if cursor in seen_cursors or page_num + 1 >= self._max_pages_per_direction:
                    raise ProviderPartialResultError("alchemy", _OPERATION)
                seen_cursors.add(cursor)
                params["pageKey"] = cursor
            except AnalysisRunStopped:
                _LOGGER.warning(
                    "history_incomplete",
                    extra={
                        "provider": "alchemy",
                        "operation": _OPERATION,
                        "direction": direction,
                        "page_count": page_num,
                        "movement_count": len(results),
                    },
                )
                raise
            except ProviderError as exc:
                _LOGGER.warning(
                    "history_incomplete",
                    extra={
                        "provider": "alchemy",
                        "operation": _OPERATION,
                        "direction": direction,
                        "page_count": page_num + 1,
                        "movement_count": len(results),
                    },
                )
                if isinstance(exc, ProviderPartialResultError):
                    raise
                if page_num == 0 and not prior_completed:
                    raise
                raise ProviderPartialResultError("alchemy", _OPERATION) from None
        _LOGGER.warning(
            "history_incomplete",
            extra={
                "provider": "alchemy",
                "operation": _OPERATION,
                "direction": direction,
                "page_count": self._max_pages_per_direction,
                "movement_count": len(results),
            },
        )
        raise ProviderPartialResultError("alchemy", _OPERATION)

    async def get_history(self, address: str) -> list[AssetMovement]:
        """Load all incoming and outgoing native/ERC-20 movements for one block snapshot.

        Self-directed transfers appear in both API directions; identical transfer
        IDs are deduplicated, while contradictory duplicates fail closed.

        :param address: Ethereum wallet address.
        :return: Complete provider-neutral movement list for the snapshot.
        :raises ProviderPartialResultError: If any direction or page is incomplete.
        """
        wallet = _address(address)
        snapshot = await self._rpc.get_block_number()
        out = await self._get_direction(
            wallet, direction="outgoing", through_block=snapshot, prior_completed=False
        )
        incoming = await self._get_direction(
            wallet, direction="incoming", through_block=snapshot, prior_completed=True
        )
        try:
            movements = _merge_directions(out, incoming)
        except ProviderPartialResultError:
            _LOGGER.warning(
                "history_incomplete",
                extra={"provider": "alchemy", "operation": _OPERATION},
            )
            raise
        _LOGGER.info(
            "history_completed",
            extra={
                "provider": "alchemy",
                "operation": _OPERATION,
                "movement_count": len(movements),
            },
        )
        return movements

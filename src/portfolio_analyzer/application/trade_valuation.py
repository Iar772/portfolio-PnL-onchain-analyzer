"""Conservative swap USD valuation from an explicitly trusted stablecoin leg.

A single observed USD value is reused for both disposal proceeds and acquired
basis. This module does not create FIFO lots or recognize taxable disposals.
"""

from dataclasses import dataclass
from datetime import timedelta
from decimal import Context, localcontext
from enum import StrEnum
from typing import Literal

from portfolio_analyzer.application.ports import PriceProvider
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.events import Swap
from portfolio_analyzer.domain.money import UsdAmount
from portfolio_analyzer.domain.stablecoins import StablecoinRegistry


class SwapValuationStatus(StrEnum):
    """Explain why a swap does or does not have a known USD valuation."""

    VALUED = "valued"
    NO_STABLECOIN_ANCHOR = "no_stablecoin_anchor"
    PRICE_UNAVAILABLE = "price_unavailable"
    PRICE_OUTSIDE_WINDOW = "price_outside_window"


@dataclass(frozen=True, slots=True)
class SwapUsdValuation:
    """Keep one trade USD value for two accounting uses.

    :param swap: Original economic event, without modifying its meaning.
    :param anchor_asset_id: Trusted asset chosen as the quote anchor, if any.
    :param anchor_side: Whether that asset was outgoing or incoming.
    :param quote: Observed historical market quote, if provided.
    :param value_usd: One shared USD amount, or None when unknown.
    :param status: Explicit valuation outcome (never unknown as zero).
    :raises TypeError: If field types are inconsistent.
    :raises ValueError: If a known/unknown valuation is contradictory.
    """

    swap: Swap
    anchor_asset_id: AssetId | None
    anchor_side: Literal["out", "in"] | None
    quote: PriceQuote | None
    value_usd: UsdAmount | None
    status: SwapValuationStatus

    def __post_init__(self) -> None:
        """Protect the accounting result from contradictory construction.

        :raises TypeError: For wrong domain types.
        :raises ValueError: For inconsistent provenance or status.
        """
        if not isinstance(self.swap, Swap):
            raise TypeError("swap must be a Swap")
        if not isinstance(self.status, SwapValuationStatus):
            raise TypeError("status must be SwapValuationStatus")
        if self.anchor_asset_id is not None and not isinstance(self.anchor_asset_id, AssetId):
            raise TypeError("anchor_asset_id must be AssetId or None")
        if self.quote is not None and not isinstance(self.quote, PriceQuote):
            raise TypeError("quote must be PriceQuote or None")
        if self.value_usd is not None and not isinstance(self.value_usd, UsdAmount):
            raise TypeError("value_usd must be UsdAmount or None")
        if self.anchor_side not in ("out", "in", None):
            raise ValueError("anchor_side must be 'out', 'in' or None")
        if self.anchor_side is None:
            if self.anchor_asset_id is not None:
                raise ValueError("missing anchor_side with anchor_asset_id")
        else:
            selected_asset_id = (
                self.swap.amount_out.asset_id
                if self.anchor_side == "out"
                else self.swap.amount_in.asset_id
            )
            if self.anchor_asset_id != selected_asset_id:
                raise ValueError("anchor identity must match its actual swap leg")
        if self.status is SwapValuationStatus.VALUED:
            if self.quote is None or self.value_usd is None or self.anchor_side is None:
                raise ValueError("valued swaps require anchor, quote and value")
            if self.value_usd.value < 0:
                raise ValueError("trade USD value must be nonnegative")
            selected = self.swap.amount_out if self.anchor_side == "out" else self.swap.amount_in
            if self.value_usd != _usd_value(selected, self.quote):
                raise ValueError("value_usd must match the selected leg and its quote")
        elif self.value_usd is not None:
            raise ValueError("unknown swaps must not have a USD value")
        if self.status is SwapValuationStatus.NO_STABLECOIN_ANCHOR:
            if (
                self.anchor_asset_id is not None
                or self.anchor_side is not None
                or self.quote is not None
            ):
                raise ValueError("unanchored swaps cannot contain anchor provenance")
        elif self.anchor_side is None or self.anchor_asset_id is None:
            raise ValueError("anchored statuses require a selected asset")
        if self.status is SwapValuationStatus.PRICE_UNAVAILABLE and self.quote is not None:
            raise ValueError("unavailable prices cannot have a quote")
        if self.status is SwapValuationStatus.PRICE_OUTSIDE_WINDOW and self.quote is None:
            raise ValueError("out-of-window status requires an observed quote")

    @property
    def disposal_proceeds(self) -> UsdAmount | None:
        """Return the same USD valuation used for acquired basis.

        :return: Known disposal proceeds, or None when valuation is unknown.
        """
        return self.value_usd

    @property
    def acquired_basis(self) -> UsdAmount | None:
        """Return the identical USD valuation used for disposal proceeds.

        :return: Known acquired basis, or None when valuation is unknown.
        """
        return self.value_usd

    @property
    def is_known(self) -> bool:
        """Report whether the trade has one observed USD value.

        :return: True only when valued from a timely historical quote.
        """
        return self.status is SwapValuationStatus.VALUED


def select_stablecoin_side(swap: Swap, registry: StablecoinRegistry) -> Literal["out", "in"] | None:
    """Select a trusted USD stablecoin leg by contract identity only.

    Outgoing is preferred when both legs are trusted stablecoins. This avoids
    two separate valuations for one economic trade.

    :param swap: Classified two-asset economic swap.
    :param registry: Trusted identity-based USD stablecoin registry.
    :return: Anchor side, or None when neither leg is allowlisted.
    :raises TypeError: For invalid domain model inputs.
    """
    if not isinstance(swap, Swap):
        raise TypeError("swap must be a Swap")
    if not isinstance(registry, StablecoinRegistry):
        raise TypeError("registry must be a StablecoinRegistry")
    if registry.is_trusted(swap.amount_out.asset_id):
        return "out"
    if registry.is_trusted(swap.amount_in.asset_id):
        return "in"
    return None


def _require_quote_window(max_distance: timedelta) -> None:
    """Validate the maximum acceptable historical observation drift.

    :param max_distance: Allowed absolute distance from the swap timestamp.
    :raises TypeError: If the limit is not a timedelta.
    :raises ValueError: If nonpositive or exceeds the 1-hour lookup policy.
    """
    if not isinstance(max_distance, timedelta):
        raise TypeError("max_distance must be a timedelta")
    if not timedelta(0) < max_distance <= timedelta(hours=1):
        raise ValueError("max_distance must be positive and at most one hour")


def _usd_value(amount: AssetAmount, quote: PriceQuote) -> UsdAmount:
    """Multiply an exact asset quantity by an observed USD quote without rounding.

    :param amount: Known token quantity selected as the anchor leg.
    :param quote: Historical USD observation for the same asset.
    :return: Exactly multiplied USD amount.
    """
    required_precision = max(
        80,
        len(amount.quantity.as_tuple().digits) + len(quote.price_usd.value.as_tuple().digits) + 2,
    )
    with localcontext(Context(prec=required_precision)):
        return UsdAmount(amount.quantity * quote.price_usd.value)


def calculate_stablecoin_swap_valuation(
    swap: Swap,
    registry: StablecoinRegistry,
    quote: PriceQuote | None,
    *,
    max_distance: timedelta = timedelta(minutes=15),
) -> SwapUsdValuation:
    """Calculate one deterministic USD value from a verified stablecoin leg.

    The registry's nominal $1 peg is never used as an observed market price.
    This function is synchronous and performs no network I/O.

    :param swap: Classified transaction event.
    :param registry: Trusted contract allowlist.
    :param quote: Historical USD quote for the selected stablecoin, or None.
    :param max_distance: Maximum quote-to-transaction time difference.
    :return: Known proceeds/basis together, or an explicit unknown outcome.
    :raises TypeError: For invalid types.
    :raises ValueError: For invalid quote window.
    """
    _require_quote_window(max_distance)
    side = select_stablecoin_side(swap, registry)
    if quote is not None and not isinstance(quote, PriceQuote):
        raise TypeError("quote must be PriceQuote or None")
    if side is None:
        return SwapUsdValuation(
            swap, None, None, None, None, SwapValuationStatus.NO_STABLECOIN_ANCHOR
        )

    amount: AssetAmount = swap.amount_out if side == "out" else swap.amount_in
    if quote is None:
        return SwapUsdValuation(
            swap, amount.asset_id, side, None, None, SwapValuationStatus.PRICE_UNAVAILABLE
        )
    if abs(quote.observed_at - swap.metadata.timestamp) > max_distance:
        return SwapUsdValuation(
            swap, amount.asset_id, side, quote, None, SwapValuationStatus.PRICE_OUTSIDE_WINDOW
        )

    value = _usd_value(amount, quote)
    return SwapUsdValuation(swap, amount.asset_id, side, quote, value, SwapValuationStatus.VALUED)


async def value_swap_by_stablecoin(
    swap: Swap,
    registry: StablecoinRegistry,
    provider: PriceProvider,
    *,
    max_distance: timedelta = timedelta(minutes=15),
) -> SwapUsdValuation:
    """Fetch one historical stablecoin quote then value one swap consistently.

    This application orchestration deliberately uses one request and one USD
    result. Provider errors propagate, never masquerading as unknown prices.

    :param swap: Classified two-leg economic swap.
    :param registry: Allowlist of contract identities.
    :param provider: Vendor-independent historical USD price provider.
    :param max_distance: Maximum quote-to-transaction time difference.
    :return: Shared valuation, or explicit unknown without fabricated prices.
    :raises ProviderError: When the historical-price provider fails.
    :raises TypeError: For invalid model inputs or provider results.
    :raises ValueError: For invalid observation window.
    """
    _require_quote_window(max_distance)
    side = select_stablecoin_side(swap, registry)
    if side is None:
        return calculate_stablecoin_swap_valuation(swap, registry, None, max_distance=max_distance)
    asset_id = swap.amount_out.asset_id if side == "out" else swap.amount_in.asset_id
    quote = await provider.get_historical_price(asset_id, swap.metadata.timestamp)
    return calculate_stablecoin_swap_valuation(swap, registry, quote, max_distance=max_distance)

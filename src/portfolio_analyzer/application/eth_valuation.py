"""Historical ETH-side swap valuation, including canonical WETH9 as an ETH proxy.

ETH and WETH remain distinct AssetId values. WETH price attribution uses
native Ethereum ETH/USD as the observation source, never an arbitrary ticker.
"""

from datetime import timedelta
from typing import Literal

from portfolio_analyzer.application.ports import PriceProvider
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.trade_valuation import (
    SwapUsdValuation,
    SwapValuationStatus,
    calculate_swap_valuation_for_side,
    validate_quote_window,
)
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.events import Swap
from portfolio_analyzer.ethereum.known_contracts import MAINNET_WETH9

MAINNET_ETH = AssetId(chain_id=1, contract_address=None)
_ETH_ANCHORS = frozenset((MAINNET_ETH, MAINNET_WETH9))


def select_eth_side(swap: Swap) -> Literal["out", "in"] | None:
    """Find a canonical Mainnet ETH/WETH leg in an actual two-asset swap.

    ETH-to-WETH or WETH-to-ETH conversion is intentionally rejected: that
    operation belongs to Wrap/Unwrap and must preserve basis without a sale.

    :param swap: Classified two-leg Ethereum swap.
    :return: Outgoing or incoming side, or None when neither is ETH/WETH.
    :raises TypeError: If input is not a Swap.
    :raises ValueError: If a direct ETH/WETH conversion is passed as a swap.
    """
    if not isinstance(swap, Swap):
        raise TypeError("swap must be a Swap")

    is_out = swap.amount_out.asset_id in _ETH_ANCHORS
    is_in = swap.amount_in.asset_id in _ETH_ANCHORS
    if is_out and is_in:
        raise ValueError("ETH/WETH conversion must be Wrap/Unwrap, not a valued Swap")
    if is_out:
        return "out"
    if is_in:
        return "in"
    return None


def calculate_eth_swap_valuation(
    swap: Swap,
    quote: PriceQuote | None,
    *,
    max_distance: timedelta = timedelta(minutes=15),
) -> SwapUsdValuation:
    """Calculate one transaction-time USD trade value using historical ETH/USD.

    For a WETH leg, use the WETH quantity with a native ETH quote; record
    native ETH explicitly as pricing_asset_id in the shared valuation model.

    :param swap: Previously classified simple two-leg swap.
    :param quote: Historical ETH/USD quote for the timestamp, or None.
    :param max_distance: Maximum quote-to-transaction timestamp difference.
    :return: Shared trade proceeds/basis, or explicit unknown valuation.
    :raises TypeError: For wrong model, quote, or window types.
    :raises ValueError: For direct conversion or invalid quote window.
    """
    side = select_eth_side(swap)
    return calculate_swap_valuation_for_side(
        swap,
        side,
        quote,
        missing_anchor_status=SwapValuationStatus.NO_ETH_ANCHOR,
        max_distance=max_distance,
        pricing_asset_id=MAINNET_ETH if side is not None else None,
    )


async def value_swap_by_eth(
    swap: Swap,
    provider: PriceProvider,
    *,
    max_distance: timedelta = timedelta(minutes=15),
) -> SwapUsdValuation:
    """Fetch historical native ETH/USD and value a simple ETH-anchored swap.

    Provider lookup is intentionally skipped for swaps without an ETH/WETH
    leg. Upstream provider errors are not converted into missing prices.

    :param swap: Classified economic swap with Ethereum asset IDs.
    :param provider: Vendor-neutral historical price provider.
    :param max_distance: Maximum accepted historical observation age.
    :return: One consistent USD value or an explicit unknown status.
    :raises ProviderError: If the price provider reports transport/API failure.
    :raises TypeError: For invalid inputs or provider return types.
    :raises ValueError: For direct ETH/WETH conversion or invalid window.
    """
    validate_quote_window(max_distance)
    side = select_eth_side(swap)
    if side is None:
        return calculate_eth_swap_valuation(swap, None, max_distance=max_distance)

    historical_quote = await provider.get_historical_price(MAINNET_ETH, swap.metadata.timestamp)
    return calculate_eth_swap_valuation(swap, historical_quote, max_distance=max_distance)

"""Conservative historical USD valuation for two volatile ERC-20 swap legs.

Both observed USD values must be timely and positive. Acceptance requires an
explicit evidence-backed divergence threshold; there is no default threshold.
The outgoing leg sets the single accounting value only after cross-checking
against the incoming leg; disagreements are explicit unknown valuations.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Context, Decimal, localcontext

from portfolio_analyzer.application.ports import PriceProvider
from portfolio_analyzer.application.price_divergence import (
    DivergenceDecision,
    PriceDivergencePolicy,
)
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.trade_valuation import (
    SwapUsdValuation,
    SwapValuationStatus,
    _usd_value,
    calculate_swap_valuation_for_side,
    validate_quote_window,
)
from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.events import Swap
from portfolio_analyzer.domain.money import UsdAmount
from portfolio_analyzer.domain.stablecoins import StablecoinRegistry
from portfolio_analyzer.ethereum.known_contracts import MAINNET_WETH9


@dataclass(frozen=True, slots=True)
class VolatilePriceComparison:
    """Preserve independent market observations and one shared accounting result.

    :param valuation: The existing USD valuation model used by accounting.
    :param outgoing_quote: Historical quote of the sold token, if available.
    :param incoming_quote: Historical quote of the received token, if available.
    :param outgoing_value_usd: Observed USD value of the outgoing token, if valid.
    :param incoming_value_usd: Observed USD value of the incoming token, if valid.
    :param relative_spread: Relative USD disagreement, if both values are positive.
    :param divergence_policy: The exact calibration gate used to decide the outcome.
    :raises TypeError: If quotation or value evidence has incorrect types.
    :raises ValueError: If evidence is internally inconsistent.
    """

    valuation: SwapUsdValuation
    outgoing_quote: PriceQuote | None
    incoming_quote: PriceQuote | None
    outgoing_value_usd: UsdAmount | None
    incoming_value_usd: UsdAmount | None
    relative_spread: Decimal | None
    divergence_policy: PriceDivergencePolicy = field(default_factory=PriceDivergencePolicy)

    def __post_init__(self) -> None:
        """Ensure a transparent comparison cannot contradict its valuation.

        :raises TypeError: For incorrect evidence types.
        :raises ValueError: For contradictory quote/value pairs or status.
        """
        if not isinstance(self.divergence_policy, PriceDivergencePolicy):
            raise TypeError("divergence_policy must be PriceDivergencePolicy")
        if not isinstance(self.valuation, SwapUsdValuation):
            raise TypeError("valuation must be SwapUsdValuation")
        for quote in (self.outgoing_quote, self.incoming_quote):
            if quote is not None and not isinstance(quote, PriceQuote):
                raise TypeError("comparison quotes must be PriceQuote or None")
        for value in (self.outgoing_value_usd, self.incoming_value_usd):
            if value is not None and not isinstance(value, UsdAmount):
                raise TypeError("comparison values must be UsdAmount or None")
        if self.outgoing_value_usd is not None and self.outgoing_quote is None:
            raise ValueError("outgoing value requires its quote")
        if self.incoming_value_usd is not None and self.incoming_quote is None:
            raise ValueError("incoming value requires its quote")
        if self.valuation.status is SwapValuationStatus.NO_VOLATILE_PAIR and any(
            evidence is not None
            for evidence in (
                self.outgoing_quote,
                self.incoming_quote,
                self.outgoing_value_usd,
                self.incoming_value_usd,
                self.relative_spread,
            )
        ):
            raise ValueError("unsupported volatile pairs cannot contain priced evidence")
        swap = self.valuation.swap
        if self.outgoing_quote is not None and self.outgoing_value_usd != _usd_value(
            swap.amount_out, self.outgoing_quote
        ):
            raise ValueError("outgoing value must match outgoing quantity and quote")
        if self.incoming_quote is not None and self.incoming_value_usd != _usd_value(
            swap.amount_in, self.incoming_quote
        ):
            raise ValueError("incoming value must match incoming quantity and quote")
        if self.relative_spread is not None:
            if not isinstance(self.relative_spread, Decimal):
                raise TypeError("relative_spread must be Decimal or None")
            if not self.relative_spread.is_finite() or self.relative_spread < 0:
                raise ValueError("relative_spread must be finite and nonnegative")
            if self.outgoing_value_usd is None or self.incoming_value_usd is None:
                raise ValueError("spread requires two observed USD values")
            if self.outgoing_value_usd.value <= 0 or self.incoming_value_usd.value <= 0:
                raise ValueError("spread requires positive observed USD values")
            if self.relative_spread != _relative_spread(
                self.outgoing_value_usd, self.incoming_value_usd
            ):
                raise ValueError("spread must match observed USD values")
        if self.valuation.is_known:
            if self.valuation.quote != self.outgoing_quote:
                raise ValueError("accounting quote must be the outgoing quote")
            if self.valuation.anchor_side != "out":
                raise ValueError("volatile accounting must use the outgoing anchor")
            if self.valuation.value_usd != self.outgoing_value_usd:
                raise ValueError("accepted USD value must equal outgoing market value")
            if self.incoming_value_usd is None or self.relative_spread is None:
                raise ValueError("accepted swap must have a complete comparison")
            if (
                self.divergence_policy.evaluate(self.relative_spread)
                is not DivergenceDecision.ACCEPTED
            ):
                raise ValueError("accepted swap requires calibrated, matching divergence policy")
        if self.valuation.status in (
            SwapValuationStatus.PRICE_DIVERGENCE_UNCALIBRATED,
            SwapValuationStatus.VOLATILE_PRICE_DISAGREEMENT,
        ):
            if self.relative_spread is None:
                raise ValueError("divergence status requires a computed spread")
            expected_decision = (
                DivergenceDecision.UNCALIBRATED
                if self.valuation.status is SwapValuationStatus.PRICE_DIVERGENCE_UNCALIBRATED
                else DivergenceDecision.REJECTED
            )
            if self.divergence_policy.evaluate(self.relative_spread) is not expected_decision:
                raise ValueError("divergence status contradicts its stored policy")

    @property
    def disposal_proceeds(self) -> UsdAmount | None:
        """Expose the same accepted USD amount used by FIFO disposal.

        :return: Accepted proceeds or None when the market comparison failed.
        """
        return self.valuation.disposal_proceeds

    @property
    def acquired_basis(self) -> UsdAmount | None:
        """Expose the identical accepted USD amount as acquisition basis.

        :return: Accepted acquisition basis or None when unknown.
        """
        return self.valuation.acquired_basis


def is_volatile_erc20_pair(swap: Swap, registry: StablecoinRegistry) -> bool:
    """Check chain-qualified ERC-20 identities without relying on ticker labels.

    :param swap: One classified two-asset swap.
    :param registry: Trusted stablecoins that must use their own anchor policy.
    :return: True only for two Ethereum Mainnet non-stablecoin non-WETH tokens.
    :raises TypeError: For invalid swap or registry inputs.
    """
    if not isinstance(swap, Swap):
        raise TypeError("swap must be a Swap")
    if not isinstance(registry, StablecoinRegistry):
        raise TypeError("registry must be StablecoinRegistry")
    assets: tuple[AssetId, AssetId] = (
        swap.amount_out.asset_id,
        swap.amount_in.asset_id,
    )
    return all(
        asset.chain_id == 1
        and not asset.is_native
        and asset != MAINNET_WETH9
        and not registry.is_trusted(asset)
        for asset in assets
    )


def _unknown(
    swap: Swap,
    status: SwapValuationStatus,
) -> SwapUsdValuation:
    """Return unknown value without pretending the trade had zero USD value.

    :param swap: Original economic swap.
    :param status: Explanation of why price evidence was rejected.
    :return: Shared unvalued accounting result.
    """
    if status is SwapValuationStatus.NO_VOLATILE_PAIR:
        return SwapUsdValuation(swap, None, None, None, None, status)
    return SwapUsdValuation(
        swap,
        swap.amount_out.asset_id,
        "out",
        None,
        None,
        status,
        pricing_asset_id=swap.amount_out.asset_id,
    )


def _relative_spread(outgoing: UsdAmount, incoming: UsdAmount) -> Decimal:
    """Compare exact nonzero USD leg valuations with extended precision.

    :param outgoing: Total market value of tokens sent.
    :param incoming: Total market value of tokens received.
    :return: Absolute difference divided by the larger value.
    """
    digits = max(len(outgoing.value.as_tuple().digits), len(incoming.value.as_tuple().digits))
    with localcontext(Context(prec=max(80, digits * 2 + 20))):
        return abs(outgoing.value - incoming.value) / max(outgoing.value, incoming.value)


def calculate_volatile_swap_valuation(
    swap: Swap,
    registry: StablecoinRegistry,
    outgoing_quote: PriceQuote | None,
    incoming_quote: PriceQuote | None,
    *,
    max_distance: timedelta = timedelta(minutes=15),
    divergence_policy: PriceDivergencePolicy | None = None,
) -> VolatilePriceComparison:
    """Cross-check both volatile ERC-20 legs and accept one common USD value.

    Two individually priced sides are not automatically reliable: stale, zero,
    divergent or as-yet uncalibrated observations are returned as unknown.
    For accepted trades the outgoing USD value provides BOTH proceeds and
    acquired basis, with independent incoming evidence retained for audit.

    :param swap: Two-sided classified Ethereum swap.
    :param registry: Stablecoins excluded from volatile-pair valuation.
    :param outgoing_quote: Observed USD price of disposed ERC-20 or None.
    :param incoming_quote: Observed USD price of acquired ERC-20 or None.
    :param max_distance: Allowed distance from transaction timestamp.
    :param divergence_policy: Evidence-backed tolerance, or None for unknown valuation.
    :return: Observations plus the existing common accounting valuation.
    :raises TypeError: For invalid input or quote types.
    :raises ValueError: For invalid comparison policy.
    """
    validate_quote_window(max_distance)
    if divergence_policy is not None and not isinstance(divergence_policy, PriceDivergencePolicy):
        raise TypeError("divergence_policy must be PriceDivergencePolicy or None")
    policy = divergence_policy or PriceDivergencePolicy()
    eligible = is_volatile_erc20_pair(swap, registry)
    for quote in (outgoing_quote, incoming_quote):
        if quote is not None and not isinstance(quote, PriceQuote):
            raise TypeError("historical quotes must be PriceQuote or None")
    if not eligible:
        return VolatilePriceComparison(
            _unknown(swap, SwapValuationStatus.NO_VOLATILE_PAIR),
            None,
            None,
            None,
            None,
            None,
            policy,
        )

    outgoing_usd = (
        _usd_value(swap.amount_out, outgoing_quote) if outgoing_quote is not None else None
    )
    incoming_usd = (
        _usd_value(swap.amount_in, incoming_quote) if incoming_quote is not None else None
    )
    is_missing = outgoing_quote is None or incoming_quote is None
    outside_window = any(
        abs(quote.observed_at - swap.metadata.timestamp) > max_distance
        for quote in (outgoing_quote, incoming_quote)
        if quote is not None
    )
    is_nonpositive = any(
        value is not None and value.value <= 0 for value in (outgoing_usd, incoming_usd)
    )
    spread: Decimal | None = None
    if outgoing_usd is not None and incoming_usd is not None and not is_nonpositive:
        spread = _relative_spread(outgoing_usd, incoming_usd)

    if is_missing:
        status = SwapValuationStatus.PRICE_UNAVAILABLE
    elif outside_window or is_nonpositive:
        status = SwapValuationStatus.VOLATILE_PRICE_UNRELIABLE
    elif spread is not None:
        decision = policy.evaluate(spread)
        if decision is DivergenceDecision.UNCALIBRATED:
            status = SwapValuationStatus.PRICE_DIVERGENCE_UNCALIBRATED
        elif decision is DivergenceDecision.REJECTED:
            status = SwapValuationStatus.VOLATILE_PRICE_DISAGREEMENT
        else:
            status = SwapValuationStatus.VALUED
    else:
        status = SwapValuationStatus.VOLATILE_PRICE_UNRELIABLE

    valuation = (
        calculate_swap_valuation_for_side(
            swap,
            "out",
            outgoing_quote,
            missing_anchor_status=SwapValuationStatus.NO_VOLATILE_PAIR,
            max_distance=max_distance,
            pricing_asset_id=swap.amount_out.asset_id,
        )
        if status is SwapValuationStatus.VALUED
        else _unknown(swap, status)
    )
    return VolatilePriceComparison(
        valuation,
        outgoing_quote,
        incoming_quote,
        outgoing_usd,
        incoming_usd,
        spread,
        policy,
    )


async def value_swap_by_volatile(
    swap: Swap,
    registry: StablecoinRegistry,
    provider: PriceProvider,
    *,
    max_distance: timedelta = timedelta(minutes=15),
    divergence_policy: PriceDivergencePolicy | None = None,
) -> VolatilePriceComparison:
    """Fetch both historical ERC-20 quotes at the transaction timestamp.

    Price queries are sequential and bounded, never unbounded gather. Transport
    and provider failures propagate rather than becoming absent market prices.

    :param swap: Classified simple two-sided swap.
    :param registry: Trusted stablecoin contracts excluded from this policy.
    :param provider: Async vendor-neutral historical USD prices.
    :param max_distance: Allowed distance from the transaction timestamp.
    :param divergence_policy: Evidence-backed tolerance, or None for unknown valuation.
    :return: Independent evidence plus a shared USD accounting valuation.
    :raises ProviderError: On provider failure (no silent unknown masking).
    :raises TypeError: On invalid input or returned quote types.
    :raises ValueError: For invalid valuation policy.
    """
    validate_quote_window(max_distance)
    if divergence_policy is not None and not isinstance(divergence_policy, PriceDivergencePolicy):
        raise TypeError("divergence_policy must be PriceDivergencePolicy or None")
    policy = divergence_policy or PriceDivergencePolicy()
    if not is_volatile_erc20_pair(swap, registry):
        return calculate_volatile_swap_valuation(
            swap,
            registry,
            None,
            None,
            max_distance=max_distance,
            divergence_policy=policy,
        )
    outgoing_quote = await provider.get_historical_price(
        swap.amount_out.asset_id,
        swap.metadata.timestamp,
    )
    if outgoing_quote is not None and not isinstance(outgoing_quote, PriceQuote):
        raise TypeError("PriceProvider returned an invalid outgoing quote")
    incoming_quote = await provider.get_historical_price(
        swap.amount_in.asset_id,
        swap.metadata.timestamp,
    )
    if incoming_quote is not None and not isinstance(incoming_quote, PriceQuote):
        raise TypeError("PriceProvider returned an invalid incoming quote")
    return calculate_volatile_swap_valuation(
        swap,
        registry,
        outgoing_quote,
        incoming_quote,
        max_distance=max_distance,
        divergence_policy=policy,
    )

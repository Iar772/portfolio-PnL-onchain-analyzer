"""Deterministic, conservative USD valuation of known token quantities."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext

from portfolio_analyzer.application.ports import CurrentPriceProvider
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.money import UsdAmount
from portfolio_analyzer.ethereum.balance_models import UnresolvedTokenBalance


@dataclass(frozen=True, slots=True)
class PricedAsset:
    """Represent a valued current asset with an observed USD quote.

    :param balance: Exact known quantity from chain data.
    :param quote: Timestamped trusted USD price.
    :param value_usd: Exact calculated known USD value.
    """

    balance: AssetAmount
    quote: PriceQuote
    value_usd: UsdAmount


@dataclass(frozen=True, slots=True)
class UnpricedAsset:
    """Preserve a known quantity whose market price is unavailable.

    :param balance: Known balance excluded from the known USD total.
    """

    balance: AssetAmount


@dataclass(frozen=True, slots=True)
class CurrentPortfolioValuation:
    """Distinguish priced assets from incomplete portfolio valuation.

    A known USD total is not a claim of the complete portfolio value when
    any asset is unpriced or its amount is unresolved.

    :param priced: Current holdings with known USD quotes.
    :param unpriced: Current holdings without trustworthy USD quotes.
    :param unresolved_balances: Raw balances without trustworthy token amounts.
    :param known_total_usd: Sum of only the priced holdings.
    """

    priced: tuple[PricedAsset, ...]
    unpriced: tuple[UnpricedAsset, ...]
    unresolved_balances: tuple[UnresolvedTokenBalance, ...]
    known_total_usd: UsdAmount

    @property
    def is_complete(self) -> bool:
        """Report whether all supplied assets could be valued.

        :return: False if either missing prices or unresolved quantities remain.
        """
        return not self.unpriced and not self.unresolved_balances


def value_current_balances(
    balances: Iterable[AssetAmount],
    quotes: Mapping[AssetId, PriceQuote | None],
    *,
    unresolved: Iterable[UnresolvedTokenBalance] = (),
) -> CurrentPortfolioValuation:
    """Value holdings without ever counting unknown price as a zero price.

    The computation is pure; it does no I/O and never rounds through float.

    :param balances: Known normalized nonnegative asset balances.
    :param quotes: Current quotes keyed by chain and contract, not symbol.
    :param unresolved: Unknown amounts from ERC-20 balances, if any.
    :return: Explicit partial-or-complete USD valuation.
    :raises ValueError: For duplicate asset identities or negative amounts.
    :raises TypeError: For invalid quotes or input types.
    """
    if not isinstance(quotes, Mapping):
        raise TypeError("quotes must be a mapping")
    priced_inputs: list[tuple[AssetAmount, PriceQuote]] = []
    missing: list[UnpricedAsset] = []
    seen: set[AssetId] = set()
    for balance in balances:
        if not isinstance(balance, AssetAmount):
            raise TypeError("balances must contain AssetAmount")
        if balance.quantity < 0 or balance.asset_id in seen:
            raise ValueError("portfolio balances must be unique and nonnegative")
        seen.add(balance.asset_id)
        if balance.quantity == 0:
            continue
        quote = quotes.get(balance.asset_id)
        if quote is None:
            missing.append(UnpricedAsset(balance))
        elif not isinstance(quote, PriceQuote):
            raise TypeError("known quotes must be PriceQuote instances")
        else:
            priced_inputs.append((balance, quote))
    unresolved_entries = tuple(unresolved)
    for item in unresolved_entries:
        if not isinstance(item, UnresolvedTokenBalance):
            raise TypeError("unresolved must contain UnresolvedTokenBalance")
        if item.asset_id in seen:
            raise ValueError("asset must not be both known and unresolved")
        seen.add(item.asset_id)

    # Decimal's default context can silently discard tiny amounts when a
    # portfolio combines extreme magnitudes (1e100 USD + 1e-100 USD). Bound
    # required precision by the complete exponent span of every product,
    # including room for carry when summing multiple assets.
    if priced_inputs:
        high = max(
            balance.quantity.adjusted() + quote.price_usd.value.adjusted() + 1
            for balance, quote in priced_inputs
        )
        low = min(
            int(balance.quantity.as_tuple().exponent)
            + int(quote.price_usd.value.as_tuple().exponent)
            for balance, quote in priced_inputs
        )
        precision = max(80, high - low + len(str(len(priced_inputs))) + 4)
    else:
        precision = 80
    with localcontext(Context(prec=precision)):
        priced = tuple(
            PricedAsset(balance, quote, UsdAmount(balance.quantity * quote.price_usd.value))
            for balance, quote in priced_inputs
        )
        known_total = UsdAmount(sum((item.value_usd.value for item in priced), Decimal(0)))
    return CurrentPortfolioValuation(
        priced=priced,
        unpriced=tuple(missing),
        unresolved_balances=unresolved_entries,
        known_total_usd=known_total,
    )


async def price_current_balances(
    balances: Iterable[AssetAmount],
    provider: CurrentPriceProvider,
    *,
    unresolved: Iterable[UnresolvedTokenBalance] = (),
) -> CurrentPortfolioValuation:
    """Obtain current prices sequentially without unbounded gather.

    :param balances: Current normalized holdings.
    :param provider: Replaceable current quote interface.
    :param unresolved: Unknown on-chain quantities not priced in this step.
    :return: Conservative valuation with explicit unpriced and unresolved assets.
    :raises ProviderError: If a provider request fails rather than returning None.
    """
    holdings = tuple(balances)
    quotes: dict[AssetId, PriceQuote | None] = {}
    for holding in holdings:
        if not isinstance(holding, AssetAmount):
            raise TypeError("balances must contain AssetAmount")
        if holding.quantity <= 0 or holding.asset_id in quotes:
            continue
        quotes[holding.asset_id] = await provider.get_current_price(holding.asset_id)
    return value_current_balances(holdings, quotes, unresolved=unresolved)

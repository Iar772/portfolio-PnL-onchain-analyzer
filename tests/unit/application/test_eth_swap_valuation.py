"""Canonical Ethereum ETH/WETH historical USD valuation and provenance tests."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from portfolio_analyzer.application.eth_valuation import (
    MAINNET_ETH,
    calculate_eth_swap_valuation,
    select_eth_side,
    value_swap_by_eth,
)
from portfolio_analyzer.application.provider_errors import ProviderTimeoutError
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.trade_valuation import (
    SwapUsdValuation,
    SwapValuationStatus,
    calculate_stablecoin_swap_valuation,
)
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.events import EventMetadata, Swap
from portfolio_analyzer.domain.money import UsdAmount, UsdPrice
from portfolio_analyzer.ethereum.known_contracts import MAINNET_WETH9
from portfolio_analyzer.ethereum.stablecoins import MAINNET_USD_STABLECOINS, MAINNET_USDC

TX_TIME = datetime(2024, 3, 11, 12, 0, tzinfo=UTC)
LINK = AssetId(1, "0x" + "aa" * 20)
LDO = AssetId(1, "0x" + "bb" * 20)
FAKE_WETH = AssetId(1, "0x" + "cc" * 20)


def make_swap(
    out: AssetId = MAINNET_ETH,
    incoming: AssetId = LINK,
    out_qty: str = "0.5",
    in_qty: str = "50",
) -> Swap:
    """Construct deterministic two-leg swap with exact quantities."""
    return Swap(
        metadata=EventMetadata("0x" + "aa" * 32, 20_000_000, TX_TIME),
        amount_out=AssetAmount(out, Decimal(out_qty)),
        amount_in=AssetAmount(incoming, Decimal(in_qty)),
    )


def make_quote(
    value: str = "3000",
    timestamp: datetime = TX_TIME,
) -> PriceQuote:
    """Construct observed historical quote with explicit provenance."""
    return PriceQuote(UsdPrice(Decimal(value)), timestamp, "historical_eth_test")


class FakePrices:
    """Record calls to vendor-independent historical provider contract."""

    def __init__(self, result: PriceQuote | None) -> None:
        """Store one fake quote, including legitimate absent price."""
        self.result = result
        self.history_calls: list[tuple[AssetId, datetime]] = []
        self.current_calls = 0

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Return configured quote while capturing lookup identity and time."""
        self.history_calls.append((asset_id, at))
        return self.result

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Reject current lookup during historical trade valuation."""
        self.current_calls += 1
        raise AssertionError("Only historical quote can value historical trade")


@pytest.mark.parametrize(
    ("outgoing", "incoming", "side"),
    [
        (MAINNET_ETH, LINK, "out"),
        (LINK, MAINNET_ETH, "in"),
        (MAINNET_WETH9, LINK, "out"),
        (LINK, MAINNET_WETH9, "in"),
        (LDO, LINK, None),
        (FAKE_WETH, LINK, None),
        (AssetId(8453, None), AssetId(8453, "0x" + "aa" * 20), None),
        (AssetId(8453, MAINNET_WETH9.contract_address), AssetId(8453, "0x" + "aa" * 20), None),
    ],
)
def test_selects_only_trusted_mainnet_assets(
    outgoing: AssetId,
    incoming: AssetId,
    side: str | None,
) -> None:
    """Token labels and contracts on different networks cannot become ETH anchors."""
    assert select_eth_side(make_swap(outgoing, incoming)) == side


@pytest.mark.parametrize("outgoing", [MAINNET_ETH, MAINNET_WETH9])
@pytest.mark.parametrize("incoming", [MAINNET_ETH, MAINNET_WETH9])
def test_eth_weth_conversion_must_use_basis_preserving_events(
    outgoing: AssetId,
    incoming: AssetId,
) -> None:
    """Protect actual ETH/WETH conversion against accidental taxable valuation."""
    if outgoing == incoming:
        with pytest.raises(ValueError):
            make_swap(outgoing, incoming)
    else:
        with pytest.raises(ValueError, match="Wrap/Unwrap"):
            select_eth_side(make_swap(outgoing, incoming))


@pytest.mark.parametrize(
    ("outgoing", "incoming", "out_qty", "in_qty", "expected", "side"),
    [
        (MAINNET_ETH, LINK, "0.5", "50", "1500.0", "out"),
        (LINK, MAINNET_ETH, "50", "0.5", "1500.0", "in"),
        (MAINNET_WETH9, LINK, "2.1", "80", "6300.0", "out"),
        (LINK, MAINNET_WETH9, "80", "2.1", "6300.0", "in"),
        (MAINNET_ETH, MAINNET_USDC, "1", "3000", "3000", "out"),
        (MAINNET_WETH9, LDO, "0.000000000000000001", "2", "0.000000000000003000", "out"),
    ],
)
def test_both_trade_legs_share_one_observed_usd_value(
    outgoing: AssetId,
    incoming: AssetId,
    out_qty: str,
    in_qty: str,
    expected: str,
    side: str,
) -> None:
    """Asset distinction survives proxy pricing and one value funds proceeds and basis."""
    swap = make_swap(outgoing, incoming, out_qty, in_qty)
    observation = make_quote()
    result = calculate_eth_swap_valuation(swap, observation)
    assert isinstance(result, SwapUsdValuation)
    assert result.swap is swap
    assert result.quote is observation
    assert result.status is SwapValuationStatus.VALUED
    assert result.anchor_side == side
    assert result.anchor_asset_id == (outgoing if side == "out" else incoming)
    assert result.pricing_asset_id == MAINNET_ETH
    assert result.disposal_proceeds == UsdAmount(Decimal(expected))
    assert result.acquired_basis is result.disposal_proceeds
    assert result.is_known


@pytest.mark.parametrize("price", ["0", "2800.12345678", "1e-20"])
def test_explicit_known_price_including_zero_is_not_missing(price: str) -> None:
    """Known zero differs from no historical observation."""
    result = calculate_eth_swap_valuation(make_swap(), make_quote(price))
    assert result.is_known
    with localcontext() as context:
        context.prec = 90
        assert result.value_usd == UsdAmount(Decimal("0.5") * Decimal(price))


@pytest.mark.parametrize("outgoing", [MAINNET_ETH, MAINNET_WETH9])
def test_missing_price_returns_unknown_not_zero(outgoing: AssetId) -> None:
    """No quote must not create fake proceeds or a fake new cost basis."""
    result = calculate_eth_swap_valuation(make_swap(outgoing), None)
    assert result.status is SwapValuationStatus.PRICE_UNAVAILABLE
    assert result.quote is None
    assert result.value_usd is None
    assert result.disposal_proceeds is None
    assert result.acquired_basis is None
    assert result.anchor_asset_id == outgoing
    assert result.pricing_asset_id == MAINNET_ETH
    assert not result.is_known


@pytest.mark.parametrize(
    ("drift", "status"),
    [
        (timedelta(0), SwapValuationStatus.VALUED),
        (timedelta(minutes=-15), SwapValuationStatus.VALUED),
        (timedelta(minutes=15), SwapValuationStatus.VALUED),
        (timedelta(minutes=15, microseconds=1), SwapValuationStatus.PRICE_OUTSIDE_WINDOW),
        (timedelta(minutes=-30), SwapValuationStatus.PRICE_OUTSIDE_WINDOW),
        (timedelta(days=10), SwapValuationStatus.PRICE_OUTSIDE_WINDOW),
    ],
)
def test_observed_timestamp_window_is_honored(
    drift: timedelta,
    status: SwapValuationStatus,
) -> None:
    """Never overwrite observed_at or accept an observation beyond the policy window."""
    observation = make_quote(timestamp=TX_TIME + drift)
    outcome = calculate_eth_swap_valuation(make_swap(), observation)
    assert outcome.status is status
    assert outcome.quote is observation
    assert observation.observed_at == TX_TIME + drift
    if status is SwapValuationStatus.PRICE_OUTSIDE_WINDOW:
        assert outcome.value_usd is None
        assert not outcome.is_known


def test_offset_aware_quote_is_normalized_and_compared_in_utc() -> None:
    """Offsets must not change whether an observed price is sufficiently close."""
    shifted = datetime(2024, 3, 11, 15, 10, tzinfo=timezone(timedelta(hours=3)))
    result = calculate_eth_swap_valuation(make_swap(), make_quote(timestamp=shifted))
    assert result.status is SwapValuationStatus.VALUED
    assert result.quote is not None
    assert result.quote.observed_at == TX_TIME + timedelta(minutes=10)


def test_unanchored_swap_never_uses_unrelated_quote() -> None:
    """An injected quote cannot accidentally price an LDO/LINK trade."""
    result = calculate_eth_swap_valuation(make_swap(LDO, LINK), make_quote())
    assert result.status is SwapValuationStatus.NO_ETH_ANCHOR
    assert result.anchor_asset_id is None
    assert result.pricing_asset_id is None
    assert result.quote is None
    assert result.value_usd is None


@pytest.mark.parametrize("window", [timedelta(0), timedelta(seconds=-1), timedelta(hours=2)])
def test_invalid_distance_is_rejected(window: timedelta) -> None:
    """Bad policy must be rejected instead of causing unknown valuation."""
    with pytest.raises(ValueError):
        calculate_eth_swap_valuation(make_swap(), make_quote(), max_distance=window)


def test_invalid_quote_type_rejected() -> None:
    """Wrong response model must not get interpreted as missing price."""
    with pytest.raises(TypeError):
        calculate_eth_swap_valuation(make_swap(), object())  # type: ignore[arg-type]


def test_wrong_window_type_rejected() -> None:
    """Window must always be a timedelta, not an integer or string."""
    with pytest.raises(TypeError):
        calculate_eth_swap_valuation(make_swap(), None, max_distance="15m")  # type: ignore[arg-type]


def test_invalid_swap_type_rejected() -> None:
    """Direct valuation input must be a domain Swap."""
    with pytest.raises(TypeError):
        select_eth_side(object())  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", [MAINNET_ETH, MAINNET_WETH9])
async def test_async_provider_always_queries_native_eth_by_transaction_time(
    anchor: AssetId,
) -> None:
    """WETH proxy valuation must query ETH, not WETH contract or a ticker string."""
    provider = FakePrices(make_quote("2500"))
    outcome = await value_swap_by_eth(make_swap(anchor), provider)
    assert outcome.value_usd == UsdAmount(Decimal("1250.0"))
    assert provider.history_calls == [(MAINNET_ETH, TX_TIME)]
    assert provider.current_calls == 0


@pytest.mark.asyncio
async def test_async_incoming_weth_values_using_native_eth_observation() -> None:
    """Incoming canonical WETH uses ETH price with original WETH amount identity."""
    provider = FakePrices(make_quote("3000"))
    result = await value_swap_by_eth(make_swap(LDO, MAINNET_WETH9), provider)
    assert result.anchor_side == "in"
    assert result.anchor_asset_id == MAINNET_WETH9
    assert result.pricing_asset_id == MAINNET_ETH
    assert result.value_usd == UsdAmount(Decimal("150000"))
    assert provider.history_calls == [(MAINNET_ETH, TX_TIME)]


@pytest.mark.asyncio
async def test_async_missing_quote_retains_unknown_both_legs() -> None:
    """No historical ETH point leaves proceeds and acquired basis unknown."""
    provider = FakePrices(None)
    outcome = await value_swap_by_eth(make_swap(MAINNET_WETH9), provider)
    assert outcome.status is SwapValuationStatus.PRICE_UNAVAILABLE
    assert outcome.disposal_proceeds is None
    assert outcome.acquired_basis is None
    assert provider.history_calls == [(MAINNET_ETH, TX_TIME)]


@pytest.mark.asyncio
async def test_async_no_eth_side_does_not_fetch_provider() -> None:
    """Unsupported pair must not make needless historical API calls."""
    provider = FakePrices(make_quote())
    outcome = await value_swap_by_eth(make_swap(LDO, LINK), provider)
    assert outcome.status is SwapValuationStatus.NO_ETH_ANCHOR
    assert provider.history_calls == []


@pytest.mark.asyncio
async def test_async_invalid_window_rejected_before_provider_call() -> None:
    """Malformed window must fail before network IO."""
    provider = FakePrices(make_quote())
    with pytest.raises(ValueError):
        await value_swap_by_eth(make_swap(), provider, max_distance=timedelta(0))
    assert provider.history_calls == []


@pytest.mark.asyncio
async def test_async_eth_weth_conversion_rejected_before_provider_call() -> None:
    """Conversion cannot sneak into taxable swap path."""
    provider = FakePrices(make_quote())
    with pytest.raises(ValueError, match="Wrap/Unwrap"):
        await value_swap_by_eth(make_swap(MAINNET_ETH, MAINNET_WETH9), provider)
    assert provider.history_calls == []


@pytest.mark.asyncio
async def test_provider_errors_are_not_masked_as_unknown() -> None:
    """Transport failure is not the same as legitimately missing historical price."""

    class FailingPrices(FakePrices):
        """Simulate retryable error after provider failure."""

        async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
            """Raise typed provider error without returning missing quote."""
            raise ProviderTimeoutError("alchemy", "get_historical_price")

    with pytest.raises(ProviderTimeoutError):
        await value_swap_by_eth(make_swap(), FailingPrices(None))


@pytest.mark.asyncio
async def test_async_malformed_return_type_remains_error() -> None:
    """Malformed provider response should fail type checking, not become UNKNOWN."""
    provider = FakePrices(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        await value_swap_by_eth(make_swap(), provider)


def test_stablecoin_valuation_still_uses_stablecoin_price_identity() -> None:
    """Refactoring shared valuation must preserve the stablecoin anchor and its price."""
    swap = make_swap(MAINNET_USDC, LINK, "1000", "50")
    result = calculate_stablecoin_swap_valuation(swap, MAINNET_USD_STABLECOINS, make_quote("0.995"))
    assert result.status is SwapValuationStatus.VALUED
    assert result.anchor_asset_id == MAINNET_USDC
    assert result.pricing_asset_id == MAINNET_USDC
    assert result.value_usd == UsdAmount(Decimal("995.000"))


@pytest.mark.parametrize(
    "invalid",
    [
        {"pricing_asset_id": "ETH"},
        {"pricing_asset_id": AssetId(8453, None)},
        {"status": SwapValuationStatus.NO_ETH_ANCHOR},
        {"status": SwapValuationStatus.NO_ETH_ANCHOR, "pricing_asset_id": MAINNET_ETH},
    ],
)
def test_invalid_provenance_or_status_construction_is_rejected(
    invalid: dict[str, object],
) -> None:
    """A consumer cannot forge inconsistent valuation source or missing-anchor status."""
    known = calculate_eth_swap_valuation(make_swap(), make_quote())
    fields: dict[str, object] = {
        "swap": known.swap,
        "anchor_asset_id": known.anchor_asset_id,
        "anchor_side": known.anchor_side,
        "quote": known.quote,
        "value_usd": known.value_usd,
        "status": known.status,
        "pricing_asset_id": known.pricing_asset_id,
    }
    fields.update(invalid)
    with pytest.raises((ValueError, TypeError)):
        SwapUsdValuation(**fields)  # type: ignore[arg-type]


def test_valuation_is_immutable() -> None:
    """Known USD amount and evidence are immutable once calculated."""
    result = calculate_eth_swap_valuation(make_swap(), make_quote())
    with pytest.raises(FrozenInstanceError):
        result.pricing_asset_id = LINK  # type: ignore[misc]


def test_large_quantity_keeps_small_tail_without_rounding() -> None:
    """A huge amount times a small price must not lose trailing decimal digits."""
    amount = "123456789012345678901234567890.000000000000000001"
    price = "0.0000000000123456789"
    result = calculate_eth_swap_valuation(make_swap(MAINNET_WETH9, LINK, amount), make_quote(price))
    with localcontext() as context:
        context.prec = 120
        assert result.value_usd == UsdAmount(Decimal(amount) * Decimal(price))

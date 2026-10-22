"""Stablecoin-side swap valuation and shared USD accounting semantics."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from portfolio_analyzer.application.provider_errors import ProviderTimeoutError
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.trade_valuation import (
    SwapUsdValuation,
    SwapValuationStatus,
    calculate_stablecoin_swap_valuation,
    select_stablecoin_side,
    value_swap_by_stablecoin,
)
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.events import EventMetadata, Swap
from portfolio_analyzer.domain.money import UsdAmount, UsdPrice
from portfolio_analyzer.ethereum.stablecoins import (
    MAINNET_USD_STABLECOINS,
    MAINNET_USDC,
    MAINNET_USDT,
)

TX_TIME = datetime(2024, 3, 11, 12, 0, tzinfo=UTC)
LINK = AssetId(1, "0x" + "a1" * 20)
LDO = AssetId(1, "0x" + "b2" * 20)
FAKE_USDC = AssetId(1, "0x" + "c3" * 20)
ETH = AssetId(1, None)
REGISTRY = MAINNET_USD_STABLECOINS


def make_swap(
    asset_out: AssetId = MAINNET_USDC,
    asset_in: AssetId = LINK,
    quantity_out: str = "1000",
    quantity_in: str = "50",
) -> Swap:
    """Create a deterministic classified two-asset swap for tests."""
    return Swap(
        metadata=EventMetadata("0x" + "ab" * 32, 20_000_000, TX_TIME),
        amount_out=AssetAmount(asset_out, Decimal(quantity_out)),
        amount_in=AssetAmount(asset_in, Decimal(quantity_in)),
    )


def quote(price: str = "1", when: datetime = TX_TIME) -> PriceQuote:
    """Create an observed price with a real timestamp and source."""
    return PriceQuote(UsdPrice(Decimal(price)), when, "historical_price_test")


class FakeHistoricalPriceProvider:
    """Fake async price provider recording actual historical lookup arguments."""

    def __init__(self, response: PriceQuote | None) -> None:
        """Store a deterministic response for the next lookup."""
        self.response = response
        self.requests: list[tuple[AssetId, datetime]] = []

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Record one historical quote request and return the configured answer."""
        self.requests.append((asset_id, at))
        return self.response

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Reject current pricing in a historical valuation test."""
        raise AssertionError(f"Current price must not be used: {asset_id}")


@pytest.mark.parametrize(
    ("asset_out", "asset_in", "side"),
    [
        (MAINNET_USDC, LINK, "out"),
        (MAINNET_USDT, LDO, "out"),
        (ETH, MAINNET_USDC, "in"),
        (LDO, MAINNET_USDT, "in"),
        (MAINNET_USDC, MAINNET_USDT, "out"),
        (MAINNET_USDT, MAINNET_USDC, "out"),
        (ETH, LINK, None),
        (FAKE_USDC, LINK, None),
        (LINK, LDO, None),
        (AssetId(8453, MAINNET_USDC.contract_address), AssetId(8453, LINK.contract_address), None),
    ],
)
def test_anchor_selection_uses_only_contract_and_chain(
    asset_out: AssetId, asset_in: AssetId, side: str | None
) -> None:
    """Do not trust ticker labels, other networks, or both sides independently."""
    assert select_stablecoin_side(make_swap(asset_out, asset_in), REGISTRY) == side


@pytest.mark.parametrize("quote_price", ["1", "0.998", "1.025", "0"])
def test_observed_stablecoin_price_values_both_accounting_legs(quote_price: str) -> None:
    """A single observed quote must create identical proceeds and new basis."""
    swap = make_swap()
    observed_quote = quote(quote_price)
    outcome = calculate_stablecoin_swap_valuation(swap, REGISTRY, observed_quote)
    expected = Decimal("1000") * Decimal(quote_price)
    assert outcome.status is SwapValuationStatus.VALUED
    assert outcome.is_known
    assert outcome.anchor_asset_id == MAINNET_USDC
    assert outcome.anchor_side == "out"
    assert outcome.quote is observed_quote
    assert outcome.value_usd == UsdAmount(expected)
    assert outcome.disposal_proceeds is outcome.value_usd
    assert outcome.acquired_basis is outcome.disposal_proceeds
    assert outcome.swap is swap


def test_incoming_stablecoin_values_disposed_nonstable_asset() -> None:
    """LINK to USDT must use the stablecoin received as the only USD anchor."""
    swap = make_swap(LINK, MAINNET_USDT, "50", "985")
    outcome = calculate_stablecoin_swap_valuation(swap, REGISTRY, quote("1.002"))
    assert outcome.anchor_side == "in"
    assert outcome.value_usd == UsdAmount(Decimal("986.970"))
    assert outcome.acquired_basis is outcome.disposal_proceeds


def test_two_stablecoins_do_not_double_count_or_fetch_two_prices() -> None:
    """A trusted-token pair uses only the chosen outgoing anchor."""
    swap = make_swap(MAINNET_USDC, MAINNET_USDT, "1000", "999")
    outcome = calculate_stablecoin_swap_valuation(swap, REGISTRY, quote("0.999"))
    assert outcome.value_usd == UsdAmount(Decimal("999.000"))
    assert outcome.anchor_asset_id == MAINNET_USDC


@pytest.mark.parametrize(
    ("timestamp", "expected_status"),
    [
        (TX_TIME, SwapValuationStatus.VALUED),
        (TX_TIME - timedelta(minutes=15), SwapValuationStatus.VALUED),
        (TX_TIME + timedelta(minutes=15), SwapValuationStatus.VALUED),
        (TX_TIME + timedelta(minutes=15, microseconds=1), SwapValuationStatus.PRICE_OUTSIDE_WINDOW),
        (TX_TIME - timedelta(hours=4), SwapValuationStatus.PRICE_OUTSIDE_WINDOW),
    ],
)
def test_historical_time_window_does_not_hide_stale_or_future_observations(
    timestamp: datetime, expected_status: SwapValuationStatus
) -> None:
    """Use the observed timestamp, not a fabricated transaction timestamp."""
    price = quote("1", timestamp)
    outcome = calculate_stablecoin_swap_valuation(make_swap(), REGISTRY, price)
    assert outcome.status is expected_status
    assert outcome.quote is price
    assert outcome.is_known == (expected_status is SwapValuationStatus.VALUED)
    if expected_status is not SwapValuationStatus.VALUED:
        assert outcome.acquired_basis is None
        assert outcome.disposal_proceeds is None


def test_missing_price_preserves_unknown_for_both_legs() -> None:
    """Registry membership alone cannot create a USD trade value."""
    outcome = calculate_stablecoin_swap_valuation(make_swap(), REGISTRY, None)
    assert outcome.status is SwapValuationStatus.PRICE_UNAVAILABLE
    assert outcome.anchor_asset_id == MAINNET_USDC
    assert outcome.value_usd is None
    assert outcome.disposal_proceeds is None
    assert outcome.acquired_basis is None
    assert not outcome.is_known


def test_no_allowlisted_asset_never_values_using_an_unrelated_quote() -> None:
    """An unrelated historical ETH quote cannot accidentally price LDO/LINK."""
    outcome = calculate_stablecoin_swap_valuation(make_swap(LDO, LINK), REGISTRY, quote("100"))
    assert outcome.status is SwapValuationStatus.NO_STABLECOIN_ANCHOR
    assert outcome.anchor_side is None
    assert outcome.quote is None
    assert outcome.value_usd is None


def test_symbol_spoofing_is_not_an_anchor() -> None:
    """A fake ERC-20 with a USDC symbol would not be allowlisted."""
    assert not REGISTRY.is_trusted(FAKE_USDC)
    result = calculate_stablecoin_swap_valuation(make_swap(FAKE_USDC), REGISTRY, quote())
    assert result.status is SwapValuationStatus.NO_STABLECOIN_ANCHOR


def test_observed_quote_is_not_replaced_with_registry_peg() -> None:
    """A depegged stablecoin must be valued using its observed market quote."""
    result = calculate_stablecoin_swap_valuation(make_swap(), REGISTRY, quote("0.88"))
    assert result.value_usd == UsdAmount(Decimal("880.00"))
    assert REGISTRY.find(MAINNET_USDC).peg_target_usd == Decimal("1")  # type: ignore[union-attr]


def test_decimal_multiplication_preserves_full_financial_precision() -> None:
    """Multiplication must not truncate amounts in the ambient Decimal context."""
    swap = make_swap(quantity_out="123456789012345678901234567890.12345678901234567890")
    price = quote("0.9999999999999999999999999999999999999")
    outcome = calculate_stablecoin_swap_valuation(swap, REGISTRY, price)
    # Coefficient precision can exceed Decimal's global context.
    with localcontext() as decimal_context:
        decimal_context.prec = 100
        expected = swap.amount_out.quantity * price.price_usd.value
    assert outcome.value_usd == UsdAmount(expected)


@pytest.mark.parametrize("limit", [timedelta(0), timedelta(seconds=-1), timedelta(hours=2)])
def test_invalid_quote_window_is_rejected(limit: timedelta) -> None:
    """An unlimited quote tolerance should not silently bypass timestamp policy."""
    with pytest.raises(ValueError):
        calculate_stablecoin_swap_valuation(make_swap(), REGISTRY, quote(), max_distance=limit)


@pytest.mark.parametrize("limit", [None, "15m", 5])
def test_wrong_quote_window_type_is_rejected(limit: object) -> None:
    """Only a timedelta is accepted as the historical quality policy."""
    with pytest.raises(TypeError):
        calculate_stablecoin_swap_valuation(
            make_swap(),
            REGISTRY,
            quote(),
            max_distance=limit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad", [None, "USDC", 23])
def test_registry_must_be_a_registry(bad: object) -> None:
    """No conversion from raw symbols or other untyped registry inputs."""
    with pytest.raises(TypeError):
        select_stablecoin_side(make_swap(), bad)  # type: ignore[arg-type]


def test_invalid_quote_type_is_not_treated_as_missing() -> None:
    """A broken provider DTO is not equivalent to a genuinely missing quote."""
    with pytest.raises(TypeError):
        calculate_stablecoin_swap_valuation(make_swap(), REGISTRY, "1")  # type: ignore[arg-type]


def test_result_dataclass_cannot_be_mutated() -> None:
    """Prevent accidental assignment of different basis and proceeds values."""
    result = calculate_stablecoin_swap_valuation(make_swap(), REGISTRY, quote())
    with pytest.raises(FrozenInstanceError):
        result.value_usd = UsdAmount(Decimal("2"))  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anchor_side": None, "anchor_asset_id": MAINNET_USDC},
        {"anchor_side": "in", "anchor_asset_id": MAINNET_USDC},
        {"anchor_side": "bad", "anchor_asset_id": MAINNET_USDC},
        {"value_usd": UsdAmount(Decimal("1"))},
        {"status": SwapValuationStatus.VALUED, "quote": None},
        {"status": SwapValuationStatus.NO_STABLECOIN_ANCHOR, "quote": quote()},
        {"status": SwapValuationStatus.PRICE_UNAVAILABLE, "quote": quote()},
        {"status": SwapValuationStatus.PRICE_OUTSIDE_WINDOW, "quote": None},
    ],
)
def test_result_rejects_inconsistent_known_and_unknown_states(kwargs: dict[str, object]) -> None:
    """Prevent a claimed known valuation without a valid single source."""
    args: dict[str, object] = {
        "swap": make_swap(),
        "anchor_asset_id": MAINNET_USDC,
        "anchor_side": "out",
        "quote": quote(),
        "value_usd": UsdAmount(Decimal("1000")),
        "status": SwapValuationStatus.VALUED,
    }
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        SwapUsdValuation(**args)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_async_provider_called_once_with_stablecoin_and_tx_timestamp() -> None:
    """Request exactly one stablecoin historical quote for both accounting uses."""
    price_provider = FakeHistoricalPriceProvider(quote("0.9998"))
    swap = make_swap()
    result = await value_swap_by_stablecoin(swap, REGISTRY, price_provider)
    assert price_provider.requests == [(MAINNET_USDC, TX_TIME)]
    assert result.value_usd == UsdAmount(Decimal("999.8000"))
    assert result.disposal_proceeds is result.acquired_basis


@pytest.mark.asyncio
async def test_async_incoming_stablecoin_queries_received_asset() -> None:
    """A swap receiving USDT must query USDT, not LINK."""
    price_provider = FakeHistoricalPriceProvider(quote("1.01"))
    swap = make_swap(LINK, MAINNET_USDT, "50", "900")
    outcome = await value_swap_by_stablecoin(swap, REGISTRY, price_provider)
    assert price_provider.requests == [(MAINNET_USDT, TX_TIME)]
    assert outcome.value_usd == UsdAmount(Decimal("909.00"))


@pytest.mark.asyncio
async def test_async_no_stablecoin_skips_provider_entirely() -> None:
    """No price lookup for unsupported trade shapes in this valuation stage."""
    price_provider = FakeHistoricalPriceProvider(quote())
    outcome = await value_swap_by_stablecoin(make_swap(ETH, LINK), REGISTRY, price_provider)
    assert outcome.status is SwapValuationStatus.NO_STABLECOIN_ANCHOR
    assert price_provider.requests == []


@pytest.mark.asyncio
async def test_async_missing_price_is_not_zero_or_fabricated_one() -> None:
    """A successful lookup returning None has a diagnostic unknown result."""
    price_provider = FakeHistoricalPriceProvider(None)
    outcome = await value_swap_by_stablecoin(make_swap(), REGISTRY, price_provider)
    assert outcome.status is SwapValuationStatus.PRICE_UNAVAILABLE
    assert outcome.value_usd is None
    assert price_provider.requests == [(MAINNET_USDC, TX_TIME)]


@pytest.mark.asyncio
async def test_async_provider_errors_are_not_swallowed_as_missing_price() -> None:
    """A timeout must propagate rather than incorrectly assign unknown pricing."""

    class TimeoutProvider(FakeHistoricalPriceProvider):
        """Fake provider raising an explicit infrastructure timeout."""

        async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
            """Simulate a terminal request timeout."""
            raise ProviderTimeoutError("alchemy", "get_historical_price")

    with pytest.raises(ProviderTimeoutError):
        await value_swap_by_stablecoin(make_swap(), REGISTRY, TimeoutProvider(None))


@pytest.mark.asyncio
async def test_async_wrong_type_is_not_a_missing_quote() -> None:
    """Do not swallow broken provider responses."""

    class InvalidProvider(FakeHistoricalPriceProvider):
        """Deliberately break the provider contract at runtime."""

        async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
            """Return a wrong type to exercise runtime validation."""
            return "missing"  # type: ignore[return-value]

    with pytest.raises(TypeError):
        await value_swap_by_stablecoin(make_swap(), REGISTRY, InvalidProvider(None))


def test_price_timestamp_normalizes_to_utc() -> None:
    """Timezone offsets must not affect comparison to the transaction timestamp."""
    when = datetime(2024, 3, 11, 15, 0, tzinfo=timezone(timedelta(hours=3)))
    result = calculate_stablecoin_swap_valuation(make_swap(), REGISTRY, quote(when=when))
    assert result.status is SwapValuationStatus.VALUED
    assert result.quote is not None
    assert result.quote.observed_at == TX_TIME


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("swap", None),
        ("status", "valued"),
        ("anchor_asset_id", "0x" + "a1" * 20),
        ("quote", "historical"),
        ("value_usd", Decimal("1000")),
        ("value_usd", UsdAmount(Decimal("-1"))),
    ],
)
def test_valuation_model_rejects_wrong_types_and_negative_trade_value(
    field: str, invalid: object
) -> None:
    """Financial result construction must reject malformed typed fields."""
    kwargs: dict[str, object] = {
        "swap": make_swap(),
        "anchor_asset_id": MAINNET_USDC,
        "anchor_side": "out",
        "quote": quote(),
        "value_usd": UsdAmount(Decimal("1000")),
        "status": SwapValuationStatus.VALUED,
    }
    kwargs[field] = invalid
    with pytest.raises((TypeError, ValueError)):
        SwapUsdValuation(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"anchor_asset_id": None, "anchor_side": None, "status": SwapValuationStatus.VALUED},
        {
            "anchor_asset_id": None,
            "anchor_side": None,
            "value_usd": None,
            "status": SwapValuationStatus.PRICE_UNAVAILABLE,
        },
        {
            "anchor_asset_id": None,
            "anchor_side": None,
            "value_usd": None,
            "status": SwapValuationStatus.NO_STABLECOIN_ANCHOR,
        },
        {
            "status": SwapValuationStatus.PRICE_UNAVAILABLE,
            "value_usd": None,
        },
        {
            "status": SwapValuationStatus.PRICE_OUTSIDE_WINDOW,
            "value_usd": None,
            "quote": None,
        },
    ],
)
def test_valuation_model_does_not_accept_incoherent_status(overrides: dict[str, object]) -> None:
    """Each status must expose exactly the provenance it claims to have."""
    kwargs: dict[str, object] = {
        "swap": make_swap(),
        "anchor_asset_id": MAINNET_USDC,
        "anchor_side": "out",
        "quote": quote(),
        "value_usd": UsdAmount(Decimal("1000")),
        "status": SwapValuationStatus.VALUED,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        SwapUsdValuation(**kwargs)  # type: ignore[arg-type]


def test_valuation_rejects_invalid_swap_to_anchor_selection() -> None:
    """Only a confirmed domain Swap can enter stablecoin trade valuation."""
    with pytest.raises(TypeError):
        select_stablecoin_side(None, REGISTRY)  # type: ignore[arg-type]

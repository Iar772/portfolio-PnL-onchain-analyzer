"""Conservative two-sided historical volatile ERC-20 swap pricing tests."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from portfolio_analyzer.application.price_divergence import PriceDivergencePolicy
from portfolio_analyzer.application.provider_errors import ProviderRateLimitError
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.trade_valuation import SwapValuationStatus
from portfolio_analyzer.application.volatile_valuation import (
    VolatilePriceComparison,
    calculate_volatile_swap_valuation,
    is_volatile_erc20_pair,
    value_swap_by_volatile,
)
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.events import EventMetadata, Swap
from portfolio_analyzer.domain.money import UsdAmount, UsdPrice
from portfolio_analyzer.ethereum.known_contracts import MAINNET_WETH9
from portfolio_analyzer.ethereum.stablecoins import MAINNET_USD_STABLECOINS, MAINNET_USDC


def fixture_policy(value: Decimal) -> PriceDivergencePolicy:
    """Use synthetic tolerance solely to test deterministic comparison branches."""
    return PriceDivergencePolicy(value, "unit-fixture:synthetic-threshold")


TEST_POLICY = fixture_policy(Decimal("0.05"))


TX_TIME = datetime(2024, 9, 10, 12, 0, tzinfo=UTC)
LINK = AssetId(1, "0x" + "aa" * 20)
LDO = AssetId(1, "0x" + "bb" * 20)
FAKE_USDC = AssetId(1, "0x" + "cc" * 20)
ETH = AssetId(1, None)
REGISTRY = MAINNET_USD_STABLECOINS


def make_swap(
    outgoing: AssetId = LDO,
    incoming: AssetId = LINK,
    out_qty: str = "50",
    in_qty: str = "100",
) -> Swap:
    """Build exact non-native swap event for comparisons."""
    return Swap(
        EventMetadata("0x" + "ab" * 32, 20_000_000, TX_TIME),
        AssetAmount(outgoing, Decimal(out_qty)),
        AssetAmount(incoming, Decimal(in_qty)),
    )


def quote(price: str, *, offset_minutes: int = 0) -> PriceQuote:
    """Build a timestamped historical market observation."""
    return PriceQuote(
        UsdPrice(Decimal(price)),
        TX_TIME + timedelta(minutes=offset_minutes),
        "offline_test",
    )


class FakePrices:
    """Provide per-identity quotes and capture timestamped lookups."""

    def __init__(self, quotes: dict[AssetId, PriceQuote | None]) -> None:
        """Store deterministic responses for a historical pricing fixture."""
        self.quotes = quotes
        self.calls: list[tuple[AssetId, datetime]] = []

    async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote | None:
        """Return configured historical observations in exact call order."""
        self.calls.append((asset_id, at))
        return self.quotes.get(asset_id)

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Fail if current pricing is used in historical trade valuation."""
        raise AssertionError("Current price cannot value a historical swap")


@pytest.mark.parametrize(
    ("outgoing", "incoming", "eligible"),
    [
        (LDO, LINK, True),
        (FAKE_USDC, LINK, True),
        (LINK, LDO, True),
        (MAINNET_USDC, LINK, False),
        (LINK, MAINNET_USDC, False),
        (ETH, LINK, False),
        (LINK, ETH, False),
        (MAINNET_WETH9, LINK, False),
        (LINK, MAINNET_WETH9, False),
        (AssetId(8453, "0x" + "aa" * 20), AssetId(8453, "0x" + "bb" * 20), False),
    ],
)
def test_pair_eligibility_uses_verified_asset_identity(
    outgoing: AssetId,
    incoming: AssetId,
    eligible: bool,
) -> None:
    """Accept only two Mainnet volatile ERC-20s; never ticker aliases."""
    assert is_volatile_erc20_pair(make_swap(outgoing, incoming), REGISTRY) is eligible


def test_both_priced_and_consistent_share_one_accounting_value() -> None:
    """Never independently calculate acquired basis from the incoming quote."""
    swap = make_swap()
    result = calculate_volatile_swap_valuation(
        swap, REGISTRY, quote("2"), quote("1"), divergence_policy=TEST_POLICY
    )
    assert result.valuation.swap is swap
    assert result.valuation.status is SwapValuationStatus.VALUED
    assert result.valuation.anchor_side == "out"
    assert result.valuation.anchor_asset_id == LDO
    assert result.valuation.pricing_asset_id == LDO
    assert result.outgoing_value_usd == UsdAmount(Decimal("100"))
    assert result.incoming_value_usd == UsdAmount(Decimal("100"))
    assert result.relative_spread == Decimal("0")
    assert result.disposal_proceeds is result.acquired_basis
    assert result.disposal_proceeds is result.valuation.value_usd
    assert result.acquired_basis == UsdAmount(Decimal("100"))


@pytest.mark.parametrize(
    ("out_price", "in_price", "status"),
    [
        ("2", "1.01", SwapValuationStatus.VALUED),
        ("2", "1.05", SwapValuationStatus.VALUED),
        ("2", "1.06", SwapValuationStatus.VOLATILE_PRICE_DISAGREEMENT),
        ("2", "0.8", SwapValuationStatus.VOLATILE_PRICE_DISAGREEMENT),
        ("0", "1", SwapValuationStatus.VOLATILE_PRICE_UNRELIABLE),
        ("2", "0", SwapValuationStatus.VOLATILE_PRICE_UNRELIABLE),
        ("0", "0", SwapValuationStatus.VOLATILE_PRICE_UNRELIABLE),
    ],
)
def test_comparison_handles_agreement_divergence_and_zero(
    out_price: str,
    in_price: str,
    status: SwapValuationStatus,
) -> None:
    """Outlying or zero volatile quotes cannot create invented known basis."""
    result = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, quote(out_price), quote(in_price), divergence_policy=TEST_POLICY
    )
    assert result.valuation.status is status
    if status is SwapValuationStatus.VALUED:
        assert result.acquired_basis is result.disposal_proceeds
    else:
        assert result.acquired_basis is None
        assert result.disposal_proceeds is None
        assert result.outgoing_value_usd is not None
        assert result.incoming_value_usd is not None


@pytest.mark.parametrize("missing_side", ["out", "in", "both"])
def test_missing_any_historical_quote_keeps_both_cost_values_unknown(missing_side: str) -> None:
    """One good quote alone is insufficient evidence for volatile-pair valuation."""
    out_quote = None if missing_side in ("out", "both") else quote("2")
    in_quote = None if missing_side in ("in", "both") else quote("1")
    result = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, out_quote, in_quote, divergence_policy=TEST_POLICY
    )
    assert result.valuation.status is SwapValuationStatus.PRICE_UNAVAILABLE
    assert not result.valuation.is_known
    assert result.disposal_proceeds is None
    assert result.acquired_basis is None


@pytest.mark.parametrize(("drift_out", "drift_in"), [(16, 0), (0, -16), (16, -16)])
def test_stale_either_quote_is_unreliable(drift_out: int, drift_in: int) -> None:
    """Always compare each observed_at against the actual transaction timestamp."""
    result = calculate_volatile_swap_valuation(
        make_swap(),
        REGISTRY,
        quote("2", offset_minutes=drift_out),
        quote("1", offset_minutes=drift_in),
        divergence_policy=TEST_POLICY,
    )
    assert result.valuation.status is SwapValuationStatus.VOLATILE_PRICE_UNRELIABLE
    assert result.valuation.value_usd is None


@pytest.mark.parametrize("drift", [-15, 0, 15])
def test_both_quotes_within_inclusive_window_accepted(drift: int) -> None:
    """The configured observation window includes its own endpoints."""
    result = calculate_volatile_swap_valuation(
        make_swap(),
        REGISTRY,
        quote("2", offset_minutes=drift),
        quote("1", offset_minutes=drift),
        divergence_policy=TEST_POLICY,
    )
    assert result.valuation.is_known


def test_misaligned_observation_times_are_transparently_kept() -> None:
    """Timestamp provenance for both independent price observations is retained."""
    first = quote("2", offset_minutes=-13)
    second = quote("1", offset_minutes=14)
    result = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, first, second, divergence_policy=TEST_POLICY
    )
    assert result.valuation.is_known
    assert result.outgoing_quote is first
    assert result.incoming_quote is second


def test_threshold_can_be_configured_without_replacing_price_quotes() -> None:
    """Allow deliberate risk-policy tuning while preserving identical USD legs."""
    assert PriceDivergencePolicy().max_relative_spread is None
    swap = make_swap()
    strict = calculate_volatile_swap_valuation(
        swap, REGISTRY, quote("2"), quote("1.06"), divergence_policy=TEST_POLICY
    )
    permissive = calculate_volatile_swap_valuation(
        swap,
        REGISTRY,
        quote("2"),
        quote("1.06"),
        divergence_policy=fixture_policy(Decimal("0.10")),
    )
    assert strict.valuation.status is SwapValuationStatus.VOLATILE_PRICE_DISAGREEMENT
    assert permissive.valuation.status is SwapValuationStatus.VALUED
    assert permissive.valuation.value_usd == UsdAmount(Decimal("100"))


@pytest.mark.parametrize(
    "bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-0.01"), Decimal("1.1")]
)
def test_invalid_tolerance_rejected(bad: Decimal) -> None:
    """Never silently use meaningless tolerances including NaN and infinity."""
    with pytest.raises(ValueError):
        calculate_volatile_swap_valuation(
            make_swap(),
            REGISTRY,
            quote("2"),
            quote("1"),
            divergence_policy=fixture_policy(bad),
        )


@pytest.mark.parametrize("bad", [0.05, "0.05"])
def test_invalid_tolerance_type_rejected(bad: object) -> None:
    """Reject binary floats and unrelated input objects at runtime."""
    with pytest.raises(TypeError):
        calculate_volatile_swap_valuation(
            make_swap(),
            REGISTRY,
            quote("2"),
            quote("1"),
            divergence_policy=fixture_policy(bad),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad", [timedelta(0), timedelta(hours=2), -timedelta(minutes=1)])
def test_invalid_window_rejected(bad: timedelta) -> None:
    """Historical observation windows must use the shared application policy."""
    with pytest.raises(ValueError):
        calculate_volatile_swap_valuation(
            make_swap(),
            REGISTRY,
            quote("2"),
            quote("1"),
            max_distance=bad,
            divergence_policy=TEST_POLICY,
        )


def test_unsupported_pairs_do_not_use_volatility_pricing() -> None:
    """Stablecoin and ETH/WETH routes must be valued by their own policies."""
    result = calculate_volatile_swap_valuation(
        make_swap(MAINNET_USDC, LINK),
        REGISTRY,
        quote("1"),
        quote("1"),
        divergence_policy=TEST_POLICY,
    )
    assert result.valuation.status is SwapValuationStatus.NO_VOLATILE_PAIR
    assert result.valuation.anchor_asset_id is None
    assert result.outgoing_quote is None
    assert result.incoming_quote is None


@pytest.mark.parametrize("invalid", ["2", 12, object()])
def test_invalid_quote_types_rejected(invalid: object) -> None:
    """Application layer must reject corrupt provider response types."""
    with pytest.raises(TypeError):
        calculate_volatile_swap_valuation(
            make_swap(),
            REGISTRY,
            invalid,
            quote("1"),  # type: ignore[arg-type]
            divergence_policy=TEST_POLICY,
        )


def test_extremely_precise_micro_and_large_token_values_are_preserved() -> None:
    """High-precision cross-quote arithmetic cannot silently lose token value."""
    swap = make_swap(
        out_qty="12345678901234567890123456789.123456789012345678",
        in_qty="12345678901234567890123456789.123456789012345678",
    )
    result = calculate_volatile_swap_valuation(
        swap,
        REGISTRY,
        quote("0.000000000000000000009"),
        quote("0.000000000000000000009"),
        divergence_policy=TEST_POLICY,
    )
    with localcontext() as ctx:
        ctx.prec = 120
        expected = swap.amount_out.quantity * Decimal("0.000000000000000000009")
    assert result.valuation.is_known
    assert result.acquired_basis == UsdAmount(expected)


@pytest.mark.asyncio
async def test_fetches_two_distinct_assets_at_same_transaction_timestamp() -> None:
    """Only historical, identity-based prices may feed volatile valuation."""
    fake = FakePrices({LDO: quote("2"), LINK: quote("1")})
    result = await value_swap_by_volatile(
        make_swap(), REGISTRY, fake, divergence_policy=TEST_POLICY
    )
    assert result.valuation.is_known
    assert fake.calls == [(LDO, TX_TIME), (LINK, TX_TIME)]


@pytest.mark.asyncio
async def test_fetches_incoming_even_if_outgoing_missing() -> None:
    """Retain audit provenance without pretending that one quote is enough."""
    fake = FakePrices({LDO: None, LINK: quote("1")})
    result = await value_swap_by_volatile(
        make_swap(), REGISTRY, fake, divergence_policy=TEST_POLICY
    )
    assert not result.valuation.is_known
    assert result.valuation.status is SwapValuationStatus.PRICE_UNAVAILABLE
    assert fake.calls == [(LDO, TX_TIME), (LINK, TX_TIME)]


@pytest.mark.asyncio
async def test_unsupported_pair_does_not_call_api() -> None:
    """Do not spend two costly historical calls on ETH- or stablecoin-backed swaps."""
    fake = FakePrices({MAINNET_USDC: quote("1"), LINK: quote("1")})
    result = await value_swap_by_volatile(
        make_swap(MAINNET_USDC, LINK), REGISTRY, fake, divergence_policy=TEST_POLICY
    )
    assert result.valuation.status is SwapValuationStatus.NO_VOLATILE_PAIR
    assert fake.calls == []


@pytest.mark.asyncio
async def test_provider_error_propagates_without_becoming_unknown() -> None:
    """429 errors must remain retryable provider errors, not absent USD prices."""

    class BrokenPrices(FakePrices):
        """Fail on historical queries with a typed provider rate-limit error."""

        async def get_historical_price(self, asset_id: AssetId, at: datetime) -> None:
            """Raise a typed infrastructure error rather than return None."""
            raise ProviderRateLimitError("alchemy", "historical_price")

    with pytest.raises(ProviderRateLimitError):
        await value_swap_by_volatile(
            make_swap(), REGISTRY, BrokenPrices({}), divergence_policy=TEST_POLICY
        )


@pytest.mark.asyncio
async def test_invalid_provider_return_does_not_become_unpriced() -> None:
    """Malformed price provider models must fail before pricing proceeds."""
    fake = FakePrices({LDO: "invalid", LINK: quote("1")})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        await value_swap_by_volatile(make_swap(), REGISTRY, fake, divergence_policy=TEST_POLICY)
    assert fake.calls == [(LDO, TX_TIME)]


@pytest.mark.asyncio
async def test_invalid_policy_does_not_issue_any_external_request() -> None:
    """Validate risk configuration before consuming any price API calls."""
    fake = FakePrices({LDO: quote("2"), LINK: quote("1")})
    with pytest.raises(ValueError):
        await value_swap_by_volatile(
            make_swap(),
            REGISTRY,
            fake,
            divergence_policy=fixture_policy(Decimal("NaN")),
        )
    assert fake.calls == []


def test_result_is_immutable_and_not_a_second_accounting_amount() -> None:
    """Comparison wraps existing accounting result and has no competing proceeds."""
    result = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, quote("2"), quote("1"), divergence_policy=TEST_POLICY
    )
    with pytest.raises(FrozenInstanceError):
        result.relative_spread = Decimal("0.5")  # type: ignore[misc]
    assert result.disposal_proceeds is result.valuation.value_usd


@pytest.mark.parametrize("bad", ["100", Decimal("NaN"), Decimal("-1")])
def test_result_rejects_bogus_spread_evidence(bad: object) -> None:
    """A public result object cannot carry malformed relative disagreement."""
    accepted = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, quote("2"), quote("1"), divergence_policy=TEST_POLICY
    )
    error = TypeError if isinstance(bad, str) else ValueError
    with pytest.raises(error):
        VolatilePriceComparison(
            accepted.valuation,
            accepted.outgoing_quote,
            accepted.incoming_quote,
            accepted.outgoing_value_usd,
            accepted.incoming_value_usd,
            bad,  # type: ignore[arg-type]
            accepted.divergence_policy,
        )


def test_comparison_rejects_forged_incoming_value_and_spread() -> None:
    """Independent market evidence cannot be forged after acceptance."""
    base = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, quote("2"), quote("1"), divergence_policy=TEST_POLICY
    )
    with pytest.raises(ValueError, match="incoming value must match"):
        VolatilePriceComparison(
            base.valuation,
            base.outgoing_quote,
            base.incoming_quote,
            base.outgoing_value_usd,
            UsdAmount(Decimal("99")),
            Decimal("0.01"),
            base.divergence_policy,
        )
    with pytest.raises(ValueError, match="spread must match"):
        VolatilePriceComparison(
            base.valuation,
            base.outgoing_quote,
            base.incoming_quote,
            base.outgoing_value_usd,
            base.incoming_value_usd,
            Decimal("0.01"),
            base.divergence_policy,
        )


def test_comparison_rejects_wrong_accounting_anchor_quote() -> None:
    """Never swap a different market quote into already-approved accounting."""
    base = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, quote("2"), quote("1"), divergence_policy=TEST_POLICY
    )
    with pytest.raises(ValueError, match="accounting quote must"):
        VolatilePriceComparison(
            base.valuation,
            quote("2", offset_minutes=1),
            base.incoming_quote,
            base.outgoing_value_usd,
            base.incoming_value_usd,
            base.relative_spread,
            base.divergence_policy,
        )


def test_comparison_rejects_unsupported_pair_with_quote_evidence() -> None:
    """Do not silently imply an ineligible token was evaluated as volatile."""
    base = calculate_volatile_swap_valuation(
        make_swap(MAINNET_USDC, LINK), REGISTRY, None, None, divergence_policy=TEST_POLICY
    )
    with pytest.raises(ValueError, match="unsupported volatile pairs"):
        VolatilePriceComparison(base.valuation, quote("1"), None, None, None, None)


@pytest.mark.parametrize(
    ("field", "replacement", "error", "fragment"),
    [
        ("valuation", None, TypeError, "valuation must"),
        ("outgoing_quote", "invalid", TypeError, "comparison quotes"),
        ("incoming_quote", "invalid", TypeError, "comparison quotes"),
        ("outgoing_value_usd", Decimal("100"), TypeError, "comparison values"),
        ("incoming_value_usd", Decimal("100"), TypeError, "comparison values"),
    ],
)
def test_result_rejects_wrong_field_types(
    field: str,
    replacement: object,
    error: type[Exception],
    fragment: str,
) -> None:
    """Reject forged mutable or incorrectly typed diagnostic evidence."""
    from dataclasses import replace

    base = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, quote("2"), quote("1"), divergence_policy=TEST_POLICY
    )
    with pytest.raises(error, match=fragment):
        replace(base, **{field: replacement})


@pytest.mark.parametrize("field", ["outgoing_value_usd", "incoming_value_usd"])
def test_result_rejects_value_without_quote(field: str) -> None:
    """USD evidence must always be traceable to a real price observation."""
    from dataclasses import replace

    base = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, None, None, divergence_policy=TEST_POLICY
    )
    with pytest.raises(ValueError, match="value requires its quote"):
        replace(base, **{field: UsdAmount(Decimal("100"))})


def test_result_rejects_forged_outgoing_market_value() -> None:
    """Both value calculations must agree with their own observed prices."""
    from dataclasses import replace

    base = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, quote("2"), quote("1"), divergence_policy=TEST_POLICY
    )
    with pytest.raises(ValueError, match="outgoing value must match"):
        replace(base, outgoing_value_usd=UsdAmount(Decimal("101")))


def test_result_rejects_spread_without_two_legs() -> None:
    """A one-sided observed quote cannot have a meaningful comparison spread."""
    from dataclasses import replace

    base = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, None, quote("1"), divergence_policy=TEST_POLICY
    )
    with pytest.raises(ValueError, match="spread requires two"):
        replace(base, relative_spread=Decimal("0"))


def test_result_rejects_spread_for_zero_price() -> None:
    """Zero-price volatile observations are never reliable cross-checks."""
    from dataclasses import replace

    base = calculate_volatile_swap_valuation(
        make_swap(), REGISTRY, quote("0"), quote("1"), divergence_policy=TEST_POLICY
    )
    with pytest.raises(ValueError, match="spread requires positive"):
        replace(base, relative_spread=Decimal("1"))


def test_result_rejects_valuation_using_incoming_as_accounting_anchor() -> None:
    """A comparison cannot silently switch which leg sets the single USD basis."""
    from dataclasses import replace

    from portfolio_analyzer.application.trade_valuation import calculate_swap_valuation_for_side

    swap = make_swap()
    base = calculate_volatile_swap_valuation(
        swap, REGISTRY, quote("2"), quote("2"), divergence_policy=TEST_POLICY
    )
    incoming_based = calculate_swap_valuation_for_side(
        swap,
        "in",
        quote("2"),
        missing_anchor_status=SwapValuationStatus.NO_ETH_ANCHOR,
        pricing_asset_id=LINK,
    )
    with pytest.raises(ValueError, match="outgoing anchor"):
        replace(base, valuation=incoming_based)


@pytest.mark.parametrize("invalid", [None, "wrong", 17])
def test_pair_eligibility_rejects_bad_swap(invalid: object) -> None:
    """Malformed domain inputs are errors, not merely unsupported pairs."""
    with pytest.raises(TypeError):
        is_volatile_erc20_pair(invalid, REGISTRY)  # type: ignore[arg-type]


def test_pair_eligibility_rejects_bad_registry() -> None:
    """Whitelist must be an actual registry rather than attacker-controlled labels."""
    with pytest.raises(TypeError):
        is_volatile_erc20_pair(make_swap(), None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_invalid_second_provider_quote_is_not_treated_as_missing() -> None:
    """Corrupt incoming API result is an error even after successful outgoing lookup."""
    fake = FakePrices({LDO: quote("2"), LINK: "broken"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="incoming quote"):
        await value_swap_by_volatile(make_swap(), REGISTRY, fake, divergence_policy=TEST_POLICY)
    assert fake.calls == [(LDO, TX_TIME), (LINK, TX_TIME)]

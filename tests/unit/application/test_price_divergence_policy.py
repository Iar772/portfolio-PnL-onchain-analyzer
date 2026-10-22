"""Uncalibrated-by-default price disagreement rules and auditability tests."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from portfolio_analyzer.application.price_divergence import (
    DivergenceDecision,
    PriceDivergencePolicy,
)
from portfolio_analyzer.application.provider_errors import ProviderRateLimitError
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.trade_valuation import SwapValuationStatus
from portfolio_analyzer.application.volatile_valuation import (
    calculate_volatile_swap_valuation,
    value_swap_by_volatile,
)
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.events import EventMetadata, Swap
from portfolio_analyzer.domain.money import UsdPrice
from portfolio_analyzer.ethereum.stablecoins import MAINNET_USD_STABLECOINS

TIME = datetime(2025, 1, 2, 3, 0, tzinfo=UTC)
OUT = AssetId(1, "0x" + "fa" * 20)
IN = AssetId(1, "0x" + "fb" * 20)


def sample_swap() -> Swap:
    """Return a synthetic two-sided non-stablecoin ERC-20 transaction."""
    return Swap(
        EventMetadata("0x" + "ab" * 32, 20_000_000, TIME),
        AssetAmount(OUT, Decimal("50")),
        AssetAmount(IN, Decimal("100")),
    )


def historical(price: str, shift: int = 0) -> PriceQuote:
    """Build a synthetic historical observation with preserved provenance."""
    return PriceQuote(
        UsdPrice(Decimal(price)),
        TIME + timedelta(minutes=shift),
        "synthetic_price_feed",
    )


@pytest.mark.parametrize("spread", [Decimal("0"), Decimal("0.001"), Decimal("0.999")])
def test_uncalibrated_policy_never_accepts_even_matching_prices(spread: Decimal) -> None:
    """Missing calibration must not silently accept any volatile valuation."""
    assert PriceDivergencePolicy().evaluate(spread) is DivergenceDecision.UNCALIBRATED


def test_default_matching_quotes_keep_both_accounting_values_unknown() -> None:
    """Zero observed divergence is evidence, not proof of an execution price."""
    result = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("2"),
        historical("1"),
    )
    assert result.valuation.status is SwapValuationStatus.PRICE_DIVERGENCE_UNCALIBRATED
    assert result.relative_spread == Decimal(0)
    assert result.divergence_policy.max_relative_spread is None
    assert result.outgoing_quote is not None
    assert result.incoming_quote is not None
    assert result.disposal_proceeds is None
    assert result.acquired_basis is None


def test_default_large_difference_keeps_diagnostic_without_basis() -> None:
    """Do not infer a sale or purchase cost from conflicting volatile markets."""
    result = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("2"),
        historical("2"),
    )
    assert result.relative_spread == Decimal("0.5")
    assert result.valuation.status is SwapValuationStatus.PRICE_DIVERGENCE_UNCALIBRATED
    assert result.valuation.value_usd is None
    assert result.outgoing_value_usd is not None
    assert result.incoming_value_usd is not None


def test_calibrated_threshold_is_strict_at_boundary() -> None:
    """Only an intentionally activated threshold can decide known versus unknown."""
    policy = PriceDivergencePolicy(Decimal("0.10"), "fixture:verified-swap-corpus")
    assert policy.evaluate(Decimal("0.10")) is DivergenceDecision.ACCEPTED
    assert policy.evaluate(Decimal("0.10001")) is DivergenceDecision.REJECTED
    assert policy.evaluate(Decimal("0")) is DivergenceDecision.ACCEPTED


def test_calibrated_high_difference_yields_unknown_disagreement() -> None:
    """A strongly divergent pair is UNKNOWN even with both quotes present."""
    policy = PriceDivergencePolicy(Decimal("0.03"), "fixture:verified-swap-corpus")
    result = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("2"),
        historical("1.10"),
        divergence_policy=policy,
    )
    assert result.valuation.status is SwapValuationStatus.VOLATILE_PRICE_DISAGREEMENT
    assert result.relative_spread > policy.max_relative_spread
    assert result.valuation.value_usd is None


def test_explicit_calibration_can_approve_same_observations() -> None:
    """Accept one common USD basis only under an explicitly documented policy."""
    policy = PriceDivergencePolicy(Decimal("0.05"), "fixture:verified-swap-corpus")
    result = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("2"),
        historical("1.02"),
        divergence_policy=policy,
    )
    assert result.valuation.status is SwapValuationStatus.VALUED
    assert result.divergence_policy.calibration_reference == "fixture:verified-swap-corpus"
    assert result.disposal_proceeds is result.acquired_basis


def test_missing_price_overrides_uncalibrated_spread() -> None:
    """Missing observations remain their own explicit diagnostic state."""
    result = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        None,
        historical("1"),
    )
    assert result.valuation.status is SwapValuationStatus.PRICE_UNAVAILABLE
    assert result.relative_spread is None


def test_stale_price_overrides_uncalibrated_spread() -> None:
    """Old evidence is unreliable regardless of calibration status."""
    result = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("2", 17),
        historical("1"),
    )
    assert result.valuation.status is SwapValuationStatus.VOLATILE_PRICE_UNRELIABLE


def test_zero_price_overrides_uncalibrated_spread() -> None:
    """An invalid zero price cannot masquerade as a regular divergence sample."""
    result = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("0"),
        historical("1"),
    )
    assert result.valuation.status is SwapValuationStatus.VOLATILE_PRICE_UNRELIABLE


@pytest.mark.parametrize(
    "bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-0.1"), Decimal("1.1")]
)
def test_calibrated_threshold_rejects_invalid_decimal(bad: Decimal) -> None:
    """Reject impossible or unsafe thresholds instead of guessing meaning."""
    with pytest.raises(ValueError):
        PriceDivergencePolicy(bad, "fixture:verified-swap-corpus")


@pytest.mark.parametrize("bad", [0.1, "0.1", True])
def test_threshold_requires_decimal_not_float(bad: object) -> None:
    """Avoid floating point or ambiguous threshold configuration."""
    with pytest.raises(TypeError):
        PriceDivergencePolicy(bad, "fixture:verified-swap-corpus")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_calibrated_threshold_requires_nonblank_evidence_reference(bad: str | None) -> None:
    """A numeric threshold without provenance cannot enter production valuation."""
    with pytest.raises((ValueError, TypeError)):
        PriceDivergencePolicy(Decimal("0.1"), bad)


def test_uncalibrated_policy_rejects_orphan_evidence_reference() -> None:
    """Do not create a misleading state with evidence but no threshold."""
    with pytest.raises(ValueError):
        PriceDivergencePolicy(calibration_reference="some-study")


def test_reference_is_normalized_and_policy_immutable() -> None:
    """Keep evidence identifiers stable across configuration reloads."""
    policy = PriceDivergencePolicy(Decimal("0.1"), "  report:run-001  ")
    assert policy.calibration_reference == "report:run-001"
    with pytest.raises(FrozenInstanceError):
        policy.max_relative_spread = Decimal("0.4")  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["0.2", 0.2, None])
def test_relative_spread_rejects_wrong_types(bad: object) -> None:
    """Reject malformed computed spreads rather than quietly accepting them."""
    with pytest.raises(TypeError):
        PriceDivergencePolicy().evaluate(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-0.1")])
def test_relative_spread_rejects_invalid_numbers(bad: Decimal) -> None:
    """Require finite, nonnegative comparative observations."""
    with pytest.raises(ValueError):
        PriceDivergencePolicy().evaluate(bad)


@pytest.mark.parametrize("bad", ["not policy", Decimal("0.1")])
def test_valuation_rejects_invalid_policy_object(bad: object) -> None:
    """Prevent accidental configuration from bypassing the calibration gate."""
    with pytest.raises(TypeError):
        calculate_volatile_swap_valuation(
            sample_swap(),
            MAINNET_USD_STABLECOINS,
            historical("2"),
            historical("1"),
            divergence_policy=bad,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_async_path_still_fetches_two_quotes_before_uncalibrated_unknown() -> None:
    """Collect diagnostics without creating a known USD value by default."""

    class Prices:
        """Fake provider capturing exact historical lookups."""

        def __init__(self) -> None:
            """Initialize an empty deterministic call log."""
            self.calls: list[tuple[AssetId, datetime]] = []

        async def get_historical_price(self, asset_id: AssetId, at: datetime) -> PriceQuote:
            """Return deterministic quotes for both sides of the swap."""
            self.calls.append((asset_id, at))
            return historical("2") if asset_id == OUT else historical("1")

        async def get_current_price(self, asset_id: AssetId) -> None:
            """Disallow requests to contemporary pricing for historical operations."""
            raise AssertionError("current price is not relevant")

    provider = Prices()
    result = await value_swap_by_volatile(sample_swap(), MAINNET_USD_STABLECOINS, provider)
    assert provider.calls == [(OUT, TIME), (IN, TIME)]
    assert result.valuation.status is SwapValuationStatus.PRICE_DIVERGENCE_UNCALIBRATED
    assert result.disposal_proceeds is None


@pytest.mark.asyncio
async def test_provider_failure_remains_error_not_unknown() -> None:
    """A provider outage cannot be silently conflated with an unavailable threshold."""

    class BrokenPrices:
        """Fake price service with an explicit provider failure."""

        async def get_historical_price(self, asset_id: AssetId, at: datetime) -> None:
            """Surface the source failure unchanged."""
            raise ProviderRateLimitError("alchemy", "get_historical_price")

        async def get_current_price(self, asset_id: AssetId) -> None:
            """Never serve current USD quotes in a historical operation."""
            raise AssertionError("unexpected")

    with pytest.raises(ProviderRateLimitError):
        await value_swap_by_volatile(sample_swap(), MAINNET_USD_STABLECOINS, BrokenPrices())


def test_cannot_forge_known_valuation_with_uncalibrated_policy() -> None:
    """The public immutable result must not permit calibration-free known basis."""
    from dataclasses import replace

    policy = PriceDivergencePolicy(Decimal("0.05"), "fixture:verified-swap-corpus")
    accepted = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("2"),
        historical("1"),
        divergence_policy=policy,
    )
    assert accepted.valuation.is_known
    with pytest.raises(ValueError, match="calibrated"):
        replace(accepted, divergence_policy=PriceDivergencePolicy())


def test_cannot_forge_divergence_diagnostic_under_different_policy() -> None:
    """Persisted policy must agree with the explicit unknown reason."""
    from dataclasses import replace

    original = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("2"),
        historical("2"),
    )
    assert original.valuation.status is SwapValuationStatus.PRICE_DIVERGENCE_UNCALIBRATED
    with pytest.raises(ValueError, match="contradicts"):
        replace(
            original,
            divergence_policy=PriceDivergencePolicy(
                Decimal("0.9"),
                "fixture:verified-swap-corpus",
            ),
        )


def test_research_reference_remains_in_accepted_comparison() -> None:
    """Accepted valuation must preserve provenance of its acceptance rule."""
    policy = PriceDivergencePolicy(Decimal("0.10"), "research:dataset-v2")
    result = calculate_volatile_swap_valuation(
        sample_swap(),
        MAINNET_USD_STABLECOINS,
        historical("2"),
        historical("1"),
        divergence_policy=policy,
    )
    assert result.valuation.is_known
    assert result.divergence_policy is policy
    assert result.divergence_policy.calibration_reference == "research:dataset-v2"

"""Pure application value coverage and replaceable current-price provider tests."""

from datetime import UTC, datetime
from decimal import Decimal, getcontext

import pytest

from portfolio_analyzer.application.ports import CurrentPriceProvider
from portfolio_analyzer.application.provider_models import PriceQuote
from portfolio_analyzer.application.valuation import price_current_balances, value_current_balances
from portfolio_analyzer.domain.assets import AssetAmount, AssetId
from portfolio_analyzer.domain.money import UsdAmount, UsdPrice
from portfolio_analyzer.ethereum.balance_models import UnresolvedTokenBalance

_ETH = AssetId(1, None)
_TOKEN = AssetId(1, "0x" + "ab" * 20)
_UNKNOWN = AssetId(1, "0x" + "cd" * 20)
_AT = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


def _quote(value: str) -> PriceQuote:
    """Construct a fully typed observation."""
    return PriceQuote(UsdPrice(Decimal(value)), _AT, "fake")


def test_partial_valuation_excludes_unpriced_from_known_total() -> None:
    """Missing quote never inflates or zeroes out the entire portfolio."""
    result = value_current_balances(
        [AssetAmount(_ETH, Decimal("2")), AssetAmount(_TOKEN, Decimal("100"))],
        {_ETH: _quote("3000"), _TOKEN: None},
    )
    assert result.known_total_usd == UsdAmount(Decimal("6000"))
    assert len(result.priced) == 1
    assert [entry.balance.asset_id for entry in result.unpriced] == [_TOKEN]
    assert not result.is_complete


def test_known_zero_price_is_valued_zero_without_unpriced() -> None:
    """A true known USD zero is different from an absent quote."""
    result = value_current_balances([AssetAmount(_TOKEN, Decimal("123"))], {_TOKEN: _quote("0")})
    assert result.known_total_usd.value == 0
    assert result.is_complete
    assert not result.unpriced


def test_unknown_everywhere_yields_partial_zero_known_total() -> None:
    """Known total zero is not a claim the actual portfolio is worthless."""
    result = value_current_balances([AssetAmount(_TOKEN, Decimal("1"))], {})
    assert result.known_total_usd.value == 0
    assert not result.is_complete
    assert len(result.unpriced) == 1


def test_extreme_usd_magnitudes_sum_without_dropping_small_holding() -> None:
    """Do not silently round away a tiny position next to a huge one."""
    result = value_current_balances(
        [AssetAmount(_ETH, Decimal("1e100")), AssetAmount(_TOKEN, Decimal("1e-100"))],
        {_ETH: _quote("1"), _TOKEN: _quote("1")},
    )
    assert len(result.known_total_usd.value.as_tuple().digits) >= 201
    assert result.known_total_usd.value != Decimal("1e100")
    assert result.is_complete


def test_missing_metadata_is_preserved_and_marks_incomplete() -> None:
    """Raw E4.6 unresolved tokens remain visible in valuation coverage."""
    unresolved = UnresolvedTokenBalance(_UNKNOWN, 1_234_567, "metadata_unavailable")
    result = value_current_balances(
        [AssetAmount(_ETH, Decimal("1"))],
        {_ETH: _quote("3000")},
        unresolved=[unresolved],
    )
    assert result.known_total_usd == UsdAmount(Decimal("3000"))
    assert result.unresolved_balances == (unresolved,)
    assert not result.is_complete


def test_full_valuation_and_extra_quote_not_counted() -> None:
    """Only on-chain assets contribute to the known total."""
    result = value_current_balances(
        [AssetAmount(_ETH, Decimal("1")), AssetAmount(_TOKEN, Decimal("2"))],
        {_ETH: _quote("100"), _TOKEN: _quote("25"), _UNKNOWN: _quote("900000")},
    )
    assert result.is_complete
    assert result.known_total_usd.value == 150


def test_more_than_default_decimal_precision_is_preserved() -> None:
    """Currency multiplication/sum must not round to the default 28 digits."""
    original = getcontext().prec
    try:
        getcontext().prec = 12
        qty = Decimal("123456789123456789.123456789123456789")
        price = Decimal("0.000000000000000000123456789123456789")
        result = value_current_balances(
            [AssetAmount(_ETH, qty)], {_ETH: PriceQuote(UsdPrice(price), _AT, "fake")}
        )
        assert result.priced[0].value_usd.value == Decimal(
            "0.015241578780673678530864199530864199515622620750190521"
        )
        assert result.is_complete
    finally:
        getcontext().prec = original


@pytest.mark.parametrize(
    "balances",
    [
        [AssetAmount(_ETH, Decimal("1")), AssetAmount(_ETH, Decimal("2"))],
        [AssetAmount(_TOKEN, Decimal("-1"))],
    ],
)
def test_invalid_holdings_rejected(balances: list[AssetAmount]) -> None:
    """Duplicate and negative asset amounts cannot be used for valuation."""
    with pytest.raises(ValueError):
        value_current_balances(balances, {})


def test_asset_cannot_be_both_normalized_and_unresolved() -> None:
    """Reject contradictory provider state instead of silently double counting."""
    with pytest.raises(ValueError):
        value_current_balances(
            [AssetAmount(_TOKEN, Decimal("1"))],
            {},
            unresolved=[UnresolvedTokenBalance(_TOKEN, 100, "metadata_unavailable")],
        )


def test_quotes_must_be_typed() -> None:
    """Avoid floats or raw provider responses entering deterministic valuation."""
    with pytest.raises(TypeError):
        value_current_balances([AssetAmount(_ETH, Decimal("1"))], {_ETH: 10})  # type: ignore[dict-item]


class FakeCurrentPriceProvider:
    """A pure fake with no historical price implementation."""

    def __init__(self) -> None:
        """Prepare an empty call trace."""
        self.requested: list[AssetId] = []

    async def get_current_price(self, asset_id: AssetId) -> PriceQuote | None:
        """Return ETH price only and record the requested token identity."""
        self.requested.append(asset_id)
        return _quote("2000") if asset_id == _ETH else None


@pytest.mark.asyncio
async def test_application_uses_only_current_price_protocol() -> None:
    """No implementation of historical PriceProvider is required for current use."""
    fake = FakeCurrentPriceProvider()
    provider: CurrentPriceProvider = fake
    result = await price_current_balances(
        [AssetAmount(_ETH, Decimal("3")), AssetAmount(_TOKEN, Decimal("2"))],
        provider,
    )
    assert fake.requested == [_ETH, _TOKEN]
    assert result.known_total_usd.value == 6000
    assert [item.balance.asset_id for item in result.unpriced] == [_TOKEN]
    assert not result.is_complete


@pytest.mark.asyncio
async def test_application_omits_zero_quantity_price_requests() -> None:
    """No pointless API price lookup for a zero on-chain holding."""
    fake = FakeCurrentPriceProvider()
    result = await price_current_balances([AssetAmount(_TOKEN, Decimal(0))], fake)
    assert not fake.requested
    assert result.is_complete
    assert result.known_total_usd.value == 0


def test_invalid_valuation_input_types() -> None:
    """Reject non-model types instead of accidentally pricing raw vendor data."""
    with pytest.raises(TypeError):
        value_current_balances([], [])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        value_current_balances([1], {})  # type: ignore[list-item]
    with pytest.raises(TypeError):
        value_current_balances([], {}, unresolved=[None])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_async_valuation_rejects_invalid_balances_before_provider() -> None:
    """Invalid raw values never reach fake or external price providers."""
    fake = FakeCurrentPriceProvider()
    with pytest.raises(TypeError):
        await price_current_balances(["invalid"], fake)  # type: ignore[list-item]
    assert not fake.requested

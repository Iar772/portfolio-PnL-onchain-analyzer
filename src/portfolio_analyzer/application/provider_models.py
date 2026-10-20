"""Internal provider response models shared by application ports."""

from dataclasses import dataclass
from datetime import UTC, datetime

from portfolio_analyzer.domain.money import UsdPrice


@dataclass(frozen=True, slots=True)
class PriceQuote:
    """Represent a known USD price with provenance and observation time.

    A quote is evidence for valuation, not an instruction to infer prices when
    no quote is available. A missing quote is represented by None.

    :param price_usd: Known USD price per whole asset unit, including known zero.
    :param observed_at: Timezone-aware timestamp when the price was observed.
    :param source: Nonblank identifier of the source that supplied the price.
    :raises TypeError: If a field has an invalid runtime type.
    :raises ValueError: If timestamp or source is invalid.
    """

    price_usd: UsdPrice
    observed_at: datetime
    source: str

    def __post_init__(self) -> None:
        """Validate a vendor-neutral quote and normalize its provenance.

        :raises TypeError: For wrong price, timestamp, or source types.
        :raises ValueError: For a naive timestamp or blank source.
        """
        if not isinstance(self.price_usd, UsdPrice):
            raise TypeError("PriceQuote.price_usd must be a UsdPrice")

        if not isinstance(self.observed_at, datetime):
            raise TypeError("PriceQuote.observed_at must be a datetime")

        if self.observed_at.utcoffset() is None:
            raise ValueError("PriceQuote.observed_at must be timezone-aware")

        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))

        if not isinstance(self.source, str):
            raise TypeError("PriceQuote.source must be a string")

        normalized_source = self.source.strip()

        if not normalized_source:
            raise ValueError("PriceQuote.source must not be blank")

        object.__setattr__(self, "source", normalized_source)

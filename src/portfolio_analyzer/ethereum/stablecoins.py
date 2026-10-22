"""Trusted Ethereum Mainnet USD stablecoin contract allowlist."""

from portfolio_analyzer.domain.assets import AssetId
from portfolio_analyzer.domain.stablecoins import StablecoinRegistry, UsdStablecoin

MAINNET_USDC = AssetId(
    chain_id=1,
    contract_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
)
MAINNET_USDT = AssetId(
    chain_id=1,
    contract_address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
)

MAINNET_USD_STABLECOINS = StablecoinRegistry(
    (
        UsdStablecoin(asset_id=MAINNET_USDC, symbol="USDC"),
        UsdStablecoin(asset_id=MAINNET_USDT, symbol="USDT"),
    )
)

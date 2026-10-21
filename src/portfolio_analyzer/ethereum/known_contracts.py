"""Canonical Ethereum Mainnet contract identities independent of token symbols."""

from portfolio_analyzer.domain.assets import AssetId, AssetMetadata

MAINNET_WETH9 = AssetId(
    chain_id=1,
    contract_address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
)
MAINNET_WETH9_METADATA = AssetMetadata(
    symbol="WETH",
    name="Wrapped Ether",
    decimals=18,
)

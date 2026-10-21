"""Deterministic tests of two-leg swap normalization and conservative fallbacks."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from portfolio_analyzer.domain.assets import AssetId, AssetMetadata
from portfolio_analyzer.domain.events import Swap, UnknownEvent
from portfolio_analyzer.ethereum.movements import AssetMovement, MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
    group_movements_by_transaction,
)
from portfolio_analyzer.ethereum.rpc_models import RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.swap import classify_swap_v1
from portfolio_analyzer.ethereum.wallet_deltas import (
    RawAssetDelta,
    WalletAssetDelta,
    calculate_wallet_asset_delta,
)

WALLET = "0x" + "aa" * 20
ROUTER = "0x" + "bb" * 20
POOL = "0x" + "cc" * 20
OTHER = "0x" + "dd" * 20
ETH = AssetId(1, None)
LINK = AssetId(1, "0x" + "11" * 20)
USDC = AssetId(1, "0x" + "22" * 20)
LDO = AssetId(1, "0x" + "33" * 20)
BONUS = AssetId(1, "0x" + "44" * 20)
SAME_SYMBOL_AS_LINK = AssetId(1, "0x" + "55" * 20)
WETH = AssetId(1, "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")
TX_HASH = "0x" + "ef" * 32
AT = datetime(2026, 10, 21, 15, 40, tzinfo=UTC)
TOKEN_METADATA = {
    LINK: AssetMetadata("LINK", "Chainlink", 18),
    USDC: AssetMetadata("USDC", "USD Coin", 6),
    LDO: AssetMetadata("LDO", "Lido", 18),
    BONUS: AssetMetadata("BONUS", "Bonus", 18),
    WETH: AssetMetadata("WETH", "Wrapped Ether", 18),
    SAME_SYMBOL_AS_LINK: AssetMetadata("LINK", "Imitation LINK", 6),
}
SWAP_DATA = bytes.fromhex("12aa3caf") + bytes(128)
type Facts = tuple[TransactionMovementGroup, WalletAssetDelta, RpcTransaction, RpcReceipt]


def movement(
    name: str,
    asset: AssetId,
    amount: int,
    *,
    sender: str,
    receiver: str,
) -> AssetMovement:
    """Build one provider-neutral movement for a DEX-like transaction.

    :param name: Unique transfer identity.
    :param asset: Transferred Ethereum asset.
    :param amount: Unsigned integer base units.
    :param sender: Actual sender of transferred units.
    :param receiver: Actual recipient of transferred units.
    :return: Validated immutable movement fact.
    """
    return AssetMovement(
        transfer_id=name,
        tx_hash=TX_HASH,
        block_number=4321,
        timestamp=AT,
        from_address=sender,
        to_address=receiver,
        asset_id=asset,
        raw_amount=amount,
        category=MovementCategory.EXTERNAL if asset.is_native else MovementCategory.ERC20,
    )


def outgoing(asset: AssetId, raw_units: int, *, name: str = "out") -> AssetMovement:
    """Create one outgoing asset leg excluding any Ethereum execution gas.

    :param asset: Asset given up by the wallet.
    :param raw_units: Positive base-unit magnitude.
    :param name: Movement provenance ID.
    :return: Outgoing movement into a router/pool.
    """
    return movement(name, asset, raw_units, sender=WALLET, receiver=ROUTER)


def incoming(asset: AssetId, raw_units: int, *, name: str = "in") -> AssetMovement:
    """Create an incoming ERC-20/native movement from a pool.

    :param asset: Asset received by the wallet.
    :param raw_units: Positive base-unit magnitude.
    :param name: Movement provenance ID.
    :return: Incoming movement from an exchange pool.
    """
    return movement(name, asset, raw_units, sender=POOL, receiver=WALLET)


def scenario(
    *movements: AssetMovement,
    tx_from: str = WALLET,
    tx_to: str | None = ROUTER,
    tx_input: bytes = SWAP_DATA,
    tx_value: int = 0,
    success: bool | None = True,
    transaction_index: int | None = 5,
) -> Facts:
    """Make independently validated RPC facts for a single transaction.

    :param movements: Ordered transfers belonging to one transaction.
    :param tx_from: Original Ethereum transaction sender.
    :param tx_to: Called destination, or None for contract creation.
    :param tx_input: Original call data.
    :param tx_value: Sent native ETH excluding gas in wei.
    :param success: Receipt status, including legacy None.
    :param transaction_index: Optional canonical transaction ordering index.
    :return: Group, wallet delta, transaction and receipt.
    """
    group = group_movements_by_transaction(
        movements,
        transaction_indices={TX_HASH: transaction_index} if transaction_index is not None else None,
    )[0]
    delta = calculate_wallet_asset_delta(group, WALLET)
    tx = RpcTransaction(TX_HASH, tx_from, tx_to, 4321, tx_value, tx_input)
    receipt = RpcReceipt(TX_HASH, 4321, success, 84000, 2 * 10**9, ())
    return group, delta, tx, receipt


def classify(
    facts: Facts,
    *,
    metadata: dict[AssetId, AssetMetadata] | None = None,
) -> Swap | UnknownEvent | None:
    """Invoke the subject classifier with supplied trusted metadata.

    :param facts: Matching normalized transaction and RPC facts.
    :param metadata: Optional metadata override.
    :return: Domain Swap, UnknownEvent, or None.
    """
    return classify_swap_v1(*facts, token_metadata=TOKEN_METADATA if metadata is None else metadata)


@pytest.mark.parametrize(
    ("asset_out", "raw_out", "quantity_out"),
    [
        (USDC, 100_000_000, "100"),
        (ETH, 500_000_000_000_000_000, "0.5"),
        (LDO, 40 * 10**18, "40"),
    ],
)
def test_one_out_one_in_creates_same_swap_domain_model(
    asset_out: AssetId, raw_out: int, quantity_out: str
) -> None:
    """USDC, ETH and LDO purchases of LINK share the same Swap type."""
    facts = scenario(
        outgoing(asset_out, raw_out),
        incoming(LINK, 50 * 10**18),
        tx_value=raw_out if asset_out.is_native else 0,
    )
    result = classify(facts)
    assert isinstance(result, Swap)
    assert result.amount_out.asset_id == asset_out
    assert result.amount_out.quantity == Decimal(quantity_out)
    assert result.amount_in.asset_id == LINK
    assert result.amount_in.quantity == Decimal("50")
    assert result.metadata.tx_hash == TX_HASH
    assert result.metadata.block_number == 4321
    assert result.metadata.timestamp == AT
    assert not hasattr(result, "proceeds")
    assert not hasattr(result, "cost_basis")
    assert not hasattr(result, "gas")


def test_arrival_order_of_legs_does_not_determine_trade_direction() -> None:
    """Spend/receive signs, not transfer ordering, define the swap sides."""
    facts = scenario(incoming(LINK, 2 * 10**18), outgoing(USDC, 7_000_000))
    result = classify(facts)
    assert isinstance(result, Swap)
    assert result.amount_out.quantity == Decimal("7")
    assert result.amount_in.quantity == Decimal("2")


def test_irrelevant_pool_to_router_transfers_do_not_invent_wallet_legs() -> None:
    """Router internal ERC-20 hops may exist beside two wallet deltas."""
    facts = scenario(
        outgoing(USDC, 10_000_000),
        movement("middle", BONUS, 50 * 10**18, sender=ROUTER, receiver=POOL),
        incoming(LINK, 5 * 10**18),
    )
    result = classify(facts)
    assert isinstance(result, Swap)
    assert result.amount_out.quantity == Decimal("10")


def test_native_received_for_token_out_can_be_a_simple_two_leg_swap() -> None:
    """Token-to-ETH route is also one out and one in, excluding gas."""
    facts = scenario(outgoing(USDC, 50_000_000), incoming(ETH, 10**17))
    result = classify(facts)
    assert isinstance(result, Swap)
    assert result.amount_in.quantity == Decimal("0.1")


def test_two_different_tokens_with_identical_symbols_are_distinct() -> None:
    """AssetId, never displayed symbol, determines the trade pair."""
    facts = scenario(outgoing(SAME_SYMBOL_AS_LINK, 2_500_000), incoming(LINK, 10**18))
    result = classify(facts)
    assert isinstance(result, Swap)
    assert result.amount_out.asset_id != result.amount_in.asset_id
    assert result.amount_out.quantity == Decimal("2.5")


def test_native_eth_and_erc20_decimals_are_not_inferred_from_hints() -> None:
    """Amounts use 18 native decimals and verified token metadata only."""
    out = replace(outgoing(USDC, 1_000_001), decimals_hint=18)
    incoming_link = replace(incoming(LINK, 10**18), decimals_hint=6)
    result = classify(scenario(out, incoming_link))
    assert isinstance(result, Swap)
    assert result.amount_out.quantity == Decimal("1.000001")
    assert result.amount_in.quantity == Decimal("1")


def test_huge_raw_values_are_not_rounded_by_decimal_context() -> None:
    """Large amounts retain exact units even with default Decimal precision."""
    enormous = int("123456789012345678901234567890123456789")
    result = classify(scenario(outgoing(USDC, enormous), incoming(LINK, 10**18)))
    assert isinstance(result, Swap)
    assert result.amount_out.quantity == Decimal(f"{enormous}e-6")


def test_gas_from_receipt_is_not_counted_as_eth_spent() -> None:
    """The ETH outgoing leg excludes the receipt's gas-used expenditure."""
    wei = 10**18
    facts = scenario(outgoing(ETH, wei), incoming(LINK, 4 * 10**18), tx_value=wei)
    result = classify(facts)
    assert isinstance(result, Swap)
    assert result.amount_out.quantity == Decimal("1")
    gas_price = facts[3].effective_gas_price_wei
    assert gas_price is not None
    assert facts[3].gas_used * gas_price != 0


@pytest.mark.parametrize(
    "more_movements",
    [
        (
            outgoing(USDC, 100_000_000),
            incoming(LINK, 10**18),
            incoming(BONUS, 10**18, name="bonus"),
        ),
        (
            outgoing(USDC, 100_000_000),
            outgoing(LDO, 10**18, name="out_ldo"),
            incoming(LINK, 10**18),
        ),
        (
            incoming(USDC, 100_000_000, name="in_usdc"),
            incoming(LINK, 10**18),
            incoming(BONUS, 10**18, name="bonus"),
        ),
    ],
)
def test_multi_asset_transactions_are_unknown(more_movements: tuple[AssetMovement, ...]) -> None:
    """More than two wallet assets cannot be compressed to an invented Swap."""
    result = classify(scenario(*more_movements))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "multi_asset_transaction"
    assert len(result.movements) == 3
    assert result.metadata.tx_hash == TX_HASH


def test_dust_third_asset_still_counts_as_ambiguous() -> None:
    """No arbitrary USD threshold silently drops a small third trade leg."""
    result = classify(
        scenario(outgoing(USDC, 1_000_000), incoming(LINK, 10**18), incoming(BONUS, 1, name="dust"))
    )
    assert isinstance(result, UnknownEvent)
    assert any(change.quantity == Decimal("0.000000000000000001") for change in result.movements)


@pytest.mark.parametrize("sign", ["positive", "negative"])
def test_two_same_direction_assets_are_unknown(sign: str) -> None:
    """Two credits or debits are not a two-sided exchange."""
    facts = (
        scenario(incoming(LINK, 10**18), incoming(BONUS, 10**18, name="bonus"))
        if sign == "positive"
        else scenario(outgoing(USDC, 10**6), outgoing(LDO, 10**18, name="ldo"))
    )
    result = classify(facts)
    assert isinstance(result, UnknownEvent)
    assert result.reason == "not_one_in_one_out"


@pytest.mark.parametrize(
    "movements",
    [
        (incoming(LINK, 10**18),),
        (outgoing(USDC, 1_000_000),),
        (
            outgoing(USDC, 1_000_000),
            incoming(USDC, 1_000_000),
        ),
        (movement("self", LINK, 10**18, sender=WALLET, receiver=WALLET),),
    ],
)
def test_single_leg_or_net_zero_is_reserved_for_other_classifiers(
    movements: tuple[AssetMovement, ...],
) -> None:
    """A transfer or self-transfer must never become a synthetic swap."""
    assert classify(scenario(*movements)) is None


def test_simple_pair_without_received_metadata_is_unknown() -> None:
    """Unknown decimals do not create fake quantities or zero-cost inventory."""
    facts = scenario(outgoing(USDC, 1_000_000), incoming(LINK, 10**18))
    result = classify(facts, metadata={USDC: TOKEN_METADATA[USDC]})
    assert isinstance(result, UnknownEvent)
    assert result.reason == "missing_asset_metadata"
    assert len(result.movements) == 1
    assert result.movements[0].quantity == Decimal("-1")
    assert facts[1].changes[1].raw_units == 10**18


def test_multi_asset_unknown_preserves_known_parts_without_missing_metadata() -> None:
    """Unknown asset decimals only omit that normalized leg, not the event."""
    facts = scenario(
        outgoing(USDC, 10**6),
        incoming(LINK, 10**18),
        incoming(BONUS, 123, name="extra"),
    )
    result = classify(facts, metadata={USDC: TOKEN_METADATA[USDC]})
    assert isinstance(result, UnknownEvent)
    assert result.reason == "multi_asset_transaction"
    assert len(result.movements) == 1
    assert result.movements[0].quantity == Decimal("-1")


def test_corrupted_metadata_type_fails_explicitly() -> None:
    """Do not silently accept invalid metadata types in trusted lookup."""
    facts = scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18))
    invalid: dict[AssetId, AssetMetadata] = {USDC: "bad"}  # type: ignore[dict-item]
    with pytest.raises(EthereumNormalizationError, match="token metadata"):
        classify(facts, metadata=invalid)


@pytest.mark.parametrize("success", [False, None])
def test_failed_or_status_unknown_receipt_is_not_swap(success: bool | None) -> None:
    """No Swap may be emitted without a positively successful receipt."""
    result = classify(scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18), success=success))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "swap_evidence_insufficient"


@pytest.mark.parametrize("origin", [ROUTER, POOL])
def test_transaction_not_initiated_by_wallet_is_unknown(origin: str) -> None:
    """Wallet credits/debits made by a third party need separate analysis."""
    result = classify(scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18), tx_from=origin))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "swap_evidence_insufficient"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        bytes.fromhex("a9059cbb") + bytes(64),
        bytes.fromhex("23b872dd") + bytes(96),
        bytes.fromhex("095ea7b3") + bytes(64),
    ],
)
def test_plain_transfer_approve_and_empty_call_are_not_swap_evidence(data: bytes) -> None:
    """ERC-20 transfer/approve calldata does not establish exchange intent."""
    result = classify(scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18), tx_input=data))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "swap_evidence_insufficient"


def test_contract_creation_transaction_is_unknown() -> None:
    """Contract deployment plus movements is not inferred to be a swap."""
    result = classify(scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18), tx_to=None))
    assert isinstance(result, UnknownEvent)


@pytest.mark.parametrize("is_wrap", [True, False])
def test_eth_weth_pair_is_reserved_for_wrap_classifier(is_wrap: bool) -> None:
    """Canonical ETH↔WETH has basis-preserving semantics, not swap PnL."""
    if is_wrap:
        facts = scenario(outgoing(ETH, 10**18), incoming(WETH, 10**18), tx_value=10**18)
    else:
        facts = scenario(outgoing(WETH, 10**18), incoming(ETH, 10**18))
    result = classify(facts)
    assert isinstance(result, UnknownEvent)
    assert result.reason == "possible_wrap_or_unwrap"


def test_native_spend_without_matching_transaction_value_is_unknown() -> None:
    """Native wallet spend must equal sent tx.value, without mixing gas."""
    result = classify(scenario(outgoing(ETH, 10**18), incoming(LINK, 10**18), tx_value=10**18 - 1))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "native_value_mismatch"


def test_token_swap_with_unmatched_native_sent_value_is_unknown() -> None:
    """Token-to-token contract calls cannot hide native ETH output."""
    result = classify(scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18), tx_value=1))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "unmatched_native_value"


def test_movement_provenance_preserves_transaction_index_in_group() -> None:
    """Classification leaves independently proven intra-block ordering intact."""
    facts = scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18), transaction_index=7)
    assert facts[0].transaction_index == 7
    assert facts[1].transaction_index == 7
    assert isinstance(classify(facts), Swap)
    assert facts[0].transaction_index == 7


def test_missing_transaction_index_does_not_block_single_transaction_classification() -> None:
    """Per-transaction classification does not invent ordering between txs."""
    facts = scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18), transaction_index=None)
    assert facts[0].transaction_index is None
    assert isinstance(classify(facts), Swap)


@pytest.mark.parametrize(
    ("position", "replacement"),
    [
        (0, "bad"),
        (1, "bad"),
        (2, "bad"),
        (3, "bad"),
    ],
)
def test_top_level_argument_types_are_checked(position: int, replacement: str) -> None:
    """Bad inputs fail clearly rather than becoming silent UnknownEvent."""
    arguments = list(scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18)))
    arguments[position] = replacement
    with pytest.raises(TypeError):
        classify_swap_v1(*arguments, token_metadata=TOKEN_METADATA)  # type: ignore[arg-type]


def test_mapping_type_is_checked() -> None:
    """Caller must supply a metadata mapping, not an arbitrary iterable."""
    facts = scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18))
    with pytest.raises(TypeError, match="mapping"):
        classify_swap_v1(*facts, token_metadata=[])  # type: ignore[arg-type]


@pytest.mark.parametrize("wrong_field", ["hash", "block"])
def test_conflicting_rpc_receipt_provenance_is_rejected(wrong_field: str) -> None:
    """Mismatched receipt facts are not treated as an ordinary unknown swap."""
    group, delta, tx, receipt = scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18))
    invalid_receipt = (
        replace(receipt, tx_hash="0x" + "10" * 32)
        if wrong_field == "hash"
        else replace(receipt, block_number=9000)
    )
    with pytest.raises(EthereumNormalizationError):
        classify((group, delta, tx, invalid_receipt))


def test_tampered_wallet_net_delta_is_rejected() -> None:
    """Never classify externally supplied deltas differing from source movements."""
    group, delta, tx, receipt = scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18))
    modified = replace(delta, changes=(RawAssetDelta(USDC, -2 * 10**6), delta.changes[1]))
    with pytest.raises(EthereumNormalizationError, match="disagree"):
        classify((group, modified, tx, receipt))


def test_wrong_wallet_relative_delta_is_rejected() -> None:
    """A forged wallet address cannot reuse the original wallet's legs."""
    group, delta, tx, receipt = scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18))
    modified = replace(delta, wallet_address=OTHER)
    with pytest.raises(EthereumNormalizationError, match="disagree"):
        classify((group, modified, tx, receipt))


def test_multi_asset_unknown_has_same_provenance_as_swap() -> None:
    """Complex transactions preserve reliable hash/block/time for audit."""
    result = classify(
        scenario(
            outgoing(USDC, 10**6),
            incoming(LINK, 10**18),
            incoming(BONUS, 10**18, name="c"),
        )
    )
    assert isinstance(result, UnknownEvent)
    assert result.metadata.timestamp == AT
    assert result.metadata.block_number == 4321
    assert result.metadata.tx_hash == TX_HASH


def test_swap_classification_is_deterministic_and_side_effect_free() -> None:
    """Identical inputs create equal economic events and do not mutate facts."""
    facts = scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18))
    first = classify(facts)
    second = classify(facts)
    assert first == second
    assert facts[1].changes[0].raw_units == -(10**6)
    assert facts[1].changes[1].raw_units == 10**18


def test_same_net_pair_from_three_wallet_movements_is_unknown() -> None:
    """Two outgoing USDC transfers are not a simple single outgoing leg."""
    facts = scenario(
        outgoing(USDC, 4_000_000, name="part1"),
        outgoing(USDC, 6_000_000, name="part2"),
        incoming(LINK, 10**18),
    )
    result = classify(facts)
    assert isinstance(result, UnknownEvent)
    assert result.reason == "multiple_wallet_movements"
    assert len(result.movements) == 2
    assert result.movements[0].quantity == Decimal("-10")


def test_extra_wallet_self_transfer_is_not_ignored_as_proven_simple_swap() -> None:
    """A wallet's extra self-transfer can hide a complex operation."""
    facts = scenario(
        outgoing(USDC, 1_000_000),
        incoming(LINK, 10**18),
        movement("self", BONUS, 12, sender=WALLET, receiver=WALLET),
    )
    result = classify(facts)
    assert isinstance(result, UnknownEvent)
    assert result.reason == "multiple_wallet_movements"


def test_zero_value_extra_movement_is_not_a_meaningful_swap_leg() -> None:
    """Zero-unit log entries do not create an extra economic wallet movement."""
    facts = scenario(
        outgoing(USDC, 1_000_000),
        incoming(LINK, 10**18),
        movement("zero", BONUS, 0, sender=WALLET, receiver=OTHER),
    )
    assert isinstance(classify(facts), Swap)


def test_ethereum_transfer_destination_must_match_rpc_target() -> None:
    """A wallet's outgoing native ETH transfer must match RPC tx.value and to."""
    facts = scenario(outgoing(ETH, 10**18), incoming(LINK, 10**18), tx_to=OTHER, tx_value=10**18)
    result = classify(facts)
    assert isinstance(result, UnknownEvent)
    assert result.reason == "native_value_mismatch"


def test_native_internal_movement_cannot_be_a_plain_wallet_sent_value() -> None:
    """An internal ETH debit is not transaction.value for the originating EOA."""
    eth = replace(outgoing(ETH, 10**18), category=MovementCategory.INTERNAL)
    result = classify(scenario(eth, incoming(LINK, 10**18), tx_value=10**18))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "native_value_mismatch"


def test_minted_token_received_is_not_assumed_to_be_a_market_swap() -> None:
    """Receipt tokens minted from zero address may represent staking/bridging."""
    minted = movement("mint", LINK, 10**18, sender="0x" + "00" * 20, receiver=WALLET)
    result = classify(scenario(outgoing(USDC, 10**6), minted))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "possible_mint_or_burn"


def test_burned_token_sent_is_not_assumed_to_be_a_market_swap() -> None:
    """An ERC-20 burn plus incoming asset may be a bridge or redemption."""
    burned = movement("burn", USDC, 10**6, sender=WALLET, receiver="0x" + "00" * 20)
    result = classify(scenario(burned, incoming(LINK, 10**18)))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "possible_mint_or_burn"


@pytest.mark.parametrize("data", [b"x", b"xy", b"xyz"])
def test_too_short_calldata_is_not_verified_contract_swap(data: bytes) -> None:
    """A selector-less arbitrary payload is insufficient swap evidence."""
    result = classify(scenario(outgoing(USDC, 10**6), incoming(LINK, 10**18), tx_input=data))
    assert isinstance(result, UnknownEvent)
    assert result.reason == "swap_evidence_insufficient"

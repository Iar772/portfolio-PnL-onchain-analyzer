"""Conservative v1 classification of two-asset Ethereum wallet swaps."""

from collections.abc import Mapping

from portfolio_analyzer.domain.assets import AssetAmount, AssetId, AssetMetadata
from portfolio_analyzer.domain.events import EventMetadata, Swap, UnknownEvent
from portfolio_analyzer.ethereum.direct_transfer_rules import (
    _NATIVE_METADATA,
    _assert_same_transaction,
)
from portfolio_analyzer.ethereum.movements import MovementCategory
from portfolio_analyzer.ethereum.normalizer import (
    EthereumNormalizationError,
    TransactionMovementGroup,
)
from portfolio_analyzer.ethereum.rpc_models import RpcReceipt, RpcTransaction
from portfolio_analyzer.ethereum.wallet_deltas import (
    WalletAssetDelta,
    calculate_wallet_asset_delta,
)

# Contract identity, NOT ticker text: the canonical WETH9 contract on mainnet.
_WETH9 = AssetId(chain_id=1, contract_address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")
_NATIVE_ETH = AssetId(chain_id=1, contract_address=None)
_ZERO_ADDRESS = "0x" + "00" * 20
_DIRECT_ERC20_TRANSFER = bytes.fromhex("a9059cbb")
_DIRECT_ERC20_TRANSFER_FROM = bytes.fromhex("23b872dd")
_APPROVE_ERC20 = bytes.fromhex("095ea7b3")
_NOT_SWAP_SELECTORS = frozenset(
    (_DIRECT_ERC20_TRANSFER, _DIRECT_ERC20_TRANSFER_FROM, _APPROVE_ERC20)
)


def _event_metadata(group: TransactionMovementGroup) -> EventMetadata:
    """Create shared economic-event provenance from a verified group.

    :param group: A mined Ethereum transaction and its movement facts.
    :return: Immutable event metadata without inferred execution values.
    """
    return EventMetadata(
        tx_hash=group.transaction.tx_hash,
        block_number=group.transaction.block_number,
        timestamp=group.transaction.timestamp,
    )


def _asset_metadata(
    asset_id: AssetId, token_metadata: Mapping[AssetId, AssetMetadata]
) -> AssetMetadata | None:
    """Resolve trusted decimals, never deriving them from a token symbol.

    :param asset_id: Network and contract-based asset identity.
    :param token_metadata: Trusted ERC-20 metadata keyed by identity.
    :return: Native ETH or verified ERC-20 metadata, or None.
    :raises EthereumNormalizationError: For invalid supplied metadata.
    """
    if asset_id.is_native:
        return _NATIVE_METADATA
    value = token_metadata.get(asset_id)
    if value is not None and not isinstance(value, AssetMetadata):
        raise EthereumNormalizationError("token metadata must be AssetMetadata")
    return value


def _unknown(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    token_metadata: Mapping[AssetId, AssetMetadata],
    *,
    reason: str,
) -> UnknownEvent:
    """Keep known signed movements without fabricating missing token decimals.

    Raw quantities remain available on the caller's WalletAssetDelta even
    if some UnknownEvent.movements cannot be normalized yet.

    :param group: Mined transaction provenance.
    :param delta: Exact signed changes in raw base units.
    :param token_metadata: Verified metadata for normalizable changes.
    :param reason: Concise statement of classification uncertainty.
    :return: UnknownEvent with only trustworthy normalized legs.
    :raises EthereumNormalizationError: For invalid supplied metadata.
    """
    known = []
    for change in delta.changes:
        metadata = _asset_metadata(change.asset_id, token_metadata)
        if metadata is not None:
            known.append(change.to_asset_amount(metadata))
    return UnknownEvent(
        metadata=_event_metadata(group),
        reason=reason,
        movements=tuple(known),
    )


def classify_swap_v1(
    group: TransactionMovementGroup,
    delta: WalletAssetDelta,
    transaction: RpcTransaction,
    receipt: RpcReceipt,
    *,
    token_metadata: Mapping[AssetId, AssetMetadata],
) -> Swap | UnknownEvent | None:
    """Classify a plausible wallet-initiated one-out/one-in contract swap.

    Requires successful execution, nonempty contract calldata, a wallet-origin
    transaction, and **exactly** one positive and one negative distinct asset
    net delta. Pure single-leg movements return None for TransferIn/Out
    classifiers. Multi-asset and ambiguous two-leg actions become UnknownEvent.
    A canonical ETH/WETH pair is reserved for future Wrap/Unwrap handling.

    This deliberately identifies *simple swap shapes*, not a proof of DEX
    execution: for example a staking or liquidity operation can resemble a
    two-leg swap. Future specialized classifiers must take precedence whenever
    receipt logs/protocol details establish a different economic operation.
    Gas is NOT part of wallet asset deltas, event amounts, or trade legs.
    USD proceeds and cost basis are never inferred here.

    :param group: One mined transaction with provider-neutral movement facts.
    :param delta: Its exact wallet-relative changes, excluding gas.
    :param transaction: Mined Ethereum RPC transaction facts.
    :param receipt: Its provider-neutral execution receipt.
    :param token_metadata: Trusted ERC-20 metadata keyed by AssetId.
    :return: Swap, UnknownEvent or None for a single-leg transaction.
    :raises TypeError: For invalid top-level argument types.
    :raises EthereumNormalizationError: For contradictory transaction facts.
    """
    if not isinstance(group, TransactionMovementGroup):
        raise TypeError("group must be a TransactionMovementGroup")
    if not isinstance(delta, WalletAssetDelta):
        raise TypeError("delta must be a WalletAssetDelta")
    if not isinstance(transaction, RpcTransaction):
        raise TypeError("transaction must be a RpcTransaction")
    if not isinstance(receipt, RpcReceipt):
        raise TypeError("receipt must be a RpcReceipt")
    if not isinstance(token_metadata, Mapping):
        raise TypeError("token_metadata must be a mapping")

    _assert_same_transaction(group, delta, transaction, receipt)
    # Identity checks alone do not prove that caller-supplied deltas match
    # the movements; never classify a stale/partial derived delta as a swap.
    if calculate_wallet_asset_delta(group, delta.wallet_address).changes != delta.changes:
        raise EthereumNormalizationError("wallet deltas disagree with transaction movements")

    if len(delta.changes) <= 1:
        return None
    if len(delta.changes) != 2:
        return _unknown(group, delta, token_metadata, reason="multi_asset_transaction")

    outgoing = [change for change in delta.changes if change.raw_units < 0]
    incoming = [change for change in delta.changes if change.raw_units > 0]
    if len(outgoing) != 1 or len(incoming) != 1:
        return _unknown(group, delta, token_metadata, reason="not_one_in_one_out")

    spent = outgoing[0]
    received = incoming[0]
    if {spent.asset_id, received.asset_id} == {_NATIVE_ETH, _WETH9}:
        return _unknown(group, delta, token_metadata, reason="possible_wrap_or_unwrap")

    # v1 requires two *observed* wallet movements, not merely two net assets:
    # several transfers of the same token could encode a split route, refund,
    # payout or another complex operation with the same final net shape.
    wallet = delta.wallet_address
    wallet_movements = [
        movement
        for movement in group.movements
        if movement.raw_amount > 0
        and (movement.from_address == wallet or movement.to_address == wallet)
    ]
    if len(wallet_movements) != 2:
        return _unknown(group, delta, token_metadata, reason="multiple_wallet_movements")
    observed_spent = next(
        (
            movement
            for movement in wallet_movements
            if movement.from_address == wallet and movement.to_address != wallet
        ),
        None,
    )
    observed_received = next(
        (
            movement
            for movement in wallet_movements
            if movement.to_address == wallet and movement.from_address != wallet
        ),
        None,
    )
    if (
        observed_spent is None
        or observed_received is None
        or observed_spent.asset_id != spent.asset_id
        or observed_received.asset_id != received.asset_id
        or observed_spent.raw_amount != -spent.raw_units
        or observed_received.raw_amount != received.raw_units
    ):
        return _unknown(group, delta, token_metadata, reason="wallet_leg_mismatch")

    # Mint/burn and bridge-style token issuance are not ordinary market swaps.
    if (
        observed_spent.to_address == _ZERO_ADDRESS
        or observed_received.from_address == _ZERO_ADDRESS
    ):
        return _unknown(group, delta, token_metadata, reason="possible_mint_or_burn")

    if (
        receipt.success is not True
        or transaction.from_address.lower() != delta.wallet_address
        or transaction.to_address is None
        or len(transaction.input_data) < 4
        or transaction.input_data[:4] in _NOT_SWAP_SELECTORS
    ):
        return _unknown(group, delta, token_metadata, reason="swap_evidence_insufficient")

    # Native ETH spent in a wallet-originated contract call must be accounted
    # for by tx.value (not by gas, which is excluded). Refunded ETH or more
    # complex flows cannot be assumed to be a simple v1 swap.
    if spent.asset_id.is_native:
        if (
            transaction.value_wei != -spent.raw_units
            or observed_spent.category is not MovementCategory.EXTERNAL
            or observed_spent.to_address != transaction.to_address.lower()
        ):
            return _unknown(group, delta, token_metadata, reason="native_value_mismatch")
    elif transaction.value_wei != 0:
        return _unknown(group, delta, token_metadata, reason="unmatched_native_value")

    spent_metadata = _asset_metadata(spent.asset_id, token_metadata)
    received_metadata = _asset_metadata(received.asset_id, token_metadata)
    if spent_metadata is None or received_metadata is None:
        return _unknown(group, delta, token_metadata, reason="missing_asset_metadata")

    return Swap(
        metadata=_event_metadata(group),
        amount_out=AssetAmount.from_raw_units(spent.asset_id, -spent.raw_units, spent_metadata),
        amount_in=received.to_asset_amount(received_metadata),
    )

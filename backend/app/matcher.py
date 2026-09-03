"""Deterministic reconciliation engine.

Four passes, cheapest and most certain first, so the LLM is only ever
asked to resolve what the rules genuinely cannot:

  1. exact   - settlement UTR found verbatim in a single bank narration,
               amount matches to the paisa.
  2. fuzzy   - UTR found verbatim but amount differs by a plausible
               Razorpay fee (<=3%), or settlement date drifted.
  3. group   - no single bank line covers the settlement, but two or three
               bank lines that all reference the same UTR sum to it exactly
               (a partial early payout + balance, or a fee leg booked
               separately). Many-to-one, still fully deterministic -- the
               shared UTR is what ties the group together, not a guess.
  4. llm     - no settlement's UTR appears in any remaining bank line at all
               (garbled/truncated narration) -> handed to the LLM resolver,
               which must reason over free text, not just string-match.

Anything left after all four passes is an honest exception with a
reason code, never silently dropped.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from itertools import combinations
from typing import Any

FEE_TOLERANCE_PCT = 0.03
DATE_DRIFT_OK_DAYS = 3
MAX_GROUP_SIZE = 3


def _parse_date(s: str) -> date:
    return datetime.fromisoformat(s).date() if "T" not in s else datetime.fromisoformat(s).date()


def _net_refund_for_settlement(order: dict, stl: dict, order_refunds: list[dict]) -> tuple[dict, float, float]:
    """A refund only shrinks what a settlement is expected to pay out if it
    happened at or before that settlement -- Razorpay nets a pre-settlement
    refund out of the payout amount. A refund issued *after* the settlement
    is a separate, later cash event: the settlement was genuinely
    legitimate at its full amount, and a later refund must never make it
    look short. Returns (match_order, pre_settlement_refund, post_settlement_refund).
    """
    settled_at = _parse_date(stl["settled_at"])
    pre = round(sum(r["amount"] for r in order_refunds
                     if _parse_date(r["refunded_at"]) <= settled_at), 2)
    post = round(sum(r["amount"] for r in order_refunds
                      if _parse_date(r["refunded_at"]) > settled_at), 2)
    if pre == 0:
        return order, 0.0, post
    net_amount = round(max(order["amount"] - pre, 0.0), 2)
    return {"amount": net_amount, "created_at": order["created_at"]}, pre, post


def _refund_suffix(pre: float, post: float) -> str:
    if post > 0:
        return (f" (settlement is full and legitimate; order was refunded "
                f"₹{post:,.2f} afterward -- tracked as a separate event, not "
                f"netted against this settlement)")
    if pre > 0:
        return f" (net of ₹{pre:,.2f} pre-settlement refund)"
    return ""


INSTANT_SETTLEMENT_FEE_TOLERANCE_PCT = 0.05  # Razorpay's Instant Settlement product pays
# out on demand instead of on the standard T+2 cycle, and charges an additional
# convenience fee on top of the normal processing fee for that -- a genuinely
# higher deduction, not a mismatch. The exact real-world instant-settlement fee
# isn't independently confirmed here, so this is a deliberately generous (but
# still bounded, and kept under amount_mismatch's 6% floor -- see data_gen.py)
# approximation rather than a specific claimed percentage.


def _find_settlement_match(stl: dict, order: dict, utr_matches: list[dict],
                            fee_tolerance_pct: float, date_drift_ok_days: int) -> tuple | None:
    """Try to resolve one settlement against the bank lines that already
    reference its UTR (`utr_matches`). Returns (method, confidence, note,
    bank_line_ids) or None if nothing deterministic fits -- in which case
    the caller falls back to the LLM pass.

    An instant settlement (stl.get("is_instant")) gets a wider fee
    tolerance than a standard one -- its extra on-demand-payout fee is a
    real, legitimate deduction, not a sign of a genuine amount problem,
    and the standard 3% tolerance would risk misflagging it.
    """
    effective_fee_tolerance_pct = (INSTANT_SETTLEMENT_FEE_TOLERANCE_PCT
                                    if stl.get("is_instant") else fee_tolerance_pct)
    amount_diff_pct = abs(stl["amount"] - order["amount"]) / order["amount"]
    date_drift = abs((_parse_date(stl["settled_at"]) - _parse_date(order["created_at"])).days)

    single = next((b for b in utr_matches if b["amount"] == stl["amount"]), None)
    if single:
        if amount_diff_pct < 0.001:
            if date_drift <= date_drift_ok_days:
                return ("exact", 1.0, "UTR + amount matched exactly", [single["line_id"]])
            return ("fuzzy", 0.85,
                    f"UTR + amount matched exactly; settlement date drifted {date_drift}d "
                    f"beyond the {date_drift_ok_days}d policy window", [single["line_id"]])
        if amount_diff_pct <= effective_fee_tolerance_pct:
            tolerance_note = ("within instant-settlement fee tolerance" if stl.get("is_instant")
                               else "within fee tolerance")
            return ("fuzzy", 0.9,
                    f"UTR matched; amount differs by {amount_diff_pct:.1%}, {tolerance_note}",
                    [single["line_id"]])
        # a single line references this UTR but the amount is genuinely
        # wrong -- don't let a lucky group-sum below paper over that below;
        # the caller handles this as amount_mismatch, not a group match.
        return None

    if len(utr_matches) >= 2:
        for size in range(2, min(MAX_GROUP_SIZE, len(utr_matches)) + 1):
            for combo in combinations(utr_matches, size):
                total = round(sum(b["amount"] for b in combo), 2)
                if abs(total - stl["amount"]) <= 0.01:
                    return ("group_split", 0.95,
                            f"Settlement UTR not found on a single bank line, but {size} bank "
                            f"lines sharing that UTR sum exactly to the settlement amount",
                            [b["line_id"] for b in combo])
    return None


def _refund_key(r: dict) -> str:
    """refund_id is always present for synthetic data but is an optional
    upload column (see REQUIRED_REFUND_FIELDS in main.py, kept backward
    compatible with the existing payment_id/amount-only refunds.csv
    schema) -- fall back to a payment_id+amount composite so uploaded
    refunds without it still get a stable, matchable key instead of a
    KeyError.
    """
    return r.get("refund_id") or f"{r['payment_id']}_{r['amount']}"


def _match_refund_debits(refunds: list[dict], unmatched_bank: dict[str, dict],
                          log) -> tuple[list[dict], list[dict]]:
    """A refund record proves money was *promised* back to a customer, not
    that it actually left the account -- those are two different facts,
    and a books-closing decision needs the second one, not just the
    first. This looks for an outbound (negative-amount) bank line whose
    narration references the refund and whose magnitude matches, exactly
    the same "does the bank confirm this" standard settlement matching
    already applies to inbound money. A refund with no matching debit is
    a real cash-position risk (pending payout, or a genuine error) and is
    surfaced as an honest exception with the refund's own amount, not
    silently assumed to have gone out.
    """
    matches, exceptions = [], []
    for r in refunds:
        key = _refund_key(r)
        hit = next((b for b in unmatched_bank.values()
                    if b["amount"] < 0 and abs(-b["amount"] - r["amount"]) < 0.01
                    and key in b["narration"]), None)
        if hit:
            unmatched_bank.pop(hit["line_id"], None)
            matches.append({
                "refund_id": key, "payment_id": r["payment_id"],
                "bank_line_id": hit["line_id"], "amount": r["amount"],
                "method": "refund_debit_matched",
                "note": f"Refund confirmed debited from the bank on {hit['value_date']}.",
            })
            log("refund_debit_match", "matched", "rule", 1.0,
                f"Refund {key} (Rs {r['amount']:,.2f}) confirmed debited from the bank.",
                refund_id=key)
        else:
            exceptions.append({
                "type": "refund_not_debited", "refund_id": key,
                "payment_id": r["payment_id"], "amount": r["amount"],
                "reason": f"Refund {key} for Rs {r['amount']:,.2f} is recorded but no "
                          "matching outbound bank debit was found -- the money may not have "
                          "actually left the account yet.",
            })
            log("refund_debit_match", "exception", "rule", 0.0,
                f"No outbound bank debit found for refund {key}.", refund_id=key)
    return matches, exceptions


def _check_over_refunds(orders: list[dict], refunds: list[dict], log) -> list[dict]:
    """A refund record proves money was promised back, logically capped at
    the order's own value -- refunds summing to more than the order ever
    charged is a data-quality or fraud signal, not something the
    settlement-matching passes are positioned to catch (each of those
    only ever sees one settlement candidate at a time, not every refund
    for that payment summed together). Deliberately independent of
    settlement matching: this never touches `matches`, so a legitimate
    settlement -- including one a post-settlement refund correctly left
    unshrunk, see _net_refund_for_settlement -- is never invalidated by
    this check. An order can carry both a clean settlement match and an
    over-refund exception at the same time; that's the correct outcome,
    not a contradiction.
    """
    refunds_by_payment: dict[str, list[dict]] = {}
    for r in (refunds or []):
        refunds_by_payment.setdefault(r["payment_id"], []).append(r)

    exceptions = []
    for order in orders:
        order_refunds = refunds_by_payment.get(order["razorpay_payment_id"], [])
        if not order_refunds:
            continue
        total_refunded = round(sum(r["amount"] for r in order_refunds), 2)
        excess = round(total_refunded - order["amount"], 2)
        if excess > 0.01:
            exceptions.append({
                "type": "refund_amount_exceeds_order", "order_id": order["order_id"],
                "amount": excess,
                "reason": f"Total refunds (Rs {total_refunded:,.2f}) for this order exceed its "
                          f"own amount (Rs {order['amount']:,.2f}) by Rs {excess:,.2f} -- more was "
                          "refunded than was ever charged.",
            })
            log("over_refund_check", "exception", "rule", 1.0,
                f"Refund total Rs {total_refunded:,.2f} exceeds order amount Rs {order['amount']:,.2f} "
                f"by Rs {excess:,.2f}", order_id=order["order_id"])
    return exceptions


def _check_chargebacks(orders: list[dict], chargebacks: list[dict], log) -> list[dict]:
    """A chargeback is not a refund wearing a different name -- it's a
    customer-initiated, forced reversal via their card issuer, not
    something the merchant chose to do. Two concrete differences that
    matter for reconciliation: it typically carries its own penalty fee
    on top of the reversed amount (a real cost with nothing to do with
    the original order), and it can land long after settlement with no
    date-drift relationship to it (a dispute window, not a payout SLA).
    Modeled the same way as _check_over_refunds -- purely additive,
    never touches `matches` -- so a chargeback is never mistaken for a
    voluntary refund and never netted against a settlement the way a
    pre-settlement refund legitimately can be.
    """
    by_payment: dict[str, list[dict]] = {}
    for cb in (chargebacks or []):
        by_payment.setdefault(cb["payment_id"], []).append(cb)

    order_id_by_payment = {o["razorpay_payment_id"]: o["order_id"] for o in orders}
    exceptions = []
    for payment_id, payment_chargebacks in by_payment.items():
        order_id = order_id_by_payment.get(payment_id)
        for cb in payment_chargebacks:
            fee = round(cb.get("fee", 0.0) or 0.0, 2)
            total_impact = round(cb["amount"] + fee, 2)
            fee_note = f" plus a Rs {fee:,.2f} chargeback fee" if fee else ""
            exceptions.append({
                "type": "chargeback", "order_id": order_id, "payment_id": payment_id,
                "amount": total_impact,
                "reason": f"Chargeback for Rs {cb['amount']:,.2f}{fee_note} was filed against "
                          "this payment -- a customer-initiated dispute reversal, not a "
                          "merchant-initiated refund, and never netted against any settlement.",
            })
            log("chargeback_check", "exception", "rule", 1.0,
                f"Chargeback of Rs {cb['amount']:,.2f}{fee_note} filed against {payment_id}.",
                order_id=order_id, payment_id=payment_id)
    return exceptions


def reconcile(orders: list[dict], settlements: list[dict], bank_lines: list[dict],
              fee_tolerance_pct: float = FEE_TOLERANCE_PCT,
              date_drift_ok_days: int = DATE_DRIFT_OK_DAYS,
              refunds: list[dict] | None = None,
              chargebacks: list[dict] | None = None) -> dict[str, Any]:
    """fee_tolerance_pct and date_drift_ok_days are policy, not physics --
    a real merchant's actual fee schedule and settlement SLA would set
    these, so they're parameters with the current constants as defaults,
    not hardcoded thresholds baked into the pass logic below.

    `refunds` (optional -- empty for callers that don't have refund data)
    is a list of {payment_id, amount, ...} rows. A refund isn't noise to
    explain away: it changes what settlement amount is *expected*. A full
    refund means no settlement should exist at all; a partial refund means
    the settlement is genuinely order.amount - refund.amount, not a fee
    adjustment. Comparing against the raw order amount in either case
    would misfile a correct outcome as an exception.
    """
    audit: list[dict] = []
    matches: list[dict] = []
    exceptions: list[dict] = []

    orders_by_payment = {}
    for o in orders:
        orders_by_payment.setdefault(o["razorpay_payment_id"], []).append(o)

    refunds_by_payment: dict[str, list[dict]] = {}
    for r in (refunds or []):
        refunds_by_payment.setdefault(r["payment_id"], []).append(r)

    # A settlement normally covers one payment_id, but a batch settlement
    # (multiple payments consolidated by Razorpay into one settlement,
    # settled as one bank credit) legitimately covers several -- index it
    # under every payment_id it covers so any of those orders can find it.
    settlements_by_payment: dict[str, list[dict]] = {}
    for s in settlements:
        for payment_id in s.get("payment_ids") or [s["payment_id"]]:
            settlements_by_payment.setdefault(payment_id, []).append(s)

    unmatched_bank = {b["line_id"]: b for b in bank_lines}
    resolved_settlement_ids: set[str] = set()
    needs_llm: list[dict] = []

    def log(step: str, decision: str, method: str, confidence: float, note: str, **ids):
        audit.append({
            "step": step, "decision": decision, "method": method,
            "confidence": confidence, "note": note, **ids,
        })

    # ---- pass 0: batch settlements (many orders : one settlement : one
    # bank line). Handled as a single unit first, so the per-payment_id
    # loop below never sees these payment_ids as unresolved individually.
    handled_payment_ids: set[str] = set()
    seen_batch_settlement_ids: set[str] = set()
    for stl in settlements:
        covered_ids = stl.get("payment_ids") or [stl["payment_id"]]
        if len(covered_ids) <= 1 or stl["settlement_id"] in seen_batch_settlement_ids:
            continue
        seen_batch_settlement_ids.add(stl["settlement_id"])

        group_orders = [orders_by_payment[pid][0] for pid in covered_ids if pid in orders_by_payment]
        if not group_orders:
            continue
        agg_order = {"amount": sum(o["amount"] for o in group_orders),
                     "created_at": min(o["created_at"] for o in group_orders)}

        utr_matches = [b for b in unmatched_bank.values() if stl["utr"] in b["narration"]]
        if not utr_matches:
            continue  # falls through to the normal per-order loop below, which will
                       # surface this honestly (as amount_mismatch or an exception)
                       # rather than silently dropping the group

        result = _find_settlement_match(stl, agg_order, utr_matches, fee_tolerance_pct, date_drift_ok_days)
        if result is None:
            continue

        method, confidence, note, bank_line_ids = result
        group_note = f"Part of a batch settlement covering {len(group_orders)} orders. {note}"
        for o in group_orders:
            matches.append({
                "order_id": o["order_id"], "settlement_id": stl["settlement_id"],
                "bank_line_id": bank_line_ids[0], "bank_line_ids": bank_line_ids,
                "method": "batch_settlement", "confidence": confidence, "note": group_note,
            })
            handled_payment_ids.add(o["razorpay_payment_id"])
            log("match", "matched", "batch_settlement", confidence, group_note, order_id=o["order_id"])
        resolved_settlement_ids.add(stl["settlement_id"])
        for bank_line_id in bank_line_ids:
            unmatched_bank.pop(bank_line_id, None)

    # ---- pass 1 & 2: rule-based, per payment_id ----
    for payment_id, orders_for_payment in orders_by_payment.items():
        if payment_id in handled_payment_ids:
            continue
        order = orders_for_payment[0]

        # Two orders should never share a payment_id in real Razorpay data,
        # but if they did, only `order` above gets reconciled against this
        # payment_id's settlements -- the rest must be surfaced explicitly,
        # never silently skipped.
        for extra_order in orders_for_payment[1:]:
            exceptions.append({
                "type": "duplicate_order_reference", "order_id": extra_order["order_id"],
                "reason": f"Order shares razorpay_payment_id {payment_id} with "
                          f"{order['order_id']}; only one order per payment_id can be "
                          "reconciled against that payment_id's settlements.",
            })
            log("payment_id_dedupe", "exception", "rule", 1.0,
                "duplicate payment_id across orders", order_id=extra_order["order_id"])

        candidate_settlements = settlements_by_payment.get(payment_id, [])
        order_refunds = refunds_by_payment.get(payment_id, [])
        total_refund = round(sum(r["amount"] for r in order_refunds), 2)

        if not candidate_settlements:
            # total_refund > 0 guards against a near-zero order amount
            # (<= Re 1, the tolerance below) with zero actual refund
            # records -- 0 >= 0 - 1.0 is trivially true, which would
            # otherwise fabricate a "fully refunded" claim no refund
            # record actually supports.
            if total_refund > 0 and total_refund >= order["amount"] - 1.0:
                # No settlement exists, but a refund fully explains why --
                # Razorpay never generates a settlement for a fully refunded
                # payment. This is a correct, resolved outcome, not a gap.
                note = f"Order fully refunded (₹{total_refund:,.2f}); no settlement expected."
                matches.append({
                    "order_id": order["order_id"], "settlement_id": None,
                    "bank_line_id": None, "method": "refunded", "confidence": 1.0,
                    "note": note,
                })
                log("refund_check", "matched", "refunded", 1.0, note, order_id=order["order_id"])
                continue
            exceptions.append({
                "type": "no_settlement_found", "order_id": order["order_id"],
                "reason": "Order has a Razorpay payment_id but no matching settlement "
                          "record exists in this batch.",
            })
            log("settlement_lookup", "exception", "rule", 1.0,
                "no settlement row for this payment_id", order_id=order["order_id"])
            continue

        # Netting is computed per candidate settlement, not once for the
        # order -- a refund only reduces what a *specific* settlement was
        # expected to pay out if it predates that settlement (see
        # _net_refund_for_settlement). A refund issued after a settlement
        # never shrinks it.
        picked = None
        picked_stl = None
        picked_pre = picked_post = 0.0
        for stl in candidate_settlements:
            if stl["settlement_id"] in resolved_settlement_ids:
                continue
            utr_matches = [b for b in unmatched_bank.values() if stl["utr"] in b["narration"]]
            if not utr_matches:
                continue
            match_order, pre, post = _net_refund_for_settlement(order, stl, order_refunds)
            result = _find_settlement_match(stl, match_order, utr_matches, fee_tolerance_pct, date_drift_ok_days)
            if result is None:
                continue
            picked, picked_stl, picked_pre, picked_post = result, stl, pre, post
            break

        if picked:
            method, confidence, note, bank_line_ids = picked
            stl = picked_stl
            refund_suffix = _refund_suffix(picked_pre, picked_post)
            matches.append({
                "order_id": order["order_id"], "settlement_id": stl["settlement_id"],
                "bank_line_id": bank_line_ids[0], "bank_line_ids": bank_line_ids,
                "method": method, "confidence": confidence, "note": note + refund_suffix,
            })
            resolved_settlement_ids.add(stl["settlement_id"])
            for bank_line_id in bank_line_ids:
                unmatched_bank.pop(bank_line_id, None)
            log("match", "matched", method, confidence, note + refund_suffix, order_id=order["order_id"])
        else:
            stl = candidate_settlements[0]
            match_order, pre, post = _net_refund_for_settlement(order, stl, order_refunds)
            refund_suffix = _refund_suffix(pre, post)
            amount_diff_pct = abs(stl["amount"] - match_order["amount"]) / (match_order["amount"] or order["amount"] or 1.0)
            if amount_diff_pct > fee_tolerance_pct:
                # amount is genuinely off, not fee-sized -> not an LLM question
                exceptions.append({
                    "type": "amount_mismatch", "order_id": order["order_id"],
                    "settlement_id": stl["settlement_id"],
                    "reason": f"Settlement amount {stl['amount']} differs from order "
                              f"amount {order['amount']}{refund_suffix} by {amount_diff_pct:.1%}, "
                              "outside plausible Razorpay fee range.",
                })
                log("match", "exception", "rule", 1.0,
                    "amount diff exceeds fee tolerance", order_id=order["order_id"])
            else:
                # amount lines up with what we'd expect, but this settlement's
                # UTR wasn't found verbatim in any remaining bank line -> the
                # narration is garbled/truncated. Free-text reasoning territory.
                needs_llm.append({
                    "order_id": order["order_id"], "settlement_id": stl["settlement_id"],
                    "expected_utr": stl["utr"], "expected_amount": stl["amount"],
                    # cosmetic LLM-prompt context only, not part of
                    # REQUIRED_ORDER_FIELDS -- uploaded orders.csv from a
                    # real user has no reason to include it
                    "customer": order.get("customer", "unknown"),
                })
                log("match", "deferred_to_llm", "rule", 0.0,
                    "settlement UTR not found verbatim in any bank line", order_id=order["order_id"])

        if len(candidate_settlements) > 1:
            # `stl` here is whichever settlement was actually used above (the
            # matched one, or candidate_settlements[0] for the primary
            # exception) -- not necessarily the first item in the list. A
            # settlement that matches later in the list (e.g. because its
            # sibling's UTR wasn't found in any bank line) must still be
            # excluded here, or it silently vanishes: neither matched nor
            # flagged as its own exception.
            for extra in candidate_settlements:
                if extra["settlement_id"] not in resolved_settlement_ids and extra is not stl:
                    exceptions.append({
                        "type": "duplicate_candidate", "order_id": order["order_id"],
                        "settlement_id": extra["settlement_id"],
                        "reason": "More than one settlement row references this payment_id; "
                                  "only one was matched to a bank line.",
                    })
                    log("dedupe", "exception", "rule", 0.8,
                        "unresolved duplicate settlement", order_id=order["order_id"])

    refund_matches, refund_exceptions = _match_refund_debits(refunds or [], unmatched_bank, log)
    exceptions.extend(refund_exceptions)
    exceptions.extend(_check_over_refunds(orders, refunds, log))
    exceptions.extend(_check_chargebacks(orders, chargebacks or [], log))

    return {
        "matches": matches,
        "exceptions": exceptions,
        "audit_trail": audit,
        "needs_llm": needs_llm,
        "unmatched_bank_lines": list(unmatched_bank.values()),
        "refund_matches": refund_matches,
    }


DEFAULT_LLM_CONFIDENCE_THRESHOLD = 0.6


LLM_BATCH_TIME_BUDGET_SECONDS = 6.0
LLM_MAX_CONCURRENCY = 5  # bounded, so a large batch can't open dozens of concurrent
                         # connections to the LLM provider at once


def _narrow_llm_candidates(case: dict, candidates: list[dict], tolerance_pct: float = 0.05) -> list[dict]:
    """Every LLM call used to send the full remaining bank-line pool as
    candidates, most of which have a wildly different amount and could
    never plausibly be this case's match -- needlessly large prompts that
    burn through a provider's tokens-per-minute budget fast (hit in
    practice: Groq's free tier caps out around 6000-8000 TPM regardless
    of model, and a 33-candidate prompt alone can eat that in about 4
    calls, well under what a real batch needs). Narrows each case's
    candidates to bank lines within a generous amount window first, and
    falls back to the full list only if that window is empty, so a
    legitimate match outside it (an unusually large fee/adjustment) is
    never silently excluded -- this trades prompt size, not correctness.
    """
    expected = case["expected_amount"]
    narrow = [c for c in candidates if abs(c["amount"] - expected) <= expected * tolerance_pct]
    return narrow or candidates


def resolve_llm_verdicts(rule_result: dict, llm_resolve_fn,
                          time_budget_seconds: float = LLM_BATCH_TIME_BUDGET_SECONDS
                          ) -> list[tuple[dict, dict | None]]:
    """Call the LLM resolver exactly once per rule-deferred case, independent
    of the accept/reject confidence threshold. Split out from
    apply_llm_resolutions so a calibration sweep across many threshold
    values (see /api/calibrate) can reuse one round of LLM calls instead of
    repeating the same (slow, rate-limited, real-money-in-API-cost) calls
    once per threshold it wants to evaluate.

    Each provider call already carries its own short timeout (see
    llm_resolver.PROVIDER_CALL_TIMEOUT_SECONDS), but a batch with many
    deferred cases run one at a time would still add those up past the
    serverless function's own time limit. So this dispatches every
    deferred case to a bounded thread pool (LLM_MAX_CONCURRENCY workers --
    these are blocking network calls, not CPU-bound work, so plain
    threads parallelize them fine without an async rewrite of the FastAPI
    routes above this) and waits up to time_budget_seconds total for the
    whole batch, not per case. Whatever hasn't finished when the budget
    runs out resolves with the same offline heuristic a single provider
    failure already falls back to -- so a large batch degrades to a
    weaker (but still real, still explained) resolution strategy instead
    of risking a 504 on the whole request. Straggler threads aren't
    force-killed (Python threads can't be), just no longer waited on --
    each is already bounded by its own PROVIDER_CALL_TIMEOUT_SECONDS and
    will finish or error out in the background; its result is simply
    discarded once this function has already moved on.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
    from .llm_resolver import heuristic_resolve

    cases = rule_result["needs_llm"]
    if not cases:
        return []
    if not llm_resolve_fn:
        return [(case, None) for case in cases]

    candidates = list({b["line_id"]: b for b in rule_result["unmatched_bank_lines"]}.values())

    pool = ThreadPoolExecutor(max_workers=LLM_MAX_CONCURRENCY)
    future_to_case = {pool.submit(llm_resolve_fn, case, _narrow_llm_candidates(case, candidates)): case
                       for case in cases}
    verdict_by_order_id: dict[str, dict | None] = {}
    try:
        for future in as_completed(future_to_case, timeout=max(time_budget_seconds, 0)):
            case = future_to_case[future]
            try:
                verdict_by_order_id[case["order_id"]] = future.result()
            except Exception:
                verdict_by_order_id[case["order_id"]] = None
    except FutureTimeoutError:
        pass  # whatever hasn't finished falls back to the heuristic below
    finally:
        pool.shutdown(wait=False)  # don't block the response on stragglers

    verdicts = []
    for case in cases:
        if case["order_id"] in verdict_by_order_id:
            verdicts.append((case, verdict_by_order_id[case["order_id"]]))
        else:
            fallback = heuristic_resolve(case, candidates)
            fallback["reasoning"] = ("[LLM batch time budget exhausted; fell back to heuristic] "
                                      + fallback["reasoning"])
            verdicts.append((case, fallback))
    return verdicts


def apply_llm_resolutions(rule_result: dict, llm_resolve_fn,
                           confidence_threshold: float = DEFAULT_LLM_CONFIDENCE_THRESHOLD) -> dict:
    """Second pass: hand every rule-deferred case to the LLM resolver and
    fold its verdicts back into matches/exceptions/audit_trail. Runs even
    when there's nothing to resolve, so the shape of the result is uniform.

    `confidence_threshold` is the auto-accept bar for an LLM-proposed match
    -- a finance-ops reviewer would tune this (stricter to force more human
    review, looser to auto-clear more volume) rather than it being a fixed
    constant, so it's a parameter here, not a hardcoded value.
    """
    case_verdicts = resolve_llm_verdicts(rule_result, llm_resolve_fn)
    return apply_confidence_threshold(rule_result, case_verdicts, confidence_threshold)


def apply_confidence_threshold(rule_result: dict, case_verdicts: list[tuple[dict, dict | None]],
                                confidence_threshold: float = DEFAULT_LLM_CONFIDENCE_THRESHOLD) -> dict:
    """Apply one accept/reject bar to a set of already-computed LLM verdicts.
    Pure and cheap (no LLM calls) -- this is what lets a calibration sweep
    evaluate many threshold values against the same fixed verdicts.
    """
    matches = list(rule_result["matches"])
    exceptions = list(rule_result["exceptions"])
    audit = list(rule_result["audit_trail"])
    unmatched_bank = {b["line_id"]: b for b in rule_result["unmatched_bank_lines"]}

    for case, verdict in case_verdicts:
        if verdict and verdict.get("bank_line_id") in unmatched_bank and verdict.get("confidence", 0) >= confidence_threshold:
            bank_hit = unmatched_bank.pop(verdict["bank_line_id"])
            matches.append({
                "order_id": case["order_id"], "settlement_id": case["settlement_id"],
                "bank_line_id": bank_hit["line_id"], "method": "llm",
                "confidence": verdict["confidence"], "note": verdict["reasoning"],
            })
            audit.append({
                "step": "llm_resolve", "decision": "matched", "method": "llm",
                "confidence": verdict["confidence"], "note": verdict["reasoning"],
                "order_id": case["order_id"],
            })
        else:
            reason = (verdict or {}).get("reasoning", "LLM resolver returned no confident candidate")
            exceptions.append({
                "type": "unrecognized_narration", "order_id": case["order_id"],
                "settlement_id": case["settlement_id"],
                "reason": f"Expected UTR {case['expected_utr']} not found in any bank "
                          f"narration; LLM could not confidently resolve it either. {reason}",
            })
            audit.append({
                "step": "llm_resolve", "decision": "exception", "method": "llm",
                "confidence": (verdict or {}).get("confidence", 0.0), "note": reason,
                "order_id": case["order_id"],
            })

    for bank_line_id, bl in unmatched_bank.items():
        exceptions.append({
            "type": "unrecognized_bank_line", "bank_line_id": bank_line_id,
            "amount": bl["amount"],
            "reason": f"Bank line for amount {bl['amount']} on {bl['value_date']} does not "
                      "correspond to any order in this batch.",
        })
        audit.append({
            "step": "bank_line_sweep", "decision": "exception", "method": "rule",
            "confidence": 1.0, "note": "bank line unclaimed after all passes",
            "bank_line_id": bank_line_id,
        })

    return {"matches": matches, "exceptions": exceptions, "audit_trail": audit,
            "refund_matches": rule_result.get("refund_matches", [])}


VALID_OVERRIDE_ACTIONS = {"accept_match", "reject_match", "manual_match"}


def apply_override(matches: list[dict], exceptions: list[dict], audit_trail: list[dict],
                    action: str, order_id: str, bank_line_id: str | None = None,
                    reviewer_note: str | None = None) -> dict[str, Any]:
    """A human reviewer's decision, folded into the same matches/exceptions/
    audit_trail shape the automated passes produce -- but tagged
    method="human_override" so the audit trail always shows what a person
    changed versus what the system decided. Pure function over the arrays
    the client already holds (from /api/run); nothing is stored server-side,
    keeping the whole API stateless.
    """
    if action not in VALID_OVERRIDE_ACTIONS:
        raise ValueError(f"Unknown override action: {action!r}. Must be one of {VALID_OVERRIDE_ACTIONS}.")

    matches = [dict(m) for m in matches]
    exceptions = [dict(e) for e in exceptions]
    audit_trail = list(audit_trail)
    note = (reviewer_note or "").strip()
    timestamp = datetime.now(timezone.utc).isoformat()

    def log_override(decision: str, confidence: float, override_note: str, **ids):
        audit_trail.append({
            "step": "human_review", "decision": decision, "method": "human_override",
            "confidence": confidence, "note": override_note, "timestamp": timestamp, **ids,
        })

    if action == "accept_match":
        match = next((m for m in matches if m["order_id"] == order_id), None)
        if match is None:
            raise ValueError(f"No existing match for order_id {order_id!r} to accept.")
        match["confidence"] = 1.0
        match["note"] = f"Human-reviewed and accepted.{(' ' + note) if note else ''}"
        log_override("matched", 1.0, match["note"], order_id=order_id)

    elif action == "reject_match":
        match = next((m for m in matches if m["order_id"] == order_id), None)
        if match is None:
            raise ValueError(f"No existing match for order_id {order_id!r} to reject.")
        matches.remove(match)
        reason = note or "Reviewer rejected the automated match; needs manual investigation."
        exceptions.append({"type": "human_rejected_match", "order_id": order_id, "reason": reason})
        log_override("exception", 0.0, reason, order_id=order_id)

    elif action == "manual_match":
        if not bank_line_id:
            raise ValueError("manual_match requires bank_line_id.")
        exceptions = [e for e in exceptions
                      if e.get("order_id") != order_id and e.get("bank_line_id") != bank_line_id]
        matches = [m for m in matches
                   if m["order_id"] != order_id and m.get("bank_line_id") != bank_line_id]
        reason = note or "Manually matched by reviewer."
        matches.append({
            "order_id": order_id, "settlement_id": None, "bank_line_id": bank_line_id,
            "method": "human_override", "confidence": 1.0, "note": reason,
        })
        log_override("matched", 1.0, reason, order_id=order_id, bank_line_id=bank_line_id)

    return {"matches": matches, "exceptions": exceptions, "audit_trail": audit_trail}


def summarize(orders: list[dict], result: dict, refunds: list[dict] | None = None,
              refund_matches: list[dict] | None = None) -> dict:
    total = len(orders)
    matched = len(result["matches"])
    by_method = {}
    for m in result["matches"]:
        by_method[m["method"]] = by_method.get(m["method"], 0) + 1

    amount_by_order_id = {o["order_id"]: o["amount"] for o in orders}

    # A refund isn't reconciled *settlement* money -- a fully refunded order
    # had nothing settle at all, and a partially refunded one settled less
    # than its order amount. Counting the full order amount as "reconciled"
    # for those would overstate what actually moved, which is exactly the
    # kind of claim this project is built to avoid making.
    refund_total_by_order_id: dict[str, float] = {}
    if refunds:
        payment_id_by_order_id = {o["order_id"]: o["razorpay_payment_id"] for o in orders}
        refund_total_by_payment: dict[str, float] = {}
        for r in refunds:
            refund_total_by_payment[r["payment_id"]] = round(
                refund_total_by_payment.get(r["payment_id"], 0.0) + r["amount"], 2)
        for order_id, payment_id in payment_id_by_order_id.items():
            if payment_id in refund_total_by_payment:
                refund_total_by_order_id[order_id] = refund_total_by_payment[payment_id]

    total_amount_matched = 0.0
    total_amount_refunded = 0.0
    for m in result["matches"]:
        order_amount = amount_by_order_id.get(m["order_id"], 0.0)
        refund_amount = refund_total_by_order_id.get(m["order_id"], 0.0)
        total_amount_refunded += refund_amount
        if m["method"] == "refunded":
            continue  # fully refunded -- zero net settlement money, correctly excluded
        total_amount_matched += max(round(order_amount - refund_amount, 2), 0.0)
    total_amount_matched = round(total_amount_matched, 2)
    total_amount_refunded = round(total_amount_refunded, 2)

    # One order can carry more than one exception (e.g. a duplicate
    # settlement flagged alongside the order's own primary outcome) -- sum
    # each order's amount once, not once per exception, so this can't
    # overstate exposure. Exceptions with no order_id (an orphan bank line)
    # carry their own "amount" and are added on top, since they're real
    # money movements not attributable to any order.
    #
    # refund_amount_exceeds_order and chargeback are both different in
    # kind from every other exception type here: they're deliberately
    # additive (see _check_over_refunds / _check_chargebacks) and can sit
    # on an order whose settlement is otherwise cleanly matched --
    # folding either into the same "count the whole order amount" rule
    # would inflate at-risk by the full order value over what's actually
    # anomalous (the excess, or the chargeback's own amount+fee). Both
    # carry their own accurate `amount`, so they're summed like an
    # orphan-bank-line exception instead.
    ADDITIVE_EXCEPTION_TYPES = {"refund_amount_exceeds_order", "chargeback"}
    excepted_order_ids = {e["order_id"] for e in result["exceptions"]
                           if e.get("order_id") and e["type"] not in ADDITIVE_EXCEPTION_TYPES}
    total_amount_at_risk = round(
        sum(amount_by_order_id.get(oid, 0.0) for oid in excepted_order_ids)
        + sum(e.get("amount", 0.0) for e in result["exceptions"]
              if not e.get("order_id") or e["type"] in ADDITIVE_EXCEPTION_TYPES), 2)

    # Cash position: a refund record proves money was promised back, not
    # that it left the account -- these three numbers separate "refunded
    # on paper" from "confirmed out the door" so a controller isn't
    # closing the books on an assumption. See _match_refund_debits.
    debited_refund_keys = {m["refund_id"] for m in (refund_matches or [])}
    total_refund_amount_debited = round(
        sum(r["amount"] for r in (refunds or []) if _refund_key(r) in debited_refund_keys), 2)
    total_refund_amount_undebited = round(
        sum(r["amount"] for r in (refunds or []) if _refund_key(r) not in debited_refund_keys), 2)

    return {
        "total_orders": total,
        "matched": matched,
        "match_rate": round(matched / total, 4) if total else 0.0,
        "by_method": by_method,
        "exception_count": len(result["exceptions"]),
        "total_amount_matched": total_amount_matched,
        "total_amount_at_risk": total_amount_at_risk,
        "total_amount_refunded": total_amount_refunded,
        "refund_count": len(refunds or []),
        "total_refund_amount_debited": total_refund_amount_debited,
        "total_refund_amount_undebited": total_refund_amount_undebited,
    }


DEFAULT_MATERIALITY_THRESHOLD = 5000.0


def classify_exceptions(orders: list[dict], exceptions: list[dict],
                         materiality_threshold: float = DEFAULT_MATERIALITY_THRESHOLD) -> list[dict]:
    """Attach a rupee amount and a priority tier (high/medium/low) to every
    exception, so a reviewer sees which ones actually matter first instead
    of an unordered list. This is a display/triage concern computed here
    over the matcher's output, not baked into the matching passes
    themselves -- the underlying exception records are unchanged, just
    annotated. Works for uploaded data too (no ground truth needed).
    """
    amount_by_order_id = {o["order_id"]: o["amount"] for o in orders}
    classified = []
    for e in exceptions:
        e = dict(e)
        amount = e.get("amount")
        if amount is None and e.get("order_id"):
            amount = amount_by_order_id.get(e["order_id"], 0.0)
        e["amount"] = round(amount or 0.0, 2)
        if e["amount"] >= materiality_threshold:
            e["priority"] = "high"
        elif e["amount"] >= materiality_threshold * 0.3:
            e["priority"] = "medium"
        else:
            e["priority"] = "low"
        classified.append(e)
    return classified


def closing_verdict(classified_exceptions: list[dict],
                     materiality_threshold: float = DEFAULT_MATERIALITY_THRESHOLD) -> dict:
    """A plain answer to the question a finance controller actually asks --
    not 'here are eight numbers, you decide' but a stated verdict, the way
    apply_override tags human decisions distinctly: this is a synthesis
    layer over data already computed, not a new decision-making pass.
    """
    material = [e for e in classified_exceptions if e["priority"] == "high"]
    material_amount = round(sum(e["amount"] for e in material), 2)
    total_amount = round(sum(e["amount"] for e in classified_exceptions), 2)
    can_close = len(material) == 0
    if can_close:
        message = "No material reconciliation discrepancies remain."
    else:
        message = (f"₹{total_amount:,.0f} remains unresolved, including "
                   f"₹{material_amount:,.0f} above your materiality threshold "
                   f"(₹{materiality_threshold:,.0f}).")
    return {
        "can_close": can_close,
        "materiality_threshold": materiality_threshold,
        "material_exception_count": len(material),
        "material_exception_amount": material_amount,
        "total_exception_amount": total_amount,
        "message": message,
    }

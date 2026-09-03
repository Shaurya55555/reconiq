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

    `refunded_at` is optional (REQUIRED_REFUND_FIELDS asks only for
    payment_id and amount) -- a refund with unknown timing is treated as
    pre-settlement (always netted), matching how the no-settlement-exists
    path elsewhere sums refunds with no date check at all, rather than
    guessing a date or crashing on the missing key.
    """
    settled_at = _parse_date(stl["settled_at"])
    def is_pre_settlement(r: dict) -> bool:
        refunded_at = r.get("refunded_at")
        return not refunded_at or _parse_date(refunded_at) <= settled_at
    pre = round(sum(r["amount"] for r in order_refunds if is_pre_settlement(r)), 2)
    post = round(sum(r["amount"] for r in order_refunds if not is_pre_settlement(r)), 2)
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


def _within_fee_tolerance(diff_amount: float, base_amount: float, tolerance_pct: float) -> bool:
    """A percentage-only comparison can reject a genuinely-intended
    boundary case by a hair purely because the settlement amount itself
    was rounded to the nearest paisa when stored -- e.g. a "true" 3%
    target of 7615.3245 is stored as 7615.32, making the real difference
    3.0001%, not 3.0000%. That's the same class of currency-rounding gap
    as the 0.01 tolerance already used for bank-line/settlement equality
    elsewhere in this file, just showing up in the percentage check
    instead. One paisa of absolute slack beyond the percentage-computed
    boundary absorbs it without loosening the tolerance for a genuine
    mismatch (a real 4%+ difference is still ~400x that buffer).
    """
    return diff_amount <= round(base_amount * tolerance_pct + 0.01, 6)


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
    # order["amount"] is the *netted* amount here (order minus any
    # pre-settlement refunds) and can legitimately be 0 for an over-refund
    # (refunds summing to more than the order was worth) -- the 1.0
    # fallback matches the guard already used for this same case in
    # reconcile()'s non-matching branch, so a settlement against an
    # already-fully-refunded order reliably fails tolerance and becomes an
    # honest exception instead of crashing.
    # round() absorbs IEEE-754 division noise (e.g. two cases that are both
    # exactly a business 3.0% difference by construction can otherwise
    # compute to 0.03 and 0.030000000000000065 respectively -- the second
    # fails a strict <= 0.03 tolerance check by a floating-point artifact,
    # not a real amount difference). 6 decimal places is 0.0001% precision,
    # far finer than any real currency distinction, so this only removes
    # float noise -- it does not loosen the tolerance itself.
    amount_diff_pct = round(abs(stl["amount"] - order["amount"]) / (order["amount"] or 1.0), 6)
    date_drift = abs((_parse_date(stl["settled_at"]) - _parse_date(order["created_at"])).days)

    # Settlement amounts can carry more decimal precision than a bank
    # statement's 2-decimal display (e.g. a settlement API returning
    # 4850.125 against a bank line rounded to 4850.13) -- exact equality
    # would treat that as "no candidate" and force an otherwise-trivial
    # case through the expensive LLM/heuristic fallback. 0.01 matches the
    # same currency-rounding tolerance already used for the group-sum
    # comparison just below, not the percentage-based fee tolerance.
    single = next((b for b in utr_matches if abs(b["amount"] - stl["amount"]) <= 0.01), None)
    if single:
        # date_drift_ok_days is a hard eligibility boundary, not a
        # confidence hint -- a settlement beyond the policy window is never
        # auto-matched here regardless of how good the UTR/amount evidence
        # is. The caller checks for exactly this case (UTR+amount would
        # otherwise qualify, but date_drift disqualifies it) and raises a
        # dedicated date_drift_exceeded exception instead of silently
        # falling through to amount_mismatch or an LLM call -- see
        # reconcile()'s else branch.
        if date_drift > date_drift_ok_days:
            return None
        if amount_diff_pct < 0.001:
            return ("exact", 1.0, "UTR + amount matched exactly", [single["line_id"]])
        if _within_fee_tolerance(abs(stl["amount"] - order["amount"]), order["amount"] or 1.0,
                                  effective_fee_tolerance_pct):
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
        # `key` is refund_id when present (synthetic data always includes
        # it, embedded directly in the narration) -- but refund_id isn't in
        # REQUIRED_REFUND_FIELDS, and a realistic uploaded refunds.csv
        # commonly won't have one. The composite payment_id_amount fallback
        # key used in that case will essentially never appear verbatim in
        # a real bank narration (a bank references the payment, not a
        # synthesized "payment_id_amount" string), which silently broke
        # refund-debit matching for every CSV upload without a refund_id
        # column -- always reporting "not debited" even when a matching
        # outbound line genuinely existed. payment_id is always present
        # (REQUIRED_REFUND_FIELDS) and is the realistic identifier a real
        # bank narration would actually reference.
        hit = next((b for b in unmatched_bank.values()
                    if b["amount"] < 0 and abs(-b["amount"] - r["amount"]) < 0.01
                    and (key in b["narration"] or r["payment_id"] in b["narration"])), None)
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
                # summarize() needs the *actual* pre-settlement refund netted
                # here, not every refund on this payment_id regardless of
                # timing -- a post-settlement refund is explicitly "not
                # netted against this settlement" in `note` above, but
                # summarize() had no way to know that and subtracted the
                # full refund total anyway, understating amount reconciled
                # for a settlement its own match note says is legitimate.
                "pre_settlement_refund": picked_pre,
            })
            resolved_settlement_ids.add(stl["settlement_id"])
            for bank_line_id in bank_line_ids:
                unmatched_bank.pop(bank_line_id, None)
            log("match", "matched", method, confidence, note + refund_suffix, order_id=order["order_id"])
        else:
            stl = candidate_settlements[0]
            match_order, pre, post = _net_refund_for_settlement(order, stl, order_refunds)
            refund_suffix = _refund_suffix(pre, post)

            stl_utr_matches = [b for b in unmatched_bank.values() if stl["utr"] in b["narration"]]
            # This settlement's own bank line, if UTR-identified and
            # currency-rounding-close to the settlement's own amount --
            # used below to decide zero/date-drift/amount_mismatch and to
            # claim the line so the end-of-pass sweep never double-counts
            # it as an unrelated orphan.
            own_bank_line = next((b for b in stl_utr_matches if abs(b["amount"] - stl["amount"]) <= 0.01), None)
            zero_bank_line = next((b for b in stl_utr_matches if b["amount"] == 0), None)
            date_drift = abs((_parse_date(stl["settled_at"]) - _parse_date(order["created_at"])).days)
            base_amount = match_order["amount"] or order["amount"] or 1.0
            amount_diff = abs(stl["amount"] - match_order["amount"])
            amount_diff_pct = round(amount_diff / base_amount, 6)
            amount_ok = _within_fee_tolerance(amount_diff, base_amount, fee_tolerance_pct)

            if zero_bank_line:
                # A bank line found by this settlement's UTR whose amount is
                # exactly zero is a distinct, more specific situation than a
                # generic amount mismatch or an unresolved narration --
                # usually a reversed/failed transfer leg on the bank's side,
                # not ambiguity worth an LLM's time.
                unmatched_bank.pop(zero_bank_line["line_id"], None)
                exceptions.append({
                    "type": "zero_amount_bank_line", "order_id": order["order_id"],
                    "settlement_id": stl["settlement_id"], "bank_line_id": zero_bank_line["line_id"],
                    "reason": "A bank line referencing this settlement's UTR was found, but its "
                              f"amount is Rs 0.00 -- likely a reversed or failed transfer leg, "
                              f"not usable to reconcile the expected Rs {stl['amount']:,.2f} settlement.",
                })
                log("match", "exception", "rule", 1.0,
                    "bank line references UTR but amount is zero", order_id=order["order_id"])
                # deliberately falls through to the duplicate_candidate check
                # below rather than `continue` -- a second candidate
                # settlement for this payment_id must still be flagged even
                # when the first one hit a zero-amount bank line.
            elif own_bank_line and amount_ok and date_drift > date_drift_ok_days:
                # date_drift_ok_days is a hard eligibility boundary (see
                # _find_settlement_match): UTR and amount both check out,
                # but this settlement arrived too late to auto-clear.
                # Distinct from amount_mismatch (the amount is fine) and
                # from needs_llm (nothing ambiguous to reason about -- the
                # date is just outside policy). Claim the bank line for the
                # same double-counting reason as the other branches here.
                unmatched_bank.pop(own_bank_line["line_id"], None)
                exceptions.append({
                    "type": "date_drift_exceeded", "order_id": order["order_id"],
                    "settlement_id": stl["settlement_id"],
                    "reason": f"UTR and amount both correspond to this settlement, but it settled "
                              f"{date_drift}d after the order{refund_suffix} -- beyond the "
                              f"{date_drift_ok_days}d policy window, so it cannot auto-clear.",
                })
                log("match", "exception", "rule", 1.0,
                    f"settlement date drifted {date_drift}d beyond the {date_drift_ok_days}d policy window",
                    order_id=order["order_id"])
            elif not amount_ok:
                # amount is genuinely off, not fee-sized -> not an LLM question.
                # This settlement's own bank line is already fully explained
                # by this amount_mismatch exception -- claim it now so the
                # end-of-pass sweep doesn't ALSO flag it as an unrelated
                # unrecognized_bank_line, double-counting the same money as
                # two separate risk line items in amount_at_risk.
                if own_bank_line:
                    unmatched_bank.pop(own_bank_line["line_id"], None)
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
                    # stl_utr_matches can be non-empty: a bank line's
                    # narration can contain this exact UTR while still
                    # being disqualified (wrong amount, e.g. a reversal
                    # line). The final exception text must not claim the
                    # UTR was "not found" when it plainly was -- that's
                    # exactly the kind of overclaim this project exists
                    # to avoid making.
                    "utr_seen_in_narration": bool(stl_utr_matches),
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


def _llm_resolution_source(verdict: dict | None) -> str:
    """A "method": "llm" match/exception can come from a genuine model
    response, or from the offline heuristic standing in for one (a
    provider call failure, or the whole batch's time budget running out --
    see resolve_llm_verdicts and llm_resolver.resolve). Both cases were
    only ever distinguishable by grepping the free-text reasoning for a
    "[...]" prefix, which made a timeout fallback look identical to
    successful AI reasoning at the method/summary level -- exactly the
    kind of thing this project's audit trail exists to not obscure. Both
    fallback paths already tag their reasoning with a "[...]" prefix by
    convention; this reads that same signal into a queryable field instead
    of leaving it as text a person has to notice.
    """
    if not verdict:
        return "none"
    return "heuristic_fallback" if verdict.get("reasoning", "").startswith("[") else "llm"


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
        resolution_source = _llm_resolution_source(verdict)
        if verdict and verdict.get("bank_line_id") in unmatched_bank and verdict.get("confidence", 0) >= confidence_threshold:
            bank_hit = unmatched_bank.pop(verdict["bank_line_id"])
            matches.append({
                "order_id": case["order_id"], "settlement_id": case["settlement_id"],
                "bank_line_id": bank_hit["line_id"], "method": "llm",
                "confidence": verdict["confidence"], "note": verdict["reasoning"],
                "llm_resolution_source": resolution_source,
            })
            audit.append({
                "step": "llm_resolve", "decision": "matched", "method": "llm",
                "confidence": verdict["confidence"], "note": verdict["reasoning"],
                "order_id": case["order_id"], "llm_resolution_source": resolution_source,
            })
        else:
            reason = (verdict or {}).get("reasoning", "LLM resolver returned no confident candidate")
            if case.get("utr_seen_in_narration"):
                utr_clause = (f"A bank line's narration contains the expected UTR "
                               f"{case['expected_utr']}, but its amount doesn't correspond to "
                               f"the expected {case['expected_amount']}")
            else:
                utr_clause = f"Expected UTR {case['expected_utr']} not found in any bank narration"
            exceptions.append({
                "type": "unrecognized_narration", "order_id": case["order_id"],
                "settlement_id": case["settlement_id"],
                "reason": f"{utr_clause}; LLM could not confidently resolve it either. {reason}",
                "llm_resolution_source": resolution_source,
            })
            audit.append({
                "step": "llm_resolve", "decision": "exception", "method": "llm",
                "confidence": (verdict or {}).get("confidence", 0.0), "note": reason,
                "order_id": case["order_id"], "llm_resolution_source": resolution_source,
            })

    for bank_line_id, bl in unmatched_bank.items():
        # An unexplained outbound line (negative amount -- money leaving the
        # account with no matching refund/order) is real exposure, same as
        # an unexplained inbound one -- the exception's `amount` is a risk
        # magnitude fed into total_amount_at_risk, not a signed ledger
        # entry, so a negative value here would silently *reduce* at-risk
        # instead of adding to it. The reason text keeps the real signed
        # value for clarity about direction.
        exceptions.append({
            "type": "unrecognized_bank_line", "bank_line_id": bank_line_id,
            "amount": abs(bl["amount"]),
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
    for m in result["matches"]:
        order_amount = amount_by_order_id.get(m["order_id"], 0.0)
        # Prefer the exact pre-settlement-refund figure reconcile() already
        # computed for this specific match (exact/fuzzy/group_split all set
        # it) -- it correctly excludes a post-settlement refund, which
        # doesn't reduce what this settlement was expected to pay out.
        # Falling back to "every refund on this payment_id" only for match
        # types that don't set it yet (batch_settlement, llm) -- not a fix
        # for those, just not a regression from previous behavior there.
        refund_amount = m.get("pre_settlement_refund", refund_total_by_order_id.get(m["order_id"], 0.0))
        if m["method"] == "refunded":
            continue  # fully refunded -- zero net settlement money, correctly excluded
        total_amount_matched += max(round(order_amount - refund_amount, 2), 0.0)
    total_amount_matched = round(total_amount_matched, 2)

    # Every refund's amount, independent of whether its order matched,
    # became an exception, or the refund's payment_id doesn't correspond to
    # any known order at all -- money that was refunded doesn't stop being
    # refunded because the settlement side failed to reconcile. Deliberately
    # NOT derived from the matches loop above (that used to silently
    # exclude refunds tied to an unmatched/excepted order or an unknown
    # payment_id, understating this figure) -- consistent with
    # total_refund_amount_debited/undebited below, which already sum every
    # refund unconditionally and this must equal the sum of.
    total_amount_refunded = round(sum(r["amount"] for r in (refunds or [])), 2)

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

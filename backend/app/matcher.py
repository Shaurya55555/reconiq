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

from datetime import date, datetime, timezone
from itertools import combinations
from typing import Any

FEE_TOLERANCE_PCT = 0.03
DATE_DRIFT_OK_DAYS = 3
MAX_GROUP_SIZE = 3


def _parse_date(s: str) -> date:
    return datetime.fromisoformat(s).date() if "T" not in s else datetime.fromisoformat(s).date()


def _find_settlement_match(stl: dict, order: dict, utr_matches: list[dict],
                            fee_tolerance_pct: float, date_drift_ok_days: int) -> tuple | None:
    """Try to resolve one settlement against the bank lines that already
    reference its UTR (`utr_matches`). Returns (method, confidence, note,
    bank_line_ids) or None if nothing deterministic fits -- in which case
    the caller falls back to the LLM pass.
    """
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
        if amount_diff_pct <= fee_tolerance_pct:
            return ("fuzzy", 0.9,
                    f"UTR matched; amount differs by {amount_diff_pct:.1%}, within fee tolerance",
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


def reconcile(orders: list[dict], settlements: list[dict], bank_lines: list[dict],
              fee_tolerance_pct: float = FEE_TOLERANCE_PCT,
              date_drift_ok_days: int = DATE_DRIFT_OK_DAYS) -> dict[str, Any]:
    """fee_tolerance_pct and date_drift_ok_days are policy, not physics --
    a real merchant's actual fee schedule and settlement SLA would set
    these, so they're parameters with the current constants as defaults,
    not hardcoded thresholds baked into the pass logic below.
    """
    audit: list[dict] = []
    matches: list[dict] = []
    exceptions: list[dict] = []

    orders_by_payment = {}
    for o in orders:
        orders_by_payment.setdefault(o["razorpay_payment_id"], []).append(o)

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

        if not candidate_settlements:
            exceptions.append({
                "type": "no_settlement_found", "order_id": order["order_id"],
                "reason": "Order has a Razorpay payment_id but no matching settlement "
                          "record exists in this batch.",
            })
            log("settlement_lookup", "exception", "rule", 1.0,
                "no settlement row for this payment_id", order_id=order["order_id"])
            continue

        picked = None
        picked_stl = None
        for stl in candidate_settlements:
            if stl["settlement_id"] in resolved_settlement_ids:
                continue
            utr_matches = [b for b in unmatched_bank.values() if stl["utr"] in b["narration"]]
            if not utr_matches:
                continue
            result = _find_settlement_match(stl, order, utr_matches, fee_tolerance_pct, date_drift_ok_days)
            if result is None:
                continue
            picked, picked_stl = result, stl
            break

        if picked:
            method, confidence, note, bank_line_ids = picked
            stl = picked_stl
            matches.append({
                "order_id": order["order_id"], "settlement_id": stl["settlement_id"],
                "bank_line_id": bank_line_ids[0], "bank_line_ids": bank_line_ids,
                "method": method, "confidence": confidence, "note": note,
            })
            resolved_settlement_ids.add(stl["settlement_id"])
            for bank_line_id in bank_line_ids:
                unmatched_bank.pop(bank_line_id, None)
            log("match", "matched", method, confidence, note, order_id=order["order_id"])
        else:
            stl = candidate_settlements[0]
            amount_diff_pct = abs(stl["amount"] - order["amount"]) / order["amount"]
            if amount_diff_pct > fee_tolerance_pct:
                # amount is genuinely off, not fee-sized -> not an LLM question
                exceptions.append({
                    "type": "amount_mismatch", "order_id": order["order_id"],
                    "settlement_id": stl["settlement_id"],
                    "reason": f"Settlement amount {stl['amount']} differs from order "
                              f"amount {order['amount']} by {amount_diff_pct:.1%}, "
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
                    "customer": order["customer"],
                })
                log("match", "deferred_to_llm", "rule", 0.0,
                    "settlement UTR not found verbatim in any bank line", order_id=order["order_id"])

        if len(candidate_settlements) > 1:
            for extra in candidate_settlements:
                if extra["settlement_id"] not in resolved_settlement_ids and extra is not candidate_settlements[0]:
                    exceptions.append({
                        "type": "duplicate_candidate", "order_id": order["order_id"],
                        "settlement_id": extra["settlement_id"],
                        "reason": "More than one settlement row references this payment_id; "
                                  "only one was matched to a bank line.",
                    })
                    log("dedupe", "exception", "rule", 0.8,
                        "unresolved duplicate settlement", order_id=order["order_id"])

    return {
        "matches": matches,
        "exceptions": exceptions,
        "audit_trail": audit,
        "needs_llm": needs_llm,
        "unmatched_bank_lines": list(unmatched_bank.values()),
    }


DEFAULT_LLM_CONFIDENCE_THRESHOLD = 0.6


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
    matches = list(rule_result["matches"])
    exceptions = list(rule_result["exceptions"])
    audit = list(rule_result["audit_trail"])
    unmatched_bank = {b["line_id"]: b for b in rule_result["unmatched_bank_lines"]}

    for case in rule_result["needs_llm"]:
        candidates = list(unmatched_bank.values())
        verdict = llm_resolve_fn(case, candidates) if llm_resolve_fn else None

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

    return {"matches": matches, "exceptions": exceptions, "audit_trail": audit}


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


def summarize(orders: list[dict], result: dict) -> dict:
    total = len(orders)
    matched = len(result["matches"])
    by_method = {}
    for m in result["matches"]:
        by_method[m["method"]] = by_method.get(m["method"], 0) + 1

    amount_by_order_id = {o["order_id"]: o["amount"] for o in orders}

    total_amount_matched = round(
        sum(amount_by_order_id.get(m["order_id"], 0.0) for m in result["matches"]), 2)

    # One order can carry more than one exception (e.g. a duplicate
    # settlement flagged alongside the order's own primary outcome) -- sum
    # each order's amount once, not once per exception, so this can't
    # overstate exposure. Exceptions with no order_id (an orphan bank line)
    # carry their own "amount" and are added on top, since they're real
    # money movements not attributable to any order.
    excepted_order_ids = {e["order_id"] for e in result["exceptions"] if e.get("order_id")}
    total_amount_at_risk = round(
        sum(amount_by_order_id.get(oid, 0.0) for oid in excepted_order_ids)
        + sum(e.get("amount", 0.0) for e in result["exceptions"] if not e.get("order_id")), 2)

    return {
        "total_orders": total,
        "matched": matched,
        "match_rate": round(matched / total, 4) if total else 0.0,
        "by_method": by_method,
        "exception_count": len(result["exceptions"]),
        "total_amount_matched": total_amount_matched,
        "total_amount_at_risk": total_amount_at_risk,
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
    can_close = len(material) == 0
    if can_close:
        message = "No material unresolved discrepancies. Safe to close."
    else:
        message = (f"{len(material)} material exception(s) totaling "
                   f"₹{material_amount:,.0f} remain unresolved "
                   f"(materiality threshold: ₹{materiality_threshold:,.0f}).")
    return {
        "can_close": can_close,
        "materiality_threshold": materiality_threshold,
        "material_exception_count": len(material),
        "material_exception_amount": material_amount,
        "message": message,
    }

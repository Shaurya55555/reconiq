"""Deterministic reconciliation engine.

Three passes, cheapest and most certain first, so the LLM is only ever
asked to resolve what the rules genuinely cannot:

  1. exact   - settlement UTR found verbatim in a bank narration, amount
               matches to the paisa.
  2. fuzzy   - UTR found verbatim but amount differs by a plausible
               Razorpay fee (<=3%), or settlement date drifted.
  3. llm     - no settlement's UTR appears in any remaining bank line
               (garbled/truncated narration) -> handed to the LLM resolver,
               which must reason over free text, not just string-match.

Anything left after all three passes is an honest exception with a
reason code, never silently dropped.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

FEE_TOLERANCE_PCT = 0.03
DATE_DRIFT_OK_DAYS = 3


def _parse_date(s: str) -> date:
    return datetime.fromisoformat(s).date() if "T" not in s else datetime.fromisoformat(s).date()


def reconcile(orders: list[dict], settlements: list[dict], bank_lines: list[dict],
              llm_resolve_fn=None) -> dict[str, Any]:
    audit: list[dict] = []
    matches: list[dict] = []
    exceptions: list[dict] = []

    orders_by_payment = {}
    for o in orders:
        orders_by_payment.setdefault(o["razorpay_payment_id"], []).append(o)

    settlements_by_payment: dict[str, list[dict]] = {}
    for s in settlements:
        settlements_by_payment.setdefault(s["payment_id"], []).append(s)

    unmatched_bank = {b["line_id"]: b for b in bank_lines}
    resolved_settlement_ids: set[str] = set()
    needs_llm: list[dict] = []

    def log(step: str, decision: str, method: str, confidence: float, note: str, **ids):
        audit.append({
            "step": step, "decision": decision, "method": method,
            "confidence": confidence, "note": note, **ids,
        })

    # ---- pass 1 & 2: rule-based, per payment_id ----
    for payment_id, orders_for_payment in orders_by_payment.items():
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
        for stl in candidate_settlements:
            if stl["settlement_id"] in resolved_settlement_ids:
                continue
            bank_hit = next((b for b in unmatched_bank.values() if stl["utr"] in b["narration"]), None)
            if bank_hit is None:
                continue

            amount_diff_pct = abs(stl["amount"] - order["amount"]) / order["amount"]
            date_drift = abs((_parse_date(stl["settled_at"]) - _parse_date(order["created_at"])).days)

            if amount_diff_pct < 0.001 and bank_hit["amount"] == stl["amount"]:
                picked = (stl, bank_hit, "exact", 1.0, "UTR + amount matched exactly")
            elif amount_diff_pct <= FEE_TOLERANCE_PCT and bank_hit["amount"] == stl["amount"]:
                picked = (stl, bank_hit, "fuzzy", 0.9,
                          f"UTR matched; amount differs by {amount_diff_pct:.1%}, within fee tolerance")
            elif date_drift > DATE_DRIFT_OK_DAYS and bank_hit["amount"] == stl["amount"]:
                picked = (stl, bank_hit, "fuzzy", 0.85,
                          f"UTR + amount matched; settlement date drifted {date_drift}d")
            else:
                continue
            break

        if picked:
            stl, bank_hit, method, confidence, note = picked
            matches.append({
                "order_id": order["order_id"], "settlement_id": stl["settlement_id"],
                "bank_line_id": bank_hit["line_id"], "method": method,
                "confidence": confidence, "note": note,
            })
            resolved_settlement_ids.add(stl["settlement_id"])
            unmatched_bank.pop(bank_hit["line_id"], None)
            log("match", "matched", method, confidence, note, order_id=order["order_id"])
        else:
            stl = candidate_settlements[0]
            amount_diff_pct = abs(stl["amount"] - order["amount"]) / order["amount"]
            if amount_diff_pct > FEE_TOLERANCE_PCT:
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
            "reason": f"Bank line for amount {bl['amount']} on {bl['value_date']} does not "
                      "correspond to any order in this batch.",
        })
        audit.append({
            "step": "bank_line_sweep", "decision": "exception", "method": "rule",
            "confidence": 1.0, "note": "bank line unclaimed after all passes",
            "bank_line_id": bank_line_id,
        })

    return {"matches": matches, "exceptions": exceptions, "audit_trail": audit}


def summarize(orders: list[dict], result: dict) -> dict:
    total = len(orders)
    matched = len(result["matches"])
    by_method = {}
    for m in result["matches"]:
        by_method[m["method"]] = by_method.get(m["method"], 0) + 1
    return {
        "total_orders": total,
        "matched": matched,
        "match_rate": round(matched / total, 4) if total else 0.0,
        "by_method": by_method,
        "exception_count": len(result["exceptions"]),
    }

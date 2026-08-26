"""Ground-truth scoring.

`data_gen.generate_batch` stamps every order with a `_truth` label (the
anomaly it was seeded with). That label is never fed to the matcher --
it exists purely so this module can grade the matcher's output against
what actually happened, the same way a held-out label set grades a
classifier. `match_rate` in `matcher.summarize` answers "how much got
matched"; this answers the harder question: "how much got matched (or
excepted) *correctly*."
"""
from __future__ import annotations

from typing import Any

# What SHOULD happen to an order, given the anomaly it was seeded with.
# "matched" orders should end up in `matches` regardless of which pass
# resolved them; "exception" orders should end up in `exceptions` with
# the specific reason code noted.
EXPECTED_OUTCOME = {
    "clean": "matched",
    "fee_adjusted": "matched",
    "date_shifted": "matched",
    "garbled_narration": "matched",
    "duplicate_settlement": "matched",
    "missing_settlement": "exception",
    "amount_mismatch": "exception",
}

EXPECTED_EXCEPTION_TYPE = {
    "missing_settlement": "no_settlement_found",
    "amount_mismatch": "amount_mismatch",
}


def score_against_ground_truth(orders: list[dict], final_result: dict[str, Any]) -> dict[str, Any]:
    matched_order_ids = {m["order_id"] for m in final_result["matches"]}

    # first exception recorded per order_id, for orders that have one
    exception_type_by_order: dict[str, str] = {}
    for e in final_result["exceptions"]:
        oid = e.get("order_id")
        if oid and oid not in exception_type_by_order:
            exception_type_by_order[oid] = e["type"]

    rows = []
    for order in orders:
        truth = order.get("_truth", "unknown")
        expected_outcome = EXPECTED_OUTCOME.get(truth, "matched")
        expected_exception_type = EXPECTED_EXCEPTION_TYPE.get(truth)

        if order["order_id"] in matched_order_ids:
            actual_outcome = "matched"
        elif order["order_id"] in exception_type_by_order:
            actual_outcome = "exception"
        else:
            actual_outcome = "unresolved"  # should never happen; scored as wrong if so

        correct = actual_outcome == expected_outcome
        if correct and expected_outcome == "exception" and expected_exception_type:
            correct = exception_type_by_order.get(order["order_id"]) == expected_exception_type

        rows.append({
            "order_id": order["order_id"], "truth": truth, "amount": order["amount"],
            "expected_outcome": expected_outcome, "actual_outcome": actual_outcome,
            "actual_exception_type": exception_type_by_order.get(order["order_id"]),
            "correct": correct,
        })

    total = len(rows)
    correct_count = sum(1 for r in rows if r["correct"])

    by_category: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = by_category.setdefault(r["truth"], {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(r["correct"])

    # Money-weighted accuracy: transaction-count accuracy treats a ₹200
    # order and a ₹2,00,000 order identically. A finance controller doesn't
    # -- a wrong decision on the big one is a materially different problem
    # than a wrong decision on the small one. Two failure modes are also
    # distinguished, since they're not equally bad:
    #  - false clear: expected an exception, but it got matched anyway --
    #    confidently wrong, money moves that shouldn't have been cleared.
    #  - safe miss: expected a match, but it landed as an exception -- the
    #    conservative failure, money sits flagged instead of moving wrong.
    total_amount = sum(r["amount"] for r in rows)
    correct_amount = sum(r["amount"] for r in rows if r["correct"])
    false_clear_amount = sum(r["amount"] for r in rows
                              if not r["correct"] and r["expected_outcome"] == "exception"
                              and r["actual_outcome"] == "matched")
    safe_miss_amount = sum(r["amount"] for r in rows
                            if not r["correct"] and r["expected_outcome"] == "matched"
                            and r["actual_outcome"] != "matched")

    return {
        "ground_truth_accuracy": round(correct_count / total, 4) if total else 0.0,
        "correct": correct_count,
        "total": total,
        "by_category": by_category,
        "misclassified": [r for r in rows if not r["correct"]],
        "amount_accuracy": round(correct_amount / total_amount, 4) if total_amount else 0.0,
        "false_clear_amount": round(false_clear_amount, 2),
        "safe_miss_amount": round(safe_miss_amount, 2),
    }

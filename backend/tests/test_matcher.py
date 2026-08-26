import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data_gen, llm_resolver, matcher, scoring


def run_pipeline(n_orders=120, seed=42):
    batch = data_gen.generate_batch(n_orders=n_orders, seed=seed)
    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"])
    final = matcher.apply_llm_resolutions(rule_result, llm_resolver.resolve)
    return batch, rule_result, final


def test_every_order_is_accounted_for():
    """Every order must end up either matched or explicitly excepted --
    the exact bug class this pipeline was built to avoid (silent drops)."""
    batch, _, final = run_pipeline()
    matched_orders = {m["order_id"] for m in final["matches"]}
    excepted_orders = {e["order_id"] for e in final["exceptions"] if "order_id" in e}
    all_orders = {o["order_id"] for o in batch["orders"]}
    assert matched_orders | excepted_orders == all_orders


def test_clean_and_fee_adjusted_orders_match_via_rules_only():
    batch, rule_result, _ = run_pipeline()
    rule_matched = {m["order_id"] for m in rule_result["matches"]}
    for order in batch["orders"]:
        if order["_truth"] in ("clean", "fee_adjusted", "date_shifted"):
            assert order["order_id"] in rule_matched, order["order_id"]


def test_missing_settlement_is_an_honest_exception():
    batch, _, final = run_pipeline()
    exception_by_order = {e["order_id"]: e for e in final["exceptions"] if "order_id" in e}
    for order in batch["orders"]:
        if order["_truth"] == "missing_settlement":
            assert exception_by_order[order["order_id"]]["type"] == "no_settlement_found"


def test_amount_mismatch_is_flagged_not_silently_matched():
    batch, _, final = run_pipeline()
    matched_orders = {m["order_id"] for m in final["matches"]}
    exception_by_order = {e["order_id"]: e for e in final["exceptions"] if "order_id" in e}
    for order in batch["orders"]:
        if order["_truth"] == "amount_mismatch":
            assert order["order_id"] not in matched_orders
            assert exception_by_order[order["order_id"]]["type"] == "amount_mismatch"


def test_garbled_narration_is_deferred_to_llm_layer():
    batch, rule_result, _ = run_pipeline()
    deferred_orders = {c["order_id"] for c in rule_result["needs_llm"]}
    for order in batch["orders"]:
        if order["_truth"] == "garbled_narration":
            assert order["order_id"] in deferred_orders


def test_heuristic_resolver_matches_garbled_utr_by_amount_and_similarity():
    case = {"order_id": "ORD1", "settlement_id": "stl1", "expected_utr": "UTR123456789",
             "expected_amount": 1000.0, "customer": "Test User"}
    candidates = [
        {"line_id": "bl1", "amount": 1000.0, "value_date": "2026-08-05", "narration": "NEFT/12345.../CUST TXN"},
        {"line_id": "bl2", "amount": 500.0, "value_date": "2026-08-05", "narration": "IMPS-987654321-SETTLEMENT RZP"},
    ]
    verdict = llm_resolver.heuristic_resolve(case, candidates)
    assert verdict["bank_line_id"] == "bl1"
    assert verdict["confidence"] >= 0.5


def test_heuristic_resolver_refuses_when_no_amount_matches():
    case = {"order_id": "ORD1", "settlement_id": "stl1", "expected_utr": "UTR123456789",
             "expected_amount": 1000.0, "customer": "Test User"}
    candidates = [{"line_id": "bl2", "amount": 999.0, "value_date": "2026-08-05", "narration": "IMPS-x"}]
    verdict = llm_resolver.heuristic_resolve(case, candidates)
    assert verdict["bank_line_id"] is None
    assert verdict["confidence"] == 0.0


def test_summary_match_rate_is_consistent_with_matches():
    batch, _, final = run_pipeline()
    summary = matcher.summarize(batch["orders"], final)
    assert summary["matched"] == len(final["matches"])
    assert summary["match_rate"] == round(summary["matched"] / summary["total_orders"], 4)


def test_deterministic_given_same_seed():
    _, _, final_a = run_pipeline(seed=7)
    _, _, final_b = run_pipeline(seed=7)
    assert len(final_a["matches"]) == len(final_b["matches"])
    assert len(final_a["exceptions"]) == len(final_b["exceptions"])


def test_ground_truth_accuracy_is_high_and_misclassifications_are_explained():
    batch, _, final = run_pipeline(n_orders=200, seed=99)
    result = scoring.score_against_ground_truth(batch["orders"], final)
    assert result["total"] == 200
    assert result["ground_truth_accuracy"] >= 0.90
    for row in result["misclassified"]:
        assert row["expected_outcome"] != row["actual_outcome"] or row["actual_exception_type"] is not None


def test_duplicate_payment_id_across_orders_is_an_explicit_exception_not_a_silent_drop():
    orders = [
        {"order_id": "ORD_A", "customer": "X", "amount": 100.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_shared"},
        {"order_id": "ORD_B", "customer": "Y", "amount": 200.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_shared"},
    ]
    result = matcher.reconcile(orders, [], [])
    exception_order_ids = {e["order_id"] for e in result["exceptions"]}
    assert "ORD_B" in exception_order_ids
    dup_exceptions = [e for e in result["exceptions"] if e["order_id"] == "ORD_B"]
    assert any(e["type"] == "duplicate_order_reference" for e in dup_exceptions)

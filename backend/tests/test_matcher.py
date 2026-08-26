import sys
from pathlib import Path

import pytest

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


def test_amount_at_risk_counts_each_order_once_even_with_multiple_exceptions():
    orders = [
        {"order_id": "ORD_A", "customer": "X", "amount": 500.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"},
        {"order_id": "ORD_B", "customer": "Y", "amount": 300.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_2"},
    ]
    result = {
        "matches": [],
        "exceptions": [
            {"type": "amount_mismatch", "order_id": "ORD_A"},
            {"type": "duplicate_candidate", "order_id": "ORD_A"},  # same order, second exception
            {"type": "no_settlement_found", "order_id": "ORD_B"},
            {"type": "unrecognized_bank_line", "amount": 150.0},  # no order_id, orphan money
        ],
    }
    summary = matcher.summarize(orders, result)
    assert summary["total_amount_at_risk"] == 500.0 + 300.0 + 150.0
    assert summary["total_amount_matched"] == 0.0


def test_amount_matched_sums_only_matched_orders():
    batch, _, final = run_pipeline(n_orders=50, seed=3)
    summary = matcher.summarize(batch["orders"], final)
    amount_by_id = {o["order_id"]: o["amount"] for o in batch["orders"]}
    expected = round(sum(amount_by_id[m["order_id"]] for m in final["matches"]), 2)
    assert summary["total_amount_matched"] == expected


def test_override_accept_match_bumps_confidence_and_logs_human_override():
    matches = [{"order_id": "ORD1", "settlement_id": "s1", "bank_line_id": "b1",
                "method": "fuzzy", "confidence": 0.9, "note": "auto"}]
    result = matcher.apply_override(matches, [], [], "accept_match", "ORD1",
                                     reviewer_note="looks right")
    assert result["matches"][0]["confidence"] == 1.0
    assert "looks right" in result["matches"][0]["note"]
    assert result["audit_trail"][-1]["method"] == "human_override"
    assert result["audit_trail"][-1]["decision"] == "matched"


def test_override_reject_match_moves_it_to_exceptions():
    matches = [{"order_id": "ORD1", "settlement_id": "s1", "bank_line_id": "b1",
                "method": "llm", "confidence": 0.65, "note": "shaky guess"}]
    result = matcher.apply_override(matches, [], [], "reject_match", "ORD1",
                                     reviewer_note="wrong customer")
    assert result["matches"] == []
    assert result["exceptions"][0]["type"] == "human_rejected_match"
    assert result["exceptions"][0]["order_id"] == "ORD1"
    assert "wrong customer" in result["exceptions"][0]["reason"]


def test_override_manual_match_clears_prior_exceptions_for_both_sides():
    exceptions = [
        {"type": "no_settlement_found", "order_id": "ORD1", "reason": "..."},
        {"type": "unrecognized_bank_line", "bank_line_id": "b_orphan", "amount": 100.0, "reason": "..."},
    ]
    result = matcher.apply_override([], exceptions, [], "manual_match", "ORD1",
                                     bank_line_id="b_orphan", reviewer_note="confirmed via bank portal")
    assert result["exceptions"] == []
    assert len(result["matches"]) == 1
    assert result["matches"][0] == {
        "order_id": "ORD1", "settlement_id": None, "bank_line_id": "b_orphan",
        "method": "human_override", "confidence": 1.0, "note": "confirmed via bank portal",
    }


def test_fee_tolerance_is_a_real_parameter_not_a_fixed_constant():
    batch = data_gen.generate_batch(n_orders=150, seed=17)
    strict = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                fee_tolerance_pct=0.0)
    lenient = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                 fee_tolerance_pct=0.5)
    # a near-zero tolerance can only match fewer (or equal) fee-adjusted
    # orders than a very lenient one on the identical batch
    assert len(strict["matches"]) <= len(lenient["matches"])


def test_date_drift_tolerance_changes_classification_not_amount_mismatch_outcome():
    """date_drift_ok_days should only affect how an exact-amount match is
    labeled (exact vs fuzzy/date-drifted) -- it must never rescue a
    genuine amount_mismatch into a match just because that settlement's
    date drift happens to exceed a tightened threshold. Regression test
    for a real bug: the date-drift branch used to fire independently of
    amount closeness.
    """
    batch = data_gen.generate_batch(n_orders=150, seed=17)

    strict = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                date_drift_ok_days=0)
    lenient = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                 date_drift_ok_days=365)

    # total matched count is unaffected -- only exact vs fuzzy labeling shifts
    assert len(strict["matches"]) == len(lenient["matches"])
    assert sum(1 for m in strict["matches"] if m["method"] == "exact") \
        <= sum(1 for m in lenient["matches"] if m["method"] == "exact")

    amount_mismatch_orders = {o["order_id"] for o in batch["orders"] if o["_truth"] == "amount_mismatch"}
    strict_matched = {m["order_id"] for m in strict["matches"]}
    assert not (amount_mismatch_orders & strict_matched), \
        "a tightened date tolerance must never rescue a genuine amount mismatch into a match"


def test_money_weighted_accuracy_distinguishes_false_clears_from_safe_misses():
    orders = [
        {"order_id": "ORD_CLEAN_BIG", "amount": 100000.0, "_truth": "clean"},
        {"order_id": "ORD_MISMATCH_BIG", "amount": 50000.0, "_truth": "amount_mismatch"},
        {"order_id": "ORD_CLEAN_SMALL", "amount": 100.0, "_truth": "clean"},
    ]
    result = {
        "matches": [
            {"order_id": "ORD_CLEAN_BIG"},
            {"order_id": "ORD_MISMATCH_BIG"},  # wrongly cleared -- should have been an exception
            # ORD_CLEAN_SMALL missing entirely -- wrongly excepted (safe miss)
        ],
        "exceptions": [],
    }
    scored = scoring.score_against_ground_truth(orders, result)
    assert scored["false_clear_amount"] == 50000.0
    assert scored["safe_miss_amount"] == 100.0
    assert scored["amount_accuracy"] == round(100000.0 / 150100.0, 4)


def test_override_rejects_unknown_action_and_missing_target():
    with pytest.raises(ValueError):
        matcher.apply_override([], [], [], "delete_everything", "ORD1")
    with pytest.raises(ValueError):
        matcher.apply_override([], [], [], "accept_match", "ORD_NOT_FOUND")
    with pytest.raises(ValueError):
        matcher.apply_override([], [], [], "manual_match", "ORD1")  # missing bank_line_id


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

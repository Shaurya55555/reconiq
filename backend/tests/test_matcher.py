import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data_gen, llm_resolver, matcher, scoring


def run_pipeline(n_orders=120, seed=42):
    batch = data_gen.generate_batch(n_orders=n_orders, seed=seed)
    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                     refunds=batch["refunds"])
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
    """A refunded (method="refunded") order contributes zero net settlement
    money even though it's matched -- nothing settled for it. A
    partial-refund order contributes order.amount minus its refund, not
    the raw order amount -- that's genuinely what settled."""
    batch, _, final = run_pipeline(n_orders=50, seed=3)
    summary = matcher.summarize(batch["orders"], final, refunds=batch["refunds"])
    amount_by_id = {o["order_id"]: o["amount"] for o in batch["orders"]}
    payment_id_by_order_id = {o["order_id"]: o["razorpay_payment_id"] for o in batch["orders"]}
    refund_by_payment: dict[str, float] = {}
    for r in batch["refunds"]:
        refund_by_payment[r["payment_id"]] = refund_by_payment.get(r["payment_id"], 0.0) + r["amount"]

    expected = 0.0
    for m in final["matches"]:
        if m["method"] == "refunded":
            continue
        refund = refund_by_payment.get(payment_id_by_order_id[m["order_id"]], 0.0)
        expected += max(amount_by_id[m["order_id"]] - refund, 0.0)
    assert summary["total_amount_matched"] == round(expected, 2)


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


def test_corruption_rate_controls_the_clean_fraction():
    clean_batch = data_gen.generate_batch(n_orders=300, seed=1, corruption_rate=0.05)
    dirty_batch = data_gen.generate_batch(n_orders=300, seed=1, corruption_rate=0.6)

    clean_fraction_a = sum(1 for o in clean_batch["orders"] if o["_truth"] == "clean") / 300
    clean_fraction_b = sum(1 for o in dirty_batch["orders"] if o["_truth"] == "clean") / 300

    assert clean_fraction_a > 0.85  # ~95% clean requested
    assert clean_fraction_b < 0.55  # ~40% clean requested
    assert clean_fraction_a > clean_fraction_b


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


# ---- many-to-one (split settlement) group matching ----

def test_split_settlement_orders_are_resolved_via_group_split_deterministically():
    """split_settlement is generated as one settlement whose money arrives
    as two separate bank lines sharing its UTR. This must resolve without
    any LLM at all -- the shared UTR makes it a rules problem, not a
    reasoning problem."""
    batch = data_gen.generate_batch(n_orders=200, seed=99)
    split_order_ids = {o["order_id"] for o in batch["orders"] if o["_truth"] == "split_settlement"}
    assert split_order_ids, "expected at least one split_settlement order at this seed"

    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"])
    matched_by_rules = {m["order_id"]: m for m in rule_result["matches"]}

    for order_id in split_order_ids:
        assert order_id in matched_by_rules, f"{order_id} should resolve in the rule pass alone"
        assert matched_by_rules[order_id]["method"] == "group_split"
        assert len(matched_by_rules[order_id]["bank_line_ids"]) >= 2


def test_group_split_match_bank_line_ids_sum_to_settlement_amount():
    order = {"order_id": "ORD1", "customer": "Test", "amount": 1000.0,
             "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"}
    settlement = {"settlement_id": "stl1", "payment_id": "pay_1", "utr": "UTR555000111",
                  "amount": 1000.0, "settled_at": "2026-08-02"}
    bank_lines = [
        {"line_id": "bl_a", "narration": "NEFT-UTR555000111-RAZORPAY SETTLEMENT PART",
         "amount": 400.0, "value_date": "2026-08-02"},
        {"line_id": "bl_b", "narration": "NEFT-UTR555000111-RAZORPAY SETTLEMENT PART",
         "amount": 600.0, "value_date": "2026-08-03"},
    ]
    result = matcher.reconcile([order], [settlement], bank_lines)
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert match["method"] == "group_split"
    assert set(match["bank_line_ids"]) == {"bl_a", "bl_b"}
    assert not result["exceptions"]


def test_group_split_does_not_rescue_a_genuine_amount_mismatch():
    """A single bank line referencing the UTR with the wrong amount must
    not be silently absorbed by the group-sum search reaching for other,
    unrelated bank lines to make up the difference."""
    order = {"order_id": "ORD1", "customer": "Test", "amount": 1000.0,
             "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"}
    settlement = {"settlement_id": "stl1", "payment_id": "pay_1", "utr": "UTR555000111",
                  "amount": 1000.0, "settled_at": "2026-08-02"}
    bank_lines = [
        # references the right UTR but the amount is genuinely wrong (not a split leg)
        {"line_id": "bl_wrong", "narration": "NEFT-UTR555000111-RAZORPAY SETTLEMENT",
         "amount": 700.0, "value_date": "2026-08-02"},
        # an unrelated bank line that happens to make the sum work -- must not be grabbed
        {"line_id": "bl_unrelated", "narration": "NEFT-UTR999888777-RAZORPAY SETTLEMENT",
         "amount": 300.0, "value_date": "2026-08-02"},
    ]
    result = matcher.reconcile([order], [settlement], bank_lines)
    assert not result["matches"]
    assert result["needs_llm"] or result["exceptions"], \
        "a genuinely wrong single-line amount must surface, not vanish"


def test_group_split_only_combines_lines_sharing_the_settlement_utr():
    """Two bank lines that happen to sum to the settlement amount but do
    NOT reference its UTR must never be combined into a false group match."""
    order = {"order_id": "ORD1", "customer": "Test", "amount": 1000.0,
             "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"}
    settlement = {"settlement_id": "stl1", "payment_id": "pay_1", "utr": "UTR555000111",
                  "amount": 1000.0, "settled_at": "2026-08-02"}
    bank_lines = [
        {"line_id": "bl_a", "narration": "NEFT-UTR111222333-OTHER", "amount": 400.0, "value_date": "2026-08-02"},
        {"line_id": "bl_b", "narration": "NEFT-UTR444555666-OTHER", "amount": 600.0, "value_date": "2026-08-02"},
    ]
    result = matcher.reconcile([order], [settlement], bank_lines)
    assert not result["matches"]


# ---- many-to-one, the mirror direction (batch settlement: N orders : 1 settlement : 1 bank line) ----

def test_batch_settlement_orders_are_resolved_without_llm():
    """batch_settlement orders share one real settlement record covering
    multiple payment_ids (modelling Razorpay's actual batch-settlement
    behaviour), matched to one bank line. Must resolve entirely in the
    rule pass -- no LLM needed, since the settlement's own UTR ties the
    group to its bank line deterministically."""
    batch = data_gen.generate_batch(n_orders=200, seed=99)
    batch_order_ids = {o["order_id"] for o in batch["orders"] if o["_truth"] == "batch_settlement"}
    assert batch_order_ids, "expected at least one batch_settlement order at this seed"

    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"])
    matched_by_rules = {m["order_id"]: m for m in rule_result["matches"]}

    for order_id in batch_order_ids:
        assert order_id in matched_by_rules, f"{order_id} should resolve in the rule pass alone"
        assert matched_by_rules[order_id]["method"] == "batch_settlement"


def test_batch_settlement_matches_every_member_order_to_the_shared_settlement():
    orders = [
        {"order_id": "ORD1", "customer": "A", "amount": 400.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"},
        {"order_id": "ORD2", "customer": "B", "amount": 600.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_2"},
        {"order_id": "ORD3", "customer": "C", "amount": 300.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_3"},
    ]
    settlement = {
        "settlement_id": "stl1", "payment_id": "pay_1", "payment_ids": ["pay_1", "pay_2", "pay_3"],
        "utr": "UTR700800900", "amount": 1300.0, "settled_at": "2026-08-02",
    }
    bank_lines = [
        {"line_id": "bl_batch", "narration": "NEFT-UTR700800900-RAZORPAY SETTLEMENT BATCH 3 ORDERS",
         "amount": 1300.0, "value_date": "2026-08-02"},
    ]
    result = matcher.reconcile(orders, [settlement], bank_lines)
    assert len(result["matches"]) == 3
    assert {m["order_id"] for m in result["matches"]} == {"ORD1", "ORD2", "ORD3"}
    for m in result["matches"]:
        assert m["method"] == "batch_settlement"
        assert m["bank_line_ids"] == ["bl_batch"]
        assert m["settlement_id"] == "stl1"
    assert not result["exceptions"]


def test_batch_settlement_does_not_falsely_match_when_amounts_dont_sum():
    """If the individual orders' amounts don't actually add up to the
    settlement's reported amount, that's a genuine discrepancy -- it
    must surface honestly, never be silently waved through."""
    orders = [
        {"order_id": "ORD1", "customer": "A", "amount": 400.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"},
        {"order_id": "ORD2", "customer": "B", "amount": 600.0,
         "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_2"},
    ]
    settlement = {
        "settlement_id": "stl1", "payment_id": "pay_1", "payment_ids": ["pay_1", "pay_2"],
        "utr": "UTR700800900", "amount": 1300.0,  # should be 1000.0 -- deliberately wrong
        "settled_at": "2026-08-02",
    }
    bank_lines = [
        {"line_id": "bl_batch", "narration": "NEFT-UTR700800900-RAZORPAY SETTLEMENT BATCH 2 ORDERS",
         "amount": 1300.0, "value_date": "2026-08-02"},
    ]
    result = matcher.reconcile(orders, [settlement], bank_lines)
    assert not any(m["method"] == "batch_settlement" for m in result["matches"])
    assert result["exceptions"], "a batch settlement whose total doesn't match its members must not be silently accepted"


def test_batch_and_split_settlements_coexist_in_the_same_batch():
    """The two mirror-direction group-matching passes (batch_settlement
    and group_split) must not interfere with each other when both
    anomaly types appear in the same synthetic batch."""
    batch = data_gen.generate_batch(n_orders=300, seed=7)
    truths_present = {o["_truth"] for o in batch["orders"]}
    assert "batch_settlement" in truths_present
    assert "split_settlement" in truths_present

    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"])
    final = matcher.apply_llm_resolutions(rule_result, llm_resolver.resolve)
    methods_used = {m["method"] for m in final["matches"]}
    assert "batch_settlement" in methods_used
    assert "group_split" in methods_used


# ---- exception priority classification and closing verdict ----

def test_classify_exceptions_tags_priority_by_amount():
    orders = [
        {"order_id": "ORD_HIGH", "amount": 50000.0},
        {"order_id": "ORD_MED", "amount": 2000.0},
        {"order_id": "ORD_LOW", "amount": 100.0},
    ]
    exceptions = [
        {"type": "amount_mismatch", "order_id": "ORD_HIGH", "reason": "..."},
        {"type": "amount_mismatch", "order_id": "ORD_MED", "reason": "..."},
        {"type": "amount_mismatch", "order_id": "ORD_LOW", "reason": "..."},
    ]
    classified = matcher.classify_exceptions(orders, exceptions, materiality_threshold=5000.0)
    by_order = {e["order_id"]: e for e in classified}
    assert by_order["ORD_HIGH"]["priority"] == "high"
    assert by_order["ORD_MED"]["priority"] == "medium"
    assert by_order["ORD_LOW"]["priority"] == "low"
    assert by_order["ORD_HIGH"]["amount"] == 50000.0


def test_classify_exceptions_uses_own_amount_for_orphan_bank_line_exceptions():
    exceptions = [{"type": "unrecognized_bank_line", "bank_line_id": "bl1", "amount": 8000.0, "reason": "..."}]
    classified = matcher.classify_exceptions([], exceptions, materiality_threshold=5000.0)
    assert classified[0]["priority"] == "high"
    assert classified[0]["amount"] == 8000.0


def test_closing_verdict_blocks_close_when_material_exceptions_exist():
    classified = [
        {"type": "amount_mismatch", "order_id": "ORD1", "amount": 50000.0, "priority": "high"},
        {"type": "amount_mismatch", "order_id": "ORD2", "amount": 100.0, "priority": "low"},
    ]
    verdict = matcher.closing_verdict(classified, materiality_threshold=5000.0)
    assert verdict["can_close"] is False
    assert verdict["material_exception_count"] == 1
    assert verdict["material_exception_amount"] == 50000.0
    assert verdict["total_exception_amount"] == 50100.0
    assert "50,100" in verdict["message"] and "50,000" in verdict["message"]


def test_closing_verdict_allows_close_when_only_non_material_exceptions_exist():
    classified = [{"type": "amount_mismatch", "order_id": "ORD1", "amount": 100.0, "priority": "low"}]
    verdict = matcher.closing_verdict(classified, materiality_threshold=5000.0)
    assert verdict["can_close"] is True
    assert verdict["material_exception_count"] == 0


def test_closing_verdict_allows_close_with_zero_exceptions():
    verdict = matcher.closing_verdict([], materiality_threshold=5000.0)
    assert verdict["can_close"] is True
    assert "no material" in verdict["message"].lower()


# ---- refund-aware reconciliation ----

def test_full_refund_resolves_without_settlement_and_without_exception():
    """A fully refunded order genuinely has no settlement in real Razorpay
    data -- the refund record is what explains the absence, so this must
    resolve as method="refunded", not surface as a no_settlement_found
    exception."""
    order = {"order_id": "ORD1", "customer": "Test", "amount": 1000.0,
             "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"}
    refund = {"refund_id": "rfnd1", "payment_id": "pay_1", "amount": 1000.0,
              "refunded_at": "2026-08-02T00:00:00", "type": "full"}
    result = matcher.reconcile([order], [], [], refunds=[refund])
    assert not result["exceptions"]
    assert len(result["matches"]) == 1
    assert result["matches"][0]["method"] == "refunded"


def test_partial_refund_matches_against_net_of_refund_amount():
    """A partial refund genuinely shrinks the settlement -- order.amount
    minus refund.amount -- so the settlement/bank credit for that smaller
    amount must match cleanly, not surface as a fee-tolerance-busting
    amount_mismatch."""
    order = {"order_id": "ORD1", "customer": "Test", "amount": 1000.0,
             "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"}
    refund = {"refund_id": "rfnd1", "payment_id": "pay_1", "amount": 400.0,
              "refunded_at": "2026-08-01T12:00:00", "type": "partial"}
    settlement = {"settlement_id": "stl1", "payment_id": "pay_1", "utr": "UTR555000111",
                  "amount": 600.0, "settled_at": "2026-08-02"}
    bank_lines = [{"line_id": "bl1", "narration": "NEFT-UTR555000111-RAZORPAY SETTLEMENT",
                   "amount": 600.0, "value_date": "2026-08-02"}]
    result = matcher.reconcile([order], [settlement], bank_lines, refunds=[refund])
    assert not result["exceptions"]
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert match["method"] == "exact"
    assert "refund" in match["note"].lower()


def test_refund_does_not_mask_a_genuine_amount_mismatch():
    """A refund only explains a gap it actually accounts for. If the
    settlement amount is wrong even after netting out the refund, this
    must still surface as amount_mismatch -- the refund record must never
    become a blanket excuse for any discrepancy."""
    order = {"order_id": "ORD1", "customer": "Test", "amount": 1000.0,
             "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"}
    refund = {"refund_id": "rfnd1", "payment_id": "pay_1", "amount": 400.0,
              "refunded_at": "2026-08-01T12:00:00", "type": "partial"}
    # expected net is 600, but the settlement is only 450 -- a genuine
    # mismatch on top of the refund, not explained by it
    settlement = {"settlement_id": "stl1", "payment_id": "pay_1", "utr": "UTR555000111",
                  "amount": 450.0, "settled_at": "2026-08-02"}
    bank_lines = [{"line_id": "bl1", "narration": "NEFT-UTR555000111-RAZORPAY SETTLEMENT",
                   "amount": 450.0, "value_date": "2026-08-02"}]
    result = matcher.reconcile([order], [settlement], bank_lines, refunds=[refund])
    assert not result["matches"]
    assert any(e["type"] == "amount_mismatch" for e in result["exceptions"])


def test_partial_refund_with_no_settlement_still_surfaces_as_exception():
    """A refund that doesn't cover the full order amount can't explain a
    missing settlement on its own -- this must stay a no_settlement_found
    exception, not be waved through as "refunded"."""
    order = {"order_id": "ORD1", "customer": "Test", "amount": 1000.0,
             "created_at": "2026-08-01T00:00:00", "razorpay_payment_id": "pay_1"}
    refund = {"refund_id": "rfnd1", "payment_id": "pay_1", "amount": 400.0,
              "refunded_at": "2026-08-01T12:00:00", "type": "partial"}
    result = matcher.reconcile([order], [], [], refunds=[refund])
    assert not result["matches"]
    assert any(e["type"] == "no_settlement_found" for e in result["exceptions"])


def test_summarize_excludes_refunded_orders_from_amount_reconciled():
    """A fully refunded order's original amount must never be counted as
    reconciled settlement money -- nothing settled for it. It should show
    up only in total_amount_refunded, not total_amount_matched."""
    orders = [{"order_id": "ORD1", "amount": 1000.0, "razorpay_payment_id": "pay_1"}]
    result = {
        "matches": [{"order_id": "ORD1", "method": "refunded", "settlement_id": None,
                     "bank_line_id": None, "confidence": 1.0, "note": "fully refunded"}],
        "exceptions": [],
    }
    refunds = [{"refund_id": "rfnd1", "payment_id": "pay_1", "amount": 1000.0,
                "refunded_at": "2026-08-02", "type": "full"}]
    summary = matcher.summarize(orders, result, refunds=refunds)
    assert summary["total_amount_matched"] == 0.0
    assert summary["total_amount_refunded"] == 1000.0


def test_generated_refund_and_partial_refund_orders_resolve_correctly_end_to_end():
    """Full pipeline sanity check: refunded/partial_refund orders seeded by
    data_gen must actually resolve as "matched" (via method "refunded" or
    a normal settlement method), never as an exception, once refunds are
    threaded through reconcile()."""
    batch, _, final = run_pipeline(n_orders=250, seed=99)
    matched_order_ids = {m["order_id"] for m in final["matches"]}
    refunded_order_ids = {o["order_id"] for o in batch["orders"] if o["_truth"] == "refunded"}
    partial_refund_order_ids = {o["order_id"] for o in batch["orders"] if o["_truth"] == "partial_refund"}
    assert refunded_order_ids, "expected at least one fully refunded order at this seed"
    assert partial_refund_order_ids, "expected at least one partially refunded order at this seed"
    assert refunded_order_ids <= matched_order_ids
    assert partial_refund_order_ids <= matched_order_ids


# ---- confidence-threshold calibration plumbing ----

def test_apply_llm_resolutions_matches_resolve_then_apply_threshold_split():
    """apply_llm_resolutions must be exactly equivalent to calling
    resolve_llm_verdicts followed by apply_confidence_threshold with the
    same threshold -- the calibration sweep depends on this split being a
    pure refactor, not a behavior change."""
    batch = data_gen.generate_batch(n_orders=150, seed=17)
    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                     refunds=batch["refunds"])

    combined = matcher.apply_llm_resolutions(rule_result, llm_resolver.resolve, confidence_threshold=0.6)

    case_verdicts = matcher.resolve_llm_verdicts(rule_result, llm_resolver.resolve)
    split = matcher.apply_confidence_threshold(rule_result, case_verdicts, confidence_threshold=0.6)

    assert {m["order_id"] for m in combined["matches"]} == {m["order_id"] for m in split["matches"]}
    assert len(combined["exceptions"]) == len(split["exceptions"])


def test_stricter_confidence_threshold_never_accepts_more_llm_matches():
    batch = data_gen.generate_batch(n_orders=150, seed=17)
    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                     refunds=batch["refunds"])
    case_verdicts = matcher.resolve_llm_verdicts(rule_result, llm_resolver.resolve)

    strict = matcher.apply_confidence_threshold(rule_result, case_verdicts, confidence_threshold=0.9)
    lenient = matcher.apply_confidence_threshold(rule_result, case_verdicts, confidence_threshold=0.5)

    strict_llm_matches = sum(1 for m in strict["matches"] if m["method"] == "llm")
    lenient_llm_matches = sum(1 for m in lenient["matches"] if m["method"] == "llm")
    assert strict_llm_matches <= lenient_llm_matches

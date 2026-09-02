import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_run_endpoint_returns_full_stateless_payload():
    res = client.post("/api/run", json={"n_orders": 40, "seed": 11})
    assert res.status_code == 200
    body = res.json()
    for key in ("run_id", "summary", "orders", "settlements", "bank_lines",
                "matches", "exceptions", "audit_trail", "ground_truth", "verdict"):
        assert key in body
    assert len(body["orders"]) == 40
    assert body["summary"]["total_orders"] == 40
    assert "can_close" in body["verdict"]
    for e in body["exceptions"]:
        assert "priority" in e and e["priority"] in ("high", "medium", "low")
        assert "amount" in e
    assert "llm_elapsed_seconds" in body["summary"]
    assert 0.0 <= body["summary"]["llm_elapsed_seconds"] <= body["summary"]["elapsed_seconds"]


def test_run_endpoint_includes_refunds_and_resolves_them_without_exceptions():
    res = client.post("/api/run", json={"n_orders": 250, "seed": 99})
    assert res.status_code == 200
    body = res.json()
    assert "refunds" in body
    assert body["refunds"], "expected at least one refund at this seed"
    assert "total_amount_refunded" in body["summary"]

    refunded_payment_ids = {r["payment_id"] for r in body["refunds"] if r["type"] == "full"}
    refunded_order_ids = {o["order_id"] for o in body["orders"]
                           if o["razorpay_payment_id"] in refunded_payment_ids}
    matched_order_ids = {m["order_id"] for m in body["matches"]}
    assert refunded_order_ids <= matched_order_ids


def test_run_endpoint_respects_custom_materiality_threshold():
    # a very low threshold should make even small exceptions "high" priority
    # and therefore block the closing verdict, if there are any exceptions at all
    res = client.post("/api/run", json={"n_orders": 60, "seed": 21, "materiality_threshold": 1.0})
    assert res.status_code == 200
    body = res.json()
    if body["exceptions"]:
        assert any(e["priority"] == "high" for e in body["exceptions"])
        assert body["verdict"]["can_close"] is False


def test_override_endpoint_rejects_a_shaky_match_and_updates_summary():
    run = client.post("/api/run", json={"n_orders": 60, "seed": 21}).json()

    shaky = next((m for m in run["matches"] if m["confidence"] < 0.95), None)
    assert shaky is not None, "expected at least one non-exact match in this batch"

    res = client.post("/api/override", json={
        "orders": run["orders"], "matches": run["matches"], "exceptions": run["exceptions"],
        "audit_trail": run["audit_trail"], "action": "reject_match",
        "order_id": shaky["order_id"], "reviewer_note": "test rejection",
    })
    assert res.status_code == 200
    body = res.json()

    assert not any(m["order_id"] == shaky["order_id"] for m in body["matches"])
    rejected = [e for e in body["exceptions"] if e.get("order_id") == shaky["order_id"]]
    assert rejected and rejected[0]["type"] == "human_rejected_match"

    last_audit = body["audit_trail"][-1]
    assert last_audit["method"] == "human_override"
    assert last_audit["order_id"] == shaky["order_id"]

    assert body["summary"]["matched"] == run["summary"]["matched"] - 1


def test_run_endpoint_includes_rules_only_counterfactual():
    run = client.post("/api/run", json={"n_orders": 100, "seed": 5}).json()
    rules_only = run["rules_only_summary"]

    assert "llm" not in rules_only["by_method"]
    # rules-only can only match fewer or equal orders than rules+AI on the same batch
    assert rules_only["matched"] <= run["summary"]["matched"]
    assert rules_only["exception_count"] >= run["summary"]["exception_count"]


def test_run_upload_endpoint_reconciles_hand_built_data_with_no_ground_truth():
    orders = [{"order_id": "ORD1", "amount": 1000.0, "razorpay_payment_id": "pay_1",
               "created_at": "2026-08-01T00:00:00", "customer": "Test"}]
    settlements = [{"settlement_id": "stl1", "payment_id": "pay_1", "utr": "UTR123456789",
                     "amount": 1000.0, "settled_at": "2026-08-02"}]
    bank_lines = [{"line_id": "bl1", "narration": "NEFT-UTR123456789-RAZORPAY",
                    "amount": 1000.0, "value_date": "2026-08-02"}]

    res = client.post("/api/run-upload", json={
        "orders": orders, "settlements": settlements, "bank_lines": bank_lines,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["total_orders"] == 1
    assert body["summary"]["matched"] == 1
    assert "ground_truth_accuracy" not in body["summary"]  # no fabricated accuracy claim


def test_run_upload_endpoint_accepts_a_comma_delimited_payment_ids_column():
    """A single real Razorpay settlement can cover more than one payment
    (a batch settlement) -- the upload path previously had no way to
    represent that as one row, only one payment_id per settlement. This
    covers the fix: a CSV upload's settlement row carries a payment_ids
    column (comma-delimited, as it would arrive from a quoted CSV field)
    and both orders resolve via the batch_settlement method against one
    bank line, not as two separate exceptions."""
    orders = [
        {"order_id": "ORD1", "amount": 600.0, "razorpay_payment_id": "pay_1",
         "created_at": "2026-08-01T00:00:00", "customer": "A"},
        {"order_id": "ORD2", "amount": 400.0, "razorpay_payment_id": "pay_2",
         "created_at": "2026-08-01T00:00:00", "customer": "B"},
    ]
    settlements = [
        {"settlement_id": "stl1", "payment_id": "pay_1", "payment_ids": "pay_1, pay_2",
         "utr": "UTR999888777", "amount": 1000.0, "settled_at": "2026-08-02"},
    ]
    bank_lines = [
        {"line_id": "bl1", "narration": "NEFT-UTR999888777-RAZORPAY",
         "amount": 1000.0, "value_date": "2026-08-02"},
    ]

    res = client.post("/api/run-upload", json={
        "orders": orders, "settlements": settlements, "bank_lines": bank_lines,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["matched"] == 2
    methods = {m["order_id"]: m["method"] for m in body["matches"]}
    assert methods == {"ORD1": "batch_settlement", "ORD2": "batch_settlement"}


def test_run_upload_endpoint_rejects_missing_required_columns():
    res = client.post("/api/run-upload", json={
        "orders": [{"order_id": "ORD1", "amount": 100.0}],  # missing required fields
        "settlements": [], "bank_lines": [],
    })
    assert res.status_code == 400
    assert "missing required column" in res.json()["detail"]


def test_benchmark_endpoint_runs_across_corruption_rates():
    res = client.post("/api/benchmark", json={
        "n_orders": 60, "corruption_rates": [0.1, 0.3, 0.5], "seed_base": 100, "use_llm": False,
    })
    assert res.status_code == 200
    body = res.json()
    assert len(body["results"]) == 3
    for r in body["results"]:
        assert r["total_orders"] == 60
        assert 0.0 <= r["ground_truth_accuracy"] <= 1.0
    # noisier batches should generally produce more exceptions, not fewer
    assert body["results"][0]["exception_count"] <= body["results"][-1]["exception_count"]


def test_benchmark_endpoint_averages_multiple_seeds_per_corruption_level():
    """A single batch at one corruption rate could be a lucky or unlucky
    draw -- this locks in that /api/benchmark now runs n_seeds independent
    batches per rate and reports mean + spread, not one batch's numbers."""
    res = client.post("/api/benchmark", json={
        "n_orders": 60, "corruption_rates": [0.2], "seed_base": 5, "n_seeds": 4, "use_llm": False,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["n_seeds"] == 4
    r = body["results"][0]
    assert r["n_seeds"] == 4
    for key in ("match_rate", "ground_truth_accuracy", "exception_count"):
        assert f"{key}_min" in r and f"{key}_max" in r
        assert r[f"{key}_min"] <= r[key] <= r[f"{key}_max"]


def test_calibrate_endpoint_sweeps_thresholds_on_one_fixed_batch():
    res = client.post("/api/calibrate", json={
        "n_orders": 150, "seed": 17, "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
    })
    assert res.status_code == 200
    body = res.json()
    assert len(body["results"]) == 5
    # thresholds come back sorted, and a stricter bar can only auto-accept
    # fewer (or equal) LLM cases than a looser one on the same fixed verdicts
    thresholds = [r["confidence_threshold"] for r in body["results"]]
    assert thresholds == sorted(thresholds)
    accepted_counts = [r["llm_cases_auto_accepted"] for r in body["results"]]
    assert accepted_counts == sorted(accepted_counts, reverse=True)
    for r in body["results"]:
        assert 0.0 <= r["ground_truth_accuracy"] <= 1.0
        assert r["llm_cases_auto_accepted"] <= body["needs_llm_count"]


def test_override_endpoint_returns_400_for_unknown_action():
    run = client.post("/api/run", json={"n_orders": 20, "seed": 4}).json()
    res = client.post("/api/override", json={
        "orders": run["orders"], "matches": run["matches"], "exceptions": run["exceptions"],
        "audit_trail": run["audit_trail"], "action": "delete_everything", "order_id": "ORD1001",
    })
    assert res.status_code == 400


def test_ask_endpoint_can_answer_about_a_matched_order_not_just_exceptions():
    """Regression test: /api/ask used to only receive summary + exceptions,
    so a question about any matched order looked like missing data even
    though the order was right there in the run. matches must be threaded
    through end to end."""
    run = client.post("/api/run", json={"n_orders": 30, "seed": 6}).json()
    matched_order_id = run["matches"][0]["order_id"]

    res = client.post("/api/ask", json={
        "question": f"what happened to {matched_order_id}?",
        "summary": run["summary"], "exceptions": run["exceptions"], "matches": run["matches"],
    })
    assert res.status_code == 200
    assert matched_order_id in res.json()["answer"]


def test_ask_endpoint_answers_closing_verdict_and_rules_vs_ai_questions():
    """Regression coverage for the chat expansion: verdict and
    rules_only_summary must flow end to end from /api/run through to
    /api/ask so questions about closing status and AI uplift are
    answerable, not just questions about a single named order."""
    run = client.post("/api/run", json={"n_orders": 30, "seed": 6}).json()

    res = client.post("/api/ask", json={
        "question": "can I close the books?",
        "summary": run["summary"], "exceptions": run["exceptions"], "matches": run["matches"],
        "verdict": run["verdict"],
    })
    assert res.status_code == 200
    assert len(res.json()["answer"]) > 0

    res2 = client.post("/api/ask", json={
        "question": "does the AI actually help here?",
        "summary": run["summary"], "exceptions": run["exceptions"], "matches": run["matches"],
        "rules_only_summary": run["rules_only_summary"],
    })
    assert res2.status_code == 200
    assert len(res2.json()["answer"]) > 0

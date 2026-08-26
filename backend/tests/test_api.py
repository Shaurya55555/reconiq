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
                "matches", "exceptions", "audit_trail", "ground_truth"):
        assert key in body
    assert len(body["orders"]) == 40
    assert body["summary"]["total_orders"] == 40


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


def test_override_endpoint_returns_400_for_unknown_action():
    run = client.post("/api/run", json={"n_orders": 20, "seed": 4}).json()
    res = client.post("/api/override", json={
        "orders": run["orders"], "matches": run["matches"], "exceptions": run["exceptions"],
        "audit_trail": run["audit_trail"], "action": "delete_everything", "order_id": "ORD1001",
    })
    assert res.status_code == 400

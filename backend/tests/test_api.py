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
    for key in ("run_id", "summary", "orders", "matches", "exceptions", "audit_trail", "ground_truth"):
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


def test_override_endpoint_returns_400_for_unknown_action():
    run = client.post("/api/run", json={"n_orders": 20, "seed": 4}).json()
    res = client.post("/api/override", json={
        "orders": run["orders"], "matches": run["matches"], "exceptions": run["exceptions"],
        "audit_trail": run["audit_trail"], "action": "delete_everything", "order_id": "ORD1001",
    })
    assert res.status_code == 400

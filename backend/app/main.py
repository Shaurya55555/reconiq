from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import data_gen, llm_resolver, matcher, scoring, store

load_dotenv()

app = FastAPI(title="ReconIQ", description="Multi-source reconciliation agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


class RunRequest(BaseModel):
    n_orders: int = 70
    seed: int | None = None
    confidence_threshold: float | None = None
    fee_tolerance_pct: float = matcher.FEE_TOLERANCE_PCT
    date_drift_ok_days: int = matcher.DATE_DRIFT_OK_DAYS


class AskRequest(BaseModel):
    question: str
    summary: dict
    exceptions: list[dict]


def _apply_ground_truth_to_summary(summary: dict, ground_truth: dict) -> None:
    summary["ground_truth_accuracy"] = ground_truth["ground_truth_accuracy"]
    summary["amount_accuracy"] = ground_truth["amount_accuracy"]
    summary["false_clear_amount"] = ground_truth["false_clear_amount"]
    summary["safe_miss_amount"] = ground_truth["safe_miss_amount"]


class BenchmarkRequest(BaseModel):
    n_orders: int = 80
    corruption_rates: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
    seed_base: int | None = None
    confidence_threshold: float | None = None
    fee_tolerance_pct: float = matcher.FEE_TOLERANCE_PCT
    date_drift_ok_days: int = matcher.DATE_DRIFT_OK_DAYS
    use_llm: bool = True


class OverrideRequest(BaseModel):
    orders: list[dict]
    matches: list[dict]
    exceptions: list[dict]
    audit_trail: list[dict]
    action: str
    order_id: str
    bank_line_id: str | None = None
    reviewer_note: str | None = None


@app.post("/api/run")
def run_reconciliation(req: RunRequest):
    """Stateless by design: the full result is returned in one response so
    the frontend never needs a follow-up lookup by run_id. This is what
    lets the same code run identically on a long-lived uvicorn process
    and on a cold-starting serverless function with no shared memory
    between invocations -- there's nothing to share.
    """
    threshold = req.confidence_threshold
    if threshold is None:
        threshold = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", matcher.DEFAULT_LLM_CONFIDENCE_THRESHOLD))

    t0 = time.perf_counter()
    batch = data_gen.generate_batch(n_orders=req.n_orders, seed=req.seed)
    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                     fee_tolerance_pct=req.fee_tolerance_pct,
                                     date_drift_ok_days=req.date_drift_ok_days)

    # Rules-only counterfactual: what the deterministic passes alone would
    # have produced, with every LLM-deferred case landing as an honest
    # exception instead of being resolved. Computed on the same batch so
    # the "why does the LLM exist" comparison is apples-to-apples, not a
    # separate cherry-picked run.
    rules_only = matcher.apply_llm_resolutions(rule_result, llm_resolve_fn=None, confidence_threshold=threshold)
    rules_only_summary = matcher.summarize(batch["orders"], rules_only)
    rules_only_ground_truth = scoring.score_against_ground_truth(batch["orders"], rules_only)
    _apply_ground_truth_to_summary(rules_only_summary, rules_only_ground_truth)

    final = matcher.apply_llm_resolutions(rule_result, llm_resolver.resolve, confidence_threshold=threshold)
    elapsed = time.perf_counter() - t0

    summary = matcher.summarize(batch["orders"], final)
    summary["throughput_records_per_sec"] = round(req.n_orders / elapsed, 1) if elapsed > 0 else None
    summary["elapsed_seconds"] = round(elapsed, 3)
    summary["llm_provider"] = os.getenv("LLM_PROVIDER", "heuristic (offline fallback)")
    summary["confidence_threshold"] = threshold
    summary["fee_tolerance_pct"] = req.fee_tolerance_pct
    summary["date_drift_ok_days"] = req.date_drift_ok_days

    ground_truth = scoring.score_against_ground_truth(batch["orders"], final)
    _apply_ground_truth_to_summary(summary, ground_truth)

    run_id = store.new_run({"summary": summary, "n_orders": req.n_orders})

    return {
        "run_id": run_id,
        "summary": summary,
        "orders": batch["orders"],
        "settlements": batch["settlements"],
        "bank_lines": batch["bank_lines"],
        "matches": final["matches"],
        "exceptions": final["exceptions"],
        "audit_trail": final["audit_trail"],
        "ground_truth": ground_truth,
        "rules_only_summary": rules_only_summary,
    }


@app.get("/api/runs")
def list_runs():
    """Best-effort local history -- populated only within a single
    long-lived process (e.g. `uvicorn` locally). On serverless this may
    be empty depending on which instance handles the request; the
    frontend never depends on it for its primary flow."""
    return [{"run_id": r["run_id"], "created_at": r["created_at"], "summary": r["summary"]}
            for r in store.list_runs()]


@app.post("/api/benchmark")
def benchmark(req: BenchmarkRequest):
    """Runs the real pipeline across a range of corruption rates on freshly
    generated batches, so the reported accuracy can be shown holding up (or
    degrading) as data gets messier -- one run at one corruption level
    proves nothing on its own; a curve across several does.
    """
    threshold = req.confidence_threshold
    if threshold is None:
        threshold = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", matcher.DEFAULT_LLM_CONFIDENCE_THRESHOLD))
    resolve_fn = llm_resolver.resolve if req.use_llm else None

    results = []
    for i, rate in enumerate(req.corruption_rates):
        seed = (req.seed_base + i) if req.seed_base is not None else None
        batch = data_gen.generate_batch(n_orders=req.n_orders, seed=seed, corruption_rate=rate)
        rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                         fee_tolerance_pct=req.fee_tolerance_pct,
                                         date_drift_ok_days=req.date_drift_ok_days)
        final = matcher.apply_llm_resolutions(rule_result, resolve_fn, confidence_threshold=threshold)
        summary = matcher.summarize(batch["orders"], final)
        ground_truth = scoring.score_against_ground_truth(batch["orders"], final)
        _apply_ground_truth_to_summary(summary, ground_truth)
        results.append({"corruption_rate": rate, **summary})

    return {
        "n_orders": req.n_orders,
        "use_llm": req.use_llm,
        "llm_provider": os.getenv("LLM_PROVIDER", "heuristic (offline fallback)") if req.use_llm else "disabled for this benchmark",
        "results": results,
    }


@app.post("/api/override")
def override(req: OverrideRequest):
    """A human reviewer accepting, rejecting, or manually resolving one
    order. Stateless like everything else here: the client sends back the
    orders/matches/exceptions/audit_trail it already has from /api/run (or
    a prior /api/override), and gets the mutated version back -- there's
    no server-side run to go stale or disappear on a cold start.
    """
    try:
        result = matcher.apply_override(
            req.matches, req.exceptions, req.audit_trail,
            req.action, req.order_id, req.bank_line_id, req.reviewer_note,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    summary = matcher.summarize(req.orders, result)
    summary["llm_provider"] = os.getenv("LLM_PROVIDER", "heuristic (offline fallback)")
    ground_truth = scoring.score_against_ground_truth(req.orders, result)
    _apply_ground_truth_to_summary(summary, ground_truth)

    return {
        "summary": summary,
        "orders": req.orders,
        "matches": result["matches"],
        "exceptions": result["exceptions"],
        "audit_trail": result["audit_trail"],
        "ground_truth": ground_truth,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    answer = llm_resolver.answer_question(req.question, req.summary, req.exceptions)
    return {"answer": answer}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

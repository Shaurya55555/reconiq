from __future__ import annotations

import ast
import json
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
    materiality_threshold: float = matcher.DEFAULT_MATERIALITY_THRESHOLD


class AskRequest(BaseModel):
    question: str
    summary: dict
    exceptions: list[dict]
    matches: list[dict] = []
    verdict: dict | None = None
    rules_only_summary: dict | None = None


def _apply_ground_truth_to_summary(summary: dict, ground_truth: dict) -> None:
    summary["ground_truth_accuracy"] = ground_truth["ground_truth_accuracy"]
    summary["amount_accuracy"] = ground_truth["amount_accuracy"]
    summary["false_clear_amount"] = ground_truth["false_clear_amount"]
    summary["safe_miss_amount"] = ground_truth["safe_miss_amount"]


class BenchmarkRequest(BaseModel):
    n_orders: int = 80
    corruption_rates: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
    seed_base: int | None = None
    n_seeds: int = 3
    confidence_threshold: float | None = None
    fee_tolerance_pct: float = matcher.FEE_TOLERANCE_PCT
    date_drift_ok_days: int = matcher.DATE_DRIFT_OK_DAYS
    use_llm: bool = True


AGGREGATED_BENCHMARK_METRICS = (
    "match_rate", "ground_truth_accuracy", "amount_accuracy",
    "false_clear_amount", "safe_miss_amount", "exception_count",
)


def _aggregate_seed_runs(summaries: list[dict]) -> dict:
    """A single batch at a given corruption rate proves nothing on its own
    -- it could be a lucky (or unlucky) draw. This averages n_seeds
    independent batches at the same rate and reports the spread (min/max)
    alongside the mean, so "accuracy holds up as data gets messier" is a
    claim about many runs, not one cherry-picked one.
    """
    agg: dict = {"total_orders": summaries[0]["total_orders"]}
    for key in AGGREGATED_BENCHMARK_METRICS:
        vals = [s[key] for s in summaries]
        agg[key] = round(sum(vals) / len(vals), 4)
        agg[f"{key}_min"] = round(min(vals), 4)
        agg[f"{key}_max"] = round(max(vals), 4)
    return agg


class CalibrationRequest(BaseModel):
    n_orders: int = 150
    seed: int | None = None
    thresholds: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9]
    fee_tolerance_pct: float = matcher.FEE_TOLERANCE_PCT
    date_drift_ok_days: int = matcher.DATE_DRIFT_OK_DAYS


REQUIRED_ORDER_FIELDS = {"order_id", "amount", "razorpay_payment_id", "created_at"}
REQUIRED_SETTLEMENT_FIELDS = {"settlement_id", "payment_id", "utr", "amount", "settled_at"}
REQUIRED_BANK_LINE_FIELDS = {"line_id", "narration", "amount", "value_date"}
REQUIRED_REFUND_FIELDS = {"payment_id", "amount"}
REQUIRED_CHARGEBACK_FIELDS = {"payment_id", "amount"}


class UploadRunRequest(BaseModel):
    orders: list[dict]
    settlements: list[dict]
    bank_lines: list[dict]
    refunds: list[dict] = []
    chargebacks: list[dict] = []
    confidence_threshold: float | None = None
    fee_tolerance_pct: float = matcher.FEE_TOLERANCE_PCT
    date_drift_ok_days: int = matcher.DATE_DRIFT_OK_DAYS
    materiality_threshold: float = matcher.DEFAULT_MATERIALITY_THRESHOLD


def _validate_and_coerce(rows: list[dict], required: set[str], label: str,
                          allow_negative: bool = False) -> list[dict]:
    """allow_negative defaults to False because an order/settlement/refund/
    chargeback amount is always a magnitude in real Razorpay data -- there's
    no such thing as a negative order price or a negative refund. Only
    bank_lines legitimately carries negative amounts (an outbound debit,
    e.g. a refund leaving the account), so that's the one caller that
    passes allow_negative=True. A negative amount elsewhere is invalid
    input, not a business scenario to silently accept -- summing it in
    unchecked corrupts every total downstream (total_amount_refunded going
    negative, at-risk figures becoming internally inconsistent) instead of
    surfacing as the honest upload error it actually is.
    """
    if not rows:
        raise ValueError(f"No {label} rows provided.")
    cleaned = []
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{label} row {i + 1} is missing required column(s): {', '.join(sorted(missing))}")
        row = dict(row)
        try:
            row["amount"] = float(row["amount"])
        except (TypeError, ValueError):
            raise ValueError(f"{label} row {i + 1} has a non-numeric amount: {row.get('amount')!r}")
        if not allow_negative and row["amount"] < 0:
            raise ValueError(f"{label} row {i + 1} has a negative amount: {row['amount']!r}")
        cleaned.append(row)
    return cleaned


def _coerce_settlement_payment_ids(rows: list[dict]) -> list[dict]:
    """A real Razorpay settlement can legitimately cover more than one
    payment (a batch settlement). REQUIRED_SETTLEMENT_FIELDS still asks for
    one `payment_id` per row (backward compatible with every existing
    upload and with fetch_razorpay_settlements.py's one-row-per-payment
    workaround), but a row may also carry an optional `payment_ids` column
    -- a comma-separated list in one CSV field -- to losslessly represent a
    multi-payment settlement as a single row instead. matcher.reconcile()
    already reads `payment_ids` off a settlement dict (falling back to
    `[payment_id]` when absent); this is the upload-path half of that,
    turning the raw CSV string into the list it expects. This was the
    roadmap gap called out in README's "Offline Razorpay Settlement API
    adapter" section.
    """
    out = []
    for row in rows:
        row = dict(row)
        raw = row.get("payment_ids")
        if raw:
            raw = str(raw).strip()
            ids = None
            if raw.startswith("["):
                # A CSV cell holding a JSON array (e.g. exported by a tool
                # that serializes list-valued fields that way) is a
                # plausible real-world variant of "multiple payment_ids in
                # one cell" -- try that first, rather than silently
                # producing garbage IDs like '["pay_1"' from a naive split.
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        ids = [str(p).strip() for p in parsed if str(p).strip()]
                except (ValueError, TypeError):
                    pass
                if ids is None:
                    # Python's str(list) repr uses single quotes
                    # ("['pay_1', 'pay_2']"), which isn't valid JSON --
                    # a plausible export artifact from a tool that
                    # stringified a Python list directly instead of using
                    # json.dumps. ast.literal_eval only evaluates literals
                    # (no arbitrary code execution), safe on untrusted input.
                    try:
                        parsed = ast.literal_eval(raw)
                        if isinstance(parsed, list):
                            ids = [str(p).strip() for p in parsed if str(p).strip()]
                    except (ValueError, SyntaxError, TypeError):
                        pass
            if ids is None:
                ids = [p.strip() for p in raw.split(",") if p.strip()]
            if ids:
                row["payment_ids"] = ids
        out.append(row)
    return out


def _coerce_settlement_is_instant(rows: list[dict]) -> list[dict]:
    """Optional `is_instant` column on an uploaded settlements.csv --
    CSV values arrive as strings, but matcher.py's wider instant-
    settlement fee tolerance checks `stl.get("is_instant")` as a real
    boolean. Absent or falsy-looking values stay falsy (standard
    tolerance); only an explicit true/1/yes flips it on.
    """
    out = []
    for row in rows:
        row = dict(row)
        raw = str(row.get("is_instant", "")).strip().lower()
        if raw in ("true", "1", "yes", "y"):
            row["is_instant"] = True
        elif "is_instant" in row:
            row["is_instant"] = False
        out.append(row)
    return out


def _coerce_chargeback_fee(rows: list[dict]) -> list[dict]:
    """Optional `fee` column on an uploaded chargebacks list -- CSV/JSON
    values may arrive as strings; matcher._check_chargebacks adds this
    directly to the reversed amount, so it needs to be a real float
    before it gets there. Absent or empty means no separate fee, not an
    error.
    """
    out = []
    for row in rows:
        row = dict(row)
        raw = row.get("fee")
        if raw not in (None, ""):
            try:
                row["fee"] = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"chargebacks row has a non-numeric fee: {raw!r}")
        out.append(row)
    return out


class OverrideRequest(BaseModel):
    orders: list[dict]
    matches: list[dict]
    exceptions: list[dict]
    audit_trail: list[dict]
    action: str
    order_id: str
    bank_line_id: str | None = None
    reviewer_note: str | None = None
    materiality_threshold: float = matcher.DEFAULT_MATERIALITY_THRESHOLD


def _classify_and_verdict(orders: list[dict], exceptions: list[dict], materiality_threshold: float) -> tuple[list[dict], dict]:
    classified = matcher.classify_exceptions(orders, exceptions, materiality_threshold)
    verdict = matcher.closing_verdict(classified, materiality_threshold)
    return classified, verdict


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
                                     date_drift_ok_days=req.date_drift_ok_days,
                                     refunds=batch["refunds"])

    # Rules-only counterfactual: what the deterministic passes alone would
    # have produced, with every LLM-deferred case landing as an honest
    # exception instead of being resolved. Computed on the same batch so
    # the "why does the LLM exist" comparison is apples-to-apples, not a
    # separate cherry-picked run.
    rules_only = matcher.apply_llm_resolutions(rule_result, llm_resolve_fn=None, confidence_threshold=threshold)
    rules_only_summary = matcher.summarize(batch["orders"], rules_only, refunds=batch["refunds"],
                                            refund_matches=rules_only.get("refund_matches"))
    rules_only_ground_truth = scoring.score_against_ground_truth(batch["orders"], rules_only)
    _apply_ground_truth_to_summary(rules_only_summary, rules_only_ground_truth)

    t_llm = time.perf_counter()
    final = matcher.apply_llm_resolutions(rule_result, llm_resolver.resolve, confidence_threshold=threshold)
    llm_elapsed = time.perf_counter() - t_llm
    elapsed = time.perf_counter() - t0

    summary = matcher.summarize(batch["orders"], final, refunds=batch["refunds"],
                                 refund_matches=final.get("refund_matches"))
    summary["throughput_records_per_sec"] = round(req.n_orders / elapsed, 1) if elapsed > 0 else None
    summary["elapsed_seconds"] = round(elapsed, 3)
    summary["llm_elapsed_seconds"] = round(llm_elapsed, 3)
    summary["llm_provider"] = os.getenv("LLM_PROVIDER", "heuristic (offline fallback)")
    summary["confidence_threshold"] = threshold
    summary["fee_tolerance_pct"] = req.fee_tolerance_pct
    summary["date_drift_ok_days"] = req.date_drift_ok_days

    ground_truth = scoring.score_against_ground_truth(batch["orders"], final)
    _apply_ground_truth_to_summary(summary, ground_truth)

    classified_exceptions, verdict = _classify_and_verdict(batch["orders"], final["exceptions"], req.materiality_threshold)

    run_id = store.new_run({"summary": summary, "n_orders": req.n_orders})

    return {
        "run_id": run_id,
        "summary": summary,
        "orders": batch["orders"],
        "settlements": batch["settlements"],
        "bank_lines": batch["bank_lines"],
        "refunds": batch["refunds"],
        "matches": final["matches"],
        "exceptions": classified_exceptions,
        "verdict": verdict,
        "audit_trail": final["audit_trail"],
        "ground_truth": ground_truth,
        "rules_only_summary": rules_only_summary,
        "refund_matches": final.get("refund_matches", []),
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

    Each corruption level runs n_seeds independent batches, not one -- a
    single batch at a given rate could be a lucky or unlucky draw. Reported
    figures are the mean across those seeds, with the min/max spread
    alongside, so "accuracy holds up as data gets messier" is backed by
    several runs per point, not a single cherry-picked one.
    """
    threshold = req.confidence_threshold
    if threshold is None:
        threshold = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", matcher.DEFAULT_LLM_CONFIDENCE_THRESHOLD))
    resolve_fn = llm_resolver.resolve if req.use_llm else None
    n_seeds = max(req.n_seeds, 1)

    results = []
    for i, rate in enumerate(req.corruption_rates):
        per_seed_summaries = []
        for j in range(n_seeds):
            seed = (req.seed_base + i * 1000 + j) if req.seed_base is not None else None
            batch = data_gen.generate_batch(n_orders=req.n_orders, seed=seed, corruption_rate=rate)
            rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                             fee_tolerance_pct=req.fee_tolerance_pct,
                                             date_drift_ok_days=req.date_drift_ok_days,
                                             refunds=batch["refunds"])
            final = matcher.apply_llm_resolutions(rule_result, resolve_fn, confidence_threshold=threshold)
            summary = matcher.summarize(batch["orders"], final, refunds=batch["refunds"])
            ground_truth = scoring.score_against_ground_truth(batch["orders"], final)
            _apply_ground_truth_to_summary(summary, ground_truth)
            per_seed_summaries.append(summary)
        results.append({"corruption_rate": rate, "n_seeds": n_seeds,
                         **_aggregate_seed_runs(per_seed_summaries)})

    return {
        "n_orders": req.n_orders,
        "n_seeds": n_seeds,
        "use_llm": req.use_llm,
        "llm_provider": os.getenv("LLM_PROVIDER", "heuristic (offline fallback)") if req.use_llm else "disabled for this benchmark",
        "results": results,
    }


@app.post("/api/calibrate")
def calibrate(req: CalibrationRequest):
    """LLM_CONFIDENCE_THRESHOLD (default 0.6) has always been a reasonable
    starting point, not an empirically justified one -- this sweeps the
    auto-accept bar across a range of values against the same fixed batch
    and the same fixed round of LLM verdicts, reporting coverage vs.
    false-clear rate at each point, so the default can be defended with a
    number instead of asserted.

    Reuses one rule pass + one round of LLM calls across the whole sweep
    (matcher.resolve_llm_verdicts / apply_confidence_threshold) -- repeating
    the actual LLM calls once per threshold would be both slow and, on a
    live provider, needlessly expensive.
    """
    threshold_env_default = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", matcher.DEFAULT_LLM_CONFIDENCE_THRESHOLD))

    batch = data_gen.generate_batch(n_orders=req.n_orders, seed=req.seed)
    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"],
                                     fee_tolerance_pct=req.fee_tolerance_pct,
                                     date_drift_ok_days=req.date_drift_ok_days,
                                     refunds=batch["refunds"])
    case_verdicts = matcher.resolve_llm_verdicts(rule_result, llm_resolver.resolve)

    results = []
    for t in sorted(req.thresholds):
        final = matcher.apply_confidence_threshold(rule_result, case_verdicts, confidence_threshold=t)
        summary = matcher.summarize(batch["orders"], final, refunds=batch["refunds"])
        ground_truth = scoring.score_against_ground_truth(batch["orders"], final)
        accepted = sum(1 for _, v in case_verdicts if v and v.get("confidence", 0) >= t)
        results.append({
            "confidence_threshold": t,
            "match_rate": summary["match_rate"],
            "ground_truth_accuracy": ground_truth["ground_truth_accuracy"],
            "amount_accuracy": ground_truth["amount_accuracy"],
            "false_clear_amount": ground_truth["false_clear_amount"],
            "safe_miss_amount": ground_truth["safe_miss_amount"],
            "llm_cases_auto_accepted": accepted,
        })

    return {
        "n_orders": req.n_orders,
        "needs_llm_count": len(rule_result["needs_llm"]),
        "current_default_threshold": threshold_env_default,
        "llm_provider": os.getenv("LLM_PROVIDER", "heuristic (offline fallback)"),
        "results": results,
    }


@app.post("/api/run-upload")
def run_uploaded_data(req: UploadRunRequest):
    """Bring-your-own-data: same pipeline, same policy knobs, real uploaded
    orders/settlements/bank_lines instead of synthetic ones. No ground-truth
    scoring here -- uploaded data has no seeded _truth label, so accuracy
    against a known-correct answer genuinely cannot be computed; the
    response reports match rate, amount at risk, and the exception list,
    honestly, without a fabricated accuracy number.
    """
    try:
        orders = _validate_and_coerce(req.orders, REQUIRED_ORDER_FIELDS, "orders")
        settlements = _validate_and_coerce(req.settlements, REQUIRED_SETTLEMENT_FIELDS, "settlements")
        settlements = _coerce_settlement_payment_ids(settlements)
        settlements = _coerce_settlement_is_instant(settlements)
        bank_lines = _validate_and_coerce(req.bank_lines, REQUIRED_BANK_LINE_FIELDS, "bank_lines",
                                           allow_negative=True)
        refunds = _validate_and_coerce(req.refunds, REQUIRED_REFUND_FIELDS, "refunds") if req.refunds else []
        chargebacks = (_coerce_chargeback_fee(
            _validate_and_coerce(req.chargebacks, REQUIRED_CHARGEBACK_FIELDS, "chargebacks"))
            if req.chargebacks else [])
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    threshold = req.confidence_threshold
    if threshold is None:
        threshold = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", matcher.DEFAULT_LLM_CONFIDENCE_THRESHOLD))

    t0 = time.perf_counter()
    rule_result = matcher.reconcile(orders, settlements, bank_lines,
                                     fee_tolerance_pct=req.fee_tolerance_pct,
                                     date_drift_ok_days=req.date_drift_ok_days,
                                     refunds=refunds, chargebacks=chargebacks)
    t_llm = time.perf_counter()
    final = matcher.apply_llm_resolutions(rule_result, llm_resolver.resolve, confidence_threshold=threshold)
    llm_elapsed = time.perf_counter() - t_llm
    elapsed = time.perf_counter() - t0

    summary = matcher.summarize(orders, final, refunds=refunds, refund_matches=final.get("refund_matches"))
    summary["throughput_records_per_sec"] = round(len(orders) / elapsed, 1) if elapsed > 0 else None
    summary["elapsed_seconds"] = round(elapsed, 3)
    summary["llm_elapsed_seconds"] = round(llm_elapsed, 3)
    summary["llm_provider"] = os.getenv("LLM_PROVIDER", "heuristic (offline fallback)")
    summary["confidence_threshold"] = threshold
    summary["fee_tolerance_pct"] = req.fee_tolerance_pct
    summary["date_drift_ok_days"] = req.date_drift_ok_days

    classified_exceptions, verdict = _classify_and_verdict(orders, final["exceptions"], req.materiality_threshold)

    return {
        "summary": summary,
        "orders": orders,
        "settlements": settlements,
        "bank_lines": bank_lines,
        "refunds": refunds,
        "chargebacks": chargebacks,
        "matches": final["matches"],
        "exceptions": classified_exceptions,
        "verdict": verdict,
        "audit_trail": final["audit_trail"],
        "refund_matches": final.get("refund_matches", []),
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

    classified_exceptions, verdict = _classify_and_verdict(req.orders, result["exceptions"], req.materiality_threshold)

    return {
        "summary": summary,
        "orders": req.orders,
        "matches": result["matches"],
        "exceptions": classified_exceptions,
        "verdict": verdict,
        "audit_trail": result["audit_trail"],
        "ground_truth": ground_truth,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    answer = llm_resolver.answer_question(req.question, req.summary, req.exceptions, req.matches,
                                           req.verdict, req.rules_only_summary)
    return {"answer": answer}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

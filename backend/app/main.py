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

from . import data_gen, llm_resolver, matcher, store

load_dotenv()

app = FastAPI(title="ReconIQ", description="Multi-source reconciliation agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


class RunRequest(BaseModel):
    n_orders: int = 70
    seed: int | None = None


class AskRequest(BaseModel):
    question: str


@app.post("/api/run")
def run_reconciliation(req: RunRequest):
    t0 = time.perf_counter()
    batch = data_gen.generate_batch(n_orders=req.n_orders, seed=req.seed)
    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"])
    final = matcher.apply_llm_resolutions(rule_result, llm_resolver.resolve)
    elapsed = time.perf_counter() - t0

    summary = matcher.summarize(batch["orders"], final)
    summary["throughput_records_per_sec"] = round(req.n_orders / elapsed, 1) if elapsed > 0 else None
    summary["elapsed_seconds"] = round(elapsed, 3)
    summary["llm_provider"] = os.getenv("LLM_PROVIDER", "heuristic (offline fallback)")

    run_id = store.new_run({
        "batch": batch, "result": final, "summary": summary,
        "n_orders": req.n_orders,
    })
    return {"run_id": run_id, "summary": summary}


@app.get("/api/runs")
def list_runs():
    return [{"run_id": r["run_id"], "created_at": r["created_at"], "summary": r["summary"]}
            for r in store.list_runs()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return {
        "run_id": run_id, "summary": run["summary"],
        "matches": run["result"]["matches"], "exceptions": run["result"]["exceptions"],
    }


@app.get("/api/runs/{run_id}/audit")
def get_audit(run_id: str):
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return {"run_id": run_id, "audit_trail": run["result"]["audit_trail"]}


@app.post("/api/runs/{run_id}/ask")
def ask(run_id: str, req: AskRequest):
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    answer = llm_resolver.answer_question(req.question, run["summary"], run["result"]["exceptions"])
    return {"answer": answer}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

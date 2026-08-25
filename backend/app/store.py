"""In-memory run store, keyed by run_id.

A hackathon demo doesn't need Postgres durability across restarts; the
same shape here (dict keyed by id, JSON-serializable values) drops into
a `runs` table with zero interface change if this becomes a real service.
"""
from __future__ import annotations

import time
import uuid

_RUNS: dict[str, dict] = {}


def new_run(payload: dict) -> str:
    run_id = uuid.uuid4().hex[:12]
    payload["run_id"] = run_id
    payload["created_at"] = time.time()
    _RUNS[run_id] = payload
    return run_id


def get_run(run_id: str) -> dict | None:
    return _RUNS.get(run_id)


def list_runs() -> list[dict]:
    return sorted(_RUNS.values(), key=lambda r: r["created_at"], reverse=True)

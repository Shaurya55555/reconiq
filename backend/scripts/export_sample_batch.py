"""Dumps one reproducible synthetic batch (seed=5, 100 orders) to CSV files
under data/sample/, plus the reconciliation result, so a judge can inspect
real input/output data without running the app at all.

Run from backend/: python scripts/export_sample_batch.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data_gen, llm_resolver, matcher, scoring  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({k for row in rows for k in row if not k.startswith("_")})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batch = data_gen.generate_batch(n_orders=100, seed=5)
    rule_result = matcher.reconcile(batch["orders"], batch["settlements"], batch["bank_lines"])
    final = matcher.apply_llm_resolutions(rule_result, llm_resolver.resolve)
    summary = matcher.summarize(batch["orders"], final)
    ground_truth = scoring.score_against_ground_truth(batch["orders"], final)

    write_csv(OUT_DIR / "orders.csv", batch["orders"])
    write_csv(OUT_DIR / "settlements.csv", batch["settlements"])
    write_csv(OUT_DIR / "bank_lines.csv", batch["bank_lines"])
    write_csv(OUT_DIR / "matches.csv", final["matches"])
    write_csv(OUT_DIR / "exceptions.csv", final["exceptions"])
    write_csv(OUT_DIR / "audit_trail.csv", final["audit_trail"])

    (OUT_DIR / "summary.json").write_text(json.dumps({**summary, **{
        "ground_truth_accuracy": ground_truth["ground_truth_accuracy"],
    }}, indent=2))

    print(f"Wrote sample batch + reconciliation output to {OUT_DIR}")
    print(json.dumps(summary, indent=2))
    print("ground_truth_accuracy:", ground_truth["ground_truth_accuracy"])


if __name__ == "__main__":
    main()

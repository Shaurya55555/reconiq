"""Offline Razorpay Settlement API adapter -- fetch, normalize, write CSV.

This script is deliberately isolated from the reconciliation engine: it
contains no matching logic and never imports matcher.py. Its only job is

    authenticate -> fetch -> paginate -> normalize -> write CSV

producing an orders/settlements/bank_lines-shaped CSV that can be fed
into ReconIQ's existing bring-your-own-data path (POST /api/run-upload,
or the dashboard's "Bring your own data" panel) exactly like any other
uploaded file. Run this once, offline, before a demo -- there is no live
"Connect Razorpay" button in the deployed app, and this script is never
imported by the FastAPI app.

Requires RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET in the environment (Test
Mode keys recommended -- see https://razorpay.com/docs/payments/quickstart/).
Never commit these, never print the secret, never run this against Live
Mode keys for a demo.

Endpoint used: GET /v1/settlements/recon/combined?year=YYYY&month=MM
(the "Fetch Settlement Recon Details" API), not the plain "Fetch All
Settlements" endpoint -- the plain endpoint returns settlement-level
aggregates with no payment_id at all, which can't be matched against an
order ledger. The recon-combined endpoint breaks each settlement down
into its component transactions (type: "payment" | "refund" | "transfer"
| "adjustment"), which is what a per-order reconciliation actually needs.

Known limitation, stated up front rather than hidden: a single Razorpay
settlement can legitimately cover more than one payment (a genuine batch
settlement -- the same real-world case matcher.py's `batch_settlement`
pass already models for synthetic data). The CSV upload path
(main.py::UploadRunRequest) currently accepts one payment_id per
settlement row, not a payment_ids list, so a multi-payment settlement
can't be losslessly represented as a single CSV row today. This script
does not silently drop that money: it writes one row per payment within
a multi-payment settlement (sharing the same settlement_id and UTR) and
prints an explicit warning listing which settlement_ids were split this
way, so nothing is hidden -- extending the upload schema to accept a
payment_ids column is a natural next step, not attempted here to keep
this script's blast radius to "fetch and normalize" only.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

RECON_URL = "https://api.razorpay.com/v1/settlements/recon/combined"
PAGE_SIZE = 100


def _auth() -> tuple[str, str]:
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        print("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set in the "
              "environment (use Test Mode keys). Never pass these as CLI "
              "args -- they'd end up in shell history.", file=sys.stderr)
        sys.exit(1)
    return key_id, key_secret


def fetch_recon_items(year: int, month: int, day: int | None = None) -> list[dict]:
    """Paginate through the recon-combined endpoint for one month (or one
    day, if given) and return every raw item. No filtering, no
    normalization here -- that happens in normalize_settlements below, so
    this function's only job is "get everything the API has."
    """
    key_id, key_secret = _auth()
    params = {"year": year, "month": month, "count": PAGE_SIZE, "skip": 0}
    if day is not None:
        params["day"] = day

    items: list[dict] = []
    with httpx.Client(auth=(key_id, key_secret), timeout=30.0) as client:
        while True:
            resp = client.get(RECON_URL, params=params)
            resp.raise_for_status()
            body = resp.json()
            page = body.get("items", [])
            items.extend(page)
            if len(page) < PAGE_SIZE:
                break
            params["skip"] += PAGE_SIZE
    return items


def _iso_date(unix_ts: int | None) -> str:
    if not unix_ts:
        return ""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).date().isoformat()


def normalize_settlements(items: list[dict]) -> tuple[list[dict], list[str]]:
    """Turn raw recon items into ReconIQ settlement rows.

    Only type == "payment" items carry a payment reference (via
    `entity_id`, per Razorpay's recon-combined response shape -- the
    `payment_id` field on these rows is used for refund/transfer linkage,
    not populated on payment rows themselves). Refunds, transfers, and
    fee/tax adjustment rows are skipped here -- they're a different
    reconciliation question (see the README's refund-aware matching
    section for how ReconIQ already treats refunds, once fed in via its
    own refunds.csv schema, separately from this adapter).

    Returns (rows, warnings) -- rows ready to write to settlements.csv,
    and human-readable warnings about anything this script couldn't
    losslessly represent (see module docstring).
    """
    payment_items = [i for i in items if i.get("type") == "payment" and i.get("entity_id")]

    by_settlement: dict[str, list[dict]] = {}
    for item in payment_items:
        by_settlement.setdefault(item["settlement_id"], []).append(item)

    rows: list[dict] = []
    warnings: list[str] = []
    for settlement_id, group in by_settlement.items():
        if len(group) > 1:
            warnings.append(
                f"settlement_id={settlement_id} covers {len(group)} payments "
                f"({', '.join(g['entity_id'] for g in group)}) -- written as "
                f"{len(group)} separate rows sharing one settlement_id/UTR; "
                "the current CSV upload schema has no payment_ids column, so "
                "this can't be represented as a single N:1 batch settlement "
                "row the way the synthetic generator's batch_settlement mode "
                "does. See this script's module docstring."
            )
        for item in group:
            amount = item.get("credit") or item.get("amount") or 0
            rows.append({
                "settlement_id": settlement_id,
                "payment_id": item["entity_id"],
                "utr": item.get("settlement_utr", ""),
                "amount": round(amount / 100, 2),  # paise -> rupees
                "settled_at": _iso_date(item.get("settled_at") or item.get("created_at")),
            })
    return rows, warnings


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print(f"No settlement rows to write -- {out_path} not created.", file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["settlement_id", "payment_id", "utr", "amount", "settled_at"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} settlement rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Razorpay settlements for one month and normalize "
                     "them into ReconIQ's settlements.csv schema.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("razorpay_export/settlements.csv"))
    args = parser.parse_args()

    print(f"Fetching settlement recon details for {args.year}-{args.month:02d}"
          f"{f'-{args.day:02d}' if args.day else ''} ...")
    items = fetch_recon_items(args.year, args.month, args.day)
    print(f"Fetched {len(items)} raw recon items.")

    rows, warnings = normalize_settlements(items)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    write_csv(rows, args.out)
    print("\nNext step: upload this file (alongside your own orders.csv and "
          "bank_lines.csv) via the dashboard's 'Bring your own data' panel, "
          "or POST it to /api/run-upload directly.")


if __name__ == "__main__":
    main()

"""Synthetic multi-source reconciliation data.

Three sources, mirroring a real Razorpay merchant reconciliation:
  - orders:      internal order ledger (source of truth for what was sold)
  - settlements: Razorpay settlement export (payment_id -> utr, net amount)
  - bank_lines:  raw bank statement rows (free-text narration, the ground
                 truth for money that actually moved)

A batch is seeded with a known mix of clean and broken records so the
match rate reported by the matcher can be checked against the truth
recorded in each record's `_truth` field (used only by tests/scoring,
never by the matcher itself).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Literal

CUSTOMERS = [
    "Ananya Rao", "Vikram Shah", "Priya Menon", "Rohit Verma", "Sneha Iyer",
    "Karan Malhotra", "Divya Nair", "Arjun Reddy", "Neha Kapoor", "Sameer Joshi",
    "Ishaan Gupta", "Meera Pillai", "Tanvi Desai", "Aditya Kulkarni", "Ritu Chawla",
]

Anomaly = Literal[
    "clean", "fee_adjusted", "date_shifted", "missing_settlement",
    "duplicate_settlement", "garbled_narration", "amount_mismatch",
]

ANOMALY_WEIGHTS = {
    "clean": 0.55,
    "fee_adjusted": 0.12,
    "date_shifted": 0.08,
    "missing_settlement": 0.08,
    "duplicate_settlement": 0.05,
    "garbled_narration": 0.08,
    "amount_mismatch": 0.04,
}


def _weights_for_corruption_rate(corruption_rate: float) -> dict[str, float]:
    """Redistribute ANOMALY_WEIGHTS so `corruption_rate` is the total
    probability of a non-clean order, split across the anomaly types in
    their existing relative proportions. Used by the benchmark mode to
    prove the reported accuracy holds up as data gets messier, not just
    at whatever corruption level the default weights happen to produce.
    """
    other_total = sum(v for k, v in ANOMALY_WEIGHTS.items() if k != "clean")
    weights = {"clean": 1.0 - corruption_rate}
    for k, v in ANOMALY_WEIGHTS.items():
        if k != "clean":
            weights[k] = corruption_rate * (v / other_total)
    return weights


def _weighted_anomaly(rng: random.Random, weights: dict[str, float]) -> Anomaly:
    kinds = list(weights.keys())
    vals = list(weights.values())
    return rng.choices(kinds, weights=vals, k=1)[0]


def _utr(rng: random.Random) -> str:
    return f"UTR{rng.randint(10**8, 10**9 - 1)}"


def _garble(utr: str, rng: random.Random) -> str:
    # simulate a bank narration mangling the UTR: truncation, prefix noise,
    # or swapped digits, forcing free-text reasoning to resolve it.
    body = utr.replace("UTR", "")
    choice = rng.choice(["truncate", "prefix_noise", "swap"])
    if choice == "truncate":
        return f"NEFT/{body[:5]}.../CUST TXN"
    if choice == "prefix_noise":
        return f"IMPS-{body}-SETTLEMENT RZP"
    digits = list(body)
    i = rng.randrange(len(digits) - 1)
    digits[i], digits[i + 1] = digits[i + 1], digits[i]
    return f"NEFT-{''.join(digits)}-RAZORPAY"


def generate_batch(n_orders: int = 70, seed: int | None = None,
                    corruption_rate: float | None = None) -> dict:
    """corruption_rate, if given, overrides ANOMALY_WEIGHTS' default ~45%
    non-clean mix with an explicit total anomaly probability (0..1), used
    by the benchmark mode to generate progressively messier batches.
    """
    rng = random.Random(seed)
    weights = _weights_for_corruption_rate(corruption_rate) if corruption_rate is not None else ANOMALY_WEIGHTS
    base_date = datetime(2026, 8, 1)
    orders, settlements, bank_lines = [], [], []

    for i in range(1, n_orders + 1):
        order_id = f"ORD{1000 + i}"
        customer = rng.choice(CUSTOMERS)
        amount = round(rng.uniform(299, 24999), 2)
        created_at = base_date + timedelta(days=rng.randint(0, 20), hours=rng.randint(0, 23))
        payment_id = f"pay_{rng.randint(10**10, 10**11 - 1)}"
        anomaly = _weighted_anomaly(rng, weights)

        orders.append({
            "order_id": order_id,
            "customer": customer,
            "amount": amount,
            "created_at": created_at.isoformat(),
            "razorpay_payment_id": payment_id,
            "_truth": anomaly,
        })

        if anomaly == "missing_settlement":
            continue  # no settlement row at all -> should surface as exception

        utr = _utr(rng)
        settled_at = created_at + timedelta(days=rng.randint(1, 2))
        net_amount = amount

        if anomaly == "fee_adjusted":
            fee_pct = rng.uniform(0.015, 0.028)
            net_amount = round(amount * (1 - fee_pct), 2)

        if anomaly == "amount_mismatch":
            # Deliberately outside any plausible fee range (>3%), regardless
            # of order size, so this never gets swallowed by fuzzy matching.
            net_amount = round(amount * (1 + rng.choice([-1, 1]) * rng.uniform(0.06, 0.20)), 2)

        if anomaly == "date_shifted":
            settled_at = settled_at + timedelta(days=rng.randint(6, 10))

        settlements.append({
            "settlement_id": f"stl_{rng.randint(10**9, 10**10 - 1)}",
            "payment_id": payment_id,
            "utr": utr,
            "amount": net_amount,
            "settled_at": settled_at.isoformat(),
        })

        if anomaly == "duplicate_settlement":
            dup_amount = round(net_amount * rng.uniform(0.98, 1.0), 2)
            settlements.append({
                "settlement_id": f"stl_{rng.randint(10**9, 10**10 - 1)}",
                "payment_id": payment_id,
                "utr": _utr(rng),
                "amount": dup_amount,
                "settled_at": (settled_at + timedelta(days=1)).isoformat(),
            })

        if anomaly == "garbled_narration":
            narration = _garble(utr, rng)
        else:
            narration = f"NEFT-{utr}-RAZORPAY SETTLEMENT {customer.split()[0].upper()}"

        bank_lines.append({
            "line_id": f"bl_{rng.randint(10**9, 10**10 - 1)}",
            "narration": narration,
            "amount": net_amount,
            "value_date": settled_at.date().isoformat(),
            "_utr_hint": utr,
        })

    rng.shuffle(bank_lines)
    return {"orders": orders, "settlements": settlements, "bank_lines": bank_lines}

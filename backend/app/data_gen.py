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
    "split_settlement", "batch_settlement",
]

ANOMALY_WEIGHTS = {
    "clean": 0.44,
    "fee_adjusted": 0.10,
    "date_shifted": 0.06,
    "missing_settlement": 0.06,
    "duplicate_settlement": 0.05,
    "garbled_narration": 0.06,
    "amount_mismatch": 0.04,
    "split_settlement": 0.09,
    "batch_settlement": 0.10,
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

        if anomaly == "batch_settlement":
            continue  # handled after the loop, grouped with other batch_settlement orders

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

        if anomaly == "split_settlement":
            # One settlement, real money, but the bank shows it arriving as
            # two separate credits (a partial early payout followed by the
            # balance a day or two later, or a fee adjustment booked as its
            # own line) -- both legs reference the same UTR, because it's
            # genuinely the same underlying settlement. A single 1:1 pass
            # can never resolve this; it needs a many-to-one match.
            split_pct = rng.uniform(0.35, 0.65)
            leg_1 = round(net_amount * split_pct, 2)
            leg_2 = round(net_amount - leg_1, 2)
            for leg_amount, day_offset in ((leg_1, 0), (leg_2, rng.randint(1, 2))):
                bank_lines.append({
                    "line_id": f"bl_{rng.randint(10**9, 10**10 - 1)}",
                    "narration": f"NEFT-{utr}-RAZORPAY SETTLEMENT {customer.split()[0].upper()} PART",
                    "amount": leg_amount,
                    "value_date": (settled_at + timedelta(days=day_offset)).date().isoformat(),
                    "_utr_hint": utr,
                })
        else:
            bank_lines.append({
                "line_id": f"bl_{rng.randint(10**9, 10**10 - 1)}",
                "narration": narration,
                "amount": net_amount,
                "value_date": settled_at.date().isoformat(),
                "_utr_hint": utr,
            })

    _generate_batch_settlements(orders, settlements, bank_lines, rng)

    rng.shuffle(bank_lines)
    return {"orders": orders, "settlements": settlements, "bank_lines": bank_lines}


def _generate_batch_settlements(orders: list[dict], settlements: list[dict],
                                 bank_lines: list[dict], rng: random.Random) -> None:
    """Real Razorpay behaviour, not an invented mechanism: a settlement
    commonly covers *multiple* payments batched together, settled as one
    bank credit under one UTR -- this is the mirror of split_settlement
    (there, one settlement arrives as many bank lines; here, many orders'
    payments arrive as one settlement). Modelling it this way (one real
    settlement row with a payment_ids list) is simpler and more honest
    than contriving a bank narration that somehow embeds several UTRs,
    which isn't how bank statements actually look.
    """
    pending = [o for o in orders if o["_truth"] == "batch_settlement"]
    i = 0
    while i < len(pending):
        remaining = len(pending) - i
        group_size = 1 if remaining == 1 else rng.choice([2, 3]) if remaining >= 3 else 2
        group = pending[i:i + group_size]
        i += group_size

        total_amount = round(sum(o["amount"] for o in group), 2)
        utr = _utr(rng)
        earliest_created = min(datetime.fromisoformat(o["created_at"]) for o in group)
        settled_at = earliest_created + timedelta(days=rng.randint(1, 2))

        settlements.append({
            "settlement_id": f"stl_{rng.randint(10**9, 10**10 - 1)}",
            "payment_id": group[0]["razorpay_payment_id"],  # backward-compatible single reference
            "payment_ids": [o["razorpay_payment_id"] for o in group],
            "utr": utr,
            "amount": total_amount,
            "settled_at": settled_at.isoformat(),
        })
        bank_lines.append({
            "line_id": f"bl_{rng.randint(10**9, 10**10 - 1)}",
            "narration": f"NEFT-{utr}-RAZORPAY SETTLEMENT BATCH {len(group)} ORDERS",
            "amount": total_amount,
            "value_date": settled_at.date().isoformat(),
            "_utr_hint": utr,
        })

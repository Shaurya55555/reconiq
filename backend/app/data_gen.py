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
import re
import string
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
    "split_settlement", "batch_settlement", "refunded", "partial_refund",
    "refunded_after_settlement",
]

INSTANT_SETTLEMENT_RATE = 0.15  # fraction of otherwise-clean settlements a merchant
                                 # took as an on-demand Instant Settlement payout instead
                                 # of the standard cycle
INSTANT_SETTLEMENT_FEE_RANGE = (0.032, 0.045)  # standard fee (~1.5-2.8%, see
                                                # fee_adjusted) plus Razorpay's additional
                                                # on-demand payout fee -- stays comfortably
                                                # under matcher.INSTANT_SETTLEMENT_FEE_TOLERANCE_PCT

ANOMALY_WEIGHTS = {
    "clean": 0.30,
    "fee_adjusted": 0.10,
    "date_shifted": 0.06,
    "missing_settlement": 0.06,
    "duplicate_settlement": 0.05,
    "garbled_narration": 0.06,
    "amount_mismatch": 0.04,
    "split_settlement": 0.09,
    "batch_settlement": 0.10,
    "refunded": 0.05,
    "partial_refund": 0.05,
    "refunded_after_settlement": 0.04,
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


# Real Razorpay settlement UTRs/reference numbers are prefixed by the
# receiving bank's own code (e.g. "AXISCN0841380906", "CB0042644252"),
# not a literal "UTR" string -- matching a real settlement export's look,
# not just its substring-matching *behavior* (which never cared about the
# specific prefix text either way).
BANK_UTR_PREFIXES = ["AXISCN", "HDFCR", "ICICN", "SBIN", "KKBK", "CB", "PUNB", "YESB"]


def _utr(rng: random.Random) -> str:
    prefix = rng.choice(BANK_UTR_PREFIXES)
    return f"{prefix}{rng.randint(0, 10**10 - 1):010d}"


def _random_id_suffix(rng: random.Random, length: int = 14) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def _garble(utr: str, rng: random.Random) -> str:
    # simulate a bank narration mangling the UTR: truncation, prefix noise,
    # or swapped digits, forcing free-text reasoning to resolve it. Strips
    # the leading bank-code letters (not a literal "UTR"), so this stays
    # correct regardless of which bank prefix _utr() picked.
    body = re.sub(r"^[A-Za-z]+", "", utr)
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
    orders, settlements, bank_lines, refunds = [], [], [], []

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

        if anomaly == "refunded":
            # Fully refunded before settlement -- genuinely how Razorpay
            # behaves (a full refund nets the payout to zero, so no
            # settlement is ever generated for it). No settlement, no bank
            # line; the refund record itself is what explains the absence,
            # so the matcher can tell this apart from a genuinely missing
            # settlement rather than flagging it as an exception.
            refunds.append({
                "refund_id": f"rfnd_{rng.randint(10**9, 10**10 - 1)}",
                "payment_id": payment_id,
                "amount": amount,
                "refunded_at": (created_at + timedelta(days=rng.randint(0, 3))).isoformat(),
                "type": "full",
            })
            continue

        utr = _utr(rng)
        settled_at = created_at + timedelta(days=rng.randint(1, 2))
        net_amount = amount

        if anomaly == "partial_refund":
            # Partially refunded *before* settlement -- Razorpay nets the
            # refund out of the payout, so the settlement/bank amount below
            # is genuinely order.amount - refund.amount, not a fee-sized
            # discrepancy. Real money, real settlement, just smaller than
            # the order. The refund must be dated at or before settled_at
            # (bounded to the [0, settled_at - created_at] window) -- this
            # anomaly specifically models the pre-settlement netting case;
            # see refunded_after_settlement below for the other case.
            refund_amount = round(amount * rng.uniform(0.1, 0.6), 2)
            max_offset_days = (settled_at - created_at).days
            refunds.append({
                "refund_id": f"rfnd_{rng.randint(10**9, 10**10 - 1)}",
                "payment_id": payment_id,
                "amount": refund_amount,
                "refunded_at": (created_at + timedelta(days=rng.randint(0, max_offset_days))).isoformat(),
                "type": "partial",
            })
            net_amount = round(amount - refund_amount, 2)

        if anomaly == "refunded_after_settlement":
            # The settlement already happened and was genuinely legitimate
            # at the *full* order amount -- the refund is a separate, later
            # cash event (customer asks for a refund days after the money
            # already settled) and must never be netted against this
            # settlement. net_amount stays at the full `amount`; only the
            # refund record itself is dated after settled_at.
            refunds.append({
                "refund_id": f"rfnd_{rng.randint(10**9, 10**10 - 1)}",
                "payment_id": payment_id,
                "amount": amount,
                "refunded_at": (settled_at + timedelta(days=rng.randint(1, 5))).isoformat(),
                "type": "full",
            })

        if anomaly == "fee_adjusted":
            fee_pct = rng.uniform(0.015, 0.028)
            net_amount = round(amount * (1 - fee_pct), 2)

        if anomaly == "amount_mismatch":
            # Deliberately outside any plausible fee range (>3%), regardless
            # of order size, so this never gets swallowed by fuzzy matching.
            net_amount = round(amount * (1 + rng.choice([-1, 1]) * rng.uniform(0.06, 0.20)), 2)

        if anomaly == "date_shifted":
            settled_at = settled_at + timedelta(days=rng.randint(6, 10))

        # Instant Settlement is an independent, orthogonal modifier (real
        # merchants mix instant and standard payouts), not its own anomaly
        # type -- only layered onto otherwise-legitimate settlements
        # (clean/fee_adjusted), never onto a deliberately broken one, so it
        # never masks or gets confused with a genuine anomaly under test.
        # Its higher fee (matcher.INSTANT_SETTLEMENT_FEE_TOLERANCE_PCT
        # covers it) replaces whatever standard fee was set above.
        is_instant = anomaly in ("clean", "fee_adjusted") and rng.random() < INSTANT_SETTLEMENT_RATE
        if is_instant:
            fee_pct = rng.uniform(*INSTANT_SETTLEMENT_FEE_RANGE)
            net_amount = round(amount * (1 - fee_pct), 2)

        settlement_row = {
            "settlement_id": f"setl_{_random_id_suffix(rng)}",
            "payment_id": payment_id,
            "utr": utr,
            "amount": net_amount,
            "settled_at": settled_at.isoformat(),
        }
        if is_instant:
            settlement_row["is_instant"] = True
        settlements.append(settlement_row)

        if anomaly == "duplicate_settlement":
            dup_amount = round(net_amount * rng.uniform(0.98, 1.0), 2)
            settlements.append({
                "settlement_id": f"setl_{_random_id_suffix(rng)}",
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
    _generate_refund_debits(refunds, bank_lines, rng)

    rng.shuffle(bank_lines)
    return {"orders": orders, "settlements": settlements, "bank_lines": bank_lines, "refunds": refunds}


REFUND_DEBIT_RATE = 0.82  # not every refund clears the bank promptly -- some are still
                          # pending or genuinely stuck, which is exactly the risk
                          # cash-position matching (matcher._match_refund_debits) exists
                          # to catch instead of assuming a refund record means money moved


def _generate_refund_debits(refunds: list[dict], bank_lines: list[dict], rng: random.Random) -> None:
    """A refund record is a promise, not proof the money left the account.
    Model that gap directly: most refunds get a matching outbound
    (negative-amount) bank line a day or two later, referencing the
    refund_id the same way a settlement's bank line references its UTR --
    but some fraction deliberately don't, so a real batch always has at
    least one genuinely undebited refund for the cash-position pass to
    surface, not just a hypothetical case."""
    for r in refunds:
        if rng.random() >= REFUND_DEBIT_RATE:
            continue
        refunded_at = datetime.fromisoformat(r["refunded_at"])
        debit_date = refunded_at + timedelta(days=rng.randint(0, 2))
        bank_lines.append({
            "line_id": f"bl_{rng.randint(10**9, 10**10 - 1)}",
            "narration": f"REFUND-{r['refund_id']}-RAZORPAY PAYOUT",
            "amount": -r["amount"],
            "value_date": debit_date.date().isoformat(),
        })


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
            "settlement_id": f"setl_{_random_id_suffix(rng)}",
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

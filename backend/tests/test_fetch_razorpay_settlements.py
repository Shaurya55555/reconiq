import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_razorpay_settlements import normalize_refunds, normalize_settlements  # noqa: E402


def test_single_payment_settlement_normalizes_to_one_clean_row():
    items = [
        {"type": "payment", "entity_id": "pay_ABC123", "settlement_id": "stl_1",
         "settlement_utr": "UTR111", "credit": 98500, "settled_at": 1735689600},
    ]
    rows, warnings = normalize_settlements(items)
    assert warnings == []
    assert rows == [{
        "settlement_id": "stl_1", "payment_id": "pay_ABC123", "utr": "UTR111",
        "amount": 985.0, "settled_at": "2025-01-01",
    }]


def test_multi_payment_settlement_writes_one_row_per_payment_and_warns():
    """A settlement covering >1 payment can't be losslessly represented as
    a single CSV row (no payment_ids column in the upload schema) -- this
    locks in that it's still fully written out (never silently dropped),
    just as separate rows sharing the settlement_id/UTR, with an explicit
    warning naming exactly which settlement was split this way."""
    items = [
        {"type": "payment", "entity_id": "pay_A", "settlement_id": "stl_batch",
         "settlement_utr": "UTR222", "credit": 40000, "settled_at": 1735689600},
        {"type": "payment", "entity_id": "pay_B", "settlement_id": "stl_batch",
         "settlement_utr": "UTR222", "credit": 60000, "settled_at": 1735689600},
    ]
    rows, warnings = normalize_settlements(items)
    assert len(rows) == 2
    assert {r["payment_id"] for r in rows} == {"pay_A", "pay_B"}
    assert all(r["settlement_id"] == "stl_batch" and r["utr"] == "UTR222" for r in rows)
    assert len(warnings) == 1
    assert "stl_batch" in warnings[0] and "2 payments" in warnings[0]


def test_refund_and_transfer_rows_are_excluded_not_miscounted_as_payments():
    """Only type == "payment" rows carry a usable payment reference; refund
    and transfer rows must never leak into the settlements CSV as if they
    were ordinary payments."""
    items = [
        {"type": "payment", "entity_id": "pay_X", "settlement_id": "stl_2",
         "settlement_utr": "UTR333", "credit": 10000, "settled_at": 1735689600},
        {"type": "refund", "entity_id": "rfnd_1", "payment_id": "pay_X",
         "settlement_id": "stl_2", "settlement_utr": "UTR333", "debit": 2000},
        {"type": "transfer", "entity_id": "trf_1", "settlement_id": "stl_2",
         "settlement_utr": "UTR333", "debit": 500},
    ]
    rows, warnings = normalize_settlements(items)
    assert len(rows) == 1
    assert rows[0]["payment_id"] == "pay_X"


def test_amount_converts_paise_to_rupees():
    items = [{"type": "payment", "entity_id": "pay_1", "settlement_id": "stl_3",
              "settlement_utr": "UTR444", "credit": 123456, "settled_at": 1735689600}]
    rows, _ = normalize_settlements(items)
    assert rows[0]["amount"] == 1234.56


def test_normalize_refunds_extracts_refund_rows_from_the_same_recon_feed():
    """The recon-combined response already carries refund line items
    alongside the payment rows this script was already fetching for
    settlements.csv -- this is what actually closes the gap where refund
    data used to require a separately hand-supplied file."""
    items = [
        {"type": "payment", "entity_id": "pay_X", "settlement_id": "stl_2",
         "settlement_utr": "UTR333", "credit": 10000, "settled_at": 1735689600},
        {"type": "refund", "entity_id": "rfnd_1", "payment_id": "pay_X",
         "settlement_id": "stl_2", "settlement_utr": "UTR333", "debit": 2000,
         "created_at": 1735776000},
        {"type": "transfer", "entity_id": "trf_1", "settlement_id": "stl_2",
         "settlement_utr": "UTR333", "debit": 500},
    ]
    refund_rows = normalize_refunds(items)
    assert refund_rows == [{
        "payment_id": "pay_X", "amount": 20.0, "refund_id": "rfnd_1",
        "refunded_at": "2025-01-02",
    }]


def test_normalize_refunds_ignores_payment_and_transfer_rows():
    items = [
        {"type": "payment", "entity_id": "pay_X", "settlement_id": "stl_2",
         "credit": 10000, "settled_at": 1735689600},
        {"type": "transfer", "entity_id": "trf_1", "settlement_id": "stl_2", "debit": 500},
    ]
    assert normalize_refunds(items) == []


def test_normalize_refunds_converts_paise_to_rupees():
    items = [{"type": "refund", "entity_id": "rfnd_9", "payment_id": "pay_9",
              "debit": 123456, "created_at": 1735689600}]
    rows = normalize_refunds(items)
    assert rows[0]["amount"] == 1234.56

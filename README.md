# ReconIQ

A multi-source reconciliation agent for the Razorpay AI Buildathon —
**Track 04: AI Finance Controller**.

## What it solves

Merchants reconcile money across three places that never agree perfectly:
their own order ledger, the payment gateway's settlement export, and the
bank statement the money actually lands in. Fees shift amounts, dates
drift, UTR references get garbled by bank narration formats, and some
settlements just never show up. Today someone eyeballs a spreadsheet.

ReconIQ closes that loop automatically on a batch of orders:

1. **Exact match** — settlement UTR found verbatim in a bank line, amount
   matches to the paisa.
2. **Fuzzy match** — UTR found, but amount differs by a plausible Razorpay
   fee (≤3%) or the settlement date drifted.
3. **LLM-resolved match** — the UTR isn't found in *any* bank line because
   the narration was truncated, prefixed with bank-specific noise, or had
   digits transposed. This is handed to an LLM (or an offline heuristic
   with no key configured) to reason over the free-text narration, amount,
   and date and propose a match — the one step a plain rules engine
   genuinely cannot do.
4. **Honest exceptions** — anything still unresolved is surfaced with a
   reason code (`no_settlement_found`, `amount_mismatch`,
   `duplicate_candidate`, `unrecognized_narration`,
   `unrecognized_bank_line`), never silently dropped.

Every decision, at every stage, is written to an audit trail: which
record, which method decided it, at what confidence, and why.

## Why this design (not just a rules engine with a chatbot bolted on)

The deterministic passes exist because a match rate you can't defend
under questioning is worthless — "honest metrics" was explicit in the
brief. The LLM only runs on the residual the rules provably cannot solve
(garbled narration text), so it's doing real reasoning work, not
narrating what a SQL join already found. See `backend/app/matcher.py`
for the pass order and `backend/app/llm_resolver.py` for the resolver
and its offline fallback.

## Architecture

```
data_gen.py  --generates-->  orders.csv, settlements.csv, bank_lines.csv
                                        |
                                        v
matcher.py   --pass 1/2 (rules)-->  matches, needs_llm, exceptions
                                        |
                                        v
llm_resolver.py --pass 3 (LLM/heuristic)--> resolved matches or final exceptions
                                        |
                                        v
FastAPI (main.py) --serves--> summary, exceptions, audit trail, chat Q&A
                                        |
                                        v
frontend/index.html (static dashboard, no build step)
```

`llm_resolver.py` is provider-pluggable via `LLM_PROVIDER`
(`openai` / `anthropic` / `gemini` / `ollama`); with none configured it
runs a deterministic offline heuristic (amount match + digit-sequence
similarity) so the whole app works with zero API keys.

## Running it

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/`, click **Run reconciliation**. Optionally
copy `backend/.env.example` to `backend/.env` and set `LLM_PROVIDER` +
an API key to use a real model for the exception-resolution and Q&A
layers instead of the offline heuristic.

Tests:

```bash
cd backend
pytest tests -q
```

## What broke, and how I got out of it

The synthetic-data generator originally created `amount_mismatch` records
with a fixed absolute drift (₹50–400). For small orders that's clearly
outside any fee range, but for a large order (say ₹20,000) a ₹400 drift
is under the 3% fee-tolerance band the matcher uses — so the matcher
correctly classified those as *fuzzy fee-adjusted matches* instead of
*exceptions*, and a test written to check "amount mismatches must never
silently match" failed. The bug was in the data generator conflating an
absolute rupee drift with a percentage-of-amount drift; fixed by
generating the mismatch as a percentage (6–20%) of the order amount,
which is guaranteed to exceed the fee-tolerance band regardless of order
size. Left the failing test in `backend/tests/test_matcher.py` as
`test_amount_mismatch_is_flagged_not_silently_matched` — it's the one
that would have caught this in a real batch before a human noticed a
mismatched exception silently marked "matched."

## Stack

Python / FastAPI backend, no database (in-memory run store — swaps for
Postgres with no interface change), static HTML/JS dashboard (no build
step, so nothing to break on someone else's machine during judging).

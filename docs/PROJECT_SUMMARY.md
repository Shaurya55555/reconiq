# ReconIQ — Project Summary

**Repo:** https://github.com/Shaurya55555/reconiq
**Live demo:** https://reconiq-mocha.vercel.app

Built for the Razorpay AI Buildathon — **Track 04: AI Finance Controller**.

---

## What Track 4 asked for

> "Build an agent that closes one finance-ops loop across a 50+ record
> batch of synthetic data, reporting its match rate and the exceptions
> it could not resolve."

**The bar (their exact words):**

> "Throughput plus measured accuracy plus an honest exception list.
> One cherry-picked match proves nothing."

---

## What ReconIQ is

A multi-source reconciliation agent: it matches an internal order ledger
against a Razorpay settlement export and a raw bank statement — the way
a finance-ops person would by hand, except verified, auditable, and able
to handle the messy cases most reconciliation tools give up on.

---

## The matching engine — four passes, cheapest and most certain first

1. **Exact match** — settlement UTR found verbatim in a bank line,
   amount matches to the paisa.
2. **Fuzzy match** — UTR found, amount differs by a plausible Razorpay
   fee (≤3%, configurable) or the settlement date drifted.
3. **Many-to-one (group split) match** — the deepening upgrade. When one
   settlement's money arrives as 2–3 separate bank credits instead of
   one (a partial early payout + balance, or a fee leg booked
   separately), the system finds the combination of bank lines sharing
   that settlement's UTR which sums exactly to it. Fully
   deterministic — a bounded combinatorial search, not an LLM guess —
   because the shared UTR proves the group belongs together and exact
   arithmetic doesn't need AI. This is the single most common reason
   real reconciliation tools fail past 1:1 matching and get pushed back
   to spreadsheets.
4. **LLM-resolved match** — only when the UTR isn't found in *any* bank
   line at all (garbled/truncated narration). A real Gemini call
   reasons over the free text; an offline heuristic fallback covers the
   no-API-key case.
5. **Honest exceptions** — anything still unresolved gets a specific
   reason code, never dropped silently.

---

## Everything built on top of the core loop

- **Ground-truth accuracy** — scored against a hidden truth label the
  matcher never sees, distinct from raw match rate.
- **Value-weighted accuracy** — order-level correctness weighted by
  rupee amount, not just transaction count.
- **False-clear vs. safe-miss split** — every wrong answer is
  categorized as dangerous (confidently wrong) or conservative
  (flagged for review), with a measured rupee amount for each.
- **Rules-only vs. Rules+AI comparison** — computed on the *same batch*
  every run, proving what the AI layer actually contributes instead of
  asserting it.
- **Corruption-rate benchmark** — runs the real pipeline across
  multiple corruption levels on fresh batches, charted, so the accuracy
  claim isn't just true for one lucky run.
- **Human-in-the-loop override** — accept / reject / manually-match any
  decision, logged distinctly as `human_override` in the audit trail so
  it's never confused with an automated call.
- **Per-decision evidence drawer** — click "Why?" on any match or
  exception to see the full lineage (order/settlement/bank amounts, UTR
  vs. narration, date drift, AI involvement, actual reasoning) —
  including a combined-amount breakdown for group matches (e.g.
  "₹11,240 = ₹5,985 + ₹5,256").
- **Bring-your-own-data** — upload real CSVs in the same schema and run
  the identical pipeline against them, not just synthetic data.
- **Full audit trail**, exportable as CSV, for every decision at every
  stage.
- **Hero-first dashboard** — leads with financial position (amount
  processed / reconciled / at risk), with policy config tucked into an
  "Advanced settings" panel instead of sitting above the fold.

---

## Benchmark — run live against production

Fresh batches at each corruption level, real Gemini, run directly
against the deployed site (not a canned example):

| Corruption | Match Rate | Ground-truth Acc | Value-weighted Acc | False-clear ₹ |
|---:|---:|---:|---:|---:|
| 10% | 97.5% | 100.0% | 100.0% | ₹0 |
| 20% | 97.5% | 100.0% | 100.0% | ₹0 |
| 30% | 98.8% | 100.0% | 100.0% | ₹0 |
| 40% | 88.8% | 100.0% | 100.0% | ₹0 |
| 50% | 90.0% | 98.8% | 99.5% | ₹0 |

**False-clear amount stayed at ₹0 across every corruption level tested,
even at 50%** — the system never once confidently matched something it
should have flagged, across a genuinely harsh stress test. Match rate
dips at 40–50% because more corruption correctly produces more honest
exceptions, not the same clean rate — that's the system behaving
correctly, not degrading.

Reproduce: `POST /api/benchmark {"n_orders": 80, "corruption_rates": [0.1,0.2,0.3,0.4,0.5], "use_llm": true}`

---

## Proof it works

- **32/32 automated tests passing**, including 4 tests specifically
  verifying the group-split matcher never falsely combines unrelated
  bank lines or rescues a genuine amount mismatch.
- **CI green** on every commit (GitHub Actions).
- **Verified live in a browser**: real clicks (not just API calls) —
  running a reconciliation, opening the evidence drawer, performing a
  human override, confirming the audit trail — all working end to end
  against the actual production URL.
- **Real Gemini integration**, not a mock, with a tested offline
  heuristic fallback when no LLM key is configured.

---

## Honest limitations, stated up front

1. **Scale ceiling on the live demo.** Very large batches (~200+
   orders with many LLM calls) can occasionally hit Vercel's serverless
   duration cap or Gemini rate-limit delays under heavy usage. The
   deterministic matcher itself runs in under a millisecond regardless
   of batch size — any slowness is network/LLM latency, not the core
   engine. A 1,000-order scale test and this limitation are both
   documented in the main README.
2. **The mirror many-to-many case isn't built.** One bulk bank credit
   covering *multiple different* orders' settlements (batched by the
   gateway) is the natural next extension of the same combinatorial
   search, but needs a batch-reference narration format the generator
   doesn't produce yet. Noted explicitly in the roadmap, not hidden.
3. **Deliberately out of scope, on purpose:** cash forecasting,
   reconciliation aging as a standalone view, a general-purpose finance
   chatbot, payroll, invoicing, tax workflows, expense management, or
   any other accounting product. Track 4 asks for one loop done
   credibly, not a broader finance suite — every one of those would
   dilute that story rather than strengthen it.
4. Ground-truth accuracy on synthetic data is high partly because the
   matcher's tolerance thresholds and the generator's anomaly ranges
   were authored by the same person, together — this demonstrates the
   pipeline is internally consistent, not that it's proven against
   arbitrary real-world noise.

---

## Stack

Python / FastAPI backend, stateless API (every response is
self-contained — survives serverless cold starts by design), static
HTML/JS dashboard (no build step), deployed to Vercel as a Python
serverless function. Docker + docker-compose for local one-command run.

---

## What's left

Nothing on the engineering side — repo, live deployment, tests, and
documentation are all in sync. Remaining: record the pitch video, fill
out the application form, submit.

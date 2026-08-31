# ReconIQ — 5-minute pitch video script

Canonical demo batch, locked: **`n_orders=100, seed=42`** — chosen for
diversity (all six match methods present: exact, fuzzy, `group_split`,
`batch_settlement`, `refunded`, `llm`), not for a flattering number.
Reproduce anytime: `POST /api/run {"n_orders": 100, "seed": 42}` against
https://reconiq-mocha.vercel.app. Numbers below are from that batch —
re-verify right before recording (see `docs/PROJECT_SUMMARY.md`'s
canonical batch table) and update if a fresh run differs.

Record in OBS/Loom against the live deployment, 1080p, cursor visible.
Do one full dry run off-camera first so the LLM calls have already
warmed a network path and timings below are realistic. **Have a backup
recording or screenshots ready** in case the live site is slow or
unreachable during the actual take — never rely entirely on the network
during a recorded pitch.

---

## 0:00–0:30 — Problem

Cold open, no slides, nothing clicked yet.

> "Finance teams reconcile orders, payment settlements, and bank
> transactions by hand. It gets much harder when references are broken,
> settlements arrive split across multiple bank lines, money gets
> partially refunded, or several orders get batched into one settlement.
>
> ReconIQ automates that loop — without blindly matching everything it's
> handed."

## 0:30–1:00 — Architecture, fast

Show `docs/architecture.html` (or a screenshot of it) for a few seconds.

> "Three sources — orders, Razorpay settlements, bank statement —
> normalize into one schema. A deterministic engine tries exact match,
> fuzzy match, then group matching in both directions, then
> refund-aware matching. Only what's left unresolved goes to an LLM,
> and even then, only past a confidence gate. Everything else is an
> honest exception."

Don't linger — one breath, move on. Don't explain FastAPI, Docker, or
the stack.

## 1:00–2:15 — Run the controller

Load the live site, set the canonical batch (100 orders), click **Run
Controller**. Let it load on screen, don't cut away.

> "This is a live batch — 100 transactions, mixing clean payments, fee
> adjustments, split settlements, batch settlements, full and partial
> refunds, and garbled bank narrations."

Point at the financial position cards as they render:

> "₹12.6 lakh processed. ₹9.9 lakh reconciled. ₹2.7 lakh needs attention.
> ₹1 lakh was legitimately refunded — tracked separately, never counted
> as reconciled money it isn't."

Point at the verdict banner:

> "And the headline call: **Review before closing.** ₹2.7 lakh remains
> unresolved, ₹2.6 lakh of that above the materiality threshold. A
> finance controller doesn't need to read twenty exception rows to know
> the books aren't ready to close today — this tells them directly."

This is the moment that establishes it as a *finance controller*, not
just a matcher.

## 2:15–3:00 — The "Why?" moment

Click **Why?** on an `llm`-method match.

> "Every decision has a lineage. This one's bank narration didn't
> contain the full UTR — truncated by the bank's own export format. The
> model reasoned over the amount, the partial UTR, and the narration,
> and returned a confidence score. Above our threshold, it clears
> automatically."

Click **Why?** on a high-priority exception (e.g. a `duplicate_candidate`
or `amount_mismatch`).

> "And this one didn't clear — the evidence didn't add up, so it's
> flagged for a human, not guessed at. That's the same principle behind
> every layer of this system: match what you can prove, escalate what
> you can't resolve deterministically, and never silently guess."

This is probably the single strongest product interaction — don't rush
it, but don't overstay either (aim for well under a minute).

## 3:00–3:40 — Prove AI isn't decorative

Scroll to the Rules-only vs. Rules+AI comparison.

> "We don't claim the AI layer is useful just because a model is in the
> stack. Every run also computes what the deterministic rules alone
> would have done, on the exact same batch, no cherry-picking."

Read the real numbers on screen:

> "90% match rate and 98% accuracy with rules only. 92% match rate and
> **100% accuracy** with the AI layer added — a two-point uplift, on the
> same 100 transactions, measured every single run."

## 3:40–4:20 — Prove it's not cherry-picked

Scroll to the corruption benchmark / multi-seed section.

> "One more thing: this isn't tuned to look good on one lucky batch.
> This runs several independent batches at each of five corruption
> levels — from lightly messy to heavily corrupted — and reports the
> average with the spread across those runs.
>
> **False-clear amount stays at ₹0 at every level tested, including 50%
> corruption.** The system never once confidently matched something it
> should have flagged, across a genuinely harsh stress test — that's the
> number that actually matters for a finance system, not the accuracy
> percentage."

## 4:20–5:00 — Close

Cut back to a clean shot of the summary cards and verdict banner.

> "ReconIQ doesn't try to match everything. It matches what it can
> prove, uses AI only where deterministic rules structurally can't help,
> and leaves genuine uncertainty unresolved rather than guessing. Every
> decision — automated or human-reviewed — is in an audit trail you can
> export and check yourself.
>
> The result isn't just automation. It's an auditable finance control.
>
> Repo and live demo are both linked below."

Stop. Don't use the last 30 seconds to list technologies.

---

## Delivery notes

- Don't apologize for or over-explain a stall on camera — Vercel cold
  starts can add a beat of latency on the first request after idle.
  Just wait, don't narrate the wait.
- If a live LLM call is slow during recording, that's fine to leave in
  — it's honest (a real network call, not a canned demo) — but don't let
  dead air run past ~3 seconds; talk over it.
- **If the live Gemini call fails on camera** (rate limit, transient
  503), the offline heuristic fallback still produces a labeled,
  reasonable result — don't panic-cut. That graceful degradation is
  itself worth having happen on camera if it does; it's a real answer
  to "what happens when the AI provider is down," not a failure.
- First take will likely run 6–7 minutes. Cut kit, in order: trim the
  second "Why?" click in the 2:15–3:00 section first; if still over,
  shorten the architecture beat at 0:30–1:00 to a single sentence.
- Do the dry run early enough (not minutes before recording) that a
  Gemini free-tier rate limit from testing has time to clear.

# ReconIQ — 5-minute pitch video script

Three differentiators, deep, not a feature tour: **(1) money-weighted
accuracy / false-clear rate, (2) rules-only vs. rules+AI on the same
batch, (3) human override with a distinct audit entry.** The corruption
benchmark gets a brief closing beat. Everything else (evidence drawer,
CSV upload, configurable policy) stays out of the script — mention only
if a question comes up, don't tour it.

Record in OBS/Loom against the live deployment (https://reconiq-mocha.vercel.app),
1080p, cursor visible. Do one full dry run off-camera first so the LLM
calls have already warmed a network path and timings below are realistic.

---

## 0:00–0:30 — Cold open, no slides

Just the screen, live site loaded, nothing clicked yet.

> "This is ReconIQ, an AI finance controller for reconciliation — Track 4
> of the Razorpay buildathon. Before I show you a feature tour, I want to
> show you the one number that actually matters for a finance system,
> and then prove it."

Click **Run reconciliation**. Let it load on screen — don't cut away.

## 0:30–1:30 — Money-weighted accuracy and false-clear rate

Cards are on screen: ground-truth accuracy, amount reconciled, amount at
risk, amount accuracy, false-clear amount, safe-miss amount.

> "Most reconciliation demos report one number — match rate, how much got
> auto-resolved. That tells you volume, not correctness. ReconIQ reports
> two more things most don't: ground-truth accuracy, scored against a
> label the matcher never sees — and money-weighted accuracy, because a
> ₹200 order and a ₹2 lakh order are not the same mistake.
>
> And then it splits the wrong answers into two kinds, because they are
> not equally bad. A **false clear** is confidently wrong — money moves
> that shouldn't have. A **safe miss** is conservative — it just sits
> flagged for review. Across every batch I've run, false-clear amount has
> been zero. The system is tuned to prefer 'I don't know' over a
> confident wrong answer — which is the entire thesis of the project."

Point at the ₹0 false-clear card specifically.

## 1:30–2:45 — Rules-only vs. rules+AI, same batch

Scroll to the comparison table.

> "Here's the part I actually want you to scrutinize: does the AI layer
> earn its place, or is it decoration on top of a rules engine? Every
> run also computes what the deterministic rules alone would have done
> on the *exact same batch* — no cherry-picking, no separate run."

Read the two columns out loud with real numbers on screen (e.g. "82.9%
match rate and 91.4% accuracy with rules only — 91.4% match rate and
100% accuracy with the AI layer added").

> "The AI only ever touches what the rules already gave up on — a
> settlement whose bank reference got truncated or garbled by the bank's
> own narration format. It's not narrating a decision the rules already
> made. It's making a decision the rules structurally can't."

Optionally: click one "Why?" button on an `llm`-method match here, show
the reasoning text for 5 seconds, then close it. Keep it brief — this is
a supporting beat, not the second differentiator.

## 2:45–3:45 — Human override, live

Scroll to the exception list.

> "This isn't a fire-and-forget batch job. A finance-ops reviewer can
> act on any decision."

Click **Manually match** on a real exception (pick a bank line from the
dropdown first, on camera). Let the dashboard re-render.

> "That override is now in the audit trail, tagged distinctly —
> `human_override` — so six months from now nobody can confuse a
> decision a person made with one the system made on its own."

Scroll to the audit trail, point at the `human_override` row sitting
right above the automated `llm_resolve`/`rule` rows.

## 3:45–4:30 — Corruption benchmark (brief)

Scroll to the benchmark section, either run it live if time allows or
show a pre-run chart.

> "One more thing, quickly: this isn't tuned to look good on one lucky
> batch. This chart re-runs the whole pipeline across five corruption
> levels, from lightly messy to heavily corrupted data. Accuracy holds up
> across the range — and false-clear stays at zero at every level."

## 4:30–5:00 — Close

Cut back to a clean shot of the summary cards.

> "Three things: it tells you not just what got matched, but what got
> matched *correctly*, weighted by money. It proves the AI layer earns
> its place instead of asserting it. And every decision — automated or
> human — is in an audit trail you can export and check yourself.
>
> This is ReconIQ. Repo and live demo are both linked below."

---

## Delivery notes

- Don't apologize for or over-explain the two failed screenshot/timeout
  quirks if something stalls for a second on camera — Vercel cold starts
  can add a beat of latency on the first request after idle. Just wait,
  don't narrate the wait.
- If a live LLM call is slow during recording, that's fine to leave in —
  it's honest (real network call, not a canned demo), but don't let dead
  air run past ~3 seconds; talk over it.
- Have a fallback: if the live Gemini call fails on camera (rate limit,
  transient 503), the offline heuristic fallback still produces a
  reasonable result — don't panic-cut, the graceful degradation IS one of
  the things worth having happen on camera if it does.
- First take will likely run 6–7 minutes. Cut kit: the "Why?" drill-down
  in section 2 is the first thing to trim if you're over time.

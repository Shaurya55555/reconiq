"""Pluggable LLM layer, used only for the cases rules could not resolve:
a settlement whose UTR does not appear verbatim in any remaining bank
line, because the narration was truncated, prefixed with bank-specific
noise, or had digits transposed.

Provider is selected by env var LLM_PROVIDER (openai | anthropic | gemini |
ollama). With no API key configured, `heuristic_resolve` runs instead so
the demo works fully offline -- it is a real (if weaker) resolution
strategy, not a stub, and every verdict still carries its reasoning and
confidence into the audit trail exactly like a model call would.
"""
from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher

SYSTEM_PROMPT = """You are a payments reconciliation analyst. You are given one \
settlement that could not be matched to a bank statement line by exact UTR lookup, \
and a shortlist of unmatched bank lines. Decide which bank line (if any) is the \
same payment, reasoning from the narration text, amount and date. Respond with \
strict JSON: {"bank_line_id": "<id or null>", "confidence": <0..1>, "reasoning": "<one sentence>"}. \
If no candidate is plausible, return bank_line_id null and explain why."""


def _build_user_prompt(case: dict, candidates: list[dict]) -> str:
    lines = [
        f"Settlement to resolve: expected_utr={case['expected_utr']!r}, "
        f"expected_amount={case['expected_amount']}, customer={case['customer']!r}",
        "Candidate bank lines:",
    ]
    for c in candidates:
        lines.append(f"  - id={c['line_id']!r} amount={c['amount']} date={c['value_date']} "
                      f"narration={c['narration']!r}")
    return "\n".join(lines)


def heuristic_resolve(case: dict, candidates: list[dict]) -> dict:
    """Offline fallback: score candidates by amount match + fuzzy string
    similarity between the expected UTR and the digits embedded in the
    narration. Deterministic, explainable, no network call.
    """
    expected_digits = re.sub(r"\D", "", case["expected_utr"])
    best, best_score = None, 0.0

    for c in candidates:
        if c["amount"] != case["expected_amount"]:
            continue
        narration_digits = re.sub(r"\D", "", c["narration"])
        sim = SequenceMatcher(None, expected_digits, narration_digits).ratio()
        if sim > best_score:
            best, best_score = c, sim

    if best and best_score >= 0.5:
        return {
            "bank_line_id": best["line_id"],
            "confidence": round(min(0.5 + best_score * 0.4, 0.95), 2),
            "reasoning": (
                f"Amount matched exactly ({best['amount']}) and the narration's digit "
                f"sequence is {best_score:.0%} similar to the expected UTR "
                f"{case['expected_utr']} once punctuation is stripped."
            ),
        }
    return {
        "bank_line_id": None,
        "confidence": 0.0,
        "reasoning": "No candidate bank line shares this settlement's amount and a "
                     "sufficiently similar digit sequence to the expected UTR.",
    }


def _call_openai(case, candidates) -> dict | None:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(case, candidates)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def _call_anthropic(case, candidates) -> dict | None:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(case, candidates)}],
    )
    text = resp.content[0].text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else None


def _call_gemini(case, candidates) -> dict | None:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=os.getenv("LLM_MODEL", "gemini-flash-lite-latest"),
        contents=SYSTEM_PROMPT + "\n\n" + _build_user_prompt(case, candidates),
    )
    match = re.search(r"\{.*\}", resp.text, re.DOTALL)
    return json.loads(match.group(0)) if match else None


def _call_ollama(case, candidates) -> dict | None:
    import httpx
    resp = httpx.post(
        f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/generate",
        json={
            "model": os.getenv("LLM_MODEL", "llama3.1"),
            "prompt": SYSTEM_PROMPT + "\n\n" + _build_user_prompt(case, candidates),
            "stream": False,
            "format": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])


_PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "ollama": _call_ollama,
}


QA_SYSTEM_PROMPT = """You are a reconciliation analyst answering a question about one \
completed reconciliation run. You are given the run's summary metrics, its exception \
list (records that could not be auto-matched), and -- if the question named a specific \
order -- that order's own match or exception record. Answer only from this data, in 2-4 \
sentences. If the question asks for something not present in the data, say so explicitly \
rather than guessing."""

_ORDER_ID_RE = re.compile(r"\bORD[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b", re.IGNORECASE)


def _find_order_record(question: str, exceptions: list[dict], matches: list[dict]) -> dict | None:
    """A question naming a specific order (e.g. "ORD1053") needs that
    order's own record, not just the aggregate summary/exception list --
    the exception list alone only covers orders that DIDN'T match, so a
    question about a matched order would otherwise look like the data is
    simply missing. Only the order's exact record is looked up (not a
    bulk dump of every match), so this stays cheap regardless of batch
    size.
    """
    m = _ORDER_ID_RE.search(question)
    if not m:
        return None
    order_id = m.group(0).upper()
    for e in exceptions:
        if str(e.get("order_id", "")).upper() == order_id:
            return {"order_id": order_id, "status": "exception", "record": e}
    for match in matches:
        if str(match.get("order_id", "")).upper() == order_id:
            return {"order_id": order_id, "status": "matched", "record": match}
    return {"order_id": order_id, "status": "not_found", "record": None}


def _heuristic_answer(question: str, summary: dict, exceptions: list[dict],
                       matches: list[dict] | None = None) -> str:
    matches = matches or []
    found = _find_order_record(question, exceptions, matches)
    if found:
        oid, status, record = found["order_id"], found["status"], found["record"]
        if status == "matched":
            return (f"{oid} matched via method '{record.get('method')}' "
                     f"(confidence {record.get('confidence')}): {record.get('note', '')}")
        if status == "exception":
            return f"{oid} is an exception ({record.get('type')}): {record.get('reason', '')}"
        return (f"{oid} isn't in this run's {summary['total_orders']}-order batch -- "
                "it's not among the matches or the exceptions.")

    q = question.lower()
    by_type: dict[str, int] = {}
    for e in exceptions:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1

    if "match rate" in q or "how many matched" in q:
        return (f"{summary['matched']} of {summary['total_orders']} orders matched "
                f"({summary['match_rate']:.1%}), with {summary['exception_count']} exceptions.")
    if "exception" in q or "unresolved" in q or "unmatched" in q:
        if not by_type:
            return "There are no exceptions in this run -- every order matched."
        breakdown = ", ".join(f"{v} {k}" for k, v in sorted(by_type.items(), key=lambda x: -x[1]))
        return f"There are {len(exceptions)} exceptions: {breakdown}."
    if "throughput" in q or "how fast" in q or "how long" in q:
        return (f"This run processed {summary['total_orders']} orders in "
                f"{summary.get('elapsed_seconds', '?')}s "
                f"({summary.get('throughput_records_per_sec', '?')} records/sec).")
    if by_type:
        breakdown = ", ".join(f"{v} {k}" for k, v in sorted(by_type.items(), key=lambda x: -x[1]))
        return (f"I can't answer that precisely offline, but for context: match rate is "
                f"{summary['match_rate']:.1%} and open exceptions are: {breakdown}.")
    return (f"I can't answer that precisely offline (no LLM_PROVIDER configured), but this "
            f"run's match rate was {summary['match_rate']:.1%} with no open exceptions.")


def answer_question(question: str, summary: dict, exceptions: list[dict],
                     matches: list[dict] | None = None) -> str:
    matches = matches or []
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider not in _PROVIDERS:
        return _heuristic_answer(question, summary, exceptions, matches)

    found = _find_order_record(question, exceptions, matches)
    order_context = f"\nNamed order lookup: {json.dumps(found)}" if found else ""
    context = (f"Summary: {json.dumps(summary)}\n"
               f"Exceptions ({len(exceptions)}): {json.dumps(exceptions[:50])}"
               f"{order_context}\n"
               f"Question: {question}")
    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": QA_SYSTEM_PROMPT},
                          {"role": "user", "content": context}],
                temperature=0,
            )
            return resp.choices[0].message.content
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=300, system=QA_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            return resp.content[0].text
        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            resp = client.models.generate_content(
                model=os.getenv("LLM_MODEL", "gemini-flash-lite-latest"),
                contents=QA_SYSTEM_PROMPT + "\n\n" + context,
            )
            return resp.text
        if provider == "ollama":
            import httpx
            resp = httpx.post(
                f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/generate",
                json={"model": os.getenv("LLM_MODEL", "llama3.1"),
                      "prompt": QA_SYSTEM_PROMPT + "\n\n" + context, "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["response"]
    except Exception as exc:
        return f"[{provider} call failed: {exc}] " + _heuristic_answer(question, summary, exceptions, matches)
    return _heuristic_answer(question, summary, exceptions, matches)


def resolve(case: dict, candidates: list[dict]) -> dict:
    """Single entry point the matcher calls. Falls back to the offline
    heuristic on missing config or any provider error, so a demo never
    dies mid-run because of a network blip or a missing key.
    """
    if not candidates:
        return {"bank_line_id": None, "confidence": 0.0,
                "reasoning": "No unmatched bank lines remain to consider."}

    provider = os.getenv("LLM_PROVIDER", "").lower()
    fn = _PROVIDERS.get(provider)
    if fn is None:
        return heuristic_resolve(case, candidates)

    try:
        verdict = fn(case, candidates)
        if verdict and "bank_line_id" in verdict and "confidence" in verdict:
            return verdict
    except Exception as exc:  # provider down, bad key, quota, etc.
        fallback = heuristic_resolve(case, candidates)
        fallback["reasoning"] = f"[{provider} call failed: {exc}; fell back to heuristic] " + fallback["reasoning"]
        return fallback
    return heuristic_resolve(case, candidates)

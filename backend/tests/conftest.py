"""Tests must be deterministic and offline regardless of a developer's
local .env -- force the heuristic path even if LLM_PROVIDER/GEMINI_API_KEY
are set on the machine running pytest. CI never has these set anyway; this
just makes `pytest` behave the same on a laptop with a live key configured.

Set (not deleted) to "" rather than popped: app.main calls load_dotenv()
on import with the default override=False, which only fills in a variable
if it's *absent* -- an empty string still counts as present, so this
survives that import instead of being silently overwritten by .env.
"""
import os

for _var in ("LLM_PROVIDER", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_MODEL"):
    os.environ[_var] = ""

"""Vercel Python entrypoint. Re-exports the FastAPI ASGI app from
backend/app/main.py so the @vercel/python runtime can serve it directly --
same application code runs identically locally (via uvicorn) and here.
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402  (re-exported for the Vercel runtime)

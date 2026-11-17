"""Settings read from environment variables (optionally from a local .env file, which is never committed)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI", "")            # empty -> file storage
MONGODB_DB = os.getenv("MONGODB_DB", "maintenai")
DATA_DIR = Path(os.getenv("MAINTENAI_DATA_DIR", ROOT / "data" / "app"))
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o]
FRONTEND_DIST = ROOT / "frontend" / "dist"
RESULTS_PATH = ROOT / "artifacts" / "metrics" / "results.json"
FIGURES_DIR = ROOT / "reports" / "figures"
MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "20000"))

#!/usr/bin/env bash
# Start the MaintenAI web application on http://localhost:8000 (macOS / Linux).
#   bash run_web.sh            set up if needed, train models if missing, start the server
#   bash run_web.sh --dev      also start the Vite dev server on :5173 (needs Node.js)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
PORT="${PORT:-8000}"
"$PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), f"Python 3.10-3.13 needed, found {sys.version.split()[0]}"'
if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null && ! brew list libomp >/dev/null 2>&1; then
  echo "Installing libomp (needed by XGBoost)..."; brew install libomp
fi
[[ -d .venv ]] || "$PY" -m venv .venv
source .venv/bin/activate
if [[ ! -f .venv/.deps-ok ]] || [[ requirements.txt -nt .venv/.deps-ok ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  touch .venv/.deps-ok
fi
[[ -f data/raw/ai4i2020.csv ]] || { echo "Missing data/raw/ai4i2020.csv - see README 'Getting the dataset'"; exit 1; }
# Train the deployed models the first time (about 2 minutes), or when the library versions changed.
python - <<'PY' || python -m ml.inference.train
import sys
from ml.inference.predictor import BundleError, Predictor
try:
    Predictor("engineered"); Predictor("raw")
except BundleError as e:
    print("Models need training:", e); sys.exit(1)
PY
if [[ ! -f frontend/dist/index.html ]]; then
  if command -v npm >/dev/null; then (cd frontend && npm ci && npm run build)
  else echo "frontend/dist is missing and Node.js is not installed; the API will run without the web pages."; fi
fi
if [[ "${1:-}" == "--dev" ]]; then
  (cd frontend && [[ -d node_modules ]] || npm ci; cd frontend && npm run dev) &
  echo "Dev UI: http://localhost:5173"
fi
echo "MaintenAI: http://localhost:$PORT   (API docs: http://localhost:$PORT/docs)   Ctrl+C to stop"
( sleep 3; command -v open >/dev/null && open "http://localhost:$PORT" ) >/dev/null 2>&1 &
exec python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port "$PORT"

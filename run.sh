#!/usr/bin/env sh
# KNULLD - set up (first run only, ~5 min) and start the app. Needs Python 3.12+ and internet on first run.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "[1/3] Creating Python environment..."
  python3 -m venv .venv || { echo "Python 3.12+ not found."; exit 1; }
  echo "[2/3] Installing PyTorch (CPU build)..."
  .venv/bin/python -m pip install -q --upgrade pip
  if [ "$(uname)" = "Linux" ]; then CPU="--index-url https://download.pytorch.org/whl/cpu"; fi
  .venv/bin/python -m pip install -q torch==2.11.0 $CPU || { rm -rf .venv; echo "Install failed."; exit 1; }
  echo "[3/3] Installing the rest..."
  .venv/bin/python -m pip install -q -r requirements.txt || { rm -rf .venv; echo "Install failed."; exit 1; }
fi
echo "Starting KNULLD at http://localhost:8765  (Ctrl+C to stop)"
.venv/bin/python app.py

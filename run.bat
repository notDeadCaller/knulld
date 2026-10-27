@echo off
rem KNULLD - set up (first run only, ~5 min) and start the app. Needs Python 3.12+ and internet on first run.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [1/3] Creating Python environment...
  python -m venv .venv || (echo Python 3.12+ not found. Install it from python.org, tick "Add to PATH", then run this again. & pause & exit /b 1)
  echo [2/3] Installing PyTorch, CPU build...
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\python -m pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu || goto fail
  echo [3/3] Installing the rest...
  .venv\Scripts\python -m pip install -q -r requirements.txt || goto fail
)
echo Starting KNULLD at http://localhost:8765  (close this window to stop)
.venv\Scripts\python app.py
pause
exit /b
:fail
rmdir /s /q .venv
echo Install failed - check the internet connection and run this again.
pause
exit /b 1

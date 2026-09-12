import os
import sys
import subprocess
from pathlib import Path

# Ensure project directory is in PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# If executed with a Python outside .venv, automatically forward to project .venv
venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
    try:
        result = subprocess.run([str(venv_python)] + sys.argv, cwd=str(BASE_DIR))
        sys.exit(result.returncode)
    except Exception:
        pass

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Starting Multi-Document AI Knowledge Agent Platform")
    print(" Web UI & API available at: http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


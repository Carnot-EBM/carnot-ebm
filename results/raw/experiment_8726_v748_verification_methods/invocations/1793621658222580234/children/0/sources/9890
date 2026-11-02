"""REQ-REPORT-8407: complete local Python service timing uses no current LLM load."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.python_transaction_runner_8407 import main

if __name__ == "__main__":
    raise SystemExit(main())

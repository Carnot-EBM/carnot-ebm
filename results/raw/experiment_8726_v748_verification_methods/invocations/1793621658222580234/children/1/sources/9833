"""REQ-REPORT-8329: import the checkout directly for unbuffered CPU execution."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
print("[exp8329] phase=start_no_model_load completed=0 pending=1", flush=True)

from carnot.reporting.kv260_workload_runner_8329 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

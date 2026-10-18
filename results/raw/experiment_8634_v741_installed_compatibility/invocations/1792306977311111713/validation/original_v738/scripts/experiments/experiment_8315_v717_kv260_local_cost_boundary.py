"""REQ-REPORT-8315: direct CPU-only CLI imports this checkout from any directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
print("[exp8315] phase=start_no_model_load completed=0 pending=1", flush=True)

from carnot.reporting.kv260_local_cost_execution_8315 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

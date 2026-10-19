"""REQ-VERIFY-8356: direct execution must reach the same checked CPU runner."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.kv260_workload_runner_8356 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

"""REQ-REPORT-8227: bind direct CLI imports to this checkout from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.concurrency_execution_8227 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

"""REQ-REPORT-8335: direct invocation restores imports from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.reserved_prediction_execution_8335 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

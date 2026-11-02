"""REQ-REPORT-8484: standalone CPU reconciliation preserves every historical primary."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v729_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

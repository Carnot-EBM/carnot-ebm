"""REQ-REPORT-8289: import the capstone in fresh processes outside the checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v715_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

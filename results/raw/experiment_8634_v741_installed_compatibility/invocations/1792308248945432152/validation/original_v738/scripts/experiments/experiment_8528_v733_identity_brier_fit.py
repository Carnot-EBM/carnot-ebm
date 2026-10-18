"""REQ-REPORT-8528: the direct script keeps numerical and validation work reusable."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v733_identity_brier_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

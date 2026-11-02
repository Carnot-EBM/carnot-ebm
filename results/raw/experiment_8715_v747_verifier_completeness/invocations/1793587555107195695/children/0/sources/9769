"""REQ-REPORT-8242: resolve this checkout from a private working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.independent_concurrent_execution_8242 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

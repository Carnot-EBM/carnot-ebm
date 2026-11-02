"""REQ-REPORT-8236: make this checkout importable from a private working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.qualified_concurrency_execution_8236 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

"""REQ-REPORT-8368: thin entry point for typed runtime closure qualification."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.runtime_closure_execution_8368 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

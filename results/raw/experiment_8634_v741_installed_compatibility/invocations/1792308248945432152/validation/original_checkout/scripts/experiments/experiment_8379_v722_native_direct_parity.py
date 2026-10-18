"""REQ-REPORT-8379: allow direct invocation without an ambient import path."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.native_direct_runner_8379 import main

if __name__ == "__main__":
    raise SystemExit(main())

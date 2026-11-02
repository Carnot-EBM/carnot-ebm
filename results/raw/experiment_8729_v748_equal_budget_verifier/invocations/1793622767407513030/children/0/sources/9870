"""REQ-VERIFY-8376: the script entry works without an ambient import path."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.direct_atomic_runner_8376 import main

if __name__ == "__main__":
    raise SystemExit(main())

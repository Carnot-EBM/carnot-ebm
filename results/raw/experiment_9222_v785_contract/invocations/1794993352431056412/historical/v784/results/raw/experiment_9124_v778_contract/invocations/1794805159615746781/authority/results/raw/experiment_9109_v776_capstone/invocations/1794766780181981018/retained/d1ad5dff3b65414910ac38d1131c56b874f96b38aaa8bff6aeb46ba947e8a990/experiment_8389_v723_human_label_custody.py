"""REQ-VERIFY-8389: direct CLI execution keeps current model activity at zero."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.human_label_custody_runner_8389 import main

if __name__ == "__main__":
    raise SystemExit(main())

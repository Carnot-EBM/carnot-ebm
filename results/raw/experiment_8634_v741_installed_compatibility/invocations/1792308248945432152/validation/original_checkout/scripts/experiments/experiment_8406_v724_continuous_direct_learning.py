"""REQ-REPORT-8406: direct serving evidence uses no current LLM invocation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.continuous_direct_runner_8406 import main

if __name__ == "__main__":
    raise SystemExit(main())

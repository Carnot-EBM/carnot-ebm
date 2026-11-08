"""REQ-REPORT-8924: direct canary CLI supports foreign-directory cold replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.qwen_natural_runner_8924 import main

if __name__ == "__main__":
    raise SystemExit(main())

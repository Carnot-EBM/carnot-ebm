#!/usr/bin/env python3
"""REQ-REPORT-8045: resolve shipped code from an arbitrary caller directory."""

from pathlib import Path
import sys

print("Exp8045 start: resolving local code; model invocations=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8045_v697_scorer_workspace import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

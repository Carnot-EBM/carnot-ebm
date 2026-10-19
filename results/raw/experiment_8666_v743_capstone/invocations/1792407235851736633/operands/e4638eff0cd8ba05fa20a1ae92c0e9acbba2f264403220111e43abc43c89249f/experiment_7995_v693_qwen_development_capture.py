#!/usr/bin/env python3
"""REQ-REPORT-7995: use repo imports even during external-CWD cold replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7995_v693_qwen_development_capture import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run REQ-REPORT-7981 with imports bound to this checkout from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7981_v692_qwen_stream_capture import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

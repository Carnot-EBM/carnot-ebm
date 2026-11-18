"""REQ-REPORT-9107-V776: score or refuse the archived human panel from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.qwen_judge_runner_9107 import main

raise SystemExit(main())

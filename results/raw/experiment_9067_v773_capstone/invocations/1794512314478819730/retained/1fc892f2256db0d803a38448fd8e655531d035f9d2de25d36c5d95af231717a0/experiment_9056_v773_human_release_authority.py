"""REQ-REPORT-9056-AUTHORITY: run the pinned release audit from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.mdh_human_release_runner_9056 import main

raise SystemExit(main())

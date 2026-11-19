"""REQ-REPORT-9098-V776: qualify the fixed GGUF runtime from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.embedding_runtime_runner_9098 import main

raise SystemExit(main())

"""REQ-REPORT-8989: run a model-free audit of retained RAGognize release bytes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.natural_annotation_runner_8989 import main

raise SystemExit(main())

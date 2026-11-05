"""REQ-REPORT-8802: direct deterministic-verifier prototype CLI; no LLM calls."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.causal_memory_runner_8802 import main

if __name__ == "__main__":
    raise SystemExit(main())

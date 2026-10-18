#!/usr/bin/env python3
"""REQ-REPORT-8022: use repository code even from a private working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8022_v695_likelihood_protocol import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

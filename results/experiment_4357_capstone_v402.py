#!/usr/bin/env python3
"""Run Exp 4357 v402 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4357, SCENARIO-CAPSTONE-4357.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.capstone_v402_4357 import write_artifact  # noqa: E402


if __name__ == "__main__":
    output = write_artifact(REPO_ROOT)
    print(output.read_text(encoding="utf-8"), end="")

#!/usr/bin/env python3
"""Run Exp 3840 publication-gate regression confirmation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting import publication_gate_regression_3840 as exp3840  # noqa: E402


def main() -> int:
    output = exp3840.run(ROOT)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(artifact["honest_verdict"])
    print(f"wrote {output.relative_to(ROOT)}")
    return 1 if str(artifact["honest_verdict"]).startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())

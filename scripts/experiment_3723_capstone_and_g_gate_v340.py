#!/usr/bin/env python3
"""Run Exp 3723 v340 convergence capstone and hardened G-gate recheck."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting.capstone_and_g_gate_v340_3723 import write_artifact  # noqa: E402


def main() -> int:
    output_path = write_artifact(REPO_ROOT)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

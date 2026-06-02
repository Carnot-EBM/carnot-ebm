#!/usr/bin/env python3
"""Run Exp 3707 selection-diagnosis formal closure."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(ROOT / "python"))

from carnot.reporting import selection_diagnosis_formal_closure_3707 as exp  # noqa: E402


def main() -> int:
    output = exp.write_artifact(ROOT)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

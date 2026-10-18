#!/usr/bin/env python3
"""Run Exp 3706 held-out reconciliation for the shipped detector."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(ROOT / "python"))


def _load_exp_module():
    return importlib.import_module("carnot.pipeline.reconcile_shipped_detector_heldout_3706")


exp = _load_exp_module()


def main() -> int:
    output = exp.write_artifact(
        ROOT,
        tests_run=[
            ".venv/bin/pytest tests/python/test_experiment_3706_reconcile_shipped_detector_heldout.py -q",
            ".venv/bin/pytest tests/python/test_second_pair_detector_3671.py -q",
            ".venv/bin/pytest tests/python -q",
        ],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

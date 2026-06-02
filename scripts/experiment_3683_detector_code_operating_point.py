#!/usr/bin/env python3
"""Run Exp 3683 detector code operating point hardening."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def _load_exp_module():
    path = ROOT / "python/carnot/pipeline/detector_code_operating_point_3683.py"
    spec = importlib.util.spec_from_file_location("detector_code_operating_point_3683", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load_exp_module()


def main() -> int:
    output = exp.write_artifact(
        ROOT,
        tests_run=[
            "pytest tests/python/test_experiment_3683_detector_code_operating_point.py"
        ],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

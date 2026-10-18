#!/usr/bin/env python3
"""Run Exp 3657 deployable second-pair detector."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

_MODULE_PATH = PYTHON_DIR / "carnot/pipeline/second_pair_detector.py"
_SPEC = importlib.util.spec_from_file_location("carnot_second_pair_detector", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"could not load detector module at {_MODULE_PATH}")
_DETECTOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DETECTOR
_SPEC.loader.exec_module(_DETECTOR)
OUTPUT_REL_PATH = _DETECTOR.OUTPUT_REL_PATH
build_artifact = _DETECTOR.build_artifact


def main() -> int:
    artifact = build_artifact(REPO_ROOT)
    output_path = REPO_ROOT / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

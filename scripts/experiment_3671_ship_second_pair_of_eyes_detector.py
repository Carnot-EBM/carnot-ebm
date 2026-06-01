#!/usr/bin/env python3
"""Exp 3671: ship the calibrated second-pair detector surface.

Spec: REQ-SPOE-3671, REQ-SPOE-3671-ARTIFACT.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> int:
    """Write the Exp 3671 terminal artifact."""

    module_path = REPO_ROOT / "python/carnot/pipeline/second_pair_detector.py"
    spec = importlib.util.spec_from_file_location("second_pair_detector_3671", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load detector module: {module_path}")
    detector = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = detector
    spec.loader.exec_module(detector)

    output = detector.write_ship_artifact(
        REPO_ROOT,
        output_path=detector.SHIP_OUTPUT_REL_PATH,
        tests_run=[
            ".venv/bin/pytest tests/python/test_second_pair_detector_3671.py -q",
            ".venv/bin/coverage run --source=python/carnot/pipeline/second_pair_detector.py -m pytest -o addopts='' tests/python/test_second_pair_detector_3671.py -q",
            ".venv/bin/coverage report --include='python/carnot/pipeline/second_pair_detector.py' --fail-under=100 --show-missing",
            ".venv/bin/python scripts/check_spec_coverage.py",
            ".venv/bin/pytest tests/python -q",
        ],
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

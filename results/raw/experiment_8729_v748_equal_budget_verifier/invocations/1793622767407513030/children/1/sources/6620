"""Experiment 1112: LLM failure exemplar corpus and Carnot cascade scoring.

Spec: REQ-VERIFY-1112, SCENARIO-VERIFY-1112
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

for _pkg in ["carnot", "carnot.eval", "carnot.verify", "carnot.models"]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [str(PYTHON_DIR / _pkg.replace(".", "/"))]  # type: ignore[attr-defined]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

from carnot.eval.llm_failure_exemplars import run_experiment  # noqa: E402


def main() -> None:
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

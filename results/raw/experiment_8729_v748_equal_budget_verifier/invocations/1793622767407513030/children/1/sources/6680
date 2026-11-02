#!/usr/bin/env python3
"""Experiment 1174: BiKA multiply-free SOS-KAN hardware analysis.

Spec refs: REQ-KAN-1174, SCENARIO-KAN-1174.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_bika_module() -> ModuleType:
    """Load bika_analysis without importing the heavier hardware package init."""
    module_path = PYTHON_DIR / "carnot" / "hardware" / "bika_analysis.py"
    spec = importlib.util.spec_from_file_location("_carnot_bika_analysis_direct", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bika = _load_bika_module()

EXP1148_PATH = bika.EXP1148_PATH
EXP1162_PATH = bika.EXP1162_PATH
DELIVERABLE = bika.DELIVERABLE_PATH


def main() -> int:
    """Write the Exp 1174 BiKA hardware analysis deliverable."""
    artifact = bika.run_experiment(
        exp1148_path=EXP1148_PATH,
        exp1162_path=EXP1162_PATH,
        deliverable_path=DELIVERABLE,
    )
    print(f"Standard SOS-KAN RM : {artifact['standard_kan_rm']}")
    print(f"Compressed SOS-KAN RM: {artifact['compressed_kan_rm']}")
    print(f"BiKA SOS-KAN NABS   : {artifact['bika_kan_nabs']}")
    print(f"NPU verdict         : {artifact['npu_feasibility_verdict']}")
    print(f"Honest verdict      : {artifact['honest_verdict']}")
    print(f"Deliverable         : {DELIVERABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

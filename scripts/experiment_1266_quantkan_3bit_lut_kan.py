#!/usr/bin/env python3
"""Experiment 1266: QuantKAN 3-bit PTQ plus LUT-KAN simulation.

Spec: REQ-KAN-1266, SCENARIO-KAN-1266
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

QUANTKAN_MODULE_PATH = PYTHON_DIR / "carnot" / "models" / "sos_kan_quantkan_lut.py"
_SPEC = importlib.util.spec_from_file_location("sos_kan_quantkan_lut_exp1266", QUANTKAN_MODULE_PATH)
quantkan: Any = importlib.util.module_from_spec(_SPEC)
sys.modules["sos_kan_quantkan_lut_exp1266"] = quantkan
assert _SPEC.loader is not None
_SPEC.loader.exec_module(quantkan)

BASELINE_PATH = PROJECT_ROOT / "results" / "experiment_1199_kantize_soskan_4bit_quantization.json"
CORPUS_PATH = PROJECT_ROOT / "results" / "fover_corpus_v5.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1266_quantkan_3bit_lut_kan.json"


def main() -> int:
    """Write the Exp 1266 deliverable JSON."""
    artifact = quantkan.run_experiment(
        baseline_path=BASELINE_PATH,
        corpus_path=CORPUS_PATH,
        deliverable_path=OUTPUT_PATH,
    )
    print(f"Wrote {OUTPUT_PATH}")
    print(
        f"verdict: {artifact['honest_verdict']} "
        f"(3bit_auroc={artifact['quantkan_3bit_auroc']:.4f}, "
        f"lut_speedup={artifact['lut_kan_speedup']:.1f}x)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

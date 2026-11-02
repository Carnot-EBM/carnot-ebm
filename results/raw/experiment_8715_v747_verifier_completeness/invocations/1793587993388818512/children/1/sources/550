"""Exp 3351 GateMate Latency Benchmark.

Spec refs: REQ-HW-103, SCENARIO-HW-103.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3351_gatemate_latency_benchmark.json"
)

INFERENCE_SUBSTRATE = "hardware_smoke"


def run_experiment() -> dict[str, Any]:
    started = time.perf_counter()
    
    # The GateMate n=16 tile has no host communication interface (AXI/UIO) in RTL.
    # It is blocked until a communication protocol over USB/JTAG or dedicated 
    # pins is implemented.
    artifact = {
        "experiment": 3351,
        "honest_verdict": "blocked_no_io_interface_in_rtl",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "gatemate_latency_us": None,
        "speedup_vs_cpu": None,
        "duration_s": time.perf_counter() - started,
        "blocked_reasons": ["GateMate n=16 RTL lacks a host communication interface (AXI/UIO)."],
    }
    
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    artifact = run_experiment()
    print(json.dumps({"honest_verdict": artifact["honest_verdict"], "result": str(RESULT_PATH)}))
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

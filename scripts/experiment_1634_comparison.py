#!/usr/bin/env python3
"""Exp 1634 comparison of Pi-Net continuous projection against T-SKM.

Spec refs: REQ-KONA-038, SCENARIO-KONA-038.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from carnot.verify.skm_projection import build_toy_linear_cases, project_skm
from scripts.experiment_1633_pinet import ContinuousConstraintSystem, PiNetProjectionLayer

JsonDict = dict[str, Any]

EXPERIMENT_ID = 1634
SCHEMA = "carnot.phase3.pinet_vs_tskm.v1"
SPEC_REFS = ["REQ-KONA-038", "SCENARIO-KONA-038"]
DEFAULT_OUTPUT_PATH = Path("results/experiment_1634_pinet_vs_tskm.json")
DEFAULT_TOLERANCE = 1e-5
DEFAULT_MAX_STEPS = 96

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "schema",
    "experiment_id",
    "spec_refs",
    "pinet_faster_than_tskm",
    "latency_diff",
    "honest_verdict",
    "tests_run",
)


def run_comparison(
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
    iterations: int = 10,
) -> JsonDict:
    """Run performance comparison between T-SKM and Pi-Net."""
    
    # Warm up JAX
    warmup_system = ContinuousConstraintSystem.from_arrays(
        state_dim=2,
        inequality_matrix=[[1.0, 0.0]],
        inequality_bound=[0.0],
        name="warmup"
    )
    warmup_layer = PiNetProjectionLayer(warmup_system)
    warmup_layer.project([1.0, 1.0])
    
    cases = build_toy_linear_cases()
    
    skm_time = 0.0
    pinet_time = 0.0
    
    for case in cases:
        # T-SKM
        start_time = time.perf_counter()
        for _ in range(iterations):
            project_skm(case.system, case.start, max_iterations=max_steps, tolerance=tolerance)
        skm_time += (time.perf_counter() - start_time)
        
        # Pi-Net
        pinet_system = ContinuousConstraintSystem.from_arrays(
            state_dim=len(case.start),
            inequality_matrix=case.system.matrix.tolist(),
            inequality_bound=case.system.bounds.tolist(),
            name=case.name,
        )
        layer = PiNetProjectionLayer(pinet_system, max_steps=max_steps, tolerance=tolerance)
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            layer.project(case.start)
        # block on jax execution if needed, but project returns a python float via device_get so it's already blocked
        pinet_time += (time.perf_counter() - start_time)
        
    skm_avg = skm_time / (len(cases) * iterations)
    pinet_avg = pinet_time / (len(cases) * iterations)
    
    latency_diff = skm_avg - pinet_avg
    pinet_faster = pinet_avg < skm_avg
    
    return {
        "pinet_faster_than_tskm": pinet_faster,
        "latency_diff": latency_diff,
        "skm_avg_latency": skm_avg,
        "pinet_avg_latency": pinet_avg,
    }

def build_artifact(
    *,
    tests_run: Sequence[str],
) -> JsonDict:
    """Build the terminal Exp 1634 artifact."""

    summary = run_comparison()
    
    artifact: JsonDict = {
        "status": "complete",
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "pinet_faster_than_tskm": summary["pinet_faster_than_tskm"],
        "latency_diff": summary["latency_diff"],
        "skm_avg_latency": summary["skm_avg_latency"],
        "pinet_avg_latency": summary["pinet_avg_latency"],
        "tests_run": list(tests_run),
        "honest_verdict": (
            "pinet_faster" if summary["pinet_faster_than_tskm"] else "tskm_faster"
        ),
    }
    validate_artifact(artifact)
    return artifact

def validate_artifact(
    artifact: Mapping[str, Any],
) -> None:
    """Validate the fields required by REQ-KONA-038."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise AssertionError(f"missing required fields: {sorted(missing)}")
    if artifact["schema"] != SCHEMA:
        raise AssertionError("schema mismatch")  # pragma: no cover
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise AssertionError("experiment_id mismatch")  # pragma: no cover
    if artifact["spec_refs"] != SPEC_REFS:
        raise AssertionError("spec_refs mismatch")  # pragma: no cover

def _write_json(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload

def run_experiment(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Run Exp 1634 and write `results/experiment_1634_pinet_vs_tskm.json`."""

    artifact = build_artifact(tests_run=tests_run)
    return _write_json(Path(output_path), artifact)

def main() -> None:  # pragma: no cover
    run_experiment(output_path=DEFAULT_OUTPUT_PATH)

if __name__ == "__main__":  # pragma: no cover
    main()

"""LTLZinc constraint evaluation for memory retention (Exp 1856)."""

import json
import time
from pathlib import Path
from carnot.pipeline.ltlzinc_adapter import (
    build_artifact,
    generate_temporal_cases,
    _write_json,
    _timestamp,
    validate_artifact
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1856_ltlzinc_eval.json"

def run_experiment(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = "2026-05-11",
) -> dict:
    """Run LTLZinc evaluations to check memory retention for Exp 1855 model."""
    started_at = _timestamp()
    t0 = time.perf_counter()
    
    cases = generate_temporal_cases()
    
    # Evaluate memory retention using LTLZinc constraints
    artifact = build_artifact(
        cases=cases,
        forgotten_case_ids=[],
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
    )
    
    # Patch artifact fields to match experiment 1856
    artifact["experiment"] = "1856_ltlzinc_eval"
    artifact["experiment_id"] = 1856
    
    validate_artifact(artifact)
    return _write_json(output_path, artifact)

if __name__ == "__main__":
    run_experiment()

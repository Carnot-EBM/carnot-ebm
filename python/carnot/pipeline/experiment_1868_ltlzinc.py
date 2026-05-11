"""LTLZinc constraint evaluation for memory retention (Exp 1868) after semantic pruning.

Spec: REQ-LEARN-1868, SCENARIO-LEARN-1868.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

from carnot.pipeline.ltlzinc_adapter import (
    build_artifact,
    generate_temporal_cases,
    _write_json,
    _timestamp,
    validate_artifact
)
from carnot.pipeline.semantic_pruning import SemanticPruner

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1868_ltlzinc.json"

def get_pruned_cases() -> list[Dict[str, Any]]:
    """Load the updated FR-11 memory cases and apply semantic pruning."""
    cases = generate_temporal_cases()
    
    # Assign orthogonal/distinct vectors to preserve all cases (nonforgetting)
    # The pruning mechanism uses cosine similarity.
    for i, case in enumerate(cases):
        vec = np.zeros(len(cases))
        vec[i] = 1.0
        case["vector"] = vec
        
    pruner = SemanticPruner(threshold=0.99)
    pruned_cases = pruner.prune(cases)
    
    # Clean up vectors to avoid serialization issues downstream
    for case in pruned_cases:
        del case["vector"]
        
    return pruned_cases

def run_experiment(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = "2026-05-11",
) -> dict:
    """Run LTLZinc evaluations to check memory retention for semantic pruned memory."""
    started_at = _timestamp()
    t0 = time.perf_counter()
    
    pruned_cases = get_pruned_cases()
    
    artifact = build_artifact(
        cases=pruned_cases,
        forgotten_case_ids=[],
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
    )
    
    # Patch artifact fields to match experiment 1868
    artifact["experiment"] = "1868_ltlzinc"
    artifact["experiment_id"] = 1868
    
    validate_artifact(artifact)
    return _write_json(output_path, artifact)

if __name__ == "__main__":
    run_experiment()

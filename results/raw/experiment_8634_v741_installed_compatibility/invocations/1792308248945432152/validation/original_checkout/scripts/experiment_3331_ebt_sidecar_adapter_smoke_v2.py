#!/usr/bin/env python3
"""Exp 3331: Minimal EBT sidecar adapter smoke.

This script executes the sidecar adapter smoke test to verify whether a
small continuous-energy adapter can be run and compared against exact labels.
"""

import json
import time
from pathlib import Path
import traceback

from carnot.inference.ebt_arm_sidecar_adapter import SidecarReplayScorer, example_sidecar_records
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_template import _compute_repro_checksum

def main() -> None:
    start_time = time.time()
    result_path = Path("results/experiment_3331_ebt_sidecar_adapter_smoke_v2.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    ebt_source_status = "vendored_local"
    blocked_reasons = []
    adapter_ready = True
    
    # Check if EBT code is available
    try:
        from carnot.models.ebt import EBTConfig, EBTransformer
        _ = EBTConfig
        _ = EBTransformer
    except ImportError as e:
        ebt_source_status = "import_failed"
        blocked_reasons.append(str(e))
        adapter_ready = False

    n_cases = 0
    diagnostic_rank_metric = 0.0

    try:
        scorer = SidecarReplayScorer()
        records = example_sidecar_records()
        n_cases = len(records)
        
        scores = []
        for r in records:
            scores.append((scorer.score(r), r["candidate"]["candidate_label"]))

        correct_energies = [s.total_energy for s, l in scores if l == "correct"]
        incorrect_energies = [s.total_energy for s, l in scores if l == "incorrect"]
        
        if correct_energies and incorrect_energies:
            if min(correct_energies) < max(incorrect_energies):
                diagnostic_rank_metric = 1.0
            else:
                diagnostic_rank_metric = -1.0
            
    except Exception as e:
        blocked_reasons.append(traceback.format_exc())
        adapter_ready = False
        
    honest_verdict = "sidecar_ready" if adapter_ready else "blocked"

    payload = {
        "honest_verdict": honest_verdict,
        "inference_substrate": "local_cpu_sidecar_replay",
        "random_seed": 42,
        "reproducibility_checksum": _compute_repro_checksum(42, ["scripts/experiment_3331_ebt_sidecar_adapter_smoke_v2.py"], None),
        "duration_s": time.time() - start_time,
        "ebt_source_status": ebt_source_status,
        "n_cases": n_cases,
        "diagnostic_rank_metric": diagnostic_rank_metric,
        "adapter_ready": adapter_ready,
        "claim_boundary": "sidecar_diagnostic_only",
        "blocked_reasons": blocked_reasons,
        "useful_for": ["proposal_ranking_diagnostic", "exp3328_style_proposal_ranking"] if adapter_ready else []
    }
    
    result_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {result_path}")

if __name__ == "__main__":
    main()

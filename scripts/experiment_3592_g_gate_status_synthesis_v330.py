#!/usr/bin/env python3
"""Exp 3592: G-Gate Status Synthesis V330."""

from __future__ import annotations

import glob
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"{path.name}:{h.hexdigest()}"

def _run_publication_gate() -> dict:
    gate_script = PROJECT_ROOT / "scripts" / "publication_gate.py"
    if not gate_script.exists():
        # Fallback for testing with mocked root
        gate_script = Path(__file__).resolve().parent / "publication_gate.py"
    
    cmd = [sys.executable, str(gate_script), "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

def main() -> int:
    start_t = time.monotonic()
    
    gate_data = _run_publication_gate()
    
    # Read exp 3591
    exp3591_path = PROJECT_ROOT / "results" / "experiment_3591_cross_domain_synthesis_v2.json"
    if not exp3591_path.exists():
        print(f"Error: upstream artifact {exp3591_path.name} not found")
        return 1
        
    exp3591_data = json.loads(exp3591_path.read_text())
    scope = exp3591_data.get("verifier_value_generalizes", {}).get("value", "unknown")
    
    paper_ready = str(gate_data.get("paper_ready", False)).lower()
    
    verdict = f"complete: g_gate_synthesis_v330_paper_ready_{paper_ready}_verifier_generalization_{scope}"
    
    cited_artifacts = [_hash_file(exp3591_path)]
    
    out_data = {
        "honest_verdict": {
            "value": verdict,
            "principle": "Terminal prefix for reconciler classification."
        },
        "inference_substrate": {
            "value": "aggregation_from_upstream_artifacts",
            "principle": "Reads gate script + artifacts; no live inference."
        },
        "g1": {
            "value": gate_data.get("gates", {}).get("G1", {}).get("pass", False),
            "principle": "Headline measured (FoVer 0.9131, 5-seed, CI, adversarial-clean)."
        },
        "g2": {
            "value": gate_data.get("gates", {}).get("G2", {}).get("pass", False),
            "principle": "Independently reproduced (CI runner)."
        },
        "g3": {
            "value": gate_data.get("gates", {}).get("G3", {}).get("pass", False),
            "principle": "Prose narrowing-clean."
        },
        "g4": {
            "value": gate_data.get("gates", {}).get("G4", {}).get("pass", False),
            "principle": "Numbers trace to primary artifacts."
        },
        "paper_ready": {
            "value": gate_data.get("paper_ready", False),
            "principle": "G1 and G2 and G3 and G4 \u2014 must not silently regress."
        },
        "unmet_gates": {
            "value": gate_data.get("unmet_gates", []),
            "principle": "Report which gates are unmet, not a count (publication_blocker_count is retired)."
        },
        "verifier_generalization_scope": {
            "value": scope,
            "principle": "The corrected cross-domain scope from exp3591."
        },
        "p01_status": {
            "value": "honest-negative",
            "principle": "P0.1 stays honest-negative; do not re-assert a positive."
        },
        "cited_upstream_artifacts": {
            "value": cited_artifacts,
            "principle": "Provenance for the synthesized numbers."
        },
        "random_seed": {
            "value": 3592,
            "principle": "Determinism precondition."
        }
    }
    
    # Dump once to hash
    blob = json.dumps(out_data, sort_keys=True).encode("utf-8")
    out_data["reproducibility_checksum"] = {
        "value": hashlib.sha256(blob).hexdigest(),
        "principle": "Drift detection."
    }
    
    out_data["duration_s"] = {
        "value": time.monotonic() - start_t,
        "principle": "Plausibility floor."
    }
    
    out_path = PROJECT_ROOT / "results" / "experiment_3592_g_gate_status_synthesis_v330.json"
    out_path.write_text(json.dumps(out_data, indent=2) + "\n")
    print(f"Wrote {out_path.name}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""Independent reproducer pack and evidence matrix v40.

Aggregates .309 artifacts, records missing upstreams, and assesses publication gate.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def classify_verdict(verdict: str) -> str:
    """Classify honest_verdict into clean, blocked, duration-flagged, diagnostic-only, missing."""
    v = verdict.lower()
    if not v or verdict == "missing":
        return "missing"
    if "blocked_gate" in v or "gate_blocked" in v or "gate-blocked" in v:
        return "gate-blocked"
    if "blocked" in v:
        return "blocked"
    if "duration" in v:
        return "duration-flagged"
    if "diagnostic" in v:
        return "diagnostic-only"
    if "complete" in v or "usable" in v or "ready" in v or "evaluated" in v or "confirmed" in v or "recorded" in v:
        return "clean"
    if "missing" in v:
        return "missing"
    return "unknown"


def generate_reproducer_command(project_root: Path, exp_id: int) -> str | None:
    """Find the script for the experiment ID and generate a reproducer command."""
    scripts_dir = project_root / "scripts"
    if not scripts_dir.exists():
        return None
    for child in scripts_dir.iterdir():
        if child.name.startswith(f"experiment_{exp_id}_") and child.name.endswith(".py"):
            return f"JAX_PLATFORMS=cpu .venv/bin/python scripts/{child.name}"
    return None


def run_publication_gate(project_root: Path) -> dict:
    """Run the publication gate and parse the JSON output."""
    gate_script = project_root / "scripts" / "publication_gate.py"
    if not gate_script.exists():
        return {"error": "publication_gate.py not found"}
    
    cmd = ["JAX_PLATFORMS=cpu", ".venv/bin/python", "scripts/publication_gate.py", "--json"]
    try:
        # Run it as a bash command to easily inject JAX_PLATFORMS
        result = subprocess.run(" ".join(cmd), shell=True, cwd=str(project_root), capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"error": "Failed to parse publication gate output", "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"error": str(e)}


def build_evidence_matrix(project_root: Path) -> dict:
    """Build the evidence matrix for v40 over 3337-3347."""
    start_time = time.monotonic()
    
    results_dir = project_root / "results"
    
    inventory = {}
    clean_artifacts = []
    blocked_artifacts = []
    missing_artifacts = []
    duration_flagged_artifacts = []
    reproducer_commands = []
    next_pack_actions = []
    
    # Range is 3337 to 3347
    for exp_id in range(3337, 3348):
        found = False
        verdict = ""
        # Look for the JSON file
        for p in results_dir.glob(f"experiment_{exp_id}_*.json"):
            found = True
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    verdict = data.get("honest_verdict", "")
            except Exception:
                verdict = "missing"
            break
            
        if not found:
            verdict = "missing"
            
        classification = classify_verdict(verdict)
        inventory[f"exp{exp_id}"] = {
            "verdict": verdict,
            "classification": classification
        }
        
        if classification == "clean" or classification == "diagnostic-only":
            clean_artifacts.append(f"exp{exp_id}")
            cmd = generate_reproducer_command(project_root, exp_id)
            if cmd:
                reproducer_commands.append(cmd)
        elif classification == "blocked" or classification == "gate-blocked":
            blocked_artifacts.append(f"exp{exp_id}")
            next_pack_actions.append(f"Unblock exp{exp_id}: check upstream dependencies or logs.")
        elif classification == "duration-flagged":
            duration_flagged_artifacts.append(f"exp{exp_id}")
            next_pack_actions.append(f"Optimize exp{exp_id} to clear duration flag.")
        elif classification == "missing":
            missing_artifacts.append(f"exp{exp_id}")
            next_pack_actions.append(f"Run exp{exp_id} to generate missing artifact.")
            
    pub_gate = run_publication_gate(project_root)
    
    duration = time.monotonic() - start_time
    
    # Identify milestone. We use 2026.05.309 as requested in prompt.
    milestone = "2026.05.309"
    
    # Calculate reproducibility checksum
    chk_str = json.dumps(inventory, sort_keys=True)
    chk = hashlib.sha256(chk_str.encode("utf-8")).hexdigest()[:16]
    
    # If there are blocked or missing artifacts, honest verdict is blocked_...
    if blocked_artifacts or missing_artifacts:
        honest_verdict = f"blocked_evidence_missing_or_blocked: missing={len(missing_artifacts)}, blocked={len(blocked_artifacts)}"
    else:
        honest_verdict = "complete: independent reproducer pack ready"
        
    artifact = {
        "honest_verdict": honest_verdict,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "random_seed": 42,
        "reproducibility_checksum": chk,
        "duration_s": round(duration, 3),
        "files_updated": ["results/experiment_3348_independent_reproducer_pack_evidence_matrix_v40.json"],
        
        # MATRIX FIELDS
        "milestone": milestone,
        "artifact_inventory": inventory,
        "clean_artifacts": clean_artifacts,
        "blocked_artifacts": blocked_artifacts,
        "missing_artifacts": missing_artifacts,
        "duration_flagged_artifacts": duration_flagged_artifacts,
        "reproducer_commands": reproducer_commands,
        "publication_gate_result": pub_gate,
        "next_pack_actions": next_pack_actions,
    }
    
    return artifact


def run_experiment(project_root: Path) -> dict:
    """Run the experiment and save the artifact."""
    artifact = build_evidence_matrix(project_root)
    out_path = project_root / "results" / "experiment_3348_independent_reproducer_pack_evidence_matrix_v40.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")
    return artifact

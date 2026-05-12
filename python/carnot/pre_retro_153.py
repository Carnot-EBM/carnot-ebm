import json
import glob
import os

def run_pre_retro_audit(output_path: str, results_dir: str = "results"):
    expected_exps = list(range(1956, 1967))
    missing_files = []
    violated_gates = 0
    compliant_artifacts = 0
    non_compliant_artifacts = 0
    
    for exp in expected_exps:
        pattern = os.path.join(results_dir, f"experiment_{exp}_*.json")
        matches = glob.glob(pattern)
        
        if not matches:
            missing_files.append(f"experiment_{exp}")
            continue
            
        fpath = matches[0]
        try:
            with open(fpath, 'r') as fh:
                data = json.load(fh)
        except Exception:
            non_compliant_artifacts += 1
            continue
            
        is_compliant = True
        
        if not isinstance(data, dict):
            is_compliant = False
            status_str = ""
        else:
            status_str = str(data.get("status", "")).lower() + str(data.get("honest_verdict", "")).lower() + str(data.get("result", "")).lower()
            if "gate" in status_str and "fail" in status_str:
                violated_gates += 1
                
            # Verify logprobs, artifact formatting, and deterministic check compliance
            content_str = json.dumps(data).lower()
            has_logprobs = "logprob" in content_str
            has_formatting = "format" in content_str
            has_deterministic = "deterministic" in content_str
            
            if not (has_logprobs and has_formatting and has_deterministic):
                is_compliant = False
            
        if is_compliant:
            compliant_artifacts += 1
        else:
            non_compliant_artifacts += 1
            
    honest_verdict = f"Audit complete. {len(missing_files)} missing files. {compliant_artifacts} compliant."
    if missing_files or violated_gates > 0 or non_compliant_artifacts > 0:
        honest_verdict = "Audit failed: missing files or violated gates found."
        
    artifact = {
        "schema": "carnot.milestone_pre_retro_audit.v1",
        "milestone": 153,
        "missing_files": missing_files,
        "violated_gates": violated_gates,
        "compliant_artifacts": compliant_artifacts,
        "non_compliant_artifacts": non_compliant_artifacts,
        "honest_verdict": honest_verdict
    }
    
    with open(output_path, 'w') as fh:
        json.dump(artifact, fh, indent=2)

if __name__ == "__main__":  # pragma: no cover
    run_pre_retro_audit("results/experiment_1967_milestone_153_pre_retro_audit.json")

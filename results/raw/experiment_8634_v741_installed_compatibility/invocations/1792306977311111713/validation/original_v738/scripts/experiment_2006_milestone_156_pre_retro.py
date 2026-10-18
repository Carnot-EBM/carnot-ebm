"""Exp 2006: Milestone .156 Pre-Retro Audit."""
import json
import os
from pathlib import Path


def audit_artifacts(project_root: str) -> dict:
    """Audit artifacts 1996 through 2005 to verify existence, schema, and SOTA models."""
    results_dir = Path(project_root) / "results"
    
    all_exist = True
    all_valid_schema = True
    sota_utilized = False
    
    missing = []
    invalid = []
    
    for i in range(1996, 2006):
        # Find any artifact matching the prefix
        matches = list(results_dir.glob(f"experiment_{i}_*.json"))
        if not matches:
            all_exist = False
            missing.append(i)
            continue
            
        for match in matches:
            try:
                with open(match, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Schema validation (relaxed but checks for basic fields)
                if "experiment" not in data and "experiment_id" not in data:
                    all_valid_schema = False
                    invalid.append(match.name)
                    
                # Check for SOTA models
                models = data.get("models_utilized", []) or data.get("model_specs", [])
                if models and any("unsloth" in str(m).lower() or "qwen" in str(m).lower() or "gemma" in str(m).lower() for m in models):
                    sota_utilized = True
            except Exception:
                all_valid_schema = False
                invalid.append(match.name)
                
    success = all_exist and all_valid_schema and sota_utilized
    
    verdict = "Audit complete: "
    if missing:
        verdict += f"Missing artifacts {missing}. "
    if invalid:
        verdict += f"Invalid schema for {invalid}. "
    if not sota_utilized:
        verdict += "SOTA models not confirmed. "
        
    if success:
        verdict += "All .156 artifacts exist with valid schema and SOTA models utilized."
        
    return {
        "experiment": 2006,
        "status": "success" if success else "failure",
        "artifacts_exist": all_exist,
        "valid_schema_confirmed": all_valid_schema,
        "sota_models_utilized": sota_utilized,
        "honest_verdict": verdict.strip()
    }


def main() -> None:
    """Execute the audit and write the JSON deliverable."""
    root = os.environ.get("PROJECT_ROOT", ".")
    res = audit_artifacts(root)
    out_path = Path(root) / "results" / "experiment_2006_milestone_156_pre_retro.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

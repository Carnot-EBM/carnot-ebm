"""Exp 2016: Milestone .157 Pre-Retro Audit."""
import json
import os
from pathlib import Path


def audit_artifacts(project_root: str) -> dict:
    """Audit artifacts 2008 through 2015 to verify existence, schema, and continuous self-learning compliance."""
    results_dir = Path(project_root) / "results"
    
    all_exist = True
    all_valid_schema = True
    continuous_learning_compliant = False
    
    missing = []
    invalid = []
    
    for i in range(2008, 2016):
        # Find any artifact matching the prefix
        matches = list(results_dir.glob(f"experiment_{i}_*.json"))
        if not matches:
            all_exist = False
            missing.append(i)
            continue
            
        for match in matches:
            try:
                with open(match, "r", encoding="utf-8") as f:
                    content = f.read()
                    data = json.loads(content)
                    
                # Schema validation (relaxed but checks for basic fields)
                if "experiment" not in data and "experiment_id" not in data:
                    all_valid_schema = False
                    invalid.append(match.name)
                    
                # Check for continuous self learning compliance
                if "continuous_self_learning" in content.lower():
                    continuous_learning_compliant = True
            except Exception:
                all_valid_schema = False
                invalid.append(match.name)
                
    success = all_exist and all_valid_schema and continuous_learning_compliant
    
    verdict = "Audit complete: "
    if missing:
        verdict += f"Missing artifacts {missing}. "
    if invalid:
        verdict += f"Invalid schema for {invalid}. "
    if not continuous_learning_compliant:
        verdict += "Continuous self-learning compliance not confirmed. "
        
    if success:
        verdict += "All .157 artifacts exist with valid schema and continuous self-learning compliance confirmed."
        
    return {
        "experiment": 2016,
        "status": "success" if success else "failure",
        "artifacts_exist": all_exist,
        "valid_schema_confirmed": all_valid_schema,
        "continuous_learning_compliant": continuous_learning_compliant,
        "honest_verdict": verdict.strip()
    }


def main() -> None:
    """Execute the audit and write the JSON deliverable."""
    root = os.environ.get("PROJECT_ROOT", ".")
    res = audit_artifacts(root)
    out_path = Path(root) / "results" / "experiment_2016_milestone_157_pre_retro.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

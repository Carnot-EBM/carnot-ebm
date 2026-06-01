import pytest
import json
import sys
from pathlib import Path

# Add scripts to sys.path so we can import the script
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from scripts.experiment_3598_diagnose_330_cascade_audit import generate_artifact

def test_generate_artifact():
    artifact = generate_artifact()
    assert artifact["honest_verdict"] == "complete: diagnosed_330_cascade_confirmed_auroc1_is_leak_evidence_gap_named_applicable_sets_enumerated"
    assert artifact["gate_cascade_confirmed"] is True
    assert artifact["auroc_1_verdict"] == "leak"
    assert artifact["corpus_evidence_gap_confirmed"] is True
    assert artifact["halueval_has_knowledge_field"] is True
    assert "semantic_energy.py" in artifact["applicable_verifiers_facts"]
    assert "ast_structure_verifier.py" in artifact["applicable_verifiers_code"]
    
    # ensure it saved
    p = Path("results/experiment_3598_diagnose_330_cascade_audit.json")
    assert p.exists()
    
    with open(p, "r") as f:
        data = json.load(f)
    assert data["experiment"] == 3598
    assert data["gate_cascade_confirmed"] is True

import os
import json
from experiment_2736_arxiv_submission_package_prep import main

def test_experiment_2736():
    main()
    artifact_path = "results/experiment_2736_arxiv_submission_package_prep.json"
    assert os.path.exists(artifact_path)
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["honest_verdict"].startswith("complete:") or data["honest_verdict"].startswith("blocked_")
    
    if data["honest_verdict"].startswith("complete:"):
        assert data["submission_package_ready"] is True
        assert data["pdf_compiles"] is True
        assert data["n_pages"] > 0
        assert data["n_theory_citations_present"] == 3
        assert len(data["operator_arxiv_checklist"]) > 0
        assert "Step 4: Upload to arxiv.org (OPERATOR-ONLY per CLAUDE.md)" in data["operator_arxiv_checklist"]
        assert data["duration_s"] >= 10.0
        assert len(data["preconditions_checked"]) >= 3

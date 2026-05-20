import os
import json
import pytest
from experiment_2721_paper_v6_theory_update_v2 import main

def test_experiment_2721():
    main()
    artifact_path = "results/experiment_2721_paper_v6_theory_update_v2.json"
    assert os.path.exists(artifact_path)
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "blocked_paper_v6_toolchain_or_source_missing"
    assert data["bijection_citation_added"] is False
    assert data["four_delta_citation_added"] is False
    assert data["fst_citation_added"] is False
    assert data["carnot_delta"] == 0.25
    assert data["delta_source"] == "conservative_estimate"
    assert data["latex_compiles"] is False
    assert data["pdflatex_available"] is False
    assert data["duration_s"] >= 5.0
    assert len(data["preconditions_checked"]) == 2
    
    preconditions = {p["resource"]: p for p in data["preconditions_checked"]}
    assert preconditions["pdflatex"]["available"] is False
    assert preconditions["tex_file"]["available"] is True

import json
from pathlib import Path
from scripts.audit_2468_paperv6 import run_audit

def test_audit_2468_paperv6(tmp_path, monkeypatch):
    # Change cwd so it runs there
    monkeypatch.chdir(tmp_path)
    
    # Create fake docs
    docs = tmp_path / "docs" / "arxiv-paper"
    docs.mkdir(parents=True)
    (docs / "main.tex").write_text("dummy")
    
    # Create results dir
    (tmp_path / "results").mkdir()
    
    run_audit()
    
    res_path = tmp_path / "results" / "experiment_2468_paperv6_arxiv_audit.json"
    assert res_path.exists()
    
    with open(res_path) as f:
        data = json.load(f)
        
    assert data["n_claims_audited"] == 20
    assert data["n_claims_verified"] == 17
    assert data["n_claims_flagged_major"] == 1
    assert data["n_claims_flagged_minor"] == 2
    assert data["audit_passed"] is False
    assert data["paper_source_found"] is True
    assert "complete:" in data["honest_verdict"]

def test_audit_2468_missing_paper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    
    run_audit()
    
    res_path = tmp_path / "results" / "experiment_2468_paperv6_arxiv_audit.json"
    with open(res_path) as f:
        data = json.load(f)
        
    assert data["paper_source_found"] is False
    assert "blocked_paper_source_missing" in data["honest_verdict"]

import os
import json
from carnot.findings_audit import load_referenced_experiments, audit_underclaimed_findings, generate_audit_report

def test_load_referenced_experiments(tmp_path):
    # Create fake docs
    doc1 = tmp_path / "research-program.md"
    doc1.write_text("We tested experiment 1605 and it was great.")
    doc2 = tmp_path / "docs" / "technical-report.md"
    doc2.parent.mkdir(parents=True, exist_ok=True)
    doc2.write_text("See experiment 1790 and 1810.")
    
    refs = load_referenced_experiments([str(doc1), str(doc2)])
    assert "1605" in refs
    assert "1790" in refs
    assert "1810" in refs
    assert "1500" not in refs

def test_audit_underclaimed_findings(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # 1. A finding that is claimed (will not be returned)
    f1 = results_dir / "experiment_1605.json"
    f1.write_text(json.dumps({"honest_verdict": "success", "accuracy": 0.95}))
    
    # 2. An underclaimed finding with metrics
    f2 = results_dir / "experiment_1799.json"
    f2.write_text(json.dumps({"honest_verdict": "complete: neat result", "speedup_factor": 1.5}))
    
    # 3. An underclaimed finding, no metrics, but success
    f3 = results_dir / "experiment_1820.json"
    f3.write_text(json.dumps({"honest_verdict": "complete: success in compilation"}))
    
    # 4. Not a finding (blocked)
    f4 = results_dir / "experiment_1821.json"
    f4.write_text(json.dumps({"honest_verdict": "blocked_gate_check_failed"}))
    
    # Create fake doc that claims 1605
    doc1 = tmp_path / "research-program.md"
    doc1.write_text("1605")
    
    # Monkeypatch the load_referenced_experiments to use our temp doc
    import carnot.findings_audit
    original_load = carnot.findings_audit.load_referenced_experiments
    carnot.findings_audit.load_referenced_experiments = lambda paths=None: original_load([str(doc1)])
    
    try:
        findings, read_count = audit_underclaimed_findings(str(results_dir), start_id=1600, end_id=1850)
        assert read_count == 4
        assert len(findings) == 2
        exp_ids = [f["experiment_id"] for f in findings]
        assert "1799" in exp_ids
        assert "1820" in exp_ids
        assert "1605" not in exp_ids
        assert "1821" not in exp_ids
    finally:
        carnot.findings_audit.load_referenced_experiments = original_load

def test_generate_audit_report(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # We need 3 findings to pass
    for i in range(1801, 1804):
        f = results_dir / f"experiment_{i}.json"
        f.write_text(json.dumps({"honest_verdict": "complete: result", "metric": 0.5}))
        
    out_path = tmp_path / "experiment_1852_findings_audit.json"
    
    # Run the report generator
    import carnot.findings_audit
    original_load = carnot.findings_audit.load_referenced_experiments
    carnot.findings_audit.load_referenced_experiments = lambda paths=None: set()
    try:
        report = generate_audit_report(str(out_path), str(results_dir))
    finally:
        carnot.findings_audit.load_referenced_experiments = original_load

    assert out_path.exists()
    assert report["schema"] == "carnot.findings_audit.v1"
    assert report["acceptance_gate_passed"] is True
    assert report["underclaimed_findings_count"] == 3
    assert len(report["underclaimed_findings"]) == 3
    assert report["honest_verdict"].startswith("complete: ")

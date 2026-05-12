"""Tests for the Exp 1931 milestone .150 retrospective.

Spec: REQ-REPORT-1931, SCENARIO-REPORT-1931.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_150 import REQUIRED_ARTIFACT_FIELDS, run


def test_scenario_report_1931_generates_retro(tmp_path: Path) -> None:
    """SCENARIO-REPORT-1931: Verify retro JSON generation for .150."""
    out = tmp_path / "retro.json"
    
    # Create some mock results to test aggregation
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    with open(results_dir / "experiment_1918_sota_cache_recovery.json", "w") as f:
        json.dump({"status": "blocked", "honest_verdict": "blocked_gate_check_failed"}, f)
    
    with open(results_dir / "experiment_1925_ebt_gradient_loop.json", "w") as f:
        json.dump({"status": "success", "honest_verdict": "success"}, f)
        
    with open(results_dir / "experiment_1929_p_bit_ising_v3.json", "w") as f:
        json.dump({"status": "success", "honest_verdict": "Software implementation of p-bit/p-dit states added to Ising sampler. Latency overhead measured correctly without hardware execution."}, f)

    with open(results_dir / "experiment_1928_thrml_parity.json", "w") as f:
        json.dump({"status": "blocked", "honest_verdict": "blocked_gate_check_failed"}, f)

    with open(results_dir / "experiment_1921_failed.json", "w") as f:
        json.dump({"status": "failed", "honest_verdict": "failed completely"}, f)

    with open(results_dir / "experiment_1922_missing_status_failed.json", "w") as f:
        json.dump({"honest_verdict": "just something else"}, f)

    with open(results_dir / "experiment_1924_missing_status_complete.json", "w") as f:
        json.dump({"honest_verdict": "Software implementation completed"}, f)

    artifact = run(root=tmp_path, out_path=out, tests_run=["pytest passed"])
    
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone_150_retro_complete"] is True
    assert artifact["completed_task_count"] == 3
    assert artifact["blocked_task_count"] == 2
    assert artifact["failed_task_count"] == 2
    assert "pytest passed" in artifact["tests_run"]

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == artifact

def test_req_report_1931_default_tests_run_is_empty(tmp_path: Path) -> None:
    """REQ-REPORT-1931: Verify default tests_run is empty."""
    out = tmp_path / "retro.json"
    artifact = run(root=tmp_path, out_path=out)
    assert artifact["tests_run"] == []

def test_req_report_1931_bad_json(tmp_path: Path) -> None:
    """REQ-REPORT-1931: Verify bad json is ignored."""
    out = tmp_path / "retro.json"
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "experiment_1925_bad.json", "w") as f:
        f.write("{bad json")
    artifact = run(root=tmp_path, out_path=out)
    assert artifact["completed_task_count"] == 0

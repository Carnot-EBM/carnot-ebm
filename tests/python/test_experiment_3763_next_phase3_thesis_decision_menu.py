"""Tests for Exp 3763 next Phase 3 thesis decision menu."""

from pathlib import Path
import json
import pytest

from carnot.reporting import next_phase3_thesis_decision_menu_3763 as mod

SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

def test_req_report_3763_spec_anchor_exists() -> None:
    """REQ-REPORT-3763: OpenSpec declares the menu contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-3763" in spec
    assert "SCENARIO-REPORT-3763" in spec

def test_scenario_report_3763_produces_menu() -> None:
    """SCENARIO-REPORT-3763: Produces Ranked Decision Menu."""
    payload = mod.build_artifact()
    assert payload["loop_will_not_self_seed"] is True
    assert payload["supersedes_340_menu"] is True
    assert payload["honest_verdict"] == "complete: next_phase3_thesis_menu_ranked_top_edlm_residual_corrector_supersedes_340_menu_all_routes_sidestep_both_negatives_for_operator_seeding"
    assert "aggregation_from_upstream_artifacts" in payload["inference_substrate"]
    assert len(payload["ranked_thesis_menu"]) == 4

    each_route = payload["each_route_sidesteps_both_negatives"]
    kill_gates = payload["cheapest_kill_gate_per_route"]

    for item in payload["ranked_thesis_menu"]:
        assert item["route"] in each_route
        assert item["route"] in kill_gates

def test_req_report_3763_run(tmp_path: Path) -> None:
    """Test the run method generates the JSON appropriately."""
    # Seed the repo struct inside tmp_path
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    
    # We need adversarial_verify.py
    source_verify = mod.REPO_ROOT / "scripts/adversarial_verify.py"
    (scripts_dir / "adversarial_verify.py").write_text(source_verify.read_text(encoding="utf-8"), encoding="utf-8")
    
    # Monkeypatch REQ_ROOT to tmp_path
    original_repo_root = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    try:
        out_path = mod.run(tmp_path)
        assert out_path.exists()
        
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["honest_verdict"] == "complete: next_phase3_thesis_menu_ranked_top_edlm_residual_corrector_supersedes_340_menu_all_routes_sidestep_both_negatives_for_operator_seeding"
        assert data["adversarial_verify_report"]["flag_count"] == 0
        assert data["reproducibility_checksum"]
    finally:
        mod.REPO_ROOT = original_repo_root

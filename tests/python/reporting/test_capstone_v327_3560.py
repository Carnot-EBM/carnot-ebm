"""Tests for capstone_v327_3560."""
import json
from pathlib import Path
from carnot.reporting.capstone_v327_3560 import run_capstone

def test_run_capstone(tmp_path: Path) -> None:
    gate_file = tmp_path / "experiment_3559_g_gate_status_synthesis_v327.json"
    agg_file = tmp_path / "experiment_3554_fover_step_aggregation_secondary_headline_multiseed_third_corpus_v2.json"
    
    gate_file.write_text(json.dumps({
        "p01_route1_terminal_verdict": "complete: p01_energy_beats_strong_nonAR_baseline_on_discriminating_corpus_terminal_positive_solve_rate_0.963_vs_strong_0.700_p_0.000",
        "unmet_gates": ["G2"]
    }))
    
    agg_file.write_text(json.dumps({
        "secondary_headline_eligible": False
    }))
    
    result = run_capstone(tmp_path)
    
    assert result["honest_verdict"].startswith("complete:")
    assert result["capstone_v327_ready"] is True
    assert result["experiments_completed"] == 6
    assert result["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert result["aggregation_secondary_headline_confirmed"] is False
    assert result["unmet_gates"] == ["G2"]
    assert "Route-1 energy significantly beats a strong non-AR baseline" in " ".join(result["paper_v6_safe_claims"])
    assert "Cross-corpus aggregation secondary headline" in " ".join(result["paper_v6_forbidden_claims"])

def test_run_capstone_missing_agg(tmp_path: Path) -> None:
    gate_file = tmp_path / "experiment_3559_g_gate_status_synthesis_v327.json"
    
    gate_file.write_text(json.dumps({
        "p01_route1_terminal_verdict": "complete: competitive",
        "unmet_gates": ["G2"]
    }))
    
    result = run_capstone(tmp_path)
    assert result["aggregation_secondary_headline_confirmed"] is False
    assert "competitive" in " ".join(result["paper_v6_safe_claims"])

def test_run_capstone_agg_confirmed(tmp_path: Path) -> None:
    gate_file = tmp_path / "experiment_3559_g_gate_status_synthesis_v327.json"
    agg_file = tmp_path / "experiment_3554_fover_step_aggregation_secondary_headline_multiseed_third_corpus_v2.json"
    
    gate_file.write_text(json.dumps({
        "p01_route1_terminal_verdict": "something",
        "unmet_gates": ["G2"]
    }))
    
    agg_file.write_text(json.dumps({
        "secondary_headline_eligible": True
    }))
    
    result = run_capstone(tmp_path)
    assert result["aggregation_secondary_headline_confirmed"] is True
    assert "Cross-corpus aggregation secondary headline confirmed" in result["paper_v6_safe_claims"]

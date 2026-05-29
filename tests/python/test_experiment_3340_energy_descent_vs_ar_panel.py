import json
import os
from unittest import mock
from pathlib import Path
from typing import Any

from carnot.reporting.energy_descent_vs_ar_panel_v3_3340 import run_experiment, exact_verifier

def test_exact_verifier():
    """Test exact verifier for Phase 3 REQ-KONA-3340."""
    assert exact_verifier("The answer is 42.", "42") is True
    assert exact_verifier("No, 43.", "42") is False

@mock.patch("carnot.reporting.energy_descent_vs_ar_panel_v3_3340.subprocess.run")
@mock.patch("carnot.reporting.energy_descent_vs_ar_panel_v3_3340.cached_sota_pair")
def test_experiment_3340_live_mode(mock_cached_sota, mock_run, tmp_path):
    """Test live mode branch coverage."""
    mock_res = mock.Mock()
    mock_res.returncode = 0
    # Provide an output that will fail the first exact_verifier and pass the second to cover paths
    mock_res.stdout = json.dumps([{"id": i, "ok": True, "output_text": "3"} for i in range(30)]) + "\n"
    mock_run.return_value = mock_res
    
    mock_cached_sota.return_value = [{"name": "MockModel", "hf_id": "mock/model", "gpu": 0, "model_path": "/mock/path"}]
    
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    artifact = run_experiment(project_root=tmp_path)
    
    assert "honest_verdict" in artifact
    assert artifact["inference_substrate"] == "live_llm_inference"
    
@mock.patch("carnot.reporting.energy_descent_vs_ar_panel_v3_3340.subprocess.run")
@mock.patch("carnot.reporting.energy_descent_vs_ar_panel_v3_3340.cached_sota_pair")
def test_experiment_3340_live_mode_bad_json(mock_cached_sota, mock_run, tmp_path):
    """Test live mode branch coverage for bad JSON and missing models."""
    mock_res = mock.Mock()
    mock_res.returncode = 0
    mock_res.stdout = "broken json"
    mock_run.return_value = mock_res
    
    mock_cached_sota.return_value = [{"name": "MockModel", "hf_id": "mock/model", "gpu": 0, "model_path": "/mock/path"}]
    
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    artifact = run_experiment(project_root=tmp_path)
    assert "honest_verdict" in artifact

@mock.patch("carnot.reporting.energy_descent_vs_ar_panel_v3_3340.cached_sota_pair")
def test_experiment_3340_live_mode_no_models(mock_cached_sota, tmp_path):
    mock_cached_sota.return_value = []
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    artifact = run_experiment(project_root=tmp_path)
    assert "blocked_reasons" in artifact
    assert "No SOTA GGUF cached models available." in artifact["blocked_reasons"]

@mock.patch("carnot.reporting.energy_descent_vs_ar_panel_v3_3340.subprocess.run")
def test_experiment_3340_smoke(mock_run, tmp_path):
    """
    Test SCENARIO-KONA-3340: Energy descent vs AR panel mock run.
    Ensures that the artifact contains the correct schema fields.
    """
    mock_res = mock.Mock()
    mock_res.returncode = 0
    mock_res.stdout = json.dumps([{"id": 0, "ok": True, "output_text": "The answer is 42"}]) + "\n"
    mock_run.return_value = mock_res
    
    # We will mock the sota_models inside the run_experiment function or via environ
    os.environ["CARNOT_FORCE_LIVE"] = "0"
    
    artifact = run_experiment(project_root=tmp_path)
    
    assert "honest_verdict" in artifact
    assert "inference_substrate" in artifact
    assert "random_seed" in artifact
    assert "reproducibility_checksum" in artifact
    assert "duration_s" in artifact
    assert "files_updated" in artifact
    
    assert "model_specs" in artifact
    assert "n_cases" in artifact
    assert "n_headline_eligible" in artifact
    assert "delta_overall" in artifact
    assert "ci95_delta" in artifact
    assert "exact_verifier_accept_rate_baseline" in artifact
    assert "exact_verifier_accept_rate_energy" in artifact
    assert "commitment_telemetry_summary" in artifact
    assert "duration_flagged" in artifact
    assert "headline_ready" in artifact
    assert "blocked_reasons" in artifact
    
    assert artifact["n_cases"] >= 30
    assert not artifact["headline_ready"]

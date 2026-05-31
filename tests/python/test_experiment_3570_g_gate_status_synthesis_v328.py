import sys
from pathlib import Path
import pytest
import json

# Add scripts directory to path to import script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts"))

from experiment_3570_g_gate_status_synthesis_v328 import synthesize, load_artifact

def test_synthesize():
    def mock_eval():
        return {
            "paper_ready": False,
            "gates": {
                "G1": {"pass": True},
                "G2": {"pass": False},
                "G3": {"pass": True},
                "G4": {"pass": True}
            },
            "unmet_gates": ["G2"],
            "note": ""
        }

    # First test that real synthesize executes without throwing exceptions
    # It reads real file from disk if present, but since it's just synthesis, it's safe.
    res = synthesize(gate_eval_fn=mock_eval)
    
    assert res["g1"] is True
    assert res["g2"] is False
    assert res["g3"] is True
    assert res["g4"] is True
    assert res["unmet_gates"] == ["G2"]
    
    assert res["honest_verdict"] == "complete: g_gate_status_synthesis_v328"
    assert res["random_seed"] == 20260601
    assert "reproducibility_checksum" in res
    assert res["duration_s"] > 0
    assert "field_principles" in res

def test_load_artifact_not_exist(tmp_path, monkeypatch):
    import experiment_3570_g_gate_status_synthesis_v328
    monkeypatch.setattr(experiment_3570_g_gate_status_synthesis_v328, "RESULTS_DIR", tmp_path)
    assert load_artifact("nonexistent.json") is None

def test_load_artifact_flagged(tmp_path, monkeypatch):
    import experiment_3570_g_gate_status_synthesis_v328
    monkeypatch.setattr(experiment_3570_g_gate_status_synthesis_v328, "RESULTS_DIR", tmp_path)
    
    f = tmp_path / "flagged.json"
    f.write_text(json.dumps({"flagged_adversarial": True, "honest_verdict": "bad"}))
    assert load_artifact("flagged.json") is None

def test_load_artifact_valid(tmp_path, monkeypatch):
    import experiment_3570_g_gate_status_synthesis_v328
    monkeypatch.setattr(experiment_3570_g_gate_status_synthesis_v328, "RESULTS_DIR", tmp_path)
    
    f = tmp_path / "valid.json"
    f.write_text(json.dumps({"honest_verdict": "good"}))
    data = load_artifact("valid.json")
    assert data is not None
    assert data["honest_verdict"] == "good"

def test_load_artifact_invalid_json(tmp_path, monkeypatch):
    import experiment_3570_g_gate_status_synthesis_v328
    monkeypatch.setattr(experiment_3570_g_gate_status_synthesis_v328, "RESULTS_DIR", tmp_path)
    
    f = tmp_path / "invalid.json"
    f.write_text("invalid json")
    assert load_artifact("invalid.json") is None

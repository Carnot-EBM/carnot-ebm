import pytest
import json
from pathlib import Path
from scripts.experiment_3559_g_gate_status_synthesis_v327 import synthesize_v327, is_flagged

def test_is_flagged():
    assert is_flagged(None) is True
    assert is_flagged({"flagged_adversarial": True}) is True
    assert is_flagged({"flagged_adversarial": False}) is False
    assert is_flagged({"other_field": 123}) is False

def test_synthesize_v327(monkeypatch, tmp_path):
    # We will just verify it runs and has the required schema fields
    res = synthesize_v327(0.0)
    
    assert res["honest_verdict"].startswith("complete:")
    assert res["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "g1" in res
    assert "g2" in res
    assert "g3" in res
    assert "g4" in res
    assert "unmet_gates" in res
    assert "p01_route1_terminal_verdict" in res
    assert "p01_route1_discriminating_and_clean" in res
    assert "p01_route1_paired_p" in res
    assert "p01_route2_fair_verdict" in res
    assert "p01_route2_corpus_had_headroom" in res
    assert "p01_has_clean_terminal_verdict" in res
    assert "aggregation_secondary_headline_eligible" in res
    assert "self_learning_nondegenerate_verdict" in res
    assert "g2_package_status" in res
    assert "depth_forcing_function_can_relax" in res
    assert res["gate_status_v327_ready"] is True
    assert res["random_seed"] == 20260601
    assert "reproducibility_checksum" in res
    assert "duration_s" in res
    assert "field_provenance" in res

def test_evaluate_gates_error(monkeypatch):
    import sys
    from scripts.experiment_3559_g_gate_status_synthesis_v327 import evaluate_gates
    monkeypatch.setitem(sys.modules, 'scripts.publication_gate', None)
    res = evaluate_gates()
    assert res["gates"]["G1"]["pass"] is False
    assert "G1" in res["unmet_gates"]

def test_main(monkeypatch):
    from scripts.experiment_3559_g_gate_status_synthesis_v327 import main
    # test it runs without error
    main()

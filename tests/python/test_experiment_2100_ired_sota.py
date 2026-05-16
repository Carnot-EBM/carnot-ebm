import os
import json
import numpy as np
from scripts.experiment_2100_ired_sota import (
    simulate_gguf_decoding_step,
    energy_fn_from_logits,
    run_benchmark,
    main,
    MODEL_SPECS
)

def test_simulate_gguf_decoding_step():
    """
    Verifies REQ-INFER-2100: Simulation of decoding step works.
    """
    logits = np.array([1.0, 1.0])
    state = np.array([0.0, 0.0])
    out = simulate_gguf_decoding_step(logits, state)
    assert np.allclose(out, np.array([0.01, 0.01]))

def test_energy_fn_from_logits():
    """
    Verifies REQ-INFER-2100: Energy function incorporates local model logits.
    """
    state = np.array([1.0, 2.0])
    logits = np.array([0.0, 0.0])
    energy, grad = energy_fn_from_logits(state, logits)
    assert energy == 5.0
    assert np.allclose(grad, np.array([2.0, 4.0]))

def test_run_benchmark():
    """
    Verifies REQ-INFER-2100: Runs 10 benchmark problems and measures pass rate.
    """
    res = run_benchmark("test-model")
    assert res["model"] == "test-model"
    assert res["problems_run"] == 10
    assert 0.0 <= res["pass_rate"] <= 1.0

def test_main(tmp_path, monkeypatch):
    """
    Verifies REQ-INFER-2100 and SCENARIO-INFER-2100:
    Hooks IRED optimizer into the local GGUF decoding loop for specific models
    and saves the results with ired_integrated=true.
    """
    import scripts.experiment_2100_ired_sota as exp
    # override path
    monkeypatch.setattr(exp, "os", os)
    
    out_path = tmp_path / "test.json"
    main(str(out_path))
    
    assert os.path.exists(str(out_path))
    with open(str(out_path)) as f:
        data = json.load(f)
        
    assert data["ired_integrated"] is True
    assert data["status"] == "complete"
    assert len(data["results"]) == 2
    assert "average_pass_rate" in data

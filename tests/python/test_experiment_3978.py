import pytest
import sys
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
import experiment_3978_verifier_vs_judge_efficiency as exp

def test_compute_ci():
    ci = exp.compute_ci(1.0, 5)
    assert ci["high"] == 1.0
    ci = exp.compute_ci(0.5, 10)
    assert ci["low"] < 0.5 < ci["high"]
    ci = exp.compute_ci(1.0, 0)
    assert ci == {"low": 0, "high": 0}

def test_create_programs():
    progs = exp.create_programs()
    assert len(progs) == 5
    import numpy as np
    dummy_s = np.zeros((64, 64), dtype=int)
    dummy_a = (6, 5, 5)
    
    # Test p1
    assert np.array_equal(progs[0]["fn"](dummy_s, dummy_a), dummy_s)
    
    # Test p2
    out2 = progs[1]["fn"](dummy_s, dummy_a)
    assert out2[5, 5] == 1
    
    # Test p3
    out3 = progs[2]["fn"](dummy_s, dummy_a)
    assert out3.max() == 2
    
    # Test p4
    out4 = progs[3]["fn"](dummy_s, dummy_a)
    assert out4[4, 5] == 3
    
    # Test p5
    out5 = progs[4]["fn"](dummy_s, dummy_a)
    assert out5.shape == (64, 64)

def test_run_blocked_no_gguf(monkeypatch, tmp_path):
    monkeypatch.setattr(exp, "get_gguf_path", lambda: None)
    
    # Change current working directory to tmp_path to not overwrite real results
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    
    exp.run()
    
    with open("results/experiment_3978_verifier_vs_judge_efficiency.json") as f:
        data = json.load(f)
    assert data["honest_verdict"] == "blocked_judge_gguf_not_cached"

def test_run_blocked_no_programs(monkeypatch, tmp_path):
    monkeypatch.setattr(exp, "get_gguf_path", lambda: "dummy.gguf")
    monkeypatch.setattr(exp, "_collect", lambda *args, **kwargs: [])
    
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    
    exp.run()
    
    with open("results/experiment_3978_verifier_vs_judge_efficiency.json") as f:
        data = json.load(f)
    assert data["honest_verdict"] == "blocked_no_induced_programs"


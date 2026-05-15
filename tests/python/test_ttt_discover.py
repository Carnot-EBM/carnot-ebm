import numpy as np
from carnot.pipeline.ttt_discover import calculate_entropic_utility, TTTDiscoverLoop

def test_calculate_entropic_utility():
    # Certain distribution (low entropy -> high utility)
    probs_certain = np.array([1.0, 0.0, 0.0])
    utility_certain = calculate_entropic_utility(probs_certain)
    
    # Uniform distribution (high entropy -> low utility)
    probs_uniform = np.array([1/3, 1/3, 1/3])
    utility_uniform = calculate_entropic_utility(probs_uniform)
    
    assert utility_certain > utility_uniform
    assert utility_certain > -1e-5  # Almost 0

def test_ttt_discover_loop():
    loop = TTTDiscoverLoop(model_specs="unsloth/Qwen3.6-35B-A3B-GGUF")
    samples = ["def foo(): pass", "def bar(): return 1"]
    
    results = loop.evaluate(samples)
    
    assert len(results) == 2
    for r in results:
        assert "entropic_utility" in r
        assert r["model"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
        assert r["verified"] is True

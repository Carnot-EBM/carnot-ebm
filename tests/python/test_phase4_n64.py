import pytest
from carnot.phase4_n64 import bootstrap_ci, run_n64_scaling_experiment

def test_bootstrap_ci():
    """Verify REQ-PHASE4-004 bootstrap logic"""
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    lower, upper = bootstrap_ci(data, n_bootstraps=100, ci=95, seed=42)
    assert lower <= upper
    assert lower >= 0.0
    assert upper <= 0.6

def test_run_n64_scaling_experiment():
    """Verify SCENARIO-PHASE4-3 measurement at n=64"""
    # use fewer bootstraps in test for speed
    import carnot.phase4_n64
    original_bootstrap = carnot.phase4_n64.bootstrap_ci
    
    def mocked_bootstrap(data, n_bootstraps=10, ci=95, seed=42):
        return original_bootstrap(data, n_bootstraps=10, ci=ci, seed=seed)
        
    carnot.phase4_n64.bootstrap_ci = mocked_bootstrap
    
    try:
        res = run_n64_scaling_experiment(n_spins=64, mld_steps=2, n_seeds=2, random_seed=42, git_rev="test")
        
        assert "model_specs" in res
        assert res["model_specs"]["n_spins"] == 64
        assert res["model_specs"]["n_seeds"] == 2
        assert res["model_specs"]["mld_steps"] == 2
        assert "reproducibility_checksum" in res
        assert "n_samples" in res
        assert res["n_samples"] == 4
        assert "delta_alpha_bootstrap_ci_95" in res
        assert len(res["delta_alpha_bootstrap_ci_95"]) == 2
        assert res["honest_verdict"].startswith("complete:")
    finally:
        carnot.phase4_n64.bootstrap_ci = original_bootstrap

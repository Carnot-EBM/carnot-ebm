import pytest
from carnot.fr11.multicorpus_deploy_verifier_diversity_grounding_v1 import run_multicorpus_deploy, _weaver_diverse_scores

def test_run_multicorpus_deploy_blocked():
    """Test battery gate with degenerate or too few corpora."""
    battery = [[{"is_correct": False} for _ in range(100)]]
    res = run_multicorpus_deploy(battery)
    assert "blocked" in res["honest_verdict"]

def test_run_multicorpus_deploy_success():
    """Test full loop runs correctly on valid mock battery."""
    # Create 2 valid corpora with 40% accuracy
    c1 = [{"is_correct": True} for _ in range(40)] + [{"is_correct": False} for _ in range(60)]
    c2 = [{"is_correct": True} for _ in range(50)] + [{"is_correct": False} for _ in range(50)]
    battery = [c1, c2]
    res = run_multicorpus_deploy(battery)
    
    assert "fr11_deploys_across_nondegenerate_battery" in res["honest_verdict"]
    assert res["n_battery_corpora"] == 2
    assert res["collapse_detected_control_beta0_any"] is True
    assert res["collapse_prevented_deploy_single_all_corpora"] is True
    assert res["pass_rate_vs_true_accuracy_distinct_assert"] is True

def test_weaver_diverse_scores():
    traces = [{"is_correct": True}, {"is_correct": False}]
    scores = _weaver_diverse_scores(traces, aw=0.045, k=3, seed=42)
    assert len(scores) == 2
    assert scores.shape == (2,)


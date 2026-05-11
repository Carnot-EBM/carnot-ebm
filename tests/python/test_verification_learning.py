import json
from pathlib import Path
from carnot.pipeline.verification_learning import VerificationLearningProxy

# Test traces to REQ-LEARN-1854, SCENARIO-LEARN-1854

def test_initialization():
    # Test without constraints
    proxy = VerificationLearningProxy()
    assert proxy.constraints == []
    
    # Test with constraints
    constraints = [{"type": "must_contain", "value": "test"}]
    proxy_with_c = VerificationLearningProxy(constraints=constraints)
    assert proxy_with_c.constraints == constraints

def test_score_constraint_satisfaction():
    constraints = [
        {"type": "must_contain", "value": "hello"},
        {"type": "must_not_contain", "value": "bad"},
        {"type": "unknown_type", "value": "whatever"}
    ]
    proxy = VerificationLearningProxy(constraints=constraints)
    
    unlabelled_data = [
        {"id": "1", "text": "hello world"}, # contains hello, doesn't contain bad, unknown counts -> 3/3
        {"id": "2", "text": "hello bad world"}, # contains hello, contains bad, unknown counts -> 2/3
        {"id": "3", "text": "good world"}, # doesn't contain hello, doesn't contain bad, unknown counts -> 2/3
    ]
    
    scores = proxy.score_constraint_satisfaction(unlabelled_data)
    assert scores["1"] == 1.0
    assert abs(scores["2"] - 2.0/3.0) < 1e-6
    assert abs(scores["3"] - 2.0/3.0) < 1e-6

def test_score_without_constraints():
    proxy = VerificationLearningProxy()
    unlabelled_data = [{"id": "1", "text": "hello world"}]
    scores = proxy.score_constraint_satisfaction(unlabelled_data)
    assert scores["1"] == 1.0

def test_compute_proxy_loss():
    constraints = [{"type": "must_contain", "value": "hello"}]
    proxy = VerificationLearningProxy(constraints=constraints)
    
    unlabelled_data = [
        {"id": "1", "text": "hello world"}, # 1.0 score
        {"id": "2", "text": "goodbye world"} # 0.0 score
    ]
    
    # Average score is 0.5, loss should be 0.5
    loss = proxy.compute_proxy_loss(unlabelled_data)
    assert loss == 0.5
    
    # Empty data
    assert proxy.compute_proxy_loss([]) == 0.0

def test_run_experiment_and_save(tmp_path):
    constraints = [{"type": "must_contain", "value": "hello"}]
    proxy = VerificationLearningProxy(constraints=constraints)
    
    unlabelled_data = [
        {"id": "1", "text": "hello world"},
        {"id": "2", "text": "goodbye world"}
    ]
    
    result_file = tmp_path / "experiment_1854_vl_proxy.json"
    
    result = proxy.run_experiment_and_save(unlabelled_data, str(result_file))
    
    assert result["experiment_id"] == "1854"
    # Loss is 0.5, logic says if loss < 0.5 then success else needs_improvement
    assert result["honest_verdict"] == "vl_proxy_needs_improvement"
    assert result["proxy_loss"] == 0.5
    assert result["constraint_count"] == 1
    assert result["data_count"] == 2
    
    assert result_file.exists()
    with open(result_file) as f:
        saved_data = json.load(f)
    assert saved_data == result

def test_run_experiment_success(tmp_path):
    constraints = [{"type": "must_contain", "value": "hello"}]
    proxy = VerificationLearningProxy(constraints=constraints)
    unlabelled_data = [{"id": "1", "text": "hello world"}]
    result_file = tmp_path / "experiment_1854_vl_proxy.json"
    result = proxy.run_experiment_and_save(unlabelled_data, str(result_file))
    assert result["honest_verdict"] == "vl_proxy_success"

def test_run_experiment_empty_data(tmp_path):
    proxy = VerificationLearningProxy()
    result_file = tmp_path / "experiment_1854_vl_proxy.json"
    result = proxy.run_experiment_and_save([], str(result_file))
    assert result["honest_verdict"] == "vl_proxy_empty_data"

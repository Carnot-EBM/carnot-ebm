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
        {"type": "unknown_type", "value": "whatever"},
    ]
    proxy = VerificationLearningProxy(constraints=constraints)

    unlabelled_data = [
        {
            "id": "1",
            "text": "hello world",
        },  # contains hello, doesn't contain bad, unknown counts -> 3/3
        {
            "id": "2",
            "text": "hello bad world",
        },  # contains hello, contains bad, unknown counts -> 2/3
        {
            "id": "3",
            "text": "good world",
        },  # doesn't contain hello, doesn't contain bad, unknown counts -> 2/3
    ]

    scores = proxy.score_constraint_satisfaction(unlabelled_data)
    assert scores["1"] == 1.0
    assert abs(scores["2"] - 2.0 / 3.0) < 1e-6
    assert abs(scores["3"] - 2.0 / 3.0) < 1e-6


def test_score_without_constraints():
    proxy = VerificationLearningProxy()
    unlabelled_data = [{"id": "1", "text": "hello world"}]
    scores = proxy.score_constraint_satisfaction(unlabelled_data)
    assert scores["1"] == 1.0


def test_compute_proxy_loss():
    constraints = [{"type": "must_contain", "value": "hello"}]
    proxy = VerificationLearningProxy(constraints=constraints)

    unlabelled_data = [
        {"id": "1", "text": "hello world"},  # 1.0 score
        {"id": "2", "text": "goodbye world"},  # 0.0 score
    ]

    # Average score is 0.5, loss should be 0.5
    loss = proxy.compute_proxy_loss(unlabelled_data)
    assert loss == 0.5

    # Empty data
    assert proxy.compute_proxy_loss([]) == 0.0


def test_run_experiment_and_save(tmp_path):
    constraints = [{"type": "must_contain", "value": "hello"}]
    proxy = VerificationLearningProxy(constraints=constraints)

    unlabelled_data = [{"id": "1", "text": "hello world"}, {"id": "2", "text": "goodbye world"}]

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


def test_rust_equivalence_1861(tmp_path):
    import json
    from carnot._rust import RustVerificationLearningProxy

    constraints = [
        {"type": "must_contain", "value": "hello"},
        {"type": "must_not_contain", "value": "bad"},
        {"type": "unknown_type", "value": "whatever"},
    ]
    py_proxy = VerificationLearningProxy(constraints=constraints)
    rs_proxy = RustVerificationLearningProxy(constraints=constraints)

    unlabelled_data = [
        {"id": "1", "text": "hello world"},
        {"id": "2", "text": "hello bad world"},
        {"id": "3", "text": "good world"},
    ]

    # Test scores equivalence
    py_scores = py_proxy.score_constraint_satisfaction(unlabelled_data)
    rs_scores = rs_proxy.score_constraint_satisfaction(unlabelled_data)

    assert py_scores.keys() == rs_scores.keys()
    for k in py_scores:
        assert abs(py_scores[k] - rs_scores[k]) < 1e-6

    # Test loss equivalence
    py_loss = py_proxy.compute_proxy_loss(unlabelled_data)
    rs_loss = rs_proxy.compute_proxy_loss(unlabelled_data)

    assert abs(py_loss - rs_loss) < 1e-6

    # Save the output to results/experiment_1861_equivalence.json
    result = {
        "experiment_id": "1861",
        "honest_verdict": "equivalence_verified",
        "py_loss": py_loss,
        "rs_loss": rs_loss,
        "scores_match": True,
    }
    # Write into the test's own sandbox, NOT into results/.
    #
    # This previously wrote to a hardcoded absolute path pointing at the committed
    # results/experiment_1861_equivalence.json. That was destructive in two ways:
    #
    #  1. It DELETED historical record. The committed artifact carries eleven keys
    #     including `corrigendum_2026_05_187_audit` and four adversarial-verification
    #     fields; the dict built above has six and none of those. Every run silently
    #     pruned a corrigendum, which the project's never-prune rule forbids.
    #  2. It made an unrelated test order-dependent. test_audit_1796.py asserts that
    #     `corrigendum_2026_05_187_audit` IS present in that same file, so whenever
    #     this test happened to run first -- which under pytest-xdist depends on
    #     sharding, not on anything deterministic -- that assertion failed for a
    #     reason nowhere near the test that actually caused it.
    #
    # The equivalence assertions above are the real content of this test; serialising
    # the result is incidental, so it keeps being exercised, just somewhere safe.
    output_path = tmp_path / "experiment_1861_equivalence.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Keep asserting the write actually happened and round-trips, so redirecting it
    # into the sandbox does not quietly reduce this test's coverage.
    assert output_path.is_file()
    assert json.loads(output_path.read_text()) == result

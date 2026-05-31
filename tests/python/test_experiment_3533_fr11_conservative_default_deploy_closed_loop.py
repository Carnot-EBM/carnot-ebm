"""Tests for Experiment 3533 — FR-11 Conservative-Default Beta Deploy Closed Loop v1.

References: REQ-FR11-CLD-005, SCENARIO-FR11-CLD-005
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from carnot.fr11.conservative_default_deploy_closed_loop_v1 import (
    CONSERVATIVE_DEFAULT_BETA,
    ENTROPY_COLLAPSE_THRESHOLD,
    FRESH_DEPLOY_CONFIG,
    N_ITERATIONS,
    QUALITY_DEGRADATION_TOLERANCE,
    RANDOM_SEED,
    run_arm_closed_loop,
    run_conservative_default_deploy_closed_loop,
)


# ---------------------------------------------------------------------------
# Unit tests for run_arm_closed_loop
# ---------------------------------------------------------------------------


def _make_traces(n_correct: int, n_wrong: int) -> list[dict]:
    return [{"is_correct": True}] * n_correct + [{"is_correct": False}] * n_wrong


def test_run_arm_closed_loop_returns_required_keys():
    """run_arm_closed_loop returns all keys the top-level aggregator consumes.

    References: REQ-FR11-CLD-005
    """
    traces = _make_traces(2, 3)
    at_risk = np.array([0.9, 0.8, 0.2, 0.1, 0.3])
    result = run_arm_closed_loop(traces, at_risk, n_iterations=5, entropy_beta=0.5,
                                  config_name="test", arm_label="DEPLOY")
    for key in (
        "collapse_detected", "initial_entropy", "initial_pass_rate",
        "initial_true_accuracy", "final_entropy", "final_mode_mass",
        "entropy_drop_ratio", "final_pass_rate", "final_true_accuracy", "final_gap",
    ):
        assert key in result, f"Missing key: {key}"


def test_run_arm_closed_loop_beta0_concentrates_mass():
    """With beta=0 and highly-discriminating verifier scores, the distribution
    concentrates on high-scoring traces (mode_mass grows).

    References: SCENARIO-FR11-CLD-005
    """
    n = 20
    traces = [{"is_correct": i < 5} for i in range(n)]
    at_risk = np.array([0.95 if i < 5 else 0.05 for i in range(n)])
    result = run_arm_closed_loop(
        traces, at_risk, n_iterations=50, entropy_beta=0.0,
        config_name="test", arm_label="CONTROL"
    )
    # With strong signal + beta=0, mass concentrates on few traces
    assert result["final_mode_mass"] > result["initial_true_accuracy"] * 0.5


def test_run_arm_closed_loop_conservative_beta_maintains_entropy():
    """With beta=0.5 (conservative), entropy stays higher than with beta=0.

    References: REQ-FR11-CLD-005
    """
    n = 20
    traces = [{"is_correct": i < 5} for i in range(n)]
    at_risk = np.array([0.95 if i < 5 else 0.05 for i in range(n)])

    result_deploy = run_arm_closed_loop(
        traces, at_risk, n_iterations=50, entropy_beta=0.5,
        config_name="test", arm_label="DEPLOY"
    )
    result_control = run_arm_closed_loop(
        traces, at_risk, n_iterations=50, entropy_beta=0.0,
        config_name="test", arm_label="CONTROL"
    )
    # Conservative beta maintains higher entropy than beta=0
    assert result_deploy["final_entropy"] >= result_control["final_entropy"]


def test_run_arm_closed_loop_gap_is_pass_minus_true():
    """final_gap == final_pass_rate - final_true_accuracy (verifier gap)."""
    traces = [{"is_correct": True}, {"is_correct": False}]
    at_risk = np.array([0.8, 0.3])
    result = run_arm_closed_loop(traces, at_risk, n_iterations=3, entropy_beta=0.0,
                                  config_name="test", arm_label="A")
    assert abs(result["final_gap"] - (result["final_pass_rate"] - result["final_true_accuracy"])) < 1e-9


# ---------------------------------------------------------------------------
# Unit tests for constants
# ---------------------------------------------------------------------------


def test_conservative_default_beta_is_half():
    """Conservative default beta=0.5 matches the exp3521 recommendation."""
    assert CONSERVATIVE_DEFAULT_BETA == 0.5


def test_fresh_deploy_config_is_distinct_from_prior_sets():
    """FRESH_DEPLOY_CONFIG.active_weight must not be in any prior fit/selection set.

    References: REQ-FR11-CLD-005 (fresh corpus, not refit)
    """
    prior_aws = {0.05, 0.10, 0.146, 0.30, 0.07, 0.20, 0.06, 0.08, 0.18, 0.22}
    assert FRESH_DEPLOY_CONFIG["active_weight"] not in prior_aws


def test_random_seed_not_experiment_number():
    """RANDOM_SEED must not equal the experiment number 3533 (content-derived)."""
    assert RANDOM_SEED != 3533


def test_n_iterations_at_least_200():
    """N_ITERATIONS >= 200 is required for the deployment stress depth."""
    assert N_ITERATIONS >= 200


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


def _write_traces_file(tmp_path, n_correct: int = 3, n_wrong: int = 7) -> str:
    traces = (
        [{"is_correct": True}] * n_correct +
        [{"is_correct": False}] * n_wrong
    )
    p = tmp_path / "traces.jsonl"
    with open(p, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    return str(p)


def test_run_conservative_default_deploy_closed_loop_smoke(tmp_path):
    """Smoke: runs to completion on a tiny corpus and returns all required fields.

    References: REQ-FR11-CLD-005
    """
    p = _write_traces_file(tmp_path)
    res = run_conservative_default_deploy_closed_loop(
        traces_path=p,
        n_iterations=10,
        seed=42,
        fresh_config={"name": "smoke_test", "active_weight": 0.04},
    )
    required = [
        "honest_verdict", "inference_substrate", "n_steps", "fresh_corpus_used",
        "conservative_default_beta", "deployed_alpha_t_margin",
        "collapse_detected_deploy_arm", "collapse_detected_control_beta0",
        "deploy_arm_final_true_accuracy", "quality_maintained",
        "pass_rate_vs_true_accuracy_distinct_assert",
        "random_seed", "reproducibility_checksum", "duration_s",
    ]
    for key in required:
        assert key in res, f"Missing required field: {key}"
    assert res["honest_verdict"].startswith("complete:")
    assert res["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert res["fresh_corpus_used"] is True
    assert res["conservative_default_beta"] == 0.5
    assert res["pass_rate_vs_true_accuracy_distinct_assert"] is True
    assert res["duration_s"] >= 1.0


def test_run_conservative_default_deploy_blocked_on_empty_corpus(tmp_path):
    """Empty corpus emits a blocked verdict, not a fabricated success.

    References: REQ-FR11-CLD-005 (precondition check)
    """
    p = tmp_path / "empty.jsonl"
    p.touch()
    res = run_conservative_default_deploy_closed_loop(
        traces_path=str(p), n_iterations=5, seed=42,
        fresh_config={"name": "test", "active_weight": 0.04},
    )
    assert res["honest_verdict"] == "complete: blocked_fr11_module_or_traces_unavailable"


def test_deploy_prevents_collapse_control_collapses(tmp_path):
    """With a weak grounding signal (low AW), a mocked run confirms the acceptance gate:
    deploy arm does NOT collapse, control arm DOES collapse.

    References: SCENARIO-FR11-CLD-005
    """
    from unittest.mock import patch

    p = _write_traces_file(tmp_path, n_correct=1, n_wrong=9)

    # Mock run_arm_closed_loop: DEPLOY = no collapse, CONTROL = collapse
    call_count = [0]
    def _mock_arm(traces, at_risk, n_iterations, entropy_beta, config_name, arm_label):
        call_count[0] += 1
        if entropy_beta > 0:
            # DEPLOY arm
            return {
                "collapse_detected": False,
                "initial_entropy": 2.0, "initial_pass_rate": 0.3, "initial_true_accuracy": 0.1,
                "final_entropy": 1.8, "final_mode_mass": 0.1,
                "entropy_drop_ratio": 0.1, "final_pass_rate": 0.7, "final_true_accuracy": 0.12,
                "final_gap": 0.58,
            }
        else:
            # CONTROL arm
            return {
                "collapse_detected": True,
                "initial_entropy": 2.0, "initial_pass_rate": 0.3, "initial_true_accuracy": 0.1,
                "final_entropy": 0.05, "final_mode_mass": 0.9,
                "entropy_drop_ratio": 0.975, "final_pass_rate": 1.0, "final_true_accuracy": 1e-50,
                "final_gap": 1.0,
            }

    with patch(
        "carnot.fr11.conservative_default_deploy_closed_loop_v1.run_arm_closed_loop",
        side_effect=_mock_arm,
    ):
        res = run_conservative_default_deploy_closed_loop(
            traces_path=p, n_iterations=5, seed=42,
            fresh_config={"name": "mock_test", "active_weight": 0.04},
        )

    assert res["collapse_detected_deploy_arm"] is False
    assert res["collapse_detected_control_beta0"] is True
    assert res["acceptance_gates"]["G1_deploys_end_to_end"] is True
    assert res["honest_verdict"] == (
        "complete: conservative_default_beta_deploys_end_to_end_prevents_collapse_"
        "to_N200_quality_maintained"
    )
    assert call_count[0] == 2  # exactly two arms were run


def test_quality_not_maintained_gives_tuning_verdict(tmp_path):
    """When deploy arm prevents collapse but quality drops significantly, verdict
    signals over-regularization rather than full success.

    References: SCENARIO-FR11-CLD-005
    """
    from unittest.mock import patch

    p = _write_traces_file(tmp_path)

    def _mock_arm_quality_drop(traces, at_risk, n_iterations, entropy_beta, config_name, arm_label):
        if entropy_beta > 0:
            return {
                "collapse_detected": False,
                "initial_entropy": 2.0, "initial_pass_rate": 0.5, "initial_true_accuracy": 0.2,
                "final_entropy": 1.9, "final_mode_mass": 0.05,
                "entropy_drop_ratio": 0.05,
                # final_true is well below initial (0.2 - 0.001 tolerance) = 0.199 minimum
                "final_pass_rate": 0.6, "final_true_accuracy": 0.01, "final_gap": 0.59,
            }
        else:
            return {
                "collapse_detected": True,
                "initial_entropy": 2.0, "initial_pass_rate": 0.5, "initial_true_accuracy": 0.2,
                "final_entropy": 0.01, "final_mode_mass": 0.95,
                "entropy_drop_ratio": 0.995, "final_pass_rate": 1.0, "final_true_accuracy": 1e-60,
                "final_gap": 1.0,
            }

    with patch(
        "carnot.fr11.conservative_default_deploy_closed_loop_v1.run_arm_closed_loop",
        side_effect=_mock_arm_quality_drop,
    ):
        res = run_conservative_default_deploy_closed_loop(
            traces_path=p, n_iterations=5, seed=42,
            fresh_config={"name": "quality_drop_test", "active_weight": 0.04},
        )

    assert res["collapse_detected_deploy_arm"] is False
    assert res["collapse_detected_control_beta0"] is True
    assert res["quality_maintained"] is False
    assert "over_regularizes" in res["honest_verdict"]


def test_neither_arm_prevents_collapse_gives_new_mechanism_verdict(tmp_path):
    """When deploy arm also collapses, verdict directs toward a new mechanism.

    References: SCENARIO-FR11-CLD-005
    """
    from unittest.mock import patch

    p = _write_traces_file(tmp_path)

    def _mock_arm_all_collapse(traces, at_risk, n_iterations, entropy_beta, config_name, arm_label):
        return {
            "collapse_detected": True,
            "initial_entropy": 2.0, "initial_pass_rate": 0.5, "initial_true_accuracy": 0.2,
            "final_entropy": 0.01, "final_mode_mass": 0.95,
            "entropy_drop_ratio": 0.995, "final_pass_rate": 1.0, "final_true_accuracy": 1e-60,
            "final_gap": 1.0,
        }

    with patch(
        "carnot.fr11.conservative_default_deploy_closed_loop_v1.run_arm_closed_loop",
        side_effect=_mock_arm_all_collapse,
    ):
        res = run_conservative_default_deploy_closed_loop(
            traces_path=p, n_iterations=5, seed=42,
            fresh_config={"name": "all_collapse_test", "active_weight": 0.04},
        )

    assert res["collapse_detected_deploy_arm"] is True
    assert "new_mechanism" in res["honest_verdict"]


def test_default_fresh_config_used_when_none_passed(tmp_path):
    """Calling without fresh_config falls back to FRESH_DEPLOY_CONFIG (line 233).

    References: REQ-FR11-CLD-005
    """
    from unittest.mock import patch

    p = _write_traces_file(tmp_path)

    def _mock_arm_ok(traces, at_risk, n_iterations, entropy_beta, config_name, arm_label):
        collapsed = entropy_beta == 0.0
        return {
            "collapse_detected": collapsed,
            "initial_entropy": 2.0, "initial_pass_rate": 0.3, "initial_true_accuracy": 0.1,
            "final_entropy": 0.05 if collapsed else 1.8, "final_mode_mass": 0.9 if collapsed else 0.1,
            "entropy_drop_ratio": 0.975 if collapsed else 0.1,
            "final_pass_rate": 1.0 if collapsed else 0.7,
            "final_true_accuracy": 1e-50 if collapsed else 0.12,
            "final_gap": 1.0 if collapsed else 0.58,
        }

    with patch(
        "carnot.fr11.conservative_default_deploy_closed_loop_v1.run_arm_closed_loop",
        side_effect=_mock_arm_ok,
    ):
        res = run_conservative_default_deploy_closed_loop(
            traces_path=p, n_iterations=5, seed=42,
            fresh_config=None,  # triggers the default path
        )
    # Should have used FRESH_DEPLOY_CONFIG
    assert res["fresh_config"]["name"] == FRESH_DEPLOY_CONFIG["name"]
    assert res["honest_verdict"].startswith("complete:")


def test_alpha_t_margin_definition():
    """deployed_alpha_t_margin = final_entropy_deploy - ENTROPY_COLLAPSE_THRESHOLD.

    A positive margin means the deploy arm is above the collapse entropy threshold.

    References: REQ-FR11-CLD-005
    """
    from unittest.mock import patch

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"is_correct": True}) + "\n")
        f.write(json.dumps({"is_correct": False}) + "\n")
        tmpf = f.name

    try:
        expected_entropy = 2.5  # above collapse threshold

        def _mock_arm_entropy(traces, at_risk, n_iterations, entropy_beta, config_name, arm_label):
            if entropy_beta > 0:
                return {
                    "collapse_detected": False,
                    "initial_entropy": 3.0, "initial_pass_rate": 0.5, "initial_true_accuracy": 0.5,
                    "final_entropy": expected_entropy, "final_mode_mass": 0.05,
                    "entropy_drop_ratio": 0.1, "final_pass_rate": 0.8, "final_true_accuracy": 0.5,
                    "final_gap": 0.3,
                }
            else:
                return {
                    "collapse_detected": True,
                    "initial_entropy": 3.0, "initial_pass_rate": 0.5, "initial_true_accuracy": 0.5,
                    "final_entropy": 0.05, "final_mode_mass": 0.9,
                    "entropy_drop_ratio": 0.98, "final_pass_rate": 1.0, "final_true_accuracy": 0.0,
                    "final_gap": 1.0,
                }

        with patch(
            "carnot.fr11.conservative_default_deploy_closed_loop_v1.run_arm_closed_loop",
            side_effect=_mock_arm_entropy,
        ):
            res = run_conservative_default_deploy_closed_loop(
                traces_path=tmpf, n_iterations=5, seed=42,
                fresh_config={"name": "margin_test", "active_weight": 0.04},
            )

        expected_margin = expected_entropy - ENTROPY_COLLAPSE_THRESHOLD
        assert abs(res["deployed_alpha_t_margin"] - expected_margin) < 1e-9
    finally:
        os.unlink(tmpf)

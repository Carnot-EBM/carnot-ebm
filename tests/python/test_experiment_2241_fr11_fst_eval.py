"""Tests for Exp 2241 FR-11 Fast-Slow Training evaluation.

Spec: REQ-LEARN-2241, SCENARIO-LEARN-2241.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import fr11_fst_eval as mod


def test_req_learn_2241_builds_30_example_synthetic_fover_corpus() -> None:
    """REQ-LEARN-2241: corpus is 30 synthetic arithmetic CoT rows."""

    corpus = mod.build_synthetic_fover_corpus()

    assert len(corpus) == 30
    assert {example.error_type for example in corpus} == set(mod.ERROR_TYPES)
    assert all("FoVer arithmetic case" in example.question for example in corpus)
    assert all(example.correct_answer != example.wrong_answer for example in corpus)


def test_scenario_learn_2241_fst_passes_sample_efficiency_and_kl_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2241: FST beats parameter-only RL on the gates."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.run_experiment(output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_fst_eval_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["n_corpus"] == 30
    assert artifact["sample_efficiency_ratio"] >= 2.0
    assert artifact["kl_drift_ratio"] <= 0.5
    assert artifact["utility_delta"] > 0.0

    regimes = artifact["regimes"]
    assert regimes["A_parameter_only_rl"]["iterations"] == 5
    assert regimes["B_fast_slow_training"]["iterations"] == 5
    assert regimes["B_fast_slow_training"]["slow_weights_frozen"] is True
    assert regimes["B_fast_slow_training"]["fst_certificate"]["fast_update_count"] == 5
    assert (
        "FST verifier-output summary" in regimes["B_fast_slow_training"]["prompt_prefix_samples"][0]
    )


def test_req_learn_2241_blocked_artifact_when_fast_slow_file_is_missing(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-2241-1: missing FST module writes an honest blocked artifact."""

    output = tmp_path / mod.OUTPUT_FILE
    missing_fast_slow = tmp_path / "missing_fast_slow.py"

    artifact = mod.run_experiment(output_path=output, fast_slow_path=missing_fast_slow)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_fst_module_missing"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_fst_eval_passed"] is False
    assert artifact["n_corpus"] == 0
    assert artifact["preconditions_checked"][0]["status"] == "failed"


def test_req_learn_2241_acceptance_boolean_matches_gate_formula(tmp_path: Path) -> None:
    """REQ-LEARN-2241-2: pass boolean is exactly the two-gate conjunction."""

    artifact = mod.run_experiment(output_path=tmp_path / mod.OUTPUT_FILE)

    expected = artifact["sample_efficiency_ratio"] >= 2.0 and artifact["kl_drift_ratio"] <= 0.5
    assert artifact["fr11_fst_eval_passed"] is expected
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)

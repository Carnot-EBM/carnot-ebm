"""Tests for Exp5570 active-spline online KAN energy.

Spec refs: REQ-LEARN-5570,
SCENARIO-LEARN-5570-STREAM,
SCENARIO-LEARN-5570-ACTIVE-SPLINE,
SCENARIO-LEARN-5570-ROLLBACK,
SCENARIO-LEARN-5570-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_5570_spline_local_kan_online_energy as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5570_spline_local_kan_online_energy.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5570_spline_local_kan_online_energy.py "
    "-m pytest tests/python/test_experiment_5570_spline_local_kan_online_energy.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5570_spline_local_kan_online_energy.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def _artifact(tmp_path: Path) -> dict[str, object]:
    return exp.build_artifact(
        root=REPO,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        checkpoint_dir=tmp_path / "checkpoints",
    )


def test_req_learn_5570_spec_declares_online_kan_contract() -> None:
    """REQ-LEARN-5570: OpenSpec anchors active-spline KAN adaptation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5570") :]

    for marker in (
        "REQ-LEARN-5570",
        "SCENARIO-LEARN-5570-STREAM",
        "SCENARIO-LEARN-5570-ACTIVE-SPLINE",
        "SCENARIO-LEARN-5570-ROLLBACK",
        "SCENARIO-LEARN-5570-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "active-spline-only online KAN",
        "paired confidence interval excluding zero",
        "rollback reproduces the pre-update checksum",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert exp.FIELD_PRINCIPLES[field]


def test_scenario_learn_5570_stream_builds_sessions_and_holdout() -> None:
    """SCENARIO-LEARN-5570-STREAM: exact corpus rows become online sessions."""

    dataset = exp.build_dataset(root=REPO)
    session_ids = {row.session_id for row in dataset.online_rows}
    online_ids = {row.row_id for row in dataset.online_rows}
    holdout_ids = {row.row_id for row in dataset.holdout_rows}

    assert dataset.n_rows == 120
    assert len(dataset.online_rows) == 80
    assert len(dataset.holdout_rows) == 40
    assert len(dataset.sessions) == 4
    assert session_ids == set(exp.REQUIRED_FAMILIES)
    assert online_ids.isdisjoint(holdout_ids)
    assert all(row.features.shape == (exp.FEATURE_DIM,) for row in dataset.rows)
    assert all(row.label in (-1, 1) for row in dataset.rows)
    assert dataset.dataset_path == exp.DATASET_RELATIVE_PATH.as_posix()
    assert exp.future_holdout_update_leakage(dataset) == 0


def test_scenario_learn_5570_active_update_only_touches_active_knots() -> None:
    """SCENARIO-LEARN-5570-ACTIVE-SPLINE: inactive coefficients stay fixed."""

    model = exp.OnlineKANEnergyModel(seed=11, n_params=8, init_scale=0.0)
    row = exp.FeatureRow(
        row_id="unit-row",
        family="unit",
        partition="train",
        session_id="unit",
        label=1,
        accepted_by_exact_validator=True,
        features=np.array([1.0, 0.0, 0.5, 0.0, -0.25, 0.0, 0.0, 0.0]),
    )
    before = model.coefficients.copy()
    receipt = exp.apply_online_update(
        model,
        [row],
        learning_rate=0.2,
        arm=exp.ACTIVE_ARM,
    )

    active_mask = row.features != 0.0
    assert receipt.arm == exp.ACTIVE_ARM
    assert receipt.touched_indices == [0, 2, 4]
    assert np.any(model.coefficients[active_mask] != before[active_mask])
    assert np.all(model.coefficients[~active_mask] == before[~active_mask])
    assert receipt.touched_fraction == pytest.approx(3 / 8)

    dense = exp.OnlineKANEnergyModel(seed=11, n_params=8, init_scale=0.0)
    dense_receipt = exp.apply_online_update(
        dense,
        [row],
        learning_rate=0.2,
        arm=exp.DENSE_ARM,
    )
    assert dense_receipt.touched_fraction == pytest.approx(1.0)
    assert len(dense_receipt.touched_indices) == dense.n_params

    frozen = exp.OnlineKANEnergyModel(seed=11, n_params=8, init_scale=0.0)
    frozen_before = frozen.coefficients.copy()
    frozen_receipt = exp.apply_online_update(
        frozen,
        [row],
        learning_rate=0.2,
        arm=exp.FROZEN_ARM,
    )
    assert frozen_receipt.touched_indices == []
    assert frozen_receipt.update_count == 0
    assert np.all(frozen.coefficients == frozen_before)


def test_scenario_learn_5570_rollback_restores_checkpoint(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5570-ROLLBACK: checksum matches pre-update state."""

    model = exp.OnlineKANEnergyModel(seed=5570, n_params=exp.FEATURE_DIM)
    checkpoint = exp.write_checkpoint(
        model,
        tmp_path,
        seed=5570,
        session_id="unit-session",
        phase="pre-promotion",
    )
    checksum_before = model.checksum()
    model.coefficients += 1.0
    assert model.checksum() != checksum_before

    restored = exp.OnlineKANEnergyModel.from_checkpoint(checkpoint.path)
    model.restore(restored.snapshot())
    assert model.checksum() == checksum_before
    assert exp.rollback_checksum_match(checkpoint, model) is True


def test_scenario_learn_5570_online_arms_and_gate_metrics(tmp_path: Path) -> None:
    """REQ-LEARN-5570-2/5: active arm adapts safely against controls."""

    dataset = exp.build_dataset(root=REPO)
    result = exp.run_online_experiment(
        dataset,
        seeds=exp.DEFAULT_SEEDS,
        replay_budget=exp.DEFAULT_REPLAY_BUDGET,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result["arms"] == list(exp.ARM_NAMES)
    assert result["seeds"] == list(exp.DEFAULT_SEEDS)
    assert result["heldout_exact_error_by_arm"][exp.ACTIVE_ARM] < result[
        "heldout_exact_error_by_arm"
    ][exp.FROZEN_ARM]
    assert result["paired_ci_active_vs_frozen"]["lower"] > 0.0
    assert result["prior_family_regression"] <= 0.02
    assert result["unsafe_false_accept_delta"] <= 0.0
    assert result["rollback_checksum_match"] is True
    assert result["active_update_summary"]["touched_spline_fraction"] < result[
        "dense_update_summary"
    ]["touched_spline_fraction"]
    assert result["active_update_summary"]["parameter_diff_norm"] > 0.0
    assert result["active_update_summary"]["update_count"] > 0
    assert all(Path(item["path"]).exists() for item in result["checkpoint_paths"])
    assert exp.kan_ready(result) is True


def test_scenario_learn_5570_artifact_fields_and_validation(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5570-ARTIFACT: receipt exposes required gate evidence."""

    destination = tmp_path / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        checkpoint_dir=tmp_path / "checkpoints",
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert exp.validate_artifact(artifact) is True
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]

    assert artifact["continuous_self_learning_target"] is True
    assert artifact["dataset_path"] == exp.DATASET_RELATIVE_PATH.as_posix()
    assert artifact["n_rows"] == 120
    assert artifact["arms"] == list(exp.ARM_NAMES)
    assert artifact["exact_feedback_only"] is True
    assert artifact["weights_mutated"] is True
    assert artifact["active_spline_update"] is True
    assert 0.0 < artifact["touched_spline_fraction"] < 1.0
    assert artifact["update_count"] > 0
    assert artifact["parameter_diff_norm"] > 0.0
    assert artifact["update_latency_ms"][exp.ACTIVE_ARM]["total"] > 0.0
    assert artifact["forward_adaptation_delta"] > 0.0
    assert artifact["prior_family_regression"] <= 0.02
    assert artifact["unsafe_false_accept_delta"] <= 0.0
    assert artifact["replay_budget"] == exp.DEFAULT_REPLAY_BUDGET
    assert artifact["rollback_checksum_match"] is True
    assert artifact["promotion_thresholds"]["paired_ci_excludes_zero"] is True
    assert artifact["kan_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_req_learn_5570_artifact_gate_fails_closed(tmp_path: Path) -> None:
    """REQ-LEARN-5570-5: kan_ready cannot pass unsafe or inconsistent gates."""

    artifact = _artifact(tmp_path)
    assert exp.validate_artifact(artifact) is True

    bad_cases = [
        ("continuous_self_learning_target", False, "continuous_self_learning_target"),
        ("dataset_path", "wrong.json", "dataset_path"),
        ("sessions", [], "sessions"),
        ("n_rows", 119, "n_rows"),
        ("seeds", [5570], "seeds"),
        ("arms", list(exp.ARM_NAMES[:-1]), "arms"),
        ("exact_feedback_only", False, "exact_feedback_only"),
        ("weights_mutated", False, "weights_mutated"),
        ("active_spline_update", False, "active_spline_update"),
        ("touched_spline_fraction", 1.0, "touched_spline_fraction"),
        ("update_count", 0, "update_count"),
        ("parameter_diff_norm", 0.0, "parameter_diff_norm"),
        ("forward_adaptation_delta", 0.0, "forward_adaptation_delta"),
        ("prior_family_regression", 0.03, "prior_family_regression"),
        ("unsafe_false_accept_delta", 0.01, "unsafe_false_accept_delta"),
        ("replay_budget", -1, "replay_budget"),
        ("checkpoint_paths", [], "checkpoint_paths"),
        ("rollback_checksum_match", False, "rollback_checksum_match"),
        ("field_principles", {}, "field_principles"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    blocked = deepcopy(artifact)
    blocked["promotion_thresholds"]["paired_ci_excludes_zero"] = False
    blocked["paired_ci_active_vs_frozen"]["lower"] = 0.0
    blocked["kan_ready"] = False
    blocked["honest_verdict"] = exp.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert exp.validate_artifact(blocked) is True
    assert blocked["honest_verdict"].startswith("blocked:")

    invalid_claim = deepcopy(blocked)
    invalid_claim["kan_ready"] = True
    invalid_claim["honest_verdict"] = "complete: invalid"
    invalid_claim["reproducibility_checksum"] = exp.reproducibility_checksum(invalid_claim)
    with pytest.raises(ValueError, match="kan_ready"):
        exp.validate_artifact(invalid_claim)

    missing = deepcopy(artifact)
    missing.pop("kan_ready")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    bad_latency = deepcopy(artifact)
    bad_latency["update_latency_ms"] = {}
    bad_latency["reproducibility_checksum"] = exp.reproducibility_checksum(bad_latency)
    with pytest.raises(ValueError, match="update_latency_ms"):
        exp.validate_artifact(bad_latency)

    bad_principle_type = deepcopy(artifact)
    bad_principle_type["field_principles"] = "not-a-mapping"
    bad_principle_type["reproducibility_checksum"] = exp.reproducibility_checksum(
        bad_principle_type
    )
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle_type)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    assert exp.select_replay_rows(artifact["sessions"], current_family="x", replay_budget=0) == []
    assert exp.confidence_interval([0.25]) == {
        "mean": 0.25,
        "lower": 0.25,
        "upper": 0.25,
        "n": 1,
    }

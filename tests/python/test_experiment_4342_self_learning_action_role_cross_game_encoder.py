"""Tests for Exp 4342 action-role cross-game ARC transfer.

Spec refs: REQ-LEARN-4342, SCENARIO-LEARN-4342.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_4342_self_learning_action_role_cross_game_encoder as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
WRAPPER_PATH = REPO / "results" / "experiment_4342_self_learning_action_role_cross_game_encoder.py"


def test_req_learn_4342_spec_declares_action_role_contract() -> None:
    """REQ-LEARN-4342: OpenSpec declares the action-role artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    normalized_spec = " ".join(spec.split())

    for marker in (
        "REQ-LEARN-4342",
        "SCENARIO-LEARN-4342",
        "SCENARIO-LEARN-4342-BLOCKED",
        "experiment_4342_self_learning_action_role_cross_game_encoder.json",
        "python/carnot/experiment_4342_self_learning_action_role_cross_game_encoder.py",
        "learned_encoder_transfer_helps",
        "blocked_insufficient_game_traces",
        "action-role interaction features",
        "positive_control_passed",
        "n_held_out_games",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.FIELD_PRINCIPLES.values():
        assert " ".join(principle.split()) in normalized_spec


def test_req_learn_4342_transition_features_are_game_agnostic_roles() -> None:
    """REQ-LEARN-4342-4: features encode role/effect, not game-specific pixels."""

    before = np.zeros((4, 4), dtype=np.int16)
    before[1, 1] = 7
    after = np.zeros((4, 4), dtype=np.int16)
    after[1, 2] = 7
    terminal = np.zeros((4, 4), dtype=np.int16)
    terminal[1, 2] = 7

    features = exp.action_role_feature_map(
        action_id=4,
        data=None,
        before_grid=before,
        after_grid=after,
        terminal_grid=terminal,
    )

    assert features["role_directional"] == pytest.approx(1.0)
    assert features["role_click"] == pytest.approx(0.0)
    assert features["changed_fraction"] > 0.0
    assert features["centroid_shift"] > 0.0
    assert features["goal_alignment_gain"] > 0.0
    assert set(exp.ACTION_ROLE_FEATURE_NAMES) == set(features)

    click_noop = exp.action_role_feature_map(
        action_id=6,
        data={"x": 1, "y": 1},
        before_grid=before,
        after_grid=before,
        terminal_grid=terminal,
    )
    assert click_noop["role_click"] == pytest.approx(1.0)
    assert click_noop["is_noop"] == pytest.approx(1.0)
    assert click_noop["changed_fraction"] == pytest.approx(0.0)

    class FrameObject:
        frame = np.array([[[0, 1, 0], [0, 1, 0]]], dtype=np.int16)

    padded = exp.action_role_feature_map(
        action_id=5,
        data=None,
        before_grid=[[0, 1]],
        after_grid=FrameObject(),
        terminal_grid=None,
    )
    assert padded["role_commit"] == pytest.approx(1.0)
    assert padded["changed_fraction"] > 0.0


def test_req_learn_4342_encoder_trains_deterministic_embedding() -> None:
    """REQ-LEARN-4342-4: the action-role encoder fits and transforms rows."""

    rows = [
        [0.0, 1.0, 0.0, 0.25],
        [1.0, 0.0, 0.0, 0.50],
        [0.5, 0.5, 1.0, 0.75],
    ]
    encoder = exp.ActionRoleInteractionEncoder(feature_names=("a", "b", "c", "d")).fit(rows)

    encoded = encoder.transform(rows[0])

    assert len(encoded) == 4
    assert all(np.isfinite(encoded))
    assert encoder.n_samples == 3
    assert encoder.model_summary()["architecture"] == "action_role_interaction_standardizer"
    assert encoder.model_summary()["feature_names"] == ["a", "b", "c", "d"]
    assert encoder.transform_many(rows[:2]) == [encoder.transform(rows[0]), encoder.transform(rows[1])]

    with pytest.raises(ValueError, match="no action-role rows"):
        exp.ActionRoleInteractionEncoder().fit([])
    with pytest.raises(ValueError, match="untrained"):
        exp.ActionRoleInteractionEncoder().transform(rows[0])


def test_req_learn_4342_value_head_and_transition_scorer_are_deterministic() -> None:
    """REQ-LEARN-4342-4: encoded transition rows feed a learned value head."""

    before = np.zeros((3, 3), dtype=np.int16)
    before[1, 1] = 1
    after = np.zeros((3, 3), dtype=np.int16)
    after[1, 2] = 1
    terminal = after.copy()
    feature = exp.action_role_feature_map(
        action_id=4,
        data=None,
        before_grid=before,
        after_grid=after,
        terminal_grid=terminal,
    )
    row = [feature[name] for name in exp.ACTION_ROLE_FEATURE_NAMES]
    worse = list(row)
    worse[exp.ACTION_ROLE_FEATURE_NAMES.index("goal_alignment_gain")] = -1.0
    encoder = exp.ActionRoleInteractionEncoder().fit([row, worse])
    encoded = encoder.transform_many([row, worse])
    head = exp.ActionRoleValueHead().fit(encoded, [0.0, 5.0])
    scorer = exp.TransitionScorer(encoder, head)

    assert head.n_samples == 2
    assert head.rounded_weights()
    assert head.model_summary()["architecture"] == "linear least-squares value head with bias"
    assert head.predict(encoded[0]) < head.predict(encoded[1])
    assert scorer.score(4, None, before, after, terminal) == pytest.approx(head.predict(encoded[0]))
    assert exp._feature_vector(feature) == row

    untrained = exp.ActionRoleValueHead()
    assert untrained.predict(row) == 0.0
    assert untrained.rounded_weights() == []
    with pytest.raises(ValueError, match="no rows"):
        exp.ActionRoleValueHead().fit([], [])

    label = json.dumps({"action": 6, "data": {"x": 2, "y": 3}}, sort_keys=True)
    action_id, data = exp._label_action_data(label)
    assert action_id == 6
    assert data == {"x": 2, "y": 3}


def test_scenario_learn_4342_blocked_artifact_is_terminal_and_bare() -> None:
    """SCENARIO-LEARN-4342-BLOCKED: insufficient traces fail closed."""

    artifact = exp.build_blocked_artifact(
        usable_games=["r11l", "ls20"],
        missing_games=["wa30", "lp85"],
        preconditions_checked={
            "usable_trace_game_count": 2,
            "trm_training_stood_down": True,
        },
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_game_traces"
    assert artifact["learned_encoder_transfer_helps"] is False
    assert artifact["positive_control_passed"] is False
    assert artifact["n_held_out_games"] == 0
    assert artifact["cross_game_state_reduction"] == 0.0
    assert artifact["cross_game_state_reduction_ci95"] == [0.0, 0.0]
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["blocked_reason"] == "insufficient_game_traces"
    assert exp.artifact_schema_errors(artifact) == []


def test_req_learn_4342_summary_bootstraps_across_held_out_games() -> None:
    """REQ-LEARN-4342-7: CI and gate are across held-out games."""

    rows = [
        {
            "held_out_game": "r11l",
            "level_index": 1,
            "states_uniform": 30,
            "states_guided": 10,
            "baseline_solved": True,
            "guided_solved": True,
        },
        {
            "held_out_game": "r11l",
            "level_index": 2,
            "states_uniform": 30,
            "states_guided": 10,
            "baseline_solved": True,
            "guided_solved": True,
        },
        {
            "held_out_game": "ls20",
            "level_index": 1,
            "states_uniform": 20,
            "states_guided": 10,
            "baseline_solved": True,
            "guided_solved": True,
        },
        {
            "held_out_game": "lp85",
            "level_index": 1,
            "states_uniform": 40,
            "states_guided": 20,
            "baseline_solved": True,
            "guided_solved": True,
        },
    ]

    summary = exp.summarize_state_reduction(rows, random_seed=7, n_resamples=2000)

    assert summary["positive_control_passed"] is True
    assert summary["n_held_out_games"] == 3
    assert summary["cross_game_state_reduction"] == pytest.approx(2.4)
    assert summary["cross_game_state_reduction_ci95"][0] > 1.0
    assert summary["learned_encoder_transfer_helps"] is True
    assert summary["per_held_out_game_reduction"]["r11l"]["state_reduction"] == pytest.approx(3.0)

    empty = exp.summarize_state_reduction([], random_seed=7, n_resamples=0)
    assert empty["positive_control_passed"] is False
    assert empty["n_held_out_games"] == 0


def test_req_learn_4342_complete_artifact_and_schema_gate() -> None:
    """REQ-LEARN-4342-6: complete artifacts keep gate fields bare."""

    artifact = exp.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 10,
                "states_guided": 10,
                "baseline_solved": True,
                "guided_solved": True,
            }
        ],
        split_specs={"r11l": {"train_games": ["ls20"], "held_out_game": "r11l"}},
        model_specs_by_held_out={"r11l": {"value_head": {"weights": [0.0, 0.0]}}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b"},
        preconditions_checked={"usable_trace_game_count": 2, "trm_training_stood_down": True},
        duration_s=0.5,
        n_resamples=10,
    )

    assert artifact["learned_encoder_transfer_helps"] is False
    assert artifact["honest_verdict"] == (
        "complete: action_role_encoder_transfer_no_improvement_positive_control_passed"
    )
    assert artifact["positive_control_passed"] is True
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == exp.GAP_ID
    assert exp.artifact_schema_errors(artifact) == []

    success = exp.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 30,
                "states_guided": 10,
                "baseline_solved": True,
                "guided_solved": True,
            },
            {
                "held_out_game": "ls20",
                "level_index": 1,
                "states_uniform": 20,
                "states_guided": 10,
                "baseline_solved": True,
                "guided_solved": True,
            },
            {
                "held_out_game": "lp85",
                "level_index": 1,
                "states_uniform": 40,
                "states_guided": 20,
                "baseline_solved": True,
                "guided_solved": True,
            },
        ],
        split_specs={"r11l": {"train_games": ["ls20"], "held_out_game": "r11l"}},
        model_specs_by_held_out={"r11l": {"value_head": {"weights": [0.0, 0.0]}}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b", "lp85": "sha256:c"},
        preconditions_checked={"usable_trace_game_count": 3, "trm_training_stood_down": True},
        duration_s=0.5,
        n_resamples=2000,
    )
    assert success["honest_verdict"].startswith("success:")
    assert success["learned_encoder_transfer_helps"] is True
    assert success["missing_verifier_gaps"] == []

    control_failed = exp.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 30,
                "states_guided": 10,
                "baseline_solved": False,
                "guided_solved": True,
            }
        ],
        split_specs={"r11l": {"train_games": ["ls20"], "held_out_game": "r11l"}},
        model_specs_by_held_out={"r11l": {"value_head": {"weights": [0.0, 0.0]}}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b"},
        preconditions_checked={"usable_trace_game_count": 2, "trm_training_stood_down": True},
        duration_s=0.5,
        n_resamples=10,
    )
    assert control_failed["honest_verdict"] == "complete: action_role_encoder_positive_control_failed"

    bad = dict(artifact)
    bad["learned_encoder_transfer_helps"] = 1
    bad["positive_control_passed"] = "true"
    bad["cross_game_state_reduction"] = "1.0"
    bad["cross_game_state_reduction_ci95"] = {"lo": 1.0, "hi": 1.0}
    bad["n_held_out_games"] = True
    bad["verifier_is_oracle"] = True
    bad["preconditions_checked"] = []
    bad["random_seed"] = "4342"

    errors = exp.artifact_schema_errors(bad)

    for field in (
        "learned_encoder_transfer_helps",
        "positive_control_passed",
        "cross_game_state_reduction",
        "cross_game_state_reduction_ci95",
        "n_held_out_games",
        "verifier_is_oracle",
        "preconditions_checked",
        "random_seed",
    ):
        assert any(field in error for error in errors)

    malformed = {
        "honest_verdict": None,
        "learned_encoder_transfer_helps": True,
        "positive_control_passed": False,
        "cross_game_state_reduction": 1.0,
        "cross_game_state_reduction_ci95": [0.5, 1.5],
        "n_held_out_games": 3,
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4342,
        "reproducibility_checksum": 0,
        "model_specs": [],
        "field_principles": {"honest_verdict": "wrong"},
    }
    malformed_errors = exp.artifact_schema_errors(malformed)
    missing_errors = exp.artifact_schema_errors({})

    assert any("missing required field" in error for error in missing_errors)
    assert any("honest_verdict must be a string" in error for error in malformed_errors)
    assert any("reproducibility_checksum must be a string" in error for error in malformed_errors)
    assert any("model_specs must be an object" in error for error in malformed_errors)
    assert any("field_principles mismatch" in error for error in malformed_errors)
    assert any("requires CI95 lower bound > 1.0" in error for error in malformed_errors)
    assert any("requires reduction > 1.0" in error for error in malformed_errors)
    assert any("requires positive_control_passed=true" in error for error in malformed_errors)


def test_req_learn_4342_runner_writes_artifact(monkeypatch, tmp_path: Path) -> None:
    """REQ-LEARN-4342-6: runner writes the artifact and verifier report."""

    fake_artifact = exp.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 10,
                "states_guided": 10,
                "baseline_solved": True,
                "guided_solved": True,
            }
        ],
        split_specs={"r11l": {"train_games": ["ls20"], "held_out_game": "r11l"}},
        model_specs_by_held_out={"r11l": {"value_head": {"weights": [0.0, 0.0]}}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b"},
        preconditions_checked={"usable_trace_game_count": 2, "trm_training_stood_down": True},
        duration_s=0.5,
        n_resamples=10,
    )
    monkeypatch.setattr(exp, "evaluate_leave_one_game_out", lambda _repo: fake_artifact)
    monkeypatch.setattr(exp, "run_adversarial_verify", lambda _repo, _artifact: {"status": "clean"})

    artifact = exp.run(repo=tmp_path, write=True)

    written = tmp_path / exp.OUTPUT_REL
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
    assert artifact["adversarial_verify"] == {"status": "clean"}

    monkeypatch.setattr(exp, "evaluate_leave_one_game_out", lambda _repo: {"honest_verdict": None})
    with pytest.raises(ValueError, match="missing required field"):
        exp.run(repo=tmp_path, write=False)


def test_results_wrapper_imports_main() -> None:
    """SCENARIO-LEARN-4342: results wrapper exposes the stable CLI entrypoint."""

    namespace = runpy.run_path(str(WRAPPER_PATH), run_name="exp4342_wrapper_test")

    assert namespace["main"] is exp.main

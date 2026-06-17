"""Tests for Exp 4331 learned frame encoder cross-game ARC transfer.

Spec refs: REQ-LEARN-4331, SCENARIO-LEARN-4331.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer as exp
from carnot.agentic.arc_value_learner import LearnedVerifier


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
WRAPPER_PATH = REPO / "results" / "experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.py"


def test_req_learn_4331_spec_declares_learned_encoder_contract() -> None:
    """REQ-LEARN-4331: OpenSpec declares the learned-encoder artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    normalized_spec = " ".join(spec.split())

    for marker in (
        "REQ-LEARN-4331",
        "SCENARIO-LEARN-4331",
        "SCENARIO-LEARN-4331-BLOCKED",
        "experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.json",
        "python/carnot/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.py",
        "learned_encoder_transfer_helps",
        "blocked_insufficient_solve_traces",
        "learned convolutional feature map",
        "game-invariant ARC value",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.FIELD_PRINCIPLES.values():
        assert " ".join(principle.split()) in normalized_spec


def test_req_learn_4331_frame_tensor_encodes_raw_frames() -> None:
    """REQ-LEARN-4331-4: raw frames become fixed one-hot encoder tensors."""

    grid = np.array([[0, 1], [2, 15]], dtype=np.int16)
    tensor = exp.frame_to_tensor(grid, size=4, n_colors=16)

    assert tensor.shape == (16, 4, 4)
    assert tensor[:, 0, 0].sum() == pytest.approx(1.0)
    assert tensor[0, 0, 0] == pytest.approx(1.0)
    assert tensor[1, 0, 1] == pytest.approx(1.0)
    assert tensor[2, 1, 0] == pytest.approx(1.0)
    assert tensor[15, 1, 1] == pytest.approx(1.0)
    assert tensor[:, 3, 3].sum() == pytest.approx(1.0)

    from_list = exp.frame_to_tensor([[1]], size=2, n_colors=4)
    assert from_list[1, 0, 0] == pytest.approx(1.0)

    class FrameObject:
        frame = np.array([[[0, 2], [3, 1]]], dtype=np.int16)

    from_object = exp.frame_to_tensor(FrameObject(), size=2, n_colors=4)
    assert from_object[0, 0, 0] == pytest.approx(1.0)
    assert from_object[2, 0, 1] == pytest.approx(1.0)


def test_req_learn_4331_tiny_encoder_trains_cpu_embedding() -> None:
    """REQ-LEARN-4331-4: the learned encoder trains and emits embeddings."""

    frames = [
        np.zeros((4, 4), dtype=np.int16),
        np.ones((4, 4), dtype=np.int16),
        np.full((4, 4), 2, dtype=np.int16),
        np.full((4, 4), 3, dtype=np.int16),
    ]
    targets = [4.0, 3.0, 2.0, 1.0]
    encoder = exp.LearnedFrameEncoder(embedding_dim=5, epochs=8, seed=123).fit(frames, targets)

    embedding = encoder.transform_grid(frames[0])

    assert len(embedding) == 5
    assert all(np.isfinite(embedding))
    assert encoder.featurize(frames[1]) == encoder.transform_grid(frames[1])
    assert encoder.n_samples == 4
    assert encoder.model_summary()["architecture"] == "tiny_cpu_cnn_frame_encoder"
    assert encoder.model_summary()["training_compute"] == "CPU numpy PCA/ridge"

    with pytest.raises(ValueError, match="no frames"):
        exp.LearnedFrameEncoder().fit([], [])
    with pytest.raises(ValueError, match="untrained"):
        exp.LearnedFrameEncoder().transform_grid(frames[0])
    with pytest.raises(ValueError, match="untrained"):
        exp.LearnedFrameEncoder()._conv_pool_features(np.zeros((16, 4, 4), dtype=np.float32))

    head = LearnedVerifier(lambda _frame: [0.0])
    assert exp._round_weights(head) == []
    head.w = np.array([1.2345678912345, -2.0])
    assert exp._round_weights(head) == [1.234567891234, -2.0]


def test_scenario_learn_4331_blocked_artifact_is_terminal_and_bare() -> None:
    """SCENARIO-LEARN-4331-BLOCKED: insufficient traces fail closed."""

    artifact = exp.build_blocked_artifact(
        usable_games=["r11l", "ls20"],
        missing_games=["wa30", "lp85"],
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_solve_traces"
    assert artifact["learned_encoder_transfer_helps"] is False
    assert artifact["baseline_solves_held_out"] is False
    assert artifact["cross_game_state_reduction"] == 0.0
    assert artifact["cross_game_state_reduction_ci95"] == [0.0, 0.0]
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["blocked_reason"] == "insufficient_solve_traces"
    assert artifact["model_specs"]["llm_weight_mutation"] is False
    assert exp.artifact_schema_errors(artifact) == []


def test_req_learn_4331_bootstrap_summary_uses_learned_encoder_gate() -> None:
    """REQ-LEARN-4331: transfer helps only when reduction and CI lower exceed 1."""

    rows = [
        {
            "held_out_game": "r11l",
            "level_index": 1,
            "states_uniform": 30,
            "states_transferred": 10,
            "baseline_solved": True,
            "transferred_solved": True,
        },
        {
            "held_out_game": "ls20",
            "level_index": 1,
            "states_uniform": 20,
            "states_transferred": 10,
            "baseline_solved": True,
            "transferred_solved": True,
        },
        {
            "held_out_game": "lp85",
            "level_index": 1,
            "states_uniform": 40,
            "states_transferred": 20,
            "baseline_solved": True,
            "transferred_solved": True,
        },
    ]

    summary = exp.summarize_state_reduction(rows, random_seed=7, n_resamples=2000)

    assert summary["baseline_solves_held_out"] is True
    assert summary["cross_game_state_reduction"] == pytest.approx(2.25)
    assert summary["cross_game_state_reduction_ci95"][0] > 1.0
    assert summary["learned_encoder_transfer_helps"] is True
    assert summary["per_held_out_game_reduction"]["r11l"]["state_reduction"] == pytest.approx(3.0)

    failed_control = exp.summarize_state_reduction([], random_seed=7, n_resamples=0)
    assert failed_control["baseline_solves_held_out"] is False
    assert failed_control["cross_game_state_reduction"] == 0.0
    assert failed_control["cross_game_state_reduction_ci95"] == [0.0, 0.0]


def test_req_learn_4331_schema_rejects_non_bare_gate_fields() -> None:
    """REQ-LEARN-4331-6: required gate fields stay bare and oracle-distinct."""

    artifact = exp.build_blocked_artifact(
        usable_games=[],
        missing_games=["r11l", "ls20", "wa30"],
        duration_s=0.0,
    )
    bad = dict(artifact)
    bad["learned_encoder_transfer_helps"] = 1
    bad["baseline_solves_held_out"] = "false"
    bad["cross_game_state_reduction"] = "0.0"
    bad["cross_game_state_reduction_ci95"] = {"lo": 0.0, "hi": 0.0}
    bad["verifier_is_oracle"] = True
    bad["random_seed"] = "4331"

    errors = exp.artifact_schema_errors(bad)

    for field in (
        "learned_encoder_transfer_helps",
        "baseline_solves_held_out",
        "cross_game_state_reduction",
        "cross_game_state_reduction_ci95",
        "verifier_is_oracle",
        "random_seed",
    ):
        assert any(field in error for error in errors)


def test_req_learn_4331_complete_artifact_logs_sharpened_gap_on_null() -> None:
    """REQ-LEARN-4331-7: a powered null records the small-encoder insufficiency."""

    artifact = exp.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 10,
                "states_transferred": 10,
                "baseline_solved": True,
                "transferred_solved": True,
            }
        ],
        split_specs={"r11l": {"train_games": ["ls20", "wa30"], "held_out_game": "r11l"}},
        model_specs_by_held_out={"r11l": {"n_samples": 4, "value_head_weights": [0.0, 0.0]}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b", "wa30": "sha256:c"},
        duration_s=0.5,
        n_resamples=2000,
    )

    assert artifact["learned_encoder_transfer_helps"] is False
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == exp.GAP_ID
    assert "small learned frame encoder" in artifact["missing_verifier_gaps"][0]["failure_mode"]
    assert exp.artifact_schema_errors(artifact) == []

    success = exp.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 30,
                "states_transferred": 10,
                "baseline_solved": True,
                "transferred_solved": True,
            },
            {
                "held_out_game": "ls20",
                "level_index": 1,
                "states_uniform": 20,
                "states_transferred": 10,
                "baseline_solved": True,
                "transferred_solved": True,
            },
            {
                "held_out_game": "lp85",
                "level_index": 1,
                "states_uniform": 40,
                "states_transferred": 20,
                "baseline_solved": True,
                "transferred_solved": True,
            },
        ],
        split_specs={"r11l": {"train_games": ["ls20"], "held_out_game": "r11l"}},
        model_specs_by_held_out={"r11l": {"n_samples": 4, "value_head_weights": [0.0, 0.0]}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b", "lp85": "sha256:c"},
        duration_s=0.5,
        n_resamples=2000,
    )
    assert success["honest_verdict"].startswith("success:")
    assert success["learned_encoder_transfer_helps"] is True
    assert success["missing_verifier_gaps"] == []
    assert exp.artifact_schema_errors(success) == []

    control_failed = exp.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 10,
                "states_transferred": 5,
                "baseline_solved": False,
                "transferred_solved": True,
            }
        ],
        split_specs={"r11l": {"train_games": ["ls20"], "held_out_game": "r11l"}},
        model_specs_by_held_out={"r11l": {"n_samples": 4, "value_head_weights": [0.0, 0.0]}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b"},
        duration_s=0.5,
        n_resamples=10,
    )
    assert control_failed["honest_verdict"] == "complete: learned_frame_encoder_positive_control_failed"
    assert control_failed["baseline_solves_held_out"] is False


def test_req_learn_4331_runner_writes_result_and_gap_on_null(monkeypatch, tmp_path: Path) -> None:
    """REQ-LEARN-4331-7: null transfer writes artifact and sharpens the gap."""

    fake_artifact = exp.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 10,
                "states_transferred": 10,
                "baseline_solved": True,
                "transferred_solved": True,
            }
        ],
        split_specs={"r11l": {"train_games": ["ls20", "wa30"], "held_out_game": "r11l"}},
        model_specs_by_held_out={"r11l": {"n_samples": 4, "value_head_weights": [0.0, 0.0]}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b", "wa30": "sha256:c"},
        duration_s=0.5,
        n_resamples=2000,
    )
    monkeypatch.setattr(exp, "evaluate_leave_one_game_out", lambda _repo: fake_artifact)
    monkeypatch.setattr(exp, "run_adversarial_verify", lambda _repo, _artifact: {"status": "clean"})

    artifact = exp.run(repo=tmp_path, write=True)

    written = tmp_path / exp.OUTPUT_REL
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
    assert artifact["adversarial_verify"] == {"status": "clean"}
    assert exp.GAP_ID in gaps
    assert "small learned frame encoder over the current solved set is insufficient" in gaps

    exp.ensure_gap_logged(tmp_path, artifact)
    assert (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8") == gaps
    exp.ensure_gap_logged(tmp_path, {**artifact, "learned_encoder_transfer_helps": True})
    exp.ensure_gap_logged(tmp_path, {**artifact, "baseline_solves_held_out": False})


def test_req_learn_4331_schema_covers_error_branches(monkeypatch, tmp_path: Path) -> None:
    """REQ-LEARN-4331-6: malformed artifacts and bad runner output fail closed."""

    malformed = {
        "honest_verdict": None,
        "learned_encoder_transfer_helps": True,
        "baseline_solves_held_out": False,
        "cross_game_state_reduction": 1.0,
        "cross_game_state_reduction_ci95": [0.5, 1.5],
        "verifier_is_oracle": False,
        "random_seed": 4331,
        "reproducibility_checksum": 0,
        "model_specs": [],
        "per_held_out_game_reduction": [],
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(malformed)
    missing_errors = exp.artifact_schema_errors({})

    assert any("missing required field" in error for error in missing_errors)
    assert any("honest_verdict must be a string" in error for error in errors)
    assert any("reproducibility_checksum must be a string" in error for error in errors)
    assert any("model_specs must be an object" in error for error in errors)
    assert any("per_held_out_game_reduction must be an object" in error for error in errors)
    assert any("field_principles mismatch" in error for error in errors)
    assert any("requires CI95 lower bound > 1.0" in error for error in errors)
    assert any("requires reduction > 1.0" in error for error in errors)
    assert any("requires baseline_solves_held_out=true" in error for error in errors)

    monkeypatch.setattr(exp, "evaluate_leave_one_game_out", lambda _repo: malformed)
    with pytest.raises(ValueError, match="honest_verdict must be a string"):
        exp.run(repo=tmp_path, write=False)


def test_results_wrapper_imports_main() -> None:
    """SCENARIO-LEARN-4331: results wrapper exposes the stable CLI entrypoint."""

    namespace = runpy.run_path(str(WRAPPER_PATH), run_name="exp4331_wrapper_test")

    assert namespace["main"] is exp.main

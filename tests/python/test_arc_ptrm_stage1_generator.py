"""Tests for Exp 5574 PTRM Stage-1 ARC action-sequence generator.

Spec refs: REQ-ARC-PTRM-5574-1, REQ-ARC-PTRM-5574-2,
REQ-ARC-PTRM-5574-3, REQ-ARC-PTRM-5574-4, REQ-ARC-PTRM-5574-5,
SCENARIO-ARC-PTRM-5574-DATASET, SCENARIO-ARC-PTRM-5574-STOCHASTIC,
SCENARIO-ARC-PTRM-5574-ARTIFACT.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from carnot.agentic import arc_ptrm_stage1_generator as ptrm
from carnot.agentic.arc_ptrm_stage1_generator import (
    CarnotTrajectoryVerifier,
    REQUIRED_ARTIFACT_FIELDS,
    Stage1Config,
    Stage1Example,
    Stage1InputBatch,
    build_stage1_artifact,
    build_stage1_dataset,
    checkpoint_sha256,
    collect_preconditions,
    generate_trajectories,
    normalize_action,
    run_experiment_5574,
    select_trajectory,
    validate_stage1_artifact,
)


def _row(env: str, guid: str, step: int, action: object, progress: float) -> dict:
    return {
        "env": env,
        "guid": guid,
        "step_index": step,
        "action": action,
        "frame": [[step % 3, step % 5], [1, 2]],
        "frame_delta": 0.1 * step,
        "level_progress": progress,
    }


def _won_rows(env: str, guid: str, n: int = 12) -> list[dict]:
    rows = []
    for step in range(1, n + 1):
        rows.append(
            _row(
                env,
                guid,
                step,
                {"id": str(step % 7), "data": {"x": step, "y": step + 1}},
                1.0 if step == n else float(step) / float(n + 1),
            )
        )
    return rows


def _staged_jsonl_row(env: str, guid: str, step: int, progress: float) -> dict:
    return {
        "action": {
            "data": {"x": step % 7, "y": (step + 1) % 7},
            "id": str(step % 7),
            "reasoning": None,
        },
        "env": env,
        "frame": [[step % 4, (step + 1) % 4], [(step + 2) % 4, (step + 3) % 4]],
        "frame_delta": 0.01 * step,
        "guid": guid,
        "level_progress": progress,
        "schema": "carnot.arc_human_replay.frame_action_delta.v1",
        "source_row_index": 0,
        "step_index": step,
    }


def _write_fake_corpus(root: Path) -> None:
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    rows = []
    for env, guid in (("train_game", "train-win"), ("held_game", "held-win")):
        for step in range(1, 18):
            rows.append(_staged_jsonl_row(env, guid, step, 1.0 if step == 17 else step / 20.0))
    shard = shard_dir / "train-00000.jsonl"
    shard.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": "carnot.arc_human_replay.frame_action_delta.v1",
        "example_count": len(rows),
        "shard_count": 1,
        "shards": [
            {
                "path": "shards/train-00000.jsonl",
                "rows": len(rows),
                "sha256": checkpoint_sha256(shard),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_openspec_declares_ptrm_stage1_contract() -> None:
    """REQ-ARC-PTRM-5574-5: OpenSpec names the Stage-1 artifact contract."""

    spec = Path("openspec/capabilities/arc-trm-generator/spec.md").read_text(encoding="utf-8")

    for ref in (
        "REQ-ARC-PTRM-5574-1",
        "REQ-ARC-PTRM-5574-2",
        "REQ-ARC-PTRM-5574-3",
        "REQ-ARC-PTRM-5574-4",
        "REQ-ARC-PTRM-5574-5",
        "SCENARIO-ARC-PTRM-5574-DATASET",
        "SCENARIO-ARC-PTRM-5574-STOCHASTIC",
        "SCENARIO-ARC-PTRM-5574-ARTIFACT",
    ):
        assert ref in spec


def test_req_arc_ptrm_5574_2_normalizes_replay_action_variants() -> None:
    """REQ-ARC-PTRM-5574-2: action ids and optional coordinates are stable."""

    assert normalize_action({"id": "ACTION6", "data": {"x": 9, "y": 4}}).as_tuple() == (6, 9, 4, 1)
    assert normalize_action(4).as_tuple() == (4, -1, -1, 0)
    assert normalize_action({"id": "RESET", "data": {}}).as_tuple() == (0, -1, -1, 0)
    assert normalize_action("3").as_tuple() == (3, -1, -1, 0)
    assert normalize_action({"action": "ACTION2", "data": {"x": "5", "y": "6"}}).as_tuple() == (
        2,
        5,
        6,
        1,
    )


def test_scenario_arc_ptrm_5574_dataset_has_no_heldout_leakage() -> None:
    """SCENARIO-ARC-PTRM-5574-DATASET: won K-window split excludes held-out games."""

    rows = [
        *_won_rows("train_game", "train-win", n=12),
        *_won_rows("held_game", "held-win", n=12),
        *[_row("train_game", "train-loss", step, {"id": "1"}, 0.0) for step in range(1, 12)],
    ]
    config = Stage1Config(
        sequence_length=8, history_length=3, max_train_windows=100, max_eval_windows=100
    )

    bundle = build_stage1_dataset(rows, config=config, heldout_games=("held_game",))

    assert bundle.leakage_count == 0
    assert bundle.won_session_count == 2
    assert {example.game for example in bundle.train_examples} == {"train_game"}
    assert {example.game for example in bundle.heldout_examples} == {"held_game"}
    assert all(
        len(example.target_actions) == 8
        for example in bundle.train_examples + bundle.heldout_examples
    )
    assert all(example.history_intent_vector[-1] > 0.0 for example in bundle.train_examples)


def test_req_arc_ptrm_5574_1_preconditions_record_sentinel_scope_and_cuda() -> None:
    """REQ-ARC-PTRM-5574-1: precondition receipts fail closed without CPU fallback."""

    preconditions = collect_preconditions(
        corpus_manifest=Path("data/arc_public_demo_human_replay_corpus/manifest.json"),
        sentinel_path=Path("results/trm_runs/DO_NOT_RELAUNCH"),
        min_free_disk_gb=0.001,
        require_cuda=False,
    )

    by_name = {row["resource"]: row for row in preconditions["checks"]}
    assert by_name["human_replay_manifest"]["available"] is True
    assert by_name["sudoku_do_not_relaunch_scope"]["available"] is True
    assert by_name["disk_budget"]["available"] is True
    assert preconditions["blocked"] is False
    assert preconditions["sentinel_applies_to_arc"] is False


def test_scenario_arc_ptrm_5574_stochastic_recursion_and_verifier_selection() -> None:
    """SCENARIO-ARC-PTRM-5574-STOCHASTIC: recursion yields selectable diversity."""

    example = Stage1Example(
        game="g1",
        guid="guid",
        start_step=1,
        frame_features=[0.1, 0.2, 0.3, 0.4],
        history_actions=[1, 2, 3],
        history_coords=[(1, 1), (2, 2), (3, 3)],
        history_intent_vector=[0.25, 0.5, 0.75, 1.0],
        target_actions=[1, 2, 2, 2, 1, 2, 2, 2],
        target_coords=[(1, 1)] * 8,
    )
    batch = Stage1InputBatch.from_examples([example], action_vocab_size=8)
    verifier = CarnotTrajectoryVerifier.from_sequences(
        [[1, 2, 2, 2, 1, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2, 2]],
        action_vocab_size=8,
    )

    trajectories = generate_trajectories(
        batch,
        action_vocab_size=8,
        sequence_length=8,
        max_depth=5,
        hidden_dim=12,
        trajectories_per_input=6,
        seed=5574,
        noise_std=1.2,
        verifier=verifier,
    )
    selected = select_trajectory(trajectories)

    assert len(trajectories) == 6
    assert len({tuple(row.action_ids) for row in trajectories}) > 1
    assert all(1 <= row.halting_depth <= 5 for row in trajectories)
    assert selected.verifier_score == max(row.verifier_score for row in trajectories)
    assert selected.verifier_score_source == "carnot_action_language_model"


def test_req_arc_ptrm_5574_3_energy_fallback_can_halt_early() -> None:
    """REQ-ARC-PTRM-5574-3: dynamic halting also works without verifier selection."""

    example = Stage1Example(
        game="g1",
        guid="guid",
        start_step=1,
        frame_features=[0.0, 0.0, 0.0, 0.0],
        history_actions=[1, 1, 1],
        history_coords=[(-1, -1), (-1, -1), (-1, -1)],
        history_intent_vector=[1.0, 0.0, 1.0, 0.0],
        target_actions=[1, 1, 1, 1],
        target_coords=[(-1, -1)] * 4,
    )
    batch = Stage1InputBatch.from_examples([example], action_vocab_size=2)

    trajectories = generate_trajectories(
        batch,
        action_vocab_size=2,
        sequence_length=4,
        max_depth=4,
        hidden_dim=8,
        trajectories_per_input=1,
        seed=1,
        noise_std=0.0,
    )

    assert trajectories[0].halting_depth < 4
    assert trajectories[0].verifier_score_source == "energy_fallback"


def test_req_arc_ptrm_5574_4_verifier_validates_human_over_corrupt() -> None:
    """REQ-ARC-PTRM-5574-4: verifier is oracle-distinct and independently checked."""

    verifier = CarnotTrajectoryVerifier.from_sequences(
        [[1, 1, 2, 2, 2, 3, 3, 3], [1, 1, 2, 2, 3, 3, 3, 3]],
        action_vocab_size=8,
    )
    validation = verifier.validate_against_corruptions(seed=11, n_trials=16)

    assert validation["verifier_is_oracle"] is False
    assert validation["pairwise_human_preferred_rate"] > 0.5
    assert verifier.score([1, 1, 2, 2, 2, 3, 3, 3]) > verifier.score([7, 7, 7, 7, 7, 7, 7, 7])

    empty_validation = CarnotTrajectoryVerifier(
        action_vocab_size=8,
        unigram_counts={},
        transition_counts={},
    ).validate_against_corruptions(seed=1, n_trials=1)
    short_validation = CarnotTrajectoryVerifier.from_sequences(
        [[1]],
        action_vocab_size=8,
    ).validate_against_corruptions(seed=2, n_trials=1)
    assert empty_validation["verifier_is_oracle"] is False
    assert short_validation["verifier_is_oracle"] is False


def test_scenario_arc_ptrm_5574_artifact_is_complete_and_principled(tmp_path: Path) -> None:
    """SCENARIO-ARC-PTRM-5574-ARTIFACT: JSON fields, principles, and checkpoint hash match."""

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"ptrm checkpoint")
    expected_hash = hashlib.sha256(b"ptrm checkpoint").hexdigest()

    artifact = build_stage1_artifact(
        preconditions={"blocked": False, "checks": []},
        prior_pilot_receipts=[{"id": "v4", "verdict": "heldout_null_missing_history_intent"}],
        dataset_hashes={"manifest_sha256": "abc", "path": tmp_path, "shards": []},
        heldout_games=["held_game"],
        leakage_count=0,
        model_architecture={"name": "PTRMActionSequenceGenerator"},
        parameter_count=123,
        stochastic_noise_schedule={"type": "gaussian", "std": 0.1, "per_recursion_step": True},
        trajectories_per_input=4,
        recursion_depth_metrics={"1": {"accuracy": 0.1, "energy": 1.0}},
        overthinking_curve=[{"depth": 1, "accuracy": 0.1, "energy": 1.0}],
        controls={
            "non_recursive": {"accuracy": 0.05},
            "deterministic_fixed_depth": {"accuracy": 0.07},
        },
        positive_control_passed=True,
        verifier_selection_method={
            "name": "carnot_action_language_model",
            "validation": {"rate": 0.75},
        },
        checkpoint_path=checkpoint,
        training_duration_s=2.5,
        gpu_device_receipt={"torch_cuda_available": True, "devices": ["NVIDIA GeForce RTX 3090"]},
        stage1_training_complete=True,
        loo_verdict_reached=False,
        heldout_generalization_signal="not_preregistered_verdict",
        retire_trm_generator_line=False,
        honest_verdict="complete: stage1_ptrm_substrate_trained_remaining_loo_gate_preserved",
    )

    assert checkpoint_sha256(checkpoint) == expected_hash
    assert artifact["checkpoint_sha256"] == expected_hash
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_level_solve_claim"] is True
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["inference_substrate"] == "trained_ptrm_offline_development_proxy"
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    validate_stage1_artifact(artifact)

    out = tmp_path / "experiment_5574_ptrm_stochastic_generator_stage1.json"
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8"))["checkpoint_sha256"] == expected_hash


def test_scenario_arc_ptrm_5574_artifact_validation_rejects_bad_claims(tmp_path: Path) -> None:
    """SCENARIO-ARC-PTRM-5574-ARTIFACT: validator rejects incomplete or dishonest JSON."""

    def valid_artifact() -> dict:
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_bytes(b"ptrm checkpoint")
        return build_stage1_artifact(
            preconditions={"blocked": False, "checks": []},
            prior_pilot_receipts=[],
            dataset_hashes={"manifest_sha256": "abc", "shards": []},
            heldout_games=[],
            leakage_count=0,
            model_architecture={"name": "PTRMActionSequenceGenerator"},
            parameter_count=1,
            stochastic_noise_schedule={"type": "gaussian", "std": 0.1},
            trajectories_per_input=1,
            recursion_depth_metrics={"1": {"accuracy": 0.0, "energy": 0.0}},
            overthinking_curve=[{"depth": 1, "accuracy": 0.0, "energy": 0.0}],
            controls={},
            positive_control_passed=True,
            verifier_selection_method={"name": "carnot_action_language_model"},
            checkpoint_path=checkpoint,
            training_duration_s=0.1,
            gpu_device_receipt={"torch_cuda_available": False, "devices": []},
            stage1_training_complete=True,
            loo_verdict_reached=False,
            heldout_generalization_signal="not_preregistered_verdict",
            retire_trm_generator_line=False,
            honest_verdict="complete: validation_fixture",
        )

    artifact = valid_artifact()
    artifact.pop("track")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_stage1_artifact(artifact)

    artifact = valid_artifact()
    artifact.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        validate_stage1_artifact(artifact)

    artifact = valid_artifact()
    artifact["field_principles"].pop("track")
    with pytest.raises(ValueError, match="missing field principles"):
        validate_stage1_artifact(artifact)

    for field, bad_value, match in (
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("no_level_solve_claim", False, "no_level_solve_claim"),
        ("solve_provenance", "live_solve", "solve_provenance"),
        ("inference_substrate", "hidden_game_runtime", "inference_substrate"),
    ):
        artifact = valid_artifact()
        artifact[field] = bad_value
        with pytest.raises(ValueError, match=match):
            validate_stage1_artifact(artifact)

    artifact = valid_artifact()
    Path(artifact["checkpoint_path"]).write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_stage1_artifact(artifact)


def test_req_arc_ptrm_5574_1_blocked_runner_and_cuda_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-PTRM-5574-1: blocked runs emit no-fallback artifacts and CUDA checks work."""

    output = tmp_path / "blocked.json"
    artifact = run_experiment_5574(
        output_path=output,
        corpus_dir=tmp_path / "missing_corpus",
        run_dir=tmp_path / "blocked_run",
        config=Stage1Config(heldout_games=("held_game",)),
        require_cuda=False,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert loaded["honest_verdict"].startswith("blocked_")
    assert loaded["stage1_training_complete"] is False
    assert Path(loaded["checkpoint_path"]).exists()
    validate_stage1_artifact(loaded)

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ptrm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ptrm.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(ptrm.torch.cuda, "get_device_name", lambda index: "NVIDIA GeForce RTX 3090")
    preconditions = collect_preconditions(
        corpus_manifest=manifest,
        sentinel_path=tmp_path / "absent_sentinel",
        min_free_disk_gb=0.0,
        require_cuda=True,
    )
    cuda_check = {row["resource"]: row for row in preconditions["checks"]}["cuda_3090_class"]
    assert preconditions["blocked"] is False
    assert cuda_check["available"] is True


def test_req_arc_ptrm_5574_2_empty_frames_and_empty_train_control() -> None:
    """REQ-ARC-PTRM-5574-2: degenerate frame/train inputs fail closed."""

    model = ptrm.PTRMActionSequenceGenerator(
        history_length=1,
        sequence_length=2,
        action_vocab_size=2,
        hidden_dim=4,
    )

    assert ptrm._frame_features(None) == [0.0, 0.0, 0.0, 0.0]
    assert ptrm._frame_features([[]]) == [0.0, 0.0, 0.0, 0.0]
    assert ptrm._train_proxy_model(model, [], Stage1Config(), 2) == {
        "loss": 0.0,
        "train_token_accuracy": 0.0,
    }
    verifier = CarnotTrajectoryVerifier.from_sequences([[1, 1]], action_vocab_size=2)
    assert ptrm._recursion_metrics([], Stage1Config(), 2, verifier) == {
        "1": {"accuracy": 0.0, "energy": 0.0}
    }


def test_req_arc_ptrm_5574_5_runner_writes_tiny_valid_artifact(tmp_path: Path) -> None:
    """REQ-ARC-PTRM-5574-5: runner writes checkpointed JSON on a tiny staged corpus."""

    corpus_dir = tmp_path / "corpus"
    run_dir = tmp_path / "run"
    output = tmp_path / "experiment_5574_ptrm_stochastic_generator_stage1.json"
    _write_fake_corpus(corpus_dir)
    config = Stage1Config(
        sequence_length=8,
        history_length=3,
        max_depth=2,
        hidden_dim=16,
        trajectories_per_input=2,
        max_train_windows=8,
        max_eval_windows=4,
        batch_size=4,
        epochs=1,
        heldout_games=("held_game",),
    )

    artifact = run_experiment_5574(
        output_path=output,
        corpus_dir=corpus_dir,
        run_dir=run_dir,
        config=config,
        require_cuda=False,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    validate_stage1_artifact(loaded)
    assert loaded == artifact
    assert loaded["stage1_training_complete"] is True
    assert loaded["leakage_count"] == 0
    assert loaded["heldout_games"] == ["held_game"]
    assert Path(loaded["checkpoint_path"]).exists()
    assert loaded["checkpoint_sha256"] == checkpoint_sha256(Path(loaded["checkpoint_path"]))

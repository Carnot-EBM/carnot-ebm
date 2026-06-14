"""Tests for Exp 4211 synchronous verifier-as-reward finish.

Spec refs: REQ-CODE-4211, SCENARIO-CODE-4211-BLOCKED-PRECONDITION,
SCENARIO-CODE-4211-SYNC-ACCUMULATE, SCENARIO-CODE-4211-VERDICT-GATES.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from carnot import experiment_4211_verifier_as_reward_finish_synchronous as exp4211


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "code-verification" / "spec.md"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _stable_checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "code_verifier_reward_lora_rft_a83b52882c198954"
    corpora = root / "corpora"
    rows_a = [
        {"task_id": "t0", "prompt": "p0", "completion": "a0", "arm": "A_certified", "hidden_pass": True},
        {"task_id": "t1", "prompt": "p1", "completion": "a1", "arm": "A_certified", "hidden_pass": True},
    ]
    rows_b = [
        {
            "task_id": "t0",
            "prompt": "p0",
            "completion": "b0",
            "arm": "B_random_same_generator",
            "hidden_pass": False,
        },
        {
            "task_id": "t1",
            "prompt": "p1",
            "completion": "b1",
            "arm": "B_random_same_generator",
            "hidden_pass": False,
        },
    ]
    rows_c = [
        {"task_id": "t0", "prompt": "p0", "completion": "c0", "arm": "C_hidden_gold", "hidden_pass": True},
        {"task_id": "t1", "prompt": "p1", "completion": "c1", "arm": "C_hidden_gold", "hidden_pass": True},
    ]
    _write_jsonl(corpora / "arm_A.jsonl", rows_a)
    _write_jsonl(corpora / "arm_B.jsonl", rows_b)
    _write_jsonl(corpora / "arm_C.jsonl", rows_c)
    (root / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "arm_corpus_sizes": {"A": 2, "B": 2, "C": 2, "D": 0},
                "model_specs": {
                    "trainable_base": "google/gemma-4-E4B-it",
                    "trainable_base_is_non_qwen": True,
                    "on_policy_generator": "google/gemma-4-E4B-it",
                },
                "operating_point": {"base_passrate": 0.6, "max_new_tokens": 512, "truncation_rate": 0.0},
                "reproducibility_checksum": "sha256:a83b52882c198954",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_req_code_4211_spec_declares_artifact_contract() -> None:
    """REQ-CODE-4211: OpenSpec declares the synchronous finish contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-4211" in spec
    assert "SCENARIO-CODE-4211-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CODE-4211-SYNC-ACCUMULATE" in spec
    assert "SCENARIO-CODE-4211-VERDICT-GATES" in spec
    for field in exp4211.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4211.FIELD_PRINCIPLES


def test_scenario_code_4211_blocked_no_cached_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-CODE-4211-BLOCKED-PRECONDITION: missing base stops before training."""

    stable = _stable_checkpoint(tmp_path)
    called = False

    def fail_train(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("training must not run without a cached base")

    monkeypatch.setattr(exp4211, "find_cached_nonqwen_base", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exp4211, "_cuda_is_available", lambda: True)

    artifact = exp4211.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        train_callback=fail_train,
    )

    assert artifact["honest_verdict"] == "blocked_no_nonqwen_base_cached"
    assert artifact["verifier_label_carries_signal"] is False
    assert artifact["a_vs_b_delta"] is None
    assert artifact["a_vs_b_ci95"] is None
    assert artifact["preconditions"]["nonqwen_base_cached"] is False
    assert called is False


def test_scenario_code_4211_cuda_blocker_is_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-CODE-4211-BLOCKED-PRECONDITION: CUDA failure is terminal."""

    stable = _stable_checkpoint(tmp_path)
    monkeypatch.setattr(
        exp4211,
        "find_cached_nonqwen_base",
        lambda *_args, **_kwargs: exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
    )
    monkeypatch.setattr(exp4211, "_cuda_is_available", lambda: False)

    artifact = exp4211.run(output_path=tmp_path / "out.json", stable_checkpoint_path=stable)

    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["positive_control_confirmed"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions"]["cuda_available"] is False


def test_scenario_code_4211_sync_accumulate_calls_in_process_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CODE-4211-SYNC-ACCUMULATE: runner resumes without detach."""

    stable = _stable_checkpoint(tmp_path)
    captured: dict[str, object] = {}

    def fake_train(context: exp4211.TrainingContext) -> exp4211.TrainingOutcome:
        captured["context"] = context
        return exp4211.TrainingOutcome(
            status="partial",
            per_arm={
                "A": {"status": "trained", "completed_steps": 2},
                "B": {"status": "trained", "completed_steps": 2},
                "C": {"status": "trained", "completed_steps": 2},
            },
            accumulated_train_examples={"A": 2, "B": 2, "C": 2, "D": 0},
            runner_artifact_path=stable / "runner_artifact.json",
            progress_events=[
                {"arm": "A", "step": 1, "loss": 1.25},
                {"arm": "B", "step": 1, "loss": 1.5},
            ],
            used_detached_process=False,
        )

    monkeypatch.setattr(
        exp4211,
        "find_cached_nonqwen_base",
        lambda *_args, **_kwargs: exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
    )
    monkeypatch.setattr(exp4211, "_cuda_is_available", lambda: True)

    artifact = exp4211.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        train_callback=fake_train,
    )

    assert isinstance(captured["context"], exp4211.TrainingContext)
    assert captured["context"].mode == "in_process"
    assert captured["context"].stable_checkpoint_path == stable
    assert artifact["honest_verdict"] == "progress: accumulating_verifier_reward_training_no_eval_yet"
    assert artifact["training"]["used_detached_process"] is False
    assert artifact["accumulated_n"]["train_A"] == 2
    assert artifact["accumulated_n"]["train_B"] == 2
    assert artifact["accumulated_n"]["train_C"] == 2


def test_scenario_code_4211_verdict_gate_label_carries_signal() -> None:
    """SCENARIO-CODE-4211-VERDICT-GATES: A-vs-B is bare and CI-gated."""

    evaluation = exp4211.EvaluationOutcome(
        status="complete",
        pass_at_1={"A": 1.0, "B": 0.0, "C": 0.75, "D": 0.5},
        truncation_rate={"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
        task_rows=[
            {"task_id": "t0", "A": True, "B": False, "C": True, "D": False},
            {"task_id": "t1", "A": True, "B": False, "C": True, "D": True},
            {"task_id": "t2", "A": True, "B": False, "C": False, "D": False},
            {"task_id": "t3", "A": True, "B": False, "C": True, "D": True},
        ],
        seeds=[0, 1, 2],
        bootstrap_resamples=2000,
    )

    artifact = exp4211.build_result_artifact(
        preconditions={"nonqwen_base_cached": True, "cuda_available": True},
        stable_checkpoint_path=Path("/tmp/stable"),
        manifest={
            "model_specs": {"trainable_base": "google/gemma-4-E4B-it"},
            "operating_point": {"base_passrate": 0.5},
            "reproducibility_checksum": "sha256:abc",
        },
        corpus_sizes={"A": 4, "B": 4, "C": 4, "D": 0},
        cached_base=exp4211.CachedBase("google/gemma-4-E4B-it", Path("/tmp/model")),
        training=exp4211.TrainingOutcome(
            status="complete",
            per_arm={},
            accumulated_train_examples={"A": 4, "B": 4, "C": 4, "D": 0},
            runner_artifact_path=Path("/tmp/stable/runner_artifact.json"),
            progress_events=[],
            used_detached_process=False,
        ),
        evaluation=evaluation,
        adversarial_report=None,
        random_seed=4211,
        duration_s=65.0,
    )

    assert artifact["positive_control_confirmed"] is True
    assert artifact["a_vs_b_delta"] == pytest.approx(1.0)
    assert artifact["a_vs_b_ci95"] == [1.0, 1.0]
    assert artifact["verifier_label_carries_signal"] is True
    assert artifact["honest_verdict"] == "complete: verifier_label_carries_signal"
    assert artifact["a_vs_c_delta"] == pytest.approx(0.25)
    assert artifact["a_vs_d_delta"] == pytest.approx(0.5)


def test_scenario_code_4211_invalid_controls_stop_headline() -> None:
    """SCENARIO-CODE-4211-VERDICT-GATES: failed positive control invalidates A-vs-B."""

    evaluation = exp4211.EvaluationOutcome(
        status="complete",
        pass_at_1={"A": 0.75, "B": 0.25, "C": 0.4, "D": 0.5},
        truncation_rate={"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
        task_rows=[
            {"task_id": "t0", "A": True, "B": False, "C": False, "D": True},
            {"task_id": "t1", "A": True, "B": False, "C": True, "D": False},
            {"task_id": "t2", "A": True, "B": False, "C": False, "D": False},
            {"task_id": "t3", "A": False, "B": True, "C": True, "D": True},
        ],
        seeds=[0, 1, 2],
        bootstrap_resamples=2000,
    )

    artifact = exp4211.build_result_artifact(
        preconditions={"nonqwen_base_cached": True, "cuda_available": True},
        stable_checkpoint_path=Path("/tmp/stable"),
        manifest={"model_specs": {}, "operating_point": {"base_passrate": 0.5}},
        corpus_sizes={"A": 2, "B": 2, "C": 2, "D": 0},
        cached_base=exp4211.CachedBase("google/gemma-4-E4B-it", Path("/tmp/model")),
        training=exp4211.TrainingOutcome(
            status="complete",
            per_arm={},
            accumulated_train_examples={"A": 2, "B": 2, "C": 2, "D": 0},
            runner_artifact_path=Path("/tmp/stable/runner_artifact.json"),
            progress_events=[],
            used_detached_process=False,
        ),
        evaluation=evaluation,
        adversarial_report=None,
        random_seed=4211,
        duration_s=65.0,
    )

    assert artifact["honest_verdict"].startswith("invalid: TRAINING_INVALID")
    assert artifact["positive_control_confirmed"] is False
    assert artifact["verifier_label_carries_signal"] is False
    assert artifact["a_vs_b_delta"] == pytest.approx(0.5)


def test_req_code_4211_result_script_delegates_to_module() -> None:
    """REQ-CODE-4211: requested results script remains executable by path."""

    script = REPO / "results" / "experiment_4211_verifier_as_reward_finish_synchronous.py"
    spec = importlib.util.spec_from_file_location("exp4211_result_script", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.main is exp4211.main

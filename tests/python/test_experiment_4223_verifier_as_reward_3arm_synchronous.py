"""Tests for Exp 4223 B1-gated verifier-as-reward 3-arm finish.

Spec refs: REQ-CODE-4223, SCENARIO-CODE-4223-DEFERRED-HARNESS,
SCENARIO-CODE-4223-SYNC-ACCUMULATE, SCENARIO-CODE-4223-VERDICT-GATES.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from carnot import experiment_4211_verifier_as_reward_finish_synchronous as exp4211
from carnot import experiment_4223_verifier_as_reward_3arm_synchronous as exp4223


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "code-verification" / "spec.md"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _stable_checkpoint(tmp_path: Path, *, n_a: int = 2, n_b: int = 2, n_c: int = 2) -> Path:
    root = tmp_path / "code_verifier_reward_lora_rft_a83b52882c198954"
    corpora = root / "corpora"
    for arm, n_rows in (("A", n_a), ("B", n_b), ("C", n_c)):
        rows = [
            {
                "task_id": f"HumanEval/{index}",
                "prompt": f"Complete task {index}.",
                "completion": f"def f_{arm.lower()}_{index}(x):\n    return x\n",
                "hidden_pass": arm != "B",
            }
            for index in range(n_rows)
        ]
        _write_jsonl(corpora / f"arm_{arm}.jsonl", rows)
    (root / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "arm_corpus_sizes": {"A": n_a, "B": n_b, "C": n_c, "D": 0},
                "model_specs": {
                    "generation_checkpoint": str(tmp_path / "generation.checkpoint.json"),
                    "lora_config": {
                        "method": "LoRA-SFT",
                        "r": 16,
                        "target_modules": ["linear"],
                    },
                    "on_policy_generator": "google/gemma-4-E4B-it",
                    "qwen_train_base_forbidden": True,
                    "trainable_base": "google/gemma-4-E4B-it",
                    "trainable_base_is_non_qwen": True,
                },
                "operating_point": {"base_passrate": 0.6, "truncation_rate": 0.0},
                "reproducibility_checksum": "sha256:old",
                "youden_j": 0.4137931034482759,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def _b1_smoke(path: Path, *, passed: bool = True) -> Path:
    payload = {
        "harness_smoke_passed": passed,
        "lora_attach_path": "wrapper_inner_linear_target_modules" if passed else "",
        "model_specs": {
            "lora_attach_path": "wrapper_inner_linear_target_modules" if passed else "",
            "lora_config": {
                "method": "LoRA-SFT",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "learning_rate": 0.0002,
                "max_length": 1024,
                "target_modules": [
                    "q_proj.linear",
                    "k_proj.linear",
                    "v_proj.linear",
                    "o_proj.linear",
                    "gate_proj.linear",
                    "up_proj.linear",
                    "down_proj.linear",
                ],
                "exclude_modules": ["vision_tower"],
            },
            "on_policy_generator": "google/gemma-4-E4B-it",
            "qwen_train_base_forbidden": True,
            "trainable_base": "google/gemma-4-E4B-it",
            "trainable_base_is_non_qwen": True,
        },
        "random_seed": 4198,
        "reproducibility_checksum": "sha256:b1",
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_req_code_4223_spec_declares_b1_gated_contract() -> None:
    """REQ-CODE-4223: OpenSpec names the B1-gated synchronous finish contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-4223" in spec
    assert "SCENARIO-CODE-4223-DEFERRED-HARNESS" in spec
    assert "SCENARIO-CODE-4223-SYNC-ACCUMULATE" in spec
    assert "SCENARIO-CODE-4223-VERDICT-GATES" in spec
    for field in exp4223.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4223.FIELD_PRINCIPLES


def test_scenario_code_4223_deferred_when_b1_not_smoked(tmp_path: Path) -> None:
    """SCENARIO-CODE-4223-DEFERRED-HARNESS: failed B1 smoke stops before CUDA/training."""

    stable = _stable_checkpoint(tmp_path)
    smoke = _b1_smoke(tmp_path / "smoke.json", passed=False)
    called = False

    def fail_train(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("training must not run without a passed B1 smoke")

    artifact = exp4223.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        harness_smoke_path=smoke,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
        train_callback=fail_train,
    )

    assert artifact["honest_verdict"] == "complete_verifier_reward_deferred_harness_not_smoked"
    assert artifact["verifier_label_carries_signal"] is False
    assert artifact["a_vs_b_delta"] is None
    assert artifact["a_vs_b_ci95"] is None
    assert artifact["preconditions"]["b1_harness_smoke_passed"] is False
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert called is False


def test_scenario_code_4223_missing_or_malformed_b1_smoke_is_deferred(tmp_path: Path) -> None:
    """SCENARIO-CODE-4223-DEFERRED-HARNESS: unreadable/malformed B1 metadata is terminal."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps({"harness_smoke_passed": True, "lora_attach_path": "x", "model_specs": "bad"}),
        encoding="utf-8",
    )

    normalized = exp4223.load_harness_smoke_artifact(malformed)
    missing_artifact = exp4223.run(
        output_path=tmp_path / "missing_out.json",
        stable_checkpoint_path=_stable_checkpoint(tmp_path),
        harness_smoke_path=tmp_path / "missing.json",
        cuda_probe=lambda: True,
    )

    assert normalized["model_specs"] == {}
    assert exp4223.b1_harness_smoked(normalized) is False
    assert missing_artifact["honest_verdict"] == "complete_verifier_reward_deferred_harness_not_smoked"
    assert missing_artifact["preconditions"]["b1_harness_smoke_readable"] is False
    assert "FileNotFoundError" in missing_artifact["preconditions"]["b1_harness_smoke_error"]


def test_scenario_code_4223_cuda_blocker_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-CODE-4223-DEFERRED-HARNESS: CUDA failure stops after B1 is checked."""

    stable = _stable_checkpoint(tmp_path)
    smoke = _b1_smoke(tmp_path / "smoke.json")

    artifact = exp4223.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        harness_smoke_path=smoke,
        cuda_probe=lambda: False,
        cached_base_callback=lambda: exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
    )

    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["positive_control_confirmed"] is False
    assert artifact["preconditions"]["b1_harness_smoke_passed"] is True
    assert artifact["preconditions"]["cuda_available"] is False
    assert artifact["verifier_is_oracle"] is True


def test_scenario_code_4223_missing_cached_base_blocks_after_b1(tmp_path: Path) -> None:
    """SCENARIO-CODE-4223-SYNC-ACCUMULATE: missing non-Qwen base blocks before training."""

    stable = _stable_checkpoint(tmp_path)
    smoke = _b1_smoke(tmp_path / "smoke.json")

    artifact = exp4223.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        harness_smoke_path=smoke,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: None,
    )

    assert artifact["honest_verdict"] == "blocked_no_nonqwen_base_cached"
    assert artifact["preconditions"]["nonqwen_base_cached"] is False
    assert artifact["training"]["status"] == "not_started"


def test_scenario_code_4223_unreadable_stable_checkpoint_blocks(tmp_path: Path) -> None:
    """SCENARIO-CODE-4223-SYNC-ACCUMULATE: missing stable checkpoint is reported honestly."""

    smoke = _b1_smoke(tmp_path / "smoke.json")

    artifact = exp4223.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=tmp_path / "missing-stable",
        harness_smoke_path=smoke,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
    )

    assert artifact["honest_verdict"] == "blocked_stable_checkpoint_unreadable"
    assert artifact["preconditions"]["stable_checkpoint_readable"] is False
    assert "FileNotFoundError" in artifact["preconditions"]["stable_checkpoint_error"]


def test_scenario_code_4223_size_mismatch_blocks_random_label_control(tmp_path: Path) -> None:
    """SCENARIO-CODE-4223-SYNC-ACCUMULATE: Arm B must remain size-matched to Arm A."""

    stable = _stable_checkpoint(tmp_path, n_a=2, n_b=1, n_c=2)
    smoke = _b1_smoke(tmp_path / "smoke.json")

    artifact = exp4223.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        harness_smoke_path=smoke,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
    )

    assert artifact["honest_verdict"] == "blocked_size_matched_random_label_control_missing"
    assert artifact["preconditions"]["arms_n_matched"] is False
    assert artifact["arm_corpus_sizes"] == {"A": 2, "B": 1, "C": 2, "D": 0}


def test_scenario_code_4223_sync_accumulate_uses_b1_lora_config(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-4223-SYNC-ACCUMULATE: B1 config is carried into in-process resume."""

    stable = _stable_checkpoint(tmp_path)
    smoke = _b1_smoke(tmp_path / "smoke.json")
    captured: dict[str, object] = {}

    def fake_train(context: exp4211.TrainingContext) -> exp4211.TrainingOutcome:
        captured["context"] = context
        return exp4211.TrainingOutcome(
            status="partial",
            per_arm={
                "A": {"status": "trained", "completed_steps": 2},
                "B": {"status": "trained", "completed_steps": 2},
                "C": {"status": "trained", "completed_steps": 2},
                "D": {"status": "cold_base_eval_only"},
            },
            accumulated_train_examples={"A": 2, "B": 2, "C": 2, "D": 0},
            runner_artifact_path=stable / "runner_artifact.json",
            progress_events=[{"arm": "A", "step": 1, "loss": 0.75}],
            used_detached_process=False,
        )

    artifact = exp4223.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        harness_smoke_path=smoke,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
        train_callback=fake_train,
    )

    context = captured["context"]
    assert isinstance(context, exp4211.TrainingContext)
    assert context.mode == "in_process"
    assert artifact["training"]["used_detached_process"] is False
    assert artifact["accumulated_n"]["train_A"] == 2
    assert artifact["honest_verdict"] == "progress: accumulating_verifier_reward_training_no_eval_yet"
    assert context.manifest["model_specs"]["lora_config"]["target_modules"] == [
        "q_proj.linear",
        "k_proj.linear",
        "v_proj.linear",
        "o_proj.linear",
        "gate_proj.linear",
        "up_proj.linear",
        "down_proj.linear",
    ]
    assert artifact["model_specs"]["lora_attach_path"] == "wrapper_inner_linear_target_modules"
    assert artifact["model_specs"]["lora_config"] == context.manifest["model_specs"]["lora_config"]
    assert artifact["reproducibility_checksum"].startswith("sha256:")


def test_scenario_code_4223_training_exception_is_accumulating_artifact(tmp_path: Path) -> None:
    """SCENARIO-CODE-4223-SYNC-ACCUMULATE: live trainer failures still write progress JSON."""

    stable = _stable_checkpoint(tmp_path)
    smoke = _b1_smoke(tmp_path / "smoke.json")

    artifact = exp4223.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        harness_smoke_path=smoke,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
        train_callback=lambda _context: (_ for _ in ()).throw(RuntimeError("trainer failed")),
    )

    assert artifact["honest_verdict"] == "progress: accumulating_verifier_reward_training_no_eval_yet"
    assert artifact["training"]["status"] == "failed"
    assert artifact["training"]["error"] == "RuntimeError: trainer failed"


def test_scenario_code_4223_verdict_gate_label_carries_signal(tmp_path: Path) -> None:
    """SCENARIO-CODE-4223-VERDICT-GATES: A-vs-B is de-confounded and CI-gated."""

    stable = _stable_checkpoint(tmp_path, n_a=4, n_b=4, n_c=4)
    smoke = json.loads(_b1_smoke(tmp_path / "smoke.json").read_text(encoding="utf-8"))
    manifest, _corpus_paths, corpus_sizes = exp4223.load_b1_checkpoint_context(stable, smoke, random_seed=4198)
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
        memorization_shortcut_diagnostic={"status": "complete", "wrong_label_shortcut_probe": "A>>B"},
    )

    artifact = exp4223.build_result_artifact(
        preconditions={
            "b1_harness_smoke_passed": True,
            "nonqwen_base_cached": True,
            "cuda_available": True,
            "stable_checkpoint_readable": True,
            "arms_n_matched": True,
        },
        stable_checkpoint_path=stable,
        manifest=manifest,
        corpus_sizes=corpus_sizes,
        cached_base=exp4211.CachedBase("google/gemma-4-E4B-it", tmp_path / "model"),
        training=exp4211.TrainingOutcome(
            status="complete",
            per_arm={},
            accumulated_train_examples={"A": 4, "B": 4, "C": 4, "D": 0},
            runner_artifact_path=stable / "runner_artifact.json",
            progress_events=[],
            used_detached_process=False,
        ),
        evaluation=evaluation,
        adversarial_report=None,
        random_seed=4198,
        duration_s=65.0,
    )

    assert artifact["experiment"] == "experiment_4223_verifier_as_reward_3arm_synchronous"
    assert artifact["positive_control_confirmed"] is True
    assert artifact["a_vs_b_delta"] == pytest.approx(1.0)
    assert artifact["a_vs_b_ci95"] == [1.0, 1.0]
    assert artifact["verifier_label_carries_signal"] is True
    assert artifact["honest_verdict"] == "complete: verifier_label_carries_signal"
    assert artifact["youden_j"] == pytest.approx(0.4137931034482759)
    assert artifact["memorization_shortcut_diagnostic"]["wrong_label_shortcut_probe"] == "A>>B"


def test_req_code_4223_result_script_delegates_to_module() -> None:
    """REQ-CODE-4223: requested results script remains executable by path."""

    script = REPO / "results" / "experiment_4223_verifier_as_reward_3arm_synchronous.py"
    spec = importlib.util.spec_from_file_location("exp4223_result_script", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.main is exp4223.main

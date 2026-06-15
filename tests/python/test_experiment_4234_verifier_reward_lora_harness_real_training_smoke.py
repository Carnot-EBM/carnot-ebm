"""Tests for Exp 4234 verifier-reward LoRA real-training smoke.

Spec refs: REQ-CODE-4234, SCENARIO-CODE-4234-BLOCKED-PRECONDITION,
SCENARIO-CODE-4234-REAL-TRAINING.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from carnot import experiment_4234_verifier_reward_lora_harness_real_training_smoke as exp4234


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "code-verification" / "spec.md"
RUNNER_PATH = REPO / "scripts" / "experiments" / "verifier_reward_code_lora_rft_3arm.py"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _stable_checkpoint(tmp_path: Path, n: int = 12) -> Path:
    root = tmp_path / "code_verifier_reward_lora_rft_a83b52882c198954"
    for arm in ("A", "B", "C"):
        rows = [
            {
                "arm": f"arm_{arm}",
                "completion": f"def f_{arm.lower()}_{i}(x):\n    return x + {i}\n",
                "hidden_pass": arm != "B",
                "prompt": f"Complete HumanEval fixture {i}.",
                "task_id": f"HumanEval/{i}",
                "visible_perfect": arm != "B",
            }
            for i in range(n)
        ]
        _write_jsonl(root / "corpora" / f"arm_{arm}.jsonl", rows)
    return root


def _cached_base(tmp_path: Path, model_id: str = "google/gemma-4-12B-it") -> exp4234.CachedBase:
    cache_path = tmp_path / exp4234.hf_cache_name(model_id)
    cache_path.mkdir(parents=True)
    return exp4234.CachedBase(model_id=model_id, cache_path=cache_path)


def _passing_smoke() -> exp4234.RealTrainingSmokeResult:
    losses = [2.0 - 0.03 * idx for idx in range(exp4234.MIN_REAL_OPTIMIZER_STEPS)]
    return exp4234.RealTrainingSmokeResult(
        lora_attach_path=exp4234.STANDARD_ATTACH_PATH,
        trainable_param_count=1234,
        steps_run=exp4234.MIN_REAL_OPTIMIZER_STEPS,
        loss_initial=losses[0],
        loss_final=losses[-1],
        loss_trace=[{"step": idx + 1, "loss": loss} for idx, loss in enumerate(losses)],
        duration_s=exp4234.DEFAULT_DURATION_FLOOR_S + 1.0,
        harness_smoke_passed=True,
        lora_config=exp4234.working_lora_config(),
    )


def test_req_code_4234_spec_declares_real_training_contract() -> None:
    """REQ-CODE-4234: OpenSpec names the sustained-training smoke contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-4234" in spec
    assert "SCENARIO-CODE-4234-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CODE-4234-REAL-TRAINING" in spec
    for field in exp4234.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4234.FIELD_PRINCIPLES


def test_req_code_4234_base_selection_prefers_non_gguf_12b(tmp_path: Path) -> None:
    """REQ-CODE-4234: cached 12B HF bases beat E4B and GGUF/Qwen caches."""

    hub = tmp_path / "hub"
    (hub / "models--unsloth--gemma-4-12B-it-GGUF").mkdir(parents=True)
    (hub / "models--Qwen--Qwen3.5-0.8B").mkdir()
    (hub / "models--google--gemma-4-E4B-it").mkdir()
    (hub / "models--google--gemma-4-12B-it").mkdir()

    selected = exp4234.find_cached_nonqwen_base(hub_root=hub)

    assert selected == exp4234.CachedBase(
        model_id="google/gemma-4-12B-it",
        cache_path=hub / "models--google--gemma-4-12B-it",
    )


def test_req_code_4234_base_selection_falls_back_to_e4b(tmp_path: Path) -> None:
    """REQ-CODE-4234: missing non-GGUF 12B cache falls back only to E4B."""

    hub = tmp_path / "hub"
    (hub / "models--unsloth--gemma-4-12B-it-GGUF").mkdir(parents=True)
    (hub / "models--google--gemma-4-E4B-it").mkdir()

    selected = exp4234.find_cached_nonqwen_base(hub_root=hub)

    assert selected is not None
    assert selected.model_id == "google/gemma-4-E4B-it"
    assert "GGUF" not in selected.model_id
    assert "qwen" not in selected.model_id.lower()


def test_req_code_4234_base_selection_ignores_forbidden_only_cache(tmp_path: Path) -> None:
    """REQ-CODE-4234: Qwen/GGUF-only caches are not eligible trained bases."""

    hub = tmp_path / "hub"
    (hub / "models--unsloth--gemma-4-12B-it-GGUF").mkdir(parents=True)
    (hub / "models--Qwen--Qwen3.5-0.8B").mkdir()

    assert (
        exp4234.find_cached_nonqwen_base(
            model_ids=("Qwen/Qwen3.5-0.8B", "unsloth/gemma-4-12B-it-GGUF"),
            hub_root=hub,
        )
        is None
    )


def test_scenario_code_4234_cuda_blocker_stops_before_attach(tmp_path: Path) -> None:
    """SCENARIO-CODE-4234-BLOCKED-PRECONDITION: CUDA failure skips base and LoRA attach."""

    stable = _stable_checkpoint(tmp_path)
    called = {"base": False, "smoke": False}

    def base_callback():
        called["base"] = True
        return _cached_base(tmp_path)

    def smoke_callback(*_args, **_kwargs):
        called["smoke"] = True
        raise AssertionError("smoke must not run without CUDA")

    artifact = exp4234.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: False,
        cached_base_callback=base_callback,
        smoke_callback=smoke_callback,
    )

    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["harness_smoke_passed"] is False
    assert artifact["steps_run"] == 0
    assert artifact["trainable_param_count"] == 0
    assert called == {"base": False, "smoke": False}


def test_scenario_code_4234_base_blocker_forbids_qwen_substitute(tmp_path: Path) -> None:
    """SCENARIO-CODE-4234-BLOCKED-PRECONDITION: missing approved Gemma base blocks."""

    stable = _stable_checkpoint(tmp_path)
    called = False

    def smoke_callback(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Qwen or GGUF caches must not satisfy the base precondition")

    artifact = exp4234.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: None,
        smoke_callback=smoke_callback,
    )

    assert artifact["honest_verdict"] == "blocked_no_nonqwen_base_cached"
    assert artifact["harness_smoke_passed"] is False
    assert artifact["model_specs"]["qwen_train_base_forbidden"] is True
    assert called is False


def test_scenario_code_4234_success_artifact_records_real_training_fields(tmp_path: Path) -> None:
    """SCENARIO-CODE-4234-REAL-TRAINING: B2 gate sees bare sustained-training fields."""

    stable = _stable_checkpoint(tmp_path)
    cached = _cached_base(tmp_path)

    def smoke_callback(fixture, *, cached_base, random_seed, min_steps, duration_floor_s):
        assert len(fixture.rows) == exp4234.FIXTURE_SIZE
        assert cached_base == cached
        assert random_seed == exp4234.RANDOM_SEED
        assert min_steps == exp4234.MIN_REAL_OPTIMIZER_STEPS
        assert duration_floor_s == exp4234.DEFAULT_DURATION_FLOOR_S
        return _passing_smoke()

    artifact = exp4234.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: cached,
        smoke_callback=smoke_callback,
    )
    persisted = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))

    assert persisted == artifact
    assert artifact["honest_verdict"] == "complete: verifier_reward_lora_real_training_smoke_passed"
    assert artifact["harness_smoke_passed"] is True
    assert artifact["steps_run"] == exp4234.MIN_REAL_OPTIMIZER_STEPS
    assert artifact["loss_final"] < artifact["loss_initial"]
    assert artifact["trainable_param_count"] == 1234
    assert artifact["lora_attach_path"] == exp4234.STANDARD_ATTACH_PATH
    assert artifact["verifier_is_oracle"] is True
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert artifact["model_specs"]["trainable_base"] == cached.model_id
    assert artifact["model_specs"]["lora_config"]["target_modules"] == exp4234.STANDARD_LORA_TARGET_MODULES


def test_scenario_code_4234_short_duration_fails_loudly(tmp_path: Path) -> None:
    """SCENARIO-CODE-4234-REAL-TRAINING: too-fast steps are a blocked short-circuit."""

    stable = _stable_checkpoint(tmp_path)
    cached = _cached_base(tmp_path)
    short = _passing_smoke()
    short = exp4234.RealTrainingSmokeResult(
        lora_attach_path=short.lora_attach_path,
        trainable_param_count=short.trainable_param_count,
        steps_run=short.steps_run,
        loss_initial=short.loss_initial,
        loss_final=short.loss_final,
        loss_trace=short.loss_trace,
        duration_s=1.0,
        harness_smoke_passed=True,
        lora_config=short.lora_config,
    )

    artifact = exp4234.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: cached,
        smoke_callback=lambda *_args, **_kwargs: short,
    )

    assert artifact["honest_verdict"] == "blocked_lora_training_cannot_run_in_window"
    assert artifact["harness_smoke_passed"] is False
    assert artifact["steps_run"] == exp4234.MIN_REAL_OPTIMIZER_STEPS
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert not artifact["honest_verdict"].startswith("progress")
    assert artifact["smoke_failure_reason"] == "duration_below_plausibility_floor"


def test_scenario_code_4234_loss_must_move_and_steps_must_reach_floor() -> None:
    """SCENARIO-CODE-4234-REAL-TRAINING: gate rejects frozen or incomplete traces."""

    frozen = exp4234.RealTrainingSmokeResult(
        lora_attach_path=exp4234.STANDARD_ATTACH_PATH,
        trainable_param_count=10,
        steps_run=20,
        loss_initial=1.0,
        loss_final=1.0,
        loss_trace=[{"step": idx + 1, "loss": 1.0} for idx in range(20)],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4234.working_lora_config(),
    )
    incomplete = exp4234.RealTrainingSmokeResult(
        lora_attach_path=exp4234.STANDARD_ATTACH_PATH,
        trainable_param_count=10,
        steps_run=19,
        loss_initial=2.0,
        loss_final=1.0,
        loss_trace=[{"step": idx + 1, "loss": 2.0 - idx * 0.01} for idx in range(19)],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4234.working_lora_config(),
    )

    assert exp4234.real_training_gate(frozen).passed is False
    assert exp4234.real_training_gate(frozen).reason == "loss_did_not_move"
    assert exp4234.real_training_gate(incomplete).passed is False
    assert exp4234.real_training_gate(incomplete).reason == "insufficient_optimizer_steps"


def test_scenario_code_4234_missing_or_nonfinite_loss_is_not_real_training() -> None:
    """SCENARIO-CODE-4234-REAL-TRAINING: missing/nonfinite losses fail the gate."""

    missing = exp4234.RealTrainingSmokeResult(
        lora_attach_path=exp4234.STANDARD_ATTACH_PATH,
        trainable_param_count=10,
        steps_run=20,
        loss_initial=None,
        loss_final=1.0,
        loss_trace=[],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4234.working_lora_config(),
    )
    nonfinite = exp4234.RealTrainingSmokeResult(
        lora_attach_path=exp4234.STANDARD_ATTACH_PATH,
        trainable_param_count=10,
        steps_run=20,
        loss_initial=float("nan"),
        loss_final=1.0,
        loss_trace=[],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4234.working_lora_config(),
    )

    assert exp4234.real_training_gate(missing).reason == "missing_loss_trace"
    assert exp4234.real_training_gate(nonfinite).reason == "non_finite_loss"


def test_req_code_4234_jsonable_handles_to_dict_and_item(tmp_path: Path) -> None:
    """REQ-CODE-4234: artifact serialization normalizes helper-like values."""

    class ToDict:
        def to_dict(self):
            return {"path": tmp_path}

    class Item:
        def item(self):
            return 7

    assert exp4234._jsonable(ToDict()) == {"path": str(tmp_path)}
    assert exp4234._jsonable(Item()) == 7


def test_scenario_code_4234_runner_gate_helper_matches_artifact_gate() -> None:
    """SCENARIO-CODE-4234-REAL-TRAINING: shared runner rejects fake-short traces."""

    spec = importlib.util.spec_from_file_location("verifier_reward_code_lora_rft_3arm_under_4234_test", RUNNER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    passed = module._real_training_smoke_gate(
        trainable_param_count=10,
        loss_trace=[2.0 - idx * 0.05 for idx in range(20)],
        duration_s=15.0,
    )
    short = module._real_training_smoke_gate(
        trainable_param_count=10,
        loss_trace=[2.0 - idx * 0.05 for idx in range(20)],
        duration_s=1.0,
    )

    assert passed == (True, None)
    assert short == (False, "duration_below_plausibility_floor")


def test_scenario_code_4234_smoke_exception_is_blocked_not_progress(tmp_path: Path) -> None:
    """SCENARIO-CODE-4234-REAL-TRAINING: live training exceptions fail loudly."""

    stable = _stable_checkpoint(tmp_path)
    cached = _cached_base(tmp_path)

    artifact = exp4234.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: cached,
        smoke_callback=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("oom")),
    )

    assert artifact["honest_verdict"] == "blocked_lora_training_cannot_run_in_window"
    assert artifact["smoke_failure_reason"] == "RuntimeError: oom"
    assert artifact["harness_smoke_passed"] is False

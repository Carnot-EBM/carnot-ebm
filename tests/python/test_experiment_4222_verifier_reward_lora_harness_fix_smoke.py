"""Tests for Exp 4222 verifier-reward LoRA harness smoke.

Spec refs: REQ-CODE-4222, SCENARIO-CODE-4222-BLOCKED-PRECONDITION,
SCENARIO-CODE-4222-STANDARD-LORA-ATTACH.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4222_verifier_reward_lora_harness_fix_smoke as exp4222


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "code-verification" / "spec.md"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _stable_checkpoint(tmp_path: Path, n: int = 4) -> Path:
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


def test_req_code_4222_spec_declares_smoke_contract() -> None:
    """REQ-CODE-4222: OpenSpec names the attach-smoke artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-4222" in spec
    assert "SCENARIO-CODE-4222-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CODE-4222-STANDARD-LORA-ATTACH" in spec
    for field in exp4222.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4222.FIELD_PRINCIPLES


def test_scenario_code_4222_cuda_blocker_stops_before_attach(tmp_path: Path) -> None:
    """SCENARIO-CODE-4222-BLOCKED-PRECONDITION: CUDA failure skips LoRA attach."""

    stable = _stable_checkpoint(tmp_path)
    cache = tmp_path / "models--google--gemma-4-E4B-it"
    cache.mkdir()
    called = False

    def fail_smoke(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("smoke must not run without CUDA")

    artifact = exp4222.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cache_path=cache,
        cuda_probe=lambda: False,
        smoke_callback=fail_smoke,
    )

    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["harness_smoke_passed"] is False
    assert artifact["trainable_param_count"] == 0
    assert artifact["preconditions"]["cuda_available"] is False
    assert called is False


def test_scenario_code_4222_cache_blocker_forbids_qwen_substitute(tmp_path: Path) -> None:
    """SCENARIO-CODE-4222-BLOCKED-PRECONDITION: missing Gemma cache blocks training."""

    stable = _stable_checkpoint(tmp_path)
    qwen_cache = tmp_path / "models--Qwen--Qwen3.5-0.8B"
    qwen_cache.mkdir()
    called = False

    def fail_smoke(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Qwen cache must not satisfy the non-Qwen precondition")

    artifact = exp4222.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cache_path=tmp_path / "models--google--gemma-4-E4B-it",
        cuda_probe=lambda: True,
        smoke_callback=fail_smoke,
    )

    assert artifact["honest_verdict"] == "blocked_no_nonqwen_base_cached"
    assert artifact["model_specs"]["trainable_base"] == exp4222.MODEL_ID
    assert artifact["model_specs"]["trainable_base_is_non_qwen"] is True
    assert artifact["model_specs"]["qwen_train_base_forbidden"] is True
    assert called is False


def test_req_code_4222_fixture_and_checksum_are_deterministic(tmp_path: Path) -> None:
    """REQ-CODE-4222: fixture rows and LoRA config define reproducibility checksum."""

    stable = _stable_checkpoint(tmp_path, n=4)
    for path in (stable / "corpora").glob("*.jsonl"):
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    fixture = exp4222.load_or_build_fixture(stable, fixture_size=8)
    lora_config = exp4222.working_lora_config()

    checksum = exp4222.reproducibility_checksum(fixture.rows, lora_config, random_seed=4198)
    repeated = exp4222.reproducibility_checksum(fixture.rows, lora_config, random_seed=4198)

    assert fixture.source == "stable_checkpoint_corpora"
    assert len(fixture.rows) == 8
    assert fixture.corpus_sizes == {"A": 4, "B": 4, "C": 4}
    assert checksum == repeated
    assert checksum.startswith("sha256:")
    assert lora_config["target_modules"] == exp4222.STANDARD_LORA_TARGET_MODULES


def test_req_code_4222_fixture_fallback_and_jsonable_edges(tmp_path: Path) -> None:
    """REQ-CODE-4222: fallback fixtures and serialization helpers are deterministic."""

    class ToDict:
        def to_dict(self):
            return {"path": tmp_path}

    class Item:
        def item(self):
            return 9

    fallback = exp4222.load_or_build_fixture(tmp_path / "missing", fixture_size=8)
    short = exp4222.load_or_build_fixture(_stable_checkpoint(tmp_path / "short", n=1), fixture_size=8)

    assert fallback.source == "tiny_operating_point_fixture"
    assert len(fallback.rows) == 8
    assert short.source == "stable_checkpoint_corpora"
    assert len(short.rows) == 3
    assert exp4222._jsonable(tmp_path) == str(tmp_path)
    assert exp4222._jsonable(ToDict()) == {"path": str(tmp_path)}
    assert exp4222._jsonable(fallback)["source"] == "tiny_operating_point_fixture"
    assert exp4222._jsonable(Item()) == 9


def test_scenario_code_4222_standard_lora_attach_uses_projection_targets() -> None:
    """SCENARIO-CODE-4222-STANDARD-LORA-ATTACH: PEFT attaches and finite step passes."""

    seen: dict[str, object] = {}

    class FakeParam:
        def __init__(self, requires_grad: bool, n: int) -> None:
            self.requires_grad = requires_grad
            self._n = n

        def numel(self) -> int:
            return self._n

    class FakeModel:
        def parameters(self):
            return [FakeParam(True, 11), FakeParam(False, 5), FakeParam(True, 7)]

    class FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def fake_get_peft_model(model, config):
        seen["target_modules"] = config.kwargs["target_modules"]
        return model

    result = exp4222.run_lora_attach_and_step(
        [{"prompt": "Complete f.", "completion": "def f(x):\n    return x\n"}],
        model_id=exp4222.MODEL_ID,
        random_seed=4198,
        load_model=lambda model_id: FakeModel(),
        load_tokenizer=lambda model_id: object(),
        lora_config_cls=FakeLoraConfig,
        get_peft_model_fn=fake_get_peft_model,
        step_fn=lambda *_args, **_kwargs: 0.125,
        seed_fn=lambda _seed: None,
    )

    assert result.harness_smoke_passed is True
    assert result.lora_attach_path == exp4222.STANDARD_ATTACH_PATH
    assert result.trainable_param_count == 18
    assert result.loss == pytest.approx(0.125)
    assert seen["target_modules"] == exp4222.STANDARD_LORA_TARGET_MODULES


def test_scenario_code_4222_wrapper_inner_linear_fallback_attaches() -> None:
    """SCENARIO-CODE-4222-STANDARD-LORA-ATTACH: wrapper rejection retries inner Linear."""

    attempts: list[list[str]] = []

    class FakeParam:
        requires_grad = True

        def numel(self) -> int:
            return 23

    class FakeModel:
        def parameters(self):
            return [FakeParam()]

    class FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def fake_get_peft_model(model, config):
        target_modules = config.kwargs["target_modules"]
        attempts.append(target_modules)
        if target_modules == exp4222.STANDARD_LORA_TARGET_MODULES:
            raise ValueError("Target module Gemma4ClippableLinear(...) is not supported")
        return model

    result = exp4222.run_lora_attach_and_step(
        [{"prompt": "Complete f.", "completion": "def f(x):\n    return x\n"}],
        model_id=exp4222.MODEL_ID,
        random_seed=4198,
        load_model=lambda _model_id: FakeModel(),
        load_tokenizer=lambda _model_id: object(),
        lora_config_cls=FakeLoraConfig,
        get_peft_model_fn=fake_get_peft_model,
        step_fn=lambda *_args, **_kwargs: 0.25,
        seed_fn=lambda _seed: None,
    )

    assert attempts == [exp4222.STANDARD_LORA_TARGET_MODULES, exp4222.INNER_LINEAR_LORA_TARGET_MODULES]
    assert result.harness_smoke_passed is True
    assert result.lora_attach_path == exp4222.WRAPPER_INNER_LINEAR_ATTACH_PATH
    assert result.trainable_param_count == 23
    assert result.lora_config["target_modules"] == exp4222.INNER_LINEAR_LORA_TARGET_MODULES


def test_scenario_code_4222_zero_trainable_lora_is_failed_attach() -> None:
    """SCENARIO-CODE-4222-STANDARD-LORA-ATTACH: zero LoRA params cannot pass."""

    class FrozenParam:
        requires_grad = False

        def numel(self) -> int:
            return 99

    class FrozenModel:
        def parameters(self):
            return [FrozenParam()]

    class FakeLoraConfig:
        def __init__(self, **_kwargs) -> None:
            pass

    result = exp4222.run_lora_attach_and_step(
        [{"prompt": "p", "completion": "c"}],
        model_id=exp4222.MODEL_ID,
        random_seed=4198,
        load_model=lambda _model_id: FrozenModel(),
        load_tokenizer=lambda _model_id: object(),
        lora_config_cls=FakeLoraConfig,
        get_peft_model_fn=lambda model, _config: model,
        seed_fn=lambda _seed: None,
    )

    assert result.harness_smoke_passed is False
    assert result.trainable_param_count == 0
    assert result.error == "blocked_no_trainable_lora_parameters"


def test_scenario_code_4222_success_artifact_records_bare_gate_fields(tmp_path: Path) -> None:
    """SCENARIO-CODE-4222-STANDARD-LORA-ATTACH: result artifact is B2-gateable."""

    stable = _stable_checkpoint(tmp_path)
    cache = tmp_path / "models--google--gemma-4-E4B-it"
    cache.mkdir()

    def fake_smoke(fixture, *, random_seed):
        assert len(fixture.rows) == 8
        assert random_seed == 4198
        return exp4222.SmokeResult(
            harness_smoke_passed=True,
            lora_attach_path=exp4222.STANDARD_ATTACH_PATH,
            trainable_param_count=1234,
            loss=0.5,
            progress_events=[{"step": 1, "loss": 0.5}],
        )

    artifact = exp4222.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cache_path=cache,
        cuda_probe=lambda: True,
        smoke_callback=fake_smoke,
    )
    persisted = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))

    assert persisted == artifact
    assert artifact["honest_verdict"] == "complete: verifier_reward_lora_harness_smoke_passed"
    assert artifact["harness_smoke_passed"] is True
    assert artifact["trainable_param_count"] == 1234
    assert artifact["lora_attach_path"] == exp4222.STANDARD_ATTACH_PATH
    assert artifact["verifier_is_oracle"] is True
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert artifact["model_specs"]["lora_config"]["target_modules"] == exp4222.STANDARD_LORA_TARGET_MODULES


def test_scenario_code_4222_failed_and_default_smoke_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CODE-4222-STANDARD-LORA-ATTACH: failed smoke is not gateable."""

    stable = _stable_checkpoint(tmp_path)
    cache = tmp_path / "models--google--gemma-4-E4B-it"
    cache.mkdir()

    failed = exp4222.build_artifact(
        preconditions={"cuda_available": True, "nonqwen_base_cached": True},
        fixture=exp4222.load_or_build_fixture(stable),
        smoke=exp4222.SmokeResult(False, exp4222.STANDARD_ATTACH_PATH, 0, float("nan"), [], "non_finite_loss"),
        cache_path=cache,
        lora_config=exp4222.working_lora_config(),
        random_seed=4198,
        duration_s=0.1,
    )

    assert failed["honest_verdict"] == "failed: verifier_reward_lora_harness_smoke_failed"
    assert failed["acceptance_gate"]["satisfied"] is False

    monkeypatch.setattr(
        exp4222,
        "run_lora_attach_and_step",
        lambda rows, **_kwargs: exp4222.SmokeResult(True, exp4222.STANDARD_ATTACH_PATH, 5, 0.25, []),
    )
    default_artifact = exp4222.run(
        output_path=tmp_path / "default.json",
        stable_checkpoint_path=stable,
        cache_path=cache,
        cuda_probe=lambda: True,
    )
    assert default_artifact["harness_smoke_passed"] is True

    raised_artifact = exp4222.run(
        output_path=tmp_path / "raised.json",
        stable_checkpoint_path=stable,
        cache_path=cache,
        cuda_probe=lambda: True,
        smoke_callback=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("attach failed")),
    )
    assert raised_artifact["honest_verdict"] == "failed: verifier_reward_lora_harness_smoke_failed"
    assert raised_artifact["training_step"]["error"] == "RuntimeError: attach failed"

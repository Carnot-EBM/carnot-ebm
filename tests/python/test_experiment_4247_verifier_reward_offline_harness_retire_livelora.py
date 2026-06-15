"""Tests for Exp 4247 offline reward-weighted SFT harness retirement.

Spec refs: REQ-CODE-4247, SCENARIO-CODE-4247-BLOCKED-PRECONDITION,
SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4247_verifier_reward_offline_harness_retire_livelora as exp4247


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "code-verification" / "spec.md"


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


def _cached_base(tmp_path: Path, model_id: str = "google/gemma-4-E4B-it") -> exp4247.CachedBase:
    cache_path = tmp_path / exp4247.hf_cache_name(model_id)
    cache_path.mkdir(parents=True)
    return exp4247.CachedBase(model_id=model_id, cache_path=cache_path)


def _passing_smoke() -> exp4247.OfflineTrainingSmokeResult:
    losses = [2.0 - 0.03 * idx for idx in range(exp4247.MIN_REAL_OPTIMIZER_STEPS)]
    return exp4247.OfflineTrainingSmokeResult(
        lora_attach_path=exp4247.STANDARD_ATTACH_PATH,
        trainable_param_count=4321,
        steps_run=exp4247.MIN_REAL_OPTIMIZER_STEPS,
        loss_initial=losses[0],
        loss_final=losses[-1],
        loss_trace=[{"step": idx + 1, "loss": loss, "reward_weight": 1.0} for idx, loss in enumerate(losses)],
        duration_s=exp4247.DEFAULT_DURATION_FLOOR_S + 1.0,
        harness_smoke_passed=True,
        lora_config=exp4247.working_lora_config(),
    )


def test_req_code_4247_spec_declares_offline_retirement_contract() -> None:
    """REQ-CODE-4247: OpenSpec names the offline smoke and retirement fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-4247" in spec
    assert "SCENARIO-CODE-4247-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING" in spec
    for field in exp4247.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4247.FIELD_PRINCIPLES


def test_req_code_4247_base_selection_prefers_e4b_and_ignores_forbidden(tmp_path: Path) -> None:
    """REQ-CODE-4247: cached E4B HF base beats fallback, Qwen, and GGUF caches."""

    hub = tmp_path / "hub"
    (hub / "models--unsloth--gemma-4-12B-it-GGUF").mkdir(parents=True)
    (hub / "models--Qwen--Qwen3.5-0.8B").mkdir()
    (hub / "models--unsloth--gemma-4-12B-it").mkdir()
    (hub / "models--google--gemma-4-E4B-it").mkdir()

    selected = exp4247.find_cached_nonqwen_base(hub_root=hub)

    assert selected == exp4247.CachedBase(
        model_id="google/gemma-4-E4B-it",
        cache_path=hub / "models--google--gemma-4-E4B-it",
    )


def test_req_code_4247_base_selection_falls_back_only_to_unsloth_12b(tmp_path: Path) -> None:
    """REQ-CODE-4247: missing E4B falls back to non-GGUF unsloth 12B."""

    hub = tmp_path / "hub"
    (hub / "models--unsloth--gemma-4-12B-it-GGUF").mkdir(parents=True)
    (hub / "models--unsloth--gemma-4-12B-it").mkdir()

    selected = exp4247.find_cached_nonqwen_base(hub_root=hub)

    assert selected is not None
    assert selected.model_id == "unsloth/gemma-4-12B-it"
    assert "GGUF" not in selected.model_id
    assert "qwen" not in selected.model_id.lower()


def test_req_code_4247_base_selection_blocks_forbidden_only_cache(tmp_path: Path) -> None:
    """REQ-CODE-4247: Qwen/GGUF-only caches do not satisfy the training base gate."""

    hub = tmp_path / "hub"
    (hub / "models--unsloth--gemma-4-12B-it-GGUF").mkdir(parents=True)
    (hub / "models--Qwen--Qwen3.5-0.8B").mkdir()

    assert (
        exp4247.find_cached_nonqwen_base(
            model_ids=("Qwen/Qwen3.5-0.8B", "unsloth/gemma-4-12B-it-GGUF"),
            hub_root=hub,
        )
        is None
    )


def test_req_code_4247_fixture_precomputes_reward_weights_and_checksum(tmp_path: Path) -> None:
    """REQ-CODE-4247: fixture rows carry deterministic offline reward weights."""

    stable = _stable_checkpoint(tmp_path, n=4)
    fixture = exp4247.load_or_build_weighted_fixture(stable, fixture_size=9)
    checksum = exp4247.reproducibility_checksum(
        fixture_rows=fixture.rows,
        lora_config=exp4247.working_lora_config(),
        weighting_scheme=exp4247.reward_weighting_scheme(),
        model_id="google/gemma-4-E4B-it",
        random_seed=exp4247.RANDOM_SEED,
    )
    repeated = exp4247.reproducibility_checksum(
        fixture_rows=fixture.rows,
        lora_config=exp4247.working_lora_config(),
        weighting_scheme=exp4247.reward_weighting_scheme(),
        model_id="google/gemma-4-E4B-it",
        random_seed=exp4247.RANDOM_SEED,
    )

    assert fixture.source == "stable_checkpoint_corpora"
    assert fixture.corpus_sizes == {"A": 4, "B": 4, "C": 4}
    assert [row["reward_source"] for row in fixture.rows[:3]] == [
        "verifier_certified",
        "same_generator_random_label_control",
        "hidden_gold_positive_control",
    ]
    assert [row["reward_weight"] for row in fixture.rows[:3]] == [1.0, 0.25, 1.0]
    assert checksum == repeated
    assert checksum.startswith("sha256:")


def test_req_code_4247_fixture_fallback_and_jsonable_edges(tmp_path: Path) -> None:
    """REQ-CODE-4247: fallback fixtures and artifact serialization stay deterministic."""

    class ToDict:
        def to_dict(self):
            return {"path": tmp_path}

    class Item:
        def item(self):
            return 5

    fallback = exp4247.load_or_build_weighted_fixture(tmp_path / "missing", fixture_size=16)
    short = _stable_checkpoint(tmp_path / "short", n=1)
    for path in (short / "corpora").glob("*.jsonl"):
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    short_fixture = exp4247.load_or_build_weighted_fixture(short, fixture_size=16)

    assert fallback.source == "tiny_operating_point_fixture"
    assert len(fallback.rows) == 16
    assert short_fixture.source == "stable_checkpoint_corpora"
    assert len(short_fixture.rows) == 3
    assert {row["reward_source"] for row in fallback.rows} == {
        "verifier_certified",
        "same_generator_random_label_control",
        "hidden_gold_positive_control",
    }
    assert exp4247._jsonable(tmp_path) == str(tmp_path)
    assert exp4247._jsonable(ToDict()) == {"path": str(tmp_path)}
    assert exp4247._jsonable(fallback)["source"] == "tiny_operating_point_fixture"
    assert exp4247._jsonable(Item()) == 5


def test_scenario_code_4247_cuda_blocker_stops_before_attach(tmp_path: Path) -> None:
    """SCENARIO-CODE-4247-BLOCKED-PRECONDITION: CUDA failure skips base and smoke."""

    stable = _stable_checkpoint(tmp_path)
    called = {"base": False, "smoke": False}

    def base_callback():
        called["base"] = True
        return _cached_base(tmp_path)

    def smoke_callback(*_args, **_kwargs):
        called["smoke"] = True
        raise AssertionError("smoke must not run without CUDA")

    artifact = exp4247.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: False,
        cached_base_callback=base_callback,
        smoke_callback=smoke_callback,
    )

    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["live_lora_retired"] is True
    assert artifact["harness_smoke_passed"] is False
    assert artifact["steps_run"] == 0
    assert artifact["trainable_param_count"] == 0
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert called == {"base": False, "smoke": False}


def test_scenario_code_4247_base_blocker_forbids_qwen_substitute(tmp_path: Path) -> None:
    """SCENARIO-CODE-4247-BLOCKED-PRECONDITION: missing approved Gemma base blocks."""

    stable = _stable_checkpoint(tmp_path)
    called = False

    def smoke_callback(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Qwen or GGUF caches must not satisfy the base precondition")

    artifact = exp4247.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: None,
        smoke_callback=smoke_callback,
    )

    assert artifact["honest_verdict"] == "blocked_no_nonqwen_base_cached"
    assert artifact["live_lora_retired"] is True
    assert artifact["harness_smoke_passed"] is False
    assert artifact["model_specs"]["qwen_train_base_forbidden"] is True
    assert called is False


def test_scenario_code_4247_success_artifact_records_offline_training_fields(tmp_path: Path) -> None:
    """SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING: B2 sees bare offline smoke fields."""

    stable = _stable_checkpoint(tmp_path)
    cached = _cached_base(tmp_path)

    def smoke_callback(fixture, *, cached_base, random_seed, min_steps, duration_floor_s):
        assert len(fixture.rows) == exp4247.FIXTURE_SIZE
        assert cached_base == cached
        assert random_seed == exp4247.RANDOM_SEED
        assert min_steps == exp4247.MIN_REAL_OPTIMIZER_STEPS
        assert duration_floor_s == exp4247.DEFAULT_DURATION_FLOOR_S
        return _passing_smoke()

    artifact = exp4247.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: cached,
        smoke_callback=smoke_callback,
    )
    persisted = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))

    assert persisted == artifact
    assert artifact["honest_verdict"] == "complete: verifier_reward_offline_reward_weighted_smoke_passed"
    assert artifact["live_lora_retired"] is True
    assert artifact["harness_smoke_passed"] is True
    assert artifact["steps_run"] == exp4247.MIN_REAL_OPTIMIZER_STEPS
    assert artifact["loss_final"] < artifact["loss_initial"]
    assert artifact["trainable_param_count"] == 4321
    assert artifact["lora_attach_path"] == exp4247.STANDARD_ATTACH_PATH
    assert artifact["verifier_is_oracle"] is True
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert artifact["model_specs"]["trainable_base"] == cached.model_id
    assert artifact["model_specs"]["lora_config"]["target_modules"] == exp4247.STANDARD_LORA_TARGET_MODULES
    assert artifact["model_specs"]["offline_reward_weighting_scheme"]["control_weight"] == 0.25
    assert artifact["model_specs"]["offline_reward_weighting_scheme"]["live_generation"] is False
    assert artifact["model_specs"]["offline_reward_weighting_scheme"]["rl_loop"] is False


def test_scenario_code_4247_short_duration_fails_loudly(tmp_path: Path) -> None:
    """SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING: fake-short steps are blocked."""

    stable = _stable_checkpoint(tmp_path)
    cached = _cached_base(tmp_path)
    short = _passing_smoke()
    short = exp4247.OfflineTrainingSmokeResult(
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

    artifact = exp4247.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: cached,
        smoke_callback=lambda *_args, **_kwargs: short,
    )

    assert artifact["honest_verdict"] == "blocked_offline_reward_weighted_training_cannot_run_in_window"
    assert artifact["harness_smoke_passed"] is False
    assert artifact["steps_run"] == exp4247.MIN_REAL_OPTIMIZER_STEPS
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert not artifact["honest_verdict"].startswith("progress")
    assert artifact["smoke_failure_reason"] == "duration_below_plausibility_floor"


def test_scenario_code_4247_loss_and_steps_gate() -> None:
    """SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING: frozen or incomplete traces fail."""

    frozen = exp4247.OfflineTrainingSmokeResult(
        lora_attach_path=exp4247.STANDARD_ATTACH_PATH,
        trainable_param_count=10,
        steps_run=20,
        loss_initial=1.0,
        loss_final=1.0,
        loss_trace=[{"step": idx + 1, "loss": 1.0} for idx in range(20)],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4247.working_lora_config(),
    )
    incomplete = exp4247.OfflineTrainingSmokeResult(
        lora_attach_path=exp4247.STANDARD_ATTACH_PATH,
        trainable_param_count=10,
        steps_run=19,
        loss_initial=2.0,
        loss_final=1.0,
        loss_trace=[{"step": idx + 1, "loss": 2.0 - idx * 0.01} for idx in range(19)],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4247.working_lora_config(),
    )

    assert exp4247.offline_training_gate(frozen).reason == "loss_did_not_move"
    assert exp4247.offline_training_gate(incomplete).reason == "insufficient_optimizer_steps"


def test_scenario_code_4247_lora_attach_path_and_config_helpers() -> None:
    """SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING: attach path and PEFT config are explicit."""

    class FakeParam:
        def __init__(self, requires_grad: bool, n: int) -> None:
            self.requires_grad = requires_grad
            self._n = n

        def numel(self) -> int:
            return self._n

    class FakeModel:
        def parameters(self):
            return [FakeParam(True, 3), FakeParam(False, 100), FakeParam(True, 4)]

    missing_attach = exp4247.OfflineTrainingSmokeResult(
        lora_attach_path="",
        trainable_param_count=10,
        steps_run=20,
        loss_initial=2.0,
        loss_final=1.0,
        loss_trace=[],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4247.working_lora_config(),
    )
    kwargs = exp4247._lora_config_kwargs(exp4247.working_lora_config())

    assert exp4247.trainable_param_count(FakeModel()) == 7
    assert kwargs["target_modules"] == exp4247.STANDARD_LORA_TARGET_MODULES
    assert kwargs["exclude_modules"] == ["vision_tower"]
    assert exp4247.offline_training_gate(missing_attach).reason == "missing_lora_attach_path"


def test_scenario_code_4247_missing_or_nonfinite_loss_is_not_real_training() -> None:
    """SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING: missing/nonfinite losses fail."""

    missing = exp4247.OfflineTrainingSmokeResult(
        lora_attach_path=exp4247.STANDARD_ATTACH_PATH,
        trainable_param_count=10,
        steps_run=20,
        loss_initial=None,
        loss_final=1.0,
        loss_trace=[],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4247.working_lora_config(),
    )
    nonfinite = exp4247.OfflineTrainingSmokeResult(
        lora_attach_path=exp4247.STANDARD_ATTACH_PATH,
        trainable_param_count=10,
        steps_run=20,
        loss_initial=float("nan"),
        loss_final=1.0,
        loss_trace=[],
        duration_s=20.0,
        harness_smoke_passed=True,
        lora_config=exp4247.working_lora_config(),
    )

    assert exp4247.offline_training_gate(missing).reason == "missing_loss_trace"
    assert exp4247.offline_training_gate(nonfinite).reason == "non_finite_loss"


def test_scenario_code_4247_smoke_exception_is_blocked_not_progress(tmp_path: Path) -> None:
    """SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING: live exceptions fail loudly."""

    stable = _stable_checkpoint(tmp_path)
    cached = _cached_base(tmp_path)

    artifact = exp4247.run(
        output_path=tmp_path / "out.json",
        stable_checkpoint_path=stable,
        cuda_probe=lambda: True,
        cached_base_callback=lambda: cached,
        smoke_callback=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("oom")),
    )

    assert artifact["honest_verdict"] == "blocked_offline_reward_weighted_training_cannot_run_in_window"
    assert artifact["smoke_failure_reason"] == "RuntimeError: oom"
    assert artifact["harness_smoke_passed"] is False


def test_scenario_code_4247_weighted_loss_uses_reward_weight() -> None:
    """SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING: scalar losses are reward-weighted."""

    assert exp4247.weighted_loss_value(2.0, 0.25) == pytest.approx(0.5)
    assert exp4247.weighted_loss_value(2.0, 1.0) == pytest.approx(2.0)

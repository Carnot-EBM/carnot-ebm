"""Exp 4234 verifier-reward LoRA real-training smoke.

Spec refs: REQ-CODE-4234, SCENARIO-CODE-4234-BLOCKED-PRECONDITION,
SCENARIO-CODE-4234-REAL-TRAINING.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot import experiment_4222_verifier_reward_lora_harness_fix_smoke as exp4222


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4234_verifier_reward_lora_harness_real_training_smoke.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_STABLE_CHECKPOINT = exp4222.DEFAULT_STABLE_CHECKPOINT
THREE_ARM_RUNNER = REPO_ROOT / "scripts" / "experiments" / "verifier_reward_code_lora_rft_3arm.py"
RANDOM_SEED = 4198
FIXTURE_SIZE = 24
MIN_REAL_OPTIMIZER_STEPS = 20
DEFAULT_DURATION_FLOOR_S = 10.0
LOSS_MOVE_MARGIN = 1e-4
STANDARD_ATTACH_PATH = exp4222.STANDARD_ATTACH_PATH
WRAPPER_INNER_LINEAR_ATTACH_PATH = exp4222.WRAPPER_INNER_LINEAR_ATTACH_PATH
STANDARD_LORA_TARGET_MODULES = exp4222.STANDARD_LORA_TARGET_MODULES
INNER_LINEAR_LORA_TARGET_MODULES = exp4222.INNER_LINEAR_LORA_TARGET_MODULES
LORA_EXCLUDE_MODULES = exp4222.LORA_EXCLUDE_MODULES
APPROVED_NONQWEN_BASES = (
    "unsloth/gemma-4-12B-it",
    "google/gemma-4-12B-it",
    "google/gemma-4-E4B-it",
)
SPEC_REFS = [
    "REQ-CODE-4234",
    "SCENARIO-CODE-4234-BLOCKED-PRECONDITION",
    "SCENARIO-CODE-4234-REAL-TRAINING",
]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "harness_smoke_passed",
    "steps_run",
    "loss_initial",
    "loss_final",
    "lora_attach_path",
    "trainable_param_count",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A passing REAL-training smoke OR an honest blocked_* is COMPLETE -- "
        "B2 gates on the smoke so a non-training harness surfaces HERE, loudly, not as another "
        "fake-short progress artifact."
    ),
    "harness_smoke_passed": (
        "BARE bool: B2's gate compares this raw value (gated-fields-must-be-bare); true iff LoRA "
        "attached with >0 trainable params AND >=20 real optimizer steps ran AND loss moved "
        "(final<initial) AND wall-clock exceeded the plausibility floor."
    ),
    "steps_run": (
        "BARE int >=20 -- proves real optimizer steps ran (the .391 failure short-circuited before "
        "training); a low value means the harness still does not train."
    ),
    "loss_initial": "First-step training loss -- paired with loss_final to prove the loss actually MOVED.",
    "loss_final": (
        "Final-step training loss; loss_final < loss_initial by a margin is the real-training signal "
        "the .391 smoke lacked."
    ),
    "lora_attach_path": (
        "Which fix worked (standard AutoModelForCausalLM target_modules, or wrapper .linear patch) -- "
        "the diagnostic so B2 does not re-hit Gemma4ClippableLinear."
    ),
    "trainable_param_count": "BARE int >0 -- proves LoRA actually attached.",
    "verifier_is_oracle": (
        "BARE bool=true -- HONEST: the eventual reward is the execution oracle (RLVR/RLEF reward axis), "
        "NOT a moat claim (Circularity Discipline)."
    ),
    "model_specs": (
        "The NON-Qwen SOTA base + the working LoRA config; required methodology + the recipe B2 reuses."
    ),
    "random_seed": "Determinism precondition; torch + LoRA init seeded so the smoke is reproducible.",
    "reproducibility_checksum": (
        "Hash of the fixture + LoRA config; lets a third party confirm the same harness inputs."
    ),
}


@dataclass(frozen=True)
class CachedBase:
    """Approved standard HuggingFace base selected for real LoRA training."""

    model_id: str
    cache_path: Path


@dataclass(frozen=True)
class RealTrainingSmokeResult:
    """Serializable summary of the bounded real-training smoke."""

    lora_attach_path: str
    trainable_param_count: int
    steps_run: int
    loss_initial: float | None
    loss_final: float | None
    loss_trace: Sequence[Mapping[str, Any]]
    duration_s: float
    harness_smoke_passed: bool
    error: str | None = None
    lora_config: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _cuda_is_available() -> bool:  # pragma: no cover - live environment probe
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def hf_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def find_cached_nonqwen_base(
    *,
    model_ids: Sequence[str] = APPROVED_NONQWEN_BASES,
    hub_root: str | Path | None = None,
) -> CachedBase | None:
    root = Path(hub_root) if hub_root is not None else Path.home() / ".cache" / "huggingface" / "hub"
    for model_id in model_ids:
        lowered = model_id.lower()
        if "qwen" in lowered or "gguf" in lowered:
            continue
        cache_path = root / hf_cache_name(model_id)
        if cache_path.is_dir():
            return CachedBase(model_id=model_id, cache_path=cache_path)
    return None


def working_lora_config(target_modules: Sequence[str] | None = None) -> dict[str, Any]:
    return exp4222.working_lora_config(target_modules or STANDARD_LORA_TARGET_MODULES)


def load_or_build_fixture(stable_checkpoint_path: str | Path, *, fixture_size: int = FIXTURE_SIZE) -> exp4222.Fixture:
    return exp4222.load_or_build_fixture(stable_checkpoint_path, fixture_size=fixture_size)


def reproducibility_checksum(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    lora_config: Mapping[str, Any],
    model_id: str,
    random_seed: int,
) -> str:
    payload = {
        "fixture_rows": _jsonable(list(fixture_rows)),
        "lora_config": _jsonable(lora_config),
        "model_id": str(model_id),
        "random_seed": int(random_seed),
        "spec_refs": SPEC_REFS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def real_training_gate(
    smoke: RealTrainingSmokeResult,
    *,
    min_steps: int = MIN_REAL_OPTIMIZER_STEPS,
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S,
    min_loss_delta: float = LOSS_MOVE_MARGIN,
) -> GateResult:
    if int(smoke.trainable_param_count) <= 0:
        return GateResult(False, "blocked_no_trainable_lora_parameters")
    if int(smoke.steps_run) < int(min_steps):
        return GateResult(False, "insufficient_optimizer_steps")
    if smoke.loss_initial is None or smoke.loss_final is None:
        return GateResult(False, "missing_loss_trace")
    initial = float(smoke.loss_initial)
    final = float(smoke.loss_final)
    if not math.isfinite(initial) or not math.isfinite(final):
        return GateResult(False, "non_finite_loss")
    if final >= initial - float(min_loss_delta):
        return GateResult(False, "loss_did_not_move")
    if float(smoke.duration_s) < float(duration_floor_s):
        return GateResult(False, "duration_below_plausibility_floor")
    return GateResult(True, None)


def _empty_fixture() -> exp4222.Fixture:
    return exp4222.Fixture(rows=[], source="not_loaded_precondition_blocked", corpus_sizes={})


def _empty_smoke(error: str | None = None) -> RealTrainingSmokeResult:
    return RealTrainingSmokeResult(
        lora_attach_path="",
        trainable_param_count=0,
        steps_run=0,
        loss_initial=None,
        loss_final=None,
        loss_trace=[],
        duration_s=0.0,
        harness_smoke_passed=False,
        error=error,
        lora_config=working_lora_config(),
    )


def _coerce_smoke_result(raw: Mapping[str, Any]) -> RealTrainingSmokeResult:  # pragma: no cover - live adapter
    return RealTrainingSmokeResult(
        lora_attach_path=str(raw.get("lora_attach_path") or ""),
        trainable_param_count=int(raw.get("trainable_param_count") or 0),
        steps_run=int(raw.get("steps_run") or 0),
        loss_initial=raw.get("loss_initial") if isinstance(raw.get("loss_initial"), (int, float)) else None,
        loss_final=raw.get("loss_final") if isinstance(raw.get("loss_final"), (int, float)) else None,
        loss_trace=raw.get("loss_trace") if isinstance(raw.get("loss_trace"), list) else [],
        duration_s=float(raw.get("duration_s") or 0.0),
        harness_smoke_passed=raw.get("harness_smoke_passed") is True,
        error=raw.get("error") if isinstance(raw.get("error"), str) else None,
        lora_config=raw.get("lora_config") if isinstance(raw.get("lora_config"), Mapping) else working_lora_config(),
    )


def _load_runner_module() -> Any:  # pragma: no cover - live adapter
    spec = importlib.util.spec_from_file_location("verifier_reward_code_lora_rft_3arm", THREE_ARM_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner {THREE_ARM_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verifier_reward_code_lora_rft_3arm", module)
    spec.loader.exec_module(module)
    return module


def _run_live_smoke(
    fixture: exp4222.Fixture,
    *,
    cached_base: CachedBase,
    random_seed: int,
    min_steps: int,
    duration_floor_s: float,
) -> RealTrainingSmokeResult:  # pragma: no cover - live GPU/model path
    runner = _load_runner_module()
    raw = runner.run_real_training_smoke(
        fixture.rows,
        model_id=cached_base.model_id,
        seed=random_seed,
        min_steps=min_steps,
        duration_floor_s=duration_floor_s,
        min_loss_delta=LOSS_MOVE_MARGIN,
    )
    return _coerce_smoke_result(raw)


def _model_specs(
    *,
    cached_base: CachedBase | None,
    lora_config: Mapping[str, Any],
    lora_attach_path: str,
) -> dict[str, Any]:
    model_id = cached_base.model_id if cached_base is not None else ""
    cache_path = cached_base.cache_path if cached_base is not None else Path("")
    return {
        "trainable_base": model_id,
        "trainable_base_cache_path": str(cache_path),
        "trainable_base_is_non_qwen": bool(model_id and "qwen" not in model_id.lower()),
        "on_policy_generator": model_id,
        "qwen_train_base_forbidden": True,
        "load_method": f'transformers.AutoModelForCausalLM.from_pretrained("{model_id}")' if model_id else "",
        "lora_attach_path": lora_attach_path,
        "lora_config": _jsonable(lora_config),
        "runner": "scripts/experiments/verifier_reward_code_lora_rft_3arm.py",
    }


def build_artifact(
    *,
    preconditions: Mapping[str, Any],
    fixture: exp4222.Fixture,
    smoke: RealTrainingSmokeResult,
    cached_base: CachedBase | None,
    random_seed: int,
    duration_s: float,
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S,
) -> dict[str, Any]:
    gate = real_training_gate(smoke, duration_floor_s=duration_floor_s)
    gate_passed = bool(smoke.harness_smoke_passed and gate.passed and smoke.lora_attach_path)
    failure_reason = smoke.error or gate.reason
    if not preconditions.get("cuda_available", False):
        verdict = "blocked_cuda_unavailable"
    elif not preconditions.get("nonqwen_base_cached", False):
        verdict = "blocked_no_nonqwen_base_cached"
    elif gate_passed:
        verdict = "complete: verifier_reward_lora_real_training_smoke_passed"
    else:
        verdict = "blocked_lora_training_cannot_run_in_window"
    lora_config = dict(smoke.lora_config or working_lora_config())
    model_id = cached_base.model_id if cached_base is not None else ""
    checksum = reproducibility_checksum(
        fixture_rows=fixture.rows,
        lora_config=lora_config,
        model_id=model_id,
        random_seed=random_seed,
    )
    accepted = bool(
        (
            gate_passed
            and int(smoke.steps_run) >= MIN_REAL_OPTIMIZER_STEPS
            and smoke.loss_initial is not None
            and smoke.loss_final is not None
            and float(smoke.loss_final) < float(smoke.loss_initial)
            and int(smoke.trainable_param_count) > 0
            and smoke.lora_attach_path
        )
        or verdict.startswith("blocked_")
    )
    return {
        "experiment": "experiment_4234_verifier_reward_lora_harness_real_training_smoke",
        "schema": "carnot.experiment_4234_verifier_reward_lora_harness_real_training_smoke.v1",
        "honest_verdict": verdict,
        "harness_smoke_passed": gate_passed,
        "steps_run": int(smoke.steps_run),
        "loss_initial": smoke.loss_initial,
        "loss_final": smoke.loss_final,
        "lora_attach_path": str(smoke.lora_attach_path),
        "trainable_param_count": int(smoke.trainable_param_count),
        "verifier_is_oracle": True,
        "model_specs": _model_specs(cached_base=cached_base, lora_config=lora_config, lora_attach_path=smoke.lora_attach_path),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "loss_trace": _jsonable(smoke.loss_trace),
        "smoke_failure_reason": failure_reason,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "preconditions": _jsonable(preconditions),
        "fixture": {
            "source": fixture.source,
            "size": len(fixture.rows),
            "corpus_sizes": fixture.corpus_sizes,
        },
        "acceptance_gate": {
            "condition": (
                "harness_smoke_passed true with steps_run>=20 AND loss_final<loss_initial AND "
                "trainable_param_count>0 AND lora_attach_path recorded, OR an honest blocked_* verdict"
            ),
            "satisfied": accepted,
        },
        "duration_floor_s": float(duration_floor_s),
        "training_duration_s": round(float(smoke.duration_s), 6),
        "created_at": _utc_now(),
        "duration_s": round(float(duration_s), 6),
    }


def write_artifact(artifact: Mapping[str, Any], path: str | Path = DEFAULT_OUTPUT) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    stable_checkpoint_path: str | Path = DEFAULT_STABLE_CHECKPOINT,
    random_seed: int = RANDOM_SEED,
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S,
    cuda_probe: Callable[[], bool] = _cuda_is_available,
    cached_base_callback: Callable[[], CachedBase | None] = find_cached_nonqwen_base,
    smoke_callback: Callable[..., RealTrainingSmokeResult] | None = None,
) -> dict[str, Any]:
    started = time.time()
    stable = Path(stable_checkpoint_path)
    preconditions: dict[str, Any] = {
        "cuda_available": bool(cuda_probe()),
        "stable_checkpoint_path": str(stable),
        "qwen_train_base_forbidden": True,
        "fixture_size": 0,
        "fixture_source": "not_loaded_precondition_blocked",
    }
    cached_base: CachedBase | None = None
    fixture = _empty_fixture()
    smoke = _empty_smoke()

    if preconditions["cuda_available"]:
        cached_base = cached_base_callback()
        preconditions["nonqwen_base_cached"] = cached_base is not None
        preconditions["cached_base"] = _jsonable(cached_base)
        if cached_base is not None:
            fixture = load_or_build_fixture(stable, fixture_size=FIXTURE_SIZE)
            preconditions.update(
                {
                    "stable_checkpoint_readable": fixture.source == "stable_checkpoint_corpora",
                    "fixture_size": len(fixture.rows),
                    "fixture_source": fixture.source,
                    "corpus_sizes": fixture.corpus_sizes,
                }
            )
            try:
                smoke = (
                    smoke_callback(
                        fixture,
                        cached_base=cached_base,
                        random_seed=random_seed,
                        min_steps=MIN_REAL_OPTIMIZER_STEPS,
                        duration_floor_s=duration_floor_s,
                    )
                    if smoke_callback is not None
                    else _run_live_smoke(
                        fixture,
                        cached_base=cached_base,
                        random_seed=random_seed,
                        min_steps=MIN_REAL_OPTIMIZER_STEPS,
                        duration_floor_s=duration_floor_s,
                    )
                )
            except Exception as exc:
                smoke = _empty_smoke(f"{type(exc).__name__}: {exc}")
    else:
        preconditions["nonqwen_base_cached"] = False
        preconditions["cached_base"] = None

    artifact = build_artifact(
        preconditions=preconditions,
        fixture=fixture,
        smoke=smoke,
        cached_base=cached_base,
        random_seed=random_seed,
        duration_s=time.time() - started,
        duration_floor_s=duration_floor_s,
    )
    write_artifact(artifact, output_path)
    return _jsonable(artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stable-checkpoint", type=Path, default=DEFAULT_STABLE_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--duration-floor-s", type=float, default=DEFAULT_DURATION_FLOOR_S)
    args = parser.parse_args(argv)
    artifact = run(
        output_path=args.out,
        stable_checkpoint_path=args.stable_checkpoint,
        random_seed=args.seed,
        duration_floor_s=args.duration_floor_s,
    )
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(f"   harness_smoke_passed={artifact['harness_smoke_passed']}", flush=True)
    print(f"   steps_run={artifact['steps_run']}", flush=True)
    print(f"   loss_initial={artifact['loss_initial']} loss_final={artifact['loss_final']}", flush=True)
    return 0 if artifact["acceptance_gate"]["satisfied"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main(sys.argv[1:]))

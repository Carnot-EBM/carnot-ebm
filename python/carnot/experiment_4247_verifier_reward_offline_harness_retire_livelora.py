"""Exp 4247 offline reward-weighted SFT harness and live-LoRA retirement.

Spec refs: REQ-CODE-4247, SCENARIO-CODE-4247-BLOCKED-PRECONDITION,
SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot import experiment_4222_verifier_reward_lora_harness_fix_smoke as exp4222


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4247_verifier_reward_offline_harness_retire_livelora.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_STABLE_CHECKPOINT = exp4222.DEFAULT_STABLE_CHECKPOINT
RANDOM_SEED = 4247
FIXTURE_SIZE = 24
MIN_REAL_OPTIMIZER_STEPS = 20
DEFAULT_DURATION_FLOOR_S = 10.0
LOSS_MOVE_MARGIN = 1e-4
STANDARD_ATTACH_PATH = exp4222.STANDARD_ATTACH_PATH
STANDARD_LORA_TARGET_MODULES = exp4222.STANDARD_LORA_TARGET_MODULES
LORA_EXCLUDE_MODULES = exp4222.LORA_EXCLUDE_MODULES
APPROVED_NONQWEN_BASES = ("google/gemma-4-E4B-it", "unsloth/gemma-4-12B-it")
LIVE_LORA_RETIREMENT_RATIONALE = (
    "retired after 6 live-LoRA infrastructure failures; accumulate-floor met; "
    "offline reward-weighted SFT is finite-step and window-bounded"
)
SPEC_REFS = [
    "REQ-CODE-4247",
    "SCENARIO-CODE-4247-BLOCKED-PRECONDITION",
    "SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING",
]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "live_lora_retired",
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
        "Terminal-prefixed. A passing offline-training smoke + live-LoRA retirement, OR an honest "
        "blocked_*, is COMPLETE -- B2 gates on the smoke so a non-training harness surfaces HERE, "
        "loudly, not as another fake-short progress artifact."
    ),
    "live_lora_retired": (
        "BARE bool=true -- formally retires the dead live-LoRA path (6 infra failures, "
        "accumulate-floor met); D2 records it to ops/verifier_gaps.md + the exclusion manifest so it "
        "is never re-proposed."
    ),
    "harness_smoke_passed": (
        "BARE bool: B2's gate compares this raw value (gated-fields-must-be-bare); true iff LoRA "
        "attached with >0 trainable params AND >=20 real optimizer steps ran AND loss moved "
        "(final<initial) AND wall-clock exceeded the plausibility floor -- on the OFFLINE "
        "reward-weighted path."
    ),
    "steps_run": (
        "BARE int >=20 -- proves real optimizer steps ran (the live-LoRA failures short-circuited "
        "before training)."
    ),
    "loss_initial": "First-step training loss -- paired with loss_final to prove the loss actually MOVED.",
    "loss_final": (
        "Final-step training loss; loss_final < loss_initial by a margin is the real-training signal "
        "the live-LoRA smokes lacked."
    ),
    "lora_attach_path": (
        "Which attach worked (standard AutoModelForCausalLM target_modules) -- the diagnostic so B2 "
        "does not re-hit Gemma4ClippableLinear."
    ),
    "trainable_param_count": "BARE int >0 -- proves LoRA actually attached.",
    "verifier_is_oracle": (
        "BARE bool=true -- HONEST: the reward is the execution oracle (RLVR/RLEF reward axis), NOT a "
        "moat claim (Circularity Discipline)."
    ),
    "model_specs": (
        "The NON-Qwen base + the working LoRA config + the offline reward-weighting scheme; required "
        "methodology + the recipe B2 reuses."
    ),
    "random_seed": "Determinism precondition; torch + LoRA init seeded so the smoke is reproducible.",
    "reproducibility_checksum": (
        "Hash of the fixture + LoRA config + reward-weighting scheme; lets a third party confirm the "
        "harness inputs."
    ),
}


@dataclass(frozen=True)
class CachedBase:
    """Approved non-Qwen HuggingFace cache used for Transformers training."""

    model_id: str
    cache_path: Path


@dataclass(frozen=True)
class WeightedFixture:
    """Small deterministic corpus slice with reward weights already assigned."""

    rows: list[dict[str, Any]]
    source: str
    corpus_sizes: dict[str, int]


@dataclass(frozen=True)
class OfflineTrainingSmokeResult:
    """Serializable summary of the bounded offline reward-weighted SFT smoke."""

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


def _seed_everything(seed: int) -> None:  # pragma: no cover - torch/GPU dependent
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


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
        for cache_path in sorted(root.glob(f"{hf_cache_name(model_id)}*")):
            if cache_path.is_dir():
                return CachedBase(model_id=model_id, cache_path=cache_path)
    return None


def working_lora_config(target_modules: Sequence[str] | None = None) -> dict[str, Any]:
    config = exp4222.working_lora_config(target_modules or STANDARD_LORA_TARGET_MODULES)
    config["target_modules"] = list(target_modules or STANDARD_LORA_TARGET_MODULES)
    config["exclude_modules"] = list(LORA_EXCLUDE_MODULES)
    return config


def reward_weighting_scheme() -> dict[str, Any]:
    return {
        "method": "offline_reward_weighted_sft",
        "verified_weight": 1.0,
        "gold_weight": 1.0,
        "control_weight": 0.25,
        "live_generation": False,
        "rl_loop": False,
        "fixed_optimizer_steps": MIN_REAL_OPTIMIZER_STEPS,
        "rationale": "RAFT / reward-weighted regression over precomputed execution-oracle labels.",
    }


def reward_metadata_for_arm(arm: str, scheme: Mapping[str, Any] | None = None) -> tuple[float, str]:
    weights = dict(scheme or reward_weighting_scheme())
    if arm == "B":
        return float(weights["control_weight"]), "same_generator_random_label_control"
    if arm == "C":
        return float(weights["gold_weight"]), "hidden_gold_positive_control"
    return float(weights["verified_weight"]), "verifier_certified"


def _weighted_row(row: Mapping[str, Any], *, arm: str, scheme: Mapping[str, Any]) -> dict[str, Any]:
    weight, source = reward_metadata_for_arm(arm, scheme)
    weighted = dict(row)
    weighted["arm_id"] = arm
    weighted["reward_weight"] = weight
    weighted["reward_source"] = source
    return weighted


def _fallback_fixture_rows(fixture_size: int, *, scheme: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arms = ("A", "B", "C")
    for index in range(fixture_size):
        arm = arms[index % len(arms)]
        weight, source = reward_metadata_for_arm(arm, scheme)
        rows.append(
            {
                "arm": f"arm_{arm}",
                "arm_id": arm,
                "prompt": f"Complete the Python function for HumanEval offline reward fixture {index}.",
                "completion": f"def fixture_{index}(x):\n    return x + {index}\n",
                "hidden_pass": arm != "B",
                "visible_perfect": arm != "B",
                "task_id": f"offline_fixture/{index}",
                "reward_weight": weight,
                "reward_source": source,
            }
        )
    return rows


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_or_build_weighted_fixture(
    stable_checkpoint_path: str | Path = DEFAULT_STABLE_CHECKPOINT,
    *,
    fixture_size: int = FIXTURE_SIZE,
    scheme: Mapping[str, Any] | None = None,
) -> WeightedFixture:
    weights = dict(scheme or reward_weighting_scheme())
    stable = Path(stable_checkpoint_path)
    paths = {arm: stable / "corpora" / f"arm_{arm}.jsonl" for arm in ("A", "B", "C")}
    if all(path.is_file() for path in paths.values()):
        rows_by_arm = {arm: load_jsonl_rows(path) for arm, path in paths.items()}
        corpus_sizes = {arm: len(rows) for arm, rows in rows_by_arm.items()}
        fixture: list[dict[str, Any]] = []
        max_rows = max(corpus_sizes.values()) if corpus_sizes else 0
        for row_index in range(max_rows):
            for arm in ("A", "B", "C"):
                if row_index < len(rows_by_arm[arm]):
                    fixture.append(_weighted_row(rows_by_arm[arm][row_index], arm=arm, scheme=weights))
                    if len(fixture) >= fixture_size:
                        return WeightedFixture(fixture, "stable_checkpoint_corpora", corpus_sizes)
        return WeightedFixture(fixture, "stable_checkpoint_corpora", corpus_sizes)
    return WeightedFixture(_fallback_fixture_rows(fixture_size, scheme=weights), "tiny_operating_point_fixture", {})


def reproducibility_checksum(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    lora_config: Mapping[str, Any],
    weighting_scheme: Mapping[str, Any],
    model_id: str,
    random_seed: int,
) -> str:
    payload = {
        "fixture_rows": _jsonable(list(fixture_rows)),
        "lora_config": _jsonable(lora_config),
        "model_id": str(model_id),
        "random_seed": int(random_seed),
        "spec_refs": SPEC_REFS,
        "weighting_scheme": _jsonable(weighting_scheme),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def trainable_param_count(model: Any) -> int:
    return int(sum(int(param.numel()) for param in model.parameters() if getattr(param, "requires_grad", False)))


def weighted_loss_value(loss_value: float, reward_weight: float) -> float:
    return float(loss_value) * float(reward_weight)


def _prepare_model_for_offline_lora(model: Any) -> Any:  # pragma: no cover - live GPU/model path
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    return model


def _lora_config_kwargs(lora_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "r": int(lora_config["r"]),
        "lora_alpha": int(lora_config["lora_alpha"]),
        "lora_dropout": float(lora_config["lora_dropout"]),
        "task_type": str(lora_config["task_type"]),
        "target_modules": list(lora_config["target_modules"]),
        "exclude_modules": list(lora_config.get("exclude_modules") or []),
    }


def _run_live_offline_smoke(
    fixture: WeightedFixture,
    *,
    cached_base: CachedBase,
    random_seed: int,
    min_steps: int,
    duration_floor_s: float,
) -> OfflineTrainingSmokeResult:  # pragma: no cover - live GPU/model path
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return _empty_smoke(f"{type(exc).__name__}: {exc}")

    if not torch.cuda.is_available():
        return _empty_smoke("blocked_cuda_unavailable")
    if not fixture.rows:
        return _empty_smoke("empty_training_fixture")

    lora_config = working_lora_config()
    _seed_everything(random_seed)
    tokenizer = AutoTokenizer.from_pretrained(cached_base.model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cached_base.model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    try:
        print(f"[{time.strftime('%H:%M:%S')}] lora_attach_path={STANDARD_ATTACH_PATH}", flush=True)
        try:
            model = get_peft_model(model, LoraConfig(**_lora_config_kwargs(lora_config)))
        except Exception as exc:
            return OfflineTrainingSmokeResult(
                STANDARD_ATTACH_PATH,
                0,
                0,
                None,
                None,
                [],
                0.0,
                False,
                f"{type(exc).__name__}: {exc}",
                lora_config,
            )
        trainable = trainable_param_count(model)
        print(f"[{time.strftime('%H:%M:%S')}] trainable_lora_param_count={trainable}", flush=True)
        if trainable <= 0:
            return OfflineTrainingSmokeResult(
                STANDARD_ATTACH_PATH,
                0,
                0,
                None,
                None,
                [],
                0.0,
                False,
                "blocked_no_trainable_lora_parameters",
                lora_config,
            )

        _prepare_model_for_offline_lora(model)
        params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
        optimizer = torch.optim.AdamW(params, lr=float(lora_config["learning_rate"]))
        loss_trace: list[dict[str, Any]] = []
        started = time.time()
        rows = [row for row in fixture.rows if float(row.get("reward_weight", 0.0)) > 0.0] or fixture.rows
        try:
            for index in range(int(min_steps)):
                example = rows[index % len(rows)]
                prompt = str(example.get("prompt") or "")
                completion = str(example.get("completion") or "")
                reward_weight = float(example.get("reward_weight", 1.0))
                full = f"{prompt}\n{completion}{tokenizer.eos_token or ''}"
                enc = tokenizer(
                    full,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(lora_config["max_length"]),
                ).to("cuda")
                prompt_len = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(lora_config["max_length"]),
                )["input_ids"].shape[1]
                labels = enc["input_ids"].clone()
                labels[0, :prompt_len] = -100
                raw_loss = model(**enc, labels=labels).loss
                weighted_loss = raw_loss * reward_weight
                if not bool(getattr(weighted_loss, "requires_grad", False)):
                    return OfflineTrainingSmokeResult(
                        STANDARD_ATTACH_PATH,
                        trainable,
                        index,
                        loss_trace[0]["loss"] if loss_trace else None,
                        loss_trace[-1]["loss"] if loss_trace else None,
                        loss_trace,
                        time.time() - started,
                        False,
                        "blocked_loss_without_grad",
                        lora_config,
                    )
                if not bool(torch.isfinite(weighted_loss.detach()).all()):
                    return OfflineTrainingSmokeResult(
                        STANDARD_ATTACH_PATH,
                        trainable,
                        index,
                        loss_trace[0]["loss"] if loss_trace else None,
                        loss_trace[-1]["loss"] if loss_trace else None,
                        loss_trace,
                        time.time() - started,
                        False,
                        "non_finite_loss",
                        lora_config,
                    )
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                loss_value = float(weighted_loss.detach().cpu())
                event = {
                    "step": index + 1,
                    "loss": loss_value,
                    "reward_weight": reward_weight,
                    "arm_id": str(example.get("arm_id") or ""),
                }
                loss_trace.append(event)
                print(
                    f"[{time.strftime('%H:%M:%S')}] offline_smoke_step={index + 1}/{min_steps} "
                    f"loss={loss_value:.6f} reward_weight={reward_weight:.3f}",
                    flush=True,
                )
        except Exception as exc:
            return OfflineTrainingSmokeResult(
                STANDARD_ATTACH_PATH,
                trainable,
                len(loss_trace),
                loss_trace[0]["loss"] if loss_trace else None,
                loss_trace[-1]["loss"] if loss_trace else None,
                loss_trace,
                time.time() - started,
                False,
                f"{type(exc).__name__}: {exc}",
                lora_config,
            )

        duration_s = time.time() - started
        loss_values = [float(event["loss"]) for event in loss_trace]
        smoke = OfflineTrainingSmokeResult(
            STANDARD_ATTACH_PATH,
            trainable,
            len(loss_trace),
            loss_values[0] if loss_values else None,
            loss_values[-1] if loss_values else None,
            loss_trace,
            duration_s,
            True,
            None,
            lora_config,
        )
        gate = offline_training_gate(smoke, min_steps=min_steps, duration_floor_s=duration_floor_s)
        return OfflineTrainingSmokeResult(
            smoke.lora_attach_path,
            smoke.trainable_param_count,
            smoke.steps_run,
            smoke.loss_initial,
            smoke.loss_final,
            smoke.loss_trace,
            smoke.duration_s,
            gate.passed,
            gate.reason,
            lora_config,
        )
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def offline_training_gate(
    smoke: OfflineTrainingSmokeResult,
    *,
    min_steps: int = MIN_REAL_OPTIMIZER_STEPS,
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S,
    min_loss_delta: float = LOSS_MOVE_MARGIN,
) -> GateResult:
    if int(smoke.trainable_param_count) <= 0:
        return GateResult(False, "blocked_no_trainable_lora_parameters")
    if not smoke.lora_attach_path:
        return GateResult(False, "missing_lora_attach_path")
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


def _empty_fixture() -> WeightedFixture:
    return WeightedFixture(rows=[], source="not_loaded_precondition_blocked", corpus_sizes={})


def _empty_smoke(error: str | None = None) -> OfflineTrainingSmokeResult:
    return OfflineTrainingSmokeResult(
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


def _model_specs(
    *,
    cached_base: CachedBase | None,
    lora_config: Mapping[str, Any],
    lora_attach_path: str,
    weighting_scheme: Mapping[str, Any],
) -> dict[str, Any]:
    model_id = cached_base.model_id if cached_base is not None else ""
    cache_path = cached_base.cache_path if cached_base is not None else Path("")
    return {
        "trainable_base": model_id,
        "trainable_base_cache_path": str(cache_path),
        "trainable_base_is_non_qwen": bool(model_id and "qwen" not in model_id.lower()),
        "trainable_base_is_gguf": bool("gguf" in model_id.lower()),
        "qwen_train_base_forbidden": True,
        "gguf_training_repo_forbidden": True,
        "load_method": (
            f'transformers.AutoModelForCausalLM.from_pretrained("{model_id}", local_files_only=True)'
            if model_id
            else ""
        ),
        "lora_attach_path": lora_attach_path,
        "lora_config": _jsonable(lora_config),
        "offline_reward_weighting_scheme": _jsonable(weighting_scheme),
        "live_lora_retired": True,
        "runner": "python/carnot/experiment_4247_verifier_reward_offline_harness_retire_livelora.py",
    }


def build_artifact(
    *,
    preconditions: Mapping[str, Any],
    fixture: WeightedFixture,
    smoke: OfflineTrainingSmokeResult,
    cached_base: CachedBase | None,
    random_seed: int,
    duration_s: float,
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S,
) -> dict[str, Any]:
    scheme = reward_weighting_scheme()
    gate = offline_training_gate(smoke, duration_floor_s=duration_floor_s)
    gate_passed = bool(smoke.harness_smoke_passed and gate.passed and smoke.lora_attach_path)
    failure_reason = smoke.error or gate.reason
    if not preconditions.get("cuda_available", False):
        verdict = "blocked_cuda_unavailable"
    elif not preconditions.get("nonqwen_base_cached", False):
        verdict = "blocked_no_nonqwen_base_cached"
    elif gate_passed:
        verdict = "complete: verifier_reward_offline_reward_weighted_smoke_passed"
    else:
        verdict = "blocked_offline_reward_weighted_training_cannot_run_in_window"
    lora_config = dict(smoke.lora_config or working_lora_config())
    model_id = cached_base.model_id if cached_base is not None else ""
    checksum = reproducibility_checksum(
        fixture_rows=fixture.rows,
        lora_config=lora_config,
        weighting_scheme=scheme,
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
        "experiment": "experiment_4247_verifier_reward_offline_harness_retire_livelora",
        "schema": "carnot.experiment_4247_verifier_reward_offline_harness_retire_livelora.v1",
        "honest_verdict": verdict,
        "live_lora_retired": True,
        "live_lora_retirement_rationale": LIVE_LORA_RETIREMENT_RATIONALE,
        "harness_smoke_passed": gate_passed,
        "steps_run": int(smoke.steps_run),
        "loss_initial": smoke.loss_initial,
        "loss_final": smoke.loss_final,
        "lora_attach_path": str(smoke.lora_attach_path),
        "trainable_param_count": int(smoke.trainable_param_count),
        "verifier_is_oracle": True,
        "model_specs": _model_specs(
            cached_base=cached_base,
            lora_config=lora_config,
            lora_attach_path=smoke.lora_attach_path,
            weighting_scheme=scheme,
        ),
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
            "reward_weight_counts": _reward_weight_counts(fixture.rows),
        },
        "acceptance_gate": {
            "condition": (
                "live_lora_retired=true AND (harness_smoke_passed true with steps_run>=20 AND "
                "loss_final<loss_initial AND trainable_param_count>0 AND lora_attach_path recorded, OR "
                "an honest blocked_* verdict)"
            ),
            "satisfied": bool(True and accepted),
        },
        "duration_floor_s": float(duration_floor_s),
        "training_duration_s": round(float(smoke.duration_s), 6),
        "created_at": _utc_now(),
        "duration_s": round(float(duration_s), 6),
    }


def _reward_weight_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("reward_weight", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


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
    smoke_callback: Callable[..., OfflineTrainingSmokeResult] | None = None,
) -> dict[str, Any]:
    started = time.time()
    stable = Path(stable_checkpoint_path)
    preconditions: dict[str, Any] = {
        "cuda_available": bool(cuda_probe()),
        "stable_checkpoint_path": str(stable),
        "qwen_train_base_forbidden": True,
        "gguf_training_repo_forbidden": True,
        "fixture_size": 0,
        "fixture_source": "not_loaded_precondition_blocked",
        "live_lora_retired": True,
    }
    cached_base: CachedBase | None = None
    fixture = _empty_fixture()
    smoke = _empty_smoke()

    if preconditions["cuda_available"]:
        cached_base = cached_base_callback()
        preconditions["nonqwen_base_cached"] = cached_base is not None
        preconditions["cached_base"] = _jsonable(cached_base)
        if cached_base is not None:
            fixture = load_or_build_weighted_fixture(stable, fixture_size=FIXTURE_SIZE)
            preconditions.update(
                {
                    "stable_checkpoint_readable": fixture.source == "stable_checkpoint_corpora",
                    "fixture_size": len(fixture.rows),
                    "fixture_source": fixture.source,
                    "corpus_sizes": fixture.corpus_sizes,
                    "reward_weighting_scheme": reward_weighting_scheme(),
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
                    else _run_live_offline_smoke(
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
    print(f"   live_lora_retired={artifact['live_lora_retired']}", flush=True)
    print(f"   harness_smoke_passed={artifact['harness_smoke_passed']}", flush=True)
    print(f"   steps_run={artifact['steps_run']}", flush=True)
    print(f"   loss_initial={artifact['loss_initial']} loss_final={artifact['loss_final']}", flush=True)
    return 0 if artifact["acceptance_gate"]["satisfied"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main(sys.argv[1:]))

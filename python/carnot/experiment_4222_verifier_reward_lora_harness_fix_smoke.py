"""Exp 4222 verifier-reward LoRA attach smoke.

Spec refs: REQ-CODE-4222, SCENARIO-CODE-4222-BLOCKED-PRECONDITION,
SCENARIO-CODE-4222-STANDARD-LORA-ATTACH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4222_verifier_reward_lora_harness_fix_smoke.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_STABLE_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "verifier_reward_3arm_lora_rft"
    / "code_verifier_reward_lora_rft_a83b52882c198954"
)
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_CACHE_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-E4B-it"
RANDOM_SEED = 4198
FIXTURE_SIZE = 8
STANDARD_ATTACH_PATH = "standard_auto_model_for_causal_lm_target_modules"
WRAPPER_INNER_LINEAR_ATTACH_PATH = "wrapper_inner_linear_target_modules"
STANDARD_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
INNER_LINEAR_LORA_TARGET_MODULES = [f"{name}.linear" for name in STANDARD_LORA_TARGET_MODULES]
LORA_EXCLUDE_MODULES = ["vision_tower"]
SPEC_REFS = [
    "REQ-CODE-4222",
    "SCENARIO-CODE-4222-BLOCKED-PRECONDITION",
    "SCENARIO-CODE-4222-STANDARD-LORA-ATTACH",
]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "harness_smoke_passed",
    "lora_attach_path",
    "trainable_param_count",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A passing harness smoke OR an honest blocked_* is COMPLETE -- B2 gates on "
        "the smoke so a broken harness surfaces HERE, not as another failed training run."
    ),
    "harness_smoke_passed": (
        "BARE bool: B2's gate compares this raw value (gated-fields-must-be-bare); true iff LoRA "
        "attached with >0 trainable params AND 1 training step ran AND loss was finite on the fixture."
    ),
    "lora_attach_path": (
        "Which fix worked (standard AutoModelForCausalLM target_modules, or wrapper .linear patch) -- "
        "the load-bearing diagnostic so B2 and future runs do not re-hit the Gemma4ClippableLinear rejection."
    ),
    "trainable_param_count": (
        "BARE int >0 -- proves LoRA actually attached (the .390 failure was 0 attach); a zero here means "
        "the fix did not take."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- HONEST: the eventual reward is the execution oracle (RLVR/RLEF reward axis), "
        "NOT a moat claim (Circularity Discipline)."
    ),
    "model_specs": (
        "The NON-Qwen base + the working LoRA config (target_modules that attached); required methodology "
        "+ the recipe B2 reuses."
    ),
    "random_seed": "Determinism precondition; torch + LoRA init seeded so the smoke is reproducible.",
    "reproducibility_checksum": (
        "Hash of the fixture + LoRA config; lets a third party confirm the same harness inputs."
    ),
}


@dataclass(frozen=True)
class Fixture:
    rows: list[dict[str, Any]]
    source: str
    corpus_sizes: dict[str, int]


@dataclass(frozen=True)
class SmokeResult:
    harness_smoke_passed: bool
    lora_attach_path: str
    trainable_param_count: int
    loss: float | None
    progress_events: list[dict[str, Any]]
    error: str | None = None
    lora_config: dict[str, Any] | None = None


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


def _seed_torch(seed: int) -> None:  # pragma: no cover - torch install/GPU dependent
    import random

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def working_lora_config(target_modules: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "method": "LoRA-SFT",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "max_length": 1024,
        "target_modules": list(target_modules or STANDARD_LORA_TARGET_MODULES),
        "exclude_modules": list(LORA_EXCLUDE_MODULES),
    }


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


def _fallback_fixture_rows(fixture_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(fixture_size):
        rows.append(
            {
                "arm": "fixture_operating_point",
                "prompt": f"Complete the Python function for HumanEval fixture {index}.",
                "completion": f"def fixture_{index}(x):\n    return x + {index}\n",
                "hidden_pass": True,
                "visible_perfect": True,
                "task_id": f"fixture/{index}",
            }
        )
    return rows


def load_or_build_fixture(
    stable_checkpoint_path: str | Path = DEFAULT_STABLE_CHECKPOINT,
    *,
    fixture_size: int = FIXTURE_SIZE,
) -> Fixture:
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
                    fixture.append(dict(rows_by_arm[arm][row_index]))
                    if len(fixture) >= fixture_size:
                        return Fixture(rows=fixture, source="stable_checkpoint_corpora", corpus_sizes=corpus_sizes)
        return Fixture(rows=fixture, source="stable_checkpoint_corpora", corpus_sizes=corpus_sizes)
    return Fixture(rows=_fallback_fixture_rows(fixture_size), source="tiny_operating_point_fixture", corpus_sizes={})


def reproducibility_checksum(
    fixture_rows: Sequence[Mapping[str, Any]],
    lora_config: Mapping[str, Any],
    *,
    random_seed: int,
) -> str:
    payload = {
        "fixture_rows": _jsonable(list(fixture_rows)),
        "lora_config": _jsonable(lora_config),
        "random_seed": int(random_seed),
        "spec_refs": SPEC_REFS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def trainable_param_count(model: Any) -> int:
    return int(sum(int(param.numel()) for param in model.parameters() if getattr(param, "requires_grad", False)))


def _lora_config_kwargs(lora_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "r": int(lora_config["r"]),
        "lora_alpha": int(lora_config["lora_alpha"]),
        "lora_dropout": float(lora_config["lora_dropout"]),
        "task_type": str(lora_config["task_type"]),
        "target_modules": list(lora_config["target_modules"]),
        "exclude_modules": list(lora_config.get("exclude_modules") or []),
    }


def _live_training_step(
    model: Any,
    tokenizer: Any,
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    lora_config: Mapping[str, Any],
    random_seed: int,
) -> float:  # pragma: no cover - live GPU/model path
    import torch

    _seed_torch(random_seed)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    example = fixture_rows[0]
    prompt = str(example.get("prompt") or "")
    completion = str(example.get("completion") or "")
    eos = tokenizer.eos_token or ""
    full_text = f"{prompt}\n{completion}{eos}"
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=int(lora_config["max_length"])).to("cuda")
    prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(lora_config["max_length"]))[
        "input_ids"
    ].shape[1]
    labels = enc["input_ids"].clone()
    labels[0, :prompt_len] = -100
    params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
    optimizer = torch.optim.AdamW(params, lr=float(lora_config["learning_rate"]))
    loss = model(**enc, labels=labels).loss
    if not bool(torch.isfinite(loss.detach()).all()):
        return float("nan")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())


def run_lora_attach_and_step(
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    random_seed: int,
    load_model: Callable[[str], Any] | None = None,
    load_tokenizer: Callable[[str], Any] | None = None,
    lora_config_cls: Callable[..., Any] | None = None,
    get_peft_model_fn: Callable[[Any, Any], Any] | None = None,
    step_fn: Callable[..., float] | None = None,
    seed_fn: Callable[[int], None] | None = None,
) -> SmokeResult:
    if load_model is None or load_tokenizer is None or lora_config_cls is None or get_peft_model_fn is None:
        import torch  # pragma: no cover
        from peft import LoraConfig, get_peft_model  # pragma: no cover
        from transformers import AutoModelForCausalLM, AutoTokenizer  # pragma: no cover

        load_model = lambda selected_model_id: AutoModelForCausalLM.from_pretrained(  # pragma: no cover
            selected_model_id,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        load_tokenizer = AutoTokenizer.from_pretrained  # pragma: no cover
        lora_config_cls = LoraConfig  # pragma: no cover
        get_peft_model_fn = get_peft_model  # pragma: no cover
    if step_fn is None:
        step_fn = _live_training_step

    (seed_fn or _seed_torch)(random_seed)
    model = load_model(model_id)
    tokenizer = load_tokenizer(model_id)
    attached_path = STANDARD_ATTACH_PATH
    attached_lora_config = working_lora_config()
    for target_modules, attach_path in (
        (STANDARD_LORA_TARGET_MODULES, STANDARD_ATTACH_PATH),
        (INNER_LINEAR_LORA_TARGET_MODULES, WRAPPER_INNER_LINEAR_ATTACH_PATH),
    ):
        lora_config = working_lora_config(target_modules)
        peft_config = lora_config_cls(**_lora_config_kwargs(lora_config))
        print(f"lora_attach_path={attach_path}", flush=True)
        try:
            model = get_peft_model_fn(model, peft_config)
            attached_path = attach_path
            attached_lora_config = lora_config
            break
        except ValueError as exc:
            if attach_path == STANDARD_ATTACH_PATH and "Gemma4ClippableLinear" in str(exc):
                print("standard attach rejected Gemma4ClippableLinear; retrying inner .linear modules", flush=True)
                continue
            raise
    trainable = trainable_param_count(model)
    print(f"trainable_param_count={trainable}", flush=True)
    if trainable <= 0:
        return SmokeResult(
            False,
            attached_path,
            0,
            None,
            [],
            "blocked_no_trainable_lora_parameters",
            attached_lora_config,
        )
    loss = float(
        step_fn(
            model,
            tokenizer,
            fixture_rows,
            lora_config=attached_lora_config,
            random_seed=random_seed,
        )
    )
    finite = math.isfinite(loss)
    progress = [{"step": 1, "loss": loss, "finite_loss": finite}]
    print(f"step=1 loss={loss:.6f} finite_loss={finite}", flush=True)
    return SmokeResult(
        finite,
        attached_path,
        trainable,
        loss,
        progress,
        None if finite else "non_finite_loss",
        attached_lora_config,
    )


def _model_specs(cache_path: Path, lora_config: Mapping[str, Any], *, lora_attach_path: str) -> dict[str, Any]:
    return {
        "trainable_base": MODEL_ID,
        "trainable_base_cache_path": str(cache_path),
        "trainable_base_is_non_qwen": True,
        "on_policy_generator": MODEL_ID,
        "qwen_train_base_forbidden": True,
        "load_method": 'transformers.AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it")',
        "lora_attach_path": lora_attach_path,
        "lora_config": _jsonable(lora_config),
        "runner": "scripts/experiments/verifier_reward_code_lora_rft_3arm.py",
    }


def build_artifact(
    *,
    preconditions: Mapping[str, Any],
    fixture: Fixture,
    smoke: SmokeResult,
    cache_path: Path,
    lora_config: Mapping[str, Any],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    if not preconditions.get("cuda_available", False):
        verdict = "blocked_cuda_unavailable"
    elif not preconditions.get("nonqwen_base_cached", False):
        verdict = "blocked_no_nonqwen_base_cached"
    elif smoke.harness_smoke_passed and smoke.trainable_param_count > 0:
        verdict = "complete: verifier_reward_lora_harness_smoke_passed"
    else:
        verdict = "failed: verifier_reward_lora_harness_smoke_failed"
    attached_lora_config = smoke.lora_config or dict(lora_config)
    checksum = reproducibility_checksum(fixture.rows, attached_lora_config, random_seed=random_seed)
    accepted = bool(
        (smoke.harness_smoke_passed and smoke.trainable_param_count > 0 and smoke.lora_attach_path)
        or verdict.startswith("blocked_")
    )
    return {
        "experiment": "experiment_4222_verifier_reward_lora_harness_fix_smoke",
        "schema": "carnot.experiment_4222_verifier_reward_lora_harness_fix_smoke.v1",
        "honest_verdict": verdict,
        "harness_smoke_passed": bool(smoke.harness_smoke_passed),
        "lora_attach_path": str(smoke.lora_attach_path),
        "trainable_param_count": int(smoke.trainable_param_count),
        "verifier_is_oracle": True,
        "model_specs": _model_specs(cache_path, attached_lora_config, lora_attach_path=smoke.lora_attach_path),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "preconditions": _jsonable(preconditions),
        "fixture": {
            "source": fixture.source,
            "size": len(fixture.rows),
            "corpus_sizes": fixture.corpus_sizes,
        },
        "training_step": {
            "loss": smoke.loss,
            "finite_loss": bool(smoke.loss is not None and math.isfinite(float(smoke.loss))),
            "progress_events": _jsonable(smoke.progress_events),
            "error": smoke.error,
        },
        "acceptance_gate": {
            "condition": (
                "harness_smoke_passed true with trainable_param_count>0 and lora_attach_path recorded, "
                "OR an honest blocked_* verdict"
            ),
            "satisfied": accepted,
        },
        "honest_verdict_principle": FIELD_PRINCIPLES["honest_verdict"],
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
    cache_path: str | Path = MODEL_CACHE_PATH,
    random_seed: int = RANDOM_SEED,
    cuda_probe: Callable[[], bool] = _cuda_is_available,
    smoke_callback: Callable[..., SmokeResult] | None = None,
) -> dict[str, Any]:
    started = time.time()
    cache = Path(cache_path)
    fixture = load_or_build_fixture(stable_checkpoint_path, fixture_size=FIXTURE_SIZE)
    lora_config = working_lora_config()
    preconditions = {
        "cuda_available": bool(cuda_probe()),
        "cached_base": {"model_id": MODEL_ID, "cache_path": str(cache)},
        "nonqwen_base_cached": bool(cache.is_dir()),
        "qwen_train_base_forbidden": True,
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "stable_checkpoint_readable": fixture.source == "stable_checkpoint_corpora",
        "fixture_size": len(fixture.rows),
        "fixture_source": fixture.source,
    }
    smoke = SmokeResult(False, STANDARD_ATTACH_PATH, 0, None, [])
    if preconditions["cuda_available"] and preconditions["nonqwen_base_cached"]:
        try:
            if smoke_callback is None:
                smoke = run_lora_attach_and_step(fixture.rows, model_id=MODEL_ID, random_seed=random_seed)
            else:
                smoke = smoke_callback(fixture, random_seed=random_seed)
        except Exception as exc:
            smoke = SmokeResult(
                False,
                STANDARD_ATTACH_PATH,
                0,
                None,
                [],
                f"{type(exc).__name__}: {exc}",
            )
    artifact = build_artifact(
        preconditions=preconditions,
        fixture=fixture,
        smoke=smoke,
        cache_path=cache,
        lora_config=lora_config,
        random_seed=random_seed,
        duration_s=time.time() - started,
    )
    write_artifact(artifact, output_path)
    return _jsonable(artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stable-checkpoint", type=Path, default=DEFAULT_STABLE_CHECKPOINT)
    parser.add_argument("--cache-path", type=Path, default=MODEL_CACHE_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    artifact = run(
        output_path=args.out,
        stable_checkpoint_path=args.stable_checkpoint,
        cache_path=args.cache_path,
        random_seed=args.seed,
    )
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(f"   harness_smoke_passed={artifact['harness_smoke_passed']}", flush=True)
    print(f"   trainable_param_count={artifact['trainable_param_count']}", flush=True)
    return 0 if artifact["acceptance_gate"]["satisfied"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main(sys.argv[1:]))

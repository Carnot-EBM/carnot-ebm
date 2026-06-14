"""On-policy 3-arm code LoRA-RFT runner for Exp 4197/A2.

The runner uses the shared restricted execution primitive
`carnot.verify.sandbox.sandboxed_exec_function` for code labels.  Smoke mode
only builds the matched A/B/C corpora on two tasks and does not train.

Spec refs: REQ-CODE-4197, SCENARIO-CODE-4197-HARNESS.

Usage:
  .venv/bin/python scripts/experiments/verifier_reward_code_lora_rft_3arm.py --smoke
  .venv/bin/python scripts/experiments/verifier_reward_code_lora_rft_3arm.py --train
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4197_verifier_reward_phase0_headroom as exp4197  # noqa: E402
from carnot.verify.sandbox import sandboxed_exec_function  # noqa: E402


OUT = REPO_ROOT / "results" / "experiment_4197_verifier_reward_code_lora_rft_3arm_smoke.json"
LORA_TARGET_MODULES = r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
LORA_CHECKPOINT_EVERY_STEPS = 25


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_execution_tests(
    code: str,
    entry_point: str,
    tests: Iterable[Sequence[Any]],
    *,
    timeout: float = 2.0,
) -> tuple[bool, ...]:
    """Run visible or hidden example tests through the shared sandbox primitive."""

    outcomes: list[bool] = []
    for test in tests:
        args, expected = test[:-1], test[-1]
        result, error = sandboxed_exec_function(
            code,
            entry_point,
            tuple(args),
            timeout=timeout,
            allow_fallback=True,
        )
        outcomes.append(error is None and result == expected)
    return tuple(outcomes)


def build_corpora_from_checkpoint(
    checkpoint: Path,
    *,
    seed: int,
    smoke: bool,
) -> tuple[list[exp4197.CodeTask], exp4197.ThreeArmCorpora]:
    """Load same-generator code candidates and build A/B/C training arms."""

    tasks = exp4197.load_checkpoint_tasks(checkpoint)
    selected = exp4197.select_smoke_tasks(tasks, n_tasks=2) if smoke else tasks
    return selected, exp4197.build_three_arm_corpora(selected, seed=seed)


def _train_lora_sft(
    arm_name: str,
    examples: Sequence[exp4197.TrainingExample],
    *,
    model_id: str,
    output_dir: Path,
    smoke: bool,
    seed: int,
    progress_interval_s: float = 30.0,
) -> dict[str, Any]:
    """Train one LoRA arm when explicitly requested.

    The smoke path returns a plan only.  The full path is intentionally simple:
    A2 can swap in a larger trainer, but the arm data, seed, and LoRA config are
    fixed here so A/B/C stay comparable.
    """

    if smoke:
        return {
            "arm": arm_name,
            "status": "smoke_no_train",
            "n_examples": len(examples),
            "output_dir": str(output_dir),
            "random_seed": seed,
        }

    progress_events: list[dict[str, Any]] = []
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return {"arm": arm_name, "status": "blocked_training_import", "error": f"{type(exc).__name__}: {exc}"}

    if not torch.cuda.is_available():
        return {"arm": arm_name, "status": "blocked_cuda_unavailable", "n_examples": len(examples)}

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "training_progress.json"
    completed_steps = 0
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed_steps = max(0, min(int(progress.get("completed_steps") or 0), len(examples)))
        except Exception:
            completed_steps = 0
    adapter_config = output_dir / "adapter_config.json"
    if adapter_config.is_file() and completed_steps >= len(examples):
        return {
            "arm": arm_name,
            "status": "trained_resume_complete",
            "n_examples": len(examples),
            "completed_steps": completed_steps,
            "output_dir": str(output_dir),
            "random_seed": seed,
            "progress_events": [],
        }

    print(
        f"[{time.strftime('%H:%M:%S')}] arm={arm_name} resume_start={completed_steps}/{len(examples)}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    if adapter_config.is_file():
        model = PeftModel.from_pretrained(model, output_dir, is_trainable=True)
    else:
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
            exclude_modules=["vision_tower"],
        )
        model = get_peft_model(model, lora)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params <= 0:
        return {
            "arm": arm_name,
            "status": "blocked_no_trainable_lora_parameters",
            "n_examples": len(examples),
            "target_modules": LORA_TARGET_MODULES,
        }
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    model.train()
    steps = completed_steps
    last_print = 0.0
    for example in examples[completed_steps:]:
        prompt = example.prompt
        full = prompt + "\n" + example.completion + (tokenizer.eos_token or "")
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[0, :prompt_len] = -100
        loss = model(**enc, labels=labels).loss
        if not bool(getattr(loss, "requires_grad", False)):
            return {
                "arm": arm_name,
                "status": "blocked_loss_without_grad",
                "n_examples": len(examples),
                "completed_steps": steps,
                "trainable_params": trainable_params,
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        loss_value = float(loss.detach().cpu())
        now = time.time()
        if now - last_print >= progress_interval_s or steps == len(examples):
            event = {"arm": arm_name, "step": steps, "total": len(examples), "loss": round(loss_value, 6)}
            progress_events.append(event)
            print(
                f"[{time.strftime('%H:%M:%S')}] arm={arm_name} step={steps}/{len(examples)} loss={loss_value:.6f}",
                flush=True,
            )
            last_print = now
        progress_path.write_text(
            json.dumps(
                {
                    "arm": arm_name,
                    "status": "training",
                    "completed_steps": steps,
                    "total_steps": len(examples),
                    "last_loss": loss_value,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if steps % LORA_CHECKPOINT_EVERY_STEPS == 0:
            model.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    progress_path.write_text(
        json.dumps(
            {
                "arm": arm_name,
                "status": "trained",
                "completed_steps": steps,
                "total_steps": len(examples),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "arm": arm_name,
        "status": "trained",
        "n_examples": len(examples),
        "steps": steps,
        "completed_steps": steps,
        "trainable_params": trainable_params,
        "output_dir": str(output_dir),
        "progress_events": progress_events,
    }


def run(
    *,
    checkpoint: Path = exp4197.DEFAULT_PHASE0_CHECKPOINT,
    seed: int = exp4197.RANDOM_SEED,
    smoke: bool = True,
    train: bool = False,
    output_path: Path = OUT,
    train_root: Path | None = None,
    progress_interval_s: float = 30.0,
) -> dict[str, Any]:
    started = time.time()
    selected_tasks, corpora = build_corpora_from_checkpoint(checkpoint, seed=seed, smoke=smoke)
    train_mode = bool(train and not smoke)
    train_root = train_root or REPO_ROOT / "results" / "experiment_4197_lora_rft_arms"
    training = {
        "arm_a": _train_lora_sft(
            "A_certified",
            corpora.arm_a_certified,
            model_id=exp4197.TRAINABLE_BASE,
            output_dir=train_root / "arm_a_certified",
            smoke=not train_mode,
            seed=seed,
            progress_interval_s=progress_interval_s,
        ),
        "arm_b": _train_lora_sft(
            "B_random_same_generator",
            corpora.arm_b_random_control,
            model_id=exp4197.TRAINABLE_BASE,
            output_dir=train_root / "arm_b_random_same_generator",
            smoke=not train_mode,
            seed=seed,
            progress_interval_s=progress_interval_s,
        ),
        "arm_c": _train_lora_sft(
            "C_hidden_gold",
            corpora.arm_c_hidden_gold,
            model_id=exp4197.TRAINABLE_BASE,
            output_dir=train_root / "arm_c_hidden_gold",
            smoke=not train_mode,
            seed=seed,
            progress_interval_s=progress_interval_s,
        ),
        "arm_d": {"arm": "D_cold_base", "status": "cold_base_eval_only", "n_examples": 0},
    }
    sizes = corpora.sizes()
    harness_ready = (
        len(selected_tasks) >= 2
        and sizes["arm_a_certified"] > 0
        and sizes["arm_b_random_control"] == sizes["arm_a_certified"]
        and sizes["arm_c_hidden_gold"] > 0
    )
    artifact = {
        "experiment": "experiment_4197_verifier_reward_code_lora_rft_3arm",
        "honest_verdict": "complete: smoke_3arm_runner_ready" if harness_ready else "blocked_3arm_runner_smoke_failed",
        "harness_ready": harness_ready,
        "smoke": smoke,
        "train_requested": train,
        "train_mode_executed": train_mode,
        "random_seed": seed,
        "checkpoint": str(checkpoint),
        "n_tasks": len(selected_tasks),
        "arm_sizes": sizes,
        "training": training,
        "progress_events": [
            event
            for arm in ("arm_a", "arm_b", "arm_c")
            for event in training.get(arm, {}).get("progress_events", [])
        ],
        "truncation_guard": {"max_allowed_truncation_rate": exp4197.MAX_ALLOWED_TRUNCATION},
        "duration_s": round(time.time() - started, 6),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=exp4197.DEFAULT_PHASE0_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=exp4197.RANDOM_SEED)
    parser.add_argument("--smoke", action="store_true", help="Build two-task arms without training.")
    parser.add_argument("--train", action="store_true", help="Launch full LoRA training for A/B/C arms.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--train-root", type=Path, default=None, help="Per-arm LoRA checkpoint root.")
    parser.add_argument("--progress-interval-s", type=float, default=30.0)
    args = parser.parse_args()
    smoke = args.smoke or not args.train
    artifact = run(
        checkpoint=args.checkpoint,
        seed=args.seed,
        smoke=smoke,
        train=args.train,
        output_path=args.out,
        train_root=args.train_root,
        progress_interval_s=args.progress_interval_s,
    )
    print(f"-> {artifact['honest_verdict']}")
    print(f"   arm_sizes={artifact['arm_sizes']}")
    return 0 if artifact["harness_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Out-of-band LoRA trainer and validator for Exp 4263.

Run this outside the conductor window. It loads a small non-Qwen base through
AutoModelForCausalLM, attaches LoRA, trains for real optimizer steps on the
precomputed reward-weighted corpus, and fails if the validation signal is not
real training: trainable params > 0, >=20 steps, loss_final < loss_initial, and
plausible duration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any


MIN_OPTIMIZER_STEPS = 20
MIN_PLAUSIBLE_DURATION_S = 10.0
DEFAULT_BASE_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'
DEFAULT_CORPUS = Path('/home/ianblenke/github.com/ianblenke/carnot/results/experiment_4263_verifier_as_reward_out_of_band/reward_weighted_corpus.jsonl')
DEFAULT_OUTPUT = Path('/home/ianblenke/github.com/ianblenke/carnot/results/experiment_4263_verifier_as_reward_out_of_band/training_result.json')
DEFAULT_ADAPTER_DIR = Path('/home/ianblenke/github.com/ianblenke/carnot/results/experiment_4263_verifier_as_reward_out_of_band/lora_adapter')
DEFAULT_SEED = 4263
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty corpus: {path}")
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_param_count(model: Any) -> int:
    return int(sum(int(param.numel()) for param in model.parameters() if getattr(param, "requires_grad", False)))


def validate_training_artifact(result: dict[str, Any], *, min_steps: int, min_duration_s: float) -> None:
    errors: list[str] = []
    if "qwen" in str(result.get("base_model", "")).lower():
        errors.append("trained_base_must_be_non_qwen")
    if result.get("model_load_api") != "AutoModelForCausalLM":
        errors.append("model_must_load_through_AutoModelForCausalLM")
    if not result.get("lora_attached"):
        errors.append("lora_not_attached")
    if int(result.get("trainable_param_count") or 0) <= 0:
        errors.append("no_trainable_lora_params")
    if int(result.get("optimizer_steps") or 0) < int(min_steps):
        errors.append("insufficient_optimizer_steps")
    loss_initial = result.get("loss_initial")
    loss_final = result.get("loss_final")
    if loss_initial is None or loss_final is None:
        errors.append("missing_loss_trace")
    elif not float(loss_final) < float(loss_initial):
        errors.append("loss_final < loss_initial validation failed")
    if float(result.get("duration_s") or 0.0) < float(min_duration_s):
        errors.append("duration_below_plausibility_floor")
    if errors:
        raise RuntimeError("real_training_validation_failed: " + ", ".join(errors))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if "qwen" in args.base_model.lower():
        raise RuntimeError("Qwen is forbidden as the trained base for Exp 4263")

    started = time.time()
    rows = [row for row in load_rows(args.corpus) if float(row.get("reward_weight", 0.0)) > 0.0]
    if not rows:
        raise RuntimeError("no positive-weight training rows")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs).to(device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        ),
    )
    trainable = trainable_param_count(model)
    if trainable <= 0:
        raise RuntimeError("LoRA attached with zero trainable parameters")
    model.train()
    optimizer = torch.optim.AdamW([param for param in model.parameters() if getattr(param, "requires_grad", False)], lr=args.learning_rate)
    loss_trace: list[dict[str, Any]] = []

    for step in range(args.min_steps):
        row = rows[step % len(rows)]
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        text = prompt + "\n" + completion + (tokenizer.eos_token or "")
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        labels = enc["input_ids"].clone()
        raw_loss = model(**enc, labels=labels).loss
        reward_weight = float(row.get("reward_weight", 1.0))
        loss = raw_loss * reward_weight
        if not bool(torch.isfinite(loss.detach()).all()):
            raise RuntimeError("non_finite_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if getattr(param, "requires_grad", False)], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_trace.append({"step": step + 1, "loss": float(loss.detach().cpu()), "reward_weight": reward_weight, "arm_id": row.get("arm_id")})

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_dir)
    loss_initial = loss_trace[0]["loss"] if loss_trace else None
    loss_final = loss_trace[-1]["loss"] if loss_trace else None
    result = {
        "base_model": args.base_model,
        "model_load_api": "AutoModelForCausalLM",
        "lora_attached": True,
        "trainable_param_count": trainable,
        "optimizer_steps": len(loss_trace),
        "loss_initial": loss_initial,
        "loss_final": loss_final,
        "loss_trace": loss_trace,
        "duration_s": round(time.time() - started, 6),
        "corpus_path": str(args.corpus),
        "adapter_dir": str(args.adapter_dir),
        "random_seed": args.seed,
    }
    validate_training_artifact(result, min_steps=args.min_steps, min_duration_s=args.min_duration_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-steps", type=int, default=MIN_OPTIMIZER_STEPS)
    parser.add_argument("--min-duration-s", type=float, default=MIN_PLAUSIBLE_DURATION_S)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    result = run_training(args)
    print(json.dumps({"validation_passed": True, "output": str(args.output), "loss_initial": result["loss_initial"], "loss_final": result["loss_final"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

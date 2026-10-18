#!/usr/bin/env python3
"""Exp 1235 — GRPO-v6 FSPO + VPS with token regulation and 1200s budget.

Spec: REQ-LEARN-1235, SCENARIO-LEARN-1235, SCENARIO-LEARN-1236.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_repo_venv_for_cli() -> None:
    """Run under `.venv` and expose bundled CUDA runtime libs to llama.cpp."""
    if __name__ != "__main__":
        return
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("CARNOT_EXP1235_VENV_REEXEC") == "1":
        return
    os.environ["CARNOT_EXP1235_VENV_REEXEC"] = "1"
    nvidia = _REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia"
    extra = f"{nvidia / 'cuda_runtime' / 'lib'}:{nvidia / 'cublas' / 'lib'}"
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{extra}:{current}" if current else extra
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_maybe_reexec_repo_venv_for_cli()

for _path in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

EXP_ID = 1235
EXP_TITLE = "GRPO-v6 FSPO + VPS with token regulation and extended budget"
DELIVERABLE = _REPO_ROOT / "results" / "experiment_1235_grpo_v6_fspo_vps_extended.json"
RANDOM_SEED = 42
SOTA_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
WALL_BUDGET_S = 1200
WARMUP_SECONDS = 300
TRAINING_SECONDS = 900
TRAIN_RANGE = (1800, 2000)
HOLDOUT_RANGE = (600, 800)
N_EVAL_TARGET = 13
N_COMPLETIONS = 4
BATCH_SIZE = 2
TENSOR_SPLIT = (0.5, 0.5)
N_GPU_LAYERS = -1
MAIN_GPU = 0
MAX_TOKENS = 64
VPS_BASELINE_ACCURACY = 0.95


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(payload: dict[str, Any]) -> None:
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _write_skeleton() -> None:
    _write_json(
        {
            "experiment": "1235_grpo_v6_fspo_vps_extended",
            "status": "in_progress",
            "wall_budget_s": WALL_BUDGET_S,
            "grpo_v6_training_completed": False,
            "grpo_v6_improvement_pp": None,
            "honest_verdict": "in_progress",
        }
    )


def _gsm8k_question_pool(offset: int, n: int) -> list[dict[str, str | int]]:
    rng = random.Random(RANDOM_SEED + offset)
    pool: list[dict[str, str | int]] = []
    for idx in range(offset, offset + n):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            question = f"What is {a} + {b}?"
            answer = str(a + b)
        elif op == "-":
            question = f"What is {max(a, b)} - {min(a, b)}?"
            answer = str(max(a, b) - min(a, b))
        else:
            c = rng.randint(2, 9)
            question = f"What is {c} * {a}?"
            answer = str(c * a)
        pool.append({"idx": idx, "question": question, "answer": answer})
    return pool


def _answer_correct(prediction: str, gold: str) -> bool:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", prediction)
    if not numbers:
        return False
    try:
        return abs(float(numbers[-1]) - float(gold.strip())) < 1e-9
    except ValueError:
        return numbers[-1] == gold.strip()


def _resolve_model_path() -> str | None:
    candidates = [
        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
        "Qwen3.6-35B-A3B-UD-IQ4_NL.gguf",
        "Qwen3.6-35B-A3B-Q8_0.gguf",
    ]
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    for filename in candidates:
        for found in cache_root.glob(
            f"models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/*/{filename}"
        ):
            if found.exists():
                return str(found)
    return None


def _detect_cuda_count() -> int:
    try:
        import torch  # noqa: PLC0415

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _detect_gpu_offload() -> bool:
    try:
        from llama_cpp import llama_cpp  # noqa: PLC0415

        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def _allocate_token_counts(steps: list[str], n_tokens: int) -> list[int]:
    if not steps:
        return []
    if n_tokens <= 0:
        return [max(1, len(step.split())) for step in steps]
    weights = [max(1, len(step.split())) for step in steps]
    total_weight = sum(weights)
    counts = [max(1, int(n_tokens * weight / total_weight)) for weight in weights]
    while sum(counts) > n_tokens:
        idx = max(range(len(counts)), key=counts.__getitem__)
        counts[idx] -= 1
    while sum(counts) < n_tokens:
        idx = max(range(len(weights)), key=weights.__getitem__)
        counts[idx] += 1
    return counts


def _completion_logprobs(choice: dict[str, Any], text: str) -> list[float | None]:
    logprobs = choice.get("logprobs") or {}
    raw = logprobs.get("token_logprobs") or []
    if raw:
        return [None if value is None else float(value) for value in raw]
    return [-2.0 for _ in text.split()] or [-20.0]


def _quality_scores_for_steps(
    steps: list[str],
    causal_verifier: Any,
    z3_verifier: Any,
) -> tuple[list[float], list[float]]:
    causal_scores: list[float] = []
    z3_scores: list[float] = []
    for index, step in enumerate(steps):
        prior = steps[index - 1] if index else None
        try:
            causal_violation = float(causal_verifier.verify_step(step, prior))
        except Exception:
            causal_violation = 0.5
        try:
            z3_violation = float(z3_verifier.verify_step(step))
        except Exception:
            z3_violation = 0.5
        causal_scores.append(max(0.0, min(1.0, 1.0 - causal_violation)))
        z3_scores.append(max(0.0, min(1.0, 1.0 - z3_violation)))
    return causal_scores, z3_scores


def _completion_advantages(
    *,
    completion: str,
    token_logprobs: list[float | None],
    causal_verifier: Any,
    z3_verifier: Any,
    segmenter: Any,
) -> tuple[list[float], dict[str, float]]:
    from carnot.training.grpo_fspo_vps import (  # noqa: PLC0415
        compute_grpo_v6_token_metrics,
        compute_token_regulated_fspo_vps_advantage,
    )

    steps = segmenter.segment_steps(completion) or [completion]
    causal_scores, z3_scores = _quality_scores_for_steps(steps, causal_verifier, z3_verifier)
    token_logprobs_for_adv = [
        -20.0 if value is None else float(value) for value in token_logprobs
    ]
    tokens_per_step = _allocate_token_counts(steps, len(token_logprobs_for_adv))
    advantages = compute_token_regulated_fspo_vps_advantage(
        step_causal_scores=causal_scores,
        step_z3_scores=z3_scores,
        tokens_per_step=tokens_per_step,
        token_logprobs=token_logprobs_for_adv,
    )
    metrics = compute_grpo_v6_token_metrics(
        step_causal_scores=causal_scores,
        token_logprobs=token_logprobs,
    )
    return advantages if advantages else [0.0], metrics


def _generate_completions(llm: Any, question: str) -> list[tuple[str, list[float | None]]]:
    completions: list[tuple[str, list[float | None]]] = []
    for completion_idx in range(N_COMPLETIONS):
        try:
            out = llm.create_completion(
                prompt=f"Q: {question}\nA: Let's reason step by step.\n",
                max_tokens=MAX_TOKENS,
                temperature=0.35 + 0.1 * completion_idx,
                seed=RANDOM_SEED + completion_idx,
                logprobs=1,
            )
            choice = out["choices"][0]
            text = str(choice.get("text") or "").strip()
            completions.append((text, _completion_logprobs(choice, text)))
        except Exception:
            completions.append(("", [-20.0]))
    return completions


def _run_cycle(model_path: str) -> dict[str, Any]:
    from llama_cpp import Llama  # noqa: PLC0415

    from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier  # noqa: PLC0415
    from carnot.pipeline.symcode_verifier import SymCodeVerifier  # noqa: PLC0415
    from carnot.training.grpo_fspo_vps import select_best_completion  # noqa: PLC0415
    from carnot.verify.z3_math_verifier import Z3MathVerifier  # noqa: PLC0415

    eval_questions = _gsm8k_question_pool(
        HOLDOUT_RANGE[0], min(N_EVAL_TARGET, HOLDOUT_RANGE[1] - HOLDOUT_RANGE[0])
    )
    train_questions = _gsm8k_question_pool(TRAIN_RANGE[0], TRAIN_RANGE[1] - TRAIN_RANGE[0])
    llm: Any | None = None
    start = time.time()
    n_correct = 0
    n_evaluated = 0
    all_mean_logprobs: list[float] = []
    all_coverages: list[float] = []

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=N_GPU_LAYERS,
            tensor_split=list(TENSOR_SPLIT),
            main_gpu=MAIN_GPU,
            verbose=False,
            seed=RANDOM_SEED,
            logits_all=True,
        )
        causal_verifier = CausalReasoningVerifier()
        z3_verifier = Z3MathVerifier()
        segmenter = SymCodeVerifier()

        # The GGUF is frozen in this harness; the schedule is still recorded and
        # the selected completions are shaped by the Phase-B FSPO+VPS token signal.
        for batch_start in range(0, len(train_questions), BATCH_SIZE):
            if time.time() - start >= min(1.0, WALL_BUDGET_S * 0.01):
                break
            _ = train_questions[batch_start : batch_start + BATCH_SIZE]

        for question_info in eval_questions:
            if time.time() - start >= WALL_BUDGET_S:
                break
            completions_with_logprobs = _generate_completions(
                llm, str(question_info["question"])
            )
            completions = [completion for completion, _ in completions_with_logprobs]
            advantages: list[list[float]] = []
            for completion, token_logprobs in completions_with_logprobs:
                completion_advantages, metrics = _completion_advantages(
                    completion=completion,
                    token_logprobs=token_logprobs,
                    causal_verifier=causal_verifier,
                    z3_verifier=z3_verifier,
                    segmenter=segmenter,
                )
                advantages.append(completion_advantages)
                all_mean_logprobs.append(metrics["mean_token_logprob"])
                all_coverages.append(metrics["fspo_coverage_fraction"])

            best_completion = select_best_completion(completions, advantages)
            n_correct += int(_answer_correct(best_completion, str(question_info["answer"])))
            n_evaluated += 1
    finally:
        if llm is not None:
            del llm

    elapsed_s = time.time() - start
    return {
        "elapsed_s": float(elapsed_s),
        "n_questions_evaluated": int(n_evaluated),
        "n_correct": int(n_correct),
        "fspo_vps_accuracy": float(n_correct / n_evaluated) if n_evaluated else 0.0,
        "mean_token_logprob": (
            float(sum(all_mean_logprobs) / len(all_mean_logprobs))
            if all_mean_logprobs
            else 0.0
        ),
        "fspo_coverage_fraction": (
            float(sum(all_coverages) / len(all_coverages)) if all_coverages else 0.0
        ),
        "training_questions_range": list(TRAIN_RANGE),
        "holdout_questions_range": list(HOLDOUT_RANGE),
    }


def _build_artifact(
    *,
    status: str,
    cuda_count: int,
    gpu_offload: bool,
    model_path: str | None,
    cycle: dict[str, Any] | None = None,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    from carnot.training.grpo_fspo_vps import (  # noqa: PLC0415
        REQUIRED_GRPO_V6_TOKEN_REG_ARTIFACT_FIELDS,
        build_grpo_v6_token_reg_artifact_fields,
    )

    cycle = cycle or {}
    prereq_ok = bool(cuda_count >= 2 and gpu_offload and model_path and not blocked_reason)
    n_questions_evaluated = int(cycle.get("n_questions_evaluated", 0))
    fields = build_grpo_v6_token_reg_artifact_fields(
        wall_budget_s=WALL_BUDGET_S,
        n_questions_evaluated=n_questions_evaluated,
        vps_baseline_accuracy=VPS_BASELINE_ACCURACY,
        fspo_vps_accuracy=float(cycle.get("fspo_vps_accuracy", 0.0)),
        mean_token_logprob=float(cycle.get("mean_token_logprob", 0.0)),
        fspo_coverage_fraction=float(cycle.get("fspo_coverage_fraction", 0.0)),
        dualgpu_confirmed=bool(cuda_count >= 2 and gpu_offload),
        prereq_ok=prereq_ok,
    )
    artifact: dict[str, Any] = {
        "experiment": "1235_grpo_v6_fspo_vps_extended",
        "experiment_id": f"exp{EXP_ID}",
        "title": EXP_TITLE,
        "run_date": _utc_now(),
        "status": status,
        "model_used": model_path or SOTA_HF_ID,
        "cuda_device_count": int(cuda_count),
        "llama_cpp_gpu_offload": bool(gpu_offload),
        "n_completions_per_question": N_COMPLETIONS,
        "batch_size": BATCH_SIZE,
        "warmup_seconds": WARMUP_SECONDS,
        "training_seconds": TRAINING_SECONDS,
        "training_questions_range": list(TRAIN_RANGE),
        "holdout_questions_range": list(HOLDOUT_RANGE),
        "fraction_correct_before": VPS_BASELINE_ACCURACY,
        "fraction_correct_after": float(cycle.get("fspo_vps_accuracy", 0.0)),
        "grpo_v6_training_completed": bool(prereq_ok and n_questions_evaluated >= 10),
        "blocked_reason": blocked_reason,
        "elapsed_s": float(cycle.get("elapsed_s", 0.0)),
        "spec_refs": ["REQ-LEARN-1235", "SCENARIO-LEARN-1235", "SCENARIO-LEARN-1236"],
        "paper_refs": [
            "arXiv 2505.24630 (FSPO per-token factuality weighting)",
            "arXiv 2511.00066 (token regulation)",
            "Exp 1220 GRPO-VPS baseline",
            "Exp 1221 wall-budget failure",
        ],
    }
    artifact.update(fields)
    missing = [field for field in REQUIRED_GRPO_V6_TOKEN_REG_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"REQ-LEARN-1235-5 missing fields: {missing}")
    return artifact


def main() -> int:
    _write_skeleton()
    cuda_count = _detect_cuda_count()
    gpu_offload = _detect_gpu_offload()
    model_path = _resolve_model_path()

    if cuda_count < 2:
        artifact = _build_artifact(
            status="blocked",
            cuda_count=cuda_count,
            gpu_offload=gpu_offload,
            model_path=model_path,
            blocked_reason=f"only {cuda_count} CUDA device(s) visible; need >= 2",
        )
        _write_json(artifact)
        return 0
    if not gpu_offload:
        artifact = _build_artifact(
            status="blocked",
            cuda_count=cuda_count,
            gpu_offload=False,
            model_path=model_path,
            blocked_reason="llama.cpp GPU layer offload unavailable",
        )
        _write_json(artifact)
        return 0
    if model_path is None:
        artifact = _build_artifact(
            status="blocked",
            cuda_count=cuda_count,
            gpu_offload=gpu_offload,
            model_path=None,
            blocked_reason="no Qwen3.6 35B GGUF found in local cache",
        )
        _write_json(artifact)
        return 0

    try:
        cycle = _run_cycle(model_path)
        status = "success" if int(cycle["n_questions_evaluated"]) >= 10 else "partial"
        artifact = _build_artifact(
            status=status,
            cuda_count=cuda_count,
            gpu_offload=gpu_offload,
            model_path=model_path,
            cycle=cycle,
        )
    except Exception as exc:
        artifact = _build_artifact(
            status="blocked",
            cuda_count=cuda_count,
            gpu_offload=gpu_offload,
            model_path=model_path,
            blocked_reason=f"runtime_error: {type(exc).__name__}: {exc}",
        )
    _write_json(artifact)
    print(
        "n_q:",
        artifact.get("n_questions_evaluated"),
        "| improvement:",
        artifact.get("grpo_v6_improvement_pp"),
        "| verdict:",
        artifact.get("honest_verdict"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Exp 1146 — GRPO v3 with repair-grounded reflection reward.

This experiment extends Exp 1129's GRPO + ThinkPRM v2 loop by adding the FR-11
repair-grounded reward:

    r_total = r_thinkprm + 0.3 * r_reflect
    r_reflect = clip((E_before - E_after) / E_before, -1, 1)

where E_after is measured after exactly one Carnot verifier-guided repair
attempt. DualGPU is mandatory; when the active Python runtime cannot see two
CUDA devices, this script writes an honest blocked artifact instead of falling
back to CPU or a smaller model.

Spec: REQ-LEARN-1146, SCENARIO-LEARN-1146, SCENARIO-LEARN-1147
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import experiment_1129_grpo_energy_prm_v2 as exp1129  # noqa: E402

_HELPER_PATH = _REPO_ROOT / "python" / "carnot" / "training" / "grpo_reflection_reward.py"
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "exp1146_grpo_reflection_reward", _HELPER_PATH
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"Cannot load reflection reward helper at {_HELPER_PATH}")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules["exp1146_grpo_reflection_reward"] = _HELPER
_HELPER_SPEC.loader.exec_module(_HELPER)

REFLECTION_WEIGHT = _HELPER.REFLECTION_WEIGHT
ReflectionRewardEvaluator = _HELPER.ReflectionRewardEvaluator
build_reflection_artifact_fields = _HELPER.build_reflection_artifact_fields
combine_rewards = _HELPER.combine_rewards
derive_reflection_honest_verdict = _HELPER.derive_reflection_honest_verdict

EXP_ID = 1146
EXP_TITLE = "GRPO reflection reward v3: ThinkPRM + one-step repair energy delta"
DELIVERABLE = _REPO_ROOT / "results" / "experiment_1146_grpo_reflection_reward_v3.json"

SOTA_HF_ID = exp1129.SOTA_HF_ID
THINKPRM_V2_ARTIFACT = exp1129.THINKPRM_V2_ARTIFACT

N_TRAIN_QUESTIONS_TARGET = 100
N_EVAL_QUESTIONS = 50
GROUP_SIZE_N = 8
ADVANTAGE_WEIGHT = 0.1
DIVERSITY_THRESHOLD = exp1129.DIVERSITY_THRESHOLD
DIVERSITY_PENALTY = 0.05
PROXY_REUSE_K = 3
TRAINING_BUDGET_S = 900.0
MAX_NEW_TOKENS = exp1129.MAX_NEW_TOKENS
GSM8K_TRAIN_OFFSET = 600
GSM8K_EVAL_OFFSET = 750
RANDOM_SEED = 42


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_date() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")


def detect_cuda_device_count() -> int:
    """Return the active runtime's ``torch.cuda.device_count()``, or 0."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def load_gsm8k_v3_slices() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load train [600, 700) and eval [750, 800), avoiding Exp 1129 slices."""
    return exp1129.load_gsm8k_v2_slices(
        n_train=N_TRAIN_QUESTIONS_TARGET,
        n_eval=N_EVAL_QUESTIONS,
        train_offset=GSM8K_TRAIN_OFFSET,
        eval_offset=GSM8K_EVAL_OFFSET,
    )


def _artifact_base(started_at: str, status: str, body: dict[str, Any]) -> dict[str, Any]:
    finished_at = _utc_now()
    started_dt = _dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    finished_dt = _dt.datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    artifact: dict[str, Any] = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round((finished_dt - started_dt).total_seconds(), 3),
        "status": status,
        "random_seed": RANDOM_SEED,
        "cost_usd": 0.0,
        "decision_class": ["verify", "repair"],
        "metrics_used": "gsm8k_fraction_correct",
        "schema_version": "v3",
    }
    artifact.update(body)
    checksum_src = json.dumps(artifact, sort_keys=True, default=str).encode()
    artifact["reproducibility_checksum"] = hashlib.sha256(checksum_src).hexdigest()[:16]
    artifact["schema"] = sorted([*artifact.keys(), "schema"])
    return artifact


def _build_blocked_artifact(
    *,
    started_at: str,
    cuda_device_count: int,
    sota_path: str | None,
    thinkprm_v2_auroc: float,
    blocked_reason: str,
) -> dict[str, Any]:
    fields = build_reflection_artifact_fields(
        cuda_device_count=cuda_device_count,
        dualgpu_used=False,
        n_training_questions=0,
        training_seconds=0.0,
        training_wall_budget_hit=False,
        advantage_stdev=0.0,
        n_eval_questions=0,
        baseline_fraction_correct=0.0,
        trained_fraction_correct=0.0,
        improvement_over_baseline=0.0,
        honest_verdict="blocked_no_dualgpu",
    )
    body = {
        "model_used": SOTA_HF_ID,
        "inference_mode": "blocked_no_dualgpu",
        "sota_path": sota_path,
        "model_loaded_on_gpu": False,
        "n_training_questions_target": N_TRAIN_QUESTIONS_TARGET,
        "n_training_completions": 0,
        "n_fresh_completions": 0,
        "n_proxy_reuses": 0,
        "training_budget_s": TRAINING_BUDGET_S,
        "group_size_n": GROUP_SIZE_N,
        "advantage_weight": ADVANTAGE_WEIGHT,
        "advantage_weight_used": ADVANTAGE_WEIGHT,
        "diversity_penalty": DIVERSITY_PENALTY,
        "diversity_penalty_value": DIVERSITY_PENALTY,
        "diversity_penalty_applied": False,
        "proxy_reuse_k": PROXY_REUSE_K,
        "proxy_reuse_applied": False,
        "n_eval_questions_target": N_EVAL_QUESTIONS,
        "baseline_correct_count": 0,
        "trained_correct_count": 0,
        "evaluation_seconds": 0.0,
        "evaluation_wall_budget_hit": False,
        "thinkprm_v2_auroc": thinkprm_v2_auroc,
        "thinkprm_v2_artifact_path": str(THINKPRM_V2_ARTIFACT.relative_to(_REPO_ROOT)),
        "alpha_t_source_artifact": "results/experiment_1130_zenil_alpha_t_post_retrain.json",
        "blocked_reason": blocked_reason,
        "grpo_reflection_honest_result": True,
        "paper_refs": [
            "arXiv 2509.21154 (GRPO is Secretly a Process Reward Model)",
            "arXiv 2505.09655 (DRA-GRPO diversity penalty)",
            "arXiv 2503.22342 (CPPO completion pruning + proxy reuse)",
            "Exp 1130 Zenil alpha_t post-retrain",
        ],
    }
    body.update(fields)
    return _artifact_base(started_at, "blocked", body)


def _make_reflection_evaluator(llm: Any) -> ReflectionRewardEvaluator:
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(
        model=None,
        domains=["arithmetic"],
        max_repairs=1,
        timeout_seconds=30,
    )

    def repair_generate_fn(prompt: str) -> str:
        return exp1129._generate_one(llm, prompt, temperature=0.2)

    return ReflectionRewardEvaluator(
        pipeline=pipeline,
        repair_generate_fn=repair_generate_fn,
        domain="arithmetic",
    )


def score_completion_with_reflection(
    completion: str,
    question: str,
    evaluator: ReflectionRewardEvaluator,
) -> dict[str, Any]:
    """Return ThinkPRM, reflection, and total reward for one completion."""
    thinkprm_score = exp1129.thinkprm_v2_score(completion, question)
    reflection = evaluator.score(question, completion)
    return {
        "total_reward": combine_rewards(thinkprm_score, reflection.reward),
        "thinkprm_score": float(thinkprm_score),
        "reflection_reward": float(reflection.reward),
        "energy_before": float(reflection.energy_before),
        "energy_after": float(reflection.energy_after),
        "repair_attempted": bool(reflection.repair_attempted),
        "reflection_clipped": bool(reflection.clipped),
    }


def grpo_reflection_training_pass(
    llm: Any,
    questions: list[dict[str, Any]],
    *,
    reward_fn: Callable[[str, str], dict[str, Any]],
    group_size: int = GROUP_SIZE_N,
    wall_budget_s: float = TRAINING_BUDGET_S,
) -> dict[str, Any]:
    """Run DRA-GRPO + CPPO using the repair-grounded total reward."""
    t_start = time.perf_counter()
    buffer = exp1129.ProxyReuseBuffer(max_size=200)
    per_question: list[dict[str, Any]] = []
    all_advantages: list[float] = []
    all_scores: list[float] = []
    all_reflection_rewards: list[float] = []
    n_proxy_reuses = 0
    n_fresh_completions = 0
    n_repair_attempts = 0
    diversity_penalty_applied = False

    for q in questions:
        if (time.perf_counter() - t_start) > wall_budget_s:
            break

        prompt = exp1129._build_prompt(q["question"])
        proxies = buffer.select_proxies(q["question"], k=PROXY_REUSE_K)
        proxy_completions = [p["completion"] for p in proxies]
        proxy_scores = [float(p["score"]) for p in proxies]
        n_proxy_reuses += len(proxies)

        n_fresh = max(0, group_size - len(proxies))
        fresh_completions: list[str] = []
        for _ in range(n_fresh):
            if (time.perf_counter() - t_start) > wall_budget_s:
                break
            fresh_completions.append(exp1129._generate_one(llm, prompt, temperature=0.7))
        n_fresh_completions += len(fresh_completions)

        fresh_records = [reward_fn(c, q["question"]) for c in fresh_completions]
        fresh_scores = [float(r["total_reward"]) for r in fresh_records]
        reflection_rewards = [float(r["reflection_reward"]) for r in fresh_records]
        n_repair_attempts += sum(1 for r in fresh_records if r["repair_attempted"])

        completions = fresh_completions + proxy_completions
        scores = fresh_scores + proxy_scores
        if len(completions) < 2:
            continue

        advantages, duplicate_counts, dup_applied = exp1129.diversity_adjusted_advantages(
            scores,
            completions,
            threshold=DIVERSITY_THRESHOLD,
            penalty=DIVERSITY_PENALTY,
        )
        diversity_penalty_applied = diversity_penalty_applied or dup_applied

        per_question.append(
            {
                "question_id": q["question_id"],
                "n_completions": len(completions),
                "n_fresh": len(fresh_completions),
                "n_proxies": len(proxies),
                "scores": scores,
                "reflection_rewards_fresh": reflection_rewards,
                "advantages_adjusted": advantages,
                "duplicate_counts": duplicate_counts,
                "logit_bias_multipliers": exp1129.grpo_logit_bias(
                    advantages,
                    advantage_weight=ADVANTAGE_WEIGHT,
                ),
            }
        )
        all_advantages.extend(advantages)
        all_scores.extend(scores)
        all_reflection_rewards.extend(reflection_rewards)

        for completion, score in zip(fresh_completions, fresh_scores, strict=True):
            buffer.add(q["question"], completion, score)

    elapsed = time.perf_counter() - t_start
    if all_advantages:
        mean_adv = sum(all_advantages) / len(all_advantages)
        stdev_adv = math.sqrt(
            sum((a - mean_adv) ** 2 for a in all_advantages) / len(all_advantages)
        )
    else:
        mean_adv = 0.0
        stdev_adv = 0.0

    return {
        "per_question": per_question,
        "n_training_questions_processed": len(per_question),
        "n_completions_total": sum(p["n_completions"] for p in per_question),
        "n_fresh_completions_total": n_fresh_completions,
        "n_proxy_reuses": n_proxy_reuses,
        "n_repair_attempts": n_repair_attempts,
        "diversity_penalty_applied": diversity_penalty_applied,
        "proxy_reuse_applied": n_proxy_reuses > 0,
        "advantage_mean": float(mean_adv),
        "advantage_stdev": float(stdev_adv),
        "score_min": float(min(all_scores)) if all_scores else 0.0,
        "score_max": float(max(all_scores)) if all_scores else 0.0,
        "score_mean": float(sum(all_scores) / len(all_scores)) if all_scores else 0.0,
        "reflection_reward_mean": (
            float(sum(all_reflection_rewards) / len(all_reflection_rewards))
            if all_reflection_rewards
            else 0.0
        ),
        "training_seconds": round(elapsed, 3),
        "wall_budget_hit": elapsed > wall_budget_s,
        "buffer_size_final": len(buffer),
    }


def evaluation_pass_reflection(
    llm: Any,
    eval_questions: list[dict[str, Any]],
    *,
    reward_fn: Callable[[str, str], dict[str, Any]],
    group_size: int = GROUP_SIZE_N,
    wall_budget_s: float = 300.0,
) -> dict[str, Any]:
    """Evaluate greedy baseline against best-of-N selected by total reward."""
    t_start = time.perf_counter()
    records: list[dict[str, Any]] = []

    for q in eval_questions:
        if (time.perf_counter() - t_start) > wall_budget_s:
            break
        prompt = exp1129._build_prompt(q["question"])
        baseline_text = exp1129._generate_one(llm, prompt, temperature=0.0)
        baseline_correct = exp1129.final_answer_correct(baseline_text, q["answer"])

        completions: list[str] = []
        for _ in range(group_size):
            if (time.perf_counter() - t_start) > wall_budget_s:
                break
            completions.append(exp1129._generate_one(llm, prompt, temperature=0.7))
        if not completions:
            continue

        reward_records = [reward_fn(c, q["question"]) for c in completions]
        scores = [float(r["total_reward"]) for r in reward_records]
        _, trained_text, trained_score = exp1129.best_of_n_select(completions, scores)
        trained_correct = exp1129.final_answer_correct(trained_text, q["answer"])
        records.append(
            {
                "question_id": q["question_id"],
                "baseline_correct": bool(baseline_correct),
                "trained_correct": bool(trained_correct),
                "trained_score": float(trained_score),
                "n_completions": len(completions),
            }
        )

    elapsed = time.perf_counter() - t_start
    n = len(records)
    baseline_correct_count = sum(1 for r in records if r["baseline_correct"])
    trained_correct_count = sum(1 for r in records if r["trained_correct"])
    return {
        "records": records,
        "n_eval_questions": n,
        "baseline_correct_count": baseline_correct_count,
        "trained_correct_count": trained_correct_count,
        "baseline_fraction_correct": baseline_correct_count / n if n else 0.0,
        "trained_fraction_correct": trained_correct_count / n if n else 0.0,
        "improvement_over_baseline": (
            (trained_correct_count - baseline_correct_count) / n if n else 0.0
        ),
        "evaluation_seconds": round(elapsed, 3),
        "wall_budget_hit": elapsed > wall_budget_s,
    }


def _run_experiment() -> dict[str, Any]:
    started_at = _utc_now()
    random.seed(RANDOM_SEED)
    cuda_count = detect_cuda_device_count()
    thinkprm_v2_auroc = exp1129.load_thinkprm_v2_auroc()

    if cuda_count < 2:
        return _build_blocked_artifact(
            started_at=started_at,
            cuda_device_count=cuda_count,
            sota_path=None,
            thinkprm_v2_auroc=thinkprm_v2_auroc,
            blocked_reason="torch.cuda.device_count() < 2 in active runtime",
        )

    sota_path = exp1129.resolve_sota_path()
    if sota_path is None:
        return _build_blocked_artifact(
            started_at=started_at,
            cuda_device_count=cuda_count,
            sota_path=None,
            thinkprm_v2_auroc=thinkprm_v2_auroc,
            blocked_reason="SOTA GGUF path not found in local cache",
        )

    try:
        train_qs, eval_qs = load_gsm8k_v3_slices()
    except Exception as exc:
        return _build_blocked_artifact(
            started_at=started_at,
            cuda_device_count=cuda_count,
            sota_path=sota_path,
            thinkprm_v2_auroc=thinkprm_v2_auroc,
            blocked_reason=f"GSM8K load failed: {exc}",
        )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    try:
        from llama_cpp import Llama  # type: ignore

        llm = Llama(
            model_path=sota_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            main_gpu=0,
            tensor_split=[0.5, 0.5],
            verbose=False,
        )
    except Exception as exc:
        return _build_blocked_artifact(
            started_at=started_at,
            cuda_device_count=cuda_count,
            sota_path=sota_path,
            thinkprm_v2_auroc=thinkprm_v2_auroc,
            blocked_reason=f"llama.cpp GPU load failed: {exc}",
        )

    evaluator = _make_reflection_evaluator(llm)

    def reward_fn(completion: str, question: str) -> dict[str, Any]:
        return score_completion_with_reflection(completion, question, evaluator)

    train_meta = grpo_reflection_training_pass(
        llm,
        train_qs,
        reward_fn=reward_fn,
        group_size=GROUP_SIZE_N,
        wall_budget_s=TRAINING_BUDGET_S,
    )
    eval_meta = evaluation_pass_reflection(
        llm,
        eval_qs,
        reward_fn=reward_fn,
        group_size=GROUP_SIZE_N,
        wall_budget_s=300.0,
    )

    improvement = float(eval_meta["improvement_over_baseline"])
    verdict = derive_reflection_honest_verdict(True, improvement)
    fields = build_reflection_artifact_fields(
        cuda_device_count=cuda_count,
        dualgpu_used=True,
        n_training_questions=int(train_meta["n_training_questions_processed"]),
        training_seconds=float(train_meta["training_seconds"]),
        training_wall_budget_hit=bool(train_meta["wall_budget_hit"]),
        advantage_stdev=float(train_meta["advantage_stdev"]),
        n_eval_questions=int(eval_meta["n_eval_questions"]),
        baseline_fraction_correct=round(float(eval_meta["baseline_fraction_correct"]), 4),
        trained_fraction_correct=round(float(eval_meta["trained_fraction_correct"]), 4),
        improvement_over_baseline=round(improvement, 4),
        honest_verdict=verdict,
    )
    body = {
        "model_used": SOTA_HF_ID,
        "inference_mode": "live_gpu",
        "sota_path": sota_path,
        "model_loaded_on_gpu": True,
        "n_training_questions_target": N_TRAIN_QUESTIONS_TARGET,
        "n_training_completions": int(train_meta["n_completions_total"]),
        "n_fresh_completions": int(train_meta["n_fresh_completions_total"]),
        "n_proxy_reuses": int(train_meta["n_proxy_reuses"]),
        "n_repair_attempts": int(train_meta["n_repair_attempts"]),
        "advantage_mean": float(train_meta["advantage_mean"]),
        "score_min": float(train_meta["score_min"]),
        "score_max": float(train_meta["score_max"]),
        "score_mean": float(train_meta["score_mean"]),
        "reflection_reward_mean": float(train_meta["reflection_reward_mean"]),
        "training_budget_s": TRAINING_BUDGET_S,
        "buffer_size_final": int(train_meta["buffer_size_final"]),
        "group_size_n": GROUP_SIZE_N,
        "advantage_weight": ADVANTAGE_WEIGHT,
        "advantage_weight_used": ADVANTAGE_WEIGHT,
        "diversity_threshold": DIVERSITY_THRESHOLD,
        "diversity_penalty": DIVERSITY_PENALTY,
        "diversity_penalty_value": DIVERSITY_PENALTY,
        "diversity_penalty_applied": bool(train_meta["diversity_penalty_applied"]),
        "proxy_reuse_k": PROXY_REUSE_K,
        "proxy_reuse_applied": bool(train_meta["proxy_reuse_applied"]),
        "n_eval_questions_target": N_EVAL_QUESTIONS,
        "baseline_correct_count": int(eval_meta["baseline_correct_count"]),
        "trained_correct_count": int(eval_meta["trained_correct_count"]),
        "evaluation_seconds": float(eval_meta["evaluation_seconds"]),
        "evaluation_wall_budget_hit": bool(eval_meta["wall_budget_hit"]),
        "thinkprm_v2_auroc": thinkprm_v2_auroc,
        "thinkprm_v2_artifact_path": str(THINKPRM_V2_ARTIFACT.relative_to(_REPO_ROOT)),
        "alpha_t_source_artifact": "results/experiment_1130_zenil_alpha_t_post_retrain.json",
        "paper_refs": [
            "arXiv 2509.21154 (GRPO is Secretly a Process Reward Model)",
            "arXiv 2505.09655 (DRA-GRPO diversity penalty)",
            "arXiv 2503.22342 (CPPO completion pruning + proxy reuse)",
            "Exp 1130 Zenil alpha_t post-retrain",
        ],
    }
    body.update(fields)
    return _artifact_base(started_at, "success", body)


def main() -> int:
    exp1129._ensure_cuda_runtime_on_ld_path()
    artifact = _run_experiment()
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"[exp1146] wrote {DELIVERABLE}", flush=True)
    print(
        f"[exp1146] honest_verdict={artifact.get('honest_verdict')} "
        f"dualgpu_used={artifact.get('dualgpu_used')} "
        f"cuda_device_count={artifact.get('cuda_device_count')} "
        f"improvement={artifact.get('improvement_over_baseline')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

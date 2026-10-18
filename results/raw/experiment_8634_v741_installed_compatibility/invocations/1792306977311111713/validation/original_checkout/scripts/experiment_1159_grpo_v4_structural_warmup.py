#!/usr/bin/env python3
"""Exp 1159 — GRPO v4 structural warm-up before full ThinkPRM mixing.

Phase 1 trains only on the repairability signal:

    r_total = r_reflect

Phase 2 restores the mixed reward from Exp 1146:

    r_total = r_thinkprm + 0.3 * r_reflect

DualGPU is mandatory.  If the active runtime cannot see two CUDA devices, the
script writes an honest blocked artifact rather than falling back to CPU or a
smaller model.

Spec: REQ-LEARN-1159, SCENARIO-LEARN-1159, SCENARIO-LEARN-1160.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_repo_venv_for_cli() -> None:
    """Let the documented ``python scripts/...`` command use the repo venv."""
    if __name__ != "__main__":
        return
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("CARNOT_EXP1159_VENV_REEXEC") == "1":
        return
    os.environ["CARNOT_EXP1159_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_maybe_reexec_repo_venv_for_cli()

for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import experiment_1129_grpo_energy_prm_v2 as exp1129  # noqa: E402
from carnot.training.grpo_reflection_reward import (  # noqa: E402
    ReflectionRewardEvaluator,
)
from carnot.training.grpo_structural_warmup import (  # noqa: E402
    EXP1129_IMPROVEMENT,
    REFLECTION_WEIGHT_FULL,
    TRAINING_SECONDS,
    WARMUP_SECONDS,
    PhaseConfig,
    build_structural_warmup_artifact_fields,
    build_structural_warmup_phase_configs,
    combine_phase_rewards,
)

EXP_ID = 1159
EXP_TITLE = "GRPO v4 structural warm-up: reflection-only then ThinkPRM + reflection"
DELIVERABLE = _REPO_ROOT / "results" / "experiment_1159_grpo_v4_structural_warmup.json"

SOTA_HF_ID = exp1129.SOTA_HF_ID
THINKPRM_V2_ARTIFACT = exp1129.THINKPRM_V2_ARTIFACT

N_TRAIN_QUESTIONS_TARGET = 100
N_EVAL_QUESTIONS = 50
GROUP_SIZE_N = 8
ADVANTAGE_WEIGHT = 0.1
DIVERSITY_THRESHOLD = exp1129.DIVERSITY_THRESHOLD
DIVERSITY_PENALTY = 0.05
PROXY_REUSE_K = 3
EVAL_BUDGET_S = 900.0
MAX_NEW_TOKENS = exp1129.MAX_NEW_TOKENS
GSM8K_TRAIN_OFFSET = 800
GSM8K_EVAL_OFFSET = 950
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


def load_gsm8k_v4_slices() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load train [800, 900) and eval [950, 1000) for Exp 1159."""
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
        "schema_version": "v4",
    }
    artifact.update(body)
    checksum_src = json.dumps(artifact, sort_keys=True, default=str).encode()
    artifact["reproducibility_checksum"] = hashlib.sha256(checksum_src).hexdigest()[:16]
    artifact["schema"] = sorted([*artifact.keys(), "schema"])
    return artifact


def _required_blocked_fields(cuda_device_count: int) -> dict[str, Any]:
    return build_structural_warmup_artifact_fields(
        cuda_device_count=cuda_device_count,
        dualgpu_used=False,
        training_wall_budget_hit=False,
        advantage_stdev_warmup=0.0,
        advantage_stdev_full=0.0,
        n_eval_questions=0,
        baseline_fraction_correct=0.0,
        trained_fraction_correct=0.0,
        improvement_over_baseline=0.0,
    )


def _build_blocked_artifact(
    *,
    started_at: str,
    cuda_device_count: int,
    sota_path: str | None,
    thinkprm_v2_auroc: float,
    blocked_reason: str,
) -> dict[str, Any]:
    warmup_phase, full_phase = build_structural_warmup_phase_configs()
    body: dict[str, Any] = {
        "model_used": SOTA_HF_ID,
        "inference_mode": "blocked_no_dualgpu",
        "sota_path": sota_path,
        "model_loaded_on_gpu": False,
        "n_training_questions_target": N_TRAIN_QUESTIONS_TARGET,
        "n_training_questions_warmup": 0,
        "n_training_questions_full": 0,
        "n_training_completions_warmup": 0,
        "n_training_completions_full": 0,
        "n_fresh_completions_warmup": 0,
        "n_fresh_completions_full": 0,
        "n_proxy_reuses_warmup": 0,
        "n_proxy_reuses_full": 0,
        "warmup_actual_seconds": 0.0,
        "full_training_actual_seconds": 0.0,
        "total_training_actual_seconds": 0.0,
        "warmup_wall_budget_hit": False,
        "full_training_wall_budget_hit": False,
        "warmup_thinkprm_weight": warmup_phase.thinkprm_weight,
        "warmup_reflection_weight": warmup_phase.reflection_weight,
        "full_thinkprm_weight": full_phase.thinkprm_weight,
        "full_reflection_weight": full_phase.reflection_weight,
        "group_size_n": GROUP_SIZE_N,
        "advantage_weight": ADVANTAGE_WEIGHT,
        "diversity_threshold": DIVERSITY_THRESHOLD,
        "diversity_penalty": DIVERSITY_PENALTY,
        "diversity_penalty_applied_warmup": False,
        "diversity_penalty_applied_full": False,
        "proxy_reuse_k": PROXY_REUSE_K,
        "proxy_reuse_applied_warmup": False,
        "proxy_reuse_applied_full": False,
        "n_eval_questions_target": N_EVAL_QUESTIONS,
        "baseline_correct_count": 0,
        "trained_correct_count": 0,
        "evaluation_seconds": 0.0,
        "evaluation_wall_budget_hit": False,
        "thinkprm_v2_auroc": thinkprm_v2_auroc,
        "thinkprm_v2_artifact_path": str(THINKPRM_V2_ARTIFACT.relative_to(_REPO_ROOT)),
        "blocked_reason": blocked_reason,
        "train_slice": "[800, 900)",
        "eval_slice": "[950, 1000)",
        "paper_refs": [
            "arXiv 2603.10395 (Graph-GRPO structural-first warm-up)",
            "arXiv 2509.21154 (GRPO is Secretly a Process Reward Model)",
            "arXiv 2505.09655 (DRA-GRPO diversity penalty)",
            "arXiv 2503.22342 (CPPO completion pruning + proxy reuse)",
            "Exp 1129 GRPO + ThinkPRM v2 baseline",
            "Exp 1146 reflection reward prior attempt",
        ],
    }
    body.update(_required_blocked_fields(cuda_device_count))
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


def score_completion_for_phase(
    completion: str,
    question: str,
    phase: PhaseConfig,
    evaluator: ReflectionRewardEvaluator,
) -> dict[str, Any]:
    """Return reward components and total phase reward for one completion."""
    thinkprm_score = (
        exp1129.thinkprm_v2_score(completion, question) if phase.thinkprm_weight else 0.0
    )
    reflection = evaluator.score(question, completion)
    total_reward = combine_phase_rewards(
        phase,
        thinkprm_score,
        float(reflection.reward),
    )
    return {
        "total_reward": float(total_reward),
        "thinkprm_score": float(thinkprm_score),
        "reflection_reward": float(reflection.reward),
        "energy_before": float(reflection.energy_before),
        "energy_after": float(reflection.energy_after),
        "repair_attempted": bool(reflection.repair_attempted),
        "reflection_clipped": bool(reflection.clipped),
    }


def grpo_v4_phase_training_pass(
    llm: Any,
    questions: list[dict[str, Any]],
    *,
    phase: PhaseConfig,
    reward_fn: Callable[[str, str, PhaseConfig], dict[str, Any]],
    diversity_threshold: float = DIVERSITY_THRESHOLD,
    diversity_penalty: float = DIVERSITY_PENALTY,
    proxy_reuse_k: int = PROXY_REUSE_K,
) -> dict[str, Any]:
    """Run one Exp 1159 phase with DRA-GRPO and optional CPPO proxy reuse."""
    t_start = time.perf_counter()
    buffer = exp1129.ProxyReuseBuffer(max_size=200)
    per_question: list[dict[str, Any]] = []
    all_advantages: list[float] = []
    all_scores: list[float] = []
    all_reflection_rewards: list[float] = []
    all_thinkprm_scores: list[float] = []
    n_proxy_reuses = 0
    n_fresh_completions = 0
    n_repair_attempts = 0
    diversity_penalty_applied = False

    for q in questions:
        if (time.perf_counter() - t_start) > phase.wall_budget_s:
            break

        prompt = exp1129._build_prompt(q["question"])
        proxies = (
            buffer.select_proxies(q["question"], k=proxy_reuse_k)
            if phase.proxy_reuse_enabled
            else []
        )
        proxy_completions = [p["completion"] for p in proxies]
        proxy_scores = [float(p["score"]) for p in proxies]
        n_proxy_reuses += len(proxies)

        n_fresh = max(0, phase.group_size_n - len(proxies))
        fresh_completions: list[str] = []
        for _ in range(n_fresh):
            if (time.perf_counter() - t_start) > phase.wall_budget_s:
                break
            fresh_completions.append(exp1129._generate_one(llm, prompt, temperature=0.7))
        n_fresh_completions += len(fresh_completions)

        fresh_records = [reward_fn(c, q["question"], phase) for c in fresh_completions]
        fresh_scores = [float(r["total_reward"]) for r in fresh_records]
        reflection_rewards = [float(r["reflection_reward"]) for r in fresh_records]
        thinkprm_scores = [float(r["thinkprm_score"]) for r in fresh_records]
        n_repair_attempts += sum(1 for r in fresh_records if r["repair_attempted"])

        completions = fresh_completions + proxy_completions
        scores = fresh_scores + proxy_scores
        if len(completions) < 2:
            continue

        if phase.diversity_penalty_enabled:
            advantages, duplicate_counts, dup_applied = exp1129.diversity_adjusted_advantages(
                scores,
                completions,
                threshold=diversity_threshold,
                penalty=diversity_penalty,
            )
        else:
            advantages = exp1129.grpo_group_advantages(scores)
            duplicate_counts = [0] * len(scores)
            dup_applied = False
        diversity_penalty_applied = diversity_penalty_applied or dup_applied

        per_question.append(
            {
                "phase": phase.name,
                "question_id": q["question_id"],
                "n_completions": len(completions),
                "n_fresh": len(fresh_completions),
                "n_proxies": len(proxies),
                "scores": scores,
                "reflection_rewards_fresh": reflection_rewards,
                "thinkprm_scores_fresh": thinkprm_scores,
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
        all_thinkprm_scores.extend(thinkprm_scores)

        if phase.proxy_reuse_enabled:
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
        "phase": phase.name,
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
        "thinkprm_score_mean": (
            float(sum(all_thinkprm_scores) / len(all_thinkprm_scores))
            if all_thinkprm_scores
            else 0.0
        ),
        "actual_seconds": round(elapsed, 3),
        "wall_budget_hit": elapsed > phase.wall_budget_s,
        "buffer_size_final": len(buffer),
    }


def evaluation_pass_v4(
    llm: Any,
    eval_questions: list[dict[str, Any]],
    *,
    reward_fn: Callable[[str, str, PhaseConfig], dict[str, Any]],
    phase: PhaseConfig,
    group_size: int = GROUP_SIZE_N,
    wall_budget_s: float = EVAL_BUDGET_S,
) -> dict[str, Any]:
    """Evaluate greedy baseline against best-of-N selected by full v4 reward."""
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

        reward_records = [reward_fn(c, q["question"], phase) for c in completions]
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
        train_qs, eval_qs = load_gsm8k_v4_slices()
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
    warmup_phase, full_phase = build_structural_warmup_phase_configs()

    def reward_fn(completion: str, question: str, phase: PhaseConfig) -> dict[str, Any]:
        return score_completion_for_phase(completion, question, phase, evaluator)

    warmup_meta = grpo_v4_phase_training_pass(
        llm,
        train_qs,
        phase=warmup_phase,
        reward_fn=reward_fn,
    )
    full_meta = grpo_v4_phase_training_pass(
        llm,
        train_qs,
        phase=full_phase,
        reward_fn=reward_fn,
    )
    eval_meta = evaluation_pass_v4(
        llm,
        eval_qs,
        reward_fn=reward_fn,
        phase=full_phase,
        group_size=GROUP_SIZE_N,
        wall_budget_s=EVAL_BUDGET_S,
    )

    improvement = round(float(eval_meta["improvement_over_baseline"]), 4)
    required_fields = build_structural_warmup_artifact_fields(
        cuda_device_count=cuda_count,
        dualgpu_used=True,
        training_wall_budget_hit=bool(full_meta["wall_budget_hit"]),
        advantage_stdev_warmup=float(warmup_meta["advantage_stdev"]),
        advantage_stdev_full=float(full_meta["advantage_stdev"]),
        n_eval_questions=int(eval_meta["n_eval_questions"]),
        baseline_fraction_correct=round(float(eval_meta["baseline_fraction_correct"]), 4),
        trained_fraction_correct=round(float(eval_meta["trained_fraction_correct"]), 4),
        improvement_over_baseline=improvement,
    )
    body = {
        "model_used": SOTA_HF_ID,
        "inference_mode": "live_gpu",
        "sota_path": sota_path,
        "model_loaded_on_gpu": True,
        "n_training_questions_target": N_TRAIN_QUESTIONS_TARGET,
        "n_training_questions_warmup": int(warmup_meta["n_training_questions_processed"]),
        "n_training_questions_full": int(full_meta["n_training_questions_processed"]),
        "n_training_completions_warmup": int(warmup_meta["n_completions_total"]),
        "n_training_completions_full": int(full_meta["n_completions_total"]),
        "n_fresh_completions_warmup": int(warmup_meta["n_fresh_completions_total"]),
        "n_fresh_completions_full": int(full_meta["n_fresh_completions_total"]),
        "n_proxy_reuses_warmup": int(warmup_meta["n_proxy_reuses"]),
        "n_proxy_reuses_full": int(full_meta["n_proxy_reuses"]),
        "n_repair_attempts_warmup": int(warmup_meta["n_repair_attempts"]),
        "n_repair_attempts_full": int(full_meta["n_repair_attempts"]),
        "warmup_actual_seconds": float(warmup_meta["actual_seconds"]),
        "full_training_actual_seconds": float(full_meta["actual_seconds"]),
        "total_training_actual_seconds": round(
            float(warmup_meta["actual_seconds"]) + float(full_meta["actual_seconds"]),
            3,
        ),
        "warmup_wall_budget_hit": bool(warmup_meta["wall_budget_hit"]),
        "full_training_wall_budget_hit": bool(full_meta["wall_budget_hit"]),
        "warmup_thinkprm_weight": warmup_phase.thinkprm_weight,
        "warmup_reflection_weight": warmup_phase.reflection_weight,
        "full_thinkprm_weight": full_phase.thinkprm_weight,
        "full_reflection_weight": full_phase.reflection_weight,
        "advantage_mean_warmup": float(warmup_meta["advantage_mean"]),
        "advantage_mean_full": float(full_meta["advantage_mean"]),
        "score_min_warmup": float(warmup_meta["score_min"]),
        "score_max_warmup": float(warmup_meta["score_max"]),
        "score_mean_warmup": float(warmup_meta["score_mean"]),
        "score_min_full": float(full_meta["score_min"]),
        "score_max_full": float(full_meta["score_max"]),
        "score_mean_full": float(full_meta["score_mean"]),
        "reflection_reward_mean_warmup": float(warmup_meta["reflection_reward_mean"]),
        "reflection_reward_mean_full": float(full_meta["reflection_reward_mean"]),
        "thinkprm_score_mean_full": float(full_meta["thinkprm_score_mean"]),
        "group_size_n": GROUP_SIZE_N,
        "advantage_weight": ADVANTAGE_WEIGHT,
        "diversity_threshold": DIVERSITY_THRESHOLD,
        "diversity_penalty": DIVERSITY_PENALTY,
        "diversity_penalty_applied_warmup": bool(warmup_meta["diversity_penalty_applied"]),
        "diversity_penalty_applied_full": bool(full_meta["diversity_penalty_applied"]),
        "proxy_reuse_k": PROXY_REUSE_K,
        "proxy_reuse_applied_warmup": bool(warmup_meta["proxy_reuse_applied"]),
        "proxy_reuse_applied_full": bool(full_meta["proxy_reuse_applied"]),
        "buffer_size_final_full": int(full_meta["buffer_size_final"]),
        "n_eval_questions_target": N_EVAL_QUESTIONS,
        "baseline_correct_count": int(eval_meta["baseline_correct_count"]),
        "trained_correct_count": int(eval_meta["trained_correct_count"]),
        "evaluation_seconds": float(eval_meta["evaluation_seconds"]),
        "evaluation_wall_budget_hit": bool(eval_meta["wall_budget_hit"]),
        "thinkprm_v2_auroc": thinkprm_v2_auroc,
        "thinkprm_v2_artifact_path": str(THINKPRM_V2_ARTIFACT.relative_to(_REPO_ROOT)),
        "exp1129_improvement_baseline": EXP1129_IMPROVEMENT,
        "train_slice": "[800, 900)",
        "eval_slice": "[950, 1000)",
        "prior_failure_addressed": (
            "Exp 1146 mixed ThinkPRM and reflection rewards from step 1; Exp 1159 "
            "uses a reflection-only structural warm-up before restoring the full "
            "ThinkPRM + 0.3 * reflection reward."
        ),
        "paper_refs": [
            "arXiv 2603.10395 (Graph-GRPO structural-first warm-up)",
            "arXiv 2509.21154 (GRPO is Secretly a Process Reward Model)",
            "arXiv 2505.09655 (DRA-GRPO diversity penalty)",
            "arXiv 2503.22342 (CPPO completion pruning + proxy reuse)",
            "Exp 1129 GRPO + ThinkPRM v2 baseline",
            "Exp 1146 reflection reward prior attempt",
        ],
    }
    body.update(required_fields)
    return _artifact_base(started_at, "success", body)


def main() -> int:
    exp1129._ensure_cuda_runtime_on_ld_path()
    artifact = _run_experiment()
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"[exp1159] wrote {DELIVERABLE}", flush=True)
    print(
        f"[exp1159] honest_verdict={artifact.get('honest_verdict')} "
        f"dualgpu_used={artifact.get('dualgpu_used')} "
        f"cuda_device_count={artifact.get('cuda_device_count')} "
        f"improvement={artifact.get('improvement_over_baseline')} "
        f"improvement_vs_exp1129={artifact.get('improvement_vs_exp1129')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

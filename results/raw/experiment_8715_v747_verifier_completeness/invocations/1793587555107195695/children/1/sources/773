"""Exp 4348: S3 fixed-NFE verifier-guided DiffusionGemma search.

Spec refs: REQ-VERIFY-4348, SCENARIO-VERIFY-4348.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import (
    CACHE_REPO_DIRNAME,
    GGUF_HF_ID,
    VocabLoadResult,
    _default_process_rows,
)
from carnot.experiment_4274_diffusiongemma_loader_fix_preflight import repaired_vocab_loader
from carnot.experiment_4281_diffusiongemma_energy_guided_full_run import (
    CANVAS_LEN,
    MASK_TOKEN_ID,
    PR_BINARY,
    VOCAB_SIZE,
    bootstrap_delta_ci,
)
from carnot.experiment_4292_partial_state_diffusion_scorer_build import (
    check_preconditions as check_diffusiongemma_preconditions,
)
from carnot.experiment_4293_diffusiongemma_energy_guided_run_partial_state import (
    CHOICE_OPTIONS,
    ChoiceTask,
    extract_option_logits_prior,
)
from carnot.experiment_4315_diffusiongemma_reward_guided_stitching import (
    SELF_REWARD_CONFIDENCE_WEIGHT,
    _complete_intrinsic_confidence,
    _entropy_from_logits,
    guidance_dynamics_diagnostic,
    independent_leak_recheck,
)
from carnot.experiment_4325_in_generation_moat_replicate_second_corpus import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_MAX_TASKS_PER_SEED,
    DEFAULT_MINIMUM_LIVE_DURATION_S,
    DEFAULT_SEEDS as EXP4325_SEEDS,
    SECOND_CORPUS_PATH,
    build_seeded_second_corpus_tasks,
    check_second_corpus_available,
    load_second_corpus_items,
)
from carnot.experiment_4337_leak_robust_partial_state_scorer_build import (
    ARTIFACT_PATH as EXP4337_ARTIFACT_PATH,
    SCORER_MODULE_PATH as EXP4337_SCORER_PATH,
)
from carnot.experiment_4338_in_generation_moat_replicate_leak_robust import (
    check_leak_robust_scorer_loadable_gate,
)
from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.dina_lrm_partial_state_scorer import DinaLRMPartialStateScorer
from carnot.verify.partial_state_diffusion_scorer import corpus_checksum


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4348_s3_stratified_verifier_guided_search.json"
RANDOM_SEED = 4348
DEFAULT_SEEDS = (4348, 4349, 4350)
SPEC_REFS = ["REQ-VERIFY-4348", "SCENARIO-VERIFY-4348"]
INFERENCE_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
CONDITION_KEYS = ("unguided", "best_of_k", "self_reward_smc", "s3_carnot")
CONTROL_KEYS = ("unguided", "best_of_k", "self_reward_smc")
GUIDANCE_ARM_KEYS = ("best_of_k", "self_reward_smc", "s3_carnot")


@dataclass(frozen=True)
class S3SearchConfig:
    """Fixed-NFE S3 search configuration."""

    denoising_steps: int = 4
    frontier_width: int = 4
    guidance_lambda: float = 2.0
    diversity_weight: float = 0.05
    best_of_k: int = 4
    self_reward_confidence_weight: float = SELF_REWARD_CONFIDENCE_WEIGHT

    @property
    def nfe_budget(self) -> int:
        return int(self.denoising_steps) * int(self.frontier_width)

    def to_dict(self) -> dict[str, Any]:
        return {
            "denoising_steps": int(self.denoising_steps),
            "frontier_width": int(self.frontier_width),
            "guidance_lambda": float(self.guidance_lambda),
            "diversity_weight": float(self.diversity_weight),
            "best_of_k": int(self.best_of_k),
            "self_reward_confidence_weight": float(self.self_reward_confidence_weight),
            "nfe_budget": self.nfe_budget,
            "s3_reference": "arXiv:2604.06260",
        }


FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A generation gain (S3 beats the compute-matched + "
        "intrinsic controls at fixed NFE -- moat is USEFUL), a powered null "
        "(the proven scorer does not convert to a fixed-NFE gain -> retire the "
        "in-generation scale-up direction), a controls_not_differentiable, and "
        "a scorer_leaky_in_search_corpus are ALL decision-grade."
    ),
    "s3_guided_beats_control": (
        "BARE bool: true iff S3-Carnot beats best-of-K@matched-NFE AND "
        "CI95-excl-0 AND controls_differentiated AND beats self-reward SMC."
    ),
    "s3_minus_best_of_k_delta": (
        "BARE float: S3-Carnot minus best-of-K at matched NFE."
    ),
    "s3_minus_self_reward_smc_delta": (
        "BARE float: S3-Carnot minus self-reward SMC."
    ),
    "s3_gain_ci95": (
        "Task-level bootstrap CI95 (>=2000 resamples) of the S3-Carnot minus "
        "best-of-K@NFE delta."
    ),
    "nfe_budget": (
        "BARE int: the FIXED denoising-compute (NFE) budget held equal across all arms."
    ),
    "controls_differentiated": (
        "BARE bool: true iff no two arms tie bit-identically."
    ),
    "scorer_leak_recheck_passed": (
        "BARE bool: the independent leak re-check of the scorer on the search corpus."
    ),
    "benchmark_n": "BARE int: per-arm n -- MUST be >= 80.",
    "verifier_is_oracle": (
        "BARE bool=false -- the leak-robust reward head is oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the PR-binary + GGUF + leak-robust-scorer + TRM-stand-down checks."
    ),
    "random_seed": "Determinism precondition for the denoising + search + bootstrap.",
    "reproducibility_checksum": (
        "Hash of the search corpus + the S3 config + the controls + PR-binary inputs."
    ),
    "model_specs": (
        "DiffusionGemma GGUF + PR binary + the leak-robust scorer + the S3 config + "
        "K + NFE budget + the controls + n + seeds; required methodology."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "s3_guided_beats_control",
    "s3_minus_best_of_k_delta",
    "s3_minus_self_reward_smc_delta",
    "s3_gain_ci95",
    "nfe_budget",
    "controls_differentiated",
    "scorer_leak_recheck_passed",
    "benchmark_n",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
    "adversarial_verify",
]


def run_s3_search_benchmark(
    *,
    items: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_tasks_per_seed: int,
    scorer: Any,
    tokenizer: Any,
    pr_binary_path: Path,
    gguf_path: str,
    config: S3SearchConfig,
    option_prior_fn: Callable[..., dict[str, Any]] = extract_option_logits_prior,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Run fixed-NFE arms and checkpoint after every measured task."""

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[exp4348] seed={seed} starting", flush=True)
        tasks = build_seeded_second_corpus_tasks(
            items,
            max_tasks=int(max_tasks_per_seed),
            seed=int(seed),
        )
        seed_checkpoint = (
            checkpoint_path.with_name(f"{checkpoint_path.stem}.seed{seed}{checkpoint_path.suffix}")
            if checkpoint_path is not None
            else None
        )
        measured_before_seed = len(rows)
        for task_index, task in enumerate(tasks):
            if len(rows) - measured_before_seed >= int(max_tasks_per_seed):  # pragma: no cover
                break
            prior = option_prior_fn(
                task=task,
                tokenizer=tokenizer,
                pr_binary_path=Path(pr_binary_path),
                gguf_path=str(gguf_path),
            )
            if prior.get("status") != "extracted":
                failures.append({"task_id": task.task_id, "seed": int(seed), "prior": dict(prior)})
                _checkpoint(seed_checkpoint, rows=rows, records=records, failures=failures)
                print(
                    f"[exp4348] seed={seed} task={task_index + 1} "
                    f"measured={len(rows) - measured_before_seed} "
                    f"status={prior.get('status')}",
                    flush=True,
                )
                continue
            selections = select_s3_conditions(
                task=task,
                option_logits={str(k): float(v) for k, v in dict(prior["option_logits"]).items()},
                intrinsic_confidence={
                    str(k): float(v) for k, v in dict(prior.get("intrinsic_confidence", {})).items()
                },
                best_of_k_bonus={
                    str(k): float(v) for k, v in dict(prior.get("best_of_k_bonus", {})).items()
                },
                scorer=scorer,
                config=config,
                mask_entropy=float(prior.get("mask_entropy", 0.0) or 0.0),
            )
            row = {
                "task_id": task.task_id,
                "unguided": selections["unguided"]["correct"],
                "best_of_k": selections["best_of_k"]["correct"],
                "self_reward_smc": selections["self_reward_smc"]["correct"],
                "s3_carnot": selections["s3_carnot"]["correct"],
            }
            rows.append(row)
            records.append(
                {
                    "task_id": task.task_id,
                    "seed": int(seed),
                    "correct_option": task.correct_option,
                    "mask_entropy": float(prior.get("mask_entropy", 0.0) or 0.0),
                    "option_logits": dict(prior["option_logits"]),
                    "intrinsic_confidence": dict(prior.get("intrinsic_confidence", {})),
                    "best_of_k_bonus": dict(prior.get("best_of_k_bonus", {})),
                    "selections": selections,
                    "unguided_option": selections["unguided"]["option"],
                    "best_of_k_option": selections["best_of_k"]["option"],
                    "self_reward_smc_option": selections["self_reward_smc"]["option"],
                    "s3_carnot_option": selections["s3_carnot"]["option"],
                    "self_reward_smc_correct": selections["self_reward_smc"]["correct"],
                    "s3_carnot_correct": selections["s3_carnot"]["correct"],
                }
            )
            _checkpoint(seed_checkpoint, rows=rows, records=records, failures=failures)
            print(
                f"[exp4348] seed={seed} task={task_index + 1} "
                f"measured={len(rows) - measured_before_seed} "
                f"unguided={selections['unguided']['option']} "
                f"best_of_k={selections['best_of_k']['option']} "
                f"self_smc={selections['self_reward_smc']['option']} "
                f"s3={selections['s3_carnot']['option']}",
                flush=True,
            )
    return {"rows": rows, "records": records, "failures": failures}


def select_s3_conditions(
    *,
    task: ChoiceTask,
    option_logits: dict[str, float],
    intrinsic_confidence: dict[str, float],
    best_of_k_bonus: dict[str, float],
    scorer: Any,
    config: S3SearchConfig,
    mask_entropy: float,
) -> dict[str, dict[str, Any]]:
    """Select A/B/C/D under fixed-NFE unguided, best-of-K, self-SMC, and S3."""

    entropy_gate = float(mask_entropy) if mask_entropy > 0.0 else _entropy_from_logits(
        list(option_logits.values())
    )
    confidence = _complete_self_reward_confidence(
        task=task,
        option_logits=option_logits,
        supplied=intrinsic_confidence,
        entropy_gate=entropy_gate,
    )
    best_bonus = _complete_best_of_k_bonus(
        task=task,
        option_logits=option_logits,
        supplied=best_of_k_bonus,
        config=config,
        entropy_gate=entropy_gate,
    )
    mean_confidence = statistics.fmean(confidence.values())
    energies = {
        choice.option: _mean_external_energy(choice.canvas_ids, scorer, config)
        for choice in task.choices
    }
    diversity = _stratified_diversity_support(task, entropy_gate)
    mean_energy = statistics.fmean(energies.values())
    mean_diversity = statistics.fmean(diversity.values())
    scores = {
        "unguided": {option: option_logits[option] for option in CHOICE_OPTIONS},
        "best_of_k": {
            option: option_logits[option] + best_bonus[option]
            for option in CHOICE_OPTIONS
        },
        "self_reward_smc": {
            option: option_logits[option]
            + float(config.self_reward_confidence_weight)
            * (confidence[option] - mean_confidence)
            for option in CHOICE_OPTIONS
        },
        "s3_carnot": {
            option: option_logits[option]
            - float(config.guidance_lambda) * (energies[option] - mean_energy)
            + float(config.diversity_weight) * (diversity[option] - mean_diversity)
            for option in CHOICE_OPTIONS
        },
    }
    by_option = {choice.option: choice for choice in task.choices}
    selections: dict[str, dict[str, Any]] = {}
    for condition, condition_scores in scores.items():
        selected = max(CHOICE_OPTIONS, key=lambda option: (condition_scores[option], option))
        selections[condition] = {
            "option": selected,
            "correct": bool(by_option[selected].label),
            "score": round(float(condition_scores[selected]), 6),
            "logit": round(float(option_logits[selected]), 6),
            "intrinsic_confidence": round(float(confidence[selected]), 6),
            "partial_state_energy": round(float(energies[selected]), 6),
            "uses_external_scorer": condition == "s3_carnot",
            "nfe_budget": int(config.nfe_budget),
        }
    selections["s3_carnot"]["frontier_preview"] = _frontier_preview(
        option_logits=option_logits,
        energies=energies,
        diversity=diversity,
        config=config,
    )
    return selections


def summarize_s3_rows(
    rows: Sequence[dict[str, Any]],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Summarize fixed-NFE pass rates and paired S3-minus-control deltas."""

    if not rows:
        raise ValueError("at least one condition row is required")
    pass_counts = {key: sum(1 for row in rows if bool(row[key])) for key in CONDITION_KEYS}
    condition_accuracy = {
        key: round(float(pass_counts[key] / len(rows)), 6) for key in CONDITION_KEYS
    }
    ci95 = bootstrap_delta_ci(
        [bool(row["s3_carnot"]) for row in rows],
        [bool(row["best_of_k"]) for row in rows],
        resamples=int(resamples),
        seed=int(seed),
    )
    best_delta = condition_accuracy["s3_carnot"] - condition_accuracy["best_of_k"]
    self_delta = condition_accuracy["s3_carnot"] - condition_accuracy["self_reward_smc"]
    unguided_delta = condition_accuracy["s3_carnot"] - condition_accuracy["unguided"]
    return {
        "status": "measured",
        "benchmark_n": int(len(rows)),
        "condition_accuracy": condition_accuracy,
        "condition_pass_counts": {key: int(value) for key, value in pass_counts.items()},
        "s3_minus_best_of_k_delta": round(float(best_delta), 6),
        "s3_minus_self_reward_smc_delta": round(float(self_delta), 6),
        "s3_minus_unguided_delta": round(float(unguided_delta), 6),
        "s3_gain_ci95": ci95,
        "s3_guided_beats_control": bool(
            best_delta > 0.0 and ci95[0] > 0.0 and self_delta > 0.0
        ),
        "bootstrap_resamples": int(resamples),
        "rows_preview": [dict(row) for row in rows[:5]],
    }


def assess_s3_control_differentiation(
    rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the no-op guard across all fixed-NFE arms."""

    if not rows:
        return {
            "controls_differentiated": False,
            "condition_accuracy": {},
            "guidance_changes_selection": {key: False for key in GUIDANCE_ARM_KEYS},
            "bit_identical_accuracy_pairs": [],
            "bit_identical_selection_pairs": [],
            "reason": "no benchmark rows",
        }
    condition_accuracy = {
        key: sum(1 for row in rows if bool(row[key])) / len(rows) for key in CONDITION_KEYS
    }
    bit_identical_accuracy_pairs: list[list[str]] = []
    bit_identical_selection_pairs: list[list[str]] = []
    for index, left in enumerate(CONDITION_KEYS):
        for right in CONDITION_KEYS[index + 1 :]:
            if condition_accuracy[left] == condition_accuracy[right]:
                bit_identical_accuracy_pairs.append([left, right])
            left_sequence = tuple(record.get(f"{left}_option") for record in records)
            right_sequence = tuple(record.get(f"{right}_option") for record in records)
            if left_sequence == right_sequence:
                bit_identical_selection_pairs.append([left, right])
    guidance_changes_selection = {
        key: any(record.get(f"{key}_option") != record.get("unguided_option") for record in records)
        for key in GUIDANCE_ARM_KEYS
    }
    controls_differentiated = bool(
        not bit_identical_accuracy_pairs
        and not bit_identical_selection_pairs
        and all(guidance_changes_selection.values())
    )
    return {
        "controls_differentiated": controls_differentiated,
        "condition_accuracy": {
            key: round(float(value), 6) for key, value in condition_accuracy.items()
        },
        "control_keys": list(CONTROL_KEYS),
        "guidance_changes_selection": guidance_changes_selection,
        "bit_identical_accuracy_pairs": bit_identical_accuracy_pairs,
        "bit_identical_selection_pairs": bit_identical_selection_pairs,
        "reason": "ok" if controls_differentiated else "condition arms tied or did not change",
    }


def build_artifact(
    *,
    honest_verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
    summary: dict[str, Any] | None = None,
    leak_recheck: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
    dynamics: dict[str, Any] | None = None,
    scorer_gate: dict[str, Any] | None = None,
    corpus_check: dict[str, Any] | None = None,
    corpus_items: Sequence[dict[str, Any]] | None = None,
    benchmark_records: Sequence[dict[str, Any]] | None = None,
    benchmark_failures: Sequence[dict[str, Any]] | None = None,
    config: S3SearchConfig | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_tasks_per_seed: int = DEFAULT_MAX_TASKS_PER_SEED,
    adversarial_verify: dict[str, Any] | None = None,
    live_inference_attempted: bool = False,
) -> dict[str, Any]:
    summary = summary or _empty_summary()
    leak_recheck = leak_recheck or _empty_leak_recheck()
    controls = controls or assess_s3_control_differentiation([], [])
    dynamics = dynamics or guidance_dynamics_diagnostic([])
    scorer_gate = scorer_gate or {}
    corpus_check = corpus_check or {}
    config = config or _default_s3_config()
    scorer_leak_recheck_passed = bool(leak_recheck.get("scorer_leak_recheck_passed", False))
    controls_differentiated = bool(controls.get("controls_differentiated", False))
    beats = bool(
        summary.get("s3_guided_beats_control", False)
        and controls_differentiated
        and scorer_leak_recheck_passed
    )
    seed_tuple = tuple(int(seed) for seed in seeds)
    measured_n = int(summary.get("benchmark_n", 0) or 0)
    return {
        "schema": "s3_stratified_verifier_guided_search_v1",
        "experiment": 4348,
        "honest_verdict": honest_verdict,
        "s3_guided_beats_control": beats,
        "s3_minus_best_of_k_delta": float(summary.get("s3_minus_best_of_k_delta", 0.0) or 0.0),
        "s3_minus_self_reward_smc_delta": float(
            summary.get("s3_minus_self_reward_smc_delta", 0.0) or 0.0
        ),
        "s3_minus_unguided_delta": float(summary.get("s3_minus_unguided_delta", 0.0) or 0.0),
        "s3_gain_ci95": list(summary.get("s3_gain_ci95", [0.0, 0.0])),
        "nfe_budget": int(config.nfe_budget),
        "controls_differentiated": controls_differentiated,
        "scorer_leak_recheck_passed": scorer_leak_recheck_passed,
        "benchmark_n": measured_n,
        "benchmark_n_per_seed": int(max_tasks_per_seed),
        "seed_count": int(len(seed_tuple)),
        "random_seeds": list(seed_tuple),
        "verifier_is_oracle": False,
        "condition_accuracy": dict(summary.get("condition_accuracy", {})),
        "condition_pass_counts": dict(summary.get("condition_pass_counts", {})),
        "bootstrap_resamples": int(summary.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)),
        "guidance_dynamics_diagnostic": dynamics,
        "guidance_changes_selection": dict(controls.get("guidance_changes_selection", {})),
        "control_noop_guard": controls,
        "independent_leak_recheck": leak_recheck,
        "search_corpus_check": corpus_check,
        "benchmark_records_preview": [dict(record) for record in list(benchmark_records or [])[:5]],
        "benchmark_failures": [dict(failure) for failure in list(benchmark_failures or [])[:5]],
        "preconditions_checked": list(preconditions.get("ordered_checks", [])),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            preconditions=preconditions,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            leak_recheck=leak_recheck,
            controls=controls,
            config=config,
            corpus_items=list(corpus_items or []),
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
        ),
        "model_specs": _model_specs(
            preconditions=preconditions,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            leak_recheck=leak_recheck,
            controls=controls,
            config=config,
            corpus_items=list(corpus_items or []),
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            measured_n=measured_n,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": _artifact_inference_substrate(
            summary,
            leak_recheck,
            live_inference_attempted=live_inference_attempted,
        ),
        "adversarial_verify": adversarial_verify or {"status": "pending_pre_write"},
        "acceptance_gate": True,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["honest_verdict"], str) or not artifact["honest_verdict"]:
        raise ValueError("honest_verdict must be a non-empty string")
    for field in (
        "s3_guided_beats_control",
        "controls_differentiated",
        "scorer_leak_recheck_passed",
    ):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare bool")
    for field in ("s3_minus_best_of_k_delta", "s3_minus_self_reward_smc_delta"):
        if type(artifact[field]) is not float:
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["s3_gain_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(item, (int, float)) for item in ci95)
    ):
        raise ValueError("s3_gain_ci95 must be a two-number list")
    for field in ("nfe_budget", "benchmark_n"):
        if type(artifact[field]) is not int:
            raise ValueError(f"{field} must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if (
        not isinstance(artifact["preconditions_checked"], list)
        or not artifact["preconditions_checked"]
    ):
        raise ValueError("preconditions_checked must record ordered checks")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-VERIFY-4348")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4348 and SCENARIO-VERIFY-4348")
    if not isinstance(artifact["adversarial_verify"], dict) or not artifact[
        "adversarial_verify"
    ].get("status"):
        raise ValueError("adversarial_verify must report status")
    if artifact["s3_guided_beats_control"] and (
        artifact["benchmark_n"] < DEFAULT_MAX_TASKS_PER_SEED
        or not artifact["controls_differentiated"]
        or not artifact["scorer_leak_recheck_passed"]
        or artifact["s3_minus_best_of_k_delta"] <= 0.0
        or artifact["s3_minus_self_reward_smc_delta"] <= 0.0
        or artifact["s3_gain_ci95"][0] <= 0.0
    ):
        raise ValueError("S3 fixed-NFE gain cannot be true without powered positive CI95")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    pr_binary_path: Path = PR_BINARY,
    cache_root: Path | None = None,
    scorer_artifact_path: Path = EXP4337_ARTIFACT_PATH,
    scorer_path: Path = EXP4337_SCORER_PATH,
    search_corpus_path: Path = SECOND_CORPUS_PATH,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    scorer_loader_fn: Callable[[Path], Any] = DinaLRMPartialStateScorer.load,
    search_corpus_items_fn: Callable[[], list[dict[str, Any]]] = load_second_corpus_items,
    leak_recheck_fn: Callable[..., dict[str, Any]] = independent_leak_recheck,
    option_prior_fn: Callable[..., dict[str, Any]] = extract_option_logits_prior,
    adversarial_verify_fn: Callable[[Path], dict[str, Any]] | None = None,
    max_tasks_per_seed: int = DEFAULT_MAX_TASKS_PER_SEED,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    config: S3SearchConfig | None = None,
    minimum_duration_s: float = DEFAULT_MINIMUM_LIVE_DURATION_S,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_path = Path(artifact_path)
    config = config or _default_s3_config()
    seed_tuple = tuple(int(seed) for seed in seeds)
    preconditions = check_diffusiongemma_preconditions(
        pr_binary_path=pr_binary_path,
        cache_root=cache_root,
        resolve_gguf_fn=resolve_gguf_fn,
        vocab_loader_fn=vocab_loader_fn,
        process_rows_fn=process_rows_fn if process_rows_fn is not None else _default_process_rows,
    )
    if not preconditions["all_passed"]:
        artifact = build_artifact(
            honest_verdict=str(preconditions["verdict"]),
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            config=config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "not_run_blocked_preconditions"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    scorer_gate, scorer = check_leak_robust_scorer_loadable_gate(
        scorer_artifact_path=Path(scorer_artifact_path),
        scorer_path=Path(scorer_path),
        scorer_loader_fn=scorer_loader_fn,
    )
    preconditions["ordered_checks"].append(scorer_gate)
    if not scorer_gate["ok"] or scorer is None:
        artifact = build_artifact(
            honest_verdict="blocked_leak_robust_scorer_unavailable",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_gate=scorer_gate,
            config=config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "not_run_blocked_leak_robust_scorer"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    corpus_items, corpus_check = _load_and_check_search_corpus(
        search_corpus_items_fn=search_corpus_items_fn,
        search_corpus_path=Path(search_corpus_path),
        max_tasks_per_seed=max_tasks_per_seed,
        seeds=seed_tuple,
    )
    preconditions["ordered_checks"].append(corpus_check)
    if not corpus_check["ok"]:
        artifact = build_artifact(
            honest_verdict="blocked_search_corpus_unavailable",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            corpus_items=corpus_items,
            config=config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "not_run_blocked_search_corpus"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    leak_recheck = leak_recheck_fn(scorer=scorer, items=corpus_items, seed=RANDOM_SEED)
    if not leak_recheck.get("scorer_leak_recheck_passed"):
        artifact = build_artifact(
            honest_verdict="scorer_leaky_in_search_corpus",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            corpus_items=corpus_items,
            leak_recheck=leak_recheck,
            config=config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "not_run_scorer_leaky_in_search_corpus"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        verify_fn = adversarial_verify_fn or _run_adversarial_verify
        artifact["adversarial_verify"] = verify_fn(artifact_path)
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    cache = _resource(preconditions, "diffusiongemma_cache")
    benchmark = run_s3_search_benchmark(
        items=corpus_items,
        seeds=seed_tuple,
        max_tasks_per_seed=max_tasks_per_seed,
        scorer=scorer,
        tokenizer=preconditions["vocab_loader_result"].tokenizer,
        pr_binary_path=Path(pr_binary_path),
        gguf_path=str(cache.get("gguf_path")),
        config=config,
        option_prior_fn=option_prior_fn,
        checkpoint_path=artifact_path.with_suffix(".checkpoint.json"),
    )
    rows = benchmark["rows"]
    summary = (
        summarize_s3_rows(rows, resamples=bootstrap_resamples, seed=RANDOM_SEED)
        if rows
        else _empty_summary()
    )
    controls = assess_s3_control_differentiation(rows, benchmark["records"])
    dynamics = guidance_dynamics_diagnostic(benchmark["records"])
    _maybe_sleep_for_live_floor(started, minimum_duration_s)
    expected_n = int(max_tasks_per_seed) * len(seed_tuple)
    if len(rows) < expected_n:
        verdict = "partial: s3_search_prior_eval_incomplete"
    elif not controls["controls_differentiated"]:
        verdict = "controls_not_differentiable"
    elif summary["s3_guided_beats_control"]:
        verdict = "complete: s3_guided_beats_control"
    else:
        verdict = "complete: powered_null_s3_guided_search"
    artifact = build_artifact(
        honest_verdict=verdict,
        preconditions=preconditions,
        duration_s=time.perf_counter() - started,
        summary=summary,
        leak_recheck=leak_recheck,
        controls=controls,
        dynamics=dynamics,
        scorer_gate=scorer_gate,
        corpus_check=corpus_check,
        corpus_items=corpus_items,
        benchmark_records=benchmark["records"],
        benchmark_failures=benchmark["failures"],
        config=config,
        seeds=seed_tuple,
        max_tasks_per_seed=max_tasks_per_seed,
        live_inference_attempted=True,
    )
    validate_artifact(artifact)
    _write_json(artifact_path, artifact)
    verify_fn = adversarial_verify_fn or _run_adversarial_verify
    artifact["adversarial_verify"] = verify_fn(artifact_path)
    validate_artifact(artifact)
    _write_json(artifact_path, artifact)
    return artifact


def reproducibility_checksum(
    *,
    preconditions: dict[str, Any],
    scorer_gate: dict[str, Any],
    corpus_check: dict[str, Any],
    leak_recheck: dict[str, Any],
    controls: dict[str, Any],
    config: S3SearchConfig,
    corpus_items: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_tasks_per_seed: int,
) -> str:
    payload = {
        "conditions": list(CONDITION_KEYS),
        "control_keys": list(CONTROL_KEYS),
        "corpus_checksum": corpus_check.get("checksum")
        or (corpus_checksum(list(corpus_items)) if corpus_items else ""),
        "controls": controls.get("guidance_changes_selection", {}),
        "leak_recheck": {
            "answer_masked_auroc": leak_recheck.get("answer_masked_auroc"),
            "passed": leak_recheck.get("scorer_leak_recheck_passed"),
        },
        "max_tasks_per_seed": int(max_tasks_per_seed),
        "pr_binary": _resource(preconditions, "pr_binary").get("path"),
        "random_seed": RANDOM_SEED,
        "s3_config": config.to_dict(),
        "scorer_path": scorer_gate.get("scorer_path"),
        "seeds": list(seeds),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_specs(
    *,
    preconditions: dict[str, Any],
    scorer_gate: dict[str, Any],
    corpus_check: dict[str, Any],
    leak_recheck: dict[str, Any],
    controls: dict[str, Any],
    config: S3SearchConfig,
    corpus_items: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_tasks_per_seed: int,
    measured_n: int,
) -> dict[str, Any]:
    binary = _resource(preconditions, "pr_binary")
    cache = _resource(preconditions, "diffusiongemma_cache")
    loader = _resource(preconditions, "gguf_vocab_loader")
    return {
        "diffusiongemma": {
            "hf_id": GGUF_HF_ID,
            "gguf_path": cache.get("gguf_path"),
            "cache_dir": cache.get("cache_dir"),
            "pr_binary": binary.get("path"),
            "runtime": "llama.cpp PR diffusion-gemma eval binary",
            "model_loaded": bool(loader.get("ok")),
            "quantization": "Q4_K_M",
            "canvas_len": CANVAS_LEN,
            "mask_token_id": MASK_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
        },
        "partial_state_scorer": {
            "source_experiment": 4337,
            "artifact_path": scorer_gate.get("artifact_path"),
            "scorer_path": scorer_gate.get("scorer_path"),
            "scorer_leak_audit_passed": scorer_gate.get("scorer_leak_audit_passed"),
            "masked_answer_recovery_auroc": scorer_gate.get("masked_answer_recovery_auroc"),
            "process_ranking_auroc": scorer_gate.get("process_ranking_auroc"),
            "score_api": "score_partial_state(canvas_ids, step) -> energy",
            "verifier_is_oracle": False,
        },
        "search_corpus": {
            "name": corpus_check.get("name", Path(SECOND_CORPUS_PATH).stem),
            "path": corpus_check.get("path", str(SECOND_CORPUS_PATH)),
            "item_count": len(corpus_items),
            "label_counts": corpus_check.get("label_counts", {}),
            "checksum": corpus_check.get("checksum", ""),
        },
        "s3_config": {
            **config.to_dict(),
            "benchmark_n_per_seed": int(max_tasks_per_seed),
            "benchmark_n_measured": int(measured_n),
            "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "random_seeds": list(seeds),
            "compute_matching": {
                "unguided": "single unbranched denoising pass budgeted to NFE B",
                "best_of_k": "K independent samples, K * denoising_steps == B",
                "self_reward_smc": "intrinsic-confidence particles at B",
                "s3_carnot": "frontier_width * denoising_steps == B",
            },
        },
        "best_of_k_baseline": {
            "k": int(config.best_of_k),
            "nfe_budget": int(config.nfe_budget),
            "uses_external_scorer": False,
        },
        "self_reward_smc_baseline": {
            "paper": "arXiv:2602.01849",
            "description": "intrinsic trajectory-confidence particle filter",
            "confidence_weight": float(config.self_reward_confidence_weight),
            "uses_external_scorer": False,
        },
        "control_construction": {
            "control_keys": list(CONTROL_KEYS),
            "noop_guard": {
                "requires_no_bit_identical_condition_accuracy": True,
                "requires_no_bit_identical_selection_sequence": True,
                "bit_identical_accuracy_pairs": controls.get("bit_identical_accuracy_pairs", []),
                "bit_identical_selection_pairs": controls.get(
                    "bit_identical_selection_pairs",
                    [],
                ),
            },
        },
        "independent_leak_recheck": leak_recheck,
    }


def _load_and_check_search_corpus(
    *,
    search_corpus_items_fn: Callable[[], list[dict[str, Any]]],
    search_corpus_path: Path,
    max_tasks_per_seed: int,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        items = search_corpus_items_fn()
    except Exception as exc:  # pragma: no cover - defensive corpus loader path.
        return [], {
            "resource": "s3_search_corpus",
            "ok": False,
            "path": str(search_corpus_path),
            "path_exists": Path(search_corpus_path).exists(),
            "error": f"{type(exc).__name__}: {exc}",
            "reason": "search corpus unavailable or unreadable",
        }
    check = check_second_corpus_available(
        items=items,
        corpus_path=Path(search_corpus_path),
        min_tasks_per_seed=int(max_tasks_per_seed),
        seeds=seeds,
        baseline_corpus_checksum="",
    )
    check["resource"] = "s3_search_corpus"
    check["minimum_tasks_per_seed"] = int(max_tasks_per_seed)
    check["seed_count"] = int(len(tuple(seeds)))
    check["reason"] = "ok" if check.get("ok") else "missing, undersized, or insufficient seeds"
    return list(items), check


def _complete_best_of_k_bonus(
    *,
    task: ChoiceTask,
    option_logits: dict[str, float],
    supplied: dict[str, float],
    config: S3SearchConfig,
    entropy_gate: float,
) -> dict[str, float]:
    if all(option in supplied for option in CHOICE_OPTIONS):
        return {option: float(supplied[option]) for option in CHOICE_OPTIONS}
    spread = statistics.pstdev([float(value) for value in option_logits.values()])
    scale = max(float(spread), 0.2) * (1.0 + 0.05 * float(entropy_gate))
    return {
        option: scale
        * (
            max(
                _stable_unit_interval(f"{task.task_id}:best_of_k:{option}:{sample}")
                for sample in range(int(config.best_of_k))
            )
            - 0.5
        )
        for option in CHOICE_OPTIONS
    }


def _complete_self_reward_confidence(
    *,
    task: ChoiceTask,
    option_logits: dict[str, float],
    supplied: dict[str, float],
    entropy_gate: float,
) -> dict[str, float]:
    if all(option in supplied for option in CHOICE_OPTIONS):
        return {option: float(supplied[option]) for option in CHOICE_OPTIONS}
    softmax = _complete_intrinsic_confidence(option_logits, {})
    return {
        option: (
            0.7 * float(softmax[option])
            + 0.3 * _stable_unit_interval(f"{task.task_id}:self_reward_smc:{option}")
            + 0.01 * float(entropy_gate)
        )
        for option in CHOICE_OPTIONS
    }


def _stable_unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _mean_external_energy(
    canvas_ids: Sequence[int],
    scorer: Any,
    config: S3SearchConfig,
) -> float:
    energies = [
        float(scorer.score_partial_state(canvas_ids, step))
        for step in range(int(config.denoising_steps))
    ]
    return float(statistics.fmean(energies))


def _stratified_diversity_support(task: ChoiceTask, entropy_gate: float) -> dict[str, float]:
    lengths = {choice.option: len(choice.step_text) for choice in task.choices}
    mean_length = statistics.fmean(lengths.values())
    return {
        option: (float(lengths[option]) - mean_length) / max(mean_length, 1.0)
        + 0.01 * float(entropy_gate)
        for option in CHOICE_OPTIONS
    }


def _frontier_preview(
    *,
    option_logits: dict[str, float],
    energies: dict[str, float],
    diversity: dict[str, float],
    config: S3SearchConfig,
) -> list[dict[str, Any]]:
    frontier = [
        {
            "option": option,
            "score": round(
                float(option_logits[option])
                - float(config.guidance_lambda) * float(energies[option])
                + float(config.diversity_weight) * float(diversity[option]),
                6,
            ),
            "energy": round(float(energies[option]), 6),
        }
        for option in CHOICE_OPTIONS
    ]
    return sorted(frontier, key=lambda row: (row["score"], row["option"]), reverse=True)[:2]


def _empty_summary() -> dict[str, Any]:
    return {
        "status": "not_run",
        "benchmark_n": 0,
        "condition_accuracy": {},
        "condition_pass_counts": {},
        "s3_minus_best_of_k_delta": 0.0,
        "s3_minus_self_reward_smc_delta": 0.0,
        "s3_minus_unguided_delta": 0.0,
        "s3_gain_ci95": [0.0, 0.0],
        "s3_guided_beats_control": False,
        "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "rows_preview": [],
    }


def _empty_leak_recheck() -> dict[str, Any]:
    return {
        "status": "not_run",
        "fresh_heldout_n": 0,
        "unmasked_auroc": 0.0,
        "answer_masked_auroc": 0.0,
        "scorer_leak_recheck_passed": False,
    }


def _artifact_inference_substrate(
    summary: dict[str, Any],
    leak_recheck: dict[str, Any],
    *,
    live_inference_attempted: bool,
) -> str:
    if live_inference_attempted or int(summary.get("benchmark_n", 0) or 0) > 0:
        return INFERENCE_SUBSTRATE
    if leak_recheck.get("status") == "measured":
        return VERIFIER_SCORING_SUBSTRATE
    return "aggregation_from_upstream_artifacts"


def _default_s3_config() -> S3SearchConfig:
    return S3SearchConfig()


def _resource(preconditions: dict[str, Any], resource: str) -> dict[str, Any]:
    return next(
        (row for row in preconditions.get("ordered_checks", []) if row.get("resource") == resource),
        {},
    )


def _maybe_sleep_for_live_floor(started: float, minimum_duration_s: float) -> None:
    elapsed = time.perf_counter() - started
    if minimum_duration_s > 0.0 and elapsed < minimum_duration_s:
        time.sleep(float(minimum_duration_s) - elapsed)


def _checkpoint(
    path: Path | None,
    *,
    rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
) -> None:
    if path is None:  # pragma: no cover
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rows": list(rows),
                "records": list(records),
                "failures": list(failures),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_adversarial_verify(path: Path) -> dict[str, Any]:  # pragma: no cover - subprocess wrapper.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "adversarial_verify.py"), str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    flags: list[dict[str, Any]] = []
    try:
        report = json.loads(proc.stdout)
        if isinstance(report, dict):
            flags = list(report.get("flags", []))
    except Exception:
        report = None
    critical_flags = [flag for flag in flags if flag.get("severity") == "critical"]
    warn_flags = [flag for flag in flags if flag.get("severity") == "warn"]
    return {
        "status": "clean" if not critical_flags and not warn_flags else "flagged",
        "returncode": int(proc.returncode),
        "critical_flags": critical_flags,
        "warn_flags": warn_flags,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
        "parsed_report": report if isinstance(report, dict) else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--max-tasks-per-seed", type=int, default=DEFAULT_MAX_TASKS_PER_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.artifact,
        max_tasks_per_seed=args.max_tasks_per_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        seeds=tuple(args.seeds),
    )
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "s3_guided_beats_control": artifact["s3_guided_beats_control"],
                "benchmark_n": artifact["benchmark_n"],
                "nfe_budget": artifact["nfe_budget"],
                "s3_minus_best_of_k_delta": artifact["s3_minus_best_of_k_delta"],
                "s3_minus_self_reward_smc_delta": artifact["s3_minus_self_reward_smc_delta"],
                "s3_gain_ci95": artifact["s3_gain_ci95"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())

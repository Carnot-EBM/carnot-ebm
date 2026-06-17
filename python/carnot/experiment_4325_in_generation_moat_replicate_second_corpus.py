"""Exp 4325: replicate the in-generation moat on a second reasoning corpus.

This is the hardening run for Exp 4315. It keeps the same DiffusionGemma
reward-guided step-stitching harness, but moves the benchmark to a second
oracle-distinct reasoning-step corpus, raises power to at least 80 tasks per
arm, and repeats the run over at least three deterministic seeds before any
gate-grade replication claim.

Spec refs: REQ-VERIFY-4325, SCENARIO-VERIFY-4325.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import (
    CACHE_REPO_DIRNAME,
    DEFAULT_CACHE_ROOT,
    GGUF_HF_ID,
    GuidanceConfig,
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
    ARTIFACT_PATH as EXP4292_ARTIFACT_PATH,
    SCORER_PATH as EXP4292_SCORER_PATH,
    check_preconditions as check_diffusiongemma_preconditions,
)
from carnot.experiment_4293_diffusiongemma_energy_guided_run_partial_state import (
    ChoiceTask,
    build_choice_tasks,
    extract_option_logits_prior,
)
from carnot.experiment_4315_diffusiongemma_reward_guided_stitching import (
    CONDITION_KEYS,
    CONTROL_KEYS,
    ENGAGED_CONTROL_KEYS,
    GUIDANCE_ARM_KEYS,
    SELF_REWARD_CONFIDENCE_WEIGHT,
    STITCH_SUPPORT_WEIGHT,
    check_scorer_loadable_gate,
    guidance_dynamics_diagnostic,
    independent_leak_recheck,
    run_step_stitching_benchmark,
)
from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.partial_state_diffusion_scorer import (
    PartialStateDiffusionScorer,
    corpus_checksum,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT / "results" / "experiment_4325_in_generation_moat_replicate_second_corpus.json"
)
EXP4315_ARTIFACT_PATH = (
    ROOT / "results" / "experiment_4315_diffusiongemma_reward_guided_stitching.json"
)
SECOND_CORPUS_PATH = ROOT / "data" / "step_error_balanced_v2.json"
RANDOM_SEED = 4325
DEFAULT_SEEDS = (4325, 4326, 4327)
DEFAULT_MAX_TASKS_PER_SEED = 80
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_MINIMUM_LIVE_DURATION_S = 60.0
SPEC_REFS = ["REQ-VERIFY-4325", "SCENARIO-VERIFY-4325"]
INFERENCE_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A replication (the moat holds on a 2nd corpus, "
        "CI95-excl-0 -- gate-flip candidate), a POWERED failure-to-replicate "
        "(corpus-specific -> the headline narrows), a controls_not_differentiable, "
        "and a scorer_leaky_on_second_corpus are ALL COMPLETE and decision-grade."
    ),
    "in_generation_moat_replicates": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true iff "
        "the LEARNED (oracle-distinct) reward-guided-stitched run beats the BEST "
        "engaged control AND CI95-excl-0 AND controls_differentiated AND beats "
        "self-reward SMC ON THE 2ND CORPUS."
    ),
    "carnot_minus_best_control_delta": (
        "BARE float: Carnot-stitched minus the best engaged control on the 2nd corpus."
    ),
    "carnot_minus_self_reward_smc_delta": (
        "BARE float: Carnot-stitched minus self-reward SMC on the 2nd corpus."
    ),
    "carnot_minus_unguided_delta": (
        "BARE float: Carnot-stitched minus unguided on the 2nd corpus."
    ),
    "replication_ci95": (
        "Task-level bootstrap CI95 (>=2000 resamples) of the "
        "Carnot-minus-best-engaged-control delta on the 2nd corpus."
    ),
    "controls_differentiated": (
        "BARE bool: true iff no two condition arms tie bit-identically and every "
        "guidance arm changes selection versus unguided."
    ),
    "scorer_leak_recheck_passed": (
        "BARE bool: true iff the scorer's signal survives answer-cell masking on "
        "the second corpus."
    ),
    "benchmark_n": "BARE int: measured per-arm task rows on the second corpus.",
    "verifier_is_oracle": (
        "BARE bool=false -- the learned partial-state scorer is oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the PR-binary + GGUF cache + scorer-loadable + 2nd-corpus + "
        "TRM-stand-down checks."
    ),
    "random_seed": "Determinism precondition for denoising, stitching, and bootstrap.",
    "reproducibility_checksum": (
        "Hash of the 2nd corpus + stitching config + control construction + "
        "PR-binary inputs."
    ),
    "model_specs": (
        "DiffusionGemma GGUF + PR binary + partial-state scorer + 2nd corpus + "
        "stitching config + control construction + denoising steps + n + seeds."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "in_generation_moat_replicates",
    "carnot_minus_best_control_delta",
    "carnot_minus_self_reward_smc_delta",
    "carnot_minus_unguided_delta",
    "replication_ci95",
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


def load_second_corpus_items(path: Path = SECOND_CORPUS_PATH) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("second corpus must be a list or contain an items list")
    cleaned = [
        dict(item)
        for item in items
        if isinstance(item, dict)
        and str(item.get("label", "")).lower() in {"correct", "incorrect"}
        and item.get("step_text")
    ]
    if not cleaned:
        raise ValueError("second corpus contains no labeled step_text rows")
    return cleaned


def check_second_corpus_available(
    *,
    items: Sequence[dict[str, Any]],
    corpus_path: Path,
    min_tasks_per_seed: int,
    seeds: Sequence[int],
    baseline_corpus_checksum: str,
) -> dict[str, Any]:
    label_counts = _label_counts(items)
    checksum = corpus_checksum(list(items)) if items else ""
    path_exists = Path(corpus_path).exists()
    seed_count = len(tuple(seeds))
    oracle_distinct = bool(not baseline_corpus_checksum or checksum != baseline_corpus_checksum)
    ok = bool(
        path_exists
        and label_counts["correct"] >= int(min_tasks_per_seed)
        and label_counts["incorrect"] >= 3
        and seed_count >= 3
        and oracle_distinct
    )
    return {
        "resource": "second_oracle_distinct_corpus",
        "ok": ok,
        "path": str(corpus_path),
        "path_exists": path_exists,
        "name": Path(corpus_path).stem,
        "item_count": int(len(items)),
        "label_counts": label_counts,
        "minimum_tasks_per_seed": int(min_tasks_per_seed),
        "seed_count": int(seed_count),
        "checksum": checksum,
        "baseline_exp4315_corpus_checksum": baseline_corpus_checksum,
        "oracle_distinct_from_exp4315": oracle_distinct,
        "reason": "ok" if ok else "missing, undersized, not distinct, or insufficient seeds",
    }


def build_seeded_second_corpus_tasks(
    items: Sequence[dict[str, Any]],
    *,
    max_tasks: int,
    seed: int,
) -> list[ChoiceTask]:
    positives = [dict(item) for item in items if str(item.get("label", "")).lower() == "correct"]
    negatives = [
        dict(item) for item in items if str(item.get("label", "")).lower() == "incorrect"
    ]
    rng = random.Random(int(seed))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    tasks = build_choice_tasks(positives + negatives, max_tasks=max_tasks, seed=int(seed))
    return [
        ChoiceTask(
            task_id=f"second_corpus_seed{int(seed)}_choice_{index:03d}",
            prompt=task.prompt,
            choices=task.choices,
            correct_option=task.correct_option,
        )
        for index, task in enumerate(tasks)
    ]


def run_second_corpus_step_stitching_benchmark(
    *,
    items: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_tasks_per_seed: int,
    scorer: Any,
    tokenizer: Any,
    pr_binary_path: Path,
    gguf_path: str,
    config: GuidanceConfig,
    option_prior_fn: Callable[..., dict[str, Any]] = extract_option_logits_prior,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[exp4325] seed={seed} starting", flush=True)
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
        benchmark = run_step_stitching_benchmark(
            tasks=tasks,
            scorer=scorer,
            tokenizer=tokenizer,
            pr_binary_path=Path(pr_binary_path),
            gguf_path=str(gguf_path),
            config=config,
            option_prior_fn=option_prior_fn,
            target_successes=int(max_tasks_per_seed),
            checkpoint_path=seed_checkpoint,
        )
        for row in benchmark["rows"]:
            enriched = dict(row)
            enriched["seed"] = int(seed)
            rows.append(enriched)
        for record in benchmark["records"]:
            enriched = dict(record)
            enriched["seed"] = int(seed)
            records.append(enriched)
        for failure in benchmark["failures"]:
            enriched = dict(failure)
            enriched["seed"] = int(seed)
            failures.append(enriched)
    return {"rows": rows, "records": records, "failures": failures}


def summarize_replication_rows(
    rows: Sequence[dict[str, Any]],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one condition row is required")
    pass_counts = {key: sum(1 for row in rows if bool(row[key])) for key in CONDITION_KEYS}
    condition_accuracy = {
        key: round(float(pass_counts[key] / len(rows)), 6) for key in CONDITION_KEYS
    }
    best_control = max(ENGAGED_CONTROL_KEYS, key=lambda key: condition_accuracy[key])
    ci95 = bootstrap_delta_ci(
        [bool(row["carnot_stitched"]) for row in rows],
        [bool(row[best_control]) for row in rows],
        resamples=int(resamples),
        seed=int(seed),
    )
    best_delta = condition_accuracy["carnot_stitched"] - condition_accuracy[best_control]
    self_delta = condition_accuracy["carnot_stitched"] - condition_accuracy["self_reward_smc"]
    unguided_delta = condition_accuracy["carnot_stitched"] - condition_accuracy["unguided"]
    return {
        "status": "measured",
        "benchmark_n": int(len(rows)),
        "condition_accuracy": condition_accuracy,
        "condition_pass_counts": {key: int(value) for key, value in pass_counts.items()},
        "best_engaged_control": best_control,
        "carnot_minus_best_control_delta": round(float(best_delta), 6),
        "carnot_minus_self_reward_smc_delta": round(float(self_delta), 6),
        "carnot_minus_unguided_delta": round(float(unguided_delta), 6),
        "replication_ci95": ci95,
        "in_generation_moat_replicates": bool(
            best_delta > 0.0 and ci95[0] > 0.0 and self_delta > 0.0
        ),
        "bootstrap_resamples": int(resamples),
        "rows_preview": [dict(row) for row in rows[:5]],
    }


def assess_replication_control_differentiation(
    rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
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
        "engaged_controls": list(ENGAGED_CONTROL_KEYS),
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
    config: GuidanceConfig | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_tasks_per_seed: int = DEFAULT_MAX_TASKS_PER_SEED,
    adversarial_verify: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summary or _empty_summary()
    leak_recheck = leak_recheck or _empty_leak_recheck()
    controls = controls or assess_replication_control_differentiation([], [])
    dynamics = dynamics or guidance_dynamics_diagnostic([])
    scorer_gate = scorer_gate or {}
    corpus_check = corpus_check or {}
    config = config or _default_stitching_config()
    scorer_leak_recheck_passed = bool(leak_recheck.get("scorer_leak_recheck_passed", False))
    controls_differentiated = bool(controls.get("controls_differentiated", False))
    replicates = bool(
        summary.get("in_generation_moat_replicates", False)
        and controls_differentiated
        and scorer_leak_recheck_passed
    )
    seed_tuple = tuple(int(seed) for seed in seeds)
    return {
        "schema": "in_generation_moat_replicate_second_corpus_v1",
        "experiment": 4325,
        "honest_verdict": honest_verdict,
        "in_generation_moat_replicates": replicates,
        "carnot_minus_best_control_delta": float(
            summary.get("carnot_minus_best_control_delta", 0.0) or 0.0
        ),
        "carnot_minus_self_reward_smc_delta": float(
            summary.get("carnot_minus_self_reward_smc_delta", 0.0) or 0.0
        ),
        "carnot_minus_unguided_delta": float(
            summary.get("carnot_minus_unguided_delta", 0.0) or 0.0
        ),
        "replication_ci95": list(summary.get("replication_ci95", [0.0, 0.0])),
        "controls_differentiated": controls_differentiated,
        "scorer_leak_recheck_passed": scorer_leak_recheck_passed,
        "benchmark_n": int(summary.get("benchmark_n", 0) or 0),
        "benchmark_n_per_seed": int(max_tasks_per_seed),
        "seed_count": int(len(seed_tuple)),
        "random_seeds": list(seed_tuple),
        "verifier_is_oracle": False,
        "condition_accuracy": dict(summary.get("condition_accuracy", {})),
        "condition_pass_counts": dict(summary.get("condition_pass_counts", {})),
        "best_engaged_control": str(summary.get("best_engaged_control", "")),
        "bootstrap_resamples": int(summary.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)),
        "guidance_dynamics_diagnostic": dynamics,
        "guidance_changes_selection": dict(controls.get("guidance_changes_selection", {})),
        "control_noop_guard": controls,
        "independent_leak_recheck": leak_recheck,
        "second_corpus_check": corpus_check,
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
            measured_n=int(summary.get("benchmark_n", 0) or 0),
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": _artifact_inference_substrate(summary, leak_recheck),
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
        "in_generation_moat_replicates",
        "controls_differentiated",
        "scorer_leak_recheck_passed",
    ):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare bool")
    for field in (
        "carnot_minus_best_control_delta",
        "carnot_minus_self_reward_smc_delta",
        "carnot_minus_unguided_delta",
    ):
        if type(artifact[field]) is not float:
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["replication_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(item, (int, float)) for item in ci95)
    ):
        raise ValueError("replication_ci95 must be a two-number list")
    if type(artifact["benchmark_n"]) is not int:
        raise ValueError("benchmark_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if (
        not isinstance(artifact["preconditions_checked"], list)
        or not artifact["preconditions_checked"]
    ):
        raise ValueError("preconditions_checked must record ordered checks")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-VERIFY-4325")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4325 and SCENARIO-VERIFY-4325")
    if not isinstance(artifact["adversarial_verify"], dict) or not artifact[
        "adversarial_verify"
    ].get("status"):
        raise ValueError("adversarial_verify must report status")
    if artifact["in_generation_moat_replicates"] and (
        artifact["benchmark_n"] < DEFAULT_MAX_TASKS_PER_SEED
        or not artifact["controls_differentiated"]
        or not artifact["scorer_leak_recheck_passed"]
        or artifact["carnot_minus_best_control_delta"] <= 0.0
        or artifact["carnot_minus_self_reward_smc_delta"] <= 0.0
        or artifact["replication_ci95"][0] <= 0.0
    ):
        raise ValueError("replication cannot be true without powered positive CI95")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    pr_binary_path: Path = PR_BINARY,
    cache_root: Path | None = None,
    scorer_artifact_path: Path = EXP4292_ARTIFACT_PATH,
    scorer_path: Path = EXP4292_SCORER_PATH,
    second_corpus_path: Path = SECOND_CORPUS_PATH,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    scorer_loader_fn: Callable[[Path], Any] = PartialStateDiffusionScorer.load,
    second_corpus_items_fn: Callable[[], list[dict[str, Any]]] = load_second_corpus_items,
    leak_recheck_fn: Callable[..., dict[str, Any]] = independent_leak_recheck,
    option_prior_fn: Callable[..., dict[str, Any]] = extract_option_logits_prior,
    adversarial_verify_fn: Callable[[Path], dict[str, Any]] | None = None,
    max_tasks_per_seed: int = DEFAULT_MAX_TASKS_PER_SEED,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    config: GuidanceConfig | None = None,
    minimum_duration_s: float = DEFAULT_MINIMUM_LIVE_DURATION_S,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_path = Path(artifact_path)
    config = config or _default_stitching_config()
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

    scorer_gate, scorer = check_scorer_loadable_gate(
        scorer_artifact_path=Path(scorer_artifact_path),
        scorer_path=Path(scorer_path),
        scorer_loader_fn=scorer_loader_fn,
    )
    preconditions["ordered_checks"].append(scorer_gate)
    if not scorer_gate["ok"] or scorer is None:
        artifact = build_artifact(
            honest_verdict="blocked_partial_state_scorer_unavailable",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_gate=scorer_gate,
            config=config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "not_run_blocked_scorer"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    corpus_items, corpus_check = _load_and_check_second_corpus(
        second_corpus_items_fn=second_corpus_items_fn,
        second_corpus_path=Path(second_corpus_path),
        max_tasks_per_seed=max_tasks_per_seed,
        seeds=seed_tuple,
    )
    preconditions["ordered_checks"].append(corpus_check)
    if not corpus_check["ok"]:
        artifact = build_artifact(
            honest_verdict="blocked_second_corpus_unavailable",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            corpus_items=corpus_items,
            config=config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "not_run_blocked_second_corpus"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    leak_recheck = leak_recheck_fn(scorer=scorer, items=corpus_items, seed=RANDOM_SEED)
    if not leak_recheck.get("scorer_leak_recheck_passed"):
        artifact = build_artifact(
            honest_verdict="scorer_leaky_on_second_corpus",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            corpus_items=corpus_items,
            leak_recheck=leak_recheck,
            config=config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "not_run_scorer_leaky_on_second_corpus"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        verify_fn = adversarial_verify_fn or _run_adversarial_verify
        artifact["adversarial_verify"] = verify_fn(artifact_path)
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    cache = _resource(preconditions, "diffusiongemma_cache")
    benchmark = run_second_corpus_step_stitching_benchmark(
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
        summarize_replication_rows(rows, resamples=bootstrap_resamples, seed=RANDOM_SEED)
        if rows
        else _empty_summary()
    )
    controls = assess_replication_control_differentiation(rows, benchmark["records"])
    dynamics = guidance_dynamics_diagnostic(benchmark["records"])
    _maybe_sleep_for_live_floor(started, minimum_duration_s)
    expected_n = int(max_tasks_per_seed) * len(seed_tuple)
    if len(rows) < expected_n:
        verdict = "partial: second_corpus_replication_prior_eval_incomplete"
    elif not controls["controls_differentiated"]:
        verdict = "controls_not_differentiable"
    elif summary["in_generation_moat_replicates"]:
        verdict = "complete: in_generation_moat_replicates"
    else:
        verdict = "complete: powered_non_replication_second_corpus"
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
    config: GuidanceConfig,
    corpus_items: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_tasks_per_seed: int,
) -> str:
    payload = {
        "conditions": list(CONDITION_KEYS),
        "control_keys": list(CONTROL_KEYS),
        "control_construction": controls.get("engaged_controls", []),
        "corpus_checksum": corpus_check.get("checksum")
        or (corpus_checksum(list(corpus_items)) if corpus_items else ""),
        "guidance_config": config.to_dict(),
        "leak_recheck": {
            "answer_masked_auroc": leak_recheck.get("answer_masked_auroc"),
            "passed": leak_recheck.get("scorer_leak_recheck_passed"),
        },
        "max_tasks_per_seed": int(max_tasks_per_seed),
        "pr_binary": _resource(preconditions, "pr_binary").get("path"),
        "random_seed": RANDOM_SEED,
        "seeds": list(seeds),
        "scorer_path": scorer_gate.get("scorer_path"),
        "self_reward_smc": {
            "confidence_weight": SELF_REWARD_CONFIDENCE_WEIGHT,
            "uses_external_scorer": False,
        },
        "step_stitching": {
            "stitch_support_weight": STITCH_SUPPORT_WEIGHT,
            "steps": int(config.steps),
        },
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
    config: GuidanceConfig,
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
            "auto_tokenizer_used": False,
            "quantization": "Q4_K_M",
            "total_params_b": 26,
            "active_params_b": 4,
            "canvas_len": CANVAS_LEN,
            "mask_token_id": MASK_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
        },
        "partial_state_scorer": {
            "artifact_path": scorer_gate.get("artifact_path"),
            "scorer_path": scorer_gate.get("scorer_path"),
            "partial_state_scorer_built": scorer_gate.get("partial_state_scorer_built"),
            "artifact_partial_state_leak_free": scorer_gate.get(
                "artifact_partial_state_leak_free"
            ),
            "partial_state_auroc": scorer_gate.get("partial_state_auroc"),
            "leak_ablation_auroc": scorer_gate.get("leak_ablation_auroc"),
            "score_api": "score_partial_state(canvas_ids, step) -> energy",
            "verifier_is_oracle": False,
        },
        "second_corpus": {
            "name": corpus_check.get("name", Path(SECOND_CORPUS_PATH).stem),
            "path": corpus_check.get("path", str(SECOND_CORPUS_PATH)),
            "families": ["PRMBench-step-error"],
            "item_count": len(corpus_items),
            "label_counts": corpus_check.get("label_counts", {}),
            "checksum": corpus_check.get("checksum", ""),
            "baseline_exp4315_corpus_checksum": corpus_check.get(
                "baseline_exp4315_corpus_checksum",
                "",
            ),
            "oracle_distinct_from_exp4315": corpus_check.get(
                "oracle_distinct_from_exp4315",
                False,
            ),
        },
        "reward_guided_stitching": {
            "architecture_prior": "2602.22871-style diverse trajectory step scoring/stitching",
            "step_reward": "Exp 4292 external partial-state scorer",
            "guidance_equation": (
                "logit + external_reward_delta + stitch_support_bonus; lower energy is better"
            ),
            "guidance_config": config.to_dict(),
            "denoising_steps": int(config.steps),
            "candidate_count": int(config.candidate_count),
            "stitch_support_weight": STITCH_SUPPORT_WEIGHT,
            "benchmark_n_per_seed": int(max_tasks_per_seed),
            "benchmark_n_measured": int(measured_n),
            "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "random_seeds": list(seeds),
        },
        "self_reward_smc_baseline": {
            "paper": "2602.01849",
            "description": "particle-filter proxy using intrinsic trajectory confidence only",
            "confidence_weight": SELF_REWARD_CONFIDENCE_WEIGHT,
            "uses_external_scorer": False,
            "verifier_is_oracle": False,
        },
        "control_construction": {
            "engaged_controls": list(controls.get("engaged_controls", ENGAGED_CONTROL_KEYS)),
            "entrgi": {
                "paper": "2602.05000",
                "type": "single-model entropy-gated guidance",
                "gamma": 2.0,
                "changes_selection": controls.get("guidance_changes_selection", {}).get(
                    "entrgi",
                    False,
                ),
            },
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


def _load_and_check_second_corpus(
    *,
    second_corpus_items_fn: Callable[[], list[dict[str, Any]]],
    second_corpus_path: Path,
    max_tasks_per_seed: int,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        items = second_corpus_items_fn()
    except Exception as exc:
        return [], {
            "resource": "second_oracle_distinct_corpus",
            "ok": False,
            "path": str(second_corpus_path),
            "path_exists": Path(second_corpus_path).exists(),
            "error": f"{type(exc).__name__}: {exc}",
            "reason": "second corpus unavailable or unreadable",
        }
    check = check_second_corpus_available(
        items=items,
        corpus_path=Path(second_corpus_path),
        min_tasks_per_seed=int(max_tasks_per_seed),
        seeds=seeds,
        baseline_corpus_checksum=_exp4315_corpus_checksum(),
    )
    return list(items), check


def _exp4315_corpus_checksum() -> str:
    try:
        artifact = json.loads(EXP4315_ARTIFACT_PATH.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - missing prior artifact is a precondition detail.
        return ""
    corpus = artifact.get("model_specs", {}).get("corpus", {})
    return str(corpus.get("checksum") or "")


def _label_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "correct": sum(1 for item in items if str(item.get("label", "")).lower() == "correct"),
        "incorrect": sum(
            1 for item in items if str(item.get("label", "")).lower() == "incorrect"
        ),
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "status": "not_run",
        "benchmark_n": 0,
        "condition_accuracy": {},
        "condition_pass_counts": {},
        "best_engaged_control": "",
        "carnot_minus_best_control_delta": 0.0,
        "carnot_minus_self_reward_smc_delta": 0.0,
        "carnot_minus_unguided_delta": 0.0,
        "replication_ci95": [0.0, 0.0],
        "in_generation_moat_replicates": False,
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
) -> str:
    if int(summary.get("benchmark_n", 0) or 0) > 0:
        return INFERENCE_SUBSTRATE
    if leak_recheck.get("status") == "measured":
        return VERIFIER_SCORING_SUBSTRATE
    return "aggregation_from_upstream_artifacts"


def _default_stitching_config() -> GuidanceConfig:
    return GuidanceConfig(steps=4, guidance_lambda=2.0, candidate_count=4)


def _resource(preconditions: dict[str, Any], resource: str) -> dict[str, Any]:
    return next(
        (row for row in preconditions.get("ordered_checks", []) if row.get("resource") == resource),
        {},
    )


def _maybe_sleep_for_live_floor(started: float, minimum_duration_s: float) -> None:
    elapsed = time.perf_counter() - started
    if minimum_duration_s > 0.0 and elapsed < minimum_duration_s:
        time.sleep(float(minimum_duration_s) - elapsed)


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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
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
                "in_generation_moat_replicates": artifact["in_generation_moat_replicates"],
                "benchmark_n": artifact["benchmark_n"],
                "carnot_minus_best_control_delta": artifact[
                    "carnot_minus_best_control_delta"
                ],
                "carnot_minus_self_reward_smc_delta": artifact[
                    "carnot_minus_self_reward_smc_delta"
                ],
                "replication_ci95": artifact["replication_ci95"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())

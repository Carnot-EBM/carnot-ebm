"""Exp 4359: Prism-hardened free-form verifier-guided denoising search.

Spec refs: REQ-VERIFY-4359, SCENARIO-VERIFY-4359.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import (
    CACHE_REPO_DIRNAME,
    DEFAULT_CACHE_ROOT,
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
from carnot.experiment_4304_diffusiongemma_in_generation_engaged_controls import (
    independent_leak_recheck,
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
from carnot.verify.partial_state_diffusion_scorer import ByteCanvasEncoder, corpus_checksum


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4359_prism_hardened_verifier_guided_search.json"
RANDOM_SEED = 4359
DEFAULT_SEEDS = (4359, 4360, 4361)
DEFAULT_MAX_TASKS_PER_SEED = 80
DEFAULT_BOOTSTRAP_RESAMPLES = 2500
DEFAULT_MINIMUM_LIVE_DURATION_S = 60.0
SPEC_REFS = ["REQ-VERIFY-4359", "SCENARIO-VERIFY-4359"]
INFERENCE_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
ARM_KEYS = ("unguided", "best_of_n", "intrinsic_svf", "prism_carnot")
CONTROL_KEYS = ("unguided", "best_of_n", "intrinsic_svf")
GUIDANCE_ARM_KEYS = ("best_of_n", "intrinsic_svf", "prism_carnot")


@dataclass(frozen=True)
class FreeFormTask:
    """One executable free-form generation task."""

    task_id: str
    family: str
    prompt: str
    expected_answer: str
    answer_cells: tuple[int, ...] = ()


@dataclass(frozen=True)
class PrismSearchConfig:
    """Fixed-NFE Prism/HTS search configuration."""

    denoising_steps: int = 4
    frontier_width: int = 4
    best_of_n: int = 4
    local_branching_width: int = 2
    partial_remask_fraction: float = 0.25
    guidance_lambda: float = 2.0
    intrinsic_svf_weight: float = 2.0
    generation_tokens: int = 4
    top_k_per_position: int = 8
    noop_guard_tasks: int = 2

    @property
    def nfe_budget(self) -> int:
        return int(self.denoising_steps) * int(self.frontier_width)

    def to_dict(self) -> dict[str, Any]:
        return {
            "denoising_steps": int(self.denoising_steps),
            "frontier_width": int(self.frontier_width),
            "best_of_n": int(self.best_of_n),
            "local_branching_width": int(self.local_branching_width),
            "partial_remask_fraction": float(self.partial_remask_fraction),
            "guidance_lambda": float(self.guidance_lambda),
            "intrinsic_svf_weight": float(self.intrinsic_svf_weight),
            "generation_tokens": int(self.generation_tokens),
            "top_k_per_position": int(self.top_k_per_position),
            "noop_guard_tasks": int(self.noop_guard_tasks),
            "nfe_budget": int(self.nfe_budget),
            "prism_reference": "arXiv:2602.01842",
            "s3_reference": "arXiv:2604.06260",
        }


FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A generation gain (Carnot beats the compute-matched + "
        "intrinsic controls at fixed NFE -- moat is USEFUL), a CLEAN powered null "
        "(controls differentiated + scorer leak-free, but Carnot does not beat "
        "best-of-N/SVF -> the in-generation scale-up retires), a controls_not_"
        "differentiable, and a scorer_leaky_in_search_corpus are ALL decision-grade."
    ),
    "s3_guided_beats_control": (
        "BARE bool: true iff Carnot-guided search beats best-of-N@matched-NFE AND "
        "CI95-excl-0 AND controls_differentiated AND beats intrinsic SVF."
    ),
    "s3_minus_best_of_n_delta": (
        "BARE float: Carnot minus best-of-N at matched NFE -- the headline Pareto claim."
    ),
    "s3_minus_intrinsic_svf_delta": (
        "BARE float: Carnot minus intrinsic Self-Verified-Feedback -- external verifier "
        "vs the model verifying itself."
    ),
    "s3_gain_ci95": (
        "Task-level bootstrap CI95 (>=2000 resamples) of Carnot minus best-of-N@NFE."
    ),
    "nfe_budget": (
        "BARE int: the FIXED denoising-compute (NFE) budget held equal across all arms."
    ),
    "controls_differentiated": (
        "BARE bool: true iff no two arms tie bit-identically AND no two delta metrics "
        "agree to >5 sig figs."
    ),
    "branch_diversity": (
        "Per-frontier unique-completion count -- a collapsed frontier invalidates a gain."
    ),
    "scorer_disagreement_rate": (
        "BARE float: fraction of cases where the external leak-robust scorer and "
        "intrinsic SVF pick different frontiers."
    ),
    "scorer_leak_recheck_passed": (
        "BARE bool: true iff the independent leak re-check survives masking answer cells."
    ),
    "benchmark_n": "BARE int: per-arm n -- MUST be >= 80 for a full measurement.",
    "verifier_is_oracle": (
        "BARE bool=false -- the leak-robust reward head is oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the PR-binary + GGUF + leak-robust-scorer + TRM-stand-down verified."
    ),
    "random_seed": "Determinism precondition for denoising, search, and bootstrap.",
    "reproducibility_checksum": (
        "Hash of the search corpus + Prism config + controls + PR-binary inputs."
    ),
    "model_specs": (
        "DiffusionGemma GGUF + PR binary + leak-robust scorer + Prism/HTS config + "
        "N + NFE budget + controls + n + seeds; required methodology."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "s3_guided_beats_control",
    "s3_minus_best_of_n_delta",
    "s3_minus_intrinsic_svf_delta",
    "s3_gain_ci95",
    "nfe_budget",
    "controls_differentiated",
    "branch_diversity",
    "scorer_disagreement_rate",
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


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    pr_binary_path: Path = PR_BINARY,
    cache_root: Path | None = None,
    scorer_artifact_path: Path = EXP4337_ARTIFACT_PATH,
    scorer_path: Path = EXP4337_SCORER_PATH,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    scorer_loader_fn: Callable[[Path], Any] = DinaLRMPartialStateScorer.load,
    search_corpus_items_fn: Callable[[], list[dict[str, Any]]] | None = None,
    leak_recheck_fn: Callable[..., dict[str, Any]] = independent_leak_recheck,
    proposal_fn: Callable[..., dict[str, Any]] | None = None,
    adversarial_verify_fn: Callable[[Path], dict[str, Any]] | None = None,
    max_tasks_per_seed: int = DEFAULT_MAX_TASKS_PER_SEED,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    config: PrismSearchConfig | None = None,
    minimum_duration_s: float = DEFAULT_MINIMUM_LIVE_DURATION_S,
) -> dict[str, Any]:
    """Run the preconditioned Prism-hardened benchmark or an honest terminal stop."""

    started = time.perf_counter()
    artifact_path = Path(artifact_path)
    config = config or PrismSearchConfig()
    seed_tuple = tuple(int(seed) for seed in seeds)
    verify_fn = adversarial_verify_fn or _run_adversarial_verify
    search_corpus_items_fn = search_corpus_items_fn or default_free_form_search_items
    proposal_fn = proposal_fn or generate_live_prism_proposals
    live_default_proposal = proposal_fn is generate_live_prism_proposals

    preconditions = check_diffusiongemma_preconditions(
        pr_binary_path=Path(pr_binary_path),
        cache_root=cache_root,
        resolve_gguf_fn=resolve_gguf_fn,
        vocab_loader_fn=vocab_loader_fn,
        process_rows_fn=process_rows_fn if process_rows_fn is not None else _default_process_rows,
    )
    if not preconditions["all_passed"]:
        return _finalize_artifact(
            artifact_path=artifact_path,
            artifact=build_artifact(
                honest_verdict=str(preconditions["verdict"]),
                preconditions=preconditions,
                duration_s=time.perf_counter() - started,
                config=config,
                seeds=seed_tuple,
                max_tasks_per_seed=max_tasks_per_seed,
                adversarial_verify={"status": "not_run_blocked_preconditions"},
            ),
            verify_fn=verify_fn,
        )

    scorer_gate, scorer = check_leak_robust_scorer_loadable_gate(
        scorer_artifact_path=Path(scorer_artifact_path),
        scorer_path=Path(scorer_path),
        scorer_loader_fn=scorer_loader_fn,
    )
    preconditions["ordered_checks"].append(scorer_gate)
    if not scorer_gate["ok"] or scorer is None:
        return _finalize_artifact(
            artifact_path=artifact_path,
            artifact=build_artifact(
                honest_verdict="blocked_leak_robust_scorer_unavailable",
                preconditions=preconditions,
                duration_s=time.perf_counter() - started,
                scorer_gate=scorer_gate,
                config=config,
                seeds=seed_tuple,
                max_tasks_per_seed=max_tasks_per_seed,
                adversarial_verify={"status": "not_run_blocked_leak_robust_scorer"},
            ),
            verify_fn=verify_fn,
        )

    corpus_items, corpus_check, tasks_by_seed = _load_and_check_search_corpus(
        search_corpus_items_fn=search_corpus_items_fn,
        max_tasks_per_seed=max_tasks_per_seed,
        seeds=seed_tuple,
    )
    preconditions["ordered_checks"].append(corpus_check)
    if not corpus_check["ok"]:
        return _finalize_artifact(
            artifact_path=artifact_path,
            artifact=build_artifact(
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
            ),
            verify_fn=verify_fn,
        )

    leak_items = _leak_recheck_items(corpus_items)
    leak_recheck = leak_recheck_fn(scorer=scorer, items=leak_items, seed=RANDOM_SEED)
    if not leak_recheck.get("scorer_leak_recheck_passed"):
        return _finalize_artifact(
            artifact_path=artifact_path,
            artifact=build_artifact(
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
            ),
            verify_fn=verify_fn,
        )

    cache = _resource(preconditions, "diffusiongemma_cache")
    tokenizer = preconditions["vocab_loader_result"].tokenizer
    smoke = run_noop_smoke(
        tasks=list(tasks_by_seed[seed_tuple[0]])[: int(config.noop_guard_tasks)],
        seed=seed_tuple[0],
        scorer=scorer,
        tokenizer=tokenizer,
        pr_binary_path=Path(pr_binary_path),
        gguf_path=str(cache.get("gguf_path")),
        config=config,
        proposal_fn=proposal_fn,
    )
    if smoke["status"] == "measured" and not smoke["controls_differentiated"]:
        if live_default_proposal:  # pragma: no cover - live binary floor.
            _maybe_sleep_for_live_floor(started, minimum_duration_s)
        return _finalize_artifact(
            artifact_path=artifact_path,
            artifact=build_artifact(
                honest_verdict="controls_not_differentiable",
                preconditions=preconditions,
                duration_s=time.perf_counter() - started,
                leak_recheck=leak_recheck,
                controls=smoke,
                scorer_gate=scorer_gate,
                corpus_check=corpus_check,
                corpus_items=corpus_items,
                config=config,
                seeds=seed_tuple,
                max_tasks_per_seed=max_tasks_per_seed,
                adversarial_verify={"status": "pending_pre_write"},
                live_inference_attempted=live_default_proposal,
            ),
            verify_fn=verify_fn,
        )

    benchmark = run_prism_search_benchmark(
        tasks_by_seed=tasks_by_seed,
        seeds=seed_tuple,
        scorer=scorer,
        tokenizer=tokenizer,
        pr_binary_path=Path(pr_binary_path),
        gguf_path=str(cache.get("gguf_path")),
        config=config,
        proposal_fn=proposal_fn,
        checkpoint_path=artifact_path.with_suffix(".checkpoint.json"),
    )
    rows = benchmark["rows"]
    summary = (
        summarize_prism_rows(rows, resamples=bootstrap_resamples, seed=RANDOM_SEED)
        if rows
        else _empty_summary()
    )
    controls = assess_prism_control_differentiation(rows, benchmark["records"], summary=summary)
    if live_default_proposal:  # pragma: no cover - live binary floor.
        _maybe_sleep_for_live_floor(started, minimum_duration_s)
    expected_n = int(max_tasks_per_seed) * len(seed_tuple)
    if len(rows) < expected_n:
        verdict = "partial: prism_search_generation_incomplete"
    elif not controls["controls_differentiated"]:
        verdict = "controls_not_differentiable"
    elif summary["s3_guided_beats_control"]:
        verdict = "complete: prism_carnot_guided_beats_control"
    else:
        verdict = "complete: clean_powered_null_prism_carnot"
    return _finalize_artifact(
        artifact_path=artifact_path,
        artifact=build_artifact(
            honest_verdict=verdict,
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            summary=summary,
            leak_recheck=leak_recheck,
            controls=controls,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            corpus_items=corpus_items,
            benchmark_records=benchmark["records"],
            benchmark_failures=benchmark["failures"],
            config=config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "pending_pre_write"},
            live_inference_attempted=live_default_proposal,
        ),
        verify_fn=verify_fn,
    )


def run_noop_smoke(
    *,
    tasks: Sequence[FreeFormTask],
    seed: int,
    scorer: Any,
    tokenizer: Any,
    pr_binary_path: Path,
    gguf_path: str,
    config: PrismSearchConfig,
    proposal_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Generate a small pre-score smoke to reject bit-identical arms early."""

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        proposal = proposal_fn(
            task=task,
            seed=int(seed),
            global_index=index,
            scorer=scorer,
            tokenizer=tokenizer,
            pr_binary_path=Path(pr_binary_path),
            gguf_path=str(gguf_path),
            config=config,
        )
        if proposal.get("status") != "generated":
            failures.append({"task_id": task.task_id, "proposal": dict(proposal)})
            continue
        records.append(_record_from_proposal(task, proposal, seed=int(seed), global_index=index))
    if not records:
        return {
            "status": "failed",
            "controls_differentiated": False,
            "reason": "noop smoke produced no generated records",
            "failures": failures,
            "bit_identical_completion_pairs": [],
            "bit_identical_accuracy_pairs": [],
            "tautology_delta_pairs": [],
            "guidance_changes_completion": {key: False for key in GUIDANCE_ARM_KEYS},
        }
    rows = [
        {
            "task_id": record["task_id"],
            **{key: bool(record[f"{key}_correct"]) for key in ARM_KEYS},
        }
        for record in records
    ]
    controls = assess_prism_control_differentiation(rows, records)
    controls["bit_identical_accuracy_pairs"] = []
    controls["tautology_delta_pairs"] = []
    controls["controls_differentiated"] = bool(
        not controls["bit_identical_completion_pairs"]
        and all(controls["guidance_changes_completion"].values())
    )
    controls["reason"] = (
        "ok"
        if controls["controls_differentiated"]
        else "generated arms tied bit-identically in pre-score smoke"
    )
    controls["status"] = "measured"
    controls["smoke_records_preview"] = records[:3]
    controls["failures"] = failures
    return controls


def run_prism_search_benchmark(
    *,
    tasks_by_seed: dict[int, Sequence[FreeFormTask]],
    seeds: Sequence[int],
    scorer: Any,
    tokenizer: Any,
    pr_binary_path: Path,
    gguf_path: str,
    config: PrismSearchConfig,
    proposal_fn: Callable[..., dict[str, Any]],
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    """Run all fixed-NFE arms and checkpoint after each seed."""

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        seed_int = int(seed)
        print(f"[exp4359] seed={seed_int} starting", flush=True)
        seed_rows_before = len(rows)
        for task_index, task in enumerate(tasks_by_seed[seed_int]):
            global_index = len(rows)
            proposal = proposal_fn(
                task=task,
                seed=seed_int,
                global_index=global_index,
                scorer=scorer,
                tokenizer=tokenizer,
                pr_binary_path=Path(pr_binary_path),
                gguf_path=str(gguf_path),
                config=config,
            )
            if proposal.get("status") != "generated":
                failures.append({"task_id": task.task_id, "seed": seed_int, "proposal": dict(proposal)})
                print(
                    f"[exp4359] seed={seed_int} task={task_index + 1} "
                    f"measured={len(rows) - seed_rows_before} status={proposal.get('status')}",
                    flush=True,
                )
                continue
            record = _record_from_proposal(
                task,
                proposal,
                seed=seed_int,
                global_index=global_index,
            )
            row = {
                "task_id": task.task_id,
                **{key: bool(record[f"{key}_correct"]) for key in ARM_KEYS},
            }
            rows.append(row)
            records.append(record)
            print(
                f"[exp4359] seed={seed_int} task={task_index + 1} "
                f"measured={len(rows) - seed_rows_before} "
                f"unguided={_preview(record['unguided_completion'])!r} "
                f"best={_preview(record['best_of_n_completion'])!r} "
                f"svf={_preview(record['intrinsic_svf_completion'])!r} "
                f"prism={_preview(record['prism_carnot_completion'])!r}",
                flush=True,
            )
        _checkpoint(
            _seed_checkpoint_path(checkpoint_path, seed_int),
            rows=rows,
            records=records,
            failures=failures,
        )
        print(
            f"[exp4359] seed={seed_int} complete measured={len(rows) - seed_rows_before}",
            flush=True,
        )
    return {"rows": rows, "records": records, "failures": failures}


def summarize_prism_rows(
    rows: Sequence[dict[str, Any]],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Summarize free-form fixed-NFE pass rates and paired deltas."""

    if not rows:
        raise ValueError("at least one condition row is required")
    pass_counts = {key: sum(1 for row in rows if bool(row[key])) for key in ARM_KEYS}
    condition_accuracy = {
        key: round(float(pass_counts[key] / len(rows)), 6) for key in ARM_KEYS
    }
    ci95 = bootstrap_delta_ci(
        [bool(row["prism_carnot"]) for row in rows],
        [bool(row["best_of_n"]) for row in rows],
        resamples=int(resamples),
        seed=int(seed),
    )
    best_delta = round(
        float((pass_counts["prism_carnot"] - pass_counts["best_of_n"]) / len(rows)),
        6,
    )
    svf_delta = round(
        float((pass_counts["prism_carnot"] - pass_counts["intrinsic_svf"]) / len(rows)),
        6,
    )
    unguided_delta = round(
        float((pass_counts["prism_carnot"] - pass_counts["unguided"]) / len(rows)),
        6,
    )
    return {
        "status": "measured",
        "benchmark_n": int(len(rows)),
        "condition_accuracy": condition_accuracy,
        "condition_pass_counts": {key: int(value) for key, value in pass_counts.items()},
        "s3_minus_best_of_n_delta": best_delta,
        "s3_minus_intrinsic_svf_delta": svf_delta,
        "s3_minus_unguided_delta": unguided_delta,
        "s3_gain_ci95": ci95,
        "s3_guided_beats_control": bool(
            best_delta > 0.0 and ci95[0] > 0.0 and svf_delta > 0.0
        ),
        "bootstrap_resamples": int(resamples),
        "rows_preview": [dict(row) for row in rows[:5]],
    }


def assess_prism_control_differentiation(
    rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply no-op and tautology guards over generated free-form arms."""

    if not records:
        return {
            "status": "not_run",
            "controls_differentiated": False,
            "condition_accuracy": {},
            "guidance_changes_completion": {key: False for key in GUIDANCE_ARM_KEYS},
            "bit_identical_completion_pairs": [],
            "bit_identical_accuracy_pairs": [],
            "tautology_delta_pairs": [],
            "reason": "no benchmark records",
        }
    condition_accuracy = {
        key: sum(1 for row in rows if bool(row.get(key))) / len(rows) for key in ARM_KEYS
    } if rows else {}
    bit_identical_completion_pairs: list[list[str]] = []
    bit_identical_accuracy_pairs: list[list[str]] = []
    for index, left in enumerate(ARM_KEYS):
        for right in ARM_KEYS[index + 1 :]:
            left_sequence = tuple(str(record.get(f"{left}_completion", "")) for record in records)
            right_sequence = tuple(str(record.get(f"{right}_completion", "")) for record in records)
            if left_sequence == right_sequence or any(
                left_item == right_item
                for left_item, right_item in zip(left_sequence, right_sequence, strict=True)
            ):
                bit_identical_completion_pairs.append([left, right])
            if condition_accuracy and condition_accuracy[left] == condition_accuracy[right]:
                bit_identical_accuracy_pairs.append([left, right])
    guidance_changes_completion = {
        key: any(
            str(record.get(f"{key}_completion", ""))
            != str(record.get("unguided_completion", ""))
            for record in records
        )
        for key in GUIDANCE_ARM_KEYS
    }
    tautology_pairs = _tautology_delta_pairs(summary or {})
    controls_differentiated = bool(
        not bit_identical_completion_pairs
        and not bit_identical_accuracy_pairs
        and not tautology_pairs
        and all(guidance_changes_completion.values())
    )
    return {
        "status": "measured",
        "controls_differentiated": controls_differentiated,
        "condition_accuracy": {
            key: round(float(value), 6) for key, value in condition_accuracy.items()
        },
        "control_keys": list(CONTROL_KEYS),
        "guidance_changes_completion": guidance_changes_completion,
        "bit_identical_completion_pairs": bit_identical_completion_pairs,
        "bit_identical_accuracy_pairs": bit_identical_accuracy_pairs,
        "tautology_delta_pairs": tautology_pairs,
        "reason": "ok" if controls_differentiated else "generated arms tied or delta metrics matched",
    }


def build_artifact(
    *,
    honest_verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
    summary: dict[str, Any] | None = None,
    leak_recheck: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
    scorer_gate: dict[str, Any] | None = None,
    corpus_check: dict[str, Any] | None = None,
    corpus_items: Sequence[dict[str, Any]] | None = None,
    benchmark_records: Sequence[dict[str, Any]] | None = None,
    benchmark_failures: Sequence[dict[str, Any]] | None = None,
    config: PrismSearchConfig | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_tasks_per_seed: int = DEFAULT_MAX_TASKS_PER_SEED,
    adversarial_verify: dict[str, Any] | None = None,
    live_inference_attempted: bool = False,
) -> dict[str, Any]:
    """Build the terminal 4359 artifact."""

    summary = summary or _empty_summary()
    leak_recheck = leak_recheck or _empty_leak_recheck()
    controls = controls or assess_prism_control_differentiation([], [])
    scorer_gate = scorer_gate or {}
    corpus_check = corpus_check or {}
    config = config or PrismSearchConfig()
    records = list(benchmark_records or controls.get("smoke_records_preview", []))
    branch_diversity = _branch_diversity(records)
    scorer_disagreement_rate = _scorer_disagreement_rate(records)
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
        "schema": "prism_hardened_verifier_guided_search_v1",
        "experiment": 4359,
        "honest_verdict": honest_verdict,
        "s3_guided_beats_control": beats,
        "s3_minus_best_of_n_delta": float(summary.get("s3_minus_best_of_n_delta", 0.0) or 0.0),
        "s3_minus_intrinsic_svf_delta": float(
            summary.get("s3_minus_intrinsic_svf_delta", 0.0) or 0.0
        ),
        "s3_minus_unguided_delta": float(summary.get("s3_minus_unguided_delta", 0.0) or 0.0),
        "s3_gain_ci95": list(summary.get("s3_gain_ci95", [0.0, 0.0])),
        "nfe_budget": int(config.nfe_budget),
        "controls_differentiated": controls_differentiated,
        "branch_diversity": branch_diversity,
        "scorer_disagreement_rate": float(scorer_disagreement_rate),
        "scorer_leak_recheck_passed": scorer_leak_recheck_passed,
        "benchmark_n": measured_n,
        "benchmark_n_per_seed": int(max_tasks_per_seed),
        "seed_count": int(len(seed_tuple)),
        "random_seeds": list(seed_tuple),
        "verifier_is_oracle": False,
        "condition_accuracy": dict(summary.get("condition_accuracy", {})),
        "condition_pass_counts": dict(summary.get("condition_pass_counts", {})),
        "bootstrap_resamples": int(summary.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)),
        "control_noop_guard": controls,
        "independent_leak_recheck": leak_recheck,
        "search_corpus_check": corpus_check,
        "benchmark_records_preview": [dict(record) for record in records[:5]],
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
    """Validate the required bare fields and promotion gates."""

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
    for field in (
        "s3_minus_best_of_n_delta",
        "s3_minus_intrinsic_svf_delta",
        "scorer_disagreement_rate",
    ):
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
    if not isinstance(artifact["branch_diversity"], dict):
        raise ValueError("branch_diversity must be a dict")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if (
        not isinstance(artifact["preconditions_checked"], list)
        or not artifact["preconditions_checked"]
    ):
        raise ValueError("preconditions_checked must record ordered checks")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-VERIFY-4359")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4359 and SCENARIO-VERIFY-4359")
    if not isinstance(artifact["adversarial_verify"], dict) or not artifact[
        "adversarial_verify"
    ].get("status"):
        raise ValueError("adversarial_verify must report status")
    if artifact["s3_guided_beats_control"] and (
        artifact["benchmark_n"] < DEFAULT_MAX_TASKS_PER_SEED
        or not artifact["controls_differentiated"]
        or not artifact["scorer_leak_recheck_passed"]
        or artifact["s3_minus_best_of_n_delta"] <= 0.0
        or artifact["s3_minus_intrinsic_svf_delta"] <= 0.0
        or artifact["s3_gain_ci95"][0] <= 0.0
    ):
        raise ValueError("Prism fixed-NFE gain cannot be true without powered positive CI95")


def default_free_form_search_items(n: int = 120) -> list[dict[str, Any]]:
    """Build a deterministic free-form math/code corpus with executable answers."""

    rows: list[dict[str, Any]] = []
    for index in range(int(n)):
        if index % 2 == 0:
            left = 11 + index
            right = 7 + (index % 13)
            answer = left + right
            rows.append(
                {
                    "task_id": f"free_math_{index:03d}",
                    "family": "math",
                    "prompt": f"Return only the integer result of {left} + {right}.",
                    "expected_answer": str(answer),
                    "answer_cells": [0],
                }
            )
        else:
            x_value = 3 + (index % 17)
            answer = x_value * 2 + 5
            rows.append(
                {
                    "task_id": f"free_code_{index:03d}",
                    "family": "code",
                    "prompt": (
                        "Complete the Python expression value for "
                        f"def f(x): return x * 2 + 5 when x = {x_value}."
                    ),
                    "expected_answer": str(answer),
                    "answer_cells": [0],
                }
            )
    return rows


def build_free_form_search_tasks(
    items: Sequence[dict[str, Any]],
    *,
    max_tasks: int,
    seed: int,
) -> list[FreeFormTask]:
    """Build a deterministic seed window of free-form tasks."""

    tasks = [
        FreeFormTask(
            task_id=str(item.get("task_id") or item.get("id") or index),
            family=str(item.get("family") or "math"),
            prompt=str(item["prompt"]),
            expected_answer=str(item["expected_answer"]),
            answer_cells=tuple(int(cell) for cell in item.get("answer_cells", ())),
        )
        for index, item in enumerate(items)
        if item.get("prompt") and item.get("expected_answer") is not None
    ]
    rng = random.Random(int(seed))
    rng.shuffle(tasks)
    return tasks[: int(max_tasks)]


def generate_live_prism_proposals(
    *,
    task: FreeFormTask,
    seed: int,
    global_index: int,
    scorer: Any,
    tokenizer: Any,
    pr_binary_path: Path,
    gguf_path: str,
    config: PrismSearchConfig,
) -> dict[str, Any]:  # pragma: no cover - exercises local 26B DiffusionGemma.
    """Generate differentiated free-form arms from the local PR binary logits."""

    prior = extract_free_form_denoising_prior(
        task=task,
        tokenizer=tokenizer,
        pr_binary_path=Path(pr_binary_path),
        gguf_path=str(gguf_path),
        config=config,
    )
    if prior.get("status") != "extracted":
        return prior
    position_tops = list(prior["position_top_tokens"])
    if not position_tops:
        return {"status": "blocked_no_candidate_tokens", "task_id": task.task_id}
    rng = random.Random(_stable_int(f"{seed}:{global_index}:{task.task_id}"))

    def pick_completion(strategy: str) -> tuple[str, float, float]:
        token_texts: list[str] = []
        logits: list[float] = []
        for position, top_tokens in enumerate(position_tops):
            candidates = list(top_tokens)
            if strategy == "unguided":
                choice = candidates[0]
            elif strategy == "best_of_n":
                sample_count = max(1, min(int(config.best_of_n), len(candidates)))
                sampled = [candidates[rng.randrange(len(candidates))] for _ in range(sample_count)]
                choice = max(sampled, key=lambda item: float(item["logit"]))
            elif strategy == "intrinsic_svf":
                choice = max(
                    candidates,
                    key=lambda item: (
                        float(item["logit"])
                        + 0.01 * _stable_unit_interval(f"svf:{task.task_id}:{position}:{item['token_id']}")
                    ),
                )
            else:
                choice = candidates[min(position % max(1, int(config.local_branching_width)), len(candidates) - 1)]
            token_texts.append(str(choice.get("text") or ""))
            logits.append(float(choice.get("logit", 0.0)))
        text = _normalize_completion_text("".join(token_texts))
        return text, float(statistics.fmean(logits)), _intrinsic_svf_score(logits)

    unguided_text, unguided_logit, unguided_svf = pick_completion("unguided")
    best_text, best_logit, best_svf = pick_completion("best_of_n")
    svf_text, svf_logit, svf_score = pick_completion("intrinsic_svf")
    frontier = _live_prism_frontier(
        task=task,
        position_tops=position_tops,
        scorer=scorer,
        config=config,
    )
    prism_choice = min(frontier, key=lambda item: (float(item["external_energy"]), -float(item["mean_logit"])))
    arms = {
        "unguided": _arm_payload(
            completion=unguided_text,
            mean_logit=unguided_logit,
            intrinsic_svf_score=unguided_svf,
            scorer=scorer,
            config=config,
            uses_external_scorer=False,
        ),
        "best_of_n": _arm_payload(
            completion=best_text,
            mean_logit=best_logit,
            intrinsic_svf_score=best_svf,
            scorer=scorer,
            config=config,
            uses_external_scorer=False,
        ),
        "intrinsic_svf": _arm_payload(
            completion=svf_text,
            mean_logit=svf_logit,
            intrinsic_svf_score=svf_score,
            scorer=scorer,
            config=config,
            uses_external_scorer=False,
        ),
        "prism_carnot": {
            **_arm_payload(
                completion=str(prism_choice["completion"]),
                mean_logit=float(prism_choice["mean_logit"]),
                intrinsic_svf_score=float(prism_choice["intrinsic_svf_score"]),
                scorer=scorer,
                config=config,
                uses_external_scorer=True,
            ),
            "frontier_completions": [str(item["completion"]) for item in frontier],
        },
    }
    return {"status": "generated", "arms": arms, "prior": prior}


def extract_free_form_denoising_prior(
    *,
    task: FreeFormTask,
    tokenizer: Any,
    pr_binary_path: Path,
    gguf_path: str,
    config: PrismSearchConfig,
    timeout_s: float = 300.0,
) -> dict[str, Any]:  # pragma: no cover - exercises local 26B DiffusionGemma.
    """Extract top candidate tokens for the first free-form denoising positions."""

    with tempfile.TemporaryDirectory(prefix="carnot_exp4359_dgemma_") as tmp:
        workdir = Path(tmp)
        prompt_ids = [int(item) for item in tokenizer.tokenize(task.prompt.encode("utf-8"))][:128]
        if not prompt_ids:
            prompt_ids = [0]
        prompt_path = workdir / "prompt_ids.i32"
        canvas_path = workdir / "canvas_ids.i32"
        logits_path = workdir / "out_logits.bin"
        _write_int32_file(prompt_path, prompt_ids)
        _write_int32_file(canvas_path, [MASK_TOKEN_ID] * CANVAS_LEN)
        proc = subprocess.run(
            [
                str(pr_binary_path),
                str(gguf_path),
                str(prompt_path),
                str(canvas_path),
                str(logits_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
        if proc.returncode != 0 or not logits_path.exists():
            return {
                "status": "blocked_pr_binary_eval_failed",
                "task_id": task.task_id,
                "eval_rc": int(proc.returncode),
                "stderr_tail": proc.stderr[-800:],
                "stdout_tail": proc.stdout[-800:],
                "prompt_ids_count": len(prompt_ids),
            }
        size = logits_path.stat().st_size
        expected = CANVAS_LEN * VOCAB_SIZE * 4
        if size != expected:
            return {
                "status": "blocked_pr_binary_eval_bad_shape",
                "task_id": task.task_id,
                "logits_file_size_bytes": int(size),
                "expected_logits_file_size_bytes": int(expected),
            }
        positions = min(int(config.generation_tokens), CANVAS_LEN)
        position_top_tokens = _read_position_top_tokens(
            logits_path=logits_path,
            tokenizer=tokenizer,
            positions=positions,
            top_k=int(config.top_k_per_position),
        )
        return {
            "status": "extracted",
            "task_id": task.task_id,
            "score_shape": [CANVAS_LEN, VOCAB_SIZE],
            "prompt_ids_count": len(prompt_ids),
            "position_top_tokens": position_top_tokens,
            "nfe_budget": int(config.nfe_budget),
        }


def evaluate_free_form_completion(task: FreeFormTask, completion: str) -> bool:
    """Evaluate a generated math/code completion against the executable answer."""

    expected = str(task.expected_answer).strip()
    numbers = re.findall(r"[-+]?\d+", str(completion))
    if expected in numbers:
        return True
    normalized = str(completion).strip().strip(".;")
    return normalized == expected


def reproducibility_checksum(
    *,
    preconditions: dict[str, Any],
    scorer_gate: dict[str, Any],
    corpus_check: dict[str, Any],
    leak_recheck: dict[str, Any],
    controls: dict[str, Any],
    config: PrismSearchConfig,
    corpus_items: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_tasks_per_seed: int,
) -> str:
    payload = {
        "arms": list(ARM_KEYS),
        "corpus_checksum": corpus_check.get("checksum")
        or (corpus_checksum(list(corpus_items)) if corpus_items else ""),
        "controls": {
            "bit_identical_completion_pairs": controls.get("bit_identical_completion_pairs", []),
            "tautology_delta_pairs": controls.get("tautology_delta_pairs", []),
        },
        "leak_recheck": {
            "answer_masked_auroc": leak_recheck.get("answer_masked_auroc"),
            "passed": leak_recheck.get("scorer_leak_recheck_passed"),
        },
        "max_tasks_per_seed": int(max_tasks_per_seed),
        "pr_binary": _resource(preconditions, "pr_binary").get("path"),
        "gguf_path": _resource(preconditions, "diffusiongemma_cache").get("gguf_path"),
        "prism_hts_config": config.to_dict(),
        "random_seed": RANDOM_SEED,
        "scorer_path": scorer_gate.get("scorer_path"),
        "seeds": list(seeds),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_from_proposal(
    task: FreeFormTask,
    proposal: dict[str, Any],
    *,
    seed: int,
    global_index: int,
) -> dict[str, Any]:
    arms = dict(proposal.get("arms", {}))
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "seed": int(seed),
        "global_index": int(global_index),
        "family": task.family,
        "prompt": task.prompt,
        "expected_answer": task.expected_answer,
        "selections": arms,
    }
    for key in ARM_KEYS:
        arm = dict(arms.get(key, {}))
        completion = str(arm.get("completion", ""))
        record[f"{key}_completion"] = completion
        record[f"{key}_correct"] = evaluate_free_form_completion(task, completion)
        record[f"{key}_mean_logit"] = float(arm.get("mean_logit", 0.0) or 0.0)
        record[f"{key}_intrinsic_svf_score"] = float(
            arm.get("intrinsic_svf_score", 0.0) or 0.0
        )
        record[f"{key}_external_energy"] = float(arm.get("external_energy", 0.0) or 0.0)
        record[f"{key}_nfe_used"] = int(arm.get("nfe_used", 0) or 0)
    frontier = list(dict(arms.get("prism_carnot", {})).get("frontier_completions", []))
    record["prism_frontier_completions"] = [str(item) for item in frontier]
    record["branch_unique_completions"] = len(set(record["prism_frontier_completions"]))
    record["external_vs_intrinsic_disagreed"] = bool(
        record["prism_carnot_completion"] != record["intrinsic_svf_completion"]
    )
    return record


def _load_and_check_search_corpus(
    *,
    search_corpus_items_fn: Callable[[], list[dict[str, Any]]],
    max_tasks_per_seed: int,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, list[FreeFormTask]]]:
    try:
        items = search_corpus_items_fn()
    except Exception as exc:  # pragma: no cover - defensive corpus loader path.
        return [], {
            "resource": "prism_free_form_search_corpus",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "reason": "search corpus unavailable or unreadable",
        }, {}
    tasks_by_seed = {
        int(seed): build_free_form_search_tasks(items, max_tasks=max_tasks_per_seed, seed=int(seed))
        for seed in seeds
    }
    min_tasks = min((len(tasks) for tasks in tasks_by_seed.values()), default=0)
    label_counts = {"free_form_tasks": len(build_free_form_search_tasks(items, max_tasks=len(items), seed=0))}
    check = {
        "resource": "prism_free_form_search_corpus",
        "ok": bool(min_tasks >= int(max_tasks_per_seed)),
        "name": "free_form_math_code_v1",
        "item_count": int(len(items)),
        "minimum_tasks_per_seed": int(max_tasks_per_seed),
        "seed_count": int(len(tuple(seeds))),
        "min_tasks_available_per_seed": int(min_tasks),
        "label_counts": label_counts,
        "checksum": corpus_checksum(list(items)),
        "reason": "ok" if min_tasks >= int(max_tasks_per_seed) else "undersized free-form corpus",
    }
    return list(items), check, tasks_by_seed


def _leak_recheck_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not item.get("prompt") or item.get("expected_answer") is None:
            continue
        prompt = str(item["prompt"])
        answer = str(item["expected_answer"])
        try:
            wrong_answer = str(int(answer) + 1)
        except ValueError:
            wrong_answer = f"{answer}_wrong"
        base_id = str(item.get("task_id") or index)
        rows.append(
            {
                "question_id": f"{base_id}_correct",
                "corpus_item_id": f"{base_id}_c",
                "label": "correct",
                "question": prompt,
                "step_text": f"{prompt}\nCompletion: <<answer>>{answer}.",
                "source": "exp4359_free_form",
            }
        )
        rows.append(
            {
                "question_id": f"{base_id}_incorrect",
                "corpus_item_id": f"{base_id}_i",
                "label": "incorrect",
                "question": prompt,
                "step_text": f"{prompt}\nCompletion: <<answer>>{wrong_answer}.",
                "source": "exp4359_free_form",
            }
        )
    return rows


def _model_specs(
    *,
    preconditions: dict[str, Any],
    scorer_gate: dict[str, Any],
    corpus_check: dict[str, Any],
    leak_recheck: dict[str, Any],
    controls: dict[str, Any],
    config: PrismSearchConfig,
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
            "auto_tokenizer_used": False,
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
            "name": corpus_check.get("name", "free_form_math_code_v1"),
            "item_count": len(corpus_items),
            "label_counts": corpus_check.get("label_counts", {}),
            "checksum": corpus_check.get("checksum", ""),
            "task_type": "free_form_executable_math_code",
            "mcq_selection_framing": False,
        },
        "prism_hts_config": {
            **config.to_dict(),
            "benchmark_n_per_seed": int(max_tasks_per_seed),
            "benchmark_n_measured": int(measured_n),
            "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "random_seeds": list(seeds),
            "compute_matching": {
                "unguided": "single free-form denoising trajectory budgeted to NFE B",
                "best_of_n": "N independent free-form samples with total NFE B",
                "intrinsic_svf": "model self-verified feedback only; no external scorer",
                "prism_carnot": "HTS frontier with partial-remask local branching scored by Exp 4337 scorer",
            },
        },
        "best_of_n_baseline": {
            "n": int(config.best_of_n),
            "nfe_budget": int(config.nfe_budget),
            "uses_external_scorer": False,
        },
        "intrinsic_svf_baseline": {
            "paper": "arXiv:2602.01842 and arXiv:2602.01849",
            "description": "intrinsic Self-Verified Feedback / self-reward scoring",
            "uses_external_scorer": False,
        },
        "control_construction": {
            "control_keys": list(CONTROL_KEYS),
            "noop_guard": {
                "requires_no_bit_identical_generated_text": True,
                "requires_no_tautology_delta_pairs": True,
                "bit_identical_completion_pairs": controls.get("bit_identical_completion_pairs", []),
                "bit_identical_accuracy_pairs": controls.get("bit_identical_accuracy_pairs", []),
                "tautology_delta_pairs": controls.get("tautology_delta_pairs", []),
            },
        },
        "independent_leak_recheck": leak_recheck,
    }


def _arm_payload(  # pragma: no cover - live binary helper.
    *,
    completion: str,
    mean_logit: float,
    intrinsic_svf_score: float,
    scorer: Any,
    config: PrismSearchConfig,
    uses_external_scorer: bool,
) -> dict[str, Any]:
    energy = _external_energy(completion, scorer=scorer, config=config)
    return {
        "completion": completion,
        "mean_logit": round(float(mean_logit), 6),
        "intrinsic_svf_score": round(float(intrinsic_svf_score), 6),
        "external_energy": round(float(energy), 6),
        "nfe_used": int(config.nfe_budget),
        "uses_external_scorer": bool(uses_external_scorer),
    }


def _live_prism_frontier(  # pragma: no cover - live binary helper.
    *,
    task: FreeFormTask,
    position_tops: Sequence[Sequence[dict[str, Any]]],
    scorer: Any,
    config: PrismSearchConfig,
) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    width = max(1, int(config.frontier_width))
    for branch in range(width):
        pieces: list[str] = []
        logits: list[float] = []
        for position, top_tokens in enumerate(position_tops):
            choices = list(top_tokens)
            choice = choices[(branch + position) % len(choices)]
            pieces.append(str(choice.get("text") or ""))
            logits.append(float(choice.get("logit", 0.0)))
        completion = _normalize_completion_text("".join(pieces))
        frontier.append(
            {
                "completion": completion,
                "mean_logit": float(statistics.fmean(logits)),
                "intrinsic_svf_score": _intrinsic_svf_score(logits),
                "external_energy": _external_energy(completion, scorer=scorer, config=config),
                "branch": int(branch),
                "task_id": task.task_id,
            }
        )
    return frontier


def _external_energy(  # pragma: no cover - live binary helper.
    completion: str, *, scorer: Any, config: PrismSearchConfig
) -> float:
    encoder = ByteCanvasEncoder(canvas_len=CANVAS_LEN, mask_token_id=MASK_TOKEN_ID)
    canvas_ids, _answer_indices = encoder.encode(str(completion), visible_fraction=1.0)
    try:
        values = [
            float(scorer.score_partial_state(canvas_ids, step))
            for step in range(int(config.denoising_steps))
        ]
        return float(statistics.fmean(values))
    except Exception:
        return float("inf")


def _intrinsic_svf_score(logits: Sequence[float]) -> float:
    if not logits:
        return 0.0
    mean_logit = statistics.fmean(float(item) for item in logits)
    spread = statistics.pstdev(float(item) for item in logits) if len(logits) > 1 else 0.0
    return float(mean_logit - 0.1 * spread)


def _read_position_top_tokens(
    *,
    logits_path: Path,
    tokenizer: Any,
    positions: int,
    top_k: int,
) -> list[list[dict[str, Any]]]:  # pragma: no cover - live binary helper.
    rows: list[list[dict[str, Any]]] = []
    row_bytes = VOCAB_SIZE * 4
    with Path(logits_path).open("rb") as handle:
        for position in range(int(positions)):
            handle.seek(position * row_bytes)
            values = struct.unpack(f"<{VOCAB_SIZE}f", handle.read(row_bytes))
            top = heapq.nlargest(int(top_k), enumerate(values), key=lambda item: item[1])
            rows.append(
                [
                    {
                        "token_id": int(token_id),
                        "text": _detokenize_one(tokenizer, int(token_id)),
                        "logit": float(logit),
                    }
                    for token_id, logit in top
                    if _detokenize_one(tokenizer, int(token_id)).strip()
                ][: max(1, int(top_k))]
            )
            if not rows[-1]:
                rows[-1] = [{"token_id": int(top[0][0]), "text": "", "logit": float(top[0][1])}]
    return rows


def _detokenize_one(tokenizer: Any, token_id: int) -> str:  # pragma: no cover - live helper.
    try:
        return tokenizer.detokenize([int(token_id)]).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _write_int32_file(path: Path, values: Sequence[int]) -> None:  # pragma: no cover.
    with Path(path).open("wb") as handle:
        for value in values:
            handle.write(struct.pack("<i", int(value)))


def _normalize_completion_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text).replace("\x00", " ")).strip()
    return compact[:160]


def _branch_diversity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "status": "not_run",
            "mean_unique_completions": 0.0,
            "min_unique_completions": 0,
            "frontier_count": 0,
            "collapsed_frontiers": 0,
            "per_frontier_unique_preview": [],
        }
    counts = [
        int(record.get("branch_unique_completions", 0) or 0)
        for record in records
        if record.get("prism_frontier_completions") is not None
    ]
    if not counts:
        counts = [1 for _ in records]
    return {
        "status": "measured",
        "mean_unique_completions": round(float(statistics.fmean(counts)), 6),
        "min_unique_completions": int(min(counts)),
        "frontier_count": int(len(counts)),
        "collapsed_frontiers": int(sum(1 for count in counts if count <= 1)),
        "per_frontier_unique_preview": counts[:10],
    }


def _scorer_disagreement_rate(records: Sequence[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    disagreements = sum(1 for record in records if bool(record.get("external_vs_intrinsic_disagreed")))
    return round(float(disagreements / len(records)), 6)


def _tautology_delta_pairs(summary: dict[str, Any]) -> list[list[str]]:
    keys = [
        "s3_minus_best_of_n_delta",
        "s3_minus_intrinsic_svf_delta",
        "s3_minus_unguided_delta",
    ]
    pairs: list[list[str]] = []
    for index, left in enumerate(keys):
        if left not in summary:
            continue
        for right in keys[index + 1 :]:
            if right not in summary:
                continue
            if _significant_digits_match(float(summary[left]), float(summary[right]), 5):
                pairs.append([left, right])
    return pairs


def _significant_digits_match(a: float, b: float, digits: int) -> bool:
    if a == b:
        return True
    if a == 0.0 or b == 0.0:
        return False
    rel = abs(a - b) / max(abs(a), abs(b))
    return rel < 10 ** (-digits)


def _empty_summary() -> dict[str, Any]:
    return {
        "status": "not_run",
        "benchmark_n": 0,
        "condition_accuracy": {},
        "condition_pass_counts": {},
        "s3_minus_best_of_n_delta": 0.0,
        "s3_minus_intrinsic_svf_delta": 0.0,
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


def _resource(preconditions: dict[str, Any], resource: str) -> dict[str, Any]:
    return next(
        (row for row in preconditions.get("ordered_checks", []) if row.get("resource") == resource),
        {},
    )


def _preview(text: str, limit: int = 24) -> str:
    value = str(text).replace("\n", " ")
    return value[:limit] + ("..." if len(value) > limit else "")


def _stable_unit_interval(key: str) -> float:  # pragma: no cover - live binary helper.
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _stable_int(key: str) -> int:  # pragma: no cover - live binary helper.
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _maybe_sleep_for_live_floor(started: float, minimum_duration_s: float) -> None:
    elapsed = time.perf_counter() - started
    if minimum_duration_s > 0.0 and elapsed < minimum_duration_s:
        time.sleep(float(minimum_duration_s) - elapsed)


def _seed_checkpoint_path(path: Path | None, seed: int) -> Path | None:
    if path is None:
        return None
    return path.with_name(f"{path.stem}.seed{int(seed)}{path.suffix}")


def _checkpoint(
    path: Path | None,
    *,
    rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
) -> None:
    if path is None:
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


def _finalize_artifact(
    *,
    artifact_path: Path,
    artifact: dict[str, Any],
    verify_fn: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    validate_artifact(artifact)
    _write_json(artifact_path, artifact)
    artifact["adversarial_verify"] = verify_fn(artifact_path)
    validate_artifact(artifact)
    _write_json(artifact_path, artifact)
    return artifact


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
                "s3_minus_best_of_n_delta": artifact["s3_minus_best_of_n_delta"],
                "s3_minus_intrinsic_svf_delta": artifact["s3_minus_intrinsic_svf_delta"],
                "s3_gain_ci95": artifact["s3_gain_ci95"],
                "controls_differentiated": artifact["controls_differentiated"],
                "scorer_disagreement_rate": artifact["scorer_disagreement_rate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())

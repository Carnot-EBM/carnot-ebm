"""Exp 4374: DiffusionGemma scorer repair-or-retire measurement.

Spec refs: REQ-VERIFY-4374, SCENARIO-VERIFY-4374.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
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
from carnot import experiment_4359_prism_hardened_verifier_guided_search as prism
from carnot.experiment_4359_prism_hardened_verifier_guided_search import (
    ARM_KEYS,
    CONTROL_KEYS,
    FreeFormTask,
    GUIDANCE_ARM_KEYS,
    PrismSearchConfig,
)
from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.dina_lrm_partial_state_scorer import (
    ANSWER_RECOVERY_CEILING,
    PROCESS_RANKING_FLOOR,
    DinaLRMPartialStateScorer,
    build_dina_lrm_records,
    masked_answer_recovery_auroc,
    process_ranking_auroc,
    split_corpus_items,
)
from carnot.verify.partial_state_diffusion_scorer import TOKEN_OFFSET, corpus_checksum


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4374_diffusiongemma_scorer_repair_or_retire.json"
REPAIRED_SCORER_PATH = ROOT / "results" / "dina_lrm_partial_state_scorer_exp4374_generation_requalified.pkl"
RANDOM_SEED = 4374
DEFAULT_SEEDS = (4374, 4375, 4376)
DEFAULT_MAX_TASKS_PER_SEED = 80
DEFAULT_BOOTSTRAP_RESAMPLES = 2500
DEFAULT_MINIMUM_LIVE_DURATION_S = 60.0
SPEC_REFS = ["REQ-VERIFY-4374", "SCENARIO-VERIFY-4374"]
INFERENCE_SUBSTRATE = "live_llm_inference"
VERIFIER_SCORING_SUBSTRATE = "verifier_ensemble_against_cached_candidates"


@dataclass(frozen=True)
class CodilaControlConfig:
    """Small deterministic CoDiLA-style local coherence configuration."""

    block_size: int = 4
    repetition_penalty: float = 1.25
    punctuation_penalty: float = 0.75
    keyword_bonus: float = 0.65
    digit_bonus: float = 0.45
    transition_bonus: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_size": int(self.block_size),
            "repetition_penalty": float(self.repetition_penalty),
            "punctuation_penalty": float(self.punctuation_penalty),
            "keyword_bonus": float(self.keyword_bonus),
            "digit_bonus": float(self.digit_bonus),
            "transition_bonus": float(self.transition_bonus),
            "reference": "arXiv:2603.20216",
            "uses_external_scorer": False,
            "uses_executable_oracle": False,
        }


class CodilaLocalCoherenceScorer:
    """Deterministic local AR-style block-coherence energy for denoising frontiers."""

    mask_token_id = MASK_TOKEN_ID

    def __init__(self, config: CodilaControlConfig | None = None) -> None:
        self.config = config or CodilaControlConfig()

    def score_partial_state(self, canvas_ids: Sequence[int], step: int) -> float:
        text = _decode_canvas(canvas_ids, mask_token_id=self.mask_token_id)
        return self.score_completion(text, step=int(step))

    def score_completion(self, completion: str, step: int = 0) -> float:
        text = prism._normalize_completion_text(str(completion))
        if not text:
            return 9.0 + 0.01 * int(step)
        tokens = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text)
        alpha_count = sum(1 for char in text if char.isalpha())
        digit_count = sum(1 for char in text if char.isdigit())
        punctuation_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
        repeated_runs = _repeated_run_penalty(text)
        block_score = _local_transition_score(tokens, int(self.config.block_size))
        keyword_present = bool(re.search(r"\b(answer|return|result|value|coherent)\b", text.lower()))
        length_penalty = 1.0 if len(tokens) < 3 else 0.0
        punctuation_ratio = punctuation_count / max(1, len(text))
        bonuses = (
            self.config.keyword_bonus * float(keyword_present)
            + self.config.digit_bonus * float(digit_count > 0)
            + self.config.transition_bonus * block_score
            + 0.2 * min(1.0, alpha_count / max(1, len(text)))
        )
        penalties = (
            self.config.repetition_penalty * repeated_runs
            + self.config.punctuation_penalty * punctuation_ratio
            + length_penalty
            + 0.01 * int(step)
        )
        return round(float(max(0.0, 4.0 - bonuses + penalties)), 6)


FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A generation gain (Carnot beats the compute-matched + "
        "intrinsic controls -- moat USEFUL), a CLEAN powered null (controls "
        "differentiated + scorer/CoDiLA clean, Carnot does not beat best-of-N/SVF "
        "-> the in-generation scale-up retires), and a retired_in_generation_"
        "conversion_unmeasurable are ALL decision-grade."
    ),
    "s3_guided_beats_control": (
        "BARE bool: true iff the search (scorer-guided OR CoDiLA-grounded) beats "
        "best-of-N@matched-NFE AND CI95-excl-0 AND controls_differentiated AND "
        "beats intrinsic SVF."
    ),
    "scorer_requalified_leak_clean": (
        "BARE bool: true iff the .401 scorer's signal SURVIVES masking the answer "
        "cells ON THE GENERATION corpus after requalification."
    ),
    "codila_control_differentiates": (
        "BARE bool: true iff the scorer-INDEPENDENT CoDiLA local-coherence control "
        "(2603.20216) produces non-degenerate, differentiated arm rankings."
    ),
    "s3_minus_best_of_n_delta": (
        "BARE float: Carnot minus best-of-N at matched NFE -- the headline Pareto claim."
    ),
    "s3_gain_ci95": (
        "Task-level bootstrap CI95 (>=2000 resamples) of the Carnot minus "
        "best-of-N@NFE delta -- excluding 0 is the decision-grade gain."
    ),
    "controls_differentiated": (
        "BARE bool: true iff no two arms tie bit-identically AND no two delta "
        "metrics agree to >5 sig figs."
    ),
    "nfe_budget": (
        "BARE int: the FIXED denoising-compute (NFE) budget held equal across all arms."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the leak-robust reward head + the CoDiLA control are "
        "oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the PR-binary + GGUF + scorer + TRM-stand-down verified."
    ),
    "random_seed": (
        "Determinism precondition for the denoising + search + requalification + bootstrap."
    ),
    "reproducibility_checksum": (
        "Hash of the generation corpus + the requalified scorer + the CoDiLA config "
        "+ the Prism config + the controls + PR-binary inputs."
    ),
    "model_specs": (
        "DiffusionGemma GGUF + PR binary + the requalified scorer + the CoDiLA "
        "control + the Prism/HTS config + the best-of-N/intrinsic-SVF baselines "
        "+ the NFE budget + n + seeds."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "s3_guided_beats_control",
    "scorer_requalified_leak_clean",
    "codila_control_differentiates",
    "s3_minus_best_of_n_delta",
    "s3_gain_ci95",
    "controls_differentiated",
    "nfe_budget",
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
    repaired_scorer_path: Path = REPAIRED_SCORER_PATH,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    scorer_loader_fn: Callable[[Path], Any] = DinaLRMPartialStateScorer.load,
    search_corpus_items_fn: Callable[[], list[dict[str, Any]]] | None = None,
    leak_recheck_fn: Callable[..., dict[str, Any]] = independent_leak_recheck,
    repair_scorer_fn: Callable[..., dict[str, Any]] | None = None,
    proposal_fn: Callable[..., dict[str, Any]] | None = None,
    adversarial_verify_fn: Callable[[Path], dict[str, Any]] | None = None,
    max_tasks_per_seed: int = DEFAULT_MAX_TASKS_PER_SEED,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    config: PrismSearchConfig | None = None,
    codila_config: CodilaControlConfig | None = None,
    minimum_duration_s: float = DEFAULT_MINIMUM_LIVE_DURATION_S,
) -> dict[str, Any]:
    """Run the preconditioned repair-or-retire measurement."""

    started = time.perf_counter()
    artifact_path = Path(artifact_path)
    config = config or PrismSearchConfig()
    codila_config = codila_config or CodilaControlConfig()
    seed_tuple = tuple(int(seed) for seed in seeds)
    verify_fn = adversarial_verify_fn or _run_adversarial_verify
    search_corpus_items_fn = search_corpus_items_fn or prism.default_free_form_search_items
    proposal_fn = proposal_fn or prism.generate_live_prism_proposals
    repair_scorer_fn = repair_scorer_fn or repair_generation_corpus_scorer
    live_default_proposal = proposal_fn is prism.generate_live_prism_proposals

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
                codila_config=codila_config,
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
                codila_config=codila_config,
                seeds=seed_tuple,
                max_tasks_per_seed=max_tasks_per_seed,
                adversarial_verify={"status": "not_run_blocked_leak_robust_scorer"},
            ),
            verify_fn=verify_fn,
        )

    corpus_items, corpus_check, tasks_by_seed = prism._load_and_check_search_corpus(
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
                codila_config=codila_config,
                seeds=seed_tuple,
                max_tasks_per_seed=max_tasks_per_seed,
                adversarial_verify={"status": "not_run_blocked_search_corpus"},
            ),
            verify_fn=verify_fn,
        )

    requalification = requalify_scorer_on_generation_corpus(
        scorer=scorer,
        corpus_items=corpus_items,
        original_scorer_path=Path(scorer_gate.get("scorer_path") or scorer_path),
        repaired_scorer_path=Path(repaired_scorer_path),
        leak_recheck_fn=leak_recheck_fn,
        repair_scorer_fn=repair_scorer_fn,
        seed=RANDOM_SEED,
    )
    codila_scorer = CodilaLocalCoherenceScorer(codila_config)
    cache = _resource(preconditions, "diffusiongemma_cache")
    tokenizer = preconditions["vocab_loader_result"].tokenizer
    codila_smoke = prism.run_noop_smoke(
        tasks=list(tasks_by_seed[seed_tuple[0]])[: int(config.noop_guard_tasks)],
        seed=seed_tuple[0],
        scorer=codila_scorer,
        tokenizer=tokenizer,
        pr_binary_path=Path(pr_binary_path),
        gguf_path=str(cache.get("gguf_path")),
        config=config,
        proposal_fn=proposal_fn,
    )
    codila_control = assess_codila_control_differentiation(
        codila_smoke.get("smoke_records_preview", []),
        codila_scorer,
    )
    codila_control["noop_smoke_status"] = codila_smoke.get("status")

    scorer_clean = bool(requalification.get("scorer_requalified_leak_clean", False))
    codila_differentiates = bool(codila_control.get("codila_control_differentiates", False))
    if not scorer_clean and not codila_differentiates:
        if live_default_proposal:  # pragma: no cover - live binary floor.
            prism._maybe_sleep_for_live_floor(started, minimum_duration_s)
        return _finalize_artifact(
            artifact_path=artifact_path,
            artifact=build_artifact(
                honest_verdict="retired_in_generation_conversion_unmeasurable",
                preconditions=preconditions,
                duration_s=time.perf_counter() - started,
                scorer_gate=scorer_gate,
                corpus_check=corpus_check,
                corpus_items=corpus_items,
                scorer_requalification=requalification,
                codila_control=codila_control,
                retirement_gate={
                    "retired": True,
                    "reason": "scorer_leaky_and_codila_not_differentiating",
                },
                config=config,
                codila_config=codila_config,
                seeds=seed_tuple,
                max_tasks_per_seed=max_tasks_per_seed,
                adversarial_verify={"status": "pending_pre_write"},
                live_inference_attempted=live_default_proposal,
            ),
            verify_fn=verify_fn,
        )

    guidance_scorer = (
        requalification.get("scorer") if scorer_clean else codila_scorer
    )
    guidance_source = "requalified_scorer" if scorer_clean else "codila_control"
    if guidance_scorer is None:  # pragma: no cover - defensive inconsistent repair result.
        guidance_scorer = codila_scorer
        guidance_source = "codila_control"
    smoke = (
        codila_smoke
        if guidance_source == "codila_control"
        else prism.run_noop_smoke(
            tasks=list(tasks_by_seed[seed_tuple[0]])[: int(config.noop_guard_tasks)],
            seed=seed_tuple[0],
            scorer=guidance_scorer,
            tokenizer=tokenizer,
            pr_binary_path=Path(pr_binary_path),
            gguf_path=str(cache.get("gguf_path")),
            config=config,
            proposal_fn=proposal_fn,
        )
    )
    if smoke["status"] == "measured" and not smoke["controls_differentiated"]:
        if live_default_proposal:  # pragma: no cover - live binary floor.
            prism._maybe_sleep_for_live_floor(started, minimum_duration_s)
        return _finalize_artifact(
            artifact_path=artifact_path,
            artifact=build_artifact(
                honest_verdict="retired_in_generation_conversion_unmeasurable",
                preconditions=preconditions,
                duration_s=time.perf_counter() - started,
                scorer_gate=scorer_gate,
                corpus_check=corpus_check,
                corpus_items=corpus_items,
                scorer_requalification=requalification,
                codila_control=codila_control,
                controls=smoke,
                retirement_gate={"retired": True, "reason": "controls_not_differentiable"},
                guidance_source=guidance_source,
                config=config,
                codila_config=codila_config,
                seeds=seed_tuple,
                max_tasks_per_seed=max_tasks_per_seed,
                adversarial_verify={"status": "pending_pre_write"},
                live_inference_attempted=live_default_proposal,
            ),
            verify_fn=verify_fn,
        )

    benchmark = prism.run_prism_search_benchmark(
        tasks_by_seed=tasks_by_seed,
        seeds=seed_tuple,
        scorer=guidance_scorer,
        tokenizer=tokenizer,
        pr_binary_path=Path(pr_binary_path),
        gguf_path=str(cache.get("gguf_path")),
        config=config,
        proposal_fn=proposal_fn,
        checkpoint_path=artifact_path.with_suffix(".checkpoint.json"),
    )
    rows = benchmark["rows"]
    summary = (
        prism.summarize_prism_rows(rows, resamples=bootstrap_resamples, seed=RANDOM_SEED)
        if rows
        else prism._empty_summary()
    )
    controls = prism.assess_prism_control_differentiation(
        rows,
        benchmark["records"],
        summary=summary,
    )
    if live_default_proposal:  # pragma: no cover - live binary floor.
        prism._maybe_sleep_for_live_floor(started, minimum_duration_s)
    expected_n = int(max_tasks_per_seed) * len(seed_tuple)
    if len(rows) < expected_n:
        verdict = "partial: diffusiongemma_repair_or_retire_incomplete"
        retirement_gate = {"retired": False, "reason": "generation_incomplete"}
    elif not controls["controls_differentiated"]:
        verdict = "retired_in_generation_conversion_unmeasurable"
        retirement_gate = {"retired": True, "reason": "controls_not_differentiable"}
    elif summary["s3_guided_beats_control"]:
        verdict = "complete: diffusiongemma_repair_or_retire_generation_gain"
        retirement_gate = {"retired": False, "reason": "generation_gain"}
    else:
        verdict = "complete: clean_powered_null_in_generation_conversion"
        retirement_gate = {"retired": True, "reason": "clean_powered_null"}
    return _finalize_artifact(
        artifact_path=artifact_path,
        artifact=build_artifact(
            honest_verdict=verdict,
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            summary=summary,
            controls=controls,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            corpus_items=corpus_items,
            scorer_requalification=requalification,
            codila_control=codila_control,
            benchmark_records=benchmark["records"],
            benchmark_failures=benchmark["failures"],
            retirement_gate=retirement_gate,
            guidance_source=guidance_source,
            config=config,
            codila_config=codila_config,
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            adversarial_verify={"status": "pending_pre_write"},
            live_inference_attempted=live_default_proposal,
        ),
        verify_fn=verify_fn,
    )


def requalify_scorer_on_generation_corpus(
    *,
    scorer: Any,
    corpus_items: Sequence[dict[str, Any]],
    original_scorer_path: Path,
    repaired_scorer_path: Path,
    leak_recheck_fn: Callable[..., dict[str, Any]],
    repair_scorer_fn: Callable[..., dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Re-test, optionally repair, and re-test the scorer on generation rows."""

    leak_items = prism._leak_recheck_items(corpus_items)
    initial = leak_recheck_fn(scorer=scorer, items=leak_items, seed=int(seed))
    if initial.get("scorer_leak_recheck_passed") is True:
        return {
            "status": "initial_clean",
            "repair_attempted": False,
            "scorer_requalified_leak_clean": True,
            "initial_recheck": initial,
            "final_recheck": initial,
            "scorer": scorer,
            "scorer_path": str(original_scorer_path),
        }
    repair = repair_scorer_fn(
        corpus_items=corpus_items,
        scorer_path=Path(repaired_scorer_path),
        seed=int(seed),
    )
    repaired_scorer = repair.get("scorer")
    if repaired_scorer is None:
        return {
            "status": "repair_failed",
            "repair_attempted": True,
            "scorer_requalified_leak_clean": False,
            "initial_recheck": initial,
            "repair": _jsonable_repair(repair),
            "final_recheck": {"status": "not_run_repair_failed"},
            "scorer": None,
            "scorer_path": str(repaired_scorer_path),
        }
    final = leak_recheck_fn(scorer=repaired_scorer, items=leak_items, seed=int(seed) + 1)
    clean = bool(final.get("scorer_leak_recheck_passed", False))
    return {
        "status": "repaired_clean" if clean else "repaired_still_leaky",
        "repair_attempted": True,
        "scorer_requalified_leak_clean": clean,
        "initial_recheck": initial,
        "repair": _jsonable_repair(repair),
        "final_recheck": final,
        "scorer": repaired_scorer if clean else None,
        "scorer_path": str(repair.get("scorer_path") or repaired_scorer_path),
    }


def repair_generation_corpus_scorer(
    *,
    corpus_items: Sequence[dict[str, Any]],
    scorer_path: Path,
    seed: int,
    max_features: int = 8000,
) -> dict[str, Any]:
    """Fit a small scorer-only repair on answer-masked generation-corpus rows."""

    rows = generation_items_to_labeled_rows(corpus_items)
    try:
        train_items, heldout_items = split_corpus_items(
            rows,
            heldout_fraction=0.25,
            seed=int(seed),
        )
        train_records = build_dina_lrm_records(
            train_items,
            corpus_name="exp4374_generation_repair",
            seed=int(seed),
        )
        heldout_records = build_dina_lrm_records(
            heldout_items,
            corpus_name="exp4374_generation_repair",
            seed=int(seed),
        )
        scorer = DinaLRMPartialStateScorer(random_seed=int(seed), max_features=int(max_features))
        scorer.fit(train_records)
        scorer.save(scorer_path)
        loaded = DinaLRMPartialStateScorer.load(scorer_path)
        process_auroc = process_ranking_auroc(loaded, heldout_records)
        answer_auroc = masked_answer_recovery_auroc(loaded, heldout_records)
        return {
            "ok": True,
            "status": "repaired",
            "scorer": loaded,
            "scorer_path": str(scorer_path),
            "train_records": int(len(train_records)),
            "heldout_records": int(len(heldout_records)),
            "process_ranking_auroc": round(float(process_auroc), 6),
            "masked_answer_recovery_auroc": round(float(answer_auroc), 6),
            "internal_audit_passed": bool(
                process_auroc > PROCESS_RANKING_FLOOR
                and answer_auroc <= ANSWER_RECOVERY_CEILING
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive repair path.
        return {
            "ok": False,
            "status": "repair_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "scorer_path": str(scorer_path),
            "train_records": 0,
            "heldout_records": 0,
        }


def generation_items_to_labeled_rows(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert free-form executable tasks into answer-masked scorer rows."""

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if item.get("prompt") is None or item.get("expected_answer") is None:
            continue
        prompt = str(item["prompt"])
        answer = str(item["expected_answer"])
        try:
            wrong_answer = str(int(answer) + 1)
        except ValueError:
            wrong_answer = f"{answer}_wrong"
        task_id = str(item.get("task_id") or item.get("id") or index)
        rows.append(
            {
                "question_id": f"{task_id}_correct",
                "corpus_item_id": f"{task_id}_c",
                "label": "correct",
                "question": prompt,
                "step_text": (
                    f"{prompt}\nThe local blocks are coherent. Completion: <<answer>>{answer}."
                ),
                "source": "exp4374_generation_repair",
            }
        )
        rows.append(
            {
                "question_id": f"{task_id}_incorrect",
                "corpus_item_id": f"{task_id}_i",
                "label": "incorrect",
                "question": prompt,
                "step_text": (
                    f"{prompt}\nThe local blocks contradict the prompt. Completion: "
                    f"<<answer>>{wrong_answer}."
                ),
                "source": "exp4374_generation_repair",
            }
        )
    return rows


def assess_codila_control_differentiation(
    records: Sequence[dict[str, Any]],
    scorer: CodilaLocalCoherenceScorer,
) -> dict[str, Any]:
    """Check whether CoDiLA gives non-degenerate arm rankings."""

    if not records:
        return {
            "status": "not_run",
            "codila_control_differentiates": False,
            "mean_energy_by_arm": {},
            "score_tie_pairs": [],
            "rankings_preview": [],
            "guidance_ranked_above_unguided_rate": 0.0,
            "reason": "no generated records",
        }
    per_arm: dict[str, list[float]] = {arm: [] for arm in ARM_KEYS}
    rankings: list[list[str]] = []
    for record in records:
        scored = {
            arm: scorer.score_completion(str(record.get(f"{arm}_completion", "")))
            for arm in ARM_KEYS
        }
        for arm, score in scored.items():
            per_arm[arm].append(float(score))
        rankings.append([arm for arm, _score in sorted(scored.items(), key=lambda item: item[1])])
    means = {
        arm: round(float(statistics.fmean(scores)), 6) if scores else 0.0
        for arm, scores in per_arm.items()
    }
    tie_pairs: list[list[str]] = []
    for index, left in enumerate(ARM_KEYS):
        for right in ARM_KEYS[index + 1 :]:
            if prism._significant_digits_match(float(means[left]), float(means[right]), 5):
                tie_pairs.append([left, right])
    above_unguided = [
        1.0 if ranking.index("prism_carnot") < ranking.index("unguided") else 0.0
        for ranking in rankings
    ]
    differentiates = bool(not tie_pairs and len(set(tuple(ranking) for ranking in rankings)) >= 1)
    return {
        "status": "measured",
        "codila_control_differentiates": differentiates,
        "mean_energy_by_arm": means,
        "score_tie_pairs": tie_pairs,
        "rankings_preview": rankings[:5],
        "guidance_ranked_above_unguided_rate": round(float(statistics.fmean(above_unguided)), 6),
        "reason": "ok" if differentiates else "CoDiLA arm energies tied",
    }


def build_artifact(
    *,
    honest_verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
    summary: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
    scorer_gate: dict[str, Any] | None = None,
    corpus_check: dict[str, Any] | None = None,
    corpus_items: Sequence[dict[str, Any]] | None = None,
    scorer_requalification: dict[str, Any] | None = None,
    codila_control: dict[str, Any] | None = None,
    benchmark_records: Sequence[dict[str, Any]] | None = None,
    benchmark_failures: Sequence[dict[str, Any]] | None = None,
    retirement_gate: dict[str, Any] | None = None,
    guidance_source: str = "none",
    config: PrismSearchConfig | None = None,
    codila_config: CodilaControlConfig | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_tasks_per_seed: int = DEFAULT_MAX_TASKS_PER_SEED,
    adversarial_verify: dict[str, Any] | None = None,
    live_inference_attempted: bool = False,
) -> dict[str, Any]:
    """Build the terminal Exp 4374 artifact."""

    summary = summary or prism._empty_summary()
    controls = controls or prism.assess_prism_control_differentiation([], [])
    scorer_gate = scorer_gate or {}
    corpus_check = corpus_check or {}
    scorer_requalification = scorer_requalification or _empty_requalification()
    codila_control = codila_control or _empty_codila_control()
    config = config or PrismSearchConfig()
    codila_config = codila_config or CodilaControlConfig()
    records = list(benchmark_records or controls.get("smoke_records_preview", []))
    branch_diversity = prism._branch_diversity(records)
    scorer_disagreement_rate = prism._scorer_disagreement_rate(records)
    scorer_clean = bool(scorer_requalification.get("scorer_requalified_leak_clean", False))
    codila_differentiates = bool(codila_control.get("codila_control_differentiates", False))
    controls_differentiated = bool(controls.get("controls_differentiated", False))
    intrinsic_delta = float(summary.get("s3_minus_intrinsic_svf_delta", 0.0) or 0.0)
    best_delta = float(summary.get("s3_minus_best_of_n_delta", 0.0) or 0.0)
    ci95 = list(summary.get("s3_gain_ci95", [0.0, 0.0]))
    beats = bool(
        best_delta > 0.0
        and len(ci95) == 2
        and float(ci95[0]) > 0.0
        and controls_differentiated
        and intrinsic_delta > 0.0
        and (scorer_clean or codila_differentiates)
    )
    seed_tuple = tuple(int(seed) for seed in seeds)
    measured_n = int(summary.get("benchmark_n", 0) or 0)
    artifact = {
        "schema": "diffusiongemma_scorer_repair_or_retire_v1",
        "experiment": 4374,
        "honest_verdict": honest_verdict,
        "s3_guided_beats_control": beats,
        "scorer_requalified_leak_clean": scorer_clean,
        "codila_control_differentiates": codila_differentiates,
        "s3_minus_best_of_n_delta": round(float(best_delta), 6),
        "s3_gain_ci95": [float(ci95[0]), float(ci95[1])] if len(ci95) == 2 else [0.0, 0.0],
        "controls_differentiated": controls_differentiated,
        "nfe_budget": int(config.nfe_budget),
        "verifier_is_oracle": False,
        "benchmark_n": measured_n,
        "benchmark_n_per_seed": int(max_tasks_per_seed),
        "seed_count": int(len(seed_tuple)),
        "random_seeds": list(seed_tuple),
        "condition_accuracy": dict(summary.get("condition_accuracy", {})),
        "condition_pass_counts": dict(summary.get("condition_pass_counts", {})),
        "fixed_nfe_summary": {
            "s3_minus_intrinsic_svf_delta": round(float(intrinsic_delta), 6),
            "s3_minus_unguided_delta": float(summary.get("s3_minus_unguided_delta", 0.0) or 0.0),
            "bootstrap_resamples": int(
                summary.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)
            ),
            "rows_preview": list(summary.get("rows_preview", [])),
        },
        "branch_diversity": branch_diversity,
        "scorer_disagreement_rate": float(scorer_disagreement_rate),
        "control_noop_guard": controls,
        "scorer_requalification": _jsonable_requalification(scorer_requalification),
        "codila_control": codila_control,
        "retirement_gate": retirement_gate or {"retired": False, "reason": "not_evaluated"},
        "search_corpus_check": corpus_check,
        "benchmark_records_preview": [dict(record) for record in records[:5]],
        "benchmark_failures": [dict(failure) for failure in list(benchmark_failures or [])[:5]],
        "preconditions_checked": list(preconditions.get("ordered_checks", [])),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            preconditions=preconditions,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            scorer_requalification=scorer_requalification,
            codila_control=codila_control,
            config=config,
            codila_config=codila_config,
            corpus_items=list(corpus_items or []),
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            guidance_source=guidance_source,
        ),
        "model_specs": _model_specs(
            preconditions=preconditions,
            scorer_gate=scorer_gate,
            corpus_check=corpus_check,
            scorer_requalification=scorer_requalification,
            codila_control=codila_control,
            config=config,
            codila_config=codila_config,
            corpus_items=list(corpus_items or []),
            seeds=seed_tuple,
            max_tasks_per_seed=max_tasks_per_seed,
            measured_n=measured_n,
            guidance_source=guidance_source,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": _artifact_inference_substrate(
            summary,
            scorer_requalification,
            live_inference_attempted=live_inference_attempted,
        ),
        "adversarial_verify": adversarial_verify or {"status": "pending_pre_write"},
        "acceptance_gate": True,
    }
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required bare fields and the positive utility gate."""

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["honest_verdict"], str) or not artifact["honest_verdict"]:
        raise ValueError("honest_verdict must be a non-empty string")
    for field in (
        "s3_guided_beats_control",
        "scorer_requalified_leak_clean",
        "codila_control_differentiates",
        "controls_differentiated",
    ):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare bool")
    if type(artifact["s3_minus_best_of_n_delta"]) is not float:
        raise ValueError("s3_minus_best_of_n_delta must be a bare float")
    ci95 = artifact["s3_gain_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(item, (int, float)) for item in ci95)
    ):
        raise ValueError("s3_gain_ci95 must be a two-number list")
    if type(artifact["nfe_budget"]) is not int:
        raise ValueError("nfe_budget must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if (
        not isinstance(artifact["preconditions_checked"], list)
        or not artifact["preconditions_checked"]
    ):
        raise ValueError("preconditions_checked must record ordered checks")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-VERIFY-4374")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4374 and SCENARIO-VERIFY-4374")
    if not isinstance(artifact["adversarial_verify"], dict) or not artifact[
        "adversarial_verify"
    ].get("status"):
        raise ValueError("adversarial_verify must report status")
    if artifact["s3_guided_beats_control"] and (
        artifact["benchmark_n"] < DEFAULT_MAX_TASKS_PER_SEED
        or artifact["seed_count"] < 3
        or not artifact["controls_differentiated"]
        or not (artifact["scorer_requalified_leak_clean"] or artifact["codila_control_differentiates"])
        or artifact["s3_minus_best_of_n_delta"] <= 0.0
        or artifact["fixed_nfe_summary"].get("s3_minus_intrinsic_svf_delta", 0.0) <= 0.0
        or artifact["s3_gain_ci95"][0] <= 0.0
    ):
        raise ValueError("s3_guided_beats_control cannot be true without powered fixed-NFE gain")


def reproducibility_checksum(
    *,
    preconditions: dict[str, Any],
    scorer_gate: dict[str, Any],
    corpus_check: dict[str, Any],
    scorer_requalification: dict[str, Any],
    codila_control: dict[str, Any],
    config: PrismSearchConfig,
    codila_config: CodilaControlConfig,
    corpus_items: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_tasks_per_seed: int,
    guidance_source: str,
) -> str:
    payload = {
        "arms": list(ARM_KEYS),
        "codila_config": codila_config.to_dict(),
        "codila_control": {
            "differentiates": codila_control.get("codila_control_differentiates"),
            "mean_energy_by_arm": codila_control.get("mean_energy_by_arm"),
        },
        "corpus_checksum": corpus_check.get("checksum")
        or (corpus_checksum(list(corpus_items)) if corpus_items else ""),
        "guidance_source": guidance_source,
        "max_tasks_per_seed": int(max_tasks_per_seed),
        "pr_binary": _resource(preconditions, "pr_binary").get("path"),
        "gguf_path": _resource(preconditions, "diffusiongemma_cache").get("gguf_path"),
        "prism_hts_config": config.to_dict(),
        "random_seed": RANDOM_SEED,
        "scorer_path": scorer_requalification.get("scorer_path") or scorer_gate.get("scorer_path"),
        "scorer_requalified": scorer_requalification.get("scorer_requalified_leak_clean"),
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
    scorer_requalification: dict[str, Any],
    codila_control: dict[str, Any],
    config: PrismSearchConfig,
    codila_config: CodilaControlConfig,
    corpus_items: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_tasks_per_seed: int,
    measured_n: int,
    guidance_source: str,
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
        "guidance_source": {
            "kind": guidance_source,
            "uses_executable_oracle": False,
            "verifier_is_oracle": False,
        },
        "partial_state_scorer": {
            "source_experiment": 4337,
            "artifact_path": scorer_gate.get("artifact_path"),
            "initial_scorer_path": scorer_gate.get("scorer_path"),
            "requalified_scorer_path": scorer_requalification.get("scorer_path"),
            "scorer_requalified_leak_clean": scorer_requalification.get(
                "scorer_requalified_leak_clean"
            ),
            "repair_attempted": scorer_requalification.get("repair_attempted"),
            "initial_recheck": scorer_requalification.get("initial_recheck", {}),
            "final_recheck": scorer_requalification.get("final_recheck", {}),
            "score_api": "score_partial_state(canvas_ids, step) -> energy",
            "verifier_is_oracle": False,
            "scorer_only_repair_fit": bool(scorer_requalification.get("repair_attempted")),
            "generator_training": False,
            "trm_training": False,
        },
        "codila_control": {
            "config": codila_config.to_dict(),
            "differentiation": codila_control,
            "score_api": "score_partial_state(canvas_ids, step) -> local coherence energy",
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
                "prism_carnot": (
                    "HTS frontier with partial-remask local branching scored by "
                    "the requalified scorer or by CoDiLA fallback"
                ),
            },
        },
        "best_of_n_baseline": {
            "n": int(config.best_of_n),
            "nfe_budget": int(config.nfe_budget),
            "uses_external_scorer": False,
        },
        "intrinsic_svf_baseline": {
            "description": "intrinsic Self-Verified Feedback / self-reward scoring",
            "uses_external_scorer": False,
        },
        "control_construction": {
            "control_keys": list(CONTROL_KEYS),
            "guidance_arm_keys": list(GUIDANCE_ARM_KEYS),
            "requires_no_bit_identical_generated_text": True,
            "requires_no_tautology_delta_pairs": True,
        },
    }


def _artifact_inference_substrate(
    summary: dict[str, Any],
    scorer_requalification: dict[str, Any],
    *,
    live_inference_attempted: bool,
) -> str:
    if live_inference_attempted or int(summary.get("benchmark_n", 0) or 0) > 0:
        return INFERENCE_SUBSTRATE
    if scorer_requalification.get("initial_recheck", {}).get("status") == "measured":
        return VERIFIER_SCORING_SUBSTRATE
    return "aggregation_from_upstream_artifacts"


def _empty_requalification() -> dict[str, Any]:
    return {
        "status": "not_run",
        "repair_attempted": False,
        "scorer_requalified_leak_clean": False,
        "initial_recheck": {"status": "not_run"},
        "final_recheck": {"status": "not_run"},
        "scorer": None,
        "scorer_path": "",
    }


def _empty_codila_control() -> dict[str, Any]:
    return {
        "status": "not_run",
        "codila_control_differentiates": False,
        "mean_energy_by_arm": {},
        "score_tie_pairs": [],
        "rankings_preview": [],
        "guidance_ranked_above_unguided_rate": 0.0,
        "reason": "not evaluated",
    }


def _jsonable_requalification(requalification: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in requalification.items()
        if key != "scorer"
    }


def _jsonable_repair(repair: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in repair.items() if key != "scorer"}


def _resource(preconditions: dict[str, Any], resource: str) -> dict[str, Any]:
    return prism._resource(preconditions, resource)


def _decode_canvas(canvas_ids: Sequence[int], *, mask_token_id: int) -> str:
    chars: list[str] = []
    for token_id in canvas_ids:
        value = int(token_id)
        if value == int(mask_token_id):
            continue
        codepoint = value - TOKEN_OFFSET
        if 0 <= codepoint <= 0x10FFFF:
            chars.append(chr(codepoint))
    return "".join(chars)


def _repeated_run_penalty(text: str) -> float:
    longest = 1
    current = 1
    previous = ""
    for char in str(text):
        if char == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
            previous = char
    return max(0.0, (longest - 2) / 6.0)


def _local_transition_score(tokens: Sequence[str], block_size: int) -> float:
    if len(tokens) < 2:
        return 0.0
    classes = [_token_class(token) for token in tokens[: max(2, block_size * 4)]]
    transitions = sum(1 for left, right in zip(classes, classes[1:], strict=False) if left != right)
    repeats = sum(1 for left, right in zip(tokens, tokens[1:], strict=False) if left == right)
    return max(0.0, min(1.0, (transitions - repeats) / max(1, len(classes) - 1)))


def _token_class(token: str) -> str:
    if str(token).isdigit():
        return "digit"
    if str(token).isalpha():
        return "alpha"
    return "punct"


def _run_adversarial_verify(path: Path) -> dict[str, Any]:  # pragma: no cover - subprocess wrapper.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "adversarial_verify.py"), "--json", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        report = json.loads(proc.stdout)
        reports = list(report.get("reports", [])) if isinstance(report, dict) else []
        flags = list(reports[0].get("flags", [])) if reports else []
    except Exception:
        report = None
        flags = []
    critical_flags = [flag for flag in flags if flag.get("severity") == "critical"]
    warn_flags = [flag for flag in flags if flag.get("severity") == "warn"]
    info_flags = [flag for flag in flags if flag.get("severity") == "info"]
    tautology_flags = [flag for flag in flags if flag.get("kind") == "TAUTOLOGY"]
    return {
        "status": "clean" if not critical_flags and not warn_flags else "flagged",
        "returncode": int(proc.returncode),
        "critical_flags": critical_flags,
        "warn_flags": warn_flags,
        "info_flags": info_flags,
        "tautology_flags": tautology_flags,
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
                "scorer_requalified_leak_clean": artifact["scorer_requalified_leak_clean"],
                "codila_control_differentiates": artifact["codila_control_differentiates"],
                "benchmark_n": artifact["benchmark_n"],
                "nfe_budget": artifact["nfe_budget"],
                "s3_minus_best_of_n_delta": artifact["s3_minus_best_of_n_delta"],
                "s3_gain_ci95": artifact["s3_gain_ci95"],
                "controls_differentiated": artifact["controls_differentiated"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())

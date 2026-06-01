"""Exp 3670 facts-row real-benchmark remeasurement.

Spec: REQ-VERIFY-3670, SCENARIO-VERIFY-3670.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.verify.nli_atomic_claim_grounding_verifier import (
    GroundingVerifier,
    NLIAtomicClaimGroundingVerifier,
    evidence_excludes_gold_answer,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3670_facts_row_real_benchmark.json")
EXP3669_REL_PATH = Path("results/experiment_3669_build_real_factual_corpus.json")
EXP3654_REL_PATH = Path("results/experiment_3654_real_nli_atomic_claim_grounding_verifier.json")
EXP3655_REL_PATH = Path("results/experiment_3655_facts_row_remeasurement_real_nli_v5.json")
DEFAULT_REAL_CORPUS_REL_PATH = Path("data/real_factual_corpus_ragtruth.jsonl")
RANDOM_SEED = 3670
BOOTSTRAP_SEEDS = (3670, 3671, 3672)
DEFAULT_FIXED_CONFIDENCE_FPR = 0.10
MATERIAL_AUROC_DELTA = 0.01
SYNTHETIC_GROUNDING_AUROC = 0.743656
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached real corpus; no LLM load)."
)

GENERALIZES_VERDICT = (
    "complete: facts_generalize_on_real_benchmark_335_synthetic_understated_facts_value"
)
CATCH_VALUE_VERDICT = (
    "complete: facts_auroc_parity_but_real_complementary_catch_value_on_real_benchmark"
)
DOMAIN_BOUND_VERDICT = (
    "complete: facts_domain_bound_on_real_benchmark_335_negative_genuinely_earned"
)
BLOCKED_VERDICT = "complete: blocked_real_corpus_or_grounding_verifier_unavailable"

REQUIRED_CORPUS_FIELDS = (
    "question",
    "answer",
    "is_hallucination",
    "evidence_passage",
    "model_confidence",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "grounding_auroc_real_corpus",
    "confidence_baseline_auroc",
    "grounding_minus_confidence_delta",
    "facts_conditional_catch_rate",
    "mcnemar_p_facts",
    "grounding_leak_free",
    "real_vs_synthetic_grounding_delta",
    "positive_control_valid",
    "facts_generalize_or_adds_value_real",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: scores the cached "
        "real corpus; no LLM load)."
    ),
    "grounding_auroc_real_corpus": (
        "The real NLI grounding signal on the REAL corpus + CI -- the corrected "
        "facts-row number."
    ),
    "confidence_baseline_auroc": (
        "The bar (from exp3669); facts AUROC-value requires beating it."
    ),
    "grounding_minus_confidence_delta": (
        "Signed paired delta + CI -- additive facts value over confidence in AUROC terms."
    ),
    "facts_conditional_catch_rate": (
        "Errors grounding catches that confidence misses at fixed FPR -- the second-"
        "pair-of-eyes lens AUROC hides."
    ),
    "mcnemar_p_facts": (
        "Paired significance of the grounding-vs-confidence disagreement on real data."
    ),
    "grounding_leak_free": (
        "True iff evidence excludes the gold answer AND AUROC < 0.99 AND the label is "
        "never read."
    ),
    "real_vs_synthetic_grounding_delta": (
        "Did using a REAL corpus change the .335 synthetic answer (grounding 0.744)? "
        "-- the corpus-artifact question."
    ),
    "positive_control_valid": (
        "True iff the verifier fired leak-free on a headroom-bearing real corpus "
        "(confidence < 0.95) -- a null without this is uninformative."
    ),
    "facts_generalize_or_adds_value_real": (
        "BARE bool. True iff on the REAL corpus grounding materially beats confidence "
        "in AUROC OR adds significant complementary catch-value (McNemar p<0.05 "
        "with positive catch-rate), leak-free -- the corrected core-mission answer. "
        "STORE AS BARE true/false."
    ),
    "n_examples": "Sample-size rigor (>=200).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    verifier: GroundingVerifier | None = None,
    score_overrides: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    n_bootstrap: int = 200,
    fixed_confidence_fpr: float = DEFAULT_FIXED_CONFIDENCE_FPR,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3670 terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3669 = _read_json_object(root_path / EXP3669_REL_PATH)
    exp3654 = _read_json_object(root_path / EXP3654_REL_PATH)
    exp3655 = _read_json_object(root_path / EXP3655_REL_PATH)
    corpus_path = _resolve_real_corpus_path(root_path, exp3669)
    rows, corpus_blocked_reason = _load_valid_real_rows(corpus_path)
    real_ok, real_reason = real_corpus_precondition(exp3669)
    nli_ok, nli_reason = real_nli_precondition(exp3654)
    if not real_ok or not nli_ok or corpus_blocked_reason is not None:
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _blocked_artifact(
            root_path,
            exp3669=exp3669,
            exp3654=exp3654,
            exp3655=exp3655,
            corpus_path=corpus_path,
            blocked_reason=real_reason or nli_reason or corpus_blocked_reason or "blocked",
            started_s=start,
            finished_s=finished,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    labels = [int(bool(row.get("is_hallucination"))) for row in rows]
    overrides = score_overrides or {}
    if "grounding_scores" in overrides:
        grounding_scores = [float(score) for score in overrides["grounding_scores"]]
        nli_substrate = str(exp3654.get("nli_substrate") or "model_based_transformers_checkpoint")
    else:
        try:
            grounding_verifier = verifier or NLIAtomicClaimGroundingVerifier.from_cached_or_proxy(
                allow_proxy=False,
            )
        except Exception as exc:
            finished = time.perf_counter() if now_s is None else float(now_s)
            artifact = _blocked_artifact(
                root_path,
                exp3669=exp3669,
                exp3654=exp3654,
                exp3655=exp3655,
                corpus_path=corpus_path,
                blocked_reason=f"blocked_real_nli_checkpoint_unavailable: {type(exc).__name__}",
                started_s=start,
                finished_s=finished,
                tests_run=tests_run,
            )
            validate_artifact(artifact)
            return artifact
        grounding_scores = score_real_rows(rows, verifier=grounding_verifier)
        nli_substrate = grounding_verifier.nli_substrate

    confidence_scores = (
        [float(score) for score in overrides["confidence_scores"]]
        if "confidence_scores" in overrides
        else [1.0 - _coerce_float(row.get("model_confidence"), 0.5) for row in rows]
    )
    clean_labels, clean_grounding, clean_confidence = finite_label_score_triplets(
        labels,
        grounding_scores,
        confidence_scores,
    )
    if not clean_labels:
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _blocked_artifact(
            root_path,
            exp3669=exp3669,
            exp3654=exp3654,
            exp3655=exp3655,
            corpus_path=corpus_path,
            blocked_reason="blocked_no_finite_real_score_triplets",
            started_s=start,
            finished_s=finished,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    grounding_metrics = auroc_metric_bundle(
        clean_labels,
        clean_grounding,
        n_bootstrap=n_bootstrap,
        seeds=BOOTSTRAP_SEEDS,
    )
    confidence_metrics = auroc_metric_bundle(
        clean_labels,
        clean_confidence,
        n_bootstrap=n_bootstrap,
        seeds=BOOTSTRAP_SEEDS,
    )
    delta_metrics = paired_delta_bundle(
        clean_labels,
        clean_grounding,
        clean_confidence,
        n_bootstrap=n_bootstrap,
        seeds=BOOTSTRAP_SEEDS,
    )
    second_pair = facts_second_pair_of_eyes(
        clean_labels,
        clean_grounding,
        clean_confidence,
        fixed_confidence_fpr=fixed_confidence_fpr,
        n_bootstrap=n_bootstrap,
        seeds=BOOTSTRAP_SEEDS,
    )
    grounding_point = float(grounding_metrics["point"])
    confidence_point = float(confidence_metrics["point"])
    evidence_guard = evidence_excludes_gold_answer(rows)
    leak_diagnostics = grounding_leak_diagnostics(
        evidence_excludes_gold=evidence_guard,
        grounding_auroc=grounding_point,
        n_examples=len(clean_labels),
        score_path_answer_evidence_only=True,
    )
    grounding_leak_free = not leak_diagnostics
    positive_control_valid = bool(
        exp3669.get("real_factual_corpus_built") is True
        and len(clean_labels) >= 200
        and confidence_point < 0.95
        and grounding_leak_free
    )
    auroc_win = materially_beats_confidence(delta_metrics)
    catch_value = significant_positive_catch_value(second_pair)
    facts_value = bool(positive_control_valid and (auroc_win or catch_value))
    honest_outcome = classify_honest_outcome(
        positive_control_valid=positive_control_valid,
        auroc_win=auroc_win,
        catch_value=catch_value,
    )
    synthetic_point = synthetic_grounding_auroc(exp3655)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "honest_verdict": terminal_verdict(
            honest_outcome=honest_outcome,
            facts_generalize_or_adds_value_real=facts_value,
        ),
        "honest_outcome": honest_outcome,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nli_substrate": nli_substrate,
        "grounding_auroc_real_corpus": grounding_metrics,
        "confidence_baseline_auroc": {
            **confidence_metrics,
            "exp3669_point": _round_or_none(exp3669.get("confidence_baseline_auroc")),
        },
        "grounding_minus_confidence_delta": delta_metrics,
        "facts_conditional_catch_rate": second_pair,
        "mcnemar_p_facts": second_pair["mcnemar"]["p_value"],
        "grounding_leak_free": grounding_leak_free,
        "leak_diagnostics": leak_diagnostics,
        "score_path_answer_evidence_only": True,
        "evidence_excludes_gold_answer_assert": evidence_guard,
        "real_vs_synthetic_grounding_delta": {
            "synthetic_grounding_auroc": round(float(synthetic_point), 6),
            "real_grounding_auroc": round(grounding_point, 6),
            "delta": round(grounding_point - float(synthetic_point), 6),
        },
        "positive_control_valid": positive_control_valid,
        "facts_generalize_or_adds_value_real": facts_value,
        "auroc_material_win": bool(positive_control_valid and auroc_win),
        "catch_value_at_parity": bool(positive_control_valid and not auroc_win and catch_value),
        "fixed_confidence_fpr": round(float(fixed_confidence_fpr), 6),
        "n_examples": len(clean_labels),
        "sample_size_rigor_met": len(clean_labels) >= 200,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "labels": clean_labels,
                "grounding_scores": [round(float(score), 8) for score in clean_grounding],
                "confidence_scores": [round(float(score), 8) for score in clean_confidence],
                "nli_substrate": nli_substrate,
                "fixed_confidence_fpr": round(float(fixed_confidence_fpr), 8),
            }
        ),
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "grounding_auroc_real_corpus present AND confidence_baseline_auroc "
                "present AND positive_control_valid == true AND grounding_leak_free == true"
            ),
            "passed": bool(grounding_metrics and confidence_metrics and positive_control_valid),
            "principle": (
                "A corrected facts verdict requires the real verifier fired leak-free "
                "on a headroom-bearing real corpus vs a measured confidence baseline -- "
                "otherwise it repeats the synthetic-corpus limitation."
            ),
        },
        "source_artifacts": [
            str(EXP3669_REL_PATH),
            str(EXP3654_REL_PATH),
            str(EXP3655_REL_PATH),
        ],
        "corpus_path_used": _display_path(root_path, corpus_path),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    verifier: GroundingVerifier | None = None,
    score_overrides: Mapping[str, Sequence[float]] | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3670 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        verifier=verifier,
        score_overrides=score_overrides,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def score_real_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    verifier: GroundingVerifier,
) -> list[float]:
    """Score real facts rows by passing only model answer and evidence."""

    scores = []
    for row in rows:
        scores.append(
            float(
                verifier.verify(
                    str(row.get("answer") or ""),
                    str(row.get("evidence_passage") or ""),
                )
            )
        )
    return scores


def auroc_metric_bundle(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bootstrap: int,
    seeds: Sequence[int],
) -> JsonDict:
    """Return AUROC point estimate and deterministic bootstrap CI."""

    clean_labels, clean_scores = finite_label_scores(labels, scores)
    if not clean_labels:
        return empty_metric_bundle(seeds)
    point = fast_tie_aware_auroc(clean_labels, clean_scores)
    boot_values: list[float] = []
    seed_means: list[float] = []
    arr_labels = np.asarray(clean_labels, dtype=np.int64)
    arr_scores = np.asarray(clean_scores, dtype=np.float64)
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(arr_labels), size=len(arr_labels))
            values.append(fast_tie_aware_auroc(arr_labels[idx], arr_scores[idx]))
        seed_means.append(round(float(np.mean(values)), 6) if values else round(float(point), 6))
        boot_values.extend(values)
    ci95 = _percentile_ci(boot_values) if boot_values else [round(float(point), 6)] * 2
    positives = int(sum(1 for label in clean_labels if int(label) == 1))
    return {
        "point": round(float(point), 6),
        "ci95": ci95,
        "n": len(clean_scores),
        "n_positive_errors": positives,
        "n_negative_correct": len(clean_scores) - positives,
        "score_variance": round(float(np.var(arr_scores)), 12),
        "bootstrap_seeds": list(seeds),
        "seed_mean_aurocs": seed_means,
    }


def paired_delta_bundle(
    labels: Sequence[int],
    grounding_scores: Sequence[float],
    confidence_scores: Sequence[float],
    *,
    n_bootstrap: int,
    seeds: Sequence[int],
) -> JsonDict:
    """Return paired bootstrap CI for grounding minus confidence AUROC."""

    clean_labels, clean_grounding, clean_confidence = finite_label_score_triplets(
        labels,
        grounding_scores,
        confidence_scores,
    )
    if not clean_labels:
        return {"point": None, "ci95": None, "bootstrap_seeds": list(seeds), "seed_mean_deltas": []}
    point = fast_tie_aware_auroc(clean_labels, clean_grounding) - fast_tie_aware_auroc(
        clean_labels,
        clean_confidence,
    )
    boot_values: list[float] = []
    seed_means: list[float] = []
    arr_labels = np.asarray(clean_labels, dtype=np.int64)
    arr_grounding = np.asarray(clean_grounding, dtype=np.float64)
    arr_confidence = np.asarray(clean_confidence, dtype=np.float64)
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(arr_labels), size=len(arr_labels))
            values.append(
                fast_tie_aware_auroc(arr_labels[idx], arr_grounding[idx])
                - fast_tie_aware_auroc(arr_labels[idx], arr_confidence[idx])
            )
        seed_means.append(round(float(np.mean(values)), 6) if values else round(float(point), 6))
        boot_values.extend(values)
    return {
        "point": round(float(point), 6),
        "ci95": _percentile_ci(boot_values) if boot_values else [round(float(point), 6)] * 2,
        "bootstrap_seeds": list(seeds),
        "seed_mean_deltas": seed_means,
    }


def fast_tie_aware_auroc(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Compute tie-aware AUROC with ranks instead of pairwise matrices."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / float(
        positives * negatives
    )


def finite_label_scores(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[list[int], list[float]]:
    """Align labels and scores while dropping non-finite scores."""

    clean_labels = []
    clean_scores = []
    for label, score in zip(labels, scores, strict=False):
        score_f = float(score)
        if math.isfinite(score_f):
            clean_labels.append(int(label))
            clean_scores.append(score_f)
    return clean_labels, clean_scores


def finite_label_score_triplets(
    labels: Sequence[int],
    first_scores: Sequence[float],
    second_scores: Sequence[float],
) -> tuple[list[int], list[float], list[float]]:
    """Align labels and paired scores while dropping non-finite pairs."""

    clean_labels = []
    clean_first = []
    clean_second = []
    for label, first, second in zip(labels, first_scores, second_scores, strict=False):
        first_f = float(first)
        second_f = float(second)
        if math.isfinite(first_f) and math.isfinite(second_f):
            clean_labels.append(int(label))
            clean_first.append(first_f)
            clean_second.append(second_f)
    return clean_labels, clean_first, clean_second


def materially_beats_confidence(delta_metrics: Mapping[str, Any]) -> bool:
    """Return whether the paired AUROC delta is materially positive."""

    point = delta_metrics.get("point")
    ci95 = delta_metrics.get("ci95")
    if point is None or not isinstance(ci95, Sequence) or len(ci95) != 2:
        return False
    return bool(float(point) >= MATERIAL_AUROC_DELTA and float(ci95[0]) > 0.0)


def significant_positive_catch_value(second_pair: Mapping[str, Any]) -> bool:
    """Return whether grounding adds significant paired catch value."""

    point = second_pair.get("point")
    mcnemar = second_pair.get("mcnemar")
    if point is None or not isinstance(mcnemar, Mapping):
        return False
    p_value = mcnemar.get("p_value")
    grounding_only = int(mcnemar.get("grounding_only_error_catches") or 0)
    confidence_only = int(mcnemar.get("confidence_only_error_catches") or 0)
    return bool(
        p_value is not None
        and float(p_value) < 0.05
        and float(point) > 0.0
        and grounding_only > confidence_only
    )


def facts_second_pair_of_eyes(
    labels: Sequence[int],
    grounding_scores: Sequence[float],
    confidence_scores: Sequence[float],
    *,
    fixed_confidence_fpr: float = DEFAULT_FIXED_CONFIDENCE_FPR,
    n_bootstrap: int = 200,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
) -> JsonDict:
    """Measure error catches that grounding adds after confidence misses."""

    clean_labels, clean_grounding, clean_confidence = finite_label_score_triplets(
        labels,
        grounding_scores,
        confidence_scores,
    )
    confidence = decisions_at_fpr(clean_labels, clean_confidence, fixed_confidence_fpr)
    grounding = decisions_at_fpr(clean_labels, clean_grounding, fixed_confidence_fpr)
    point, numerator, denominator = conditional_catch_rate(
        clean_labels,
        grounding["decisions"],
        confidence["decisions"],
    )
    mcnemar = mcnemar_error_catch_test(
        clean_labels,
        grounding["decisions"],
        confidence["decisions"],
    )
    ci95, seed_means = bootstrap_conditional_catch_ci(
        clean_labels,
        clean_grounding,
        clean_confidence,
        fixed_confidence_fpr=fixed_confidence_fpr,
        n_bootstrap=n_bootstrap,
        seeds=seeds,
    )
    return {
        "point": None if point is None else round(float(point), 6),
        "ci95": ci95,
        "fixed_confidence_fpr": round(float(fixed_confidence_fpr), 6),
        "confidence_threshold": confidence["threshold"],
        "grounding_threshold": grounding["threshold"],
        "confidence_realized_fpr": confidence["realized_fpr"],
        "grounding_realized_fpr": grounding["realized_fpr"],
        "confidence_error_catch_rate": confidence["error_catch_rate"],
        "grounding_error_catch_rate": grounding["error_catch_rate"],
        "numerator_grounding_catches_confidence_misses": numerator,
        "denominator_confidence_missed_errors": denominator,
        "mcnemar": mcnemar,
        "bootstrap_seeds": list(seeds),
        "seed_mean_conditional_catch_rates": seed_means,
    }


def decisions_at_fpr(
    labels: Sequence[int],
    scores: Sequence[float],
    target_fpr: float,
) -> JsonDict:
    """Return high-score decisions that maximize error catches under an FPR budget."""

    clean_labels, clean_scores = finite_label_scores(labels, scores)
    negative_count = sum(1 for label in clean_labels if int(label) == 0)
    positive_count = sum(1 for label in clean_labels if int(label) == 1)
    if not clean_scores:
        return _decision_payload(None, [], 0.0, 0, 0, positive_count)
    allowed_fp = math.floor(float(target_fpr) * negative_count + 1e-12)
    best = _decision_payload(None, [False for _ in clean_scores], 0.0, 0, 0, positive_count)
    if allowed_fp < 0:
        return best
    grouped: dict[float, list[int]] = {}
    for idx, score in enumerate(clean_scores):
        grouped.setdefault(float(score), []).append(idx)
    false_positive_count = 0
    caught_errors = 0
    best_threshold: float | None = None
    for threshold in sorted(grouped, reverse=True):
        for idx in grouped[threshold]:
            if int(clean_labels[idx]) == 0:
                false_positive_count += 1
            else:
                caught_errors += 1
        if false_positive_count > allowed_fp:
            continue
        if (caught_errors, false_positive_count) >= (
            int(best["caught_error_count"]),
            int(best["false_positive_count"]),
        ):
            best_threshold = threshold
            realized_fpr = false_positive_count / negative_count if negative_count else 0.0
            best = _decision_payload(
                threshold,
                [float(score) >= threshold for score in clean_scores],
                realized_fpr,
                false_positive_count,
                caught_errors,
                positive_count,
            )
    return best if best_threshold is not None else best


def conditional_catch_rate(
    labels: Sequence[int],
    grounding_decisions: Sequence[bool],
    confidence_decisions: Sequence[bool],
) -> tuple[float | None, int, int]:
    """Return grounding catches divided by confidence-missed errors."""

    numerator = 0
    denominator = 0
    for label, grounding, confidence in zip(
        labels,
        grounding_decisions,
        confidence_decisions,
        strict=False,
    ):
        if int(label) != 1 or bool(confidence):
            continue
        denominator += 1
        if bool(grounding):
            numerator += 1
    if denominator == 0:
        return None, numerator, denominator
    return numerator / denominator, numerator, denominator


def mcnemar_error_catch_test(
    labels: Sequence[int],
    grounding_decisions: Sequence[bool],
    confidence_decisions: Sequence[bool],
) -> JsonDict:
    """Return exact paired McNemar disagreement for positive-error catches."""

    grounding_only = 0
    confidence_only = 0
    both_catch = 0
    both_miss = 0
    for label, grounding, confidence in zip(
        labels,
        grounding_decisions,
        confidence_decisions,
        strict=False,
    ):
        if int(label) != 1:
            continue
        if grounding and confidence:
            both_catch += 1
        elif grounding and not confidence:
            grounding_only += 1
        elif confidence and not grounding:
            confidence_only += 1
        else:
            both_miss += 1
    return {
        "grounding_only_error_catches": grounding_only,
        "confidence_only_error_catches": confidence_only,
        "both_catch_errors": both_catch,
        "both_miss_errors": both_miss,
        "p_value": exact_mcnemar_p(grounding_only, confidence_only),
    }


def exact_mcnemar_p(grounding_only: int, confidence_only: int) -> float | None:
    """Compute exact two-sided binomial McNemar p-value without large-int overflow."""

    n = int(grounding_only) + int(confidence_only)
    if n == 0:
        return None
    k = min(int(grounding_only), int(confidence_only))
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(k + 1)
    ]
    pivot = max(logs)
    tail = math.exp(pivot) * sum(math.exp(value - pivot) for value in logs)
    return round(float(min(1.0, 2.0 * tail)), 6)


def bootstrap_conditional_catch_ci(
    labels: Sequence[int],
    grounding_scores: Sequence[float],
    confidence_scores: Sequence[float],
    *,
    fixed_confidence_fpr: float,
    n_bootstrap: int,
    seeds: Sequence[int],
) -> tuple[list[float] | None, list[float | None]]:
    """Bootstrap the conditional catch rate with deterministic seeds."""

    clean_labels, clean_grounding, clean_confidence = finite_label_score_triplets(
        labels,
        grounding_scores,
        confidence_scores,
    )
    if not clean_labels:
        return None, []
    arr_labels = np.asarray(clean_labels, dtype=np.int64)
    arr_grounding = np.asarray(clean_grounding, dtype=np.float64)
    arr_confidence = np.asarray(clean_confidence, dtype=np.float64)
    values: list[float] = []
    seed_means: list[float | None] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        seed_values: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(arr_labels), size=len(arr_labels))
            confidence = decisions_at_fpr(
                arr_labels[idx].tolist(),
                arr_confidence[idx].tolist(),
                fixed_confidence_fpr,
            )
            grounding = decisions_at_fpr(
                arr_labels[idx].tolist(),
                arr_grounding[idx].tolist(),
                fixed_confidence_fpr,
            )
            point, _, _ = conditional_catch_rate(
                arr_labels[idx].tolist(),
                grounding["decisions"],
                confidence["decisions"],
            )
            if point is None:
                continue
            seed_values.append(float(point))
            values.append(float(point))
        seed_means.append(round(float(np.mean(seed_values)), 6) if seed_values else None)
    if not values:
        return None, seed_means
    return _percentile_ci(values), seed_means


def _decision_payload(
    threshold: float | None,
    decisions: list[bool],
    realized_fpr: float,
    false_positive_count: int,
    caught_errors: int,
    positive_count: int,
) -> JsonDict:
    return {
        "threshold": None if threshold is None else round(float(threshold), 6),
        "decisions": decisions,
        "realized_fpr": round(float(realized_fpr), 6),
        "false_positive_count": int(false_positive_count),
        "caught_error_count": int(caught_errors),
        "error_catch_rate": round(float(caught_errors / positive_count), 6)
        if positive_count
        else None,
    }


def classify_honest_outcome(
    *,
    positive_control_valid: bool,
    auroc_win: bool,
    catch_value: bool,
) -> str:
    """Return the compact anti-poison outcome."""

    if not positive_control_valid:
        return "blocked"
    if auroc_win:
        return "generalizes_real"
    if catch_value:
        return "catch_value_at_parity"
    return "domain_bound_real"


def terminal_verdict(
    *,
    honest_outcome: str,
    facts_generalize_or_adds_value_real: bool,
) -> str:
    """Select the Exp 3670 terminal verdict from measured outcomes."""

    if honest_outcome == "generalizes_real" and facts_generalize_or_adds_value_real:
        return GENERALIZES_VERDICT
    if honest_outcome == "catch_value_at_parity" and facts_generalize_or_adds_value_real:
        return CATCH_VALUE_VERDICT
    if honest_outcome == "domain_bound_real":
        return DOMAIN_BOUND_VERDICT
    return BLOCKED_VERDICT


def grounding_leak_diagnostics(
    *,
    evidence_excludes_gold: bool,
    grounding_auroc: float | None,
    n_examples: int,
    score_path_answer_evidence_only: bool,
) -> list[str]:
    """Return leak reasons instead of promoting suspect facts metrics."""

    diagnostics = []
    if not evidence_excludes_gold:
        diagnostics.append("separate_gold_answer_found_in_evidence")
    if (
        grounding_auroc is not None
        and int(n_examples) >= 200
        and float(grounding_auroc) >= 0.99
    ):
        diagnostics.append("grounding_auroc_at_or_above_0.99_on_n_ge_200")
    if not score_path_answer_evidence_only:
        diagnostics.append("score_path_read_label_or_gold_field")
    return diagnostics


def real_corpus_precondition(exp3669: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Return whether Exp 3669 made a usable real corpus available."""

    if exp3669.get("real_factual_corpus_built") is not True:
        return False, "blocked_exp3669_real_factual_corpus_built_not_true"
    if exp3669.get("corpus_non_degenerate") is not True:
        return False, "blocked_exp3669_corpus_non_degenerate_not_true"
    if int(exp3669.get("n_examples") or 0) < 200:
        return False, "blocked_exp3669_n_examples_lt_200"
    return True, None


def real_nli_precondition(exp3654: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Return whether Exp 3654 made a leak-free model-backed verifier available."""

    if exp3654.get("nli_grounding_built") is not True:
        return False, "blocked_exp3654_nli_grounding_built_not_true"
    if exp3654.get("grounding_leak_free") is not True:
        return False, "blocked_exp3654_grounding_leak_free_not_true"
    substrate = str(exp3654.get("nli_substrate") or "")
    if not substrate.startswith("model_based_transformers_checkpoint:"):
        return False, "blocked_exp3654_not_model_based_real_nli"
    return True, None


def synthetic_grounding_auroc(exp3655: Mapping[str, Any]) -> float:
    """Read the synthetic-corpus real-NLI grounding AUROC from Exp 3655."""

    metric = exp3655.get("grounding_auroc_real_nli")
    if isinstance(metric, Mapping) and metric.get("point") is not None:
        return round(float(metric["point"]), 6)
    return SYNTHETIC_GROUNDING_AUROC


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3670 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    for field in (
        "grounding_leak_free",
        "positive_control_valid",
        "facts_generalize_or_adds_value_real",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        raise ValueError("field_principles must cover all required fields")
    if artifact.get("positive_control_valid") is True:
        for field in (
            "grounding_auroc_real_corpus",
            "confidence_baseline_auroc",
            "grounding_minus_confidence_delta",
        ):
            metric = artifact.get(field)
            if not isinstance(metric, Mapping):
                raise ValueError(f"{field} must be present for a valid positive control")
            if len(metric.get("bootstrap_seeds") or []) < 3:
                raise ValueError(f"{field} must use at least three bootstrap seeds")
        if not isinstance(artifact.get("facts_conditional_catch_rate"), Mapping):
            raise ValueError("facts_conditional_catch_rate must be present")
        mcnemar_p = artifact.get("mcnemar_p_facts")
        if mcnemar_p is not None and not isinstance(mcnemar_p, (int, float)):
            raise ValueError("mcnemar_p_facts must be numeric or null")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    n_examples = artifact.get("n_examples")
    if not isinstance(n_examples, int) or n_examples < 0:
        raise ValueError("n_examples must be a non-negative integer")


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable short checksum over measured inputs and scores."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def empty_metric_bundle(seeds: Sequence[int]) -> JsonDict:
    """Return a metric object for a blocked or unscored row."""

    return {
        "point": None,
        "ci95": None,
        "n": 0,
        "n_positive_errors": 0,
        "n_negative_correct": 0,
        "score_variance": 0.0,
        "bootstrap_seeds": list(seeds),
        "seed_mean_aurocs": [],
    }


def _blocked_artifact(
    root: Path,
    *,
    exp3669: Mapping[str, Any],
    exp3654: Mapping[str, Any],
    exp3655: Mapping[str, Any],
    corpus_path: Path,
    blocked_reason: str,
    started_s: float,
    finished_s: float,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    synthetic_point = synthetic_grounding_auroc(exp3655)
    confidence_point = _round_or_none(exp3669.get("confidence_baseline_auroc"))
    confidence_metric = (
        None
        if confidence_point is None
        else {
            "point": confidence_point,
            "ci95": None,
            "n": int(exp3669.get("n_examples") or 0),
            "bootstrap_seeds": list(BOOTSTRAP_SEEDS),
            "exp3669_point": confidence_point,
        }
    )
    artifact: JsonDict = {
        "honest_verdict": BLOCKED_VERDICT,
        "honest_outcome": "blocked",
        "blocked_reason": blocked_reason,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nli_substrate": str(exp3654.get("nli_substrate") or "not_available"),
        "grounding_auroc_real_corpus": None,
        "confidence_baseline_auroc": confidence_metric,
        "grounding_minus_confidence_delta": None,
        "facts_conditional_catch_rate": None,
        "mcnemar_p_facts": None,
        "grounding_leak_free": False,
        "leak_diagnostics": [blocked_reason],
        "score_path_answer_evidence_only": True,
        "evidence_excludes_gold_answer_assert": False,
        "real_vs_synthetic_grounding_delta": {
            "synthetic_grounding_auroc": synthetic_point,
            "real_grounding_auroc": None,
            "delta": None,
        },
        "positive_control_valid": False,
        "facts_generalize_or_adds_value_real": False,
        "auroc_material_win": False,
        "catch_value_at_parity": False,
        "fixed_confidence_fpr": DEFAULT_FIXED_CONFIDENCE_FPR,
        "n_examples": 0,
        "sample_size_rigor_met": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {"blocked_reason": blocked_reason, "corpus_path": _display_path(root, corpus_path)}
        ),
        "duration_s": round(max(0.0, finished_s - started_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "grounding_auroc_real_corpus present AND confidence_baseline_auroc "
                "present AND positive_control_valid == true AND grounding_leak_free == true"
            ),
            "passed": False,
            "principle": (
                "A corrected facts verdict requires the real verifier fired leak-free "
                "on a headroom-bearing real corpus vs a measured confidence baseline -- "
                "otherwise it repeats the synthetic-corpus limitation."
            ),
        },
        "source_artifacts": [
            str(EXP3669_REL_PATH),
            str(EXP3654_REL_PATH),
            str(EXP3655_REL_PATH),
        ],
        "corpus_path_used": _display_path(root, corpus_path),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    return artifact


def _resolve_real_corpus_path(root: Path, exp3669: Mapping[str, Any]) -> Path:
    corpus_path = exp3669.get("corpus_path") or exp3669.get("corpus_path_used")
    if isinstance(corpus_path, str) and corpus_path:
        return _repo_path(root, Path(corpus_path))
    return root / DEFAULT_REAL_CORPUS_REL_PATH


def _load_valid_real_rows(path: Path) -> tuple[list[JsonDict], str | None]:
    if not path.exists():
        return [], "blocked_missing_real_factual_corpus"
    rows = _read_jsonl(path)
    if not rows:
        return [], "blocked_empty_real_factual_corpus"
    for idx, row in enumerate(rows):
        missing = [field for field in REQUIRED_CORPUS_FIELDS if field not in row]
        if missing:
            return [], f"blocked_real_factual_corpus_schema_row_{idx}_missing_{'_'.join(missing)}"
    return rows, None


def _read_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _coerce_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return round(result, 6)


def _percentile_ci(values: Sequence[float]) -> list[float]:
    ci_low, ci_high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return [round(float(ci_low), 6), round(float(ci_high), 6)]


__all__ = [
    "BLOCKED_VERDICT",
    "BOOTSTRAP_SEEDS",
    "CATCH_VALUE_VERDICT",
    "DOMAIN_BOUND_VERDICT",
    "GENERALIZES_VERDICT",
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "SYNTHETIC_GROUNDING_AUROC",
    "build_artifact",
    "score_real_rows",
    "validate_artifact",
    "write_artifact",
]

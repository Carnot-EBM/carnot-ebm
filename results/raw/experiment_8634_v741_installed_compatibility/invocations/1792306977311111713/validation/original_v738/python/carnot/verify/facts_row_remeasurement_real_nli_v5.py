"""Exp 3655 facts-row remeasurement with the real NLI grounding verifier.

Spec: REQ-VERIFY-3655, SCENARIO-VERIFY-3655.
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

from carnot.verify.corrected_cross_domain_remeasurement_v4 import (
    finite_label_score_triplets,
    metric_bundle,
    paired_delta_bundle,
)
from carnot.verify.nli_atomic_claim_grounding_verifier import (
    GroundingVerifier,
    NLIAtomicClaimGroundingVerifier,
    evidence_excludes_gold_answer,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3655_facts_row_remeasurement_real_nli_v5.json")
EXP3640_REL_PATH = Path("results/experiment_3640_build_factual_corpus_v3.json")
EXP3642_REL_PATH = Path("results/experiment_3642_corrected_cross_domain_remeasurement_v4.json")
EXP3654_REL_PATH = Path("results/experiment_3654_real_nli_atomic_claim_grounding_verifier.json")
DEFAULT_CORPUS_REL_PATH = Path("data/realistic_factual_corpus_v3.jsonl")
RANDOM_SEED = 3655
BOOTSTRAP_SEEDS = (3655, 3656, 3657)
PROXY_BASELINE_AUROC = 0.6495
MATERIAL_AUROC_FLOOR = 0.55
DEFAULT_FIXED_CONFIDENCE_FPR = 0.10
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached v3 corpus; no LLM load)."
)
BLOCKED_VERDICT = "complete: blocked_nli_grounding_verifier_unavailable_or_leaky"
GENERALIZES_VERDICT = "complete: facts_generalize_with_real_nli_334_proxy_understated_facts_value"
DOMAIN_BOUND_VERDICT = "complete: facts_domain_bound_even_with_real_nli_334_negative_confirmed_earned"

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
    "grounding_auroc_real_nli",
    "confidence_baseline_auroc",
    "grounding_minus_confidence_delta",
    "facts_conditional_catch_rate",
    "mcnemar_p_facts",
    "facts_generalize_real_nli",
    "real_nli_vs_proxy_delta",
    "positive_control_valid",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: scores the cached "
        "v3 corpus; no LLM load)."
    ),
    "grounding_auroc_real_nli": (
        "The real NLI grounding signal on facts + CI -- the corrected facts-row number."
    ),
    "confidence_baseline_auroc": (
        "The bar to beat (0.7446) -- facts value requires beating confidence."
    ),
    "grounding_minus_confidence_delta": (
        "Signed paired delta + CI -- the additive facts value over confidence alone."
    ),
    "facts_conditional_catch_rate": (
        "Errors the grounding verifier catches that confidence misses -- the facts "
        "second-pair-of-eyes signal."
    ),
    "mcnemar_p_facts": (
        "Paired significance of the grounding-vs-confidence disagreement (not just a "
        "point estimate)."
    ),
    "facts_generalize_real_nli": (
        "BARE bool. True iff the real NLI grounding verifier materially beats 0.5 AND "
        ">= confidence baseline, leak-free -- the corrected core-mission answer. "
        "STORE AS BARE true/false."
    ),
    "real_nli_vs_proxy_delta": (
        "Did using a real model change the .334 proxy answer (0.6495)? -- the "
        "methodological correction."
    ),
    "positive_control_valid": (
        "True iff the verifier fired on a headroom-bearing corpus (confidence < 0.95) "
        "leak-free -- a null without this is uninformative."
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
    """Build the Exp 3655 terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3640 = _read_json_object(root_path / EXP3640_REL_PATH)
    exp3642 = _read_json_object(root_path / EXP3642_REL_PATH)
    exp3654 = _read_json_object(root_path / EXP3654_REL_PATH)
    corpus_path = _resolve_corpus_path(root_path, exp3640)
    rows, corpus_blocked_reason = _load_valid_v3_rows(corpus_path)
    precondition_ok, precondition_reason = real_nli_precondition(exp3654)
    if not precondition_ok or corpus_blocked_reason is not None:
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _blocked_artifact(
            root_path,
            exp3640=exp3640,
            exp3642=exp3642,
            exp3654=exp3654,
            blocked_reason=precondition_reason or corpus_blocked_reason or "blocked_unknown",
            n_examples=len(rows),
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
                allow_proxy=False
            )
        except Exception as exc:
            finished = time.perf_counter() if now_s is None else float(now_s)
            artifact = _blocked_artifact(
                root_path,
                exp3640=exp3640,
                exp3642=exp3642,
                exp3654=exp3654,
                blocked_reason=f"blocked_real_nli_checkpoint_unavailable: {type(exc).__name__}",
                n_examples=len(rows),
                started_s=start,
                finished_s=finished,
                tests_run=tests_run,
            )
            validate_artifact(artifact)
            return artifact
        grounding_scores = score_facts_rows(rows, verifier=grounding_verifier)
        nli_substrate = grounding_verifier.nli_substrate

    if "confidence_scores" in overrides:
        confidence_scores = [float(score) for score in overrides["confidence_scores"]]
    else:
        confidence_scores = [1.0 - _coerce_float(row.get("model_confidence"), 0.5) for row in rows]

    clean_labels, clean_grounding, clean_confidence = finite_label_score_triplets(
        labels,
        grounding_scores,
        confidence_scores,
    )
    if not clean_labels:
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _blocked_artifact(
            root_path,
            exp3640=exp3640,
            exp3642=exp3642,
            exp3654=exp3654,
            blocked_reason="blocked_no_finite_score_triplets",
            n_examples=0,
            started_s=start,
            finished_s=finished,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    grounding_metrics = metric_bundle(
        clean_labels,
        clean_grounding,
        n_bootstrap=n_bootstrap,
        seeds=BOOTSTRAP_SEEDS,
    )
    confidence_metrics = metric_bundle(
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
    positive_control_valid = bool(
        evidence_guard
        and exp3654.get("grounding_leak_free") is True
        and confidence_point < 0.95
        and len(clean_labels) > 0
    )
    facts_generalize = bool(
        positive_control_valid
        and grounding_point > MATERIAL_AUROC_FLOOR
        and grounding_point >= confidence_point
    )
    proxy_auroc = proxy_facts_auroc(exp3642)
    proxy_verdict = proxy_facts_verdict(exp3642)
    real_verdict = "generalizes" if facts_generalize else "domain_bound"
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "honest_verdict": terminal_verdict(
            facts_generalize_real_nli=facts_generalize,
            positive_control_valid=positive_control_valid,
        ),
        "honest_outcome": real_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nli_substrate": nli_substrate,
        "grounding_auroc_real_nli": grounding_metrics,
        "confidence_baseline_auroc": confidence_metrics,
        "grounding_minus_confidence_delta": delta_metrics,
        "facts_conditional_catch_rate": second_pair,
        "mcnemar_p_facts": second_pair["mcnemar"]["p_value"],
        "facts_generalize_real_nli": facts_generalize,
        "real_nli_vs_proxy_delta": {
            "proxy_auroc": round(float(proxy_auroc), 6),
            "real_nli_auroc": round(grounding_point, 6),
            "delta": round(grounding_point - float(proxy_auroc), 6),
            "proxy_verdict": proxy_verdict,
            "real_nli_verdict": real_verdict,
            "answer_changed": bool(proxy_verdict != real_verdict),
        },
        "positive_control_valid": positive_control_valid,
        "fixed_confidence_fpr": round(float(fixed_confidence_fpr), 6),
        "evidence_excludes_gold_answer_assert": evidence_guard,
        "score_path_answer_evidence_only": True,
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
                "grounding_auroc_real_nli present AND confidence_baseline_auroc present "
                "AND positive_control_valid == true"
            ),
            "passed": bool(grounding_metrics and confidence_metrics and positive_control_valid),
            "principle": (
                "A corrected facts verdict requires the real verifier fired on a "
                "headroom-bearing leak-free corpus vs a measured confidence baseline -- "
                "otherwise it repeats the .334 proxy limitation."
            ),
        },
        "source_artifacts": [
            str(EXP3640_REL_PATH),
            str(EXP3642_REL_PATH),
            str(EXP3654_REL_PATH),
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
    """Build and persist the Exp 3655 artifact."""

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


def score_facts_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    verifier: GroundingVerifier,
) -> list[float]:
    """Score facts rows by passing only model answer and evidence to the verifier."""

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


def real_nli_precondition(exp3654: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Return whether Exp 3654 made a usable, leak-free model-based verifier available."""

    if exp3654.get("nli_grounding_built") is not True:
        return False, "blocked_exp3654_nli_grounding_built_not_true"
    if exp3654.get("grounding_leak_free") is not True:
        return False, "blocked_exp3654_grounding_leak_free_not_true"
    substrate = str(exp3654.get("nli_substrate") or "")
    if not substrate.startswith("model_based_transformers_checkpoint:"):
        return False, "blocked_exp3654_not_model_based_real_nli"
    return True, None


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

    clean_labels, clean_scores, _ = finite_label_score_triplets(labels, scores, scores)
    negative_count = sum(1 for label in clean_labels if int(label) == 0)
    positive_count = sum(1 for label in clean_labels if int(label) == 1)
    thresholds = [math.inf] + sorted({float(score) for score in clean_scores}, reverse=True)
    best: JsonDict | None = None
    for threshold in thresholds:
        decisions = [float(score) >= float(threshold) for score in clean_scores]
        false_positive_count = sum(
            1 for label, decision in zip(clean_labels, decisions, strict=True) if label == 0 and decision
        )
        realized_fpr = false_positive_count / negative_count if negative_count else 0.0
        if realized_fpr > float(target_fpr) + 1e-12:
            continue
        caught_errors = sum(
            1 for label, decision in zip(clean_labels, decisions, strict=True) if label == 1 and decision
        )
        candidate = {
            "threshold": None if math.isinf(float(threshold)) else round(float(threshold), 6),
            "decisions": decisions,
            "realized_fpr": round(float(realized_fpr), 6),
            "false_positive_count": int(false_positive_count),
            "caught_error_count": int(caught_errors),
            "error_catch_rate": round(float(caught_errors / positive_count), 6)
            if positive_count
            else None,
        }
        if best is None or (
            int(candidate["caught_error_count"]),
            int(candidate["false_positive_count"]),
        ) > (
            int(best["caught_error_count"]),
            int(best["false_positive_count"]),
        ):
            best = candidate
    if best is None:
        return {
            "threshold": None,
            "decisions": [False for _ in clean_scores],
            "realized_fpr": 0.0,
            "false_positive_count": 0,
            "caught_error_count": 0,
            "error_catch_rate": 0.0 if positive_count else None,
        }
    return best


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
    """Compute the exact two-sided binomial McNemar p-value."""

    n = int(grounding_only) + int(confidence_only)
    if n == 0:
        return None
    k = min(int(grounding_only), int(confidence_only))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
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
            stats = facts_second_pair_of_eyes(
                arr_labels[idx].tolist(),
                arr_grounding[idx].tolist(),
                arr_confidence[idx].tolist(),
                fixed_confidence_fpr=fixed_confidence_fpr,
                n_bootstrap=0,
                seeds=(),
            )
            point = stats["point"]
            if point is None:
                continue
            value = float(point)
            seed_values.append(value)
            values.append(value)
        seed_means.append(round(float(np.mean(seed_values)), 6) if seed_values else None)
    if not values:
        return None, seed_means
    ci_low, ci_high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return [round(float(ci_low), 6), round(float(ci_high), 6)], seed_means


def terminal_verdict(
    *,
    facts_generalize_real_nli: bool,
    positive_control_valid: bool,
) -> str:
    """Select the Exp 3655 terminal verdict."""

    if not positive_control_valid:
        return BLOCKED_VERDICT
    if facts_generalize_real_nli:
        return GENERALIZES_VERDICT
    return DOMAIN_BOUND_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3655 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    for field in ("facts_generalize_real_nli", "positive_control_valid"):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    if not isinstance(artifact.get("field_principles"), Mapping):
        raise ValueError("field_principles must be present")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(artifact["field_principles"]):
        raise ValueError("field_principles must cover all required fields")
    if artifact.get("positive_control_valid") is True:
        for field in (
            "grounding_auroc_real_nli",
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


def proxy_facts_auroc(exp3642: Mapping[str, Any]) -> float:
    """Read the Exp 3642 proxy facts AUROC, falling back to the .334 headline."""

    facts = _facts_row(exp3642)
    ensemble = facts.get("ensemble_auroc") if isinstance(facts, Mapping) else None
    if isinstance(ensemble, Mapping) and ensemble.get("point") is not None:
        return round(float(ensemble["point"]), 6)
    return PROXY_BASELINE_AUROC


def proxy_facts_verdict(exp3642: Mapping[str, Any]) -> str:
    """Return the Exp 3642 proxy verdict as a compact outcome."""

    facts = _facts_row(exp3642)
    verdict = str(facts.get("domain_verdict") or "domain_bound") if facts else "domain_bound"
    return "generalizes" if verdict == "generalizes" else "domain_bound"


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable short checksum over measured inputs and scores."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _blocked_artifact(
    root: Path,
    *,
    exp3640: Mapping[str, Any],
    exp3642: Mapping[str, Any],
    exp3654: Mapping[str, Any],
    blocked_reason: str,
    n_examples: int,
    started_s: float,
    finished_s: float,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    confidence_point = _coerce_float(
        exp3640.get("confidence_baseline_auroc_on_corpus"),
        math.nan,
    )
    confidence_metric = None
    if math.isfinite(confidence_point):
        confidence_metric = {
            "point": round(float(confidence_point), 6),
            "ci95": None,
            "n": int(n_examples),
            "bootstrap_seeds": list(BOOTSTRAP_SEEDS),
        }
    artifact: JsonDict = {
        "honest_verdict": BLOCKED_VERDICT,
        "honest_outcome": "blocked",
        "blocked_reason": blocked_reason,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nli_substrate": str(exp3654.get("nli_substrate") or "not_available"),
        "grounding_auroc_real_nli": None,
        "confidence_baseline_auroc": confidence_metric,
        "grounding_minus_confidence_delta": None,
        "facts_conditional_catch_rate": None,
        "mcnemar_p_facts": None,
        "facts_generalize_real_nli": False,
        "real_nli_vs_proxy_delta": {
            "proxy_auroc": round(float(proxy_facts_auroc(exp3642)), 6),
            "real_nli_auroc": None,
            "delta": None,
            "proxy_verdict": proxy_facts_verdict(exp3642),
            "real_nli_verdict": "blocked",
            "answer_changed": None,
        },
        "positive_control_valid": False,
        "fixed_confidence_fpr": DEFAULT_FIXED_CONFIDENCE_FPR,
        "evidence_excludes_gold_answer_assert": False,
        "score_path_answer_evidence_only": True,
        "n_examples": 0,
        "sample_size_rigor_met": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "blocked_reason": blocked_reason,
                "exp3654": {
                    "nli_grounding_built": exp3654.get("nli_grounding_built"),
                    "grounding_leak_free": exp3654.get("grounding_leak_free"),
                    "nli_substrate": exp3654.get("nli_substrate"),
                },
            }
        ),
        "duration_s": round(max(0.0, finished_s - started_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "grounding_auroc_real_nli present AND confidence_baseline_auroc present "
                "AND positive_control_valid == true"
            ),
            "passed": False,
            "principle": (
                "A corrected facts verdict requires the real verifier fired on a "
                "headroom-bearing leak-free corpus vs a measured confidence baseline -- "
                "otherwise it repeats the .334 proxy limitation."
            ),
        },
        "source_artifacts": [
            str(EXP3640_REL_PATH),
            str(EXP3642_REL_PATH),
            str(EXP3654_REL_PATH),
        ],
        "corpus_path_used": _display_path(root, _resolve_corpus_path(root, exp3640)),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    return artifact


def _facts_row(exp3642: Mapping[str, Any]) -> Mapping[str, Any]:
    table = exp3642.get("generalization_table")
    if not isinstance(table, Mapping):
        return {}
    facts = table.get("facts")
    return facts if isinstance(facts, Mapping) else {}


def _load_valid_v3_rows(path: Path) -> tuple[list[JsonDict], str | None]:
    if not path.exists():
        return [], "blocked_missing_v3_facts_corpus"
    rows = _read_jsonl(path)
    if not rows:
        return [], "blocked_empty_v3_facts_corpus"
    for idx, row in enumerate(rows):
        missing = [field for field in REQUIRED_CORPUS_FIELDS if field not in row]
        if missing:
            return [], f"blocked_v3_facts_corpus_schema_row_{idx}_missing_{'_'.join(missing)}"
    return rows, None


def _resolve_corpus_path(root: Path, exp3640: Mapping[str, Any]) -> Path:
    corpus_path = exp3640.get("corpus_path_used")
    if isinstance(corpus_path, str) and corpus_path:
        return _repo_path(root, Path(corpus_path))
    return root / DEFAULT_CORPUS_REL_PATH


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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


def _coerce_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


__all__ = [
    "BOOTSTRAP_SEEDS",
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "conditional_catch_rate",
    "decisions_at_fpr",
    "facts_second_pair_of_eyes",
    "score_facts_rows",
    "validate_artifact",
    "write_artifact",
]

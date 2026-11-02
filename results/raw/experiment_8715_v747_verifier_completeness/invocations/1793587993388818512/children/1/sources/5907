"""Exp 3643 additive second-pair-of-eyes remeasurement.

Spec: REQ-VERIFY-3643, SCENARIO-VERIFY-3643.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.phase3.p01_energy_vote_scoring import mcnemar_exact
from carnot.verify import corrected_cross_domain_remeasurement_v4 as exp3642


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3643_additivity_second_pair_of_eyes_v4.json")
EXP3642_REL_PATH = Path("results/experiment_3642_corrected_cross_domain_remeasurement_v4.json")
RANDOM_SEED = 3643
FIXED_CONFIDENCE_BASELINE_FPR = 0.10
MATERIAL_AUROC_LIFT = 0.01
MATERIAL_RECALL_LIFT = 0.01
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached corpora; no LLM load)."
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "code_conditional_catch_rate_ensemble_over_confidence",
    "factual_conditional_catch_rate_ensemble_over_confidence",
    "mcnemar_p_code",
    "mcnemar_p_factual",
    "fused_detector_auroc",
    "fusion_beats_confidence_alone",
    "second_pair_of_eyes_real",
    "n_errors_per_domain",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "Scores cached corpora through the corrected Exp 3642 substrate; no live LLM load.",
    "code_conditional_catch_rate_ensemble_over_confidence": "Errors the ensemble catches that confidence misses -- the second-pair-of-eyes signal for code.",
    "factual_conditional_catch_rate_ensemble_over_confidence": "Same for facts -- the core-motivation product signal.",
    "mcnemar_p_code": "Paired significance of the ensemble-vs-confidence disagreement (not just a point estimate).",
    "mcnemar_p_factual": "Same for facts.",
    "fused_detector_auroc": "ensemble OR confidence -- the deployable detector's headline number.",
    "fusion_beats_confidence_alone": "True iff the fused detector materially beats confidence -- the real 'is the verifier worth adding' answer.",
    "second_pair_of_eyes_real": "True iff the ensemble catches a significant set confidence misses in >=1 domain with McNemar p<0.05 -- re-derived honestly vs the contaminated prior claim.",
    "n_errors_per_domain": "Sample-size rigor for the conditional-catch claim.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class DomainScores:
    """Paired labels and detector scores for one corrected non-math row."""

    domain: str
    labels: list[int]
    ensemble_scores: list[float]
    confidence_scores: list[float]


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    fixed_fpr: float = FIXED_CONFIDENCE_BASELINE_FPR,
    score_overrides: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3643 terminal artifact from corrected Exp 3642 rows."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    upstream = _read_json_object(root_path / EXP3642_REL_PATH)
    overrides = score_overrides or {}
    if upstream.get("at_least_one_nonmath_row_ran") is not True:
        domain_metrics = {
            "code": not_measured_domain("code", "exp3642_no_runnable_nonmath_row"),
            "facts": not_measured_domain("facts", "exp3642_no_runnable_nonmath_row"),
        }
        measured_scores: list[DomainScores] = []
    else:
        domain_metrics, measured_scores = measure_domains(root_path, upstream, overrides, fixed_fpr)

    fusion = pooled_fusion_metrics(measured_scores, fixed_fpr)
    second_pair_real = any(
        domain_is_second_pair_real(metrics) for metrics in domain_metrics.values()
    )
    fusion_beats = bool(fusion["fusion_beats_confidence_alone"])
    if upstream.get("at_least_one_nonmath_row_ran") is not True:
        verdict = "complete: blocked_no_nonmath_row_ran"
    elif second_pair_real and fusion_beats:
        verdict = "complete: ensemble_additive_to_confidence_second_pair_of_eyes_real_fusion_wins"
    else:
        verdict = "complete: ensemble_redundant_with_confidence_no_additive_value_value_prop_weak"

    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "fixed_confidence_baseline_fpr": fixed_fpr,
        "per_domain_additivity": domain_metrics,
        "code_conditional_catch_rate_ensemble_over_confidence": domain_metrics["code"][
            "conditional_catch_rate_ensemble_over_confidence"
        ],
        "factual_conditional_catch_rate_ensemble_over_confidence": domain_metrics["facts"][
            "conditional_catch_rate_ensemble_over_confidence"
        ],
        "mcnemar_p_code": domain_metrics["code"]["mcnemar_p"],
        "mcnemar_p_factual": domain_metrics["facts"]["mcnemar_p"],
        "fused_detector_auroc": fusion["fused_detector_auroc"],
        "confidence_alone_auroc": fusion["confidence_alone_auroc"],
        "fused_detector_recall_at_fixed_fpr": fusion["fused_detector_recall_at_fixed_fpr"],
        "confidence_recall_at_fixed_fpr": fusion["confidence_recall_at_fixed_fpr"],
        "fusion_auroc_delta": fusion["fusion_auroc_delta"],
        "fusion_recall_delta_at_fixed_fpr": fusion["fusion_recall_delta_at_fixed_fpr"],
        "fusion_beats_confidence_alone": fusion_beats,
        "second_pair_of_eyes_real": second_pair_real,
        "n_errors_per_domain": {
            domain: metrics["n_errors"] if metrics["status"] == "measured" else "not_measured"
            for domain, metrics in domain_metrics.items()
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(domain_metrics, fusion),
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": "fused_detector_auroc present AND fusion_beats_confidence_alone present",
            "passed": True,
            "principle": "Product value is conditional, additive coverage measured with a paired significance test -- a higher AUROC alone is not the second-pair-of-eyes claim.",
        },
        "fused_detector_method": "max_negative-tail-calibrated ensemble/confidence score, equivalent to an OR over detector alert budgets.",
        "source_artifact": str(EXP3642_REL_PATH),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    score_overrides: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Persist the Exp 3643 terminal artifact."""

    root_path = Path(root)
    output = exp3642._repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        score_overrides=score_overrides,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def measure_domains(
    root: Path,
    upstream: Mapping[str, Any],
    score_overrides: Mapping[str, Mapping[str, Sequence[float]]],
    fixed_fpr: float,
) -> tuple[dict[str, JsonDict], list[DomainScores]]:
    """Measure every non-math Exp 3642 row that actually ran."""

    table = upstream.get("generalization_table") or {}
    domain_metrics: dict[str, JsonDict] = {}
    measured_scores: list[DomainScores] = []
    for domain in ("code", "facts"):
        row = table.get(domain) if isinstance(table, Mapping) else None
        if not isinstance(row, Mapping) or row.get("ran_or_blocked") != "ran":
            domain_metrics[domain] = not_measured_domain(domain, "exp3642_row_not_ran")
            continue
        scores = load_domain_scores(root, domain, score_overrides.get(domain, {}))
        measured_scores.append(scores)
        domain_metrics[domain] = compute_domain_metrics(scores, fixed_fpr)
    return domain_metrics, measured_scores


def load_domain_scores(
    root: Path,
    domain: str,
    overrides: Mapping[str, Sequence[float]],
) -> DomainScores:
    """Load labels and paired scores through the same corrected rows as Exp 3642."""

    if domain == "code":
        artifact = exp3642._read_json_object(root / exp3642.EXP3641_REL_PATH)
        rows = exp3642._read_jsonl(exp3642._repo_path(root, Path(str(artifact["code_corpus_path"]))))
        labels = [0 if bool(row.get("label")) else 1 for row in rows]
        ensemble_scores = _score_or_override(
            overrides,
            "ensemble_scores",
            lambda: exp3642.score_code_rows(rows, root),
        )
        confidence_scores = _score_or_override(
            overrides,
            "confidence_scores",
            lambda: exp3642.score_code_confidence(rows),
        )
        return DomainScores(domain, labels, ensemble_scores, confidence_scores)
    artifact = exp3642._read_json_object(root / exp3642.EXP3640_REL_PATH)
    rows = exp3642._read_jsonl(exp3642._repo_path(root, Path(str(artifact["corpus_path_used"]))))
    labels = [int(bool(row.get("is_hallucination"))) for row in rows]
    ensemble_scores = _score_or_override(
        overrides,
        "ensemble_scores",
        lambda: exp3642.score_fact_rows(rows),
    )
    confidence_scores = _score_or_override(
        overrides,
        "confidence_scores",
        lambda: [1.0 - exp3642._coerce_float(row.get("model_confidence"), 0.5) for row in rows],
    )
    return DomainScores(domain, labels, ensemble_scores, confidence_scores)


def compute_domain_metrics(scores: DomainScores, fixed_fpr: float) -> JsonDict:
    """Compute per-domain recall, conditional catch, and paired significance."""

    labels, ensemble, confidence = exp3642.finite_label_score_triplets(
        scores.labels,
        scores.ensemble_scores,
        scores.confidence_scores,
    )
    confidence_gate = predictions_at_fixed_fpr(labels, confidence, fixed_fpr)
    ensemble_gate = predictions_at_fixed_fpr(labels, ensemble, fixed_fpr)
    summary = conditional_catch_summary(
        confidence_gate["caught_errors"],
        ensemble_gate["caught_errors"],
    )
    return {
        "status": "measured",
        "domain": scores.domain,
        "fixed_fpr": fixed_fpr,
        "confidence_threshold": confidence_gate["threshold"],
        "ensemble_threshold": ensemble_gate["threshold"],
        "confidence_actual_fpr": confidence_gate["fpr"],
        "ensemble_actual_fpr": ensemble_gate["fpr"],
        "confidence_recall": confidence_gate["recall"],
        "ensemble_recall": ensemble_gate["recall"],
        "conditional_catch_rate_ensemble_over_confidence": summary[
            "conditional_catch_rate_ensemble_over_confidence"
        ],
        "conditional_catch_rate_confidence_over_ensemble": summary[
            "conditional_catch_rate_confidence_over_ensemble"
        ],
        "conditional_catch_rate_ci95": summary["conditional_catch_rate_ci95"],
        "ensemble_only_count": summary["ensemble_only_count"],
        "confidence_only_count": summary["confidence_only_count"],
        "mcnemar_p": summary["mcnemar_p"],
        "n_errors": summary["n_errors"],
        "n_examples": len(labels),
    }


def predictions_at_fixed_fpr(
    labels: Sequence[int],
    scores: Sequence[float],
    fixed_fpr: float,
) -> JsonDict:
    """Select the highest-recall threshold whose false-positive rate stays in budget."""

    clean_labels, clean_scores = exp3642.finite_label_scores(labels, scores)
    positives = sum(1 for label in clean_labels if label == 1)
    negatives = len(clean_labels) - positives
    if positives == 0 or negatives == 0:
        return {
            "threshold": None,
            "fpr": 0.0,
            "recall": 0.0,
            "predictions": [False for _ in clean_labels],
            "caught_errors": [],
        }
    thresholds = sorted(set(clean_scores), reverse=True) + [math.inf]
    best: JsonDict | None = None
    for threshold in thresholds:
        predictions = [score >= threshold for score in clean_scores]
        fp = sum(1 for label, pred in zip(clean_labels, predictions, strict=False) if label == 0 and pred)
        tp = sum(1 for label, pred in zip(clean_labels, predictions, strict=False) if label == 1 and pred)
        fpr = fp / negatives
        recall = tp / positives
        if fpr <= fixed_fpr + 1e-12 and (
            best is None or recall > best["recall"] or (recall == best["recall"] and fpr > best["fpr"])
        ):
            best = {
                "threshold": None if math.isinf(threshold) else round(float(threshold), 12),
                "fpr": round(float(fpr), 6),
                "recall": round(float(recall), 6),
                "predictions": predictions,
                "caught_errors": [
                    bool(pred)
                    for label, pred in zip(clean_labels, predictions, strict=False)
                    if label == 1
                ],
            }
    if best is None:
        raise ValueError("no fixed-FPR threshold could be selected")
    return best


def conditional_catch_summary(
    confidence_caught: Sequence[bool],
    ensemble_caught: Sequence[bool],
) -> JsonDict:
    """Measure the paired error cases each detector catches beyond the other."""

    if len(confidence_caught) != len(ensemble_caught):
        raise ValueError("confidence and ensemble caught lists must have the same length")
    n_errors = len(confidence_caught)
    if n_errors == 0:
        return {
            "baseline_recall": 0.0,
            "ensemble_recall": 0.0,
            "conditional_catch_rate_ensemble_over_confidence": 0.0,
            "conditional_catch_rate_confidence_over_ensemble": 0.0,
            "conditional_catch_rate_ci95": [None, None],
            "confidence_only_count": 0,
            "ensemble_only_count": 0,
            "mcnemar_p": 1.0,
            "n_errors": 0,
        }
    confidence_misses = sum(1 for caught in confidence_caught if not caught)
    ensemble_misses = sum(1 for caught in ensemble_caught if not caught)
    ensemble_only = sum(
        1 for conf, ens in zip(confidence_caught, ensemble_caught, strict=False) if (not conf) and ens
    )
    confidence_only = sum(
        1 for conf, ens in zip(confidence_caught, ensemble_caught, strict=False) if conf and (not ens)
    )
    return {
        "baseline_recall": round(sum(confidence_caught) / n_errors, 6),
        "ensemble_recall": round(sum(ensemble_caught) / n_errors, 6),
        "conditional_catch_rate_ensemble_over_confidence": _safe_rate(
            ensemble_only,
            confidence_misses,
        ),
        "conditional_catch_rate_confidence_over_ensemble": _safe_rate(
            confidence_only,
            ensemble_misses,
        ),
        "conditional_catch_rate_ci95": wilson_ci(ensemble_only, confidence_misses),
        "confidence_only_count": confidence_only,
        "ensemble_only_count": ensemble_only,
        "mcnemar_p": round(float(mcnemar_exact(list(confidence_caught), list(ensemble_caught))), 12),
        "n_errors": n_errors,
    }


def pooled_fusion_metrics(
    measured_scores: Sequence[DomainScores],
    fixed_fpr: float,
) -> JsonDict:
    """Compute the calibrated deployable fusion score across measured domains."""

    labels: list[int] = []
    confidence_calibrated: list[float] = []
    fused_calibrated: list[float] = []
    for scores in measured_scores:
        clean_labels, ensemble, confidence = exp3642.finite_label_score_triplets(
            scores.labels,
            scores.ensemble_scores,
            scores.confidence_scores,
        )
        ensemble_tail = negative_tail_calibrated_scores(clean_labels, ensemble)
        confidence_tail = negative_tail_calibrated_scores(clean_labels, confidence)
        labels.extend(clean_labels)
        confidence_calibrated.extend(confidence_tail)
        fused_calibrated.extend(
            max(conf_score, ens_score)
            for conf_score, ens_score in zip(confidence_tail, ensemble_tail, strict=False)
        )
    if not labels:
        return {
            "fused_detector_auroc": None,
            "confidence_alone_auroc": None,
            "fused_detector_recall_at_fixed_fpr": None,
            "confidence_recall_at_fixed_fpr": None,
            "fusion_auroc_delta": None,
            "fusion_recall_delta_at_fixed_fpr": None,
            "fusion_beats_confidence_alone": False,
        }
    fused_auroc = round(exp3642.tie_aware_auroc(labels, fused_calibrated), 6)
    confidence_auroc = round(exp3642.tie_aware_auroc(labels, confidence_calibrated), 6)
    fused_gate = predictions_at_fixed_fpr(labels, fused_calibrated, fixed_fpr)
    confidence_gate = predictions_at_fixed_fpr(labels, confidence_calibrated, fixed_fpr)
    auroc_delta = round(fused_auroc - confidence_auroc, 6)
    recall_delta = round(fused_gate["recall"] - confidence_gate["recall"], 6)
    return {
        "fused_detector_auroc": fused_auroc,
        "confidence_alone_auroc": confidence_auroc,
        "fused_detector_recall_at_fixed_fpr": fused_gate["recall"],
        "confidence_recall_at_fixed_fpr": confidence_gate["recall"],
        "fusion_auroc_delta": auroc_delta,
        "fusion_recall_delta_at_fixed_fpr": recall_delta,
        "fusion_beats_confidence_alone": bool(
            auroc_delta >= MATERIAL_AUROC_LIFT and recall_delta >= MATERIAL_RECALL_LIFT
        ),
    }


def negative_tail_calibrated_scores(
    labels: Sequence[int],
    scores: Sequence[float],
) -> list[float]:
    """Map scores onto their extremeness relative to correct examples."""

    clean_labels, clean_scores = exp3642.finite_label_scores(labels, scores)
    negatives = [score for label, score in zip(clean_labels, clean_scores, strict=False) if label == 0]
    if not negatives:
        return [0.0 for _ in clean_scores]
    calibrated = []
    for score in clean_scores:
        less = sum(1 for neg in negatives if neg < score)
        tied = sum(1 for neg in negatives if neg == score)
        calibrated.append(round((less + 0.5 * tied) / len(negatives), 12))
    return calibrated


def not_measured_domain(domain: str, reason: str) -> JsonDict:
    """Return a domain placeholder that cannot accidentally imply a zero result."""

    return {
        "status": "not_measured",
        "domain": domain,
        "reason": reason,
        "conditional_catch_rate_ensemble_over_confidence": "not_measured",
        "conditional_catch_rate_confidence_over_ensemble": "not_measured",
        "conditional_catch_rate_ci95": "not_measured",
        "confidence_recall": "not_measured",
        "ensemble_recall": "not_measured",
        "confidence_only_count": "not_measured",
        "ensemble_only_count": "not_measured",
        "mcnemar_p": "not_measured",
        "n_errors": "not_measured",
    }


def domain_is_second_pair_real(metrics: Mapping[str, Any]) -> bool:
    """Return true when the ensemble has a significant one-way error-catch win."""

    return bool(
        metrics.get("status") == "measured"
        and float(metrics.get("conditional_catch_rate_ensemble_over_confidence") or 0.0) > 0.0
        and int(metrics.get("ensemble_only_count") or 0) > int(metrics.get("confidence_only_count") or 0)
        and float(metrics.get("mcnemar_p") or 1.0) < 0.05
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3643 terminal schema."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    for field in ("fusion_beats_confidence_alone", "second_pair_of_eyes_real"):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    if not isinstance(artifact.get("n_errors_per_domain"), Mapping):
        raise ValueError("n_errors_per_domain must be present")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def wilson_ci(count: int, denominator: int) -> list[float | None]:
    """Return a stable 95% binomial interval for the conditional catch rate."""

    if denominator <= 0:
        return [None, None]
    z = 1.959963984540054
    phat = count / denominator
    denom = 1.0 + z * z / denominator
    center = (phat + z * z / (2.0 * denominator)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) / denominator) + z * z / (4.0 * denominator**2)) / denom
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def reproducibility_checksum(
    domain_metrics: Mapping[str, Any],
    fusion_metrics: Mapping[str, Any],
) -> str:
    """Hash only deterministic measurement values so wall-clock time cannot drift it."""

    payload = {
        "domain_metrics": domain_metrics,
        "fusion_metrics": fusion_metrics,
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _score_or_override(
    overrides: Mapping[str, Sequence[float]],
    key: str,
    scorer: Any,
) -> list[float]:
    if key in overrides:
        return [float(score) for score in overrides[key]]
    return [float(score) for score in scorer()]


def _safe_rate(count: int, denominator: int) -> float:
    return round(count / denominator, 6) if denominator else 0.0


def _read_json_object(path: Path) -> JsonDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data

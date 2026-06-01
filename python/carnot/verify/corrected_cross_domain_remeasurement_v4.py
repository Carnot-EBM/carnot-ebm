"""Exp 3642 corrected cross-domain verifier remeasurement.

Spec: REQ-VERIFY-3642, SCENARIO-VERIFY-3642.
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

from carnot.verify.retrieval_nli_grounding_verifier import RetrievalNLIGroundingVerifier


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3642_corrected_cross_domain_remeasurement_v4.json")
EXP2837_REL_PATH = Path("results/experiment_2837_fover_memory_leakage_v3.json")
EXP3640_REL_PATH = Path("results/experiment_3640_build_factual_corpus_v3.json")
EXP3641_REL_PATH = Path("results/experiment_3641_code_corpus_verifiers_fire_transfer_v3.json")
BOOTSTRAP_SEEDS = (3642, 3643, 3644)
RANDOM_SEED = 3642
FROZEN_MATH_AUROC = 0.9131
MATERIAL_AUROC_FLOOR = 0.55
NLI_SUBSTRATE = "disclosed_text_statistical_proxy_token_support_no_gold_or_label_input"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached corpora + a small NLI checkpoint; not the FoVer LLM)."
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "nli_substrate",
    "evidence_excludes_gold_answer_assert",
    "generalization_table",
    "math_ensemble_auroc",
    "code_generalizes",
    "facts_generalize",
    "grounding_verifier_auroc",
    "grounding_leak_free",
    "positive_control_valid",
    "at_least_one_nonmath_row_ran",
    "n_examples_per_domain",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "Scores cached corpora plus the disclosed grounding substrate; does not rerun the FoVer LLM.",
    "nli_substrate": "Declares model-based NLI vs disclosed text-statistical proxy -- verifier-authenticity honesty.",
    "evidence_excludes_gold_answer_assert": "Asserts no separate gold-answer field is available to the grounding scorer -- the explicit guard against the exp3587 leak.",
    "generalization_table": "domain -> {ensemble_auroc, confidence_auroc, delta, ci, ran_or_blocked} -- the milestone's central evidence object, measured fairly.",
    "math_ensemble_auroc": "The frozen FoVer math headline (0.9131) re-stated as the row baseline.",
    "code_generalizes": "Code re-tested with execution-applicable verifiers that fired -- distinguishes a wiring bug from a real code limitation.",
    "facts_generalize": "Facts re-tested with the leak-free grounding verifier on a corpus with held-out evidence -- the core-motivation result.",
    "grounding_verifier_auroc": "The facts-row grounding signal + CI; an exact 1.0 is a leak, not a win.",
    "grounding_leak_free": "True iff evidence excludes a separate gold-answer field and grounding AUROC < 0.99 -- gates trust in the factual row.",
    "positive_control_valid": "BARE bool. True iff BOTH non-math rows actually RAN and each has confidence-baseline AUROC < 0.95.",
    "at_least_one_nonmath_row_ran": "BARE bool. True iff >=1 non-math row actually RAN with confidence-baseline headroom.",
    "n_examples_per_domain": "Sample-size rigor per row.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_bootstrap: int = 200,
    score_overrides: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the corrected Exp 3642 terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    overrides = score_overrides or {}
    math_artifact = _read_json_object(root_path / EXP2837_REL_PATH)
    facts_artifact = _read_json_object(root_path / EXP3640_REL_PATH)
    code_artifact = _read_json_object(root_path / EXP3641_REL_PATH)

    table: JsonDict = {
        "math": build_math_row(math_artifact),
        "code": build_code_row(root_path, code_artifact, overrides.get("code", {}), n_bootstrap),
        "facts": build_facts_row(
            root_path, facts_artifact, overrides.get("facts", {}), n_bootstrap
        ),
    }
    evidence_guard = bool(table["facts"].get("evidence_excludes_gold_answer_assert"))
    grounding_metrics = (
        table["facts"].get("ensemble_auroc") if table["facts"]["ran_or_blocked"] == "ran" else None
    )
    grounding_point = (
        grounding_metrics.get("point") if isinstance(grounding_metrics, Mapping) else None
    )
    grounding_leak_free = bool(
        table["facts"]["ran_or_blocked"] == "ran"
        and evidence_guard
        and grounding_point is not None
        and float(grounding_point) < 0.99
    )
    code_generalizes = row_generalizes_with_headroom(table["code"])
    facts_generalize = bool(row_generalizes_with_headroom(table["facts"]) and grounding_leak_free)
    positive_control_valid = bool(
        row_ran_with_headroom(table["code"]) and row_ran_with_headroom(table["facts"])
    )
    at_least_one_nonmath_row_ran = bool(
        row_ran_with_headroom(table["code"]) or row_ran_with_headroom(table["facts"])
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "honest_verdict": terminal_verdict(
            code_generalizes=code_generalizes,
            facts_generalize=facts_generalize,
            positive_control_valid=positive_control_valid,
            code_ran=table["code"]["ran_or_blocked"] == "ran",
            facts_ran=table["facts"]["ran_or_blocked"] == "ran",
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nli_substrate": NLI_SUBSTRATE,
        "evidence_excludes_gold_answer_assert": evidence_guard,
        "generalization_table": table,
        "math_ensemble_auroc": FROZEN_MATH_AUROC,
        "code_generalizes": code_generalizes,
        "facts_generalize": facts_generalize,
        "grounding_verifier_auroc": grounding_metrics,
        "grounding_leak_free": grounding_leak_free,
        "positive_control_valid": positive_control_valid,
        "at_least_one_nonmath_row_ran": at_least_one_nonmath_row_ran,
        "n_examples_per_domain": {
            domain: int(row.get("n_examples") or 0) for domain, row in table.items()
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(table),
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": "generalization_table present AND math_ensemble_auroc present",
            "passed": True,
            "principle": "The centerpiece always lands the math row plus the table; bare flags record downstream gates.",
        },
        "source_artifacts": [str(EXP2837_REL_PATH), str(EXP3640_REL_PATH), str(EXP3641_REL_PATH)],
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
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3642 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_math_row(exp2837: Mapping[str, Any]) -> JsonDict:
    """Restate the frozen Exp 2837 math row without rerunning FoVer."""

    ensemble_ci = exp2837.get("condition_a_production_auroc_ci95")
    baseline_ci = exp2837.get("condition_b_architecture_only_auroc_ci95")
    delta_ci = exp2837.get("learning_contribution_ci95")
    baseline_point = _round_or_none(exp2837.get("condition_b_architecture_only_auroc_mean"))
    return {
        "domain": "math",
        "ran_or_blocked": "ran",
        "source": "results/experiment_2837_fover_memory_leakage_v3.json",
        "ensemble_auroc": _metric_from_frozen(FROZEN_MATH_AUROC, ensemble_ci),
        "confidence_auroc": _metric_from_frozen(baseline_point, baseline_ci),
        "delta": _metric_from_frozen(
            _round_or_none(exp2837.get("learning_contribution")),
            delta_ci,
        ),
        "domain_verdict": "generalizes",
        "confidence_baseline_source": "exp2837_architecture_only_control_for_frozen_math_comparator",
        "class_balance": {
            "positive_errors": None,
            "negative_correct": None,
        },
        "n_examples": int(exp2837.get("n_examples") or 1000),
        "headroom": bool(baseline_point is not None and baseline_point < 0.95),
    }


def build_code_row(
    root: Path,
    code_artifact: Mapping[str, Any],
    overrides: Mapping[str, Sequence[float]],
    n_bootstrap: int,
) -> JsonDict:
    """Build the code row only when Exp 3641 says code verifiers fired."""

    if code_artifact.get("code_verifiers_fire") is not True:
        return blocked_row("code", "blocked_code_verifiers")
    corpus_path = code_artifact.get("code_corpus_path")
    if not isinstance(corpus_path, str) or not corpus_path:
        return blocked_row("code", "blocked_code_corpus")
    rows = _read_jsonl(_repo_path(root, Path(corpus_path)))
    if not rows:
        return blocked_row("code", "blocked_code_corpus")
    labels = [0 if bool(row.get("label")) else 1 for row in rows]
    if "ensemble_scores" in overrides:
        ensemble_scores = [float(score) for score in overrides["ensemble_scores"]]
    else:
        ensemble_scores = score_code_rows(rows, root)
    if "confidence_scores" in overrides:
        confidence_scores = [float(score) for score in overrides["confidence_scores"]]
    else:
        confidence_scores = score_code_confidence(rows)
    return ran_row("code", labels, ensemble_scores, confidence_scores, n_bootstrap)


def build_facts_row(
    root: Path,
    facts_artifact: Mapping[str, Any],
    overrides: Mapping[str, Sequence[float]],
    n_bootstrap: int,
) -> JsonDict:
    """Build the facts row only from a validated evidence-bearing corpus."""

    if facts_artifact.get("facts_corpus_validated") is not True:
        return blocked_row("facts", "blocked_facts_corpus")
    corpus_path = facts_artifact.get("corpus_path_used")
    if not isinstance(corpus_path, str) or not corpus_path:
        return blocked_row("facts", "blocked_facts_corpus")
    rows = _read_jsonl(_repo_path(root, Path(corpus_path)))
    if not rows:
        return blocked_row("facts", "blocked_facts_corpus")
    labels = [int(bool(row.get("is_hallucination"))) for row in rows]
    if "ensemble_scores" in overrides:
        ensemble_scores = [float(score) for score in overrides["ensemble_scores"]]
    else:
        ensemble_scores = score_fact_rows(rows)
    if "confidence_scores" in overrides:
        confidence_scores = [float(score) for score in overrides["confidence_scores"]]
    else:
        confidence_scores = [1.0 - _coerce_float(row.get("model_confidence"), 0.5) for row in rows]
    row = ran_row("facts", labels, ensemble_scores, confidence_scores, n_bootstrap)
    row["evidence_excludes_gold_answer_assert"] = evidence_excludes_gold_answer(rows)
    row["nli_substrate"] = NLI_SUBSTRATE
    return row


def ran_row(
    domain: str,
    labels: Sequence[int],
    ensemble_scores: Sequence[float],
    confidence_scores: Sequence[float],
    n_bootstrap: int,
) -> JsonDict:
    """Return one scored generalization-table row."""

    clean_labels, clean_ensemble, clean_confidence = finite_label_score_triplets(
        labels,
        ensemble_scores,
        confidence_scores,
    )
    ensemble_metrics = metric_bundle(clean_labels, clean_ensemble, n_bootstrap=n_bootstrap)
    confidence_metrics = metric_bundle(clean_labels, clean_confidence, n_bootstrap=n_bootstrap)
    delta_metrics = paired_delta_bundle(
        clean_labels,
        clean_ensemble,
        clean_confidence,
        n_bootstrap=n_bootstrap,
    )
    verdict = classify_domain(ensemble_metrics, confidence_metrics)
    positives = int(sum(1 for label in clean_labels if int(label) == 1))
    return {
        "domain": domain,
        "ran_or_blocked": "ran",
        "ensemble_auroc": ensemble_metrics,
        "confidence_auroc": confidence_metrics,
        "delta": delta_metrics,
        "domain_verdict": verdict,
        "class_balance": {
            "positive_errors": positives,
            "negative_correct": len(clean_labels) - positives,
        },
        "n_examples": len(clean_labels),
        "headroom": row_headroom(confidence_metrics),
    }


def blocked_row(domain: str, reason: str) -> JsonDict:
    """Return a blocked row without synthetic chance metrics."""

    return {
        "domain": domain,
        "ran_or_blocked": "blocked",
        "blocked_reason": reason,
        "ensemble_auroc": None,
        "confidence_auroc": None,
        "delta": None,
        "domain_verdict": "blocked",
        "class_balance": {
            "positive_errors": 0,
            "negative_correct": 0,
        },
        "n_examples": 0,
        "headroom": False,
    }


def score_fact_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    verifier: Any | None = None,
) -> list[float]:
    """Score factual rows by passing only model answer and evidence to verifier."""

    grounding_verifier = verifier or RetrievalNLIGroundingVerifier()
    scores = []
    for row in rows:
        model_answer = str(row.get("answer") or "")
        evidence_passage = str(row.get("evidence_passage") or "")
        scores.append(float(grounding_verifier.verify(model_answer, evidence_passage)))
    return scores


def score_code_rows(rows: Sequence[Mapping[str, Any]], root: Path) -> list[float]:
    """Score code rows with the execution-applicable Exp 3641 verifier set."""

    from carnot.verify import code_corpus_verifiers_fire_transfer_v3 as code_transfer

    execution = code_transfer.score_execution_verifiers(
        rows,
        root,
        verifier_imports=code_transfer.import_configured_verifiers(),
        score_overrides={},
    )
    return [float(score) for score in execution["scores"]]


def score_code_confidence(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Reuse Exp 3641's cached confidence/self-consistency baseline."""

    from carnot.verify import code_corpus_verifiers_fire_transfer_v3 as code_transfer

    return [
        float(score) for score in code_transfer.score_confidence_baseline(rows, score_overrides={})
    ]


def metric_bundle(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bootstrap: int = 200,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
) -> JsonDict:
    """Return AUROC point estimate and deterministic bootstrap CI."""

    clean_labels, clean_scores = finite_label_scores(labels, scores)
    if not clean_scores:
        return empty_metric_bundle(seeds)
    point = tie_aware_auroc(clean_labels, clean_scores)
    boot_values: list[float] = []
    seed_means: list[float] = []
    arr_labels = np.asarray(clean_labels, dtype=np.int64)
    arr_scores = np.asarray(clean_scores, dtype=np.float64)
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(arr_labels), size=len(arr_labels))
            value = tie_aware_auroc(arr_labels[idx], arr_scores[idx])
            values.append(value)
            boot_values.append(value)
        seed_means.append(round(float(np.mean(values)), 6))
    ci_low, ci_high = np.percentile(np.asarray(boot_values, dtype=np.float64), [2.5, 97.5])
    positives = int(sum(1 for label in clean_labels if int(label) == 1))
    return {
        "point": round(float(point), 6),
        "ci95": [round(float(ci_low), 6), round(float(ci_high), 6)],
        "n": len(clean_scores),
        "n_positive_errors": positives,
        "n_negative_correct": len(clean_scores) - positives,
        "score_variance": round(float(np.var(np.asarray(clean_scores, dtype=np.float64))), 12),
        "bootstrap_seeds": list(seeds),
        "seed_mean_aurocs": seed_means,
    }


def paired_delta_bundle(
    labels: Sequence[int],
    ensemble_scores: Sequence[float],
    confidence_scores: Sequence[float],
    *,
    n_bootstrap: int = 200,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
) -> JsonDict:
    """Return paired bootstrap CI for ensemble minus confidence AUROC."""

    clean_labels, clean_ensemble, clean_confidence = finite_label_score_triplets(
        labels,
        ensemble_scores,
        confidence_scores,
    )
    if not clean_labels:
        return {"point": None, "ci95": None, "bootstrap_seeds": list(seeds), "seed_mean_deltas": []}
    point = tie_aware_auroc(clean_labels, clean_ensemble) - tie_aware_auroc(
        clean_labels,
        clean_confidence,
    )
    boot_values: list[float] = []
    seed_means: list[float] = []
    arr_labels = np.asarray(clean_labels, dtype=np.int64)
    arr_ensemble = np.asarray(clean_ensemble, dtype=np.float64)
    arr_confidence = np.asarray(clean_confidence, dtype=np.float64)
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(arr_labels), size=len(arr_labels))
            value = tie_aware_auroc(arr_labels[idx], arr_ensemble[idx]) - tie_aware_auroc(
                arr_labels[idx],
                arr_confidence[idx],
            )
            values.append(value)
            boot_values.append(value)
        seed_means.append(round(float(np.mean(values)), 6))
    ci_low, ci_high = np.percentile(np.asarray(boot_values, dtype=np.float64), [2.5, 97.5])
    return {
        "point": round(float(point), 6),
        "ci95": [round(float(ci_low), 6), round(float(ci_high), 6)],
        "bootstrap_seeds": list(seeds),
        "seed_mean_deltas": seed_means,
    }


def tie_aware_auroc(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> float:
    """Compute AUROC with half credit for tied positive/negative scores."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = s[y == 1]
    negatives = s[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    greater = positives[:, None] > negatives[None, :]
    ties = positives[:, None] == negatives[None, :]
    wins = float(greater.sum()) + 0.5 * float(ties.sum())
    return wins / float(len(positives) * len(negatives))


def classify_domain(
    ensemble_metrics: Mapping[str, Any],
    confidence_metrics: Mapping[str, Any],
) -> str:
    """Classify one runnable row as generalizing or domain-bound."""

    ensemble_point = ensemble_metrics.get("point")
    confidence_point = confidence_metrics.get("point")
    if ensemble_point is None or confidence_point is None:
        return "domain_bound"
    if float(ensemble_point) > MATERIAL_AUROC_FLOOR and float(ensemble_point) >= float(
        confidence_point
    ):
        return "generalizes"
    return "domain_bound"


def row_headroom(confidence_metrics: Mapping[str, Any] | None) -> bool:
    """Return whether the confidence baseline leaves real headroom."""

    if not isinstance(confidence_metrics, Mapping):
        return False
    point = confidence_metrics.get("point")
    return bool(point is not None and float(point) < 0.95)


def row_ran_with_headroom(row: Mapping[str, Any]) -> bool:
    """Return true only for rows that ran and have baseline headroom."""

    return bool(row.get("ran_or_blocked") == "ran" and row.get("headroom") is True)


def row_generalizes_with_headroom(row: Mapping[str, Any]) -> bool:
    """Return true only for generalizing rows with valid headroom."""

    return bool(row_ran_with_headroom(row) and row.get("domain_verdict") == "generalizes")


def terminal_verdict(
    *,
    code_generalizes: bool,
    facts_generalize: bool,
    positive_control_valid: bool,
    code_ran: bool,
    facts_ran: bool,
) -> str:
    """Select the terminal verdict from honest per-row outcomes."""

    if code_generalizes and facts_generalize:
        return "complete: verifier_value_generalizes_beyond_math_329_null_was_artifact"
    if code_generalizes:
        return "complete: verifier_value_generalizes_to_code_not_facts_partial_scope"
    if facts_generalize:
        return "complete: verifier_value_generalizes_to_facts_not_code_partial_scope"
    if positive_control_valid:
        return "complete: verifier_value_math_only_EARNED_against_valid_positive_control_scoped_limitation"
    if code_ran or facts_ran:
        return "complete: verifier_value_nonmath_partial_rows_domain_bound_positive_control_invalid_no_null_asserted"
    return "complete: verifier_value_nonmath_rows_blocked_no_positive_control_no_null_asserted"


def evidence_excludes_gold_answer(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check that no separate gold-answer field is exposed inside evidence text."""

    gold_keys = ("gold_answer", "right_answer", "correct_answer", "reference_answer")
    for row in rows:
        evidence = str(row.get("evidence_passage") or "").lower()
        for key in gold_keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip() and value.strip().lower() in evidence:
                return False
    return bool(rows)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3642 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    for field in ("positive_control_valid", "at_least_one_nonmath_row_ran"):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    table = artifact.get("generalization_table")
    if not isinstance(table, Mapping) or set(table) != {"math", "code", "facts"}:
        raise ValueError("generalization_table must contain math, code, and facts rows")
    if _round_or_none(artifact.get("math_ensemble_auroc")) != FROZEN_MATH_AUROC:
        raise ValueError("math_ensemble_auroc must restate the frozen 0.9131 headline")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


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
    """Align labels and paired score vectors while dropping non-finite pairs."""

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


def reproducibility_checksum(table: Mapping[str, Any]) -> str:
    """Return a stable checksum over the measured table."""

    encoded = json.dumps(table, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _metric_from_frozen(point: float | None, ci_payload: Any) -> JsonDict:
    if point is None:
        ci = None
    elif isinstance(ci_payload, Mapping) and "low" in ci_payload and "high" in ci_payload:
        ci = [_round_or_none(ci_payload.get("low")), _round_or_none(ci_payload.get("high"))]
    else:
        ci = None
    return {"point": _round_or_none(point), "ci95": ci}


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


__all__ = [
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "score_fact_rows",
    "validate_artifact",
    "write_artifact",
]

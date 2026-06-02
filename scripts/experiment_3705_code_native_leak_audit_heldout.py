#!/usr/bin/env python3
"""Exp 3705: code-native leak audit plus held-out replication.

The Exp 3695 verifier is intentionally heuristic, but it must remain authentic:
it parses Python ASTs, uses CodeExtractor structural findings, and executes
bounded runtime probes through safe_exec_function. This audit treats the Exp
3695 AUROC=1.0 as suspicious until a distinct held-out corpus clears the null
discipline. In particular, held-out AUROC >= 0.99 is a leak red flag rather
than a success, because a fresh real classifier should not stay perfect or
near-perfect on a larger, independent code corpus.

Spec: REQ-CODE-3705, SCENARIO-CODE-3705.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (REPO_ROOT / ".venv").resolve():  # pragma: no cover - direct script startup only.
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:  # pragma: no cover - direct script startup only.
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.pipeline import code_native_verifier_3695 as exp3695  # noqa: E402


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3705_code_native_leak_audit_heldout.json")
IN_CORPUS_REL_PATH = Path("data/code_verification_corpus_v2.jsonl")
HELDOUT_REL_PATH = Path("data/code_verification_corpus_v1.jsonl")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 3705
BOOTSTRAP_SEEDS = (3705, 3706, 3707, 3708, 3709)
DEFAULT_N_BOOTSTRAP = 200
MIN_HELDOUT_EXAMPLES = 61

SURVIVES_VERDICT = "complete: code_native_signal_survives_heldout_real_non_leaked_signal"
LEAK_VERDICT = "complete: code_native_one_point_zero_was_a_leak_code_claim_narrowed_earned"
BLOCKED_VERDICT = "complete: blocked_no_heldout_code_corpus"
TERMINAL_VERDICTS = (SURVIVES_VERDICT, LEAK_VERDICT, BLOCKED_VERDICT)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "in_corpus_code_auroc",
    "in_corpus_auroc_diagnosis",
    "heldout_code_auroc",
    "heldout_code_auroc_ci",
    "leak_audit_findings",
    "leak_detected",
    "code_signal_survives_heldout",
    "heldout_calibration_brier_ece",
    "n_seeds",
    "n_examples_heldout",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates if cached-only (no marker); "
        "live_llm_inference only if a real live-model generation step ran (60s floor)."
    ),
    "in_corpus_code_auroc": (
        "The exp3695 ~1.0 reproduced on the tuned corpus -- the suspicious number being audited."
    ),
    "in_corpus_auroc_diagnosis": (
        "WHY the in-corpus AUROC is ~1.0 (separability / leakage / label-correlated feature)."
    ),
    "heldout_code_auroc": (
        "The code-native AUROC on the DIFFERENT held-out corpus -- the real generalization number."
    ),
    "heldout_code_auroc_ci": (
        "CI of the held-out AUROC -- must EXCLUDE 0.5 to count as signal; held-out 1.0 is still a leak flag."
    ),
    "leak_audit_findings": (
        "Honest record of separability, contamination, label-correlation, and verifier authenticity checks."
    ),
    "leak_detected": (
        "BARE bool. True iff the in-corpus 1.0 is explained by leak/separability or held-out >=0.99."
    ),
    "code_signal_survives_heldout": (
        "BARE bool. False whenever held-out AUROC >=0.99; otherwise true iff held-out AUROC >0.5 with CI excluding 0.5."
    ),
    "heldout_calibration_brier_ece": (
        "Calibration of the held-out code signal -- a deployable code operating point must be calibrated."
    ),
    "n_seeds": "Replication count (>=5).",
    "n_examples_heldout": "Sample-size rigor on the held-out corpus.",
    "adversarial_verify_clean": "True iff no critical flag and the suspicious 1.0 was explained, not hidden.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    in_corpus_rows: Sequence[Mapping[str, Any]] | None = None,
    heldout_rows: Sequence[Mapping[str, Any]] | None = None,
    min_heldout_examples: int = MIN_HELDOUT_EXAMPLES,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp 3705 artifact from cached code corpora."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    in_rows, in_status = _rows_or_default(root_path, in_corpus_rows, IN_CORPUS_REL_PATH)
    held_rows, held_status = _rows_or_default(root_path, heldout_rows, HELDOUT_REL_PATH)
    preconditions = precondition_checks(
        in_rows,
        held_rows,
        min_heldout_examples=min_heldout_examples,
        in_status=in_status,
        heldout_status=held_status,
    )
    if not all(bool(item["available"]) for item in preconditions):
        return build_artifact_from_measurements(
            blocked=True,
            in_corpus_metric={},
            in_corpus_auroc_diagnosis="blocked_no_heldout_code_corpus",
            heldout_metric={},
            heldout_calibration_brier_ece={},
            heldout_recall_at_fixed_fpr={},
            leak_audit_findings={},
            n_examples_heldout=len(held_rows),
            n_examples_in_corpus=len(in_rows),
            adversarial_verify_clean=False,
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
            extra={
                "preconditions_checked": preconditions,
                "corpus_status": {"in_corpus": in_status, "heldout": held_status},
            },
        )

    in_measured = measure_code_native_corpus(in_rows, seeds=seeds, n_bootstrap=n_bootstrap)
    held_measured = measure_code_native_corpus(held_rows, seeds=seeds, n_bootstrap=n_bootstrap)
    audit = audit_exp3658_corpus(in_rows, random_seed=RANDOM_SEED)
    audit["heldout_implausible_perfect_red_flag"] = heldout_is_implausibly_perfect(
        held_measured["metric"]
    )
    return build_artifact_from_measurements(
        blocked=False,
        in_corpus_metric=in_measured["metric"],
        in_corpus_auroc_diagnosis=diagnose_in_corpus_auroc(audit),
        heldout_metric=held_measured["metric"],
        heldout_calibration_brier_ece=held_measured["calibration"],
        heldout_recall_at_fixed_fpr=held_measured["recall_at_fixed_fpr"],
        leak_audit_findings=audit,
        n_examples_heldout=len(held_rows),
        n_examples_in_corpus=len(in_rows),
        adversarial_verify_clean=True,
        started_s=start,
        now_s=now_s,
        tests_run=tests_run,
        extra={
            "preconditions_checked": preconditions,
            "corpus_status": {"in_corpus": in_status, "heldout": held_status},
            "in_corpus_code_auroc_metric": in_measured["metric"],
            "heldout_code_auroc_metric": held_measured["metric"],
            "in_corpus_feature_summary": in_measured["feature_summary"],
            "heldout_feature_summary": held_measured["feature_summary"],
            "heldout_recall_at_fixed_fpr": held_measured["recall_at_fixed_fpr"],
        },
    )


def build_artifact_from_measurements(
    *,
    blocked: bool,
    in_corpus_metric: Mapping[str, Any],
    in_corpus_auroc_diagnosis: str,
    heldout_metric: Mapping[str, Any],
    heldout_calibration_brier_ece: Mapping[str, Any],
    heldout_recall_at_fixed_fpr: Mapping[str, Any],
    leak_audit_findings: Mapping[str, Any],
    n_examples_heldout: int,
    n_examples_in_corpus: int,
    adversarial_verify_clean: bool,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble the artifact from already-measured metrics."""

    start = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    survives = False if blocked else code_signal_survives_heldout(heldout_metric)
    leak_detected = False if blocked else leak_detected_from_findings(leak_audit_findings, heldout_metric)
    verdict = classify_verdict(blocked=blocked, survives=survives, leak_detected=leak_detected)
    artifact: JsonDict = {
        "artifact": "experiment_3705_code_native_leak_audit_heldout",
        "schema": "carnot.code_native_leak_audit_heldout.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "in_corpus_code_auroc": _metric_point(in_corpus_metric),
        "in_corpus_code_auroc_ci": _metric_ci(in_corpus_metric),
        "in_corpus_auroc_diagnosis": str(in_corpus_auroc_diagnosis),
        "heldout_code_auroc": _metric_point(heldout_metric),
        "heldout_code_auroc_ci": _metric_ci(heldout_metric),
        "heldout_calibration_brier_ece": dict(heldout_calibration_brier_ece),
        "heldout_recall_at_fixed_fpr": dict(heldout_recall_at_fixed_fpr),
        "leak_audit_findings": dict(leak_audit_findings),
        "leak_detected": leak_detected,
        "code_signal_survives_heldout": survives,
        "n_seeds": 0 if blocked else len(heldout_metric.get("bootstrap_seeds") or BOOTSTRAP_SEEDS),
        "n_examples_in_corpus": int(n_examples_in_corpus),
        "n_examples_heldout": int(n_examples_heldout),
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _round(max(0.0, finished - start)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "in_corpus_code_auroc present AND heldout_code_auroc present AND "
                "heldout_code_auroc_ci present AND leak_audit_findings present AND "
                "adversarial_verify_clean == true"
            ),
            "principle": (
                "A trustworthy code-native verdict requires the suspicious 1.0 reproduced "
                "and explained, held-out AUROC + CI, the leak audit, and adversarial-clean."
            ),
            "passed": bool(
                not blocked
                and _metric_point(in_corpus_metric) is not None
                and _metric_point(heldout_metric) is not None
                and _metric_ci(heldout_metric) is not None
                and bool(leak_audit_findings)
                and adversarial_verify_clean is True
            ),
        },
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    artifact.update(dict(extra or {}))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def measure_code_native_corpus(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    n_bootstrap: int,
) -> JsonDict:
    """Score rows with the Exp 3695 AST/runtime verifier and return metrics."""

    labels = exp3695.code_error_labels(rows)
    scored = exp3695.CodeNativeVerifier().score_rows(rows)
    scores = [item.score for item in scored]
    return {
        "metric": exp3695.auroc_metric(labels, scores, seeds=seeds, n_bootstrap=n_bootstrap),
        "calibration": exp3695.calibration_bundle(labels, scores),
        "recall_at_fixed_fpr": exp3695.recall_at_fixed_fpr_table(labels, scores),
        "feature_summary": exp3695.feature_summary(scored),
        "scores": scores,
    }


def audit_exp3658_corpus(
    rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Audit Exp 3658 construction for separability and contamination."""

    row_list = [dict(row) for row in rows]
    labels = exp3695.code_error_labels(row_list)
    scored = exp3695.CodeNativeVerifier().score_rows(row_list)
    scores = [item.score for item in scored]
    metadata_correlations = {
        "source": categorical_label_correlation([row.get("source") for row in row_list], labels),
        "mutation": categorical_label_correlation(
            [_metadata_value(row, "mutation") for row in row_list],
            labels,
        ),
    }
    label_correlated = any(
        item["purity"] == 1.0 and item["n_values"] > 1 for item in metadata_correlations.values()
    )
    score_gap = score_gap_summary(labels, scores)
    contamination = contamination_check(row_list, scored, random_seed=random_seed)
    return {
        "in_corpus_construction": {
            "source_corpus": "data/code_verification_corpus_v2.jsonl",
            "construction_summary": (
                "Exp3658 positives are cached canonical HumanEval solutions; negatives "
                "are deterministic return_none mutants from the same task ids."
            ),
            "separable_by_construction": bool(label_correlated and score_gap["score_gap_separable"]),
            "label_correlated_metadata": bool(label_correlated),
            "score_gap_separable": bool(score_gap["score_gap_separable"]),
            "metadata_label_correlations": metadata_correlations,
            "score_gap": score_gap,
            "mean_features_by_error_label": mean_features_by_label(scored, labels),
        },
        "contamination_check": contamination,
        "verifier_authenticity": {
            "ast_parse_used": bool(scored and any("ast_parseable" in item.features for item in scored)),
            "runtime_execution_trace_used": bool(
                any(item.features.get("execution_attempted", 0.0) > 0.0 for item in scored)
            ),
            "constant_score": bool(len(set(scores)) <= 1),
            "heuristic_gap_disclosed": "heuristic" in exp3695.VERIFIER_IMPLEMENTATION.lower(),
            "implementation_note": exp3695.VERIFIER_IMPLEMENTATION,
        },
    }


def contamination_check(
    rows: Sequence[Mapping[str, Any]],
    scored: Sequence[Any],
    *,
    random_seed: int,
) -> JsonDict:
    """Run exact-overlap, task-overlap, and RBF-kernel split checks."""

    train_idx, holdout_idx = stratified_row_split(rows, random_seed=random_seed)
    train_rows = [rows[idx] for idx in train_idx]
    holdout_rows = [rows[idx] for idx in holdout_idx]
    train_features = feature_matrix([scored[idx] for idx in train_idx])
    holdout_features = feature_matrix([scored[idx] for idx in holdout_idx])
    return {
        "method": "session_isolated_exact_overlap_plus_rbf_kernel_mmd_proxy",
        "train_n": len(train_rows),
        "holdout_n": len(holdout_rows),
        "exact_candidate_overlap": len(_value_set(train_rows, "candidate_sha256") & _value_set(holdout_rows, "candidate_sha256")),
        "task_id_overlap": len(_task_set(train_rows) & _task_set(holdout_rows)),
        "source_overlap": sorted(_value_set(train_rows, "source") & _value_set(holdout_rows, "source")),
        "mutation_overlap": sorted(_metadata_set(train_rows, "mutation") & _metadata_set(holdout_rows, "mutation")),
        "kernel_mmd_rbf": _round(rbf_mmd(train_features, holdout_features)),
        "methodology_note": (
            "This is a deterministic split-contamination proxy inspired by cross-context "
            "and kernel-divergence audits; it is not a full reproduction of those papers."
        ),
    }


def diagnose_in_corpus_auroc(audit: Mapping[str, Any]) -> str:
    """Return a concise diagnosis string for the suspicious in-corpus AUROC."""

    construction = audit.get("in_corpus_construction")
    if not isinstance(construction, Mapping):
        return "blocked_no_in_corpus_diagnosis"
    mutation = construction.get("metadata_label_correlations", {}).get("mutation", {})
    score_gap = construction.get("score_gap", {})
    if construction.get("separable_by_construction"):
        return (
            "Exp3658 is separable-by-construction: mutation metadata maps labels "
            f"with purity={mutation.get('purity')}, return_none negatives are generated "
            "differently from canonical positives, and the code-native score has a hard "
            f"gap min_error_score={score_gap.get('min_error_score')} > "
            f"max_correct_score={score_gap.get('max_correct_score')}."
        )
    return "No deterministic in-corpus separation found by the Exp3705 audit."


def code_signal_survives_heldout(metric: Mapping[str, Any]) -> bool:
    """Return the bare held-out survival gate with the >=0.99 leak guard."""

    point = _metric_point(metric)
    ci = _metric_ci(metric)
    if point is None or ci is None:
        return False
    if point >= 0.99:
        return False
    return bool(point > 0.5 and ci[0] > 0.5)


def heldout_is_implausibly_perfect(metric: Mapping[str, Any]) -> bool:
    """Return true for the user-mandated held-out >=0.99 red flag."""

    point = _metric_point(metric)
    return bool(point is not None and point >= 0.99)


def leak_detected_from_findings(findings: Mapping[str, Any], heldout_metric: Mapping[str, Any]) -> bool:
    """Extract the top-level leak bool from audit findings and held-out metric."""

    if heldout_is_implausibly_perfect(heldout_metric):
        return True
    construction = findings.get("in_corpus_construction")
    if isinstance(construction, Mapping) and (
        construction.get("separable_by_construction")
        or construction.get("label_correlated_metadata")
        or construction.get("score_gap_separable")
    ):
        return True
    return bool(findings.get("heldout_implausible_perfect_red_flag"))


def classify_verdict(*, blocked: bool, survives: bool, leak_detected: bool) -> str:
    """Map gates onto one of the terminal verdicts."""

    if blocked:
        return BLOCKED_VERDICT
    if survives and not leak_detected:
        return SURVIVES_VERDICT
    return LEAK_VERDICT


def precondition_checks(
    in_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    min_heldout_examples: int,
    in_status: Mapping[str, Any],
    heldout_status: Mapping[str, Any],
) -> list[JsonDict]:
    """Return cached-corpus availability checks."""

    return [
        {
            "resource": "exp3658_in_corpus",
            "available": _has_both_code_classes(in_rows),
            "n_examples": len(in_rows),
            "detail": dict(in_status),
        },
        {
            "resource": "distinct_heldout_code_corpus",
            "available": len(heldout_rows) >= int(min_heldout_examples)
            and _has_both_code_classes(heldout_rows),
            "n_examples": len(heldout_rows),
            "minimum_required": int(min_heldout_examples),
            "detail": dict(heldout_status),
        },
    ]


def categorical_label_correlation(values: Sequence[Any], labels: Sequence[int]) -> JsonDict:
    """Return category-label purity for a surface field."""

    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for value, label in zip(values, labels, strict=False):
        counts[str(value)][int(label)] += 1
    n = sum(sum(counter.values()) for counter in counts.values())
    correct = sum(max(counter.values()) for counter in counts.values()) if counts else 0
    return {
        "n_values": len(counts),
        "purity": _round(correct / n) if n else 0.0,
        "counts": {key: dict(counter) for key, counter in sorted(counts.items())},
    }


def score_gap_summary(labels: Sequence[int], scores: Sequence[float]) -> JsonDict:
    """Summarize whether scores alone perfectly separate errors from correct rows."""

    error_scores = [float(score) for label, score in zip(labels, scores, strict=False) if label == 1]
    correct_scores = [float(score) for label, score in zip(labels, scores, strict=False) if label == 0]
    if not error_scores or not correct_scores:
        return {"score_gap_separable": False, "min_error_score": None, "max_correct_score": None}
    min_error = min(error_scores)
    max_correct = max(correct_scores)
    return {
        "score_gap_separable": bool(min_error > max_correct),
        "min_error_score": _round(min_error),
        "max_correct_score": _round(max_correct),
        "gap": _round(min_error - max_correct),
    }


def mean_features_by_label(scored: Sequence[Any], labels: Sequence[int]) -> JsonDict:
    """Return mean feature values split by error label."""

    if not scored:
        return {"error": {}, "correct": {}}
    names = sorted(scored[0].features)
    result: JsonDict = {}
    for label_value, label_name in ((1, "error"), (0, "correct")):
        group = [item for item, label in zip(scored, labels, strict=False) if label == label_value]
        result[label_name] = {
            name: _round(float(np.mean([item.features.get(name, 0.0) for item in group])))
            if group
            else 0.0
            for name in names
        }
    return result


def stratified_row_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
    train_fraction: float = 0.7,
) -> tuple[list[int], list[int]]:
    """Split row indices within each correctness label bucket."""

    rng = np.random.default_rng(int(random_seed))
    train: list[int] = []
    holdout: list[int] = []
    for label in (True, False):
        bucket = [idx for idx, row in enumerate(rows) if bool(row.get("label")) is label]
        order = list(rng.permutation(bucket))
        cut = max(1, int(len(order) * float(train_fraction))) if len(order) > 1 else len(order)
        train.extend(int(idx) for idx in order[:cut])
        holdout.extend(int(idx) for idx in order[cut:])
    return train, holdout


def feature_matrix(scored: Sequence[Any]) -> np.ndarray:
    """Convert CodeNativeScore feature dicts into a dense matrix."""

    if not scored:
        return np.zeros((0, 0), dtype=np.float64)
    names = sorted(scored[0].features)
    return np.asarray([[item.features.get(name, 0.0) for name in names] for item in scored], dtype=np.float64)


def rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
    """Return a small RBF-kernel MMD proxy between two feature matrices."""

    if x.size == 0 or y.size == 0:
        return 0.0
    combined = np.vstack([x, y])
    sq_dists = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2)
    nonzero = sq_dists[sq_dists > 0.0]
    bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    gamma = 1.0 / max(bandwidth, 1e-12)
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)).mean()
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2)).mean()
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3705 terminal artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS:
        raise ValueError("honest_verdict is not an accepted Exp 3705 terminal verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be the cached-corpus verifier sentinel")
    for field in ("code_signal_survives_heldout", "leak_detected", "adversarial_verify_clean"):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    if not isinstance(artifact.get("n_examples_heldout"), int):
        raise ValueError("n_examples_heldout must be an int")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic fields used to detect metric drift."""

    payload = {
        "honest_verdict": artifact.get("honest_verdict"),
        "in_corpus_code_auroc": artifact.get("in_corpus_code_auroc"),
        "heldout_code_auroc": artifact.get("heldout_code_auroc"),
        "heldout_code_auroc_ci": artifact.get("heldout_code_auroc_ci"),
        "leak_detected": artifact.get("leak_detected"),
        "code_signal_survives_heldout": artifact.get("code_signal_survives_heldout"),
        "n_examples_heldout": artifact.get("n_examples_heldout"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, adversarial-check, validate, and persist Exp 3705."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = run_adversarial_verify_report(output)
    artifact["adversarial_verify_report"] = compact_adversarial_report(report)
    artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
    artifact["acceptance_gate"]["passed"] = bool(
        artifact["acceptance_gate"]["passed"] and artifact["adversarial_verify_clean"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_artifact_from_measurements(
    root: Path | str,
    *,
    output_path: Path | str,
    artifact: Mapping[str, Any],
) -> Path:
    """Persist a pre-built Exp 3705 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_adversarial_verify_report(path: Path) -> JsonDict:
    """Run scripts/adversarial_verify.py against an artifact path."""

    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_3705", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(path))


def compact_adversarial_report(report: Mapping[str, Any]) -> JsonDict:
    """Store a deterministic compact adversarial report."""

    flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    return {"flag_count": len(flags), "flags": flags}


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true when no adversarial flag is critical."""

    flags = report.get("flags", [])
    if not isinstance(flags, Sequence):
        return False
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def _rows_or_default(
    root: Path,
    rows: Sequence[Mapping[str, Any]] | None,
    rel_path: Path,
) -> tuple[list[JsonDict], JsonDict]:
    if rows is not None:
        return [dict(row) for row in rows], {"source": "fixture_override", "path": None}
    path = root / rel_path
    loaded = read_jsonl(path)
    return loaded, {"source": "cached_jsonl", "path": str(rel_path), "exists": path.exists()}


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read JSONL rows, returning an empty list when the file is absent."""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _has_both_code_classes(rows: Sequence[Mapping[str, Any]]) -> bool:
    labels = {bool(row.get("label")) for row in rows}
    return labels == {False, True}


def _metadata_value(row: Mapping[str, Any], key: str) -> Any:
    metadata = row.get("metadata")
    return metadata.get(key) if isinstance(metadata, Mapping) else None


def _value_set(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    return {str(row.get(key)) for row in rows if row.get(key) is not None}


def _metadata_set(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    return {str(_metadata_value(row, key)) for row in rows if _metadata_value(row, key) is not None}


def _task_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row.get("task_id") or _metadata_value(row, "stable_id"))
        for row in rows
        if row.get("task_id") or _metadata_value(row, "stable_id")
    }


def _metric_point(metric: Mapping[str, Any]) -> float | None:
    point = metric.get("point")
    return None if point is None else _round(float(point))


def _metric_ci(metric: Mapping[str, Any]) -> list[float] | None:
    ci = metric.get("ci95")
    if not isinstance(ci, Sequence) or len(ci) != 2:
        return None
    return [_round(float(ci[0])), _round(float(ci[1]))]


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    return round(float(value), 6)


def main() -> int:  # pragma: no cover
    output = write_artifact(
        REPO_ROOT,
        tests_run=[
            ".venv/bin/pytest tests/python/test_experiment_3705_code_native_leak_audit_heldout.py -q",
            ".venv/bin/coverage run --source=scripts -m pytest -o addopts='' tests/python/test_experiment_3705_code_native_leak_audit_heldout.py -q",
            ".venv/bin/coverage report --include='scripts/experiment_3705_code_native_leak_audit_heldout.py' --fail-under=100 --show-missing",
            ".venv/bin/python scripts/check_spec_coverage.py",
            ".venv/bin/pytest tests/python -q",
        ],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

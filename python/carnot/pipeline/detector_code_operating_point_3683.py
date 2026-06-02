"""Exp 3683 code operating point hardening for the shipped detector.

Spec: REQ-SPOE-3683, SCENARIO-SPOE-3683.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3683_detector_code_operating_point.json")
RANDOM_SEED = 3683
BOOTSTRAP_SEEDS = (3683, 3684, 3685)
DEFAULT_N_BOOTSTRAP = 200
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached balanced code corpus; no LLM load)."
)

VERDICT_RECOVERED = "complete: code_operating_point_recovered_detector_now_math_and_code"
VERDICT_MATH_ONLY = "complete: code_remains_math_only_detector_scoped_honestly"
VERDICT_BLOCKED = "complete: blocked_no_balanced_code_corpus_or_detector_module"
TERMINAL_VERDICTS = (VERDICT_RECOVERED, VERDICT_MATH_ONLY, VERDICT_BLOCKED)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "code_auroc_baseline",
    "code_auroc_dependency_aware",
    "code_auroc_recalibrated",
    "code_calibration_brier_ece_after",
    "code_recall_at_fixed_fpr",
    "module_code_path_updated",
    "e2e_test_passed",
    "code_operating_point_recovered",
    "n_examples_code",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "code_auroc_baseline": (
        "The exp3671 0.5 code baseline reconfirmed -- the bar to beat."
    ),
    "code_auroc_dependency_aware": (
        "Code AUROC with the .336 dependency-aware weighting -- does the "
        "math-validated weighting transfer to code?"
    ),
    "code_auroc_recalibrated": (
        "Code AUROC after code-specific recalibration + its CI -- must exclude "
        "0.5 to count as signal."
    ),
    "code_calibration_brier_ece_after": (
        "Calibration after the fix -- a deployable code operating point must be calibrated."
    ),
    "code_recall_at_fixed_fpr": (
        "Recall-at-FPR table for the code operating point a deployer reads."
    ),
    "module_code_path_updated": (
        "True iff the shipped module's code operating point was updated "
        "(or honestly documented math-only)."
    ),
    "e2e_test_passed": (
        "True iff the shipped surface E2E test still passes after the change."
    ),
    "code_operating_point_recovered": (
        "BARE bool. True iff a fix lifts code AUROC above 0.5 (CI excludes 0.5) "
        "AND improves calibration -- else code is honestly math-only. STORE AS "
        "BARE true/false."
    ),
    "n_examples_code": "Sample-size rigor on the code corpus.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

_SPD_MODULE_NAME = "_carnot_exp3683_second_pair_detector"


def _load_second_pair_detector() -> Any:
    module = sys.modules.get(_SPD_MODULE_NAME)
    if module is not None:
        return module
    path = Path(__file__).with_name("second_pair_detector.py")
    spec = importlib.util.spec_from_file_location(_SPD_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load second_pair_detector from {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[_SPD_MODULE_NAME] = loaded
    spec.loader.exec_module(loaded)
    return loaded


spd = _load_second_pair_detector()


@dataclass(frozen=True)
class OutcomeClassification:
    """Terminal verdict and recovered gate for a measured code operating point."""

    terminal_verdict: str
    code_operating_point_recovered: bool


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3683 code operating point artifact from cached rows."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    preconditions = check_preconditions(root_path)
    if not all(bool(item["available"]) for item in preconditions):
        return build_artifact_from_metrics(
            blocked=True,
            code_auroc_baseline={},
            code_auroc_dependency_aware={},
            code_auroc_recalibrated={},
            code_calibration_brier_ece_after={},
            code_recall_at_fixed_fpr={},
            n_examples_code=_precondition_n_examples(preconditions),
            module_code_path_updated=False,
            e2e_test_passed=False,
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
            preconditions_checked=preconditions,
        )

    rows, _status = load_balanced_code_rows(root_path)
    labels = code_error_labels(rows)
    confidence_scores = score_code_confidence(rows)
    all_examples, _corpus_status = spd.load_cached_labeled_examples(
        root_path,
        use_balanced_code_corpus=True,
    )
    baseline = measure_baseline_code_operating_point(
        all_examples,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    verifier_names, score_matrix = code_verifier_score_panel(rows, root_path)
    dependency_scores = dependency_aware_code_scores(
        labels=labels,
        score_matrix=score_matrix,
        verifier_names=verifier_names,
    )
    dependency_metric = auroc_metric(
        labels,
        dependency_scores,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    dependency_metric["verifier_names"] = list(verifier_names)
    dependency_metric["weighting"] = "exp3667_dependency_aware_crossfit"
    recalibrated = measure_code_recalibration(
        labels=labels,
        ensemble_scores=dependency_scores,
        confidence_scores=confidence_scores,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    e2e_passed = run_score_candidates_e2e(root_path, all_examples)
    module_updated = bool(getattr(spd, "CODE_OPERATING_POINT_SCOPE", ""))
    return build_artifact_from_metrics(
        blocked=False,
        code_auroc_baseline=baseline,
        code_auroc_dependency_aware=dependency_metric,
        code_auroc_recalibrated=recalibrated["code_auroc_recalibrated"],
        code_calibration_brier_ece_after=recalibrated["code_calibration_brier_ece_after"],
        code_recall_at_fixed_fpr=recalibrated["code_recall_at_fixed_fpr"],
        n_examples_code=len(rows),
        module_code_path_updated=module_updated,
        e2e_test_passed=e2e_passed,
        started_s=start,
        now_s=now_s,
        tests_run=tests_run,
        preconditions_checked=preconditions,
        extra={
            "code_recalibration_protocol": recalibrated["code_recalibration_protocol"],
            "dependency_aware_score_summary": {
                "score_variance": _round(float(np.var(np.asarray(dependency_scores)))),
                "n_scores": len(dependency_scores),
            },
        },
    )


def build_artifact_from_metrics(
    *,
    blocked: bool,
    code_auroc_baseline: Mapping[str, Any],
    code_auroc_dependency_aware: Mapping[str, Any],
    code_auroc_recalibrated: Mapping[str, Any],
    code_calibration_brier_ece_after: Mapping[str, Any],
    code_recall_at_fixed_fpr: Mapping[str, Any],
    n_examples_code: int,
    module_code_path_updated: bool,
    e2e_test_passed: bool,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    preconditions_checked: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble the terminal artifact from already-measured metrics."""

    start = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    classification = classify_outcome(
        blocked=blocked,
        code_auroc_baseline=code_auroc_baseline,
        code_auroc_recalibrated=code_auroc_recalibrated,
        code_calibration_brier_ece_after=code_calibration_brier_ece_after,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3683_detector_code_operating_point",
        "schema": "carnot.detector_code_operating_point.v1",
        "honest_verdict": classification.terminal_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "code_auroc_baseline": dict(code_auroc_baseline),
        "code_auroc_dependency_aware": dict(code_auroc_dependency_aware),
        "code_auroc_recalibrated": dict(code_auroc_recalibrated),
        "code_calibration_brier_ece_after": dict(code_calibration_brier_ece_after),
        "code_recall_at_fixed_fpr": dict(code_recall_at_fixed_fpr),
        "module_code_path_updated": bool(module_code_path_updated),
        "e2e_test_passed": bool(e2e_test_passed),
        "code_operating_point_recovered": classification.code_operating_point_recovered,
        "n_examples_code": int(n_examples_code),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _round(max(0.0, finished - start)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "code_auroc_baseline present AND code_auroc_recalibrated present "
                "AND code_calibration_brier_ece_after present"
            ),
            "passed": bool(
                not blocked
                and code_auroc_baseline
                and code_auroc_recalibrated
                and code_calibration_brier_ece_after
            ),
            "principle": (
                "A code-hardening verdict requires the baseline, the post-fix "
                "AUROC and the post-fix calibration -- a single number cannot "
                "decide whether the code operating point is deployable or math-only."
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions_checked or []],
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    artifact.update(dict(extra or {}))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def classify_outcome(
    *,
    blocked: bool,
    code_auroc_baseline: Mapping[str, Any],
    code_auroc_recalibrated: Mapping[str, Any],
    code_calibration_brier_ece_after: Mapping[str, Any],
) -> OutcomeClassification:
    """Map measured code statistics onto the three terminal outcomes."""

    if blocked:
        return OutcomeClassification(VERDICT_BLOCKED, False)
    recovered = bool(
        auroc_signal_excludes_chance(code_auroc_recalibrated)
        and calibration_improved(
            _baseline_calibration(code_auroc_baseline),
            code_calibration_brier_ece_after,
        )
    )
    verdict = VERDICT_RECOVERED if recovered else VERDICT_MATH_ONLY
    return OutcomeClassification(verdict, recovered)


def auroc_signal_excludes_chance(metric: Mapping[str, Any]) -> bool:
    """Return true only when AUROC is above chance and CI excludes 0.5."""

    point = metric.get("point")
    ci = metric.get("ci95")
    if point is None or not isinstance(ci, Sequence) or len(ci) != 2:
        return False
    return bool(float(point) > 0.5 and float(ci[0]) > 0.5)


def calibration_improved(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> bool:
    """Return true only when Brier and ECE both improve."""

    try:
        return bool(
            float(after["brier"]) < float(before["brier"])
            and float(after["ece"]) < float(before["ece"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def auroc_metric(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> JsonDict:
    """Return tie-aware AUROC plus a deterministic bootstrap CI."""

    clean_labels, clean_scores = spd.finite_label_scores(labels, scores)
    if not clean_labels or len(set(clean_labels)) < 2:
        return empty_metric(seeds)
    label_arr = np.asarray(clean_labels, dtype=np.int64)
    score_arr = np.asarray(clean_scores, dtype=np.float64)
    point = spd.tie_aware_auroc(label_arr, score_arr)
    boot_values: list[float] = []
    seed_means: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(label_arr), size=len(label_arr))
            if len(set(label_arr[idx].tolist())) < 2:
                continue
            value = spd.tie_aware_auroc(label_arr[idx], score_arr[idx])
            values.append(float(value))
            boot_values.append(float(value))
        seed_means.append(_round(float(np.mean(values))) if values else _round(point))
    if boot_values:
        ci_low, ci_high = np.percentile(np.asarray(boot_values, dtype=np.float64), [2.5, 97.5])
    else:
        ci_low = ci_high = point
    positives = int(np.sum(label_arr == 1))
    return {
        "point": _round(point),
        "ci95": [_round(float(ci_low)), _round(float(ci_high))],
        "n": len(clean_labels),
        "n_positive_errors": positives,
        "n_negative_correct": len(clean_labels) - positives,
        "bootstrap_seeds": [int(seed) for seed in seeds],
        "seed_mean_aurocs": seed_means,
    }


def empty_metric(seeds: Sequence[int]) -> JsonDict:
    """Return an empty metric bundle for blocked or one-class inputs."""

    return {
        "point": None,
        "ci95": None,
        "n": 0,
        "n_positive_errors": 0,
        "n_negative_correct": 0,
        "bootstrap_seeds": [int(seed) for seed in seeds],
        "seed_mean_aurocs": [],
    }


def calibration_bundle(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> JsonDict:
    """Return Brier and ECE for aligned finite labels and probabilities."""

    clean_labels, clean_probs = spd.finite_label_scores(labels, probabilities)
    return {
        "brier": _round(spd.brier_score(clean_labels, clean_probs)),
        "ece": _round(spd.expected_calibration_error(clean_labels, clean_probs)),
    }


def measure_code_recalibration(
    *,
    labels: Sequence[int],
    ensemble_scores: Sequence[float],
    confidence_scores: Sequence[float],
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> JsonDict:
    """Fit code-only logistic calibration and evaluate held-out code rows."""

    examples = [
        spd.LabeledDetectorExample(
            domain="code",
            label=int(label),
            ensemble_energy=float(ensemble),
            confidence_error=float(confidence),
            example_id=f"code-recal-{idx}",
        )
        for idx, (label, ensemble, confidence) in enumerate(
            zip(labels, ensemble_scores, confidence_scores, strict=False)
        )
    ]
    train, holdout = spd.stratified_train_holdout(examples, seed=RANDOM_SEED)
    if not _has_both_classes(train) or not _has_both_classes(holdout):
        return {
            "code_auroc_recalibrated": empty_metric(seeds),
            "code_calibration_brier_ece_after": {},
            "code_recall_at_fixed_fpr": {},
            "code_recalibration_protocol": {
                "method": "logistic",
                "train_n": len(train),
                "holdout_n": len(holdout),
                "blocked_reason": "one_class_train_or_holdout",
            },
        }
    detector = spd.CalibratedFusedDetector().fit(train)
    holdout_labels = [example.label for example in holdout]
    probabilities = detector.predict_proba(holdout)
    holdout_confidence = [example.confidence_error for example in holdout]
    holdout_ensemble = [example.ensemble_energy for example in holdout]
    return {
        "code_auroc_recalibrated": auroc_metric(
            holdout_labels,
            probabilities,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        ),
        "code_calibration_brier_ece_after": calibration_bundle(holdout_labels, probabilities),
        "code_recall_at_fixed_fpr": spd.recall_at_fixed_fpr_table(
            holdout_labels,
            probabilities,
            holdout_confidence,
            holdout_ensemble,
        ),
        "code_recalibration_protocol": {
            "method": "logistic",
            "train_n": len(train),
            "holdout_n": len(holdout),
            "feature_names": list(detector.feature_names),
        },
    }


def measure_baseline_code_operating_point(
    examples: Sequence[spd.LabeledDetectorExample],
    *,
    seeds: Sequence[int],
    n_bootstrap: int,
) -> JsonDict:
    """Re-measure the shipped mixed-domain calibrator on code holdout rows."""

    train, holdout = spd.stratified_train_holdout(examples, seed=spd.SHIP_RANDOM_SEED)
    code_holdout = [example for example in holdout if example.domain == "code"]
    if not _has_both_classes(train) or not _has_both_classes(code_holdout):
        return {}
    detector = spd.CalibratedFusedDetector().fit(train)
    labels = [example.label for example in code_holdout]
    fused_scores = detector.predict_proba(code_holdout)
    ensemble_scores = [example.ensemble_energy for example in code_holdout]
    confidence_scores = [example.confidence_error for example in code_holdout]
    return {
        "fused": auroc_metric(labels, fused_scores, seeds=seeds, n_bootstrap=n_bootstrap),
        "ensemble": auroc_metric(labels, ensemble_scores, seeds=seeds, n_bootstrap=n_bootstrap),
        "confidence": auroc_metric(labels, confidence_scores, seeds=seeds, n_bootstrap=n_bootstrap),
        "calibration_brier_ece": calibration_bundle(labels, fused_scores),
        "n_holdout": len(code_holdout),
    }


def dependency_aware_code_scores(
    *,
    labels: Sequence[int],
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
) -> list[float]:
    """Apply the Exp 3667 dependency-aware crossfit weighting to code scores."""

    from carnot.verify import dependency_aware_weighting_clean as exp3667

    panel = exp3667.score_weighting_panel(
        labels=np.asarray(labels, dtype=np.int64),
        score_matrix=np.asarray(score_matrix, dtype=np.float64),
        verifier_names=list(verifier_names),
        random_seed=RANDOM_SEED,
    )
    return [float(score) for score in panel["dependency_aware_proper"]]


def code_verifier_score_panel(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
) -> tuple[list[str], np.ndarray]:
    """Return row-aligned code verifier score columns for dependency weighting."""

    from carnot.verify import code_corpus_verifiers_fire_transfer_v3 as code_transfer

    ast_scores = code_transfer.ast_structure_scores(rows)
    structural_scores = structural_dependency_scores_aligned(rows, root)
    math_transfer_scores = code_transfer.score_math_signal(rows, score_overrides={})
    names = [
        "ast_structure_verifier",
        "code_structural_dependency_verifier",
        "math_signal_transfer",
    ]
    matrix = np.column_stack(
        [
            _require_length(ast_scores, len(rows), "ast_structure_verifier"),
            _require_length(
                structural_scores,
                len(rows),
                "code_structural_dependency_verifier",
            ),
            _require_length(math_transfer_scores, len(rows), "math_signal_transfer"),
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("code verifier score matrix must be finite")
    return names, matrix.astype(np.float64)


def structural_dependency_scores_aligned(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
) -> list[float]:
    """Score structural dependency checks with one output per input row."""

    from carnot.verify import code_corpus_verifiers_fire_transfer_v3 as code_transfer
    from carnot.verify import code_structural_dependency_verifier as dep

    manifests = code_transfer.load_manifest_lookup(rows, root)
    scores: list[float] = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        corpus = code_transfer.normalize_corpus(metadata.get("corpus"))
        stable_id = str(metadata.get("stable_id") or row.get("task_id") or "")
        manifest_row = manifests.get((corpus, stable_id))
        if manifest_row is None:
            scores.append(0.0)
            continue
        contract = dep.build_contract_from_manifest_row(
            corpus,
            manifest_row,
            manifest_path=str(manifest_row.get("_manifest_path") or ""),
        )
        result = dep.verify_candidate_source(
            contract,
            str(row["candidate_code"]),
            "exp3683_candidate",
            candidate_id=str(row.get("candidate_sha256") or stable_id),
        )
        scores.append(min(1.0, len(result.get("violations") or []) / 3.0))
    return scores


def check_preconditions(root: Path) -> list[JsonDict]:
    """Check balanced code corpus availability and detector module importability."""

    checks: list[JsonDict] = []
    try:
        rows, status = load_balanced_code_rows(root)
        labels = [bool(row.get("label")) for row in rows]
        balanced = bool(rows and any(labels) and not all(labels))
        checks.append(
            {
                "resource": "balanced_exp3658_code_corpus",
                "available": balanced,
                "detail": status,
                "n_examples": len(rows),
            }
        )
    except Exception as exc:
        checks.append(
            {
                "resource": "balanced_exp3658_code_corpus",
                "available": False,
                "detail": f"{type(exc).__name__}: {exc}",
                "n_examples": 0,
            }
        )
    checks.append(
        {
            "resource": "second_pair_detector_module",
            "available": all(
                callable(getattr(spd, name, None))
                for name in ("score_candidates", "stratified_train_holdout")
            ),
            "detail": "python/carnot/pipeline/second_pair_detector.py",
        }
    )
    return checks


def load_balanced_code_rows(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Load the balanced Exp 3658 code corpus rows."""

    artifact_path = root / "results/experiment_3658_code_generalization_second_corpus.json"
    corpus_path = root / "data/code_verification_corpus_v2.jsonl"
    if artifact_path.exists():
        artifact = _read_json_object(artifact_path)
        if artifact.get("second_code_corpus_path"):
            corpus_path = _repo_path(root, Path(str(artifact["second_code_corpus_path"])))
    if not corpus_path.exists():
        return [], {"status": "missing", "path": str(corpus_path)}
    rows = _read_jsonl(corpus_path)
    return rows, {"status": "loaded", "path": str(corpus_path), "n_examples": len(rows)}


def code_error_labels(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return 1 for buggy/error code and 0 for correct code."""

    return [0 if bool(row.get("label")) else 1 for row in rows]


def score_code_confidence(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Compute the Exp 3642 code confidence baseline scores."""

    from carnot.verify import corrected_cross_domain_remeasurement_v4 as exp3642

    return [float(score) for score in exp3642.score_code_confidence(rows)]


def run_score_candidates_e2e(
    root: Path,
    examples: Sequence[spd.LabeledDetectorExample],
) -> bool:
    """Call the shipped score surface on one code example."""

    code_examples = [example for example in examples if example.domain == "code"]
    if not code_examples:
        return False
    example = code_examples[0]
    try:
        result = spd.score_candidates(
            [
                spd.CandidateScoreInput(
                    candidate_id=f"exp3683-{example.example_id}",
                    domain="code",
                    text=example.example_id,
                    confidence_error=example.confidence_error,
                    ensemble_energy=example.ensemble_energy,
                )
            ],
            root=root,
            examples=examples,
        )
    except Exception:
        return False
    rows = result.get("scores")
    if not isinstance(rows, list) or not rows:
        return False
    value = rows[0].get("calibrated_error_score")
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3683 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS:
        raise ValueError("honest_verdict is not an accepted Exp 3683 terminal verdict")
    for field in (
        "code_operating_point_recovered",
        "module_code_path_updated",
        "e2e_test_passed",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3683 artifact fields."""

    payload = {
        "code_auroc_baseline": artifact.get("code_auroc_baseline"),
        "code_auroc_dependency_aware": artifact.get("code_auroc_dependency_aware"),
        "code_auroc_recalibrated": artifact.get("code_auroc_recalibrated"),
        "code_calibration_brier_ece_after": artifact.get(
            "code_calibration_brier_ece_after"
        ),
        "code_recall_at_fixed_fpr": artifact.get("code_recall_at_fixed_fpr"),
        "module_code_path_updated": artifact.get("module_code_path_updated"),
        "e2e_test_passed": artifact.get("e2e_test_passed"),
        "code_operating_point_recovered": artifact.get("code_operating_point_recovered"),
        "n_examples_code": artifact.get("n_examples_code"),
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
    """Build, validate, and persist the Exp 3683 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_artifact_from_metrics(
    root: Path | str,
    *,
    output_path: Path | str,
    **kwargs: Any,
) -> Path:
    """Persist a synthetic or pre-measured Exp 3683 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact_from_metrics(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _baseline_calibration(code_auroc_baseline: Mapping[str, Any]) -> Mapping[str, Any]:
    calibration = code_auroc_baseline.get("calibration_brier_ece")
    return calibration if isinstance(calibration, Mapping) else {}


def _has_both_classes(examples: Sequence[spd.LabeledDetectorExample]) -> bool:
    return len({example.label for example in examples}) == 2


def _require_length(scores: Sequence[float], expected: int, name: str) -> list[float]:
    if len(scores) != expected:
        raise ValueError(f"{name} returned {len(scores)} scores for {expected} rows")
    return [float(score) for score in scores]


def _precondition_n_examples(preconditions: Sequence[Mapping[str, Any]]) -> int:
    for item in preconditions:
        if item.get("resource") == "balanced_exp3658_code_corpus":
            return int(item.get("n_examples") or 0)
    return 0


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
    rows: list[JsonDict] = []
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


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    return round(float(value), 6)


__all__ = [
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "auroc_metric",
    "auroc_signal_excludes_chance",
    "build_artifact",
    "build_artifact_from_metrics",
    "calibration_bundle",
    "calibration_improved",
    "measure_code_recalibration",
    "validate_artifact",
    "write_artifact",
    "write_artifact_from_metrics",
]

"""Exp 3695 code-native verifier for the second-pair detector.

The code-native score is not a learned model. It is an honest deterministic
heuristic over real Python AST parsing, CodeExtractor structural findings, and
bounded execution probes through ``safe_exec_function``. The feature weights are
therefore a proxy operating rule, but the underlying signals actually inspect
and execute candidate code rather than relabeling the math-only verifier score.

Spec: REQ-SPOE-3695, SCENARIO-SPOE-3695.
"""

from __future__ import annotations

import ast
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

from carnot.pipeline.extract import CodeExtractor
from carnot.verify.python_types import safe_exec_function

from . import detector_code_operating_point_3683 as exp3683


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3695_code_native_verifier.json")
RANDOM_SEED = 3695
BOOTSTRAP_SEEDS = (3695, 3696, 3697)
DEFAULT_N_BOOTSTRAP = 200
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

VERDICT_RECOVERED = "complete: code_native_signal_recovered_beats_chance_floor"
VERDICT_MATH_ONLY = "complete: code_remains_math_only_code_native_signal_also_fails_earned"
VERDICT_BLOCKED = "complete: blocked_no_code_corpus_or_ast_tooling"
TERMINAL_VERDICTS = (VERDICT_RECOVERED, VERDICT_MATH_ONLY, VERDICT_BLOCKED)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "code_auroc_baseline",
    "code_native_auroc",
    "code_native_auroc_ci",
    "code_native_calibration_brier_ece",
    "code_native_recall_at_fixed_fpr",
    "code_native_verifier_implementation",
    "code_signal_recovered",
    "n_examples_code",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates if cached-only "
        "(no live-model marker); live_llm_inference only if a real generation step ran."
    ),
    "code_auroc_baseline": (
        "The exp3683 0.5 code baseline reconfirmed -- the chance floor to beat."
    ),
    "code_native_auroc": (
        "The code-NATIVE verifier AUROC -- the new signal; the headline number for this task."
    ),
    "code_native_auroc_ci": (
        "CI of the code-native AUROC -- must EXCLUDE 0.5 to count as signal."
    ),
    "code_native_calibration_brier_ece": (
        "Calibration of the code-native signal -- a deployable code operating point must be calibrated."
    ),
    "code_native_recall_at_fixed_fpr": (
        "Recall-at-FPR table a deployer reads for the code operating point."
    ),
    "code_native_verifier_implementation": (
        "Honest description of the AST/execution signal; real parsing/execution, not math relabeling."
    ),
    "code_signal_recovered": (
        "BARE bool. True iff code-native AUROC beats 0.5 with a CI excluding 0.5 "
        "AND calibration improves -- gates exp3696. STORE AS BARE true/false."
    ),
    "n_examples_code": "Sample-size rigor on the code corpus.",
    "adversarial_verify_clean": "True iff no critical flag.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

VERIFIER_IMPLEMENTATION = (
    "CodeNativeVerifier parses candidate Python with ast.parse, runs CodeExtractor "
    "for structural initialization/type/return findings, scores AST anomalies "
    "(parse failure, missing entry point, missing value return, literal None returns, "
    "early top-level returns and dead code after return), and executes deterministic "
    "signature-derived probes through safe_exec_function so runtime exceptions, None "
    "returns, and annotation type mismatches contribute an execution-trace signal. "
    "Weights are deterministic heuristic proxy weights; the inspected signals are "
    "code-native AST/runtime signals, not math-verifier scores."
)


@dataclass(frozen=True)
class CodeNativeScore:
    """One row's code-native score and transparent feature payload."""

    row_id: str
    score: float
    features: dict[str, float]
    detail: dict[str, Any]


@dataclass(frozen=True)
class OutcomeClassification:
    """Terminal verdict and recovered gate for Exp 3695."""

    terminal_verdict: str
    code_signal_recovered: bool


class CodeNativeVerifier:
    """AST/execution verifier with disclosed heuristic proxy weighting.

    The features come from real parsing and bounded execution. The final scalar
    is a hand-weighted proxy because this experiment is testing whether any
    code-native signal exists on the cached corpus before training a deployable
    learned code verifier.
    """

    def score_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[CodeNativeScore]:
        """Score rows in order with one output per input row."""

        return [self.score_row(row, idx) for idx, row in enumerate(rows)]

    def score_row(self, row: Mapping[str, Any], index: int = 0) -> CodeNativeScore:
        """Score one candidate row using AST and runtime trace features."""

        source = str(row.get("candidate_code") or row.get("code") or "")
        row_id = str(
            row.get("candidate_sha256")
            or row.get("task_id")
            or row.get("example_id")
            or f"row-{index}"
        )
        features = _empty_features()
        detail: JsonDict = {"entry_point": None, "runtime_errors": []}
        if not source.strip():
            features["parse_error"] = 1.0
            detail["parse_error"] = "empty_source"
            return CodeNativeScore(row_id, _score_features(features), features, detail)

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            features["parse_error"] = 1.0
            detail["parse_error"] = exc.msg
            return CodeNativeScore(row_id, _score_features(features), features, detail)

        features["ast_parseable"] = 1.0
        entry_point = _entry_point(row)
        function = _find_function(tree, entry_point)
        if function is None and entry_point:
            features["missing_entry_point"] = 1.0
        if function is None:
            function = _first_function(tree)
        detail["entry_point"] = function.name if function is not None else entry_point

        self._add_code_extractor_features(source, features, detail)
        if function is not None:
            _add_function_ast_features(function, features, detail)
            _add_runtime_trace_features(source, function, features, detail)
        else:
            features["missing_entry_point"] = 1.0
            features["missing_value_return"] = 1.0

        return CodeNativeScore(row_id, _score_features(features), features, detail)

    @staticmethod
    def _add_code_extractor_features(
        source: str,
        features: dict[str, float],
        detail: JsonDict,
    ) -> None:
        constraints = CodeExtractor().extract(source, domain="code")
        violations = [
            result
            for result in constraints
            if result.metadata.get("satisfied") is False
        ]
        undefined = [
            result
            for result in violations
            if result.metadata.get("kind") == "initialization"
        ]
        features["code_extractor_violation_rate"] = min(
            1.0,
            len(violations) / max(1, len(constraints)),
        )
        features["undefined_name_count_clamped"] = min(1.0, len(undefined) / 3.0)
        detail["n_code_extractor_constraints"] = len(constraints)
        detail["n_code_extractor_violations"] = len(violations)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3695 code-native verifier artifact from cached rows."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    preconditions = check_preconditions(root_path)
    if not all(bool(item["available"]) for item in preconditions):
        return build_artifact_from_metrics(
            blocked=True,
            code_auroc_baseline={},
            code_native_metric={},
            code_native_calibration_brier_ece={},
            code_native_recall_at_fixed_fpr={},
            n_examples_code=_precondition_n_examples(preconditions),
            adversarial_verify_clean=False,
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
            preconditions_checked=preconditions,
        )

    rows, _status = load_balanced_code_rows(root_path)
    labels = code_error_labels(rows)
    scored = CodeNativeVerifier().score_rows(rows)
    scores = [item.score for item in scored]
    native_metric = auroc_metric(labels, scores, seeds=seeds, n_bootstrap=n_bootstrap)
    baseline = reconfirm_code_baseline(root_path, seeds=seeds, n_bootstrap=n_bootstrap)
    calibrated = measure_code_native_calibration(
        labels=labels,
        scores=scores,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    artifact = build_artifact_from_metrics(
        blocked=False,
        code_auroc_baseline=baseline,
        code_native_metric=native_metric,
        code_native_calibration_brier_ece=calibrated["code_native_calibration_brier_ece"],
        code_native_recall_at_fixed_fpr=calibrated["code_native_recall_at_fixed_fpr"],
        n_examples_code=len(rows),
        adversarial_verify_clean=True,
        started_s=start,
        now_s=now_s,
        tests_run=tests_run,
        preconditions_checked=preconditions,
        extra={
            "code_native_auroc_metric": native_metric,
            "code_native_feature_summary": feature_summary(scored),
            "code_native_calibration_protocol": calibrated[
                "code_native_calibration_protocol"
            ],
        },
    )
    return artifact


def build_artifact_from_metrics(
    *,
    blocked: bool,
    code_auroc_baseline: Mapping[str, Any],
    code_native_metric: Mapping[str, Any],
    code_native_calibration_brier_ece: Mapping[str, Any],
    code_native_recall_at_fixed_fpr: Mapping[str, Any],
    n_examples_code: int,
    adversarial_verify_clean: bool,
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
        code_native_metric=code_native_metric,
        code_native_calibration_brier_ece=code_native_calibration_brier_ece,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3695_code_native_verifier",
        "schema": "carnot.code_native_verifier_3695.v1",
        "honest_verdict": classification.terminal_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "code_auroc_baseline": dict(code_auroc_baseline),
        "code_native_auroc": _metric_point(code_native_metric),
        "code_native_auroc_ci": _metric_ci(code_native_metric),
        "code_native_calibration_brier_ece": dict(code_native_calibration_brier_ece),
        "code_native_recall_at_fixed_fpr": dict(code_native_recall_at_fixed_fpr),
        "code_native_verifier_implementation": VERIFIER_IMPLEMENTATION,
        "code_signal_recovered": classification.code_signal_recovered,
        "n_examples_code": int(n_examples_code),
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _round(max(0.0, finished - start)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "code_auroc_baseline present AND code_native_auroc present AND "
                "code_native_auroc_ci present AND adversarial_verify_clean == true"
            ),
            "passed": bool(
                not blocked
                and code_auroc_baseline
                and _metric_point(code_native_metric) is not None
                and _metric_ci(code_native_metric) is not None
                and adversarial_verify_clean is True
            ),
            "principle": (
                "A code-native verdict requires the baseline, the code-native AUROC "
                "and its CI, and adversarial-clean status."
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
    code_native_metric: Mapping[str, Any],
    code_native_calibration_brier_ece: Mapping[str, Any],
) -> OutcomeClassification:
    """Map measured code-native statistics onto terminal outcomes."""

    if blocked:
        return OutcomeClassification(VERDICT_BLOCKED, False)
    recovered = bool(
        auroc_signal_excludes_chance(code_native_metric)
        and calibration_improved(
            _baseline_calibration(code_auroc_baseline),
            code_native_calibration_brier_ece,
        )
    )
    return OutcomeClassification(VERDICT_RECOVERED if recovered else VERDICT_MATH_ONLY, recovered)


def auroc_signal_excludes_chance(metric: Mapping[str, Any]) -> bool:
    """Return true only when AUROC is above chance and CI excludes 0.5."""

    point = metric.get("point")
    ci = metric.get("ci95")
    if point is None or not isinstance(ci, Sequence) or len(ci) != 2:
        return False
    return bool(float(point) > 0.5 and float(ci[0]) > 0.5)


def calibration_improved(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
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

    clean_labels, clean_scores = exp3683.spd.finite_label_scores(labels, scores)
    if not clean_labels or len(set(clean_labels)) < 2:
        return empty_metric(seeds)
    label_arr = np.asarray(clean_labels, dtype=np.int64)
    score_arr = np.asarray(clean_scores, dtype=np.float64)
    point = exp3683.spd.tie_aware_auroc(label_arr, score_arr)
    boot_values: list[float] = []
    seed_means: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(label_arr), size=len(label_arr))
            if len(set(label_arr[idx].tolist())) < 2:
                continue
            value = exp3683.spd.tie_aware_auroc(label_arr[idx], score_arr[idx])
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


def calibration_bundle(labels: Sequence[int], probabilities: Sequence[float]) -> JsonDict:
    """Return Brier and ECE for aligned finite labels and probabilities."""

    clean_labels, clean_probs = exp3683.spd.finite_label_scores(labels, probabilities)
    return {
        "brier": _round(exp3683.spd.brier_score(clean_labels, clean_probs)),
        "ece": _round(exp3683.spd.expected_calibration_error(clean_labels, clean_probs)),
    }


def measure_code_native_calibration(
    *,
    labels: Sequence[int],
    scores: Sequence[float],
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> JsonDict:
    """Fit code-only logistic calibration and evaluate held-out native scores."""

    examples = [
        exp3683.spd.LabeledDetectorExample(
            domain="code",
            label=int(label),
            ensemble_energy=float(score),
            confidence_error=float(score),
            example_id=f"code-native-{idx}",
        )
        for idx, (label, score) in enumerate(zip(labels, scores, strict=False))
    ]
    train, holdout = exp3683.spd.stratified_train_holdout(examples, seed=RANDOM_SEED)
    if not _has_both_classes(train) or not _has_both_classes(holdout):
        return {
            "code_native_calibration_brier_ece": {},
            "code_native_recall_at_fixed_fpr": {},
            "code_native_calibrated_auroc": empty_metric(seeds),
            "code_native_calibration_protocol": {
                "method": "logistic",
                "train_n": len(train),
                "holdout_n": len(holdout),
                "blocked_reason": "one_class_train_or_holdout",
            },
        }
    detector = exp3683.spd.CalibratedFusedDetector().fit(train)
    holdout_labels = [example.label for example in holdout]
    probabilities = detector.predict_proba(holdout)
    return {
        "code_native_calibration_brier_ece": calibration_bundle(
            holdout_labels,
            probabilities,
        ),
        "code_native_recall_at_fixed_fpr": recall_at_fixed_fpr_table(
            holdout_labels,
            probabilities,
        ),
        "code_native_calibrated_auroc": auroc_metric(
            holdout_labels,
            probabilities,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        ),
        "code_native_calibration_protocol": {
            "method": "logistic",
            "train_n": len(train),
            "holdout_n": len(holdout),
            "feature_names": list(detector.feature_names),
        },
    }


def recall_at_fixed_fpr_table(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    budgets: Sequence[float] = exp3683.spd.FPR_BUDGETS,
) -> JsonDict:
    """Return code-native recall at fixed false-positive-rate budgets."""

    clean_labels, clean_scores = exp3683.spd.finite_label_scores(labels, scores)
    rows = exp3683.spd.operating_points_at_fixed_fpr(clean_labels, clean_scores, budgets)
    return {
        _fpr_key(float(budget)): {
            "code_native_recall": row["recall"],
            "code_native_actual_fpr": row["actual_fpr"],
            "code_native_threshold": row["threshold"],
        }
        for budget, row in rows.items()
    }


def feature_summary(scored: Sequence[CodeNativeScore]) -> JsonDict:
    """Summarize feature activity without storing every row in the artifact."""

    if not scored:
        return {"n_scored": 0, "score_variance": 0.0, "mean_features": {}}
    feature_names = sorted(scored[0].features)
    return {
        "n_scored": len(scored),
        "score_variance": _round(float(np.var(np.asarray([row.score for row in scored])))),
        "n_execution_attempted": int(
            sum(row.features.get("execution_attempted", 0.0) > 0.0 for row in scored)
        ),
        "mean_features": {
            name: _round(float(np.mean([row.features.get(name, 0.0) for row in scored])))
            for name in feature_names
        },
    }


def reconfirm_code_baseline(
    root: Path,
    *,
    seeds: Sequence[int],
    n_bootstrap: int,
) -> JsonDict:
    """Re-measure the Exp 3683 chance-floor baseline on the balanced corpus."""

    examples, _status = exp3683.spd.load_cached_labeled_examples(
        root,
        use_balanced_code_corpus=True,
    )
    return exp3683.measure_baseline_code_operating_point(
        examples,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )


def check_preconditions(root: Path) -> list[JsonDict]:
    """Check balanced corpus availability and code AST/runtime tooling."""

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
    checks.append(_code_tooling_precondition())
    return checks


def load_balanced_code_rows(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Load the balanced Exp 3658 code corpus rows."""

    return exp3683.load_balanced_code_rows(root)


def code_error_labels(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return 1 for buggy/error code and 0 for correct code."""

    return exp3683.code_error_labels(rows)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3695 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS:
        raise ValueError("honest_verdict is not an accepted Exp 3695 terminal verdict")
    for field in ("code_signal_recovered", "adversarial_verify_clean"):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3695 artifact fields."""

    payload = {
        "code_auroc_baseline": artifact.get("code_auroc_baseline"),
        "code_native_auroc": artifact.get("code_native_auroc"),
        "code_native_auroc_ci": artifact.get("code_native_auroc_ci"),
        "code_native_calibration_brier_ece": artifact.get(
            "code_native_calibration_brier_ece"
        ),
        "code_native_recall_at_fixed_fpr": artifact.get("code_native_recall_at_fixed_fpr"),
        "code_signal_recovered": artifact.get("code_signal_recovered"),
        "n_examples_code": artifact.get("n_examples_code"),
        "adversarial_verify_clean": artifact.get("adversarial_verify_clean"),
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
    """Build, adversarial-check, validate, and persist the Exp 3695 artifact."""

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


def write_artifact_from_metrics(
    root: Path | str,
    *,
    output_path: Path | str,
    **kwargs: Any,
) -> Path:
    """Persist a synthetic or pre-measured Exp 3695 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact_from_metrics(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_adversarial_verify_report(path: Path) -> JsonDict:
    """Run scripts/adversarial_verify.py against an artifact path."""

    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_3695", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(path))


def compact_adversarial_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep the adversarial report small and deterministic in the artifact."""

    flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    return {"flag_count": len(flags), "flags": flags}


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true when no adversarial flag is critical."""

    flags = report.get("flags", [])
    if not isinstance(flags, Sequence):
        return False
    return not any(str(flag.get("severity", "")).lower() == "critical" for flag in flags)


def _add_function_ast_features(
    function: ast.FunctionDef,
    features: dict[str, float],
    detail: JsonDict,
) -> None:
    value_returns = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    none_returns = [
        node
        for node in value_returns
        if isinstance(node.value, ast.Constant) and node.value.value is None
    ]
    if not value_returns:
        features["missing_value_return"] = 1.0
    features["return_none_literal_rate"] = min(1.0, len(none_returns) / max(1, len(value_returns)))
    features["early_unconditional_return"] = 1.0 if _early_unconditional_return(function) else 0.0
    features["dead_code_after_return"] = 1.0 if _dead_code_after_return(function) else 0.0
    features["forbidden_call"] = 1.0 if _has_forbidden_call(function) else 0.0
    branch_loop_nodes = sum(
        isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp))
        for node in ast.walk(function)
    )
    detail["n_value_returns"] = len(value_returns)
    detail["n_branch_loop_nodes"] = int(branch_loop_nodes)


def _add_runtime_trace_features(
    source: str,
    function: ast.FunctionDef,
    features: dict[str, float],
    detail: JsonDict,
) -> None:
    probes = _probe_args(function)
    if not probes:
        return
    expected = _annotation_kind(function.returns)
    errors = 0
    none_returns = 0
    type_mismatches = 0
    runtime_errors: list[str] = []
    for args in probes:
        result, error = safe_exec_function(source, function.name, args, timeout=0.5)
        if error is not None:
            errors += 1
            runtime_errors.append(type(error).__name__)
            continue
        if result is None:
            none_returns += 1
        if expected and not _value_matches_kind(result, expected):
            type_mismatches += 1
    n = max(1, len(probes))
    features["execution_attempted"] = 1.0
    features["runtime_exception_rate"] = errors / n
    features["runtime_return_none_rate"] = none_returns / n
    features["runtime_type_mismatch_rate"] = type_mismatches / n
    detail["n_runtime_probes"] = len(probes)
    detail["runtime_errors"] = runtime_errors[:5]


def _empty_features() -> dict[str, float]:
    return {
        "ast_parseable": 0.0,
        "parse_error": 0.0,
        "missing_entry_point": 0.0,
        "code_extractor_violation_rate": 0.0,
        "undefined_name_count_clamped": 0.0,
        "missing_value_return": 0.0,
        "return_none_literal_rate": 0.0,
        "early_unconditional_return": 0.0,
        "dead_code_after_return": 0.0,
        "forbidden_call": 0.0,
        "execution_attempted": 0.0,
        "runtime_exception_rate": 0.0,
        "runtime_return_none_rate": 0.0,
        "runtime_type_mismatch_rate": 0.0,
    }


def _score_features(features: Mapping[str, float]) -> float:
    weights = {
        "parse_error": 0.85,
        "missing_entry_point": 0.35,
        "code_extractor_violation_rate": 0.15,
        "undefined_name_count_clamped": 0.12,
        "missing_value_return": 0.25,
        "return_none_literal_rate": 0.28,
        "early_unconditional_return": 0.14,
        "dead_code_after_return": 0.12,
        "forbidden_call": 0.15,
        "runtime_exception_rate": 0.25,
        "runtime_return_none_rate": 0.30,
        "runtime_type_mismatch_rate": 0.20,
    }
    return _round(min(1.0, sum(float(features.get(name, 0.0)) * weight for name, weight in weights.items())))


def _entry_point(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    value = metadata.get("entry_point") or row.get("entry_point") or row.get("function_name")
    return str(value) if value else None


def _find_function(tree: ast.AST, entry_point: str | None) -> ast.FunctionDef | None:
    if not entry_point:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    return None


def _first_function(tree: ast.AST) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _early_unconditional_return(function: ast.FunctionDef) -> bool:
    return len(function.body) > 1 and isinstance(function.body[0], ast.Return)


def _dead_code_after_return(function: ast.FunctionDef) -> bool:
    seen_return = False
    for statement in function.body:
        if seen_return:
            return True
        if isinstance(statement, ast.Return):
            seen_return = True
    return False


def _has_forbidden_call(function: ast.FunctionDef) -> bool:
    forbidden = {"eval", "exec", "open", "input", "__import__"}
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden:
                return True
    return False


def _probe_args(function: ast.FunctionDef) -> list[tuple[Any, ...]]:
    params = [arg for arg in function.args.args if arg.arg != "self"]
    if not params:
        return [()]
    values_by_param = [_values_for_kind(_annotation_kind(arg.annotation)) for arg in params]
    probes: list[tuple[Any, ...]] = []
    for idx in range(3):
        probes.append(tuple(values[min(idx, len(values) - 1)] for values in values_by_param))
    return probes


def _annotation_kind(annotation: ast.AST | None) -> str:
    if annotation is None:
        return "int"
    if isinstance(annotation, ast.Name):
        return _kind_from_text(annotation.id)
    if isinstance(annotation, ast.Subscript):
        return _annotation_kind(annotation.value)
    if isinstance(annotation, ast.Attribute):
        return _kind_from_text(annotation.attr)
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return _kind_from_text(annotation.value)
    return "int"


def _kind_from_text(text: str) -> str:
    normalized = text.lower()
    if "dict" in normalized:
        return "dict"
    if "tuple" in normalized:
        return "tuple"
    if "list" in normalized or "sequence" in normalized:
        return "list"
    if "bool" in normalized:
        return "bool"
    if "float" in normalized:
        return "float"
    if "str" in normalized:
        return "str"
    if "none" in normalized:
        return "none"
    return "int"


def _values_for_kind(kind: str) -> list[Any]:
    return {
        "dict": [{"key": 1}, {}, {"key": 0}],
        "tuple": [(1,), (), (0, 1)],
        "list": [[1, 2], [], [0]],
        "bool": [True, False, True],
        "float": [1.0, 0.5, -1.0],
        "str": ["x", "", "( )"],
        "none": [None, None, None],
        "int": [1, 0, -1],
    }.get(kind, [1, 0, -1])


def _value_matches_kind(value: Any, kind: str) -> bool:
    if kind == "none":
        return value is None
    expected = {
        "dict": dict,
        "tuple": tuple,
        "list": list,
        "bool": bool,
        "float": (float, int),
        "str": str,
        "int": int,
    }.get(kind)
    return True if expected is None else isinstance(value, expected)


def _code_tooling_precondition() -> JsonDict:
    try:
        constraints = CodeExtractor().extract("def f(x: int) -> int:\n    return x\n", domain="code")
        result, error = safe_exec_function("def f(x: int) -> int:\n    return x\n", "f", (1,))
        available = bool(constraints and error is None and result == 1)
        return {
            "resource": "code_extractor_ast_runtime_tooling",
            "available": available,
            "detail": "CodeExtractor, ast.parse, safe_exec_function",
        }
    except Exception as exc:
        return {
            "resource": "code_extractor_ast_runtime_tooling",
            "available": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _has_both_classes(examples: Sequence[Any]) -> bool:
    return len({example.label for example in examples}) == 2


def _baseline_calibration(code_auroc_baseline: Mapping[str, Any]) -> Mapping[str, Any]:
    calibration = code_auroc_baseline.get("calibration_brier_ece")
    return calibration if isinstance(calibration, Mapping) else {}


def _metric_point(metric: Mapping[str, Any]) -> float | None:
    point = metric.get("point")
    return None if point is None else _round(float(point))


def _metric_ci(metric: Mapping[str, Any]) -> list[float] | None:
    ci = metric.get("ci95")
    if not isinstance(ci, Sequence) or len(ci) != 2:
        return None
    return [_round(float(ci[0])), _round(float(ci[1]))]


def _precondition_n_examples(preconditions: Sequence[Mapping[str, Any]]) -> int:
    for item in preconditions:
        if item.get("resource") == "balanced_exp3658_code_corpus":
            return int(item.get("n_examples") or 0)
    return 0


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _fpr_key(value: float) -> str:
    return f"{value:.2f}"


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    return round(float(value), 6)


__all__ = [
    "BOOTSTRAP_SEEDS",
    "CodeNativeScore",
    "CodeNativeVerifier",
    "INFERENCE_SUBSTRATE",
    "OUTPUT_REL_PATH",
    "RANDOM_SEED",
    "REQUIRED_ARTIFACT_FIELDS",
    "adversarial_report_is_clean",
    "auroc_metric",
    "auroc_signal_excludes_chance",
    "build_artifact",
    "build_artifact_from_metrics",
    "calibration_bundle",
    "calibration_improved",
    "check_preconditions",
    "code_error_labels",
    "feature_summary",
    "load_balanced_code_rows",
    "measure_code_native_calibration",
    "recall_at_fixed_fpr_table",
    "reconfirm_code_baseline",
    "reproducibility_checksum",
    "validate_artifact",
    "write_artifact",
    "write_artifact_from_metrics",
]

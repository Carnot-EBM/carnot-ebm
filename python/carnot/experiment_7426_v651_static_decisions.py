"""Measure static decisions for human-annotated source support.

The target is whether a response is supported by its supplied source under the
RAGTruth annotations. It is not general truth and is not an entailment model.
Spec refs: REQ-AUTO-7426 and SCENARIO-AUTO-7426-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7382_v648_decision_protocol import _clopper_pearson_upper
from carnot.experiment_7385_v648_decision_training import fit_affine_transform
from carnot.experiment_7412_v650_source_features import (
    MAX_STEPS,
    SOURCE_FEATURE_NAMES,
    extract_feature_row,
    fit_gibbs_head,
    fit_logistic_control,
    gibbs_energy,
    initial_gibbs_checkpoint,
    probability_from_energy,
)
from carnot.experiment_7423_v651_annotated_protocol import (
    EVALUATOR_TOKEN,
    ProtocolReaders,
    reload_corpus,
)
from carnot.experiment_7425_v651_spline_prototype import (
    dense_design_vector,
    fit_quantile_knots,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    sidecar_reference,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
EXPERIMENT_ID = "exp7426-v651-static-decisions"
SCHEMA = "carnot.exp7426.v651.static_decisions.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7426_v651_static_decisions.json")
RAW_DIR = Path("results/raw/experiment_7426_v651_static_decisions")
MODULE_PATH = Path("python/carnot/experiment_7426_v651_static_decisions.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7426_v651_static_decisions.py")
TEST_PATH = Path("tests/python/test_experiment_7426_v651_static_decisions.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
ANNOTATED_PATH = Path("results/experiment_7423_v651_annotated_protocol.json")
SPLINE_PATH = Path("results/experiment_7425_v651_spline_prototype.json")
CORPUS_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
CORPUS_PATH = CORPUS_DIR / "corpus_manifest.json"

EXPECTED_ANNOTATED_SHA256 = (
    "sha256:4d91bd186aa7e0fd320694cc0fbc066a2c84ed85c823beccba47c357f903ba8e"
)
EXPECTED_SPLINE_SHA256 = "sha256:362874817bd332e64e54ddd527b581070ffba06c59ef021c6c104febfff47bce"
EXPECTED_CORPUS_SHA256 = "sha256:a065b2a64f2926c5bade1fd3495b1cbff23bd3b3c967bd104b973a6b118e30c0"
EXPECTED_CORPUS_MANIFEST_HASH = (
    "sha256:5e09e80789b5579194d22018a9b336558da4668cfa1facbf58d2e6c403fccdb8"
)

TRAINING_SEEDS = (65_101, 65_102, 65_103, 65_104, 65_105)
ARMS = (
    "training_prevalence",
    "raw_l2_logistic",
    "gibbs_6_4_1",
    "sparse_spline_49",
    "dense_spline_logistic",
)
CONDITIONS = ("full_source", "source_masked", "source_swapped")
PARTITIONS = ("fit", "probability_calibration", "policy_calibration", "final_test")
ACCEPT_THRESHOLDS = (0.95, 0.975, 0.99)
REJECT_THRESHOLDS = (0.01, 0.05, 0.10)
THRESHOLD_PAIRS = tuple(
    (accept, reject) for accept in ACCEPT_THRESHOLDS for reject in REJECT_THRESHOLDS
)
FAMILYWISE_ALPHA = 0.05
SIMULTANEOUS_POLICY_TESTS = len(ARMS) * len(THRESHOLD_PAIRS) * 2
ALPHA_PER_POLICY_TEST = FAMILYWISE_ALPHA / SIMULTANEOUS_POLICY_TESTS
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_510_426
PRIMARY_ARM = "sparse_spline_49"
PRIMARY_CONTROLS = ("raw_l2_logistic", "gibbs_6_4_1")
PARITY_TOLERANCE = 1e-10
LABEL_AUTHORITY = "human_annotation_source_support"
INFERENCE_SUBSTRATE = "no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
MAX_SHARD_BYTES = 8 * 1024 * 1024

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7412_v650_source_features.py"),
    Path("python/carnot/experiment_7413_v650_source_calibration.py"),
    Path("python/carnot/experiment_7423_v651_annotated_protocol.py"),
    Path("python/carnot/experiment_7425_v651_spline_prototype.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    SPEC_PATH,
    ANNOTATED_PATH,
    SPLINE_PATH,
    CORPUS_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_row_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "decision_capture_complete_score",
    "decision_value_score",
    "checkpoint_manifest",
    "policy_certificates",
    "grouped_intervals",
    "small_ebm_training",
)


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return an aware UTC timestamp for a measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - real execution boundary.
    """Print a flushed phase boundary or pending-operation heartbeat."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7426] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object and treat malformed external bytes as absent."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Name one exact prerequisite and its observed value."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def upstream_field_checks(
    annotated: Mapping[str, Any], spline: Mapping[str, Any]
) -> list[JsonDict]:
    """Check both roadmap prerequisites without starting dependent work."""

    allowed = ["positive", "circular_positive", "null"]
    rows: list[JsonDict] = []
    definitions = (
        (
            "annotated_protocol_ready",
            "exp7423-annotated-protocol",
            ANNOTATED_PATH,
            "annotated_protocol_ready_score",
            "==",
            1,
            annotated.get("annotated_protocol_ready_score"),
            None,
        ),
        (
            "annotated_verdict_allowed",
            "exp7423-annotated-protocol",
            ANNOTATED_PATH,
            "verdict_class",
            "in",
            allowed,
            annotated.get("verdict_class"),
            annotated.get("verdict_class") in allowed,
        ),
        (
            "annotated_unflagged",
            "exp7423-annotated-protocol",
            ANNOTATED_PATH,
            "flagged_adversarial",
            "==",
            False,
            annotated.get("flagged_adversarial"),
            None,
        ),
        (
            "spline_prototype_ready",
            "exp7425-spline-prototype",
            SPLINE_PATH,
            "spline_prototype_ready_score",
            "==",
            1,
            spline.get("spline_prototype_ready_score"),
            None,
        ),
        (
            "spline_verdict_allowed",
            "exp7425-spline-prototype",
            SPLINE_PATH,
            "verdict_class",
            "in",
            allowed,
            spline.get("verdict_class"),
            spline.get("verdict_class") in allowed,
        ),
        (
            "spline_unflagged",
            "exp7425-spline-prototype",
            SPLINE_PATH,
            "flagged_adversarial",
            "==",
            False,
            spline.get("flagged_adversarial"),
            None,
        ),
        (
            "annotated_milestone_identity",
            "exp7423-annotated-protocol",
            ANNOTATED_PATH,
            "milestone",
            "==",
            MILESTONE,
            annotated.get("milestone"),
            None,
        ),
        (
            "spline_milestone_identity",
            "exp7425-spline-prototype",
            SPLINE_PATH,
            "milestone",
            "==",
            MILESTONE,
            spline.get("milestone"),
            None,
        ),
    )
    for check, upstream, path, field, operator, expected, observed, passed in definitions:
        rows.append(
            _precondition(
                check,
                upstream,
                path.as_posix(),
                field,
                operator,
                expected,
                observed,
                passed=passed,
            )
        )
    return rows


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate source bytes, upstream fields, protocol, and exclusion state."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "==",
                "readable_nonempty_bytes",
                observed,
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": None,
            }
    annotated = _load_object(root / ANNOTATED_PATH)
    spline = _load_object(root / SPLINE_PATH)
    corpus = _load_object(root / CORPUS_PATH)
    hashes_to_check = {
        ANNOTATED_PATH: EXPECTED_ANNOTATED_SHA256,
        SPLINE_PATH: EXPECTED_SPLINE_SHA256,
        CORPUS_PATH: EXPECTED_CORPUS_SHA256,
    }
    for relative, expected in hashes_to_check.items():
        observed = sha256_file(root / relative) if (root / relative).is_file() else None
        checks.append(
            _precondition(
                f"exact_hash:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "sha256",
                "==",
                expected,
                observed,
            )
        )
    checks.extend(upstream_field_checks(annotated, spline))
    checks.extend(
        [
            _precondition(
                "corpus_manifest_identity",
                "exp7423-annotated-protocol",
                CORPUS_PATH.as_posix(),
                "manifest_hash",
                "==",
                EXPECTED_CORPUS_MANIFEST_HASH,
                corpus.get("manifest_hash"),
            ),
            _precondition(
                "sealed_training_seeds",
                "exp7423-annotated-protocol",
                ANNOTATED_PATH.as_posix(),
                "protocol_plan.training_seeds",
                "==",
                list(TRAINING_SEEDS),
                (annotated.get("protocol_plan") or {}).get("training_seeds"),
            ),
            _precondition(
                "sealed_training_budget",
                "exp7423-annotated-protocol",
                ANNOTATED_PATH.as_posix(),
                "protocol_plan.training_budget_steps",
                "==",
                MAX_STEPS,
                (annotated.get("protocol_plan") or {}).get("training_budget_steps"),
            ),
            _precondition(
                "spline_parameter_count",
                "exp7425-spline-prototype",
                SPLINE_PATH.as_posix(),
                "energy_definition.trainable_parameter_count",
                "==",
                49,
                (spline.get("energy_definition") or {}).get("trainable_parameter_count"),
            ),
        ]
    )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "==",
            "REQ-AUTO-7426",
            "REQ-AUTO-7426" if "REQ-AUTO-7426" in spec_text else None,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            "==",
            False,
            "experiment_id: 7426" in exclusion,
        )
    )
    if ANNOTATED_PATH.as_posix() in hashes:
        hashes[ANNOTATED_PATH.as_posix()]["original_flagged_adversarial"] = annotated.get(
            "flagged_adversarial"
        )
    if SPLINE_PATH.as_posix() in hashes:
        hashes[SPLINE_PATH.as_posix()]["original_flagged_adversarial"] = spline.get(
            "flagged_adversarial"
        )
    loaded = {
        "annotated_protocol": annotated,
        "spline_prototype": spline,
        "corpus_manifest": corpus,
    }
    return checks, hashes, loaded


def _features_from_text(row: Mapping[str, Any], source_text: str) -> JsonDict:
    """Recompute the shipped six proxies from predictor-only bounded text."""

    projected = extract_feature_row(
        {
            "row_key": row["row_key"],
            "group_id": row["group_id"],
            "partition": row["partition"],
            "question": "",
            "context": source_text,
            "answer": row.get("response_text", ""),
            "sentence": row.get("response_text", ""),
        }
    )
    return deepcopy(projected["source_features"])


def condition_predictors(predictors: Sequence[Mapping[str, Any]], condition: str) -> list[JsonDict]:
    """Apply one source condition without reading evaluator labels."""

    if condition not in CONDITIONS:
        raise ValueError("registered condition required")
    copied = [deepcopy(dict(row)) for row in predictors]
    if condition == "full_source":
        return copied
    if condition == "source_masked":
        for row in copied:
            row["source_text"] = ""
            row["features"] = _features_from_text(row, "")
        return copied
    cohorts: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in copied:
        cohorts[(str(row.get("partition")), str(row.get("task_type")))].append(row)
    for cohort in cohorts.values():
        groups = sorted({str(row.get("group_id")) for row in cohort})
        if len(groups) < 2:
            raise ValueError("source swap requires at least two groups per role and domain")
        representative = {
            group: min(
                (row for row in cohort if str(row.get("group_id")) == group),
                key=lambda item: str(item.get("row_key")),
            )
            for group in groups
        }
        source_by_group = {
            group: str(representative[group].get("source_text") or "") for group in groups
        }
        donor = {group: groups[(index + 1) % len(groups)] for index, group in enumerate(groups)}
        for row in cohort:
            source = source_by_group[donor[str(row["group_id"])]]
            row["source_text"] = source
            row["features"] = _features_from_text(row, source)
    return copied


def join_predictor_labels(
    predictors: Sequence[Mapping[str, Any]], evaluators: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Join the masked predictor view to the explicit human-label authority."""

    by_key = {str(row.get("row_key")): row for row in evaluators}
    if len(by_key) != len(evaluators):
        raise ValueError("evaluator row keys must be unique")
    output: list[JsonDict] = []
    for predictor in predictors:
        key = str(predictor.get("row_key") or "")
        evaluator = by_key.get(key)
        if evaluator is None:
            raise ValueError(f"evaluator row missing:{key}")
        label = evaluator.get("primary_label")
        sensitivity = evaluator.get("implicit_true_excluded_label")
        if label not in {0, 1} or sensitivity not in {0, 1}:
            raise ValueError(f"human label invalid:{key}")
        if evaluator.get("group_id") != predictor.get("group_id") or evaluator.get(
            "partition"
        ) != predictor.get("partition"):
            raise ValueError(f"predictor evaluator identity mismatch:{key}")
        output.append(
            {
                **deepcopy(dict(predictor)),
                "label": int(label),
                "sensitivity_label": int(sensitivity),
                "authority": LABEL_AUTHORITY,
                "authority_limit": "fallible human source-support judgment; not formal truth",
                "source_group": predictor["group_id"],
            }
        )
    if set(by_key) != {str(row.get("row_key")) for row in predictors}:
        raise ValueError("evaluator contains rows outside predictor view")
    return output


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return the exact finite six-feature matrix for all same-information arms."""

    vectors: list[list[float]] = []
    for row in rows:
        features = row.get("features")
        if not isinstance(features, Mapping) or set(features) != set(SOURCE_FEATURE_NAMES):
            raise ValueError("features must contain the exact six registered values")
        vector = [float(features[name]) for name in SOURCE_FEATURE_NAMES]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in vector):
            raise ValueError("six registered features must be finite and bounded")
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1:] != (6,):
        raise ValueError("feature matrix must have six columns")
    return matrix


def logit(probability: float) -> float:
    """Convert a finite probability to a bounded logit for affine fitting."""

    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    clipped = min(max(value, 1e-15), 1.0 - 1e-15)
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    """Evaluate sigmoid without overflow for local batch scoring."""

    positive = values >= 0.0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _adam_update(
    value: np.ndarray,
    gradient: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen float64 Adam update to one numeric array."""

    new_first = 0.9 * first + 0.1 * gradient
    new_second = 0.999 * second + 0.001 * gradient**2
    corrected_first = new_first / (1.0 - 0.9**step)
    corrected_second = new_second / (1.0 - 0.999**step)
    updated = value - 0.01 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return updated, new_first, new_second


def fit_spline_pair(
    features: Any,
    labels: Any,
    *,
    seed: int,
    steps: int = MAX_STEPS,
    heartbeat: Callable[[int], None] | None = None,
) -> JsonDict:
    """Fit sparse-block and dense fixed-basis logits from identical state."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1:] != (6,) or y.shape != (len(x),):
        raise ValueError("spline training requires a six-column matrix and matching labels")
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("spline training values must be finite")
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("spline training requires both labels")
    if not 0 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between zero and {MAX_STEPS}")
    knots = fit_quantile_knots(x)
    design = np.asarray([dense_design_vector(row, knots)[0] for row in x], dtype=np.float64)
    rng = np.random.default_rng(seed)
    initial = rng.normal(0.0, 0.01, size=48).astype(np.float64)
    sparse_coef = initial.reshape(6, 8).copy()
    dense_coef = initial.copy()
    sparse_bias = np.asarray(0.0)
    dense_bias = np.asarray(0.0)
    sparse_first = np.zeros_like(sparse_coef)
    sparse_second = np.zeros_like(sparse_coef)
    dense_first = np.zeros_like(dense_coef)
    dense_second = np.zeros_like(dense_coef)
    sparse_bias_first = np.asarray(0.0)
    sparse_bias_second = np.asarray(0.0)
    dense_bias_first = np.asarray(0.0)
    dense_bias_second = np.asarray(0.0)
    loss_trace: list[JsonDict] = []
    max_probability_gap = 0.0
    max_gradient_gap = 0.0
    for update in range(steps + 1):
        sparse_logits = design @ sparse_coef.reshape(-1) + sparse_bias
        dense_logits = design @ dense_coef + dense_bias
        sparse_probabilities = _sigmoid_array(sparse_logits)
        dense_probabilities = _sigmoid_array(dense_logits)
        probability_gap = float(np.max(np.abs(sparse_probabilities - dense_probabilities)))
        max_probability_gap = max(max_probability_gap, probability_gap)
        sparse_loss = float(
            np.mean(np.logaddexp(0.0, sparse_logits) - y * sparse_logits)
            + 0.001 * np.sum(sparse_coef**2)
        )
        dense_loss = float(
            np.mean(np.logaddexp(0.0, dense_logits) - y * dense_logits)
            + 0.001 * np.sum(dense_coef**2)
        )
        loss_trace.append(
            {
                "step": update,
                "sparse_loss": sparse_loss,
                "dense_loss": dense_loss,
                "probability_gap": probability_gap,
            }
        )
        if update == steps:
            break
        sparse_residual = (sparse_probabilities - y) / len(y)
        dense_residual = (dense_probabilities - y) / len(y)
        sparse_gradient = (design.T @ sparse_residual).reshape(6, 8) + 0.002 * sparse_coef
        dense_gradient = design.T @ dense_residual + 0.002 * dense_coef
        max_gradient_gap = max(
            max_gradient_gap,
            float(np.max(np.abs(sparse_gradient.reshape(-1) - dense_gradient))),
        )
        sparse_bias_gradient = np.asarray(np.sum(sparse_residual))
        dense_bias_gradient = np.asarray(np.sum(dense_residual))
        sparse_coef, sparse_first, sparse_second = _adam_update(
            sparse_coef, sparse_gradient, sparse_first, sparse_second, update + 1
        )
        dense_coef, dense_first, dense_second = _adam_update(
            dense_coef, dense_gradient, dense_first, dense_second, update + 1
        )
        sparse_bias, sparse_bias_first, sparse_bias_second = _adam_update(
            sparse_bias,
            sparse_bias_gradient,
            sparse_bias_first,
            sparse_bias_second,
            update + 1,
        )
        dense_bias, dense_bias_first, dense_bias_second = _adam_update(
            dense_bias,
            dense_bias_gradient,
            dense_bias_first,
            dense_bias_second,
            update + 1,
        )
        if heartbeat is not None and (update + 1) % 100 == 0:
            heartbeat(update + 1)
    sparse_initial = {
        "coef": initial.reshape(6, 8).tolist(),
        "bias": 0.0,
        "knots": knots.tolist(),
    }
    dense_initial = {"coef": initial.tolist(), "bias": 0.0, "knots": knots.tolist()}
    sparse_checkpoint = {
        "kind": "sparse_spline_49",
        "knots": knots.tolist(),
        "coef": sparse_coef.tolist(),
        "bias": float(sparse_bias),
        "initial_coef": initial.reshape(6, 8).tolist(),
        "initial_bias": 0.0,
        "initial_state_sha256": canonical_hash(sparse_initial),
    }
    dense_checkpoint = {
        "kind": "dense_spline_logistic",
        "knots": knots.tolist(),
        "coef": dense_coef.tolist(),
        "bias": float(dense_bias),
        "initial_coef": initial.tolist(),
        "initial_bias": 0.0,
        "initial_state_sha256": canonical_hash(dense_initial),
    }
    sparse_checkpoint["fitted_state_sha256"] = canonical_hash(sparse_checkpoint)
    dense_checkpoint["fitted_state_sha256"] = canonical_hash(dense_checkpoint)
    parameter_gap = float(np.max(np.abs(sparse_coef.reshape(-1) - dense_coef)))
    parameter_gap = max(parameter_gap, abs(float(sparse_bias) - float(dense_bias)))
    return {
        "seed": seed,
        "updates": steps,
        "parameter_count": 49,
        "objective": "natural_prevalence_bernoulli_loss_l2_0.001",
        "loss_trace": loss_trace,
        "sparse_checkpoint": sparse_checkpoint,
        "dense_checkpoint": dense_checkpoint,
        "max_parameter_gap": parameter_gap,
        "max_probability_gap": max_probability_gap,
        "max_gradient_gap": max_gradient_gap,
    }


def spline_parity_errors(pair: Mapping[str, Any]) -> list[str]:
    """Recompute parity from numeric checkpoints instead of trusting flags."""

    errors: list[str] = []
    try:
        sparse = pair["sparse_checkpoint"]
        dense = pair["dense_checkpoint"]
        sparse_coef = np.asarray(sparse["coef"], dtype=np.float64).reshape(-1)
        dense_coef = np.asarray(dense["coef"], dtype=np.float64).reshape(-1)
        gap = max(
            float(np.max(np.abs(sparse_coef - dense_coef))),
            abs(float(sparse["bias"]) - float(dense["bias"])),
        )
        if gap > PARITY_TOLERANCE:
            errors.append("spline_parameter_parity_mismatch")
        if pair.get("max_probability_gap", math.inf) > PARITY_TOLERANCE:
            errors.append("spline_probability_parity_mismatch")
        if pair.get("max_gradient_gap", math.inf) > PARITY_TOLERANCE:
            errors.append("spline_gradient_parity_mismatch")
    except (KeyError, TypeError, ValueError):
        errors.append("spline_parity_state_invalid")
    return errors


def _predict_state(state: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    """Score one frozen seed state over a six-feature matrix."""

    arm = str(state["arm"])
    checkpoint = state["checkpoint"]
    if arm == "training_prevalence":
        return np.full(len(features), float(checkpoint["probability"]), dtype=np.float64)
    if arm == "raw_l2_logistic":
        logits = features @ np.asarray(checkpoint["coef"], dtype=np.float64) + float(
            checkpoint["bias"]
        )
        return _sigmoid_array(logits)
    if arm == "gibbs_6_4_1":
        return np.asarray(
            [probability_from_energy(gibbs_energy(checkpoint, row)) for row in features],
            dtype=np.float64,
        )
    knots = np.asarray(checkpoint["knots"], dtype=np.float64)
    design = np.asarray([dense_design_vector(row, knots)[0] for row in features], dtype=np.float64)
    coefficients = np.asarray(checkpoint["coef"], dtype=np.float64).reshape(-1)
    return _sigmoid_array(design @ coefficients + float(checkpoint["bias"]))


def fit_ensemble_calibration(
    raw_probabilities: Any, labels: Sequence[int], *, steps: int = MAX_STEPS
) -> JsonDict:
    """Average five seed probabilities before the only affine calibration."""

    raw = np.asarray(raw_probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != len(TRAINING_SEEDS) or y.shape != (raw.shape[1],):
        raise ValueError("calibration requires five seed rows and matching labels")
    if not np.all(np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("calibration probabilities must be finite and bounded")
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("calibration requires both labels")
    means = np.mean(raw, axis=0)
    logits = [logit(float(value)) for value in means]
    affine = fit_affine_transform(logits, y.astype(int).tolist(), steps=steps)
    return {
        "seed_count": len(TRAINING_SEEDS),
        "mean_raw_probabilities": means.tolist(),
        "calibration_logits": logits,
        "affine": affine,
        "fit_order": "five_seed_probability_mean_then_affine_logit",
    }


def _apply_affine(probabilities: np.ndarray, affine: Mapping[str, Any]) -> np.ndarray:
    """Apply one fitted affine logit transform to bounded probabilities."""

    logits = np.asarray([logit(float(value)) for value in probabilities], dtype=np.float64)
    return _sigmoid_array(float(affine["slope"]) * logits + float(affine["intercept"]))


def exact_risk_certificate(harmful: Sequence[int], *, risk_budget: float) -> JsonDict:
    """Use the registered one-sided exact bound and 90-test correction."""

    if any(value not in {0, 1} for value in harmful):
        raise ValueError("harmful outcomes must be binary")
    selected = len(harmful)
    failures = sum(harmful)
    if selected == 0:
        return {
            "selected_groups": 0,
            "harmful_outcomes": 0,
            "upper_risk_bound": None,
            "risk_budget": risk_budget,
            "alpha_familywise": FAMILYWISE_ALPHA,
            "alpha_per_test": ALPHA_PER_POLICY_TEST,
            "simultaneous_test_count": SIMULTANEOUS_POLICY_TESTS,
            "certified": False,
            "action_enabled": False,
            "reason": "no_selected_groups_no_certificate",
        }
    upper = _clopper_pearson_upper(failures, selected, ALPHA_PER_POLICY_TEST)
    certified = upper <= risk_budget
    return {
        "selected_groups": selected,
        "harmful_outcomes": failures,
        "upper_risk_bound": upper,
        "risk_budget": risk_budget,
        "alpha_familywise": FAMILYWISE_ALPHA,
        "alpha_per_test": ALPHA_PER_POLICY_TEST,
        "simultaneous_test_count": SIMULTANEOUS_POLICY_TESTS,
        "certified": certified,
        "action_enabled": certified,
        "reason": "exact_bound_within_budget" if certified else "exact_bound_exceeds_budget",
    }


def typed_support_decision(probability: float | None, policy: Mapping[str, Any]) -> JsonDict:
    """Map support probability to accept, reject, or safe escalation."""

    if probability is None or not math.isfinite(float(probability)) or not 0 <= probability <= 1:
        return {"action": "escalate", "reason": "invalid_probability"}
    value = float(probability)
    if value >= float(policy["accept_threshold"]):
        if policy.get("accept_enabled") is True:
            return {"action": "accept", "reason": "certified_high_source_support"}
        return {"action": "escalate", "reason": "accept_uncertified"}
    if value <= float(policy["reject_threshold"]):
        if policy.get("reject_enabled") is True:
            return {"action": "reject", "reason": "certified_low_source_support"}
        return {"action": "escalate", "reason": "reject_uncertified"}
    return {"action": "escalate", "reason": "between_thresholds"}


def _representatives(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Choose the lowest row key in each group without reading its label."""

    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group = str(row.get("group_id") or "")
        key = str(row.get("row_key") or "")
        if not group or not key:
            raise ValueError("policy rows require group and row identity")
        if group not in selected or key < str(selected[group]["row_key"]):
            selected[group] = row
    return [selected[group] for group in sorted(selected)]


def select_policy(
    rows: Sequence[Mapping[str, Any]], probabilities: Mapping[str, float]
) -> tuple[list[JsonDict], JsonDict]:
    """Certify all nine pairs on one label-blind row per source group."""

    representatives = _representatives(rows)
    candidates: list[JsonDict] = []
    for index, (accept_threshold, reject_threshold) in enumerate(THRESHOLD_PAIRS):
        provisional = []
        for row in representatives:
            value = float(probabilities[str(row["row_key"])])
            if value >= accept_threshold:
                provisional.append("accept")
            elif value <= reject_threshold:
                provisional.append("reject")
            else:
                provisional.append("escalate")
        accept_certificate = exact_risk_certificate(
            [
                int(row["label"] == 0)
                for row, action in zip(representatives, provisional, strict=True)
                if action == "accept"
            ],
            risk_budget=0.05,
        )
        reject_certificate = exact_risk_certificate(
            [
                int(row["label"] == 1)
                for row, action in zip(representatives, provisional, strict=True)
                if action == "reject"
            ],
            risk_budget=0.10,
        )
        enabled = [
            action
            if (action == "accept" and accept_certificate["action_enabled"])
            or (action == "reject" and reject_certificate["action_enabled"])
            else "escalate"
            for action in provisional
        ]
        correct = sum(
            (action == "accept" and row["label"] == 1) or (action == "reject" and row["label"] == 0)
            for row, action in zip(representatives, enabled, strict=True)
        )
        harmful = sum(
            (action == "accept" and row["label"] == 0) or (action == "reject" and row["label"] == 1)
            for row, action in zip(representatives, enabled, strict=True)
        )
        coverage = sum(action != "escalate" for action in enabled) / max(len(enabled), 1)
        candidates.append(
            {
                "threshold_index": index,
                "accept_threshold": accept_threshold,
                "reject_threshold": reject_threshold,
                "accept_certificate": accept_certificate,
                "reject_certificate": reject_certificate,
                "representative_groups": len(representatives),
                "coverage": coverage,
                "utility": (correct - harmful) / max(len(enabled), 1),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["coverage"]),
            float(row["utility"]),
            -int(row["threshold_index"]),
        ),
    )
    return candidates, {
        "threshold_index": selected["threshold_index"],
        "accept_threshold": selected["accept_threshold"],
        "reject_threshold": selected["reject_threshold"],
        "accept_enabled": selected["accept_certificate"]["action_enabled"],
        "reject_enabled": selected["reject_certificate"]["action_enabled"],
        "accept_certificate": deepcopy(selected["accept_certificate"]),
        "reject_certificate": deepcopy(selected["reject_certificate"]),
        "representative_groups": len(representatives),
        "selection_partition": "policy_calibration",
        "selection_rule": "coverage_then_utility_then_registered_order",
    }


def _partition_rows(rows: Sequence[Mapping[str, Any]], partition: str) -> list[Mapping[str, Any]]:
    """Select one sealed role and require both human support labels."""

    selected = [row for row in rows if row.get("partition") == partition]
    if not selected or {int(row["label"]) for row in selected} != {0, 1}:
        raise ValueError(f"{partition} must contain both labels")
    return selected


def _state_row(
    arm: str, seed: int, checkpoint: Mapping[str, Any], fit: Mapping[str, Any]
) -> JsonDict:
    """Create one plain numeric seed state with its training receipt."""

    return {
        "arm": arm,
        "seed": seed,
        "checkpoint": deepcopy(dict(checkpoint)),
        "initial_checkpoint": deepcopy(dict(fit["initial_checkpoint"])),
        "initial_checkpoint_sha256": fit["initial_checkpoint_sha256"],
        "final_checkpoint_sha256": canonical_hash(checkpoint),
        "updates": int(fit["updates"]),
        "parameter_count": int(fit["parameter_count"]),
        "loss_trace": deepcopy(list(fit["loss_trace"])),
        "objective": fit["objective"],
    }


def fit_condition(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    steps: int = MAX_STEPS,
    emit: Callable[[str, str, int, int], None] | None = None,
) -> JsonDict:
    """Fit all five arms, then calibrate and select policy by sealed role."""

    if condition not in CONDITIONS:
        raise ValueError("registered condition required")
    partitions = {partition: _partition_rows(rows, partition) for partition in PARTITIONS[:-1]}
    matrices = {
        partition: feature_matrix(partition_rows)
        for partition, partition_rows in partitions.items()
    }
    labels = {
        partition: np.asarray([row["label"] for row in partition_rows], dtype=np.float64)
        for partition, partition_rows in partitions.items()
    }
    states_by_arm: dict[str, list[JsonDict]] = {arm: [] for arm in ARMS}
    parity_rows: list[JsonDict] = []
    completed = 0
    total = len(ARMS) * len(TRAINING_SEEDS)
    prevalence = float(np.mean(labels["fit"]))
    prevalence_loss = float(
        -np.mean(
            labels["fit"] * math.log(prevalence) + (1.0 - labels["fit"]) * math.log1p(-prevalence)
        )
    )
    for seed in TRAINING_SEEDS:
        prevalence_initial = {"probability": prevalence}
        fit_receipt = {
            "initial_checkpoint": prevalence_initial,
            "initial_checkpoint_sha256": canonical_hash(prevalence_initial),
            "updates": 0,
            "parameter_count": 1,
            "loss_trace": [{"step": 0, "loss": prevalence_loss}],
            "objective": "training_prevalence_constant",
        }
        states_by_arm["training_prevalence"].append(
            _state_row("training_prevalence", seed, {"probability": prevalence}, fit_receipt)
        )
        completed += 1
        if emit:
            emit("training_prevalence", "after_training", completed, total)

        if emit:
            emit("raw_l2_logistic", "before_training", completed, total)
        logistic_initial = {
            "coef": np.random.default_rng(seed).normal(0.0, 0.01, size=6).tolist(),
            "bias": 0.0,
        }
        logistic = fit_logistic_control(matrices["fit"], labels["fit"], seed=seed, steps=steps)
        states_by_arm["raw_l2_logistic"].append(
            _state_row(
                "raw_l2_logistic",
                seed,
                logistic["checkpoint"],
                {
                    "initial_checkpoint": logistic_initial,
                    "initial_checkpoint_sha256": canonical_hash(logistic_initial),
                    "updates": logistic["update_count"],
                    "parameter_count": 7,
                    "loss_trace": logistic["loss_curve"],
                    "objective": logistic["objective"],
                },
            )
        )
        completed += 1
        if emit:
            emit("raw_l2_logistic", "after_training", completed, total)

        if emit:
            emit("gibbs_6_4_1", "before_training", completed, total)
        gibbs_initial = initial_gibbs_checkpoint(seed=seed, input_dim=6)
        gibbs = fit_gibbs_head(matrices["fit"], labels["fit"], seed=seed, steps=steps)
        states_by_arm["gibbs_6_4_1"].append(
            _state_row(
                "gibbs_6_4_1",
                seed,
                gibbs["checkpoint"],
                {
                    "initial_checkpoint": gibbs_initial,
                    "initial_checkpoint_sha256": canonical_hash(gibbs_initial),
                    "updates": gibbs["update_count"],
                    "parameter_count": 33,
                    "loss_trace": gibbs["loss_curve"],
                    "objective": gibbs["objective"],
                },
            )
        )
        completed += 1
        if emit:
            emit("gibbs_6_4_1", "after_training", completed, total)

        if emit:
            emit("sparse_spline_49", "before_training", completed, total)
        pair = fit_spline_pair(matrices["fit"], labels["fit"], seed=seed, steps=steps)
        parity_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "max_parameter_gap": pair["max_parameter_gap"],
                "max_probability_gap": pair["max_probability_gap"],
                "max_gradient_gap": pair["max_gradient_gap"],
                "passed": not spline_parity_errors(pair),
            }
        )
        sparse_receipt = {
            "initial_checkpoint": {
                "coef": pair["sparse_checkpoint"]["initial_coef"],
                "bias": pair["sparse_checkpoint"]["initial_bias"],
                "knots": pair["sparse_checkpoint"]["knots"],
            },
            "initial_checkpoint_sha256": pair["sparse_checkpoint"]["initial_state_sha256"],
            "updates": pair["updates"],
            "parameter_count": 49,
            "loss_trace": pair["loss_trace"],
            "objective": pair["objective"],
        }
        dense_receipt = {
            **sparse_receipt,
            "initial_checkpoint": {
                "coef": pair["dense_checkpoint"]["initial_coef"],
                "bias": pair["dense_checkpoint"]["initial_bias"],
                "knots": pair["dense_checkpoint"]["knots"],
            },
            "initial_checkpoint_sha256": pair["dense_checkpoint"]["initial_state_sha256"],
        }
        states_by_arm["sparse_spline_49"].append(
            _state_row("sparse_spline_49", seed, pair["sparse_checkpoint"], sparse_receipt)
        )
        completed += 1
        if emit:
            emit("sparse_spline_49", "after_training", completed, total)
        states_by_arm["dense_spline_logistic"].append(
            _state_row("dense_spline_logistic", seed, pair["dense_checkpoint"], dense_receipt)
        )
        completed += 1
        if emit:
            emit("dense_spline_logistic", "after_training", completed, total)

    arm_fits: JsonDict = {}
    policy_certificates: list[JsonDict] = []
    for arm in ARMS:
        seed_states = states_by_arm[arm]
        raw_calibration = np.asarray(
            [_predict_state(state, matrices["probability_calibration"]) for state in seed_states]
        )
        calibration = fit_ensemble_calibration(
            raw_calibration, labels["probability_calibration"].astype(int).tolist(), steps=steps
        )
        raw_policy = np.asarray(
            [_predict_state(state, matrices["policy_calibration"]) for state in seed_states]
        )
        mean_policy = np.mean(raw_policy, axis=0)
        calibrated_policy = _apply_affine(mean_policy, calibration["affine"])
        probability_map = {
            str(row["row_key"]): float(value)
            for row, value in zip(partitions["policy_calibration"], calibrated_policy, strict=True)
        }
        candidates, policy = select_policy(partitions["policy_calibration"], probability_map)
        policy_certificates.append(
            {"condition": condition, "arm": arm, "candidates": candidates, "selected": policy}
        )
        calibration_summary = {
            key: value
            for key, value in calibration.items()
            if key not in {"mean_raw_probabilities", "calibration_logits"}
        }
        calibration_summary["input_probability_map_sha256"] = canonical_hash(
            {
                "row_keys": [row["row_key"] for row in partitions["probability_calibration"]],
                "mean_raw_probabilities": calibration["mean_raw_probabilities"],
                "labels": labels["probability_calibration"].astype(int).tolist(),
            }
        )
        arm_fits[arm] = {
            "arm": arm,
            "condition": condition,
            "seed_states": seed_states,
            "calibration": calibration_summary,
            "policy": policy,
            "partition_hashes": {
                partition: canonical_hash(
                    [
                        {
                            "row_key": row["row_key"],
                            "group_id": row["group_id"],
                            "label": row["label"],
                        }
                        for row in partitions[partition]
                    ]
                )
                for partition in ("fit", "probability_calibration", "policy_calibration")
            },
        }
    return {
        "condition": condition,
        "arms": arm_fits,
        "parity_rows": parity_rows,
        "policy_certificates": policy_certificates,
        "fit_row_count": len(partitions["fit"]),
    }


def _log_loss(label: int, probability: float) -> float:
    """Return finite binary log loss for one human support label."""

    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def ensemble_metric_row(
    *,
    arm: str,
    row_key: str,
    group_id: str,
    task_type: str,
    label: int,
    probability: float,
    action: str,
    condition: str = "full_source",
) -> JsonDict:
    """Build one independently reducible calibrated ensemble metric row."""

    return {
        "row_type": "ensemble_metric",
        "condition": condition,
        "arm": arm,
        "row_key": row_key,
        "group_id": group_id,
        "source_group": group_id,
        "task_type": task_type,
        "label": int(label),
        "authority": LABEL_AUTHORITY,
        "probability": float(probability),
        "action": action,
        "brier_contribution": (float(probability) - int(label)) ** 2,
        "log_loss_contribution": _log_loss(int(label), float(probability)),
    }


def score_official_rows(
    fit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Score each official row with frozen states and retain every seed."""

    if any(row.get("partition") != "final_test" for row in rows):
        raise ValueError("official scoring accepts final_test rows only")
    matrix = feature_matrix(rows)
    probability_rows: list[JsonDict] = []
    ensemble_rows: list[JsonDict] = []
    condition = str(fit["condition"])
    for arm in ARMS:
        arm_fit = fit["arms"][arm]
        seed_states = arm_fit["seed_states"]
        raw = np.asarray([_predict_state(state, matrix) for state in seed_states])
        mean_raw = np.mean(raw, axis=0)
        calibrated = _apply_affine(mean_raw, arm_fit["calibration"]["affine"])
        for row_index, row in enumerate(rows):
            decision = typed_support_decision(float(calibrated[row_index]), arm_fit["policy"])
            ensemble_rows.append(
                ensemble_metric_row(
                    condition=condition,
                    arm=arm,
                    row_key=str(row["row_key"]),
                    group_id=str(row["group_id"]),
                    task_type=str(row["task_type"]),
                    label=int(row["label"]),
                    probability=float(calibrated[row_index]),
                    action=decision["action"],
                )
            )
            for seed_index, state in enumerate(seed_states):
                probability_rows.append(
                    {
                        "row_type": "seed_probability",
                        "status": "completed",
                        "condition": condition,
                        "arm": arm,
                        "seed": state["seed"],
                        "row_key": row["row_key"],
                        "group_id": row["group_id"],
                        "source_group": row["group_id"],
                        "task_type": row["task_type"],
                        "raw_seed_probability": float(raw[seed_index, row_index]),
                        "ensemble_raw_probability": float(mean_raw[row_index]),
                        "probability": float(calibrated[row_index]),
                        "label": int(row["label"]),
                        "sensitivity_label": int(row["sensitivity_label"]),
                        "authority": LABEL_AUTHORITY,
                        "action": decision["action"],
                        "action_reason": decision["reason"],
                        "brier_contribution": (float(calibrated[row_index]) - int(row["label"]))
                        ** 2,
                        "log_loss_contribution": _log_loss(
                            int(row["label"]), float(calibrated[row_index])
                        ),
                        "cost": {"current_llm_calls": 0, "cpu_scoring_duration_s": 0.0},
                    }
                )
    return probability_rows, ensemble_rows


def excluded_probability_rows(dispositions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep every excluded unit visible without inventing a score or action."""

    return [
        {
            "row_type": "excluded_probability",
            "status": "excluded",
            "condition": condition,
            "arm": arm,
            "seed": seed,
            "row_key": None,
            "response_id": disposition.get("response_id"),
            "source_group": disposition.get("group_id"),
            "disposition": disposition.get("reason"),
            "authority": LABEL_AUTHORITY,
            "probability": None,
            "label": None,
            "action": None,
            "brier_contribution": None,
            "log_loss_contribution": None,
            "cost": {"current_llm_calls": 0, "cpu_scoring_duration_s": 0.0},
        }
        for disposition in dispositions
        for condition in CONDITIONS
        for arm in ARMS
        for seed in TRAINING_SEEDS
    ]


def write_detail_shards(
    directory: Path, rows: Sequence[Mapping[str, Any]], *, prefix: str = "rows"
) -> list[JsonDict]:
    """Write deterministic hash-bound JSONL shards below the 20 MiB limit."""

    directory.mkdir(parents=True, exist_ok=True)
    manifests: list[JsonDict] = []
    chunk: list[bytes] = []
    size = 0

    def flush() -> None:
        nonlocal chunk, size
        if not chunk:
            return
        path = directory / f"{prefix}-{len(manifests):03d}.jsonl"
        payload = b"".join(chunk)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        manifests.append(
            {
                "path": path.name,
                "rows": len(chunk),
                "size_bytes": len(payload),
                "sha256": "sha256:" + __import__("hashlib").sha256(payload).hexdigest(),
            }
        )
        chunk = []
        size = 0

    for row in rows:
        encoded = (
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_SHARD_BYTES:
            raise ValueError("single detail row exceeds shard size")
        if chunk and size + len(encoded) > MAX_SHARD_BYTES:
            flush()
        chunk.append(encoded)
        size += len(encoded)
    flush()
    return manifests


def load_detail_shards(directory: Path, manifests: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reload exact detailed evidence and reject any changed byte."""

    output: list[JsonDict] = []
    for manifest in manifests:
        path = directory / str(manifest.get("path") or "")
        if not path.is_file() or sha256_file(path) != manifest.get("sha256"):
            raise ValueError(f"detail shard hash mismatch:{manifest.get('path')}")
        if path.stat().st_size != manifest.get("size_bytes"):
            raise ValueError(f"detail shard size mismatch:{manifest.get('path')}")
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as error:
            raise ValueError(f"detail shard JSON invalid:{path.name}") from error
        if len(rows) != manifest.get("rows") or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"detail shard row mismatch:{path.name}")
        output.extend(rows)
    return output


def reduce_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce proper scores, typed risk, coverage, and domain support by arm."""

    by_arm: JsonDict = {}
    for arm in ARMS:
        selected = [row for row in rows if row.get("arm") == arm]
        if not selected:
            continue
        accepts = [row for row in selected if row.get("action") == "accept"]
        rejects = [row for row in selected if row.get("action") == "reject"]
        decisions = [*accepts, *rejects]
        domains: JsonDict = {}
        for task_type in sorted({str(row.get("task_type")) for row in selected}):
            domain = [row for row in selected if row.get("task_type") == task_type]
            domains[task_type] = {
                "rows": len(domain),
                "groups": len({row.get("group_id") for row in domain}),
                "label_counts": {
                    "0": sum(row.get("label") == 0 for row in domain),
                    "1": sum(row.get("label") == 1 for row in domain),
                },
                "brier": float(np.mean([row["brier_contribution"] for row in domain])),
                "log_loss": float(np.mean([row["log_loss_contribution"] for row in domain])),
            }
        by_arm[arm] = {
            "rows": len(selected),
            "groups": len({row.get("group_id") for row in selected}),
            "brier": float(np.mean([row["brier_contribution"] for row in selected])),
            "log_loss": float(np.mean([row["log_loss_contribution"] for row in selected])),
            "coverage": len(decisions) / len(selected),
            "accept_count": len(accepts),
            "reject_count": len(rejects),
            "escalate_count": len(selected) - len(decisions),
            "false_accept_risk": (
                sum(row["label"] == 0 for row in accepts) / len(accepts) if accepts else None
            ),
            "false_reject_risk": (
                sum(row["label"] == 1 for row in rejects) / len(rejects) if rejects else None
            ),
            "empirical_decision_risk": (
                sum(
                    (row["action"] == "accept" and row["label"] == 0)
                    or (row["action"] == "reject" and row["label"] == 1)
                    for row in decisions
                )
                / len(decisions)
                if decisions
                else None
            ),
            "by_domain": domains,
        }
    return {"by_arm": by_arm}


def _bootstrap_interval(values: np.ndarray) -> JsonDict:
    """Return one observed mean and percentile interval."""

    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def grouped_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> JsonDict:
    """Resample connected source groups for the two primary spline contrasts."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    groups = sorted({str(row.get("group_id")) for row in rows})
    if not groups:
        raise ValueError("bootstrap requires scored source groups")
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        if not arm_rows:
            raise ValueError(f"bootstrap arm missing:{arm}")
        values = {"brier": [], "log_loss": [], "coverage": []}
        for group in groups:
            group_rows = [row for row in arm_rows if str(row.get("group_id")) == group]
            if not group_rows:
                raise ValueError(f"bootstrap paired group missing:{arm}:{group}")
            values["brier"].append(
                float(np.mean([row["brier_contribution"] for row in group_rows]))
            )
            values["log_loss"].append(
                float(np.mean([row["log_loss_contribution"] for row in group_rows]))
            )
            values["coverage"].append(
                float(np.mean([row["action"] != "escalate" for row in group_rows]))
            )
        vectors[arm] = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    raw: dict[str, dict[str, np.ndarray]] = {}
    for control in PRIMARY_CONTROLS:
        raw[control] = {}
        for metric in ("brier", "log_loss", "coverage"):
            delta = vectors[PRIMARY_ARM][metric] - vectors[control][metric]
            raw[control][metric] = np.mean(delta[indices], axis=1)
    centered = np.column_stack(
        [raw[control]["brier"] - np.mean(raw[control]["brier"]) for control in PRIMARY_CONTROLS]
    )
    upper_critical = float(np.quantile(np.max(centered, axis=1), 0.95))
    contrasts: JsonDict = {}
    for control in PRIMARY_CONTROLS:
        observed = float(np.mean(vectors[PRIMARY_ARM]["brier"] - vectors[control]["brier"]))
        contrasts[control] = {
            "brier_delta": {
                "mean": observed,
                "simultaneous_upper_95": observed + upper_critical,
                "marginal_ci95": [
                    float(np.quantile(raw[control]["brier"], 0.025)),
                    float(np.quantile(raw[control]["brier"], 0.975)),
                ],
            },
            "log_loss_delta": _bootstrap_interval(raw[control]["log_loss"]),
            "coverage_delta": _bootstrap_interval(raw[control]["coverage"]),
        }
    return {
        "draws": draws,
        "seed": seed,
        "resampling_unit": "connected_source_group",
        "effective_groups": len(groups),
        "seed_average_before_resampling": True,
        "primary_contrasts": list(PRIMARY_CONTROLS),
        "simultaneous_correction_count": len(PRIMARY_CONTROLS),
        "simultaneous_method": "max_centered_statistic_two_primary_brier_contrasts",
        "upper_critical_value": upper_critical,
        "contrasts": contrasts,
    }


def _risk(metric: Mapping[str, Any]) -> float:
    """Map no decisions to zero empirical harm while coverage stays separate."""

    value = metric.get("empirical_decision_risk")
    return 0.0 if value is None else float(value)


def reduce_decision_value(
    intervals: Mapping[str, Any], metrics: Mapping[str, Any], *, parity_passed: bool
) -> JsonDict:
    """Apply the prespecified predictive and typed-policy conjunction."""

    contrasts = intervals.get("contrasts") or {}
    by_arm = metrics.get("by_arm") or {}
    primary = by_arm.get(PRIMARY_ARM) or {}
    controls = [by_arm.get(control) or {} for control in PRIMARY_CONTROLS]
    checks = {
        "spline_basis_parity": bool(parity_passed),
        "simultaneous_brier_improvement": all(
            float(
                (contrasts.get(control) or {})
                .get("brier_delta", {})
                .get("simultaneous_upper_95", math.inf)
            )
            < 0.0
            for control in PRIMARY_CONTROLS
        ),
        "non_worse_log_loss": all(
            float(primary.get("log_loss", math.inf)) <= float(control.get("log_loss", -math.inf))
            for control in controls
        ),
        "certified_coverage_at_least_0_25": float(primary.get("coverage", 0.0)) >= 0.25,
        "coverage_no_lower_than_controls": all(
            float(primary.get("coverage", 0.0)) >= float(control.get("coverage", math.inf))
            for control in controls
        ),
        "empirical_risk_no_increase": all(_risk(primary) <= _risk(control) for control in controls),
    }
    passed = all(checks.values())
    return {"checks": checks, "passed": passed, "decision_value_score": int(passed)}


def _unit_rows(
    fits: Sequence[Mapping[str, Any]], checkpoint_manifest: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Summarize every condition, arm, and seed without duplicating weights."""

    checkpoints = {(row["condition"], row["arm"], row["seed"]): row for row in checkpoint_manifest}
    rows: list[JsonDict] = []
    for fit in fits:
        condition = str(fit["condition"])
        for arm in ARMS:
            for state in fit["arms"][arm]["seed_states"]:
                checkpoint = checkpoints[(condition, arm, state["seed"])]
                rows.append(
                    {
                        "comparative_unit": f"{condition}:{arm}:{state['seed']}",
                        "condition": condition,
                        "arm": arm,
                        "seed": state["seed"],
                        "status": "completed",
                        "attempted": True,
                        "completed": True,
                        "failed": False,
                        "censored": False,
                        "checkpoint_sha256": checkpoint["sha256"],
                        "fit_partition": "fit",
                        "calibration_partition": "probability_calibration",
                        "policy_partition": "policy_calibration",
                    }
                )
    return rows


def write_checkpoints(
    root: Path, directory: Path, fits: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Persist all numeric initial, fitted, calibration, and policy states."""

    directory.mkdir(parents=True, exist_ok=True)
    manifests: list[JsonDict] = []
    for fit in fits:
        condition = str(fit["condition"])
        for arm in ARMS:
            arm_fit = fit["arms"][arm]
            for state in arm_fit["seed_states"]:
                payload = {
                    "schema": "carnot.exp7426.numeric_checkpoint.v1",
                    "condition": condition,
                    "arm": arm,
                    "seed": state["seed"],
                    "checkpoint": state["checkpoint"],
                    "initial_checkpoint": state["initial_checkpoint"],
                    "initial_checkpoint_sha256": state["initial_checkpoint_sha256"],
                    "final_checkpoint_sha256": state["final_checkpoint_sha256"],
                    "updates": state["updates"],
                    "parameter_count": state["parameter_count"],
                    "loss_trace": state["loss_trace"],
                    "objective": state["objective"],
                    "calibration": arm_fit["calibration"],
                    "policy": arm_fit["policy"],
                    "partition_hashes": arm_fit["partition_hashes"],
                }
                path = directory / f"{condition}--{arm}--{state['seed']}.json"
                atomic_json(path, payload)
                try:
                    label = path.relative_to(root).as_posix()
                except ValueError:
                    label = str(path.resolve())
                manifests.append(
                    {
                        "condition": condition,
                        "arm": arm,
                        "seed": state["seed"],
                        "path": label,
                        "sha256": sha256_file(path),
                        "byte_size": path.stat().st_size,
                        "initial_checkpoint_sha256": state["initial_checkpoint_sha256"],
                        "final_checkpoint_sha256": state["final_checkpoint_sha256"],
                        "updates": state["updates"],
                        "parameter_count": state["parameter_count"],
                    }
                )
    return manifests


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing affected and terminal receipt by exact name."""

    names = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)).issubset(names)


def _field_principles() -> dict[str, str]:
    """Explain required fields while their values stay machine-readable."""

    defaults = {
        key: "Keep this field plain, replayable, and independently reducible."
        for key in REQUIRED_FIELDS
    }
    defaults.update(
        {
            "schema": "Use a versioned plain top-level schema with experiment identity and terminal status.",
            "run_date": "Use 20260919 with actual UTC start, end, and monotonic timing.",
            "preconditions_checked": "Record exact paths, identities, and observed values before dependent work.",
            "MODEL_SPECS": "Remain empty because this run invokes no current LLM.",
            "model_invoked": "Become true only after an actual current model attempt.",
            "invocation_counts": "Count only owned current loads and generations in each terminal state.",
            "inference_substrate": "Use a truthful string and keep device detail in the detail field.",
            "inference_substrate_class": "Describe current compute and never historical inference or padding.",
            "execution_venue": "Use the closed host value; device identity is separate.",
            "duration_s": "Measure current work and separate validation, model, and cold-start phases.",
            "phase_spans": "Retain real phase times, progress boundaries, and checkpoint counts.",
            "random_seed": "Freeze all fit, policy, and bootstrap seeds.",
            "reproducibility_checksum": "Bind code, protocol, input bytes, raw rows, and validation scope.",
            "source_artifact_hashes": "Bind exact inputs and retain each upstream adversarial flag.",
            "rows": "Retain every condition, arm, seed, and terminal unit state.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
            "acceptance_gate_results": "Separate validity, completion, safety, and scientific benefit.",
            "gate_check_summary": "Name exact blocked fields including missing, zero, and null observations.",
            "verifier_is_oracle": "Human source support is fallible and is not general semantic truth.",
            "honest_verdict": "Start complete findings with complete_ and unavailable inputs with blocked_.",
            "verdict_class": "Use the closed terminal enum; valid no-benefit data are null.",
            "flagged_adversarial": "Critical findings deny readiness and scientific value.",
            "validation_receipts": "Retain exact argv, environment, exits, durations, names, and hashed logs.",
            "field_principles": "Explain field intent separately from ordinary scalar values.",
            "promotion_score": "Always remain zero; no rollout or generator update is authorized.",
            "decision_capture_complete_score": "One means complete valid paired measurement, including a null.",
            "decision_value_score": "One requires every predictive and typed-policy clause.",
            "checkpoint_manifest": "Bind numeric initial state, fitted state, probability maps, and policies.",
            "policy_certificates": "Retain exact counts, simultaneous bounds, assumptions, and disabled actions.",
            "grouped_intervals": "Retain paired group contrasts and their simultaneous correction.",
            "small_ebm_training": "Report five seeds, actual steps, loss traces, and small parameter counts.",
        }
    )
    return defaults


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Record one explicit gate and the failure mode it prevents."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(
    gates: Sequence[Mapping[str, Any]], blocked: Mapping[str, Any] | None = None
) -> JsonDict:
    """Keep scientific nulls separate from invalid or blocked evidence."""

    required_failures = [
        deepcopy(dict(row))
        for row in gates
        if row.get("category") != "scientific_benefit" and row.get("passed") is not True
    ]
    return {
        "required_checks_passed": not required_failures and blocked is None,
        "failed_required_checks": required_failures,
        "scientific_benefit_passed": all(
            row.get("passed") is True
            for row in gates
            if row.get("category") == "scientific_benefit"
        ),
        "blocked_upstream": blocked.get("upstream") if blocked else None,
        "blocked_path": blocked.get("path") if blocked else None,
        "blocked_check": blocked.get("check") if blocked else None,
        "blocked_field": blocked.get("field") if blocked else None,
        "blocked_expected": blocked.get("expected") if blocked else None,
        "blocked_observed": blocked.get("observed") if blocked else None,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding wall time and its own slot."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    for field in (
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "phase_spans",
        "validation_receipts",
        "validation_duration_s",
        "cold_start_duration_s",
    ):
        payload.pop(field, None)
    return canonical_hash(payload)


def _receipt(
    *,
    started_ns: int,
    ended_ns: int,
    spans: Sequence[Mapping[str, Any]],
    training: Mapping[str, Any],
    root: Path,
) -> JsonDict:
    """Build zero-current-model provenance with typed historical sidecars."""

    sidecars = []
    for relative in (ANNOTATED_PATH, SPLINE_PATH):
        path = root / relative
        if path.is_file():
            sidecars.append(sidecar_reference(path, root=root, scope="historical_model_receipts"))
    return build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "device": "host_cpu",
            "current_llm": False,
            "small_numeric_heads": True,
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=sidecars,
        phase_spans=spans,
        small_ebm_training=training,
    )


def build_artifact(
    *,
    root: Path,
    fits: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Sequence[Mapping[str, Any]],
    detail_shards: Sequence[Mapping[str, Any]],
    metric_shards: Sequence[Mapping[str, Any]],
    detail_directory: Path,
    metric_directory: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_ns: int,
    ended_ns: int,
    excluded_count: int,
    fixture: bool = False,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one terminal record from raw shards and frozen numeric states."""

    metric_rows = load_detail_shards(metric_directory, metric_shards)
    full_rows = [row for row in metric_rows if row.get("condition") == "full_source"]
    metrics = reduce_metrics(full_rows)
    draws = 200 if fixture else BOOTSTRAP_DRAWS
    intervals = grouped_bootstrap(full_rows, draws=draws, seed=BOOTSTRAP_SEED)
    parity_rows = [row for fit in fits for row in fit["parity_rows"]]
    parity_passed = bool(parity_rows) and all(row.get("passed") is True for row in parity_rows)
    value = reduce_decision_value(intervals, metrics, parity_passed=parity_passed)
    validation_passed = _validation_passed(validation_receipts)
    expected_units = len(fits) * len(ARMS) * len(TRAINING_SEEDS)
    units_complete = len(checkpoint_manifest) == expected_units
    detail_count = sum(int(row["rows"]) for row in detail_shards)
    metric_count = sum(int(row["rows"]) for row in metric_shards)
    expected_metric_count = len(fits) * len(ARMS) * int(metrics["by_arm"][PRIMARY_ARM]["rows"])
    capture = int(
        all(row.get("passed") is True for row in preconditions)
        and units_complete
        and parity_passed
        and detail_count > 0
        and metric_count == expected_metric_count
        and validation_passed
        and not flagged_adversarial
    )
    decision_value = int(capture == 1 and value["decision_value_score"] == 1)
    verdict_class = "positive" if decision_value else "null"
    honest = (
        "complete_positive_static_source_support_decision_value"
        if decision_value
        else "complete_null_static_source_support_no_registered_decision_benefit"
    )
    rows = _unit_rows(fits, checkpoint_manifest)
    small_training = {
        "performed": True,
        "receipt_class": "small_ebm_training",
        "current_llm_calls": 0,
        "generator_weights_fitted": False,
        "seeds": list(TRAINING_SEEDS),
        "maximum_steps": MAX_STEPS,
        "actual_steps_by_unit": [
            {
                "condition": row["condition"],
                "arm": row["arm"],
                "seed": row["seed"],
                "steps": row["updates"],
                "parameter_count": row["parameter_count"],
                "initial_checkpoint_sha256": row["initial_checkpoint_sha256"],
                "final_checkpoint_sha256": row["final_checkpoint_sha256"],
            }
            for row in checkpoint_manifest
        ],
        "loss_trace_storage": "numeric_checkpoint_files",
    }
    receipt = _receipt(
        started_ns=started_ns,
        ended_ns=ended_ns,
        spans=phase_spans,
        training=small_training,
        root=root,
    )
    gates = [
        _gate(
            "preconditions",
            "validity",
            "all",
            True,
            all(row.get("passed") is True for row in preconditions),
            all(row.get("passed") is True for row in preconditions),
            "Exact prerequisites prevent fitting against changed authority or basis bytes.",
        ),
        _gate(
            "registered_units",
            "completion",
            "==",
            expected_units,
            len(rows),
            len(rows) == expected_units,
            "Every arm, seed, and condition must remain visible.",
        ),
        _gate(
            "spline_basis_parity",
            "validity",
            "==",
            True,
            parity_passed,
            parity_passed,
            "A representation mismatch is a defect, not a benefit.",
        ),
        _gate(
            "detailed_rows_present",
            "completion",
            ">",
            0,
            detail_count,
            detail_count > 0,
            "The headline must remain tied to every seed probability and exclusion.",
        ),
        _gate(
            "affected_and_terminal_validation",
            "validation",
            "==",
            True,
            validation_passed,
            validation_passed,
            "Only the frozen scoped plan and exact terminal readers authorize capture.",
        ),
        *[
            _gate(
                check,
                "scientific_benefit",
                "==",
                True,
                observed,
                observed is True,
                "Every prespecified predictive and typed-policy clause is required.",
            )
            for check, observed in value["checks"].items()
        ],
    ]
    try:
        detail_label = detail_directory.relative_to(root).as_posix()
    except ValueError:
        detail_label = str(detail_directory.resolve())
    try:
        metric_label = metric_directory.relative_to(root).as_posix()
    except ValueError:
        metric_label = str(metric_directory.resolve())
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        **receipt,
        "model_duration_s": 0.0,
        "validation_duration_s": sum(
            float(row.get("duration_s") or 0.0) for row in validation_receipts
        ),
        "cold_start_duration_s": sum(
            float(row.get("duration_s") or 0.0)
            for row in validation_receipts
            if row.get("name") in TERMINAL_CHECK_NAMES
        ),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "random_seed": {
            "training": list(TRAINING_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
            "policy_thresholds": {
                "accept": list(ACCEPT_THRESHOLDS),
                "reject": list(REJECT_THRESHOLDS),
            },
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "detail_row_directory": detail_label,
        "probability_row_shards": [deepcopy(dict(row)) for row in detail_shards],
        "metric_row_directory": metric_label,
        "metric_row_shards": [deepcopy(dict(row)) for row in metric_shards],
        "sample_size_budget": {
            "planned_units": expected_units,
            "attempted_units": len(rows),
            "completed_units": len(rows),
            "failed_units": 0,
            "censored_units": 0,
            "unstarted_units": 0,
            "official_metric_rows": metric_count,
            "detailed_rows": detail_count,
            "excluded_source_units": excluded_count,
            "independent_test_groups": intervals["effective_groups"],
            "stop_rule": "all registered fits and one frozen official-test evaluation",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "label_authority": LABEL_AUTHORITY,
        "label_authority_limit": "fallible human source-support judgment; not formal truth or replacement verifier",
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "decision_capture_complete_score": capture,
        "decision_value_score": decision_value,
        "checkpoint_manifest": [deepcopy(dict(row)) for row in checkpoint_manifest],
        "policy_certificates": [
            deepcopy(row) for fit in fits for row in fit["policy_certificates"]
        ],
        "grouped_intervals": intervals,
        "calibration_metrics": metrics,
        "decision_value_reduction": value,
        "spline_parity_rows": parity_rows,
        "domain_level_support": {
            arm: deepcopy(metric["by_domain"]) for arm, metric in metrics["by_arm"].items()
        },
        "empirical_official_test_risks": {
            arm: {
                "false_accept_risk": metric["false_accept_risk"],
                "false_reject_risk": metric["false_reject_risk"],
                "empirical_decision_risk": metric["empirical_decision_risk"],
                "coverage": metric["coverage"],
            }
            for arm, metric in metrics["by_arm"].items()
        },
        "exchangeability_limit": "Policy bounds assume policy-calibration source groups exchange with future groups.",
        "source_ablation_role": "diagnostic_only_not_another_chance_to_pass",
        "lexical_entailment_claimed": False,
        "fixture_artifact": fixture,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute capture and value from raw shards, checkpoints, and receipts."""

    detail_dir = Path(str(value.get("detail_row_directory") or ""))
    if not detail_dir.is_absolute():
        detail_dir = root / detail_dir
    metric_dir = Path(str(value.get("metric_row_directory") or ""))
    if not metric_dir.is_absolute():
        metric_dir = root / metric_dir
    details = load_detail_shards(detail_dir, value.get("probability_row_shards") or [])
    metric_rows = load_detail_shards(metric_dir, value.get("metric_row_shards") or [])
    full_rows = [row for row in metric_rows if row.get("condition") == "full_source"]
    metrics = reduce_metrics(full_rows)
    draws = 200 if value.get("fixture_artifact") else BOOTSTRAP_DRAWS
    intervals = grouped_bootstrap(full_rows, draws=draws, seed=BOOTSTRAP_SEED)
    parity_rows = value.get("spline_parity_rows") or []
    parity = bool(parity_rows) and all(
        row.get("passed") is True
        and float(row.get("max_parameter_gap", math.inf)) <= PARITY_TOLERANCE
        and float(row.get("max_probability_gap", math.inf)) <= PARITY_TOLERANCE
        and float(row.get("max_gradient_gap", math.inf)) <= PARITY_TOLERANCE
        for row in parity_rows
    )
    decision = reduce_decision_value(intervals, metrics, parity_passed=parity)
    units = value.get("rows") or []
    expected_conditions = 1 if value.get("fixture_artifact") else len(CONDITIONS)
    expected_units = expected_conditions * len(ARMS) * len(TRAINING_SEEDS)
    unit_complete = len(units) == expected_units and all(
        row.get("status") == "completed" and row.get("completed") is True for row in units
    )
    checkpoints = value.get("checkpoint_manifest") or []
    checkpoint_complete = len(checkpoints) == expected_units
    validation = _validation_passed(value.get("validation_receipts") or [])
    preconditions = value.get("preconditions_checked") or []
    capture = int(
        bool(preconditions)
        and all(row.get("passed") is True for row in preconditions)
        and unit_complete
        and checkpoint_complete
        and bool(details)
        and bool(metric_rows)
        and parity
        and validation
        and value.get("flagged_adversarial") is False
    )
    return {
        "decision_capture_complete_score": capture,
        "decision_value_score": int(capture == 1 and decision["decision_value_score"] == 1),
        "promotion_score": 0,
        "detail_row_count": len(details),
        "metric_row_count": len(metric_rows),
        "calibration_metrics": metrics,
        "grouped_intervals": intervals,
        "decision_value_reduction": decision,
        "spline_parity_passed": parity,
    }


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check declarations, raw reduction, byte hashes, and checksum."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "promotion_score": 0,
        "verifier_is_oracle": False,
        "label_authority": LABEL_AUTHORITY,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"declaration_mismatch:{field}")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if set(value.get("field_principles") or {}) != set(REQUIRED_FIELDS):
        errors.append("field_principles_mismatch")
    if value.get("verdict_class") == "blocked":
        if not str(value.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
    else:
        try:
            reduced = independent_reduce(value, root=root)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"independent_reduction_failed:{error}")
        else:
            for field in (
                "decision_capture_complete_score",
                "decision_value_score",
                "promotion_score",
            ):
                if value.get(field) != reduced[field]:
                    errors.append(f"{field}_mismatch")
            if canonical_hash(value.get("calibration_metrics")) != canonical_hash(
                reduced["calibration_metrics"]
            ):
                errors.append("calibration_metrics_mismatch")
            if canonical_hash(value.get("grouped_intervals")) != canonical_hash(
                reduced["grouped_intervals"]
            ):
                errors.append("grouped_intervals_mismatch")
            if canonical_hash(value.get("decision_value_reduction")) != canonical_hash(
                reduced["decision_value_reduction"]
            ):
                errors.append("decision_value_reduction_mismatch")
        for row in value.get("checkpoint_manifest") or []:
            path = Path(str(row.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            if (
                not resolved.is_file()
                or sha256_file(resolved) != row.get("sha256")
                or resolved.stat().st_size != row.get("byte_size")
            ):
                errors.append(f"checkpoint_bytes_mismatch:{row.get('path')}")
        if not value.get("fixture_artifact"):
            for row in (value.get("source_artifact_hashes") or {}).values():
                if not isinstance(row, Mapping):
                    errors.append("source_artifact_hash_row_invalid")
                    continue
                path = Path(str(row.get("path") or ""))
                resolved = path if path.is_absolute() else root / path
                if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                    errors.append(f"source_artifact_hash_mismatch:{row.get('path')}")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(value, root=root))
    return list(dict.fromkeys(errors))


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish external prerequisite absence without dependent fitting."""

    started = time.monotonic_ns()
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-blocked-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"device": "host_cpu", "dependent_work_started": False},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        small_ebm_training={"performed": False, "reason": "blocked_prerequisite"},
    )
    gates = [
        _gate(
            str(failed.get("check")),
            "prerequisite",
            str(failed.get("operator") or "=="),
            failed.get("expected"),
            failed.get("observed"),
            False,
            "Dependent work cannot replace absent or changed authority evidence.",
        )
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_upstream_prerequisite",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        **receipt,
        "model_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "cold_start_duration_s": 0.0,
        "preconditions_checked": [deepcopy(dict(failed))],
        "random_seed": {"training": list(TRAINING_SEEDS), "bootstrap": BOOTSTRAP_SEED},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": len(CONDITIONS) * len(ARMS) * len(TRAINING_SEEDS),
            "attempted_units": 0,
            "completed_units": 0,
            "failed_units": 0,
            "censored_units": 0,
            "unstarted_units": len(CONDITIONS) * len(ARMS) * len(TRAINING_SEEDS),
            "stop_rule": "stop before dependent work on failed prerequisite",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates, failed),
        "verifier_is_oracle": False,
        "label_authority": LABEL_AUTHORITY,
        "honest_verdict": "blocked_upstream_prerequisite",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "decision_capture_complete_score": 0,
        "decision_value_score": 0,
        "checkpoint_manifest": [],
        "policy_certificates": [],
        "grouped_intervals": {},
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_rows() -> list[JsonDict]:
    """Create small sealed-role rows for reducer and mutation tests."""

    rows: list[JsonDict] = []
    for partition_index, partition in enumerate(PARTITIONS):
        for index in range(8):
            value = 0.1 + (index + partition_index) / 20.0
            rows.append(
                {
                    "row_key": f"{partition}-{index}",
                    "group_id": f"{partition}-group-{index}",
                    "source_group": f"{partition}-group-{index}",
                    "partition": partition,
                    "task_type": ("QA", "Summary", "Data2txt")[index % 3],
                    "features": {
                        name: float(min(max(value + feature / 100.0, 0.0), 1.0))
                        for feature, name in enumerate(SOURCE_FEATURE_NAMES)
                    },
                    "label": index % 2,
                    "sensitivity_label": index % 2,
                    "authority": LABEL_AUTHORITY,
                }
            )
    return rows


def build_fixture_artifact(
    root: Path, *, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a small complete artifact without touching repository evidence."""

    rows = _fixture_rows()
    fit = fit_condition(rows, condition="full_source", steps=1)
    official = [row for row in rows if row["partition"] == "final_test"]
    probabilities, metrics = score_official_rows(fit, official)
    detail_dir = root / "fixture-detail"
    metric_dir = root / "fixture-metric"
    detail = write_detail_shards(detail_dir, probabilities)
    metric = write_detail_shards(metric_dir, metrics, prefix="metrics")
    checkpoints = write_checkpoints(root, root / "fixture-checkpoints", [fit])
    preconditions = [_precondition("fixture", "fixture", "fixture", "ready", "==", True, True)]
    started_ns = 1_000_000_000
    ended_ns = 2_000_000_000
    artifact = build_artifact(
        root=root,
        fits=[fit],
        checkpoint_manifest=checkpoints,
        detail_shards=detail,
        metric_shards=metric,
        detail_directory=detail_dir,
        metric_directory=metric_dir,
        preconditions=preconditions,
        source_hashes={},
        validation_receipts=validation_receipts,
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        started_ns=started_ns,
        ended_ns=ended_ns,
        excluded_count=0,
        fixture=True,
    )
    return artifact


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Reload one candidate and independently validate all bound evidence."""

    value = _load_object(path)
    return validate_artifact(value, root=root) if value else ["artifact_unreadable_or_not_object"]


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    """Close one measured phase with a resumable completed-unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover - E2E only.
    """Build fresh replay, reduction, adversarial, and strict row readers."""

    python = ".venv/bin/python"
    common = (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE)
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay", (*common, "--cold-replay", str(candidate)), "candidate"
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_row_reduction",
                (*common, "--independent-reduce", str(candidate)),
                "candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
    ]


def _load_protocol_views(
    root: Path,
) -> tuple[ProtocolReaders, JsonDict]:  # pragma: no cover - E2E only.
    """Use the shipped hash-checking reader and retain raw dispositions."""

    reloaded = reload_corpus(root / CORPUS_DIR)
    return ProtocolReaders(reloaded), reloaded


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared entrypoint E2E.
    """Authenticate, fit, freeze, score once, validate, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, _loaded = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return blocked

    progress(run_started, "load_protocol", "before_benchmark")
    phase_started = time.monotonic()
    readers, reloaded = _load_protocol_views(root)
    dev_predictors = [
        row for partition in PARTITIONS[:-1] for row in readers.read_predictors(partition)
    ]
    dev_evaluators = [
        row
        for partition in PARTITIONS[:-1]
        for row in readers.read_evaluators(partition, EVALUATOR_TOKEN)
    ]
    spans.append(_span("load_protocol", phase_started, run_started, len(dev_predictors)))
    progress(run_started, "load_protocol", "after_benchmark", completed=len(dev_predictors))

    fits: list[JsonDict] = []
    condition_predictor_rows: dict[str, list[JsonDict]] = {}
    progress(run_started, "small_ebm_training", "before_training", planned=75)
    phase_started = time.monotonic()

    def emit(arm: str, event: str, completed: int, total: int) -> None:
        progress(
            run_started,
            "small_ebm_training",
            event,
            arm=arm,
            condition=CONDITIONS[len(fits)],
            completed=len(fits) * total + completed,
            total=len(CONDITIONS) * total,
        )

    for condition in CONDITIONS:
        progress(run_started, "condition_features", "before_benchmark", condition=condition)
        transformed = condition_predictors(dev_predictors, condition)
        condition_predictor_rows[condition] = transformed
        joined = join_predictor_labels(transformed, dev_evaluators)
        progress(
            run_started,
            "condition_features",
            "after_benchmark",
            condition=condition,
            completed=len(joined),
        )
        fits.append(fit_condition(joined, condition=condition, emit=emit))
    spans.append(_span("small_ebm_training", phase_started, run_started, 75))
    progress(run_started, "small_ebm_training", "after_training", completed=75)

    raw_dir = root / RAW_DIR
    progress(run_started, "checkpoint", "before_serialization", planned=75)
    phase_started = time.monotonic()
    checkpoints = write_checkpoints(root, raw_dir / "checkpoints", fits)
    spans.append(_span("checkpoint", phase_started, run_started, len(checkpoints)))
    progress(run_started, "checkpoint", "after_serialization", completed=len(checkpoints))

    progress(run_started, "official_test", "before_benchmark")
    phase_started = time.monotonic()
    official_predictors = readers.read_predictors("final_test")
    official_evaluators = readers.read_evaluators("final_test", EVALUATOR_TOKEN)
    probability_rows: list[JsonDict] = []
    metric_rows: list[JsonDict] = []
    for fit in fits:
        condition = str(fit["condition"])
        transformed = condition_predictors(official_predictors, condition)
        official = join_predictor_labels(transformed, official_evaluators)
        seed_rows, ensemble = score_official_rows(fit, official)
        probability_rows.extend(seed_rows)
        metric_rows.extend(ensemble)
        progress(
            run_started,
            "official_test",
            "checkpoint",
            condition=condition,
            completed=len(probability_rows),
        )
    excluded = excluded_probability_rows(reloaded["dispositions"])
    probability_rows.extend(excluded)
    spans.append(
        _span("official_test", phase_started, run_started, len(probability_rows) + len(metric_rows))
    )
    progress(
        run_started,
        "official_test",
        "after_benchmark",
        completed=len(probability_rows) + len(metric_rows),
    )

    progress(run_started, "evidence_shards", "before_serialization")
    phase_started = time.monotonic()
    detail_dir = raw_dir / "probability_rows"
    metric_dir = raw_dir / "metric_rows"
    detail_shards = write_detail_shards(detail_dir, probability_rows)
    metric_shards = write_detail_shards(metric_dir, metric_rows, prefix="metrics")
    spans.append(
        _span(
            "evidence_shards", phase_started, run_started, len(detail_shards) + len(metric_shards)
        )
    )
    progress(
        run_started,
        "evidence_shards",
        "after_serialization",
        completed=len(detail_shards) + len(metric_shards),
    )

    progress(run_started, "bootstrap", "before_benchmark", draws=BOOTSTRAP_DRAWS)
    phase_started = time.monotonic()
    full_metrics = [row for row in metric_rows if row["condition"] == "full_source"]
    reduce_metrics(full_metrics)
    grouped_bootstrap(full_metrics)
    spans.append(_span("bootstrap", phase_started, run_started, BOOTSTRAP_DRAWS))
    progress(run_started, "bootstrap", "after_benchmark", completed=BOOTSTRAP_DRAWS)

    private_root = Path(tempfile.mkdtemp(prefix="exp7426-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(
        run_started,
        "affected_validation",
        "before_subprocesses",
        planned=len(commands),
        plan_errors=len(plan_errors),
    )
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )

    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        path = root / relative
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "original_flagged_adversarial": None,
        }
    for row in [*detail_shards, *metric_shards]:
        directory = detail_dir if row in detail_shards else metric_dir
        path = directory / row["path"]
        label = path.relative_to(root).as_posix()
        source_hashes[label] = {
            "path": label,
            "sha256": row["sha256"],
            "original_flagged_adversarial": None,
        }
    for row in checkpoints:
        source_hashes[row["path"]] = {
            "path": row["path"],
            "sha256": row["sha256"],
            "original_flagged_adversarial": None,
        }
    candidate = build_artifact(
        root=root,
        fits=fits,
        checkpoint_manifest=checkpoints,
        detail_shards=detail_shards,
        metric_shards=metric_shards,
        detail_directory=detail_dir,
        metric_directory=metric_dir,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=affected,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        excluded_count=len(reloaded["dispositions"]),
        flagged_adversarial=not affected_reduction["passed"],
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_commands = _terminal_commands(candidate_path)
    progress(
        run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_commands)
    )
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_commands, log_dir=raw_dir / "validation/terminal"
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    final = build_artifact(
        root=root,
        fits=fits,
        checkpoint_manifest=checkpoints,
        detail_shards=detail_shards,
        metric_shards=metric_shards,
        detail_directory=detail_dir,
        metric_directory=metric_dir,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        excluded_count=len(reloaded["dispositions"]),
        flagged_adversarial=not affected_reduction["passed"] or not terminal_passed or critical,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and strict fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one strict fresh-process artifact reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, root=args.root)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, root=args.root)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        reduced = independent_reduce(value, root=args.root) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduced}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

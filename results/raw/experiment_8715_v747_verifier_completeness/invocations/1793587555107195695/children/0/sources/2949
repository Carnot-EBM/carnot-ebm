"""Measure tuned typed decisions on the preserved human-label corpus.

This experiment can measure a protocol on reused evidence. It cannot turn old
labels into a deployment guarantee for new users.

Spec refs: REQ-AUTO-7439 and SCENARIO-AUTO-7439-01 through -06.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
from carnot.experiment_7412_v650_source_features import (
    MAX_STEPS,
    SOURCE_FEATURE_NAMES,
    fit_gibbs_head,
    fit_logistic_control,
    initial_gibbs_checkpoint,
)
from carnot.experiment_7423_v651_annotated_protocol import (
    EVALUATOR_TOKEN,
    ProtocolReaders,
    reload_corpus,
)
from carnot.experiment_7426_v651_static_decisions import (
    _apply_affine,
    _predict_state,
    fit_spline_pair,
    spline_parity_errors,
)
from carnot.experiment_7436_v652_selection_protocol import (
    ALPHA_PER_CHECK,
    ACCEPT_RISK_BUDGET,
    COVERAGE_FLOOR,
    NO_ACCEPT_SENTINEL,
    NO_REJECT_SENTINEL,
    REJECT_RISK_BUDGET,
    certify_policy,
    exact_action_check,
    select_tuned_policy,
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
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7439-v652-certified-decisions"
SCHEMA = "carnot.exp7439.v652.certified_decisions.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7439_v652_certified_decisions.json")
RAW_DIR = Path("results/raw/experiment_7439_v652_certified_decisions")
ROW_DIR = RAW_DIR / "probability_rows"
CHECKPOINT_DIR = RAW_DIR / "checkpoints"
MODULE_PATH = Path("python/carnot/experiment_7439_v652_certified_decisions.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7439_v652_certified_decisions.py")
TEST_PATH = Path("tests/python/test_experiment_7439_v652_certified_decisions.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/autoresearch/spec.md"
UPSTREAM_PATH = Path("results/experiment_7436_v652_selection_protocol.json")
SEALED_PROTOCOL_PATH = Path("results/raw/experiment_7436_v652_selection_protocol/protocol.json")
CORPUS_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
CORPUS_MANIFEST_PATH = CORPUS_DIR / "corpus_manifest.json"
EXPECTED_UPSTREAM_SHA256 = "sha256:e125cf8a0bd6ab3805bf3cc5656f6aadc0d1dfb99fe6c42967db746119148f79"
EXPECTED_PROTOCOL_SHA256 = "sha256:7c006fececfe75976642a8000be70ec6d2ec3e514c5c7d81aebedb6efc23aad7"
EXPECTED_PROTOCOL_MANIFEST_HASH = (
    "sha256:50dd0c2864661de9ca68a6389213d905829b0f4568f3f09f8b9edb95a6a06905"
)
EXPECTED_CORPUS_SHA256 = "sha256:a065b2a64f2926c5bade1fd3495b1cbff23bd3b3c967bd104b973a6b118e30c0"

TRAINING_SEEDS = (65_201, 65_202, 65_203, 65_204, 65_205)
DEPLOYED_HEADS = ("gibbs_6_4_1", "sparse_spline_49", "raw_l2_logistic")
DENSE_CONTROL = "dense_spline_logistic"
PREVALENCE_CONTROL = "training_prevalence"
ALL_FITTED_HEADS = (*DEPLOYED_HEADS, DENSE_CONTROL)
OLD_POLICY = {
    "accept_threshold": 0.95,
    "reject_threshold": 0.01,
    "accept_enabled": True,
    "reject_enabled": True,
    "candidate_index": 0,
    "selection_partition": "v651_fixed_threshold_protocol",
    "frozen": True,
}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_520_439
INFERENCE_SUBSTRATE = "compact_numeric_head_training_and_exact_group_reduction"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
SCORE_TOLERANCE = 0.001

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
    Path("python/carnot/experiment_7423_v651_annotated_protocol.py"),
    Path("python/carnot/experiment_7426_v651_static_decisions.py"),
    Path("python/carnot/experiment_7436_v652_selection_protocol.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    Path("ops/verifier_gaps.md"),
    Path("openspec/capabilities/autoresearch/spec.md"),
    UPSTREAM_PATH,
    SEALED_PROTOCOL_PATH,
    CORPUS_MANIFEST_PATH,
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
    "policy_certificates",
    "probability_row_shards",
    "checkpoint_manifest",
    "evaluation_reuse_disclosure",
    "small_ebm_training",
    "deployment_certificate_valid",
    "certificate_scope",
)


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    """Return one aware UTC timestamp for an artifact boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - console boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Print each phase boundary and long-operation checkpoint immediately."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7439] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - runtime I/O.
    """Load one external object while preserving missing or malformed evidence."""

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
    passed: bool,
) -> JsonDict:
    """Keep missing, null, and zero observations distinct in gate evidence."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def upstream_field_checks(upstream: Mapping[str, Any], *, path: Path) -> list[JsonDict]:
    """Authenticate the three structured prerequisites before dependent work."""

    allowed = ["null", "positive"]
    definitions = (
        (
            "selection_protocol_ready",
            "selection_protocol_ready_score",
            "==",
            1,
            upstream.get("selection_protocol_ready_score"),
            upstream.get("selection_protocol_ready_score") == 1,
        ),
        (
            "selection_protocol_verdict",
            "verdict_class",
            "in",
            allowed,
            upstream.get("verdict_class"),
            upstream.get("verdict_class") in allowed,
        ),
        (
            "selection_protocol_unflagged",
            "flagged_adversarial",
            "==",
            False,
            upstream.get("flagged_adversarial"),
            upstream.get("flagged_adversarial") is False,
        ),
    )
    return [
        _precondition(
            check,
            "exp7436-selection-protocol",
            path.as_posix(),
            field,
            operator,
            expected,
            observed,
            passed,
        )
        for check, field, operator, expected, observed, passed in definitions
    ]


def predictor_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, JsonDict]:
    """Project only six numeric proxies and disclose every excluded field."""

    vectors: list[list[float]] = []
    all_fields: set[str] = set()
    for row in rows:
        all_fields.update(str(key) for key in row)
        features = row.get("features")
        if not isinstance(features, Mapping) or set(features) != set(SOURCE_FEATURE_NAMES):
            raise ValueError("predictor requires the exact six shipped features")
        vector = [float(features[name]) for name in SOURCE_FEATURE_NAMES]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in vector):
            raise ValueError("predictor features must be finite and bounded")
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1:] != (6,):
        raise ValueError("predictor matrix must be non-empty with six columns")
    return matrix, {
        "input_fields": list(SOURCE_FEATURE_NAMES),
        "excluded_fields": sorted(all_fields - {"features"}),
        "labels_consumed": False,
        "annotation_spans_consumed": False,
        "model_identity_consumed": False,
        "group_identity_consumed": False,
    }


def _state_row(arm: str, seed: int, checkpoint: Mapping[str, Any], receipt: JsonDict) -> JsonDict:
    """Bind one fitted numeric state to its fit-only receipt."""

    return {
        "arm": arm,
        "seed": seed,
        "checkpoint": deepcopy(dict(checkpoint)),
        "initial_checkpoint": receipt["initial_checkpoint"],
        "initial_checkpoint_sha256": canonical_hash(receipt["initial_checkpoint"]),
        "final_checkpoint_sha256": canonical_hash(checkpoint),
        "updates": receipt["updates"],
        "parameter_count": receipt["parameter_count"],
        "loss_trace": receipt["loss_trace"],
        "objective": receipt["objective"],
    }


def fit_compact_heads(rows: Sequence[Mapping[str, Any]], *, steps: int = MAX_STEPS) -> JsonDict:
    """Fit shipped Gibbs, spline, and logistic heads on fit rows only."""

    matrix, projection = predictor_matrix(rows)
    labels = np.asarray([row.get("label") for row in rows], dtype=np.float64)
    if labels.shape != (len(matrix),) or set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("fit rows require both binary labels")
    heads: dict[str, list[JsonDict]] = {head: [] for head in ALL_FITTED_HEADS}
    parity: list[JsonDict] = []
    fit_started = time.monotonic()
    for seed in TRAINING_SEEDS:
        logistic_initial = {
            "coef": np.random.default_rng(seed).normal(0.0, 0.01, size=6).tolist(),
            "bias": 0.0,
        }
        logistic = fit_logistic_control(matrix, labels, seed=seed, steps=steps)
        heads["raw_l2_logistic"].append(
            _state_row(
                "raw_l2_logistic",
                seed,
                logistic["checkpoint"],
                {
                    "initial_checkpoint": logistic_initial,
                    "updates": logistic["update_count"],
                    "parameter_count": 7,
                    "loss_trace": logistic["loss_curve"],
                    "objective": logistic["objective"],
                },
            )
        )
        gibbs_initial = initial_gibbs_checkpoint(seed=seed, input_dim=6)
        gibbs = fit_gibbs_head(matrix, labels, seed=seed, steps=steps)
        heads["gibbs_6_4_1"].append(
            _state_row(
                "gibbs_6_4_1",
                seed,
                gibbs["checkpoint"],
                {
                    "initial_checkpoint": gibbs_initial,
                    "updates": gibbs["update_count"],
                    "parameter_count": 33,
                    "loss_trace": gibbs["loss_curve"],
                    "objective": gibbs["objective"],
                },
            )
        )
        pair = fit_spline_pair(matrix, labels, seed=seed, steps=steps)
        common = {
            "updates": pair["updates"],
            "parameter_count": 49,
            "loss_trace": pair["loss_trace"],
            "objective": pair["objective"],
        }
        sparse = pair["sparse_checkpoint"]
        dense = pair["dense_checkpoint"]
        heads["sparse_spline_49"].append(
            _state_row(
                "sparse_spline_49",
                seed,
                sparse,
                {
                    **common,
                    "initial_checkpoint": {
                        "coef": sparse["initial_coef"],
                        "bias": sparse["initial_bias"],
                        "knots": sparse["knots"],
                    },
                },
            )
        )
        heads[DENSE_CONTROL].append(
            _state_row(
                DENSE_CONTROL,
                seed,
                dense,
                {
                    **common,
                    "initial_checkpoint": {
                        "coef": dense["initial_coef"],
                        "bias": dense["initial_bias"],
                        "knots": dense["knots"],
                    },
                },
            )
        )
        parity.append(
            {
                "seed": seed,
                "max_parameter_gap": pair["max_parameter_gap"],
                "max_probability_gap": pair["max_probability_gap"],
                "max_gradient_gap": pair["max_gradient_gap"],
                "passed": not spline_parity_errors(pair),
            }
        )
    prevalence = float(np.mean(labels))
    return {
        "training_seeds": list(TRAINING_SEEDS),
        "heads": heads,
        "fit_elapsed_s": time.monotonic() - fit_started,
        "fit_partition": "fit",
        "fit_row_count": len(rows),
        "fit_input_sha256": canonical_hash(
            {"matrix": matrix.tolist(), "labels": labels.astype(int).tolist()}
        ),
        "projection_receipt": projection,
        "diagnostics": {
            "dense_spline_parity": parity,
            "prevalence": {
                "probability": prevalence,
                "parameter_count": 1,
                "training_only": True,
                "pooled_into_certification": False,
            },
        },
    }


def calibrate_seed_probabilities(
    seed_probabilities: Any, labels: Sequence[int], *, steps: int = MAX_STEPS
) -> JsonDict:
    """Average five seed probabilities before fitting one scalar calibration."""

    raw = np.asarray(seed_probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 5 or y.shape != (raw.shape[1],):
        raise ValueError("calibration requires five seed rows and matching labels")
    if not np.all(np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("calibration probabilities must be finite and bounded")
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("calibration requires both labels")
    means = np.mean(raw, axis=0)
    from carnot.experiment_7426_v651_static_decisions import fit_ensemble_calibration

    fitted = fit_ensemble_calibration(raw, y.astype(int).tolist(), steps=steps)
    fitted["mean_raw_probabilities"] = means.tolist()
    return fitted


def freeze_policy(
    tuning_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    """Select exactly one threshold pair and mark it immutable."""

    policy, candidates = select_tuned_policy(tuning_rows)
    policy = deepcopy(policy)
    policy["frozen"] = True
    policy["selection_partition"] = "policy_tuning"
    return policy, candidates


def action_risk_check(harmful: Sequence[int], *, action: str, enabled: bool) -> JsonDict:
    """Report undefined action risk as null and leave disabled alpha unspent."""

    if action not in {"accept", "reject"}:
        raise ValueError("action must be accept or reject")
    if any(value not in {0, 1} for value in harmful):
        raise ValueError("harmful outcomes must be binary")
    budget = ACCEPT_RISK_BUDGET if action == "accept" else REJECT_RISK_BUDGET
    raw = exact_action_check(harmful, risk_budget=budget, enabled=enabled)
    return {
        "action": action,
        "applicable": raw["applicable"],
        "enabled_during_tuning": enabled,
        "selected_groups": raw["selected_groups"],
        "harmful_outcomes": raw["harmful_outcomes"],
        "risk": raw["empirical_risk"],
        "upper_bound": raw["upper_risk_bound"],
        "risk_budget": budget,
        "alpha_allocated": raw["alpha_allocated"],
        "alpha_spent": raw["alpha_spent"],
        "passed": raw["passed"],
        "diagnosis": raw["diagnosis"],
    }


def certify_frozen_policy(
    certification_rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> JsonDict:
    """Apply all three exact checks without changing the selected thresholds."""

    certificate = certify_policy(certification_rows, policy)
    certificate = deepcopy(certificate)
    certificate["valid"] = bool(certificate["all_checks_passed"])
    certificate["certificate_coverage_lower_bound"] = certificate["coverage_check"][
        "lower_coverage_bound"
    ]
    return certificate


def _log_loss(label: int, probability: float) -> float:
    """Return finite binary log loss for one probability row."""

    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def reduce_policy_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce probability calibration and typed policy behavior separately."""

    if not rows:
        raise ValueError("metric rows must be non-empty")
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    actions = [str(row["action"]) for row in rows]
    if any(label not in {0, 1} for label in labels):
        raise ValueError("metric labels must be binary")
    if any(action not in {"accept", "reject", "escalate"} for action in actions):
        raise ValueError("metric actions must be typed")
    accept = [label for label, action in zip(labels, actions, strict=True) if action == "accept"]
    reject = [label for label, action in zip(labels, actions, strict=True) if action == "reject"]
    correct = sum(
        (action == "accept" and label == 1) or (action == "reject" and label == 0)
        for label, action in zip(labels, actions, strict=True)
    )
    harmful_accept = sum(label == 0 for label in accept)
    harmful_reject = sum(label == 1 for label in reject)
    utility = correct - 20 * harmful_accept - 10 * harmful_reject
    return {
        "row_count": len(rows),
        "brier": float(
            np.mean(
                [
                    (probability - label) ** 2
                    for label, probability in zip(labels, probabilities, strict=True)
                ]
            )
        ),
        "log_loss": float(
            np.mean(
                [
                    _log_loss(label, probability)
                    for label, probability in zip(labels, probabilities, strict=True)
                ]
            )
        ),
        "coverage": (len(accept) + len(reject)) / len(rows),
        "accept_harm": harmful_accept / len(accept) if accept else None,
        "accept_denominator": len(accept),
        "accept_harmful": harmful_accept,
        "reject_harm": harmful_reject / len(reject) if reject else None,
        "reject_denominator": len(reject),
        "reject_harmful": harmful_reject,
        "registered_utility": utility / len(rows),
        "utility": utility / len(rows),
    }


def paired_coverage_intervals(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, JsonDict]:
    """Bootstrap two simultaneous paired coverage contrasts by source group."""

    groups = [str(row.get("group_id") or "") for row in rows]
    if not groups or any(not group for group in groups) or len(set(groups)) != len(groups):
        raise ValueError("paired bootstrap requires unique non-empty source groups")
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    values = np.asarray(
        [
            [
                float(row["tuned_spline"]) - float(row["old_spline"]),
                float(row["tuned_spline"]) - float(row["tuned_logistic"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 2), dtype=np.float64)
    for start in range(0, draws, 1_000):
        stop = min(start + 1_000, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = np.mean(values[indices], axis=1)
    names = ("tuned_spline_minus_old", "tuned_spline_minus_logistic")
    return {
        name: {
            "observed": float(np.mean(values[:, index])),
            "lower": float(np.quantile(samples[:, index], 0.0125)),
            "upper": float(np.quantile(samples[:, index], 0.9875)),
            "confidence": 0.95,
            "simultaneous_contrasts": 2,
            "tail_alpha": 0.0125,
            "draws": draws,
            "seed": seed,
            "paired_source_groups": len(values),
        }
        for index, name in enumerate(names)
    }


def _benefit_gate(check: str, observed: Any, expected: Any, passed: bool) -> JsonDict:
    """Describe one registered value gate without hiding a null observation."""

    return {
        "check": check,
        "category": "scientific_benefit",
        "operator": ">=" if "noninferiority" not in check else "<=",
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": "Every registered decision-benefit condition is conjunctive.",
    }


def reduce_decision_value(
    spline: Mapping[str, Any],
    logistic: Mapping[str, Any],
    old_spline: Mapping[str, Any],
    intervals: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Require certificate, coverage, contrasts, utility, and proper scores together."""

    old_lower = float(intervals["tuned_spline_minus_old"]["lower"])
    logistic_lower = float(intervals["tuned_spline_minus_logistic"]["lower"])
    utility_floor = max(float(logistic["utility"]), float(old_spline["utility"]))
    brier_ceiling = min(float(logistic["brier"]), float(old_spline["brier"])) + SCORE_TOLERANCE
    log_ceiling = min(float(logistic["log_loss"]), float(old_spline["log_loss"])) + SCORE_TOLERANCE
    gates = [
        _benefit_gate(
            "spline_certificate_valid",
            spline["certificate_valid"],
            True,
            spline["certificate_valid"] is True,
        ),
        _benefit_gate(
            "certificate_coverage_lower_bound",
            spline["certificate_coverage_lower_bound"],
            COVERAGE_FLOOR,
            float(spline["certificate_coverage_lower_bound"]) >= COVERAGE_FLOOR,
        ),
        _benefit_gate("coverage_delta_over_old", old_lower, 0.0, old_lower > 0.0),
        _benefit_gate("coverage_delta_over_logistic", logistic_lower, 0.0, logistic_lower > 0.0),
        _benefit_gate(
            "utility_no_worse_than_controls",
            spline["utility"],
            utility_floor,
            float(spline["utility"]) >= utility_floor,
        ),
        _benefit_gate(
            "brier_noninferiority",
            spline["brier"],
            brier_ceiling,
            float(spline["brier"]) <= brier_ceiling,
        ),
        _benefit_gate(
            "log_loss_noninferiority",
            spline["log_loss"],
            log_ceiling,
            float(spline["log_loss"]) <= log_ceiling,
        ),
    ]
    value = int(all(row["passed"] for row in gates))
    policy_gain = old_lower > 0.0
    beats_logistic = logistic_lower > 0.0
    return {
        "decision_capture_complete_score": 1,
        "decision_value_score": value,
        "gates": gates,
        "limited_policy_wrapper_finding": policy_gain and not beats_logistic,
        "energy_advantage": True if value else None,
    }


def _policy_action(probability: float, certificate: Mapping[str, Any]) -> str:
    """Apply only the deployed policy after the certificate is frozen."""

    if certificate.get("deployed_policy") != "frozen_typed_policy":
        return "escalate"
    if certificate.get("deployed_accept_enabled") and probability >= float(
        certificate["accept_threshold"]
    ):
        return "accept"
    if certificate.get("deployed_reject_enabled") and probability <= float(
        certificate["reject_threshold"]
    ):
        return "reject"
    return "escalate"


def _validation_passed(
    receipts: Sequence[Mapping[str, Any]], *, candidate: bool
) -> bool:  # pragma: no cover - runtime reduction.
    """Require the exact affected set and, for final output, terminal readers."""

    required = set(AFFECTED_CHECK_NAMES)
    if not candidate:
        required.update(TERMINAL_CHECK_NAMES)
    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("required") is True and row.get("passed") is True and row.get("exit_code") == 0
    }
    return required.issubset(passed)


def _write_row_shard(
    root: Path, rows: Sequence[Mapping[str, Any]], *, name: str = "rows-000.jsonl"
) -> list[JsonDict]:  # pragma: no cover - runtime I/O.
    """Write one deterministic row shard and return its byte-bound manifest."""

    directory = root / ROW_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    data = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)
    relative = path.relative_to(root).as_posix()
    return [{"path": relative, "sha256": sha256_file(path), "rows": len(rows)}]


def _load_row_shards(
    root: Path, manifests: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:  # pragma: no cover - runtime I/O.
    """Reload exact shard bytes before independent reduction."""

    rows: list[JsonDict] = []
    for manifest in manifests:
        path = root / str(manifest.get("path") or "")
        if not path.is_file() or sha256_file(path) != manifest.get("sha256"):
            raise ValueError("probability_row_shard_invalid")
        loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(loaded) != manifest.get("rows") or any(not isinstance(row, dict) for row in loaded):
            raise ValueError("probability_row_shard_invalid")
        rows.extend(loaded)
    return rows


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks, logs, and this checksum."""

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
        "computation_duration_s",
    ):
        payload.pop(field, None)
    return canonical_hash(payload)


def _field_principles() -> dict[str, str]:
    """Explain required field intent separately from machine-readable values."""

    explicit = {
        "schema": "Use a versioned plain top-level schema with experiment identity and status.",
        "run_date": "Use 20260920 and record actual UTC and monotonic boundaries.",
        "preconditions_checked": "Name every actual path, identity, flag, and observed prerequisite.",
        "MODEL_SPECS": "Use an empty list because no current LLM is invoked.",
        "model_invoked": "Keep current attempted model work separate from archived evidence.",
        "invocation_counts": "Reconcile attempted, completed, failed, cancelled, and in-flight calls.",
        "inference_substrate": "Describe current numeric work in a plain string.",
        "inference_substrate_class": "Declare no_model_load because only compact numeric heads run.",
        "execution_venue": "Use host and disclose CPU, CUDA, and external devices separately.",
        "duration_s": "Measure current work and separate computation, validation, and cold start.",
        "phase_spans": "Bind phase times, progress events, and completed units.",
        "random_seed": "Freeze fitting and paired resampling seeds.",
        "reproducibility_checksum": "Bind code, protocol, inputs, rows, and validation scope.",
        "source_artifact_hashes": "Preserve source byte identity and original flags.",
        "rows": "Keep each arm, seed, condition, and terminal state visible.",
        "sample_size_budget": "Separate planned, attempted, completed, failed, and unstarted units.",
        "acceptance_gate_results": "Separate validity, completion, and scientific benefit gates.",
        "gate_check_summary": "Name exact blocked or failed fields without hiding null or zero.",
        "verifier_is_oracle": "Human support annotations are fallible and not an exact oracle.",
        "honest_verdict": "Use complete for valid nulls and blocked for absent prerequisites.",
        "verdict_class": "Use the closed terminal disposition vocabulary.",
        "flagged_adversarial": "Critical validation findings cannot supply readiness.",
        "validation_receipts": "Record actual scoped commands, exits, durations, and log hashes.",
        "field_principles": "Explain field intent without wrapping gate scalars.",
        "promotion_score": "Remain zero because this milestone cannot authorize rollout.",
        "decision_capture_complete_score": "Allow complete valid measurements, including nulls.",
        "decision_value_score": "Require every registered certificate and benefit gate.",
        "policy_certificates": "Record each frozen pair, alpha use, counts, harms, and exact bounds.",
        "probability_row_shards": "Keep source predictions and labels for independent reduction.",
        "checkpoint_manifest": "Bind fitted heads and calibration to fit-only inputs.",
        "evaluation_reuse_disclosure": "Keep reused public evidence exploratory.",
        "small_ebm_training": "Report compact-head work without implying generator training.",
        "deployment_certificate_valid": "Remain false because labels were exposed historically.",
        "certificate_scope": "Use exploratory_reused_corpus for all nominal bounds.",
    }
    return {
        field: explicit.get(field, f"Record the measured {field.replace('_', ' ')} value.")
        for field in REQUIRED_FIELDS
    }


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Create one explicit gate with an ordinary scalar observation."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _receipt(
    root: Path,
    *,
    started_ns: int,
    ended_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    small_training: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - runtime provenance.
    """Build zero-current-model provenance with typed historical sidecars."""

    sidecars = [
        sidecar_reference(root / path, root=root, scope="historical_model_receipts")
        for path in (UPSTREAM_PATH,)
        if (root / path).is_file()
    ]
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "device": "host_cpu",
            "cpu_used": True,
            "cuda_used": False,
            "external_device": None,
            "current_llm": None,
            "compact_heads_fitted": list(DEPLOYED_HEADS),
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=sidecars,
        phase_spans=phase_spans,
        small_ebm_training=small_training,
    )
    receipt["started_monotonic_ns"] = 0
    receipt["ended_monotonic_ns"] = ended_ns - started_ns
    return receipt


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute certificates, metrics, contrasts, and value from bound rows."""

    try:
        rows = _load_row_shards(root, value.get("probability_row_shards") or [])
    except (OSError, ValueError, json.JSONDecodeError):
        return {"error": "probability_row_shard_invalid"}
    policies = {
        (str(row["head"]), str(row["policy_kind"])): row
        for row in value.get("policy_certificates") or []
    }
    certificates: dict[str, JsonDict] = {}
    metrics: dict[str, JsonDict] = {}
    for (head, policy_kind), policy_row in sorted(policies.items()):
        certification = [
            row
            for row in rows
            if row.get("stage") == "certification"
            and row.get("head") == head
            and row.get("policy_kind") == policy_kind
        ]
        certificate = certify_frozen_policy(certification, policy_row["frozen_policy"])
        certificates[f"{head}:{policy_kind}"] = certificate
        final_rows = [
            row
            for row in rows
            if row.get("stage") == "final_test"
            and row.get("head") == head
            and row.get("policy_kind") == policy_kind
        ]
        metrics[f"{head}:{policy_kind}"] = reduce_policy_metrics(final_rows)
    spline_rows = {
        str(row["group_id"]): row
        for row in rows
        if row.get("stage") == "final_test"
        and row.get("head") == "sparse_spline_49"
        and row.get("policy_kind") == "tuned"
    }
    old_rows = {
        str(row["group_id"]): row
        for row in rows
        if row.get("stage") == "final_test"
        and row.get("head") == "sparse_spline_49"
        and row.get("policy_kind") == "old_fixed"
    }
    logistic_rows = {
        str(row["group_id"]): row
        for row in rows
        if row.get("stage") == "final_test"
        and row.get("head") == "raw_l2_logistic"
        and row.get("policy_kind") == "tuned"
    }
    groups = sorted(set(spline_rows) & set(old_rows) & set(logistic_rows))
    paired = [
        {
            "group_id": group,
            "tuned_spline": int(spline_rows[group]["action"] != "escalate"),
            "old_spline": int(old_rows[group]["action"] != "escalate"),
            "tuned_logistic": int(logistic_rows[group]["action"] != "escalate"),
        }
        for group in groups
    ]
    intervals = paired_coverage_intervals(paired)
    spline_metric = metrics["sparse_spline_49:tuned"]
    spline_certificate = certificates["sparse_spline_49:tuned"]
    value_reduction = reduce_decision_value(
        {
            "certificate_valid": spline_certificate["valid"],
            "certificate_coverage_lower_bound": spline_certificate[
                "certificate_coverage_lower_bound"
            ],
            **spline_metric,
        },
        metrics["raw_l2_logistic:tuned"],
        metrics["sparse_spline_49:old_fixed"],
        intervals,
    )
    candidate = bool(value.get("candidate_artifact"))
    validation_ok = _validation_passed(value.get("validation_receipts") or [], candidate=candidate)
    capture = int(validation_ok and not value.get("flagged_adversarial"))
    return {
        "certificates": certificates,
        "metrics": metrics,
        "paired_coverage_intervals": intervals,
        "decision_capture_complete_score": capture,
        "decision_value_score": value_reduction["decision_value_score"] if capture else 0,
        "benefit_gates": value_reduction["gates"],
        "limited_policy_wrapper_finding": value_reduction["limited_policy_wrapper_finding"],
        "energy_advantage": value_reduction["energy_advantage"],
    }


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check declarations, bound rows, reduction, and reused-corpus limits."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"required_field_missing:{field}")
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_invalid")
    if value.get("run_date") != RUN_DATE or value.get("milestone") != MILESTONE:
        errors.append("artifact_schedule_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_invalid")
    if value.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if value.get("deployment_certificate_valid") is not False:
        errors.append("deployment_certificate_must_be_false")
    if value.get("certificate_scope") != "exploratory_reused_corpus":
        errors.append("certificate_scope_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_invalid")
    reduced = independent_reduce(value, root=root)
    if reduced.get("error"):
        errors.append(str(reduced["error"]))
    elif reduced != value.get("independent_reduction"):
        errors.append("independent_reduction_mismatch")
    if not value.get("fixture_artifact"):
        errors.extend(validate_current_work_receipt(value, root=root))
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fixture_probability_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    """Create raw certification and final rows for local artifact tests."""

    policies: list[JsonDict] = []
    raw_rows: list[JsonDict] = []
    for head_index, head in enumerate(DEPLOYED_HEADS):
        tuning = [
            {
                "row_key": f"tune-{index:03d}",
                "group_id": f"tune-{index:03d}",
                "label": index % 2,
                "probability": (index + 1) / 41,
            }
            for index in range(40)
        ]
        tuned, _ = freeze_policy(tuning)
        for policy_kind, frozen in (("tuned", tuned), ("old_fixed", deepcopy(OLD_POLICY))):
            certification = [
                {
                    "row_key": f"cert-{index:03d}",
                    "group_id": f"cert-{index:03d}",
                    "label": index % 2,
                    "probability": (index + 1 + head_index) / 84,
                }
                for index in range(80)
            ]
            certificate = certify_frozen_policy(certification, frozen)
            policies.append(
                {
                    "head": head,
                    "policy_kind": policy_kind,
                    "frozen_policy": frozen,
                    "certificate": certificate,
                }
            )
            raw_rows.extend(
                {
                    **row,
                    "stage": "certification",
                    "head": head,
                    "policy_kind": policy_kind,
                }
                for row in certification
            )
            for index in range(40):
                probability = (index + 1 + head_index) / 44
                raw_rows.append(
                    {
                        "row_key": f"final-{index:03d}",
                        "group_id": f"final-{index:03d}",
                        "label": index % 2,
                        "probability": probability,
                        "action": _policy_action(probability, certificate),
                        "stage": "final_test",
                        "head": head,
                        "policy_kind": policy_kind,
                    }
                )
    return policies, raw_rows


def build_fixture_artifact(
    root: Path, *, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a complete small artifact that is independently reducible."""

    root.mkdir(parents=True, exist_ok=True)
    policies, raw_rows = _fixture_probability_rows()
    shards = _write_row_shard(root, raw_rows, name="fixture-rows.jsonl")
    base: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_fixture_measurement",
        "run_date": RUN_DATE,
        "started_at_utc": "2026-09-20T00:00:00+00:00",
        "completed_at_utc": "2026-09-20T00:00:01+00:00",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {"device": "fixture_host_cpu", "cuda_used": False},
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 1.0,
        "phase_spans": [],
        "preconditions_checked": [
            _precondition("fixture", "fixture", "fixture", "ready", "==", True, True, True)
        ],
        "random_seed": {"training": list(TRAINING_SEEDS), "bootstrap": BOOTSTRAP_SEED},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": 30,
            "attempted_units": 30,
            "completed_units": 30,
            "failed_units": 0,
            "censored_units": 0,
            "unstarted_units": 0,
            "stop_rule": "all fixture units complete",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {"required_checks_passed": True, "failed_required_checks": []},
        "verifier_is_oracle": False,
        "honest_verdict": "complete_null_fixture_measurement",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_duration_s": 0.0,
        "cold_start_duration_s": 0.0,
        "computation_duration_s": 1.0,
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "decision_capture_complete_score": 1,
        "decision_value_score": 0,
        "policy_certificates": policies,
        "probability_row_shards": shards,
        "checkpoint_manifest": [],
        "evaluation_reuse_disclosure": "Previously exposed RAGTruth labels remain exploratory.",
        "small_ebm_training": {
            "performed": True,
            "head_fit_time_s": 0.1,
            "calibration_time_s": 0.1,
            "policy_fit_time_s": 0.1,
            "generator_weights_fitted": False,
        },
        "deployment_certificate_valid": False,
        "certificate_scope": "exploratory_reused_corpus",
        "candidate_artifact": False,
        "fixture_artifact": True,
    }
    reduced = independent_reduce(base, root=root)
    base["independent_reduction"] = reduced
    base["decision_capture_complete_score"] = reduced["decision_capture_complete_score"]
    base["decision_value_score"] = reduced["decision_value_score"]
    base["acceptance_gate_results"] = reduced["benefit_gates"]
    base["reproducibility_checksum"] = artifact_checksum(base)
    return base


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover - runtime I/O.
    """Authenticate source bytes, upstream fields, flags, and the protocol seal."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    upstream = _load_object(root / UPSTREAM_PATH)
    protocol = _load_object(root / SEALED_PROTOCOL_PATH)
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
                observed == "readable_nonempty_bytes",
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": None,
            }
    identities = (
        (UPSTREAM_PATH, EXPECTED_UPSTREAM_SHA256, "exp7436-selection-protocol"),
        (SEALED_PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "exp7436-sealed-protocol"),
        (CORPUS_MANIFEST_PATH, EXPECTED_CORPUS_SHA256, "exp7423-corpus-manifest"),
    )
    for relative, expected, name in identities:
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            _precondition(
                f"exact_hash:{name}",
                name,
                relative.as_posix(),
                "sha256",
                "==",
                expected,
                observed,
                observed == expected,
            )
        )
    checks.extend(upstream_field_checks(upstream, path=UPSTREAM_PATH))
    checks.extend(
        [
            _precondition(
                "upstream_identity",
                "exp7436-selection-protocol",
                UPSTREAM_PATH.as_posix(),
                "experiment_id",
                "==",
                "exp7436-v652-selection-protocol",
                upstream.get("experiment_id"),
                upstream.get("experiment_id") == "exp7436-v652-selection-protocol",
            ),
            _precondition(
                "sealed_protocol_identity",
                "exp7436-sealed-protocol",
                SEALED_PROTOCOL_PATH.as_posix(),
                "manifest_hash",
                "==",
                EXPECTED_PROTOCOL_MANIFEST_HASH,
                protocol.get("manifest_hash"),
                protocol.get("manifest_hash") == EXPECTED_PROTOCOL_MANIFEST_HASH,
            ),
        ]
    )
    spec_text = SPEC_PATH.read_text(encoding="utf-8") if SPEC_PATH.is_file() else ""
    checks.append(
        _precondition(
            "driving_requirement",
            "openspec/capabilities/autoresearch/spec.md",
            "openspec/capabilities/autoresearch/spec.md",
            "REQ-*",
            "==",
            "REQ-AUTO-7439",
            "REQ-AUTO-7439" if "REQ-AUTO-7439" in spec_text else None,
            "REQ-AUTO-7439" in spec_text,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    quarantined = "experiment_id: 7439" in exclusion or "exp7439" in exclusion
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            "==",
            False,
            quarantined,
            not quarantined,
        )
    )
    if UPSTREAM_PATH.as_posix() in hashes:
        hashes[UPSTREAM_PATH.as_posix()].update(
            {
                "original_verdict_class": upstream.get("verdict_class"),
                "original_flagged_adversarial": upstream.get("flagged_adversarial"),
            }
        )
    return checks, hashes, {"upstream": upstream, "protocol": protocol}


def _join_labels(
    predictors: Sequence[Mapping[str, Any]], evaluators: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:  # pragma: no cover - runtime role boundary.
    """Join explicit evaluator authority after label-blind row selection."""

    by_key = {str(row.get("row_key") or ""): row for row in evaluators}
    output: list[JsonDict] = []
    for predictor in predictors:
        key = str(predictor.get("row_key") or "")
        evaluator = by_key.get(key)
        if evaluator is None:
            raise ValueError(f"evaluator_row_missing:{key}")
        label = evaluator.get("primary_label")
        if label not in {0, 1}:
            raise ValueError(f"evaluator_label_invalid:{key}")
        if evaluator.get("group_id") != predictor.get("group_id"):
            raise ValueError(f"predictor_evaluator_group_mismatch:{key}")
        output.append({**deepcopy(dict(predictor)), "label": int(label)})
    return output


def _lowest_row_per_group(
    rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:  # pragma: no cover - runtime role boundary.
    """Choose one response per source group without reading a label."""

    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group = str(row.get("group_id") or "")
        key = str(row.get("row_key") or "")
        if not group or not key:
            raise ValueError("representative rows require group and row identity")
        if group not in selected or key < str(selected[group]["row_key"]):
            selected[group] = row
    return [deepcopy(dict(selected[group])) for group in sorted(selected)]


def _role_rows(
    predictors: Sequence[Mapping[str, Any]], group_ids: Sequence[str], *, representatives: bool
) -> list[JsonDict]:  # pragma: no cover - runtime role boundary.
    """Select only groups named by the immutable Exp7436 protocol."""

    allowed = set(group_ids)
    selected = [deepcopy(dict(row)) for row in predictors if str(row.get("group_id")) in allowed]
    observed = {str(row["group_id"]) for row in selected}
    if observed != allowed:
        raise ValueError("sealed_role_membership_mismatch")
    return _lowest_row_per_group(selected) if representatives else selected


def _seed_scores(
    states: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:  # pragma: no cover - runtime numeric path.
    """Score one predictor-only matrix with all five frozen seed states."""

    matrix, _receipt_row = predictor_matrix(rows)
    return np.asarray([_predict_state(state, matrix) for state in states], dtype=np.float64)


def _calibrated_scores(
    states: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> np.ndarray:  # pragma: no cover - runtime numeric path.
    """Average seed probabilities before applying one frozen affine map."""

    means = np.mean(_seed_scores(states, rows), axis=0)
    return _apply_affine(means, calibration["affine"])


def _score_rows(
    rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float]
) -> list[JsonDict]:  # pragma: no cover - runtime numeric path.
    """Attach probabilities after labels have been authorized for one role."""

    return [
        {**deepcopy(dict(row)), "probability": float(probability)}
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def _write_checkpoints(
    root: Path, fitted: Mapping[str, Any]
) -> list[JsonDict]:  # pragma: no cover - runtime I/O.
    """Persist each numeric head state below a hash-bound checkpoint manifest."""

    directory = root / CHECKPOINT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    manifests: list[JsonDict] = []
    for head in ALL_FITTED_HEADS:
        for state in fitted["heads"][head]:
            relative = CHECKPOINT_DIR / f"{head}--{state['seed']}.json"
            path = root / relative
            atomic_json(path, state)
            manifests.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "head": head,
                    "seed": state["seed"],
                    "fit_partition": "fit",
                    "fit_input_sha256": fitted["fit_input_sha256"],
                    "source_receipt_class": "small_ebm_training",
                }
            )
    return manifests


def _terminal_commands(
    candidate: Path,
) -> list[PlannedCommand]:  # pragma: no cover - E2E subprocess plan.
    """Build fresh replay, reduction, adversarial, and strict row checks."""

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


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover - measured runtime boundary.
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


def _build_blocked_artifact(
    failed: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - external absence boundary.
    """Publish exact external absence without fabricating dependent measurements."""

    now = utc_now()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": f"blocked_precondition_{failed.get('check')}",
        "run_date": RUN_DATE,
        "started_at_utc": now,
        "completed_at_utc": now,
        "preconditions_checked": [deepcopy(dict(failed))],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "device": "host_cpu",
            "cuda_used": False,
            "external_device": None,
        },
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": None,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": 30,
            "attempted_units": 0,
            "completed_units": 0,
            "failed_units": 0,
            "censored_units": 0,
            "unstarted_units": 30,
            "stop_rule": "stop before dependent work when an external prerequisite fails",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "required_checks_passed": False,
            "blocked_upstream": failed.get("upstream"),
            "blocked_path": failed.get("path"),
            "blocked_check": failed.get("check"),
            "blocked_field": failed.get("field"),
            "blocked_expected": failed.get("expected"),
            "blocked_observed": failed.get("observed"),
        },
        "verifier_is_oracle": False,
        "honest_verdict": f"blocked_precondition_{failed.get('check')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "decision_capture_complete_score": 0,
        "decision_value_score": 0,
        "policy_certificates": [],
        "probability_row_shards": [],
        "checkpoint_manifest": [],
        "evaluation_reuse_disclosure": (
            "No dependent evaluation ran because an authenticated prerequisite failed."
        ),
        "small_ebm_training": {"performed": False},
        "deployment_certificate_valid": False,
        "certificate_scope": "exploratory_reused_corpus",
        "candidate_artifact": False,
        "fixture_artifact": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _build_artifact(
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    unit_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    probability_shards: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    timing: Mapping[str, float],
    candidate: bool,
    flagged_adversarial: bool,
) -> JsonDict:  # pragma: no cover - terminal artifact assembly.
    """Build one schema-complete record and reduce its hash-bound raw rows."""

    small_training = {
        "performed": True,
        "receipt_class": "small_ebm_training",
        "components": list(DEPLOYED_HEADS),
        "diagnostic_components": [DENSE_CONTROL, PREVALENCE_CONTROL],
        "generator_weights_fitted": False,
        "current_llm_calls": 0,
        "training_seeds": list(TRAINING_SEEDS),
        "maximum_steps": MAX_STEPS,
        "head_fit_time_s": float(timing["head_fit_time_s"]),
        "calibration_time_s": float(timing["calibration_time_s"]),
        "policy_fit_time_s": float(timing["policy_fit_time_s"]),
    }
    receipt = _receipt(
        root,
        started_ns=started_ns,
        ended_ns=ended_ns,
        phase_spans=phase_spans,
        small_training=small_training,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_pending_independent_reduction",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        **receipt,
        "model_duration_s": 0.0,
        "computation_duration_s": sum(
            float(span.get("duration_s") or 0.0)
            for span in phase_spans
            if span.get("phase")
            in {"training", "calibration_and_policy", "certification", "final_test"}
        ),
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
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [deepcopy(dict(row)) for row in unit_rows],
        "sample_size_budget": {
            "planned_units": len(ALL_FITTED_HEADS) * len(TRAINING_SEEDS) + 1,
            "attempted_units": len(unit_rows),
            "completed_units": sum(row.get("completed") is True for row in unit_rows),
            "failed_units": sum(row.get("failed") is True for row in unit_rows),
            "censored_units": sum(row.get("censored") is True for row in unit_rows),
            "unstarted_units": sum(row.get("unstarted") is True for row in unit_rows),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "stop_rule": "all frozen fits, one certificate read, and one final-test read",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "honest_verdict": "complete_null_pending_independent_reduction",
        "verdict_class": "null",
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_checks": list(AFFECTED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECK_NAMES),
        },
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "decision_capture_complete_score": 0,
        "decision_value_score": 0,
        "policy_certificates": [deepcopy(dict(row)) for row in policies],
        "probability_row_shards": [deepcopy(dict(row)) for row in probability_shards],
        "checkpoint_manifest": [deepcopy(dict(row)) for row in checkpoints],
        "diagnostics": deepcopy(dict(diagnostics)),
        "evaluation_reuse_disclosure": (
            "RAGTruth labels and evaluation roles were previously exposed. This is exploratory "
            "protocol evidence, not a fresh guarantee for new users."
        ),
        "small_ebm_training": small_training,
        "deployment_certificate_valid": False,
        "certificate_scope": "exploratory_reused_corpus",
        "fresh_new_user_safety_claimed": False,
        "candidate_artifact": candidate,
        "fixture_artifact": False,
    }
    reduced = independent_reduce(artifact, root=root)
    capture = int(reduced["decision_capture_complete_score"])
    value = int(reduced["decision_value_score"])
    artifact["decision_capture_complete_score"] = capture
    artifact["decision_value_score"] = value
    artifact["independent_reduction"] = reduced
    validation_ok = _validation_passed(validation_receipts, candidate=candidate)
    gates = [
        _gate(
            "authenticated_preconditions",
            "validity",
            "all",
            True,
            all(row.get("passed") is True for row in preconditions),
            all(row.get("passed") is True for row in preconditions),
            "Changed upstream bytes or flags cannot enter the measurement.",
        ),
        _gate(
            "affected_and_terminal_validation",
            "validation",
            "==",
            True,
            validation_ok,
            validation_ok,
            "Only the frozen scoped commands and terminal readers authorize capture.",
        ),
        _gate(
            "dense_spline_parity",
            "validity",
            "all",
            True,
            all(row.get("passed") is True for row in diagnostics["dense_spline_parity"]),
            all(row.get("passed") is True for row in diagnostics["dense_spline_parity"]),
            "Dense basis parity is a diagnostic and never adds certification rows.",
        ),
        *[deepcopy(row) for row in reduced["benefit_gates"]],
    ]
    required_failures = [
        row["check"]
        for row in gates
        if row["category"] in {"validity", "validation"} and row["passed"] is not True
    ]
    artifact["acceptance_gate_results"] = gates
    artifact["gate_check_summary"] = {
        "required_checks_passed": not required_failures,
        "failed_required_checks": required_failures,
        "scientific_benefit_passed": value == 1,
        "blocked_upstream": None,
        "blocked_path": None,
        "blocked_check": None,
        "blocked_field": None,
        "blocked_expected": None,
        "blocked_observed": None,
    }
    if required_failures or flagged_adversarial:
        honest = "complete_disqualified_required_validation_defect"
        verdict_class = "disqualified"
    elif value == 1:
        honest = "complete_positive_registered_decision_benefit"
        verdict_class = "positive"
    elif reduced["limited_policy_wrapper_finding"]:
        honest = "complete_null_policy_wrapper_coverage_only_no_energy_advantage"
        verdict_class = "null"
    else:
        honest = "complete_null_no_registered_decision_benefit"
        verdict_class = "null"
    artifact["status"] = honest
    artifact["honest_verdict"] = honest
    artifact["verdict_class"] = verdict_class
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


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
    preconditions, source_hashes, loaded = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = _build_blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return blocked

    protocol = loaded["protocol"]
    progress(run_started, "corpus", "before_model_load")
    phase_started = time.monotonic()
    reloaded = reload_corpus(root / CORPUS_DIR)
    readers = ProtocolReaders(reloaded)
    fit_predictors = _role_rows(
        readers.read_predictors("fit"), protocol["roles"]["fit"]["group_ids"], representatives=False
    )
    probability_predictors = readers.read_predictors("probability_calibration")
    calibration_predictors = _role_rows(
        probability_predictors,
        protocol["roles"]["probability_calibration"]["group_ids"],
        representatives=True,
    )
    tuning_predictors = _role_rows(
        probability_predictors,
        protocol["roles"]["policy_tuning"]["group_ids"],
        representatives=True,
    )
    fit_rows = _join_labels(fit_predictors, readers.read_evaluators("fit", EVALUATOR_TOKEN))
    probability_evaluators = readers.read_evaluators("probability_calibration", EVALUATOR_TOKEN)
    calibration_rows = _join_labels(calibration_predictors, probability_evaluators)
    tuning_rows = _join_labels(tuning_predictors, probability_evaluators)
    spans.append(_span("corpus", phase_started, run_started, len(fit_rows)))
    progress(run_started, "corpus", "after_model_load", completed=len(fit_rows))

    progress(run_started, "training", "before_training", planned=len(ALL_FITTED_HEADS) * 5)
    phase_started = time.monotonic()
    fitted = fit_compact_heads(fit_rows)
    checkpoints = _write_checkpoints(root, fitted)
    spans.append(_span("training", phase_started, run_started, len(checkpoints)))
    progress(run_started, "training", "after_training", completed=len(checkpoints))

    progress(run_started, "calibration_and_policy", "before_benchmark", planned=3)
    phase_started = time.monotonic()
    calibrations: dict[str, JsonDict] = {}
    frozen: dict[str, JsonDict] = {}
    policy_fit_elapsed = 0.0
    calibration_elapsed = 0.0
    for index, head in enumerate(DEPLOYED_HEADS, 1):
        progress(
            run_started, "calibration_and_policy", "before_benchmark", head=head, unit=f"{index}/3"
        )
        calibration_started = time.monotonic()
        raw_calibration = _seed_scores(fitted["heads"][head], calibration_rows)
        calibration = calibrate_seed_probabilities(
            raw_calibration, [int(row["label"]) for row in calibration_rows]
        )
        calibration_elapsed += time.monotonic() - calibration_started
        tuning_probabilities = _calibrated_scores(fitted["heads"][head], tuning_rows, calibration)
        scored_tuning = _score_rows(tuning_rows, tuning_probabilities)
        policy_started = time.monotonic()
        policy, candidates = freeze_policy(scored_tuning)
        policy_fit_elapsed += time.monotonic() - policy_started
        calibration["input_sha256"] = canonical_hash(
            {
                "row_keys": [row["row_key"] for row in calibration_rows],
                "mean_probabilities": np.mean(raw_calibration, axis=0).tolist(),
                "labels": [row["label"] for row in calibration_rows],
            }
        )
        calibrations[head] = calibration
        frozen[head] = {"policy": policy, "candidate_rows": candidates}
        progress(
            run_started, "calibration_and_policy", "after_benchmark", head=head, unit=f"{index}/3"
        )
    spans.append(_span("calibration_and_policy", phase_started, run_started, len(frozen)))

    progress(run_started, "certification", "before_benchmark", planned=9)
    phase_started = time.monotonic()
    certification_predictors = _role_rows(
        readers.read_predictors("policy_calibration"),
        protocol["roles"]["certification"]["group_ids"],
        representatives=True,
    )
    certification_rows = _join_labels(
        certification_predictors,
        readers.read_evaluators("policy_calibration", EVALUATOR_TOKEN),
    )
    policies: list[JsonDict] = []
    raw_rows: list[JsonDict] = []
    for index, head in enumerate(DEPLOYED_HEADS, 1):
        probabilities = _calibrated_scores(
            fitted["heads"][head], certification_rows, calibrations[head]
        )
        scored = _score_rows(certification_rows, probabilities)
        for policy_kind, policy in (
            ("tuned", frozen[head]["policy"]),
            ("old_fixed", deepcopy(OLD_POLICY)),
        ):
            certificate = certify_frozen_policy(scored, policy)
            policies.append(
                {
                    "head": head,
                    "policy_kind": policy_kind,
                    "frozen_policy": policy,
                    "certificate": certificate,
                    "calibration": {
                        "affine": calibrations[head]["affine"],
                        "fit_order": calibrations[head]["fit_order"],
                        "input_sha256": calibrations[head]["input_sha256"],
                    },
                    "registered_nine_check_member": policy_kind == "tuned",
                }
            )
            raw_rows.extend(
                {
                    "row_key": row["row_key"],
                    "group_id": row["group_id"],
                    "label": row["label"],
                    "probability": row["probability"],
                    "stage": "certification",
                    "head": head,
                    "policy_kind": policy_kind,
                }
                for row in scored
            )
        progress(
            run_started, "certification", "benchmark_unit_complete", head=head, completed=index * 3
        )
    spans.append(_span("certification", phase_started, run_started, 9))
    progress(run_started, "certification", "after_benchmark", completed=9)

    progress(run_started, "final_test", "before_benchmark", planned=1)
    phase_started = time.monotonic()
    final_predictors = _lowest_row_per_group(readers.read_predictors("final_test"))
    final_rows = _join_labels(
        final_predictors, readers.read_evaluators("final_test", EVALUATOR_TOKEN)
    )
    policy_map = {(row["head"], row["policy_kind"]): row for row in policies}
    for head in DEPLOYED_HEADS:
        probabilities = _calibrated_scores(fitted["heads"][head], final_rows, calibrations[head])
        for policy_kind in ("tuned", "old_fixed"):
            policy_row = policy_map[(head, policy_kind)]
            certificate = policy_row["certificate"]
            raw_rows.extend(
                {
                    "row_key": row["row_key"],
                    "group_id": row["group_id"],
                    "label": row["label"],
                    "probability": float(probability),
                    "action": _policy_action(float(probability), certificate),
                    "stage": "final_test",
                    "head": head,
                    "policy_kind": policy_kind,
                }
                for row, probability in zip(final_rows, probabilities, strict=True)
            )
    spans.append(_span("final_test", phase_started, run_started, len(final_rows)))
    progress(run_started, "final_test", "after_benchmark", completed=len(final_rows))

    probability_shards = _write_row_shard(root, raw_rows)
    for manifest in [*probability_shards, *checkpoints]:
        source_hashes[manifest["path"]] = {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "original_flagged_adversarial": None,
            **(
                {"source_receipt_class": manifest["source_receipt_class"]}
                if "source_receipt_class" in manifest
                else {}
            ),
        }
    for relative in (
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("openspec/capabilities/autoresearch/spec.md"),
    ):
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "original_flagged_adversarial": None,
        }
    unit_rows = [
        {
            "row_type": "small_ebm_fit",
            "arm": head,
            "seed": state["seed"],
            "condition": "full_source",
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "status": "completed",
            "checkpoint_sha256": state["final_checkpoint_sha256"],
        }
        for head in ALL_FITTED_HEADS
        for state in fitted["heads"][head]
    ]
    unit_rows.append(
        {
            "row_type": "no_training_prevalence_control",
            "arm": PREVALENCE_CONTROL,
            "seed": None,
            "condition": "full_source",
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "status": "completed",
            **fitted["diagnostics"]["prevalence"],
        }
    )
    timing = {
        "head_fit_time_s": float(fitted["fit_elapsed_s"]),
        "calibration_time_s": calibration_elapsed,
        "policy_fit_time_s": policy_fit_elapsed,
    }

    private_root = Path(tempfile.mkdtemp(prefix="exp7439-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
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
    candidate = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        unit_rows=unit_rows,
        policies=policies,
        probability_shards=probability_shards,
        checkpoints=checkpoints,
        diagnostics=fitted["diagnostics"],
        validation_receipts=affected,
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        timing=timing,
        candidate=True,
        flagged_adversarial=not affected_reduction["passed"] or bool(plan_errors),
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_commands = _terminal_commands(candidate_path)
    progress(run_started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_commands, log_dir=root / RAW_DIR / "validation/terminal"
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
    final = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        unit_rows=unit_rows,
        policies=policies,
        probability_shards=probability_shards,
        checkpoints=checkpoints,
        diagnostics=fitted["diagnostics"],
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        timing=timing,
        candidate=False,
        flagged_adversarial=(
            not affected_reduction["passed"] or not terminal_passed or critical or bool(plan_errors)
        ),
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date and strict replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one fresh-process artifact reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = validate_artifact(value, root=args.root) if value else ["artifact_unreadable"]
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        reduced = independent_reduce(value, root=args.root) if value else {"error": "unreadable"}
        passed = bool(value) and reduced == value.get("independent_reduction")
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return int(not passed)
    run_experiment(args.root.resolve(), args.date, output_path=args.output)
    return 0

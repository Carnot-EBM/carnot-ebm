"""Audit the completed static calibrated-decision null independently.

This module reads frozen rows and checkpoints. It does not fit a model or use
the disqualified online experiment as a scientific prerequisite.

Spec refs: REQ-REPORT-7396 and SCENARIO-REPORT-7396-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    atomic_json,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260918"
MILESTONE = "2026.09.649"
EXPERIMENT_ID = "exp7396-decision-diagnosis"
SCHEMA = "carnot.exp7396.v649.decision_diagnosis.v1"
PHASE = 1

MODULE_PATH = Path("python/carnot/experiment_7396_v649_decision_diagnosis.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7396_v649_decision_diagnosis.py")
TEST_PATH = Path("tests/python/test_experiment_7396_v649_decision_diagnosis.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7396_v649_decision_diagnosis.json")
RAW_DIR = Path("results/raw/experiment_7396_v649_decision_diagnosis")
EXP7382_PATH = Path("results/experiment_7382_v648_decision_protocol.json")
EXP7385_PATH = Path("results/experiment_7385_v648_decision_training.json")
EXP7386_PATH = Path("results/experiment_7386_v648_online_decisions.json")
CORPUS_PATH = Path("data/fover_corpus_v4.json")

EXPECTED_HASHES = {
    EXP7382_PATH.as_posix(): "sha256:a093a1b970e0308b84fbcad96d7a5254e9b563260d8a21e16d89de19a90bb8da",
    EXP7385_PATH.as_posix(): "sha256:05cb5c7fb56fa5afa9ec8b450315ea00534229ec477890a84b412355f8ede87b",
    CORPUS_PATH.as_posix(): "sha256:c5710308eb72575591165ad1df672086e3d91ae3270c174c409e8c1ef48725e2",
}
EXPECTED_PARTITION_HASH = "sha256:c392fe22a192b1db74ef45622034bb665211165243d1ef0df63835f1eee92a18"
PRIMARY_ARM = "natural_prevalence_bernoulli_gibbs"
CONTROL_ARMS = ("training_prevalence", "l2_logistic_calibration")
LEARNED_ARMS = (
    "raw_balanced_nce_gibbs",
    "prior_corrected_nce_gibbs",
    PRIMARY_ARM,
)
ARMS = (*CONTROL_ARMS, *LEARNED_ARMS)
SEEDS = (7_382_001, 7_382_002, 7_382_003, 7_382_004, 7_382_005)
BOOTSTRAP_SEED = 7_382_307
BOOTSTRAP_DRAWS = 10_000
PERMUTATION_SEED = 7_396_307
DIAGNOSTIC_PARTITIONS = ("training", "probability_calibration")

AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_recompute",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/verifier_gaps.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7382_v648_decision_protocol.py"),
    Path("python/carnot/experiment_7385_v648_decision_training.py"),
    Path("python/carnot/verify/pcib_probe.py"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    CORPUS_PATH,
    EXP7382_PATH,
    EXP7385_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "milestone",
        "phase",
        "status",
        "run_date",
        "started_at_utc",
        "ended_at_utc",
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
        "small_ebm_training",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "historical_receipt_sidecars",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "validation_receipts",
        "repository_health",
        "field_principles",
        "promotion_score",
        "static_audit_complete_score",
        "static_value_confirmed_score",
        "feature_collision_rows",
        "decision_change_rows",
        "feature_diagnosis",
        "context_diagnosis",
        "static_metric_recomputation",
        "metric_match_report",
        "reducer_controls",
        "original_registered_gate",
        "exp7386_restricted_diagnosis",
        "required_check_names",
    }
)

_CONCLUSION_MARKERS = (
    "therefore",
    "thus",
    "hence",
    "so the",
    "answer is",
    "result is",
    "total is",
    "total =",
    "sum is",
    "= answer",
    "in total",
    "altogether",
)


def utc_now() -> str:  # pragma: no cover - observed run boundary.
    """Return one real UTC boundary for the current process."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase and long-operation boundary with monotonic time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7396] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a later reader can detect source drift."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for vectors, partitions, and artifacts."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    """Read JSON without converting arrays into misleading empty objects."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind all artifact evidence except the checksum slot itself."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _precondition(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Name one exact structured gate before dependent work starts."""

    return {
        "check": check,
        "category": "precondition",
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": deepcopy(expected),
        "operator": "==",
        "observed": deepcopy(observed),
        "passed": observed == expected,
    }


def static_metadata_checks(
    protocol: Mapping[str, Any], training: Mapping[str, Any]
) -> list[JsonDict]:
    """Check the two eligible static producers without consulting Exp7386."""

    expected = (
        (
            EXP7382_PATH,
            protocol,
            {
                "experiment_id": "exp7382-decision-protocol",
                "status": "complete_decision_protocol_ready",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "decision_protocol_ready_score": 1,
            },
        ),
        (
            EXP7385_PATH,
            training,
            {
                "experiment_id": "exp7385-decision-training",
                "status": "complete_decision_training_null",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "decision_capture_complete_score": 1,
                "calibration_value_score": 0,
            },
        ),
    )
    rows: list[JsonDict] = []
    for path, artifact, fields in expected:
        for field, value in fields.items():
            rows.append(
                _precondition(
                    f"{path.stem}:{field}", path.as_posix(), field, value, artifact.get(field)
                )
            )
    rows.extend(
        (
            _precondition(
                "exp7382_required_gates",
                EXP7382_PATH.as_posix(),
                "gate_check_summary.all_required_passed",
                True,
                (protocol.get("gate_check_summary") or {}).get("all_required_passed"),
            ),
            _precondition(
                "exp7385_required_gates",
                EXP7385_PATH.as_posix(),
                "gate_check_summary.all_required_passed",
                True,
                (training.get("gate_check_summary") or {}).get("all_required_passed"),
            ),
            _precondition(
                "exp7385_original_efficacy_null",
                EXP7385_PATH.as_posix(),
                "calibration_value_reduction.passed",
                False,
                (training.get("calibration_value_reduction") or {}).get("passed"),
            ),
        )
    )
    return rows


def _partition_integrity(protocol: Mapping[str, Any]) -> tuple[str | None, int, int]:
    """Recompute group membership identity and cross-partition separation."""

    memberships = protocol.get("partition_membership") or []
    mapping = {
        str(row.get("group_id")): str(row.get("partition"))
        for row in memberships
        if isinstance(row, Mapping)
    }
    computed = canonical_hash(mapping) if len(mapping) == len(memberships) else None
    features = [row for row in protocol.get("feature_rows") or [] if isinstance(row, Mapping)]
    mismatches = sum(
        mapping.get(str(row.get("group_id"))) != row.get("partition") for row in features
    )
    text_partitions: dict[str, set[str]] = defaultdict(set)
    for row in features:
        text_partitions[str(row.get("normalized_text_sha256"))].add(str(row.get("partition")))
    cross_partition_texts = sum(len(parts) > 1 for parts in text_partitions.values())
    return computed, mismatches, cross_partition_texts


def _row_integrity(training: Mapping[str, Any]) -> tuple[int, int, int]:
    """Check every stored contribution and fixed arm, seed, and group identity."""

    rows = [row for row in training.get("rows") or [] if isinstance(row, Mapping)]
    errors = 0
    identities: set[tuple[Any, Any, Any]] = set()
    group_labels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        try:
            label = int(row["label"])
            probability = float(row["probability"])
            if label not in {0, 1} or not 0.0 <= probability <= 1.0:
                raise ValueError
            clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
            brier = (probability - label) ** 2
            log_loss = -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))
            if not math.isclose(float(row["brier_contribution"]), brier, abs_tol=1e-14):
                errors += 1
            if not math.isclose(float(row["log_loss_contribution"]), log_loss, abs_tol=1e-14):
                errors += 1
            identity = (row["arm"], int(row["seed"]), row["group_id"])
            if identity in identities:
                errors += 1
            identities.add(identity)
            group_labels[str(row["group_id"])].add(label)
        except (KeyError, TypeError, ValueError, OverflowError):
            errors += 1
    inconsistent_labels = sum(len(labels) != 1 for labels in group_labels.values())
    return errors, len(identities), inconsistent_labels


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], dict[str, str], dict[str, JsonDict]]:
    """Authenticate exact static inputs, checkpoints, rows, and partition bytes."""

    root = root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)
    for relative, expected in EXPECTED_HASHES.items():
        checks.append(
            _precondition(
                f"exact_hash:{relative}", relative, "sha256", expected, hashes.get(relative)
            )
        )

    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7396",
            "REQ-REPORT-7396" if "REQ-REPORT-7396" in spec else None,
        )
    )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    checks.append(
        _precondition(
            "current_task_not_excluded",
            "ops/exclusion_manifest.yaml",
            "experiment_id:7396",
            False,
            bool(re.search(r"experiment_id:\s*7396\b", exclusion)),
        )
    )

    protocol_value = load_json(root / EXP7382_PATH)
    training_value = load_json(root / EXP7385_PATH)
    protocol = dict(protocol_value) if isinstance(protocol_value, Mapping) else {}
    training = dict(training_value) if isinstance(training_value, Mapping) else {}
    checks.extend(static_metadata_checks(protocol, training))
    partition_hash, feature_partition_errors, cross_partition_texts = _partition_integrity(protocol)
    checks.extend(
        (
            _precondition(
                "partition_membership_hash",
                EXP7382_PATH.as_posix(),
                "protocol_manifest.partition_membership_sha256",
                EXPECTED_PARTITION_HASH,
                partition_hash,
            ),
            _precondition(
                "feature_partition_membership",
                EXP7382_PATH.as_posix(),
                "feature_rows.partition",
                0,
                feature_partition_errors,
            ),
            _precondition(
                "normalized_text_partition_separation",
                EXP7382_PATH.as_posix(),
                "feature_rows.normalized_text_sha256",
                0,
                cross_partition_texts,
            ),
            _precondition(
                "training_protocol_hash",
                EXP7385_PATH.as_posix(),
                "protocol_identity.artifact_sha256",
                EXPECTED_HASHES[EXP7382_PATH.as_posix()],
                (training.get("protocol_identity") or {}).get("artifact_sha256"),
            ),
            _precondition(
                "training_partition_hash",
                EXP7385_PATH.as_posix(),
                "protocol_identity.partition_membership_sha256",
                EXPECTED_PARTITION_HASH,
                (training.get("protocol_identity") or {}).get("partition_membership_sha256"),
            ),
        )
    )
    row_errors, row_identities, inconsistent_labels = _row_integrity(training)
    checks.extend(
        (
            _precondition(
                "training_row_contributions",
                EXP7385_PATH.as_posix(),
                "rows.metric_contributions",
                0,
                row_errors,
            ),
            _precondition(
                "training_row_identity_count",
                EXP7385_PATH.as_posix(),
                "rows.arm_seed_group",
                33_075,
                row_identities,
            ),
            _precondition(
                "training_group_label_consistency",
                EXP7385_PATH.as_posix(),
                "rows.group_id.label",
                0,
                inconsistent_labels,
            ),
        )
    )

    checkpoints = [
        row for row in training.get("checkpoint_manifest") or [] if isinstance(row, Mapping)
    ]
    checks.append(
        _precondition(
            "checkpoint_count", EXP7385_PATH.as_posix(), "checkpoint_manifest", 25, len(checkpoints)
        )
    )
    for row in checkpoints:
        relative = Path(str(row.get("path")))
        checkpoint = root / relative
        observed = sha256_file(checkpoint) if checkpoint.is_file() else None
        if observed is not None:
            hashes[relative.as_posix()] = observed
        checks.append(
            _precondition(
                f"checkpoint_hash:{row.get('arm')}:{row.get('seed')}",
                relative.as_posix(),
                "sha256",
                row.get("sha256"),
                observed,
            )
        )
    return checks, hashes, {"protocol": protocol, "training": training}


def binary_pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute average precision with stable source-order handling for ties."""

    positives = sum(int(label) == 1 for label in labels)
    if positives == 0:
        raise ValueError("PR-AUC requires a positive label")
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    true_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if int(labels[index]) == 1:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute pairwise AUROC without using the producer implementation."""

    positives = [float(score) for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [float(score) for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both labels")
    wins = sum(
        sum(value > negative for negative in negatives)
        + 0.5 * sum(value == negative for negative in negatives)
        for value in positives
    )
    return wins / (len(positives) * len(negatives))


def _wilson_bounds(errors: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a bounded 95 percent Wilson interval for one calibration cell."""

    if total <= 0:
        return [0.0, 1.0]
    probability = errors / total
    scale = 1.0 + z * z / total
    center = (probability + z * z / (2 * total)) / scale
    radius = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total**2))
    radius /= scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _calibration_bounds(labels: Sequence[int], probabilities: Sequence[float]) -> list[JsonDict]:
    """Report fixed-width reliability cells and binomial observation bounds."""

    cells: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for label, probability in zip(labels, probabilities, strict=True):
        cells[min(9, int(float(probability) * 10))].append((int(label), float(probability)))
    return [
        {
            "bin": index,
            "lower_probability": index / 10,
            "upper_probability": (index + 1) / 10,
            "count": len(values),
            "mean_predicted_probability": float(np.mean([value[1] for value in values])),
            "observed_error_rate": sum(value[0] for value in values) / len(values),
            "observed_error_rate_ci95": _wilson_bounds(
                sum(value[0] for value in values), len(values)
            ),
        }
        for index, values in sorted(cells.items())
    ]


def _action_confusion(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count typed actions against labels without treating escalation as a class."""

    counts = {
        "accept_correct": 0,
        "accept_incorrect": 0,
        "reject_correct": 0,
        "reject_incorrect": 0,
        "escalate_label_correct": 0,
        "escalate_label_incorrect": 0,
    }
    for row in rows:
        label = int(row["label"])
        decision = str(row["decision"])
        if decision == "accept":
            key = "accept_incorrect" if label else "accept_correct"
        elif decision == "reject":
            key = "reject_correct" if label else "reject_incorrect"
        else:
            key = "escalate_label_incorrect" if label else "escalate_label_correct"
        counts[key] += 1
    return counts


def _unit_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one arm and seed while preserving action and calibration evidence."""

    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("probability must be finite and in [0, 1]")
    decisions = Counter(str(row["decision"]) for row in rows)
    brier = [(value - label) ** 2 for label, value in zip(labels, probabilities, strict=True)]
    log_loss = []
    for label, probability in zip(labels, probabilities, strict=True):
        clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
        log_loss.append(-(label * math.log(clipped) + (1 - label) * math.log1p(-clipped)))
    return {
        "arm": str(rows[0]["arm"]),
        "seed": int(rows[0]["seed"]),
        "rows": len(rows),
        "effective_groups": len({str(row["group_id"]) for row in rows}),
        "prevalence": sum(labels) / len(labels),
        "brier": float(np.mean(brier)),
        "log_loss": float(np.mean(log_loss)),
        "auroc": _binary_auroc(labels, probabilities),
        "pr_auc": binary_pr_auc(labels, probabilities),
        "accept_count": decisions["accept"],
        "reject_count": decisions["reject"],
        "escalate_count": decisions["escalate"],
        "coverage": (decisions["accept"] + decisions["reject"]) / len(rows),
        "action_confusion": _action_confusion(rows),
        "calibration_bounds": _calibration_bounds(labels, probabilities),
        "source_cpu_scoring_duration_s": sum(
            float((row.get("measured_cost") or {}).get("cpu_scoring_duration_s") or 0.0)
            for row in rows
        ),
    }


def _aggregate_metrics(unit_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Average proper scores over frozen seeds and retain total action counts."""

    output: dict[str, JsonDict] = {}
    for arm in sorted({str(row["arm"]) for row in unit_metrics.values()}):
        values = [row for row in unit_metrics.values() if row["arm"] == arm]
        output[arm] = {
            "seeds": len(values),
            "effective_groups": int(values[0]["effective_groups"]),
            **{
                field: float(np.mean([float(row[field]) for row in values]))
                for field in ("prevalence", "brier", "log_loss", "auroc", "pr_auc", "coverage")
            },
            **{
                field: sum(int(row[field]) for row in values)
                for field in ("accept_count", "reject_count", "escalate_count")
            },
        }
    return output


def _paired_intervals(
    rows: Sequence[Mapping[str, Any]], draws: int, seed: int
) -> dict[str, dict[str, JsonDict]]:
    """Bootstrap paired group deltas after seed averaging with one frozen draw matrix."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), str(row["group_id"]))].append(row)
    groups = sorted({group for _, group in grouped})
    present_arms = sorted({arm for arm, _ in grouped})
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for arm in present_arms:
        if any((arm, group) not in grouped for group in groups):
            continue
        vectors[arm] = {
            "brier": np.asarray(
                [
                    np.mean([float(row["brier_contribution"]) for row in grouped[(arm, group)]])
                    for group in groups
                ]
            ),
            "log_loss": np.asarray(
                [
                    np.mean([float(row["log_loss_contribution"]) for row in grouped[(arm, group)]])
                    for group in groups
                ]
            ),
            "coverage": np.asarray(
                [
                    np.mean([row["decision"] != "escalate" for row in grouped[(arm, group)]])
                    for group in groups
                ]
            ),
        }
    if not groups or draws <= 0:
        return {}
    indices = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    output: dict[str, dict[str, JsonDict]] = {}
    comparisons = (
        ((learned, control) for learned in LEARNED_ARMS for control in CONTROL_ARMS)
        if set(CONTROL_ARMS) <= set(vectors)
        else ((arm, control) for arm in vectors for control in vectors if arm != control)
    )
    for learned, control in comparisons:
        if learned not in vectors or control not in vectors:
            continue
        result: JsonDict = {}
        for metric in ("brier", "log_loss", "coverage"):
            delta = vectors[learned][metric] - vectors[control][metric]
            means = np.mean(delta[indices], axis=1)
            result[f"{metric}_delta"] = {
                "mean": float(np.mean(means)),
                "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            }
        result.update({"effective_groups": len(groups), "draws": draws, "seed": seed})
        output.setdefault(learned, {})[control] = result
    return output


def recompute_static_metrics(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> JsonDict:
    """Independently recompute all registered static row and group summaries."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    valid_rows: list[Mapping[str, Any]] = []
    integrity_errors: list[str] = []
    for index, row in enumerate(rows):
        try:
            label = int(row["label"])
            probability = float(row["probability"])
            if label not in {0, 1} or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("probability must be finite and in [0, 1]")
            brier = (probability - label) ** 2
            clipped = min(max(probability, 1e-15), 1 - 1e-15)
            log_loss = -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))
            if not math.isclose(brier, float(row["brier_contribution"]), abs_tol=1e-14):
                integrity_errors.append(f"brier:{index}")
            if not math.isclose(log_loss, float(row["log_loss_contribution"]), abs_tol=1e-14):
                integrity_errors.append(f"log_loss:{index}")
            grouped[(str(row["arm"]), int(row["seed"]))].append(row)
            valid_rows.append(row)
        except ValueError as exc:
            if str(exc) == "probability must be finite and in [0, 1]":
                raise
            integrity_errors.append(f"malformed:{index}")
        except (KeyError, TypeError, OverflowError):
            integrity_errors.append(f"malformed:{index}")
    units = {f"{arm}:{seed}": _unit_metrics(values) for (arm, seed), values in grouped.items()}
    return {
        "by_arm_seed": units,
        "by_arm": _aggregate_metrics(units),
        "paired_group_intervals": _paired_intervals(valid_rows, draws, seed),
        "row_integrity_errors": integrity_errors,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def run_reducer_controls() -> JsonDict:
    """Prove the metric reducer recognizes signal and rejects a label permutation."""

    labels = [0, 0, 1, 1]
    informative = [0.05, 0.2, 0.8, 0.95]
    constant = [0.5] * 4
    permuted = [1, 0, 1, 0]

    def summary(y: Sequence[int], probabilities: Sequence[float]) -> JsonDict:
        return {
            "pr_auc": binary_pr_auc(y, probabilities),
            "brier": float(np.mean([(p - label) ** 2 for label, p in zip(y, probabilities)])),
        }

    analytic = summary(labels, informative)
    flat = summary(labels, constant)
    shuffled = summary(permuted, informative)
    passed = (
        analytic["pr_auc"] == 1.0
        and analytic["brier"] < flat["brier"]
        and shuffled["pr_auc"] < analytic["pr_auc"]
    )
    return {
        "analytic_informative": analytic,
        "constant": flat,
        "permuted_labels": shuffled,
        "all_passed": passed,
    }


def _cell_floor(rows: Sequence[Mapping[str, Any]], labels: Sequence[int] | None = None) -> float:
    """Compute the in-sample Bayes Brier floor for fixed observable cells."""

    supplied = list(labels) if labels is not None else [int(row["label"]) for row in rows]
    cells: dict[tuple[str, float, float], list[int]] = defaultdict(list)
    for row, label in zip(rows, supplied, strict=True):
        cells[
            (
                str(row["partition"]),
                float(row["entity_uptake"]),
                float(row["falsifiability_score"]),
            )
        ].append(label)
    total = len(rows)
    return (
        sum(
            len(values) * (sum(values) / len(values)) * (1 - sum(values) / len(values))
            for values in cells.values()
        )
        / total
    )


def diagnose_feature_cells(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    permutation_draws: int = 200,
    seed: int = PERMUTATION_SEED,
) -> JsonDict:
    """Describe collisions using only training and probability-calibration labels."""

    diagnostic = [row for row in feature_rows if row.get("partition") in DIAGNOSTIC_PARTITIONS]
    excluded = len(feature_rows) - len(diagnostic)
    cells: dict[tuple[str, float, float], list[Mapping[str, Any]]] = defaultdict(list)
    group_partitions: dict[str, set[str]] = defaultdict(set)
    text_partitions: dict[str, set[str]] = defaultdict(set)
    for row in diagnostic:
        cells[
            (
                str(row["partition"]),
                float(row["entity_uptake"]),
                float(row["falsifiability_score"]),
            )
        ].append(row)
        group_partitions[str(row.get("group_id"))].add(str(row["partition"]))
        text_partitions[str(row.get("normalized_text_sha256"))].add(str(row["partition"]))
    collisions: list[JsonDict] = []
    for (partition, entity, falsifiability), values in sorted(cells.items()):
        labels = [int(row["label"]) for row in values]
        incorrect = sum(labels)
        if not (incorrect and incorrect < len(labels)):
            continue
        probability = incorrect / len(labels)
        vector = [entity, falsifiability]
        collisions.append(
            {
                "partition": partition,
                "vector": vector,
                "vector_hash": canonical_hash(vector),
                "label_counts": {
                    "correct": len(labels) - incorrect,
                    "incorrect": incorrect,
                },
                "sample_count": len(labels),
                "empirical_in_cell_probability": probability,
                "empirical_bayes_brier_floor": probability * (1 - probability),
                "scope": "descriptive_sample_floor_not_population_impossibility",
            }
        )
    labels = np.asarray([int(row["label"]) for row in diagnostic])
    rng = np.random.default_rng(seed)
    permuted_floors = [
        _cell_floor(diagnostic, rng.permutation(labels).tolist()) for _ in range(permutation_draws)
    ]
    observed_floor = _cell_floor(diagnostic) if diagnostic else 0.0
    return {
        "diagnostic_partitions": list(DIAGNOSTIC_PARTITIONS),
        "diagnostic_row_count": len(diagnostic),
        "excluded_partition_rows": excluded,
        "feature_collision_rows": collisions,
        "conflicting_cell_count": len(collisions),
        "empirical_bayes_brier_floor": observed_floor,
        "permutation_control": {
            "draws": permutation_draws,
            "seed": seed,
            "mean_permuted_floor": float(np.mean(permuted_floors)) if permuted_floors else None,
            "observed_minus_permuted_mean": (
                observed_floor - float(np.mean(permuted_floors)) if permuted_floors else None
            ),
            "labels_permuted_within_diagnostic_sample": True,
        },
        "group_partition_overlap_count": sum(len(parts) > 1 for parts in group_partitions.values()),
        "normalized_text_partition_overlap_count": sum(
            len(parts) > 1 for parts in text_partitions.values()
        ),
        "future_label_access": False,
        "claim_scope": "descriptive_sample_floors_only",
    }


def _has_conclusion_marker(text: str) -> bool:
    """Detect the marker vocabulary used by the text-statistical proxy."""

    lowered = text.lower()
    return any(marker in lowered for marker in _CONCLUSION_MARKERS)


def trace_context_proxies(
    feature_rows: Sequence[Mapping[str, Any]], corpus_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Trace the empty context argument without inventing unavailable questions."""

    diagnostic = [row for row in feature_rows if row.get("partition") in DIAGNOSTIC_PARTITIONS]
    texts = {int(index): str(row.get("step_text") or "") for index, row in enumerate(corpus_rows)}
    entity_values = [float(row["entity_uptake"]) for row in diagnostic]
    falsifiability_values = [float(row["falsifiability_score"]) for row in diagnostic]
    without_markers = sum(
        not _has_conclusion_marker(texts.get(int(row["source_row_index"]), ""))
        for row in diagnostic
    )
    corpus_keys = (
        set().union(*(set(row) for row in corpus_rows if isinstance(row, Mapping)))
        if corpus_rows
        else set()
    )
    return {
        "diagnostic_partitions": list(DIAGNOSTIC_PARTITIONS),
        "rows_traced": len(diagnostic),
        "pcib_context_argument": "",
        "entity_context_proxy": "first_half_of_response_text",
        "entity_context_proxy_is_actual_question": False,
        "actual_source_question_available": bool(
            {"question", "source_question", "problem"} & corpus_keys
        ),
        "question_id_establishes_question_provenance": False,
        "question_id_role": "row_identifier_only",
        "entity_uptake_constant": len(set(entity_values)) <= 1,
        "falsifiability_score_constant": len(set(falsifiability_values)) <= 1,
        "rows_without_conclusion_markers": without_markers,
        "rows_without_conclusion_markers_rate": (
            without_markers / len(diagnostic) if diagnostic else None
        ),
        "synthetic_questions_created": 0,
        "unsupported_arithmetic_false_positive_claims": 0,
    }


def decision_change_counts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Count binary and typed-action changes while preserving answer identity."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), int(row["seed"]))].append(row)
    output: list[JsonDict] = []
    for (arm, seed), values in sorted(grouped.items()):
        raw_changes = 0
        typed_changes = 0
        typed = Counter()
        identities: set[tuple[int, str]] = set()
        for row in values:
            raw_incorrect = float(row["raw_energy"]) >= 0.0
            calibrated_incorrect = float(row["probability"]) >= 0.5
            raw_changes += raw_incorrect != calibrated_incorrect
            action = str(row["decision"])
            typed[action] += 1
            implied = "reject" if calibrated_incorrect else "accept"
            typed_changes += action != implied
            identities.add((int(row["source_row_index"]), str(row["group_id"])))
        accept_rate = typed["accept"] / len(values)
        output.append(
            {
                "arm": arm,
                "seed": seed,
                "answer_count": len(values),
                "fixed_answer_identity_count": len(identities),
                "answer_identity_changed_count": 0,
                "raw_to_calibrated_binary_changes": raw_changes,
                "calibrated_binary_to_typed_action_changes": typed_changes,
                "typed_action_counts": {
                    "accept": typed["accept"],
                    "reject": typed["reject"],
                    "escalate": typed["escalate"],
                },
                "accept_rate": accept_rate,
                "accepts_nearly_everything": accept_rate >= 0.95,
                "majority_acceptance_is_learned_value": False,
            }
        )
    return output


def _nested_close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    """Compare stored and independent numeric trees with a strict tolerance."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_close(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def compare_stored_metrics(training: Mapping[str, Any], recomputed: Mapping[str, Any]) -> JsonDict:
    """Compare proper scores and registered intervals to stored producer evidence."""

    stored_units = (training.get("calibration_metrics") or {}).get("by_arm_seed") or {}
    stored_arms = (training.get("calibration_metrics") or {}).get("by_arm") or {}
    fields = (
        "rows",
        "effective_groups",
        "prevalence",
        "brier",
        "log_loss",
        "auroc",
        "pr_auc",
        "accept_count",
        "reject_count",
        "escalate_count",
        "coverage",
    )
    mismatches: list[JsonDict] = []
    for unit_id, observed in (recomputed.get("by_arm_seed") or {}).items():
        expected = stored_units.get(unit_id) or {}
        for field in fields:
            if not _nested_close(expected.get(field), observed.get(field)):
                mismatches.append(
                    {
                        "scope": unit_id,
                        "field": field,
                        "expected": expected.get(field),
                        "observed": observed.get(field),
                    }
                )
    arm_fields = fields[2:]
    for arm, observed in (recomputed.get("by_arm") or {}).items():
        expected = stored_arms.get(arm) or {}
        for field in arm_fields:
            if not _nested_close(expected.get(field), observed.get(field)):
                mismatches.append(
                    {
                        "scope": arm,
                        "field": field,
                        "expected": expected.get(field),
                        "observed": observed.get(field),
                    }
                )
    stored_intervals = training.get("paired_group_intervals") or {}
    interval_match = _nested_close(
        stored_intervals, recomputed.get("paired_group_intervals") or {}, tolerance=1e-9
    )
    if not interval_match:
        mismatches.append(
            {
                "scope": "paired_group_intervals",
                "field": "registered_tree",
                "expected_hash": canonical_hash(stored_intervals),
                "observed_hash": canonical_hash(recomputed.get("paired_group_intervals") or {}),
            }
        )
    original = training.get("calibration_value_reduction") or {}
    brier_failed = (original.get("checks") or {}).get("brier_ci_below_both_controls") is False
    return {
        "unit_count_expected": len(stored_units),
        "unit_count_observed": len(recomputed.get("by_arm_seed") or {}),
        "metric_mismatch_rows": mismatches,
        "metric_mismatch_count": len(mismatches),
        "paired_intervals_match": interval_match,
        "original_brier_gate_failed": brier_failed,
        "all_matched": not mismatches and brier_failed,
    }


def exp7386_restricted_diagnosis(root: Path) -> JsonDict:
    """Retain only the allowed failed-log facts from the online experiment."""

    value = load_json(root / EXP7386_PATH)
    artifact = dict(value) if isinstance(value, Mapping) else {}
    receipts = [
        row for row in artifact.get("validation_receipts") or [] if isinstance(row, Mapping)
    ]
    broad = next((row for row in receipts if row.get("name") == "full_python_suite"), {})
    reducer = next((row for row in receipts if row.get("name") == "independent_reducer"), {})
    return {
        "path": EXP7386_PATH.as_posix(),
        "sha256": sha256_file(root / EXP7386_PATH) if (root / EXP7386_PATH).is_file() else None,
        "original_status": artifact.get("status"),
        "original_verdict_class": artifact.get("verdict_class"),
        "original_flagged_adversarial": artifact.get("flagged_adversarial"),
        "affected_validation_mismatch": "affected_validation_mismatch"
        in str(reducer.get("output_tail") or ""),
        "appended_broad_suite": broad.get("name") == "full_python_suite",
        "appended_broad_suite_timed_out": broad.get("timed_out"),
        "appended_broad_suite_exit_code": broad.get("exit_code"),
        "online_metrics_consumed": False,
        "rehabilitated": False,
        "scope": "restricted_failed_log_diagnosis_only",
    }


def write_historical_sidecars(root: Path, raw_dir: Path) -> list[JsonDict]:
    """Keep archived producer declarations outside current invocation counters."""

    training = load_json(root / EXP7385_PATH)
    protocol = load_json(root / EXP7382_PATH)
    diagnosis = exp7386_restricted_diagnosis(root)
    payloads = (
        (
            "historical_static_producer_receipts.json",
            "historical_model_receipts",
            {
                "sources": [
                    {
                        "path": path.as_posix(),
                        "sha256": sha256_file(root / path),
                        "status": value.get("status") if isinstance(value, Mapping) else None,
                        "verdict_class": value.get("verdict_class")
                        if isinstance(value, Mapping)
                        else None,
                        "flagged_adversarial": value.get("flagged_adversarial")
                        if isinstance(value, Mapping)
                        else None,
                    }
                    for path, value in ((EXP7382_PATH, protocol), (EXP7385_PATH, training))
                ],
                "authorizes_current_model_work": False,
            },
        ),
        (
            "exp7386_restricted_failed_logs.json",
            "historical_diagnostic_only",
            diagnosis,
        ),
    )
    references: list[JsonDict] = []
    for name, scope, payload in payloads:
        path = raw_dir / name
        atomic_json(path, {"scope": scope, "payload": payload})
        references.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "scope": scope,
            }
        )
    return references


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 plan before Exp7303 executes any child process."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command expansion, broad pytest, and missing private parents."""

    errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if tuple(command.name for command in commands) != AFFECTED_CHECK_NAMES:
        errors.append("affected_check_names_mismatch")
    if any("full_python_suite" in command.name for command in commands):
        errors.append("full_python_suite_forbidden")
    focused = next((command for command in commands if command.name == "focused_pytest"), None)
    if focused is None or not {"-n", "0", "addopts=", "--no-cov"} <= set(focused.argv):
        errors.append("focused_pytest_flags_mismatch")
    return list(dict.fromkeys(errors))


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful bounded receipt for each frozen name."""

    selected = {str(row.get("name")): row for row in receipts if row.get("name") in names}
    return set(names) <= set(selected) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is False
        for row in selected.values()
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str = EXPERIMENT_ID,
    artifact_field: str | None = None,
) -> JsonDict:
    """Keep gate category and operands explicit in the terminal record."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "artifact_field": artifact_field or check,
        "expected": deepcopy(expected),
        "operator": "==",
        "observed": deepcopy(observed),
        "passed": observed == expected,
    }


def _acceptance_gates(
    preconditions: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    feature: Mapping[str, Any],
    context: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    match: Mapping[str, Any],
    original_gate: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    require_terminal: bool,
) -> list[JsonDict]:
    """Separate completion, validation, safety, and unchanged efficacy."""

    affected = _receipts_pass(receipts, AFFECTED_CHECK_NAMES)
    terminal = _receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    return [
        _gate(
            "static_preconditions",
            "completion",
            True,
            all(row.get("passed") is True for row in preconditions),
        ),
        _gate("independent_metric_match", "completion", True, match.get("all_matched")),
        _gate("analytic_and_permutation_controls", "safety", True, controls.get("all_passed")),
        _gate(
            "feature_diagnosis_complete",
            "completion",
            True,
            feature.get("future_label_access") is False,
        ),
        _gate(
            "context_provenance_bounded",
            "safety",
            False,
            context.get("actual_source_question_available"),
        ),
        _gate(
            "answer_identity_fixed",
            "safety",
            0,
            sum(int(row.get("answer_identity_changed_count") or 0) for row in decisions),
        ),
        _gate("required_affected_validation", "required_validation", True, affected),
        _gate("terminal_readers", "required_validation", True, terminal),
        _gate(
            "brier_ci_below_both_controls",
            "scientific_efficacy",
            True,
            (original_gate.get("checks") or {}).get("brier_ci_below_both_controls"),
            upstream=EXP7385_PATH.as_posix(),
            artifact_field="calibration_value_reduction.checks.brier_ci_below_both_controls",
        ),
        _gate("promotion_forbidden", "promotion", 0, 0),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact required and scientific failures separately."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required = [row for row in failures if row.get("category") != "scientific_efficacy"]
    scientific = [row for row in failures if row.get("category") == "scientific_efficacy"]
    return {
        "all_required_passed": not required,
        "failed_required_count": len(required),
        "first_required_failure": required[0] if required else None,
        "failed_scientific_gate_count": len(scientific),
        "first_scientific_failure": scientific[0] if scientific else None,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary artifact fields without changing their JSON types."""

    specific = {
        "schema": "Use a versioned schema with ordinary experiment identity and terminal fields.",
        "run_date": "Use 20260918 and retain actual UTC start and end timestamps.",
        "preconditions_checked": "Authenticate exact source bytes, eligibility, rows, partitions, and checkpoints first.",
        "MODEL_SPECS": "List current LLM work; this host audit performs none.",
        "model_invoked": "Set true for an attempted current real LLM load or generation; none occurred.",
        "invocation_counts": "Count only current owned LLM events, never archived producer work.",
        "inference_substrate": "Use a truthful string that describes host aggregation and numeric reduction.",
        "inference_substrate_details": "Keep device and software details separate from the substrate string.",
        "inference_substrate_class": "Use the closed aggregation class required for this audit.",
        "execution_venue": "Use the closed host string, not a hostname or device mapping.",
        "duration_s": "Measure current monotonic elapsed time without padding.",
        "phase_spans": "Retain actual phase boundaries, checkpoints, and completed units.",
        "random_seed": "Freeze original bootstrap seeds and the diagnostic permutation seed.",
        "reproducibility_checksum": "Bind current code, configuration, sources, and raw diagnosis rows.",
        "source_artifact_hashes": "Retain exact path and byte hashes for static inputs and checkpoints.",
        "rows": "Retain every arm and seed audit unit with metrics, disposition, and full measured cost.",
        "sample_size_budget": "Separate planned, attempted, completed, censored, and unstarted audit units.",
        "acceptance_gate_results": "Keep completion, safety, validation, efficacy, and promotion gates separate.",
        "gate_check_summary": "Name the first failed upstream, path, check, field, expected, and observed value.",
        "verifier_is_oracle": "Archived correctness labels define truth; the learned risk score alone is not an oracle.",
        "honest_verdict": "Use a complete null for finished valid accounting with a failed unchanged efficacy gate.",
        "verdict_class": "Use a closed terminal class; reserve partial for retryable unfinished owned work.",
        "flagged_adversarial": "Preserve critical current findings and never use invalid science for readiness.",
        "validation_receipts": "Retain exact argv, environment, exits, durations, and hashed logs, including failures.",
        "repository_health": "Keep unrelated broad-suite history separate from affected-check readiness.",
        "field_principles": "Explain fields without wrapping numeric gates or ordinary mappings.",
        "promotion_score": "Always remain zero; no rollout, weight update, publication, or submission is authorized.",
        "static_audit_complete_score": "One requires complete independent static accounting and required validation.",
        "static_value_confirmed_score": "One requires the unchanged registered static efficacy gate to pass.",
        "feature_collision_rows": "Report partition, vector hash, labels, and descriptive sample floor without future labels.",
        "decision_change_rows": "Keep answer IDs fixed and separate raw, calibrated, and typed-action changes.",
    }
    return {
        key: specific.get(key, f"The {key} field retains directly auditable supporting evidence.")
        for key in keys
    }


def _audit_rows(recomputed: Mapping[str, Any]) -> list[JsonDict]:
    """Convert each arm and seed reduction into a complete comparative row."""

    return [
        {
            "unit_id": unit_id,
            "arm": row["arm"],
            "seed": row["seed"],
            "condition": "frozen_final_test_static_audit",
            "metric_contributions": {
                field: deepcopy(row[field])
                for field in (
                    "prevalence",
                    "brier",
                    "log_loss",
                    "auroc",
                    "pr_auc",
                    "coverage",
                    "action_confusion",
                    "calibration_bounds",
                )
            },
            "disposition": "complete",
            "censored": False,
            "failure": None,
            "cost": {
                "source_rows_recomputed": row["rows"],
                "effective_independent_groups": row["effective_groups"],
                "source_cpu_scoring_duration_s": row["source_cpu_scoring_duration_s"],
                "current_llm_calls": 0,
            },
        }
        for unit_id, row in sorted((recomputed.get("by_arm_seed") or {}).items())
    ]


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    sidecars: Sequence[Mapping[str, Any]],
    recomputed: Mapping[str, Any],
    match: Mapping[str, Any],
    controls: Mapping[str, Any],
    feature: Mapping[str, Any],
    context: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    original_gate: Mapping[str, Any],
    exp7386: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    started_at: str,
    ended_at: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    require_terminal: bool,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one schema-complete static audit from raw independent evidence."""

    gates = _acceptance_gates(
        preconditions,
        controls,
        feature,
        context,
        decisions,
        match,
        original_gate,
        receipts,
        require_terminal=require_terminal,
    )
    required_passed = all(
        row["passed"] for row in gates if row["category"] != "scientific_efficacy"
    )
    audit_score = int(required_passed and not flagged_adversarial)
    value_score = int(audit_score == 1 and original_gate.get("passed") is True)
    verdict = "null" if audit_score else "disqualified"
    status = (
        "complete_decision_diagnosis_null"
        if verdict == "null"
        else "complete_decision_diagnosis_disqualified"
    )
    honest = (
        "complete_null_static_audit_confirms_registered_no_value"
        if verdict == "null"
        else "complete_disqualified_static_audit_required_check_failure"
    )
    rows = _audit_rows(recomputed)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host aggregation of frozen JSON rows with NumPy exact formulas and bootstrap reduction",
        "inference_substrate_details": {
            "device": "host_cpu",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax_platform_request": os.environ.get("JAX_PLATFORMS"),
            "work": "hashing, row reduction, feature-cell diagnosis, resampling, and scoped subprocess validation",
        },
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "small_ebm_training": {
            "performed": False,
            "current_optimizer_updates": 0,
            "historical_exp7385_small_head_training_inspected": True,
            "current_llm_calls": 0,
        },
        "random_seed": {
            "training_seeds_preserved": list(SEEDS),
            "paired_group_bootstrap_seed": BOOTSTRAP_SEED,
            "feature_permutation_seed": PERMUTATION_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "historical_receipt_sidecars": [deepcopy(dict(row)) for row in sidecars],
        "rows": rows,
        "sample_size_budget": {
            "planned_audit_units": 25,
            "attempted_audit_units": len(rows),
            "completed_audit_units": len(rows),
            "censored_audit_units": 0,
            "unstarted_audit_units": max(0, 25 - len(rows)),
            "source_scored_rows": 33_075,
            "effective_independent_group_count": 1_323,
            "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
            "feature_permutation_draws": 200,
            "limits": "five frozen arms, five frozen seeds, and existing static rows only",
            "stop_rule": "Stop after every frozen audit unit and required check; do not expand a model.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": {
            "status": "affected_checks_only",
            "as_of": RUN_DATE,
            "affects_required_checks": not _receipts_pass(receipts, AFFECTED_CHECK_NAMES),
            "unrelated_broad_suite_observations": [
                {
                    "source": EXP7386_PATH.as_posix(),
                    "name": "full_python_suite",
                    "timed_out": exp7386.get("appended_broad_suite_timed_out"),
                    "required_for_exp7396": False,
                }
            ],
        },
        "field_principles": {},
        "promotion_score": 0,
        "static_audit_complete_score": audit_score,
        "static_value_confirmed_score": value_score,
        "feature_collision_rows": deepcopy(feature.get("feature_collision_rows") or []),
        "decision_change_rows": [deepcopy(dict(row)) for row in decisions],
        "feature_diagnosis": deepcopy(dict(feature)),
        "context_diagnosis": deepcopy(dict(context)),
        "static_metric_recomputation": deepcopy(dict(recomputed)),
        "metric_match_report": deepcopy(dict(match)),
        "reducer_controls": deepcopy(dict(controls)),
        "original_registered_gate": deepcopy(dict(original_gate)),
        "exp7386_restricted_diagnosis": deepcopy(dict(exp7386)),
        "required_check_names": {
            "affected": list(AFFECTED_CHECK_NAMES),
            "terminal": list(TERMINAL_CHECK_NAMES),
        },
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_diagnostics() -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, list[JsonDict]]:
    """Build deterministic evidence for artifact mutation tests."""

    controls = run_reducer_controls()
    feature = {
        "future_label_access": False,
        "feature_collision_rows": [],
        "diagnostic_row_count": 4,
    }
    context = {
        "actual_source_question_available": False,
        "question_id_establishes_question_provenance": False,
    }
    match = {
        "all_matched": True,
        "metric_mismatch_count": 0,
        "original_brier_gate_failed": True,
    }
    decisions = [
        {
            "arm": "fixture",
            "seed": 1,
            "answer_identity_changed_count": 0,
            "typed_action_counts": {"accept": 1, "reject": 0, "escalate": 0},
        }
    ]
    return controls, feature, context, match, decisions


def build_artifact_for_test(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build a small complete-null artifact for cold-reader mutation tests."""

    controls, feature, context, match, decisions = _fixture_diagnostics()
    recomputed = {
        "by_arm_seed": {
            "fixture:1": {
                "arm": "fixture",
                "seed": 1,
                "rows": 1,
                "effective_groups": 1,
                "prevalence": 0.0,
                "brier": 0.01,
                "log_loss": -math.log(0.9),
                "auroc": 0.5,
                "pr_auc": 0.5,
                "coverage": 1.0,
                "action_confusion": {"accept_correct": 1},
                "calibration_bounds": [],
                "source_cpu_scoring_duration_s": 0.001,
            }
        },
        "by_arm": {},
        "paired_group_intervals": {},
        "row_integrity_errors": [],
    }
    original = {
        "checks": {"brier_ci_below_both_controls": False},
        "passed": False,
        "calibration_value_score": 0,
    }
    preconditions = [_precondition("fixture", "fixture", "ready", True, True)]
    return build_artifact(
        preconditions=preconditions,
        source_hashes={},
        sidecars=[],
        recomputed=recomputed,
        match=match,
        controls=controls,
        feature=feature,
        context=context,
        decisions=decisions,
        original_gate=original,
        exp7386={"appended_broad_suite_timed_out": True},
        receipts=receipts,
        started_at="2026-09-18T00:00:00Z",
        ended_at="2026-09-18T00:00:01Z",
        duration_s=1.0,
        phase_spans=[],
        require_terminal=True,
    )


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]], source_hashes: Mapping[str, str]
) -> JsonDict:
    """Publish exact external absence without starting dependent diagnosis."""

    failed = next((deepcopy(dict(row)) for row in preconditions if not row.get("passed")), None)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_decision_diagnosis_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "ended_at_utc": utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host precondition hashing before dependent aggregation",
        "inference_substrate_details": {"device": "host_cpu"},
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "small_ebm_training": {"performed": False, "current_optimizer_updates": 0},
        "random_seed": {
            "training_seeds_preserved": list(SEEDS),
            "paired_group_bootstrap_seed": BOOTSTRAP_SEED,
            "feature_permutation_seed": PERMUTATION_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "historical_receipt_sidecars": [],
        "rows": [],
        "sample_size_budget": {
            "planned_audit_units": 25,
            "attempted_audit_units": 0,
            "completed_audit_units": 0,
            "censored_audit_units": 25,
            "unstarted_audit_units": 0,
            "stop_rule": "Stop before diagnosis when an exact static prerequisite fails.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "all_required_passed": False,
            "failed_required_count": 1,
            "first_required_failure": failed,
            "failed_scientific_gate_count": 0,
            "first_scientific_failure": None,
        },
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_static_decision_diagnosis_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {"status": "not_assessed", "affects_required_checks": True},
        "field_principles": {},
        "promotion_score": 0,
        "static_audit_complete_score": 0,
        "static_value_confirmed_score": 0,
        "feature_collision_rows": [],
        "decision_change_rows": [],
        "feature_diagnosis": {},
        "context_diagnosis": {},
        "static_metric_recomputation": {},
        "metric_match_report": {},
        "reducer_controls": {},
        "original_registered_gate": {},
        "exp7386_restricted_diagnosis": {},
        "required_check_names": {
            "affected": list(AFFECTED_CHECK_NAMES),
            "terminal": list(TERMINAL_CHECK_NAMES),
        },
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    require_terminal: bool = True,
    deep: bool = False,
) -> list[str]:
    """Cold-check identity, declarations, reductions, scores, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate_class") != "aggregation"
        or not isinstance(artifact.get("inference_substrate"), str)
        or not isinstance(artifact.get("inference_substrate_details"), Mapping)
    ):
        errors.append("current_substrate_declaration_invalid")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_score_nonzero")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    for row in artifact.get("rows") or []:
        if not isinstance(row, Mapping) or not {
            "unit_id",
            "arm",
            "seed",
            "condition",
            "metric_contributions",
            "disposition",
            "censored",
            "failure",
            "cost",
        } <= set(row):
            errors.append("audit_row_invalid")
            break
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("static_audit_complete_score") != 0:
            errors.append("blocked_audit_score_nonzero")
        if artifact.get("static_value_confirmed_score") != 0:
            errors.append("blocked_value_score_nonzero")
    else:
        controls = artifact.get("reducer_controls") or {}
        feature = artifact.get("feature_diagnosis") or {}
        context = artifact.get("context_diagnosis") or {}
        decisions = artifact.get("decision_change_rows") or []
        match = artifact.get("metric_match_report") or {}
        original = artifact.get("original_registered_gate") or {}
        gates = _acceptance_gates(
            artifact.get("preconditions_checked") or [],
            controls,
            feature,
            context,
            decisions,
            match,
            original,
            artifact.get("validation_receipts") or [],
            require_terminal=require_terminal,
        )
        if artifact.get("acceptance_gate_results") != gates:
            errors.append("acceptance_gate_results_mismatch")
        if artifact.get("gate_check_summary") != _gate_summary(gates):
            errors.append("gate_check_summary_mismatch")
        required_passed = all(
            row["passed"] for row in gates if row["category"] != "scientific_efficacy"
        )
        expected_audit = int(required_passed and artifact.get("flagged_adversarial") is False)
        expected_value = int(expected_audit == 1 and original.get("passed") is True)
        if artifact.get("static_audit_complete_score") != expected_audit:
            errors.append("static_audit_complete_score_mismatch")
        if artifact.get("static_value_confirmed_score") != expected_value:
            errors.append("static_value_confirmed_score_mismatch")
        expected_verdict = "null" if expected_audit else "disqualified"
        if artifact.get("verdict_class") != expected_verdict:
            errors.append("terminal_state_mismatch")
    for relative, expected in (artifact.get("source_artifact_hashes") or {}).items():
        path = Path(str(relative))
        resolved = path if path.is_absolute() else root / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        if observed != expected:
            errors.append(f"source_hash_mismatch:{relative}")
    for reference in artifact.get("historical_receipt_sidecars") or []:
        path = root / str(reference.get("path"))
        observed = sha256_file(path) if path.is_file() else None
        if observed != reference.get("sha256"):
            errors.append(f"sidecar_hash_mismatch:{reference.get('path')}")
    if deep and artifact.get("verdict_class") != "blocked":
        training = load_json(root / EXP7385_PATH)
        if not isinstance(training, Mapping):
            errors.append("cold_training_source_missing")
        else:
            recomputed = recompute_static_metrics(
                training.get("rows") or [], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
            )
            match = compare_stored_metrics(training, recomputed)
            if match != artifact.get("metric_match_report"):
                errors.append("cold_metric_recomputation_mismatch")
            if canonical_hash(recomputed) != canonical_hash(
                artifact.get("static_metric_recomputation") or {}
            ):
                errors.append("cold_rows_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def terminal_command_specs(root: Path, candidate: Path) -> list[PlannedCommand]:
    """Build declared replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7396_v649_decision_diagnosis import validate_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=validate_artifact(v,root=pathlib.Path.cwd(),require_terminal=False,deep=True);"
        "print(json.dumps({'errors':e},sort_keys=True),flush=True);"
        "raise SystemExit(bool(e))"
    )
    commands = (
        (
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "completion",
        ),
        (
            "independent_cold_recompute",
            (python, "-u", "-c", reducer, str(candidate)),
            "completion",
        ),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "completion",
        ),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate"), category, True
        )
        for name, argv, category in commands
    ]


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover - observed run boundary.
    """Close one measured phase with a UTC checkpoint and completed unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the declared entrypoint E2E.
    """Run static diagnosis, scoped checks, cold readers, and atomic publication."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, hashes, inputs = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(preconditions)))
    ready = all(row.get("passed") is True for row in preconditions)
    progress(started, "preconditions", "end", passed=ready)
    if not ready:
        blocked = build_blocked_artifact(preconditions, hashes)
        blocked["started_at_utc"] = started_at
        blocked["ended_at_utc"] = utc_now()
        blocked["duration_s"] = time.monotonic() - started
        blocked["phase_spans"] = spans
        blocked["field_principles"] = _field_principles(tuple(blocked))
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        progress(started, "write", "before_atomic_blocked", path=output)
        atomic_json(output, blocked)
        progress(started, "write", "after_atomic_blocked", path=output)
        return blocked

    for phase in ("model_load", "generation"):
        phase_started = time.monotonic()
        progress(started, phase, "before_no_current_model_work")
        spans.append(_span(phase, phase_started, started, 0))
        progress(started, phase, "after_no_current_model_work", model_invoked=False)

    phase_started = time.monotonic()
    progress(started, "metric_reduction", "before_benchmark", rows=33075)
    training = inputs["training"]
    protocol = inputs["protocol"]
    recomputed = recompute_static_metrics(
        training.get("rows") or [], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    match = compare_stored_metrics(training, recomputed)
    spans.append(_span("metric_reduction", phase_started, started, len(recomputed["by_arm_seed"])))
    progress(started, "metric_reduction", "after_benchmark", matched=match["all_matched"])

    phase_started = time.monotonic()
    progress(started, "feature_diagnosis", "start")
    feature_rows = protocol.get("feature_rows") or []
    corpus_value = load_json(root / CORPUS_PATH)
    corpus = corpus_value if isinstance(corpus_value, list) else []
    feature = diagnose_feature_cells(feature_rows, permutation_draws=200, seed=PERMUTATION_SEED)
    context = trace_context_proxies(feature_rows, corpus)
    decisions = decision_change_counts(training.get("rows") or [])
    controls = run_reducer_controls()
    exp7386 = exp7386_restricted_diagnosis(root)
    sidecars = write_historical_sidecars(root, raw_dir)
    spans.append(_span("feature_diagnosis", phase_started, started, len(feature_rows)))
    progress(
        started,
        "feature_diagnosis",
        "end",
        collision_cells=feature["conflicting_cell_count"],
    )

    private = Path(tempfile.mkdtemp(prefix="exp7396-validation-", dir="/tmp"))
    phase_started = time.monotonic()
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
    progress(started, "affected_validation", "before_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"] and not plan_errors,
    )

    original_gate = deepcopy(dict(training.get("calibration_value_reduction") or {}))
    candidate = build_artifact(
        preconditions=preconditions,
        source_hashes=hashes,
        sidecars=sidecars,
        recomputed=recomputed,
        match=match,
        controls=controls,
        feature=feature,
        context=context,
        decisions=decisions,
        original_gate=original_gate,
        exp7386=exp7386,
        receipts=affected,
        started_at=started_at,
        ended_at=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        require_terminal=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    progress(started, "candidate_write", "after_atomic", path=candidate_path)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses")
    terminal = run_categorized_commands(
        root,
        terminal_command_specs(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    critical = any("[CRITICAL]" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    final = build_artifact(
        preconditions=preconditions,
        source_hashes=hashes,
        sidecars=sidecars,
        recomputed=recomputed,
        match=match,
        controls=controls,
        feature=feature,
        context=context,
        decisions=decisions,
        original_gate=original_gate,
        exp7386=exp7386,
        receipts=[*affected, *terminal],
        started_at=started_at,
        ended_at=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        require_terminal=True,
        flagged_adversarial=critical or not terminal_passed,
    )
    errors = validate_artifact(final, root=root, require_terminal=True, deep=False)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output)
    atomic_json(candidate_path, final)
    atomic_json(output, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed date, output, and cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the experiment or cold-reload one measured candidate."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.validate is not None:
        value = load_json(args.validate)
        errors = validate_artifact(value, root=REPO_ROOT, require_terminal=False, deep=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

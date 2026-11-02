"""Measure source-grounded calibration against three matched controls.

The learner predicts agreement with machine annotations. It does not establish
semantic truth. Source IDs and evaluator-only values never enter a feature
vector. Spec refs: REQ-AUTO-7413 and SCENARIO-AUTO-7413-*.
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
from carnot.experiment_7385_v648_decision_training import fit_affine_transform
from carnot.experiment_7410_v650_source_corpus import (
    FINAL_TEST_TOKEN,
    CorpusReaders,
    reload_corpus,
)
from carnot.experiment_7412_v650_source_features import (
    MAX_STEPS,
    SOURCE_FEATURE_NAMES,
    TRAINING_SEEDS,
    extract_feature_row,
    exact_risk_certificate,
    fit_gibbs_head,
    fit_logistic_control,
    probability_from_energy,
    typed_decision,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
EXPERIMENT_ID = "exp7413-source-calibration"
SCHEMA = "carnot.exp7413.v650.source_calibration.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7413_v650_source_calibration.json")
RAW_DIR = Path("results/raw/experiment_7413_v650_source_calibration")
MODULE_PATH = Path("python/carnot/experiment_7413_v650_source_calibration.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7413_v650_source_calibration.py")
TEST_PATH = Path("tests/python/test_experiment_7413_v650_source_calibration.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
UPSTREAM_PATH = Path("results/experiment_7412_v650_source_features.json")
PROTOCOL_PATH = Path("results/raw/experiment_7412_v650_source_features/protocol_manifest.json")
FEATURE_PATH = Path("results/raw/experiment_7412_v650_source_features/source_feature_rows.json")
CORPUS_PATH = Path("results/raw/experiment_7410_v650_source_corpus/corpus_manifest.json")
CORPUS_DIR = CORPUS_PATH.parent

EXPECTED_HASHES = {
    UPSTREAM_PATH: "sha256:b6ff5100e270646ad98961513956bcdc07073668e034f764d1bd64d20fb17874",
    PROTOCOL_PATH: "sha256:cef859226f0dc2f135992ecc594416fc93634e2ffbea44249b0470aa58175bbf",
    FEATURE_PATH: "sha256:a34e26beee7f797bb68fe69af1be00420c1c2cc3bbb1b203ebc89dd59df36864",
    CORPUS_PATH: "sha256:be0f0b29d6216eaf9a2c1a8ddefd98a5761256c2c9a091038baaf901ab4c277a",
}
EXPECTED_PROTOCOL_HASH = "sha256:65a9a5b817f4cfbef1524efa809bc1203402dfeffd5e400b93dabe2f249407af"
EXPECTED_CORPUS_HASH = "sha256:2843bdf316cb6e75f0fcd878704ca3ac937eaa43b2812ef2b73bea3106f7b5d1"

ARMS = (
    "training_prevalence",
    "l2_logistic_six_input",
    "response_only_2_4_1_gibbs",
    "source_aware_6_4_1_gibbs",
)
PRIMARY_ARM = "source_aware_6_4_1_gibbs"
BASELINE_ARMS = ARMS[:-1]
THRESHOLD_PAIRS = tuple(
    (accept, reject) for accept in (0.01, 0.025, 0.05) for reject in (0.90, 0.95, 0.99)
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_501_307
INFERENCE_SUBSTRATE = "no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
CONDITIONS = ("full_source", "source_removed", "cross_group_source_permutation")
SCORED_PARTITIONS = ("train", "probability_calibration", "policy_calibration", "final_test")
RESPONSE_FEATURE_NAMES = ("entity_uptake", "falsifiability_score")
PROHIBITED_FEATURE_NAMES = {
    "source_id",
    "article_id",
    "teacher_score",
    "gold_triple",
    "label",
    "expected_verdict",
    "source_relation",
}

REQUIRED_ARTIFACT_FIELDS = (
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
    "calibration_capture_complete_score",
    "calibration_value_score",
    "checkpoint_manifest",
    "paired_metric_rows",
    "source_ablation_rows",
    "label_authority",
)

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
    Path("python/carnot/experiment_7385_v648_decision_training.py"),
    Path("python/carnot/experiment_7396_v649_decision_diagnosis.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    Path("python/carnot/experiment_7410_v650_source_corpus.py"),
    Path("python/carnot/experiment_7412_v650_source_features.py"),
    SPEC_PATH,
    UPSTREAM_PATH,
    PROTOCOL_PATH,
    FEATURE_PATH,
    CORPUS_PATH,
)

V650_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_metric_recompute",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def utc_now() -> str:
    """Return one actual UTC boundary for a receipt or checkpoint."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush a phase or long-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7413] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Read a JSON object while treating malformed external bytes as absent."""

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
    passed: bool | None = None,
) -> JsonDict:
    """Record an exact prerequisite so a blocked result is actionable."""

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


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate exact frozen inputs before labels or optimizer work are read."""

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
        if observed:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": (
                    False
                    if relative
                    in {UPSTREAM_PATH, Path("results/experiment_7410_v650_source_corpus.json")}
                    else None
                ),
            }
    upstream = _load_object(root / UPSTREAM_PATH)
    protocol = _load_object(root / PROTOCOL_PATH)
    features = _load_object(root / FEATURE_PATH)
    corpus = _load_object(root / CORPUS_PATH)
    loaded = {
        "upstream": upstream,
        "protocol": protocol,
        "features": features,
        "corpus": corpus,
    }
    for relative, expected_hash in EXPECTED_HASHES.items():
        path = root / relative
        checks.append(
            _precondition(
                f"exact_hash:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "sha256",
                "==",
                expected_hash,
                sha256_file(path) if path.is_file() else None,
            )
        )
    verdict = upstream.get("verdict_class")
    allowed = ["positive", "circular_positive", "null"]
    checks.extend(
        [
            _precondition(
                "upstream_protocol_ready",
                "exp7412-source-features",
                UPSTREAM_PATH.as_posix(),
                "source_feature_protocol_ready_score",
                "==",
                1,
                upstream.get("source_feature_protocol_ready_score"),
            ),
            _precondition(
                "upstream_verdict_allowed",
                "exp7412-source-features",
                UPSTREAM_PATH.as_posix(),
                "verdict_class",
                "in",
                allowed,
                verdict,
                verdict in allowed,
            ),
            _precondition(
                "upstream_unflagged",
                "exp7412-source-features",
                UPSTREAM_PATH.as_posix(),
                "flagged_adversarial",
                "==",
                False,
                upstream.get("flagged_adversarial"),
            ),
            _precondition(
                "protocol_manifest_hash",
                PROTOCOL_PATH.as_posix(),
                PROTOCOL_PATH.as_posix(),
                "manifest_hash",
                "==",
                EXPECTED_PROTOCOL_HASH,
                protocol.get("manifest_hash"),
            ),
            _precondition(
                "corpus_manifest_hash",
                CORPUS_PATH.as_posix(),
                CORPUS_PATH.as_posix(),
                "manifest_hash",
                "==",
                EXPECTED_CORPUS_HASH,
                corpus.get("manifest_hash"),
            ),
            _precondition(
                "feature_row_count",
                FEATURE_PATH.as_posix(),
                FEATURE_PATH.as_posix(),
                "record_count",
                "==",
                protocol.get("feature_row_count"),
                len(features.get("records") or []),
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
            "REQ-AUTO-7413",
            "REQ-AUTO-7413" if "REQ-AUTO-7413" in spec_text else None,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            "==",
            False,
            "experiment_id: 7413" in exclusion,
        )
    )
    return checks, hashes, loaded


def _finite_vector(values: Sequence[Any], expected: int) -> list[float]:
    """Return a finite bounded feature vector with an exact frozen width."""

    vector = [float(value) for value in values]
    if len(vector) != expected or any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in vector
    ):
        raise ValueError("feature vector must have the frozen width and finite values in [0, 1]")
    return vector


def feature_vector(row: Mapping[str, Any], arm: str) -> list[float]:
    """Select only the registered values and reject extra feature fields."""

    if arm not in ARMS:
        raise ValueError(f"registered arm required: {arm}")
    source = row.get("source_features")
    response = row.get("response_only_ablation")
    if isinstance(source, Mapping) and PROHIBITED_FEATURE_NAMES.intersection(source):
        raise ValueError("prohibited feature keys are not allowed")
    if isinstance(response, Mapping) and PROHIBITED_FEATURE_NAMES.intersection(response):
        raise ValueError("prohibited feature keys are not allowed")
    if not isinstance(source, Mapping) or set(source) != set(SOURCE_FEATURE_NAMES):
        raise ValueError("source feature keys do not match the frozen protocol")
    if not isinstance(response, Mapping) or set(response) != set(RESPONSE_FEATURE_NAMES):
        raise ValueError("response feature keys do not match the frozen protocol")
    if arm == "response_only_2_4_1_gibbs":
        return _finite_vector([response[name] for name in RESPONSE_FEATURE_NAMES], 2)
    return _finite_vector([source[name] for name in SOURCE_FEATURE_NAMES], 6)


def _logit(probability: float) -> float:
    """Convert a probability to a finite logit for affine calibration."""

    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid(value: float) -> float:
    """Evaluate the logistic function without overflow."""

    return probability_from_energy(value)


def _raw_logit(unit: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    """Evaluate one registered numeric checkpoint before affine calibration."""

    arm = str(unit["arm"])
    weights = unit["weights"]
    if arm == "training_prevalence":
        return _logit(float(weights["probability"]))
    vector = np.asarray(feature_vector(row, arm), dtype=np.float64)
    if arm == "l2_logistic_six_input":
        return float(vector @ np.asarray(weights["coef"], dtype=np.float64) + weights["bias"])
    w1 = np.asarray(weights["w1"], dtype=np.float64)
    hidden_linear = w1 @ vector + np.asarray(weights["b1"], dtype=np.float64)
    hidden = hidden_linear / (1.0 + np.exp(-hidden_linear))
    return float(np.asarray(weights["w_out"], dtype=np.float64) @ hidden + weights["b_out"])


def _probability(unit: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[float, float]:
    """Return raw and calibrated probabilities from one row."""

    raw_logit = _raw_logit(unit, row)
    affine = unit["affine"]
    calibrated_logit = float(affine["slope"]) * raw_logit + float(affine["intercept"])
    return _sigmoid(raw_logit), _sigmoid(calibrated_logit)


def _scored_partition(rows: Sequence[Mapping[str, Any]], partition: str) -> list[JsonDict]:
    """Select scored rows from exactly one declared role."""

    return [
        deepcopy(dict(row))
        for row in rows
        if row.get("partition") == partition and row.get("label") in {0, 1}
    ]


def _partition_hash(rows: Sequence[Mapping[str, Any]], partition: str) -> str:
    """Bind identities and labels used by one fitting stage."""

    return canonical_hash(
        [
            {"row_key": row["row_key"], "group_id": row["group_id"], "label": row["label"]}
            for row in _scored_partition(rows, partition)
        ]
    )


def _label_blind_representatives(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Choose the lowest row key in each group without consulting its label."""

    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group = str(row.get("group_id") or "")
        key = str(row.get("row_key") or "")
        if not group or not key:
            raise ValueError("group and row identity are required")
        current = selected.get(group)
        if current is None or key < str(current["row_key"]):
            selected[group] = row
    return [deepcopy(dict(selected[group])) for group in sorted(selected)]


def _select_policy(
    unit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], JsonDict]:
    """Certify every frozen threshold pair on policy-calibration groups only."""

    policy_rows = _scored_partition(rows, "policy_calibration")
    if not policy_rows or {int(row["label"]) for row in policy_rows} != {0, 1}:
        raise ValueError("partition support requires both policy labels")
    representatives = _label_blind_representatives(policy_rows)
    evaluated: list[JsonDict] = []
    for index, (accept_threshold, reject_threshold) in enumerate(THRESHOLD_PAIRS):
        decisions = []
        for row in representatives:
            _raw, probability = _probability(unit, row)
            decisions.append(
                typed_decision(
                    probability, accept_threshold, reject_threshold, "policy-calibration"
                )["decision"]
            )
        accept_harm = [
            int(row["label"] == 1)
            for row, decision in zip(representatives, decisions, strict=True)
            if decision == "accept"
        ]
        reject_harm = [
            int(row["label"] == 0)
            for row, decision in zip(representatives, decisions, strict=True)
            if decision == "reject"
        ]
        accept_certificate = exact_risk_certificate(accept_harm, risk_budget=0.05)
        reject_certificate = exact_risk_certificate(reject_harm, risk_budget=0.10)
        enabled = [
            decision
            if (decision == "accept" and accept_certificate["action_enabled"])
            or (decision == "reject" and reject_certificate["action_enabled"])
            else "escalate"
            for decision in decisions
        ]
        decided = sum(decision != "escalate" for decision in enabled)
        correct = sum(
            (decision == "accept" and row["label"] == 0)
            or (decision == "reject" and row["label"] == 1)
            for row, decision in zip(representatives, enabled, strict=True)
        )
        harmful = sum(
            (decision == "accept" and row["label"] == 1)
            or (decision == "reject" and row["label"] == 0)
            for row, decision in zip(representatives, enabled, strict=True)
        )
        evaluated.append(
            {
                "threshold_index": index,
                "accept_threshold": accept_threshold,
                "reject_threshold": reject_threshold,
                "accept_certificate": accept_certificate,
                "reject_certificate": reject_certificate,
                "coverage": decided / len(representatives),
                "utility": (correct - harmful) / len(representatives),
                "representative_groups": len(representatives),
            }
        )
    selected = max(
        evaluated,
        key=lambda row: (
            float(row["coverage"]),
            float(row["utility"]),
            -int(row["threshold_index"]),
        ),
    )
    policy = {
        "threshold_index": selected["threshold_index"],
        "accept_threshold": selected["accept_threshold"],
        "reject_threshold": selected["reject_threshold"],
        "accept_enabled": selected["accept_certificate"]["action_enabled"],
        "reject_enabled": selected["reject_certificate"]["action_enabled"],
        "selection_partition": "policy_calibration",
        "selection_rule": "coverage_then_utility_then_registered_order",
        "representative_groups": len(representatives),
    }
    return evaluated, policy


def fit_registered_units(
    rows: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str] = ARMS,
    condition: str = "full_source",
    steps: int = MAX_STEPS,
    emit_progress: bool = False,
) -> list[JsonDict]:
    """Fit, calibrate, and select policies for registered arms and seeds."""

    if condition not in CONDITIONS or any(arm not in ARMS for arm in arms):
        raise ValueError("registered arm and condition required")
    partitions = {partition: _scored_partition(rows, partition) for partition in SCORED_PARTITIONS}
    if any(
        not part or {int(row["label"]) for row in part} != {0, 1} for part in partitions.values()
    ):
        raise ValueError("partition support requires both labels in every scored role")
    hashes = {partition: _partition_hash(rows, partition) for partition in SCORED_PARTITIONS}
    units: list[JsonDict] = []
    started = time.monotonic()
    planned = len(arms) * len(TRAINING_SEEDS)
    for arm in arms:
        training = partitions["train"]
        train_x = np.asarray([feature_vector(row, arm) for row in training], dtype=np.float64)
        train_y = np.asarray([int(row["label"]) for row in training], dtype=np.float64)
        for seed in TRAINING_SEEDS:
            fit_started = time.monotonic()
            prevalence = float(np.mean(train_y))
            if arm == "training_prevalence":
                weights: JsonDict = {"probability": prevalence}
                curve = [
                    {
                        "step": 0,
                        "loss": float(
                            -np.mean(
                                train_y * math.log(prevalence)
                                + (1 - train_y) * math.log1p(-prevalence)
                            )
                        ),
                    }
                ]
                update_count = 0
                architecture = [1]
                objective = "training_prevalence_constant"
            elif arm == "l2_logistic_six_input":
                fitted = fit_logistic_control(train_x, train_y, seed=seed, steps=steps)
                weights = fitted["checkpoint"]
                curve = fitted["loss_curve"]
                update_count = int(fitted["update_count"])
                architecture = [6, 1]
                objective = str(fitted["objective"])
            else:
                fitted = fit_gibbs_head(train_x, train_y, seed=seed, steps=steps)
                weights = fitted["checkpoint"]
                curve = fitted["loss_curve"]
                update_count = int(fitted["update_count"])
                architecture = [len(train_x[0]), 4, 1]
                objective = str(fitted["objective"])
            base: JsonDict = {
                "arm": arm,
                "seed": seed,
                "condition": condition,
                "weights": weights,
                "training_prevalence": prevalence,
                "fit_partition": "train",
            }
            calibration = partitions["probability_calibration"]
            logits = [_raw_logit(base, row) for row in calibration]
            labels = [int(row["label"]) for row in calibration]
            calibration_started = time.monotonic()
            affine = fit_affine_transform(logits, labels, steps=steps)
            calibration_duration = time.monotonic() - calibration_started
            base["affine"] = affine
            candidates, policy = _select_policy(base, rows)
            base.update(
                {
                    "selected_policy": policy,
                    "policy_candidates": candidates,
                    "architecture": architecture,
                    "score_direction": "higher_probability_means_machine_annotation_incorrect",
                    "objective": objective,
                    "optimizer_work": {
                        "name": "adam",
                        "maximum_steps": steps,
                        "weight_updates": update_count,
                        "affine_updates": int(affine["update_count"]),
                        "initial_loss": float(curve[0]["loss"]),
                        "final_loss": float(curve[-1]["loss"]),
                        "fit_duration_s": time.monotonic() - fit_started - calibration_duration,
                        "calibration_duration_s": calibration_duration,
                    },
                    "input_hashes": {
                        "training": hashes["train"],
                        "calibration": hashes["probability_calibration"],
                        "policy": hashes["policy_calibration"],
                    },
                }
            )
            units.append(base)
            if emit_progress:
                progress(
                    started, "small_ebm_training", "checkpoint", completed=len(units), total=planned
                )
    return units


def certified_decision(
    probability: float, policy: Mapping[str, Any], model_version: str
) -> JsonDict:
    """Apply a selected policy and turn every uncertified action into escalation."""

    result = typed_decision(
        probability,
        float(policy["accept_threshold"]),
        float(policy["reject_threshold"]),
        model_version,
    )
    if result["decision"] == "accept" and policy.get("accept_enabled") is not True:
        result["decision"] = "escalate"
        result["reason"] = "accept_action_uncertified_escalation"
    if result["decision"] == "reject" and policy.get("reject_enabled") is not True:
        result["decision"] = "escalate"
        result["reason"] = "reject_action_uncertified_escalation"
    return result


def log_loss_contribution(label: int, probability: float) -> float:
    """Return finite binary log loss after validating the probability."""

    if label not in {0, 1} or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("label and probability must be valid")
    clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def metric_row_for_test(
    *,
    arm: str,
    seed: int,
    row_key: str,
    group_id: str,
    label: int,
    raw_probability: float,
    probability: float,
    decision: str,
) -> JsonDict:
    """Build one internally consistent scored row for reducer tests."""

    return {
        "unit_id": f"full_source:{arm}:{seed}",
        "condition": "full_source",
        "arm": arm,
        "seed": seed,
        "row_key": row_key,
        "group_id": group_id,
        "partition": "final_test",
        "scored": True,
        "label": label,
        "label_authority": "machine_annotation",
        "raw_probability": raw_probability,
        "probability": probability,
        "decision": decision,
        "decision_reason": "fixture",
        "brier_contribution": (probability - label) ** 2,
        "log_loss_contribution": log_loss_contribution(label, probability),
        "measured_cost": {"current_llm_calls": 0, "cpu_scoring_duration_s": 0.0},
    }


def score_official_rows(
    units: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    emit_progress: bool = False,
) -> tuple[list[JsonDict], JsonDict]:
    """Score every official row once per unit while preserving unscored rows."""

    official = [row for row in rows if row.get("partition") == "final_test"]
    output: list[JsonDict] = []
    started = time.monotonic()
    for unit_index, unit in enumerate(units, start=1):
        unit_started = time.monotonic()
        staged: list[JsonDict] = []
        for row in official:
            raw_probability, probability = _probability(unit, row)
            action = certified_decision(
                probability,
                unit["selected_policy"],
                f"exp7413:{unit['condition']}:{unit['arm']}:{unit['seed']}",
            )
            label = row.get("label")
            scored = label in {0, 1}
            staged.append(
                {
                    "unit_id": f"{unit['condition']}:{unit['arm']}:{unit['seed']}",
                    "condition": unit["condition"],
                    "arm": unit["arm"],
                    "seed": unit["seed"],
                    "row_key": row["row_key"],
                    "group_id": row["group_id"],
                    "partition": "final_test",
                    "scored": scored,
                    "label": int(label) if scored else None,
                    "label_authority": "machine_annotation",
                    "raw_probability": raw_probability,
                    "probability": probability,
                    "decision": action["decision"],
                    "decision_reason": action["reason"],
                    "brier_contribution": (probability - int(label)) ** 2 if scored else None,
                    "log_loss_contribution": log_loss_contribution(int(label), probability)
                    if scored
                    else None,
                    "measured_cost": {"current_llm_calls": 0, "cpu_scoring_duration_s": 0.0},
                }
            )
        per_row = (time.monotonic() - unit_started) / len(staged) if staged else 0.0
        for staged_row in staged:
            staged_row["measured_cost"]["cpu_scoring_duration_s"] = per_row
        output.extend(staged)
        if emit_progress:
            progress(
                started, "official_scoring", "checkpoint", completed=unit_index, total=len(units)
            )
    scored_count = sum(row.get("label") in {0, 1} for row in official)
    return output, {
        "official_rows": len(official),
        "eligible_scored_rows": scored_count,
        "unscored_rows": len(official) - scored_count,
        "arm_seed_units": len(units),
        "paired_rows": len(output),
    }


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute pairwise AUROC with half credit for tied scores."""

    positive = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negative = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positive or not negative:
        raise ValueError("AUROC requires both labels")
    wins = sum(
        sum(value > other for other in negative) + 0.5 * sum(value == other for other in negative)
        for value in positive
    )
    return wins / (len(positive) * len(negative))


def binary_pr_auc(labels: Sequence[int], scores: Sequence[float], *, positive_label: int) -> float:
    """Compute average precision for either declared positive class."""

    binary = [int(label == positive_label) for label in labels]
    positives = sum(binary)
    if not positives or positives == len(binary):
        raise ValueError("PR-AUC requires both labels")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    found = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if binary[index]:
            found += 1
            total += found / rank
    return total / positives


def _unit_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one arm-seed ledger without treating escalation as a class."""

    scored = [row for row in rows if row.get("scored") is True]
    if not scored:
        raise ValueError("unit metrics require scored rows")
    labels = [int(row["label"]) for row in scored]
    probabilities = [float(row["probability"]) for row in scored]
    predicted = [int(probability >= 0.5) for probability in probabilities]
    f1_values = []
    for target in (0, 1):
        true_positive = sum(
            label == target and guess == target
            for label, guess in zip(labels, predicted, strict=True)
        )
        false_positive = sum(
            label != target and guess == target
            for label, guess in zip(labels, predicted, strict=True)
        )
        false_negative = sum(
            label == target and guess != target
            for label, guess in zip(labels, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    accepts = [row for row in scored if row["decision"] == "accept"]
    rejects = [row for row in scored if row["decision"] == "reject"]
    raw_binary = [int(float(row["raw_probability"]) >= 0.5) for row in scored]
    calibrated_binary = predicted
    return {
        "rows": len(scored),
        "effective_groups": len({str(row["group_id"]) for row in scored}),
        "brier": float(np.mean([float(row["brier_contribution"]) for row in scored])),
        "log_loss": float(np.mean([float(row["log_loss_contribution"]) for row in scored])),
        "auroc": binary_auroc(labels, probabilities),
        "incorrect_pr_auc": binary_pr_auc(labels, probabilities, positive_label=1),
        "correct_pr_auc": binary_pr_auc(
            labels, [1.0 - value for value in probabilities], positive_label=0
        ),
        "macro_f1": float(np.mean(f1_values)),
        "coverage": (len(accepts) + len(rejects)) / len(scored),
        "incorrect_accept_risk": sum(row["label"] == 1 for row in accepts) / len(accepts)
        if accepts
        else None,
        "correct_reject_risk": sum(row["label"] == 0 for row in rejects) / len(rejects)
        if rejects
        else None,
        "accept_count": len(accepts),
        "reject_count": len(rejects),
        "escalate_count": len(scored) - len(accepts) - len(rejects),
        "decision_changes": {
            "raw_to_calibrated_binary": sum(
                left != right for left, right in zip(raw_binary, calibrated_binary, strict=True)
            ),
            "calibrated_binary_to_typed_action": sum(
                ("reject" if value else "accept") != row["decision"]
                for value, row in zip(calibrated_binary, scored, strict=True)
            ),
        },
    }


def reduce_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report each unit and seed-averaged arm metrics from paired rows."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scored") is True:
            grouped[(str(row["arm"]), int(row["seed"]))].append(row)
    by_unit = {
        f"{arm}:{seed}": {"arm": arm, "seed": seed, **_unit_metrics(unit_rows)}
        for (arm, seed), unit_rows in grouped.items()
    }
    by_arm: JsonDict = {}
    for arm in ARMS:
        selected = [row for row in by_unit.values() if row["arm"] == arm]
        if not selected:
            continue
        numeric = (
            "brier",
            "log_loss",
            "auroc",
            "incorrect_pr_auc",
            "correct_pr_auc",
            "macro_f1",
            "coverage",
        )
        changes = {
            key: sum(int(row["decision_changes"][key]) for row in selected)
            for key in ("raw_to_calibrated_binary", "calibrated_binary_to_typed_action")
        }
        accepts = sum(int(row["accept_count"]) for row in selected)
        rejects = sum(int(row["reject_count"]) for row in selected)
        by_arm[arm] = {
            "seeds": len(selected),
            **{key: float(np.mean([float(row[key]) for row in selected])) for key in numeric},
            "incorrect_accept_risk": (
                sum(
                    (
                        float(row["incorrect_accept_risk"])
                        if row["incorrect_accept_risk"] is not None
                        else 0.0
                    )
                    * int(row["accept_count"])
                    for row in selected
                )
                / accepts
                if accepts
                else None
            ),
            "correct_reject_risk": (
                sum(
                    (
                        float(row["correct_reject_risk"])
                        if row["correct_reject_risk"] is not None
                        else 0.0
                    )
                    * int(row["reject_count"])
                    for row in selected
                )
                / rejects
                if rejects
                else None
            ),
            "accept_count": accepts,
            "reject_count": rejects,
            "escalate_count": sum(int(row["escalate_count"]) for row in selected),
            "decision_changes": changes,
        }
    return {"by_arm_seed": by_unit, "by_arm": by_arm}


def _interval(values: np.ndarray) -> JsonDict:
    """Return the observed bootstrap mean and its percentile interval."""

    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def paired_group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> JsonDict:
    """Average seeds within rows, then resample connected groups in pairs."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    cell: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scored") is True:
            cell[(str(row["arm"]), str(row["group_id"]), str(row["row_key"]))].append(row)
    groups = sorted({group for _arm, group, _key in cell})
    if not groups:
        raise ValueError("paired bootstrap requires scored groups")
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        values: dict[str, list[float]] = {"brier": [], "log_loss": [], "coverage": []}
        for group in groups:
            group_cells = [
                unit_rows
                for (cell_arm, cell_group, _key), unit_rows in cell.items()
                if cell_arm == arm and cell_group == group
            ]
            if not group_cells:
                raise ValueError("paired bootstrap arm coverage mismatch")
            values["brier"].append(
                float(
                    np.mean(
                        [
                            np.mean([float(row["brier_contribution"]) for row in unit_rows])
                            for unit_rows in group_cells
                        ]
                    )
                )
            )
            values["log_loss"].append(
                float(
                    np.mean(
                        [
                            np.mean([float(row["log_loss_contribution"]) for row in unit_rows])
                            for unit_rows in group_cells
                        ]
                    )
                )
            )
            values["coverage"].append(
                float(
                    np.mean(
                        [
                            np.mean([row["decision"] != "escalate" for row in unit_rows])
                            for unit_rows in group_cells
                        ]
                    )
                )
            )
        vectors[arm] = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    raw: dict[str, dict[str, np.ndarray]] = {}
    for baseline in BASELINE_ARMS:
        raw[baseline] = {}
        for metric in ("brier", "log_loss", "coverage"):
            delta = vectors[PRIMARY_ARM][metric] - vectors[baseline][metric]
            raw[baseline][metric] = np.mean(delta[indices], axis=1)
    centered = np.column_stack(
        [raw[baseline]["brier"] - np.mean(raw[baseline]["brier"]) for baseline in BASELINE_ARMS]
    )
    upper_critical = float(np.quantile(np.max(centered, axis=1), 0.95))
    lower_critical = float(np.quantile(np.min(centered, axis=1), 0.05))
    contrasts: JsonDict = {}
    for baseline in BASELINE_ARMS:
        brier_values = raw[baseline]["brier"]
        observed = float(np.mean(vectors[PRIMARY_ARM]["brier"] - vectors[baseline]["brier"]))
        contrasts[baseline] = {
            "brier_delta": {
                "mean": observed,
                "simultaneous_ci95": [observed + lower_critical, observed + upper_critical],
                "marginal_ci95": [
                    float(np.quantile(brier_values, 0.025)),
                    float(np.quantile(brier_values, 0.975)),
                ],
            },
            "log_loss_delta": _interval(raw[baseline]["log_loss"]),
            "coverage_delta": _interval(raw[baseline]["coverage"]),
        }
    return {
        "draws": draws,
        "seed": seed,
        "effective_groups": len(groups),
        "seed_average_before_group_resampling": True,
        "simultaneous_method": "max_centered_statistic_three_primary_brier_contrasts",
        "upper_critical_value": upper_critical,
        "contrasts": contrasts,
    }


def source_removed_row(row: Mapping[str, Any]) -> JsonDict:
    """Apply the frozen no-source vector while retaining response falsifiability."""

    changed = deepcopy(dict(row))
    response = changed.get("response_only_ablation")
    if not isinstance(response, Mapping) or "falsifiability_score" not in response:
        raise ValueError("response feature keys do not match the frozen protocol")
    changed["source_features"] = {
        "numeric_novelty_with_context": 0.0,
        "falsifiability_score": float(response["falsifiability_score"]),
        "normalized_number_token_overlap": 0.0,
        "normalized_content_token_overlap": 0.0,
        "max_answer_source_sentence_overlap": 0.0,
        "missing_or_empty_source": 1.0,
    }
    return changed


def cross_group_source_permutation(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Rotate representative contexts across groups within each partition."""

    output: list[JsonDict] = []
    for partition in sorted({str(row.get("partition")) for row in rows}):
        selected = [row for row in rows if row.get("partition") == partition]
        groups = sorted({str(row.get("group_id")) for row in selected})
        if len(groups) < 2:
            raise ValueError("cross-group permutation requires at least two groups per partition")
        representative = {
            group: min(
                (row for row in selected if str(row.get("group_id")) == group),
                key=lambda row: str(row.get("row_key")),
            )
            for group in groups
        }
        donor = {group: groups[(index + 1) % len(groups)] for index, group in enumerate(groups)}
        for row in selected:
            changed = deepcopy(dict(row))
            changed["context"] = representative[donor[str(row["group_id"])]].get("context", "")
            features = extract_feature_row(changed)
            output.append(
                {**features, "label": row.get("label"), "label_authority": "machine_annotation"}
            )
    return sorted(output, key=lambda row: str(row["row_key"]))


def reduce_calibration_value(intervals: Mapping[str, Any], metrics: Mapping[str, Any]) -> JsonDict:
    """Apply the registered conjunction without promoting ranking-only evidence."""

    contrasts = intervals.get("contrasts") or {}
    by_arm = metrics.get("by_arm") or {}
    primary = by_arm.get(PRIMARY_ARM) or {}
    brier = all(
        float(
            (contrasts.get(baseline) or {})
            .get("brier_delta", {})
            .get("simultaneous_ci95", [math.inf])[-1]
        )
        < 0
        for baseline in BASELINE_ARMS
    )
    log_loss = all(
        float(primary.get("log_loss", math.inf))
        <= float((by_arm.get(baseline) or {}).get("log_loss", -math.inf))
        for baseline in BASELINE_ARMS
    )
    coverage = float(primary.get("coverage", 0.0)) >= 0.25
    logistic_delta = (
        (contrasts.get("l2_logistic_six_input") or {})
        .get("coverage_delta", {})
        .get("ci95", [-math.inf])
    )
    no_coverage_loss = bool(logistic_delta) and float(logistic_delta[0]) >= 0.0
    checks = {
        "three_simultaneous_brier_upper_bounds_below_zero": brier,
        "non_worse_mean_log_loss": log_loss,
        "certified_coverage_at_least_0_25": coverage,
        "no_coverage_loss_vs_logistic": no_coverage_loss,
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "calibration_value_score": int(passed),
        "ranking_without_certified_coverage_is_diagnostic": brier and not coverage,
    }


def _field_principles() -> dict[str, str]:
    """Explain required ordinary fields without wrapping their values."""

    defaults = {
        key: "Keep this ordinary field machine-readable and replayable."
        for key in REQUIRED_ARTIFACT_FIELDS
    }
    defaults.update(
        {
            "schema": "Version ordinary top-level identity and terminal status fields.",
            "run_date": "Use 20260919 with actual UTC start and end timestamps.",
            "preconditions_checked": "Record exact paths, hashes, and resource checks before dependent work.",
            "MODEL_SPECS": "Remain empty because no current LLM is used.",
            "model_invoked": "Remain false because no model load or generation is attempted.",
            "invocation_counts": "Reduce current owned events; all current LLM counts are zero.",
            "inference_substrate": "Use a truthful string and keep CPU details separate.",
            "inference_substrate_class": "Declare no_model_load without runtime padding.",
            "execution_venue": "Use the closed host value.",
            "duration_s": "Measure monotonic current work apart from historical evidence.",
            "phase_spans": "Retain actual phase boundaries, checkpoints, and completed units.",
            "random_seed": "Freeze five fit seeds and bootstrap seed 6501307.",
            "reproducibility_checksum": "Bind code, protocol, exact inputs, checkpoints, and raw rows.",
            "source_artifact_hashes": "Hash exact paths and preserve upstream adversarial flags.",
            "rows": "Account for every main and ablation arm-seed unit.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted work.",
            "acceptance_gate_results": "Keep validity, completion, safety, validation, and benefit separate.",
            "gate_check_summary": "Name exact blocked and failed fields without hiding missing values.",
            "verifier_is_oracle": "Machine annotation agreement is not general semantic correctness.",
            "honest_verdict": "Start completed findings with complete_ and external absence with blocked_.",
            "verdict_class": "Use only the closed terminal enum; completed no-benefit work is null.",
            "flagged_adversarial": "Preserve critical findings and deny readiness when flagged.",
            "validation_receipts": "Retain argv, environment, exits, durations, and hashed logs.",
            "field_principles": "Explain fields separately while gate scalars remain ordinary numbers.",
            "promotion_score": "Always remain zero; this experiment cannot roll out or publish.",
            "calibration_capture_complete_score": "One means complete valid measurement, including a null.",
            "calibration_value_score": "One requires every registered scientific benefit gate.",
            "checkpoint_manifest": "Bind each numeric arm-seed checkpoint, fit scope, work, and byte size.",
            "paired_metric_rows": "Retain each official row, arm, seed, label, action, and cost.",
            "source_ablation_rows": "Retain fixed removal and permutation results without test tuning.",
            "label_authority": "Declare machine annotations without an exact-correctness claim.",
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
    """Record one explicit gate and its scientific or validity role."""

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
    """Reduce required checks while keeping scientific failures valid nulls."""

    failures = [
        deepcopy(dict(row))
        for row in gates
        if row.get("category") != "scientific_benefit" and row.get("passed") is not True
    ]
    return {
        "required_checks_passed": not failures and blocked is None,
        "failed_required_checks": failures,
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
    """Hash stable evidence while excluding its own checksum slot and wall time."""

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
    ):
        payload.pop(field, None)
    return canonical_hash(payload)


def _checkpoint_payload(unit: Mapping[str, Any]) -> JsonDict:
    """Strip one training state to plain numeric scoring and work evidence."""

    return {
        "schema": "carnot.exp7413.numeric_checkpoint.v1",
        "condition": unit["condition"],
        "arm": unit["arm"],
        "seed": unit["seed"],
        "architecture": unit["architecture"],
        "score_direction": unit["score_direction"],
        "weights": deepcopy(unit["weights"]),
        "affine": {"slope": unit["affine"]["slope"], "intercept": unit["affine"]["intercept"]},
        "selected_policy": deepcopy(unit["selected_policy"]),
        "optimizer_work": deepcopy(unit["optimizer_work"]),
        "input_hashes": deepcopy(unit["input_hashes"]),
    }


def write_checkpoints(root: Path, units: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Write every numeric checkpoint atomically and retain its exact byte identity."""

    checkpoint_dir = root / RAW_DIR / "checkpoints"
    rows = []
    for unit in units:
        path = checkpoint_dir / f"{unit['condition']}--{unit['arm']}--{unit['seed']}.json"
        payload = _checkpoint_payload(unit)
        atomic_json(path, payload)
        rows.append(
            {
                "condition": unit["condition"],
                "arm": unit["arm"],
                "seed": unit["seed"],
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
                "architecture": unit["architecture"],
                "score_direction": unit["score_direction"],
                "optimizer_work": deepcopy(unit["optimizer_work"]),
                "input_hashes": deepcopy(unit["input_hashes"]),
            }
        )
    return rows


def _unit_rows(
    units: Sequence[Mapping[str, Any]], checkpoint_manifest: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Summarize completed registered units without duplicating numeric weights."""

    checkpoint = {(row["condition"], row["arm"], row["seed"]): row for row in checkpoint_manifest}
    return [
        {
            "comparative_unit": f"{unit['condition']}:{unit['arm']}:{unit['seed']}",
            "condition": unit["condition"],
            "arm": unit["arm"],
            "seed": unit["seed"],
            "status": "completed",
            "checkpoint_sha256": checkpoint[(unit["condition"], unit["arm"], unit["seed"])][
                "sha256"
            ],
            "fit_partition": unit["fit_partition"],
            "calibration_partition": unit["affine"]["fitting_partition"],
            "policy_partition": unit["selected_policy"]["selection_partition"],
        }
        for unit in units
    ]


def _source_dependency(
    full_metrics: Mapping[str, Any],
    ablation_rows: Sequence[Mapping[str, Any]],
    value: Mapping[str, Any],
) -> JsonDict:
    """Report whether an observed full-condition benefit disappears without source."""

    full_brier = float(full_metrics["by_arm"][PRIMARY_ARM]["brier"])
    by_condition = {
        condition: reduce_metrics(
            [row for row in ablation_rows if row.get("condition") == condition]
        )["by_arm"][PRIMARY_ARM]
        for condition in CONDITIONS[1:]
    }
    deltas = {
        condition: float(row["brier"]) - full_brier for condition, row in by_condition.items()
    }
    return {
        "full_source_brier": full_brier,
        "ablation_brier_delta_vs_full": deltas,
        "full_registered_benefit_passed": value.get("passed") is True,
        "benefit_requires_source": value.get("passed") is True
        and all(delta > 0 for delta in deltas.values()),
        "scope": "machine_annotation_agreement_not_semantic_truth",
    }


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for every affected and terminal command."""

    names = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)).issubset(names)


def build_artifact(
    *,
    main_units: Sequence[Mapping[str, Any]],
    ablation_units: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    scoring_receipt: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_ns: int,
    flagged_adversarial: bool,
    fixture: bool = False,
) -> JsonDict:
    """Build one complete result while keeping capture independent of benefit."""

    metrics = reduce_metrics(paired_rows)
    intervals = paired_group_bootstrap(
        paired_rows, draws=200 if fixture else BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    value = reduce_calibration_value(intervals, metrics)
    validation_passed = _validation_passed(validation_receipts) if not fixture else True
    expected_main = scoring_receipt["official_rows"] * len(ARMS) * len(TRAINING_SEEDS)
    expected_ablation = scoring_receipt["official_rows"] * 2 * len(TRAINING_SEEDS)
    paired_complete = len(paired_rows) == expected_main
    ablation_complete = len(ablation_rows) == expected_ablation
    units = [*main_units, *ablation_units]
    expected_units = len(ARMS) * len(TRAINING_SEEDS) + 2 * len(TRAINING_SEEDS)
    gates = [
        _gate(
            "preconditions",
            "validity",
            "all",
            True,
            all(row.get("passed") is True for row in preconditions),
            all(row.get("passed") is True for row in preconditions),
            "Every exact input must authenticate before fitting.",
        ),
        _gate(
            "registered_units",
            "completion",
            "==",
            expected_units,
            len(units),
            len(units) == expected_units,
            "All four main arms and two matched source ablations must finish five seeds.",
        ),
        _gate(
            "paired_official_rows",
            "completion",
            "==",
            expected_main,
            len(paired_rows),
            paired_complete,
            "Every official row must remain paired across all main units.",
        ),
        _gate(
            "ablation_official_rows",
            "completion",
            "==",
            expected_ablation,
            len(ablation_rows),
            ablation_complete,
            "Both fixed source ablations must retain every official row.",
        ),
        _gate(
            "affected_and_terminal_validation",
            "validation",
            "==",
            True,
            validation_passed,
            validation_passed,
            "Every frozen affected and terminal command must pass.",
        ),
        _gate(
            "registered_scientific_value",
            "scientific_benefit",
            "==",
            True,
            value["passed"],
            value["passed"],
            "Benefit requires all Brier, log-loss, risk, and coverage checks.",
        ),
    ]
    summary = _gate_summary(gates)
    capture = int(summary["required_checks_passed"] and not flagged_adversarial)
    value_score = int(capture and value["passed"])
    invocation = build_current_work_receipt(
        run_id="exp7413-current",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "work": "host CPU NumPy small-head fitting, affine calibration, bootstrap, hashing, and JSON serialization",
            "neural_text_model_loaded": False,
            "optimizer": "small fixed Gibbs and logistic heads only",
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=duration_ns,
        phase_spans=phase_spans,
        small_ebm_training={
            "performed": True,
            "scope": "four_registered_arms_plus_two_source_ablations",
            "arm_seed_units": len(units),
            "fit_updates": sum(int(unit["optimizer_work"]["weight_updates"]) for unit in units),
            "affine_updates": sum(int(unit["optimizer_work"]["affine_updates"]) for unit in units),
            "receipts": [
                {
                    "condition": unit["condition"],
                    "arm": unit["arm"],
                    "seed": unit["seed"],
                    "optimizer_work": deepcopy(unit["optimizer_work"]),
                }
                for unit in units
            ],
        },
    )
    unit_rows = _unit_rows(units, checkpoint_manifest)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete" if capture else "disqualified",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **invocation,
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": unit_rows,
        "sample_size_budget": {
            "planned": expected_units,
            "attempted": len(units),
            "completed": len(units),
            "failed": 0,
            "censored": 0,
            "unstarted": expected_units - len(units),
            "independent_groups": intervals["effective_groups"],
            "official_rows": scoring_receipt["official_rows"],
            "eligible_scored_rows": scoring_receipt["eligible_scored_rows"],
            "unscored_rows_preserved_per_unit": scoring_receipt["unscored_rows"],
            "planned_paired_rows": expected_main,
            "planned_ablation_rows": expected_ablation,
            "stop_rule": "complete every registered arm, seed, official row, and fixed ablation without test-label tuning",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "honest_verdict": (
            "complete_positive_source_calibration_machine_annotation_scope"
            if value_score
            else "complete_null_source_calibration_no_registered_decision_benefit"
        )
        if capture
        else "complete_disqualified_source_calibration_validation",
        "verdict_class": "positive" if value_score else "null" if capture else "disqualified",
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "calibration_capture_complete_score": capture,
        "calibration_value_score": value_score,
        "checkpoint_manifest": [deepcopy(dict(row)) for row in checkpoint_manifest],
        "paired_metric_rows": [deepcopy(dict(row)) for row in paired_rows],
        "source_ablation_rows": [deepcopy(dict(row)) for row in ablation_rows],
        "label_authority": "machine_annotation",
        "calibration_metrics": metrics,
        "paired_group_bootstrap": intervals,
        "calibration_value_reduction": value,
        "source_dependency_report": _source_dependency(metrics, ablation_rows, value),
        "complete_service_cost": {
            "feature_extraction_duration_s": sum(
                float(span["duration_s"])
                for span in phase_spans
                if span["phase"] in {"load_features", "build_ablations"}
            ),
            "weight_fitting_duration_s": sum(
                float(unit["optimizer_work"]["fit_duration_s"]) for unit in units
            ),
            "affine_calibration_duration_s": sum(
                float(unit["optimizer_work"]["calibration_duration_s"]) for unit in units
            ),
            "official_scoring_duration_s": sum(
                float(row["measured_cost"]["cpu_scoring_duration_s"])
                for row in (*paired_rows, *ablation_rows)
            ),
            "current_llm_calls": 0,
        },
        "fixture_artifact": fixture,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_fixture_artifact() -> JsonDict:
    """Build a complete null fixture for cold mutation tests."""

    rows = []
    for partition in SCORED_PARTITIONS:
        for index in range(4):
            label = index % 2
            value = 0.8 if label else 0.2
            rows.append(
                {
                    "row_key": f"{partition}-{index}",
                    "group_id": f"{partition}-g{index}",
                    "partition": partition,
                    "source_features": dict(
                        zip(
                            SOURCE_FEATURE_NAMES,
                            [value, value, 1 - value, 1 - value, 1 - value, 0.0],
                            strict=True,
                        )
                    ),
                    "response_only_ablation": {
                        "entity_uptake": value,
                        "falsifiability_score": value,
                    },
                    "label": label,
                    "label_authority": "machine_annotation",
                }
            )
    main_units = fit_registered_units(rows, steps=1)
    removed = [source_removed_row(row) for row in rows]
    removed_units = fit_registered_units(
        removed, arms=(PRIMARY_ARM,), condition="source_removed", steps=1
    )
    permuted = deepcopy(rows)
    for row in permuted:
        row["source_features"] = deepcopy(removed[0]["source_features"])
    permuted_units = fit_registered_units(
        permuted, arms=(PRIMARY_ARM,), condition="cross_group_source_permutation", steps=1
    )
    checkpoint_rows = []
    for unit in (*main_units, *removed_units, *permuted_units):
        payload = _checkpoint_payload(unit)
        checkpoint_rows.append(
            {
                "condition": unit["condition"],
                "arm": unit["arm"],
                "seed": unit["seed"],
                "path": f"fixture/{unit['condition']}--{unit['arm']}--{unit['seed']}.json",
                "sha256": canonical_hash(payload),
                "byte_size": len(json.dumps(payload, sort_keys=True).encode()),
                "architecture": unit["architecture"],
                "score_direction": unit["score_direction"],
                "optimizer_work": deepcopy(unit["optimizer_work"]),
                "input_hashes": deepcopy(unit["input_hashes"]),
            }
        )
    paired, receipt = score_official_rows(main_units, rows)
    removed_rows, _ = score_official_rows(removed_units, removed)
    permuted_rows, _ = score_official_rows(permuted_units, permuted)
    return build_artifact(
        main_units=main_units,
        ablation_units=[*removed_units, *permuted_units],
        checkpoint_manifest=checkpoint_rows,
        paired_rows=paired,
        ablation_rows=[*removed_rows, *permuted_rows],
        scoring_receipt=receipt,
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes={},
        validation_receipts=[],
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00+00:00",
        duration_ns=0,
        flagged_adversarial=False,
        fixture=True,
    )


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish an unchanged external prerequisite failure as a complete block."""

    invocation = build_current_work_receipt(
        run_id="exp7413-blocked",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "work": "precondition checks only",
            "neural_text_model_loaded": False,
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        small_ebm_training={"performed": False, "receipts": []},
    )
    blocked = {
        key: failed.get(key)
        for key in ("upstream", "path", "check", "field", "expected", "observed")
    }
    gates = [
        _gate(
            str(failed.get("check")),
            "precondition",
            str(failed.get("operator") or "=="),
            failed.get("expected"),
            failed.get("observed"),
            False,
            "Dependent fitting cannot start from an ineligible upstream.",
        )
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        "preconditions_checked": [{**deepcopy(dict(failed)), "passed": False}],
        **invocation,
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": 30,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 30,
            "independent_groups": 0,
            "official_rows": 0,
            "eligible_scored_rows": 0,
            "unscored_rows_preserved_per_unit": 0,
            "planned_paired_rows": 0,
            "planned_ablation_rows": 0,
            "stop_rule": "blocked before dependent work",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates, blocked),
        "verifier_is_oracle": False,
        "honest_verdict": f"blocked_{failed.get('check')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "calibration_capture_complete_score": 0,
        "calibration_value_score": 0,
        "checkpoint_manifest": [],
        "paired_metric_rows": [],
        "source_ablation_rows": [],
        "label_authority": "machine_annotation",
        "calibration_metrics": {"by_arm_seed": {}, "by_arm": {}},
        "paired_group_bootstrap": {"contrasts": {}},
        "calibration_value_reduction": {"passed": False, "calibration_value_score": 0},
        "source_dependency_report": {},
        "complete_service_cost": {"current_llm_calls": 0},
        "fixture_artifact": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute completeness and scientific scores from raw artifact rows."""

    if value.get("verdict_class") == "blocked":
        return {"capture": 0, "value": 0, "paired_complete": True, "ablation_complete": True}
    budget = value.get("sample_size_budget") or {}
    paired = value.get("paired_metric_rows") or []
    ablation = value.get("source_ablation_rows") or []
    paired_complete = len(paired) == int(budget.get("planned_paired_rows") or -1)
    ablation_complete = len(ablation) == int(budget.get("planned_ablation_rows") or -1)
    metrics = reduce_metrics(paired) if paired else {"by_arm": {}}
    intervals = (
        paired_group_bootstrap(
            paired,
            draws=200 if value.get("fixture_artifact") else BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        )
        if paired
        else {"contrasts": {}}
    )
    reduction = reduce_calibration_value(intervals, metrics)
    required = all(
        row.get("passed") is True
        for row in value.get("acceptance_gate_results") or []
        if row.get("category") != "scientific_benefit"
    )
    capture = int(
        required
        and paired_complete
        and ablation_complete
        and value.get("flagged_adversarial") is False
    )
    return {
        "capture": capture,
        "value": int(capture and reduction["passed"]),
        "paired_complete": paired_complete,
        "ablation_complete": ablation_complete,
    }


def _row_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Recompute scored contributions and reject duplicate paired identities."""

    errors: list[str] = []
    identities: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        identity = (
            row.get("condition"),
            row.get("arm"),
            row.get("seed"),
            row.get("row_key"),
        )
        if identity in identities:
            errors.append(f"duplicate_metric_row:{index}")
        identities.add(identity)
        probability = row.get("probability")
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(float(probability))
            or not 0 <= float(probability) <= 1
        ):
            errors.append(f"probability_invalid:{index}")
            continue
        if row.get("scored") is True:
            label = row.get("label")
            if label not in {0, 1}:
                errors.append(f"scored_label_invalid:{index}")
                continue
            expected_brier = (float(probability) - int(label)) ** 2
            expected_log = log_loss_contribution(int(label), float(probability))
            if not math.isclose(
                float(row.get("brier_contribution", math.inf)),
                expected_brier,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append(f"brier_contribution_mismatch:{index}")
            if not math.isclose(
                float(row.get("log_loss_contribution", math.inf)),
                expected_log,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append(f"log_loss_contribution_mismatch:{index}")
        elif (
            row.get("label") is not None
            or row.get("brier_contribution") is not None
            or row.get("log_loss_contribution") is not None
        ):
            errors.append(f"unscored_row_invented_metric:{index}")
    return list(dict.fromkeys(errors))


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, raw reduction, counters, hashes, and declarations."""

    errors = [f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
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
        "label_authority": "machine_annotation",
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
    if set(value.get("field_principles") or {}) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if value.get("verdict_class") == "blocked":
        if (
            value.get("calibration_capture_complete_score") != 0
            or value.get("calibration_value_score") != 0
            or not str(value.get("honest_verdict") or "").startswith("blocked_")
        ):
            errors.append("blocked_disposition_invalid")
    else:
        reduced = independent_reduce(value)
        if not reduced["paired_complete"]:
            errors.append("paired_row_budget_mismatch")
        if not reduced["ablation_complete"]:
            errors.append("ablation_row_budget_mismatch")
        if value.get("calibration_capture_complete_score") != reduced["capture"]:
            errors.append("calibration_capture_complete_score_mismatch")
        if value.get("calibration_value_score") != reduced["value"]:
            errors.append("calibration_value_score_mismatch")
        paired = value.get("paired_metric_rows") or []
        ablation = value.get("source_ablation_rows") or []
        errors.extend(_row_integrity_errors([*paired, *ablation]))
        if paired:
            recomputed_metrics = reduce_metrics(paired)
            recomputed_intervals = paired_group_bootstrap(
                paired,
                draws=200 if value.get("fixture_artifact") else BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
            )
            recomputed_value = reduce_calibration_value(recomputed_intervals, recomputed_metrics)
            if canonical_hash(recomputed_metrics) != canonical_hash(
                value.get("calibration_metrics")
            ):
                errors.append("calibration_metrics_mismatch")
            if canonical_hash(recomputed_intervals) != canonical_hash(
                value.get("paired_group_bootstrap")
            ):
                errors.append("paired_group_bootstrap_mismatch")
            if canonical_hash(recomputed_value) != canonical_hash(
                value.get("calibration_value_reduction")
            ):
                errors.append("calibration_value_reduction_mismatch")
        if not value.get("fixture_artifact"):
            for row in value.get("checkpoint_manifest") or []:
                path = REPO_ROOT / str(row.get("path") or "")
                if (
                    not path.is_file()
                    or sha256_file(path) != row.get("sha256")
                    or path.stat().st_size != row.get("byte_size")
                ):
                    errors.append(f"checkpoint_bytes_mismatch:{row.get('path')}")
            for row in (value.get("source_artifact_hashes") or {}).values():
                if not isinstance(row, Mapping):
                    errors.append("source_artifact_hash_row_invalid")
                    continue
                path = Path(str(row.get("path") or ""))
                resolved = path if path.is_absolute() else REPO_ROOT / path
                if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                    errors.append(f"source_artifact_hash_mismatch:{row.get('path')}")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(value, root=REPO_ROOT))
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> list[str]:
    """Reload one terminal candidate and independently reduce its raw rows."""

    value = _load_object(path)
    if not value:
        return ["artifact_unreadable_or_not_object"]
    return validate_artifact(value)


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    """Close one measured phase with a completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _load_real_rows(
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover - entrypoint E2E.
    """Join frozen features with evaluator labels and return raw predictors too."""

    features = _load_object(root / FEATURE_PATH).get("records") or []
    feature_by_key = {str(row["row_key"]): deepcopy(dict(row)) for row in features}
    reloaded = reload_corpus(root / CORPUS_DIR)
    readers = CorpusReaders(reloaded)
    predictors: list[JsonDict] = []
    joined: list[JsonDict] = []
    for partition in SCORED_PARTITIONS:
        partition_predictors = readers.read_predictors(partition)
        labels = readers.read_labels(
            partition, token=FINAL_TEST_TOKEN if partition == "final_test" else None
        )
        label_by_key = {str(row["row_key"]): row.get("label") for row in labels}
        for predictor in partition_predictors:
            key = str(predictor["row_key"])
            base = feature_by_key[key]
            base.update({"label": label_by_key[key], "label_authority": "machine_annotation"})
            joined.append(base)
            predictors.append({**deepcopy(dict(predictor)), "label": label_by_key[key]})
    return joined, predictors


def _terminal_commands(
    candidate: Path,
) -> list[PlannedCommand]:  # pragma: no cover - entrypoint E2E.
    """Build fresh replay, independent reduction, and unchanged strict readers."""

    python = ".venv/bin/python"
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--cold-replay",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_metric_recompute",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--independent-reduce",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
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
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared entrypoint E2E.
    """Authenticate, fit, score, validate, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
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

    phase_started = time.monotonic()
    progress(run_started, "load_features", "before_benchmark")
    rows, predictors = _load_real_rows(root)
    spans.append(_span("load_features", phase_started, run_started, len(rows)))
    progress(run_started, "load_features", "after_benchmark", completed=len(rows))

    phase_started = time.monotonic()
    progress(run_started, "small_ebm_training", "before_training", planned=20)
    main_units = fit_registered_units(rows, emit_progress=True)
    spans.append(_span("main_training", phase_started, run_started, len(main_units)))
    progress(run_started, "small_ebm_training", "after_training", completed=len(main_units))

    phase_started = time.monotonic()
    progress(run_started, "build_ablations", "before_benchmark")
    removed_rows = [source_removed_row(row) for row in rows]
    permuted_rows = cross_group_source_permutation(predictors)
    spans.append(
        _span("build_ablations", phase_started, run_started, len(removed_rows) + len(permuted_rows))
    )
    progress(
        run_started,
        "build_ablations",
        "after_benchmark",
        completed=len(removed_rows) + len(permuted_rows),
    )

    phase_started = time.monotonic()
    progress(run_started, "ablation_training", "before_training", planned=10)
    removed_units = fit_registered_units(
        removed_rows, arms=(PRIMARY_ARM,), condition="source_removed", emit_progress=True
    )
    permuted_units = fit_registered_units(
        permuted_rows,
        arms=(PRIMARY_ARM,),
        condition="cross_group_source_permutation",
        emit_progress=True,
    )
    ablation_units = [*removed_units, *permuted_units]
    spans.append(_span("ablation_training", phase_started, run_started, len(ablation_units)))
    progress(run_started, "ablation_training", "after_training", completed=len(ablation_units))

    phase_started = time.monotonic()
    progress(run_started, "checkpoint_write", "before_serialization")
    checkpoint_manifest = write_checkpoints(root, [*main_units, *ablation_units])
    spans.append(_span("checkpoint_write", phase_started, run_started, len(checkpoint_manifest)))
    progress(
        run_started, "checkpoint_write", "after_serialization", completed=len(checkpoint_manifest)
    )

    phase_started = time.monotonic()
    progress(run_started, "official_scoring", "before_benchmark", planned=30)
    paired_rows, scoring_receipt = score_official_rows(main_units, rows, emit_progress=True)
    removed_scored, _ = score_official_rows(removed_units, removed_rows, emit_progress=True)
    permuted_scored, _ = score_official_rows(permuted_units, permuted_rows, emit_progress=True)
    ablation_scored = [*removed_scored, *permuted_scored]
    spans.append(
        _span(
            "official_scoring", phase_started, run_started, len(paired_rows) + len(ablation_scored)
        )
    )
    progress(
        run_started,
        "official_scoring",
        "after_benchmark",
        completed=len(paired_rows) + len(ablation_scored),
    )

    phase_started = time.monotonic()
    progress(run_started, "bootstrap", "before_benchmark", draws=BOOTSTRAP_DRAWS)
    reduce_metrics(paired_rows)
    paired_group_bootstrap(paired_rows)
    spans.append(_span("bootstrap", phase_started, run_started, BOOTSTRAP_DRAWS))
    progress(run_started, "bootstrap", "after_benchmark", completed=BOOTSTRAP_DRAWS)

    raw_dir = root / RAW_DIR
    private_root = Path(tempfile.mkdtemp(prefix="exp7413-validation-", dir="/tmp"))
    commands = build_command_plan(root, V650_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, V650_MANIFEST, commands)
    progress(
        run_started,
        "validation",
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
    reduction = reduce_affected_receipts(root, V650_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "validation",
        "after_subprocesses",
        completed=len(affected),
        passed=reduction["passed"],
    )

    for row in checkpoint_manifest:
        source_hashes[row["path"]] = {
            "path": row["path"],
            "sha256": row["sha256"],
            "original_flagged_adversarial": None,
        }
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        path = root / relative
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "original_flagged_adversarial": None,
        }
    duration_ns = int((time.monotonic() - run_started) * 1_000_000_000)
    candidate = build_artifact(
        main_units=main_units,
        ablation_units=ablation_units,
        checkpoint_manifest=checkpoint_manifest,
        paired_rows=paired_rows,
        ablation_rows=ablation_scored,
        scoring_receipt=scoring_receipt,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=affected,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_ns=duration_ns,
        flagged_adversarial=not reduction["passed"],
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
    final_duration_ns = int((time.monotonic() - run_started) * 1_000_000_000)
    final = build_artifact(
        main_units=main_units,
        ablation_units=ablation_units,
        checkpoint_manifest=checkpoint_manifest,
        paired_rows=paired_rows,
        ablation_rows=ablation_scored,
        scoring_receipt=scoring_receipt,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_ns=final_duration_ns,
        flagged_adversarial=not reduction["passed"] or not terminal_passed or critical,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed run date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one strict fresh-process artifact reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = validate_artifact(value) if value else ["artifact_unreadable_or_not_object"]
        reduced = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduced}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

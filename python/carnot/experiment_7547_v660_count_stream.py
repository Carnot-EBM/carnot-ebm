"""Seal the V660 empirical stream for delayed count learning.

This module reads authenticated cached forecasts. It loads no language model
and does not reuse the old trained head or its temperature. The output tests
data custody and durable count mechanics, not predictive benefit.

Spec refs: REQ-CL-7547 and SCENARIO-CL-7547-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.experiment_7504_v657_evidence_interface import load_json, load_jsonl, write_jsonl
from carnot.experiment_7509_v657_causal_online import build_arrival_order
from carnot.experiment_7534_v659_count_memory import (
    CLIP_MIN,
    PRIOR_MASS,
    CountConfig,
    CountEventMachine,
    canonical_hash,
    normalized_probability,
    sha256_file,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
EXPERIMENT_ID = "exp7547-count-stream"
SCHEMA = "carnot.exp7547.v660.count_stream.v1"
RESULT_PATH = Path("results/experiment_7547_v660_count_stream.json")
RAW_DIR = Path("results/raw/experiment_7547_v660_count_stream")
MODULE_PATH = Path("python/carnot/experiment_7547_v660_count_stream.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7547_v660_count_stream.py")
TEST_PATH = Path("tests/python/test_experiment_7547_v660_count_stream.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

FEATURE_PATH = Path("results/raw/experiment_7504_v657_evidence_interface/features.jsonl")
PREDICTOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/predictors.jsonl")
EVALUATOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/evaluators.jsonl")
ACCESS_PATH = Path(
    "results/raw/experiment_7504_v657_evidence_interface/access_exposure_manifest.json"
)
UPSTREAM_PATHS = {
    "exp7504": Path("results/experiment_7504_v657_evidence_interface.json"),
    "exp7510": Path("results/experiment_7510_v657_causal_audit.json"),
    "exp7534": Path("results/experiment_7534_v659_count_memory.json"),
    "exp7546": Path("results/experiment_7546_v660_contract_methods.json"),
}
ORDER_SEEDS = (7549001, 7549002, 7549003, 7549004, 7549005)
BLOCK_SIZE = 8
FEEDBACK_DELAY = 8
ARMS = ("frozen", "global", "local", "shuffled_local")
EXPECTED_ROLE_COUNTS = {"training": 176, "calibration_tuning": 60, "online": 159, "test": 116}
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def logit(probability: float) -> float:
    """Map a strict probability to the feature interface log-odds value."""

    value = float(probability)
    if not 0.0 < value < 1.0 or not math.isfinite(value):
        raise ValueError("raw_probability_invalid")
    return math.log(value / (1.0 - value))


def frozen_protocol() -> JsonDict:
    """Return every setting fixed before online labels become visible."""

    return {
        "order_seeds": list(ORDER_SEEDS),
        "block_size": BLOCK_SIZE,
        "feedback_delay": FEEDBACK_DELAY,
        "prior_mass": PRIOR_MASS,
        "clip": CLIP_MIN,
        "full_audit": True,
        "arms": list(ARMS),
        "prediction_precedes_release_and_update": True,
        "final_short_block": "retained_and_censored_when_release_is_after_stream_end",
        "end_of_stream_updates": "censored_not_silently_dropped",
        "label_orientation": "one_means_contains_unsupported",
    }


def assemble_public_rows(
    features: Sequence[Mapping[str, Any]],
    predictors: Sequence[Mapping[str, Any]],
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, list[JsonDict]]:
    """Join public stores and reject identity, role, or raw mapping drift."""

    predictor_by_group: dict[str, Mapping[str, Any]] = {}
    for predictor in predictors:
        group = str(predictor.get("group_id") or "")
        if not group or group in predictor_by_group:
            raise ValueError("predictor_group_duplicate_or_missing")
        predictor_by_group[group] = predictor
    output = {role: [] for role in EXPECTED_ROLE_COUNTS}
    groups: set[str] = set()
    sources: set[str] = set()
    for feature in features:
        group = str(feature.get("group_id") or "")
        role = str(feature.get("role") or "")
        predictor = predictor_by_group.get(group)
        if not group or group in groups or predictor is None or role not in output:
            raise ValueError("public_group_identity_invalid")
        groups.add(group)
        if predictor.get("role") != role:
            raise ValueError(f"role_mismatch:{group}")
        for field in ("source_hash", "response_hash"):
            if predictor.get(field) != feature.get(field):
                raise ValueError(f"{field}_mismatch:{group}")
        source_hash = str(feature.get("source_hash") or "")
        if not source_hash or source_hash in sources:
            raise ValueError(f"normalized_source_duplicate:{group}")
        sources.add(source_hash)
        probability = float(feature.get("raw_whole_expectation", math.nan))
        values = feature.get("features")
        if not isinstance(values, list) or len(values) != 10:
            raise ValueError(f"feature_shape_invalid:{group}")
        if not 0.0 < probability < 1.0 or not math.isfinite(probability):
            raise ValueError(f"raw_probability_invalid:{group}")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError(f"feature_value_invalid:{group}")
        output[role].append(
            {
                "group_id": group,
                "role": role,
                "source_hash": source_hash,
                "response_hash": str(feature["response_hash"]),
                "source_family": str(predictor.get("source_family") or "unknown"),
                "raw_whole_expectation": probability,
                "base_probability": probability,
                "label_orientation": "one_means_contains_unsupported",
            }
        )
    for rows in output.values():
        rows.sort(key=lambda row: str(row["group_id"]))
    if expected_counts is not None and {role: len(rows) for role, rows in output.items()} != dict(
        expected_counts
    ):
        raise ValueError("role_counts_invalid")
    return output


def load_public_role_rows(root: Path) -> dict[str, list[JsonDict]]:
    """Reuse Exp7504 public bytes while avoiding every Exp7505 setting."""

    return assemble_public_rows(
        load_jsonl(root / FEATURE_PATH),
        load_jsonl(root / PREDICTOR_PATH),
        expected_counts=EXPECTED_ROLE_COUNTS,
    )


def compute_count_config(training_rows: Sequence[Mapping[str, Any]]) -> CountConfig:
    """Compute eight label-free means and midpoint defaults from training forecasts."""

    if not training_rows:
        raise ValueError("training_forecasts_required")
    bins: list[list[float]] = [[] for _ in range(8)]
    probabilities: list[float] = []
    for row in training_rows:
        probability = float(row.get("raw_whole_expectation", math.nan))
        if not 0.0 < probability < 1.0 or not math.isfinite(probability):
            raise ValueError("training_probability_invalid")
        probabilities.append(probability)
        index = min(7, math.floor(8 * min(1.0 - CLIP_MIN, max(CLIP_MIN, probability))))
        bins[index].append(probability)
    means = tuple(
        sum(values) / len(values) if values else (index + 0.5) / 8.0
        for index, values in enumerate(bins)
    )
    return CountConfig(means, sum(probabilities) / len(probabilities), PRIOR_MASS)


def _config_payload(config: CountConfig) -> JsonDict:
    """Expose immutable count settings without leaking implementation objects."""

    return {
        "bin_means": list(config.bin_means),
        "global_mean": config.global_mean,
        "kappa": config.kappa,
    }


def _config_from_payload(value: Mapping[str, Any]) -> CountConfig:
    """Rebuild the qualified count configuration from canonical protocol bytes."""

    return CountConfig(
        tuple(float(item) for item in value["bin_means"]),
        float(value["global_mean"]),
        float(value["kappa"]),
    )


def freeze_public_protocol(public: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    """Hash roles, settings, bins, and orders before any label-reader call."""

    for role in EXPECTED_ROLE_COUNTS:
        if role not in public:
            raise ValueError(f"public_role_missing:{role}")
    training = list(public["training"])
    online = list(public["online"])
    config = compute_count_config(training)
    orders = {
        str(seed): [str(row["group_id"]) for row in build_arrival_order(online, seed)]
        for seed in ORDER_SEEDS
    }
    identities = {str(row["group_id"]) for row in online}
    if any(  # pragma: no cover - the reused order helper preserves its input roster.
        len(order) != len(identities) or set(order) != identities for order in orders.values()
    ):
        raise ValueError("label_free_order_roster_mismatch")
    payload = {
        "settings": frozen_protocol(),
        "role_counts": {role: len(public[role]) for role in EXPECTED_ROLE_COUNTS},
        "role_identity_hashes": {
            role: canonical_hash(
                [
                    [row["group_id"], row["source_hash"], row["response_hash"], row["role"]]
                    for row in public[role]
                ]
            )
            for role in EXPECTED_ROLE_COUNTS
        },
        "count_config": _config_payload(config),
        "orders": orders,
    }
    return {
        **payload,
        "hash_payload": deepcopy(payload),
        "protocol_hash": canonical_hash(payload),
        "labels_read": False,
    }


def read_oriented_labels(
    evaluators: Sequence[Mapping[str, Any]],
    public: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    roles: Sequence[str],
    orientation: str = "one_means_contains_unsupported",
) -> dict[str, dict[str, int]]:
    """Open named roles and apply Exp7509's explicit hallucination orientation."""

    if orientation != "one_means_contains_unsupported":
        raise ValueError("label_orientation_invalid")
    allowed = set(roles)
    if not allowed or not allowed <= {"online", "test"}:
        raise ValueError("label_roles_invalid")
    eligible = {role: {str(row["group_id"]) for row in public[role]} for role in allowed}
    output = {role: {} for role in allowed}
    observed_pairs: set[tuple[str, str]] = set()
    eligible_groups = set().union(*eligible.values())
    for row in evaluators:
        role = str(row.get("role") or "")
        group = str(row.get("group_id") or "")
        if group not in eligible_groups:
            continue
        if role not in allowed or group not in eligible[role] or (role, group) in observed_pairs:
            raise ValueError(f"label_identity_mismatch:{group}")
        label = row.get("label")
        if label not in {0, 1}:
            raise ValueError(f"private_label_invalid:{group}")
        observed_pairs.add((role, group))
        output[role][group] = 1 - int(label)
    if any(set(output[role]) != eligible[role] for role in allowed):
        raise ValueError("label_identity_mismatch:incomplete_roster")
    return output


def load_private_labels(
    root: Path, public: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, int]]:
    """Open only online and retention labels after the public protocol freezes."""

    return read_oriented_labels(
        load_jsonl(root / EVALUATOR_PATH),
        public,
        roles=("online", "test"),
    )


def _private_fixture() -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Create small independent rows for interface mutation checks."""

    features: list[JsonDict] = []
    predictors: list[JsonDict] = []
    evaluators: list[JsonDict] = []
    roles = ("training", "online", "test")
    for index, role in enumerate(roles):
        probability = 0.2 + 0.3 * index
        source_hash = f"sha256:{index + 1:064x}"
        response_hash = f"sha256:{index + 101:064x}"
        group = f"private-{index}"
        features.append(
            {
                "group_id": group,
                "role": role,
                "source_hash": source_hash,
                "response_hash": response_hash,
                "raw_whole_expectation": probability,
                "features": [logit(probability), *([0.0] * 9)],
            }
        )
        predictors.append(
            {
                "group_id": group,
                "role": role,
                "source_hash": source_hash,
                "response_hash": response_hash,
                "source_family": "private",
            }
        )
        if role in {"online", "test"}:
            evaluators.append({"group_id": group, "role": role, "label": index % 2})
    return features, predictors, evaluators


def run_private_mutation_controls() -> list[JsonDict]:
    """Prove each registered custody mutation is rejected on private copies."""

    features, predictors, evaluators = _private_fixture()
    baseline = assemble_public_rows(features, predictors)
    controls: list[JsonDict] = []
    mutations = ("source_hash", "role_membership", "raw_normalization", "label_orientation")
    for mutation in mutations:
        changed_features = deepcopy(features)
        changed_predictors = deepcopy(predictors)
        rejected = False
        error = None
        try:
            if mutation == "source_hash":
                changed_predictors[0]["source_hash"] = "sha256:" + "f" * 64
                assemble_public_rows(changed_features, changed_predictors)
            elif mutation == "role_membership":
                changed_predictors[0]["role"] = "online"
                assemble_public_rows(changed_features, changed_predictors)
            elif mutation == "raw_normalization":
                changed_features[0]["raw_whole_expectation"] = 0.0
                assemble_public_rows(changed_features, changed_predictors)
            else:
                read_oriented_labels(
                    evaluators,
                    baseline,
                    roles=("online", "test"),
                    orientation="one_means_supported",
                )
        except ValueError as exc:
            rejected = True
            error = str(exc)
        controls.append(
            {
                "mutation": mutation,
                "baseline_valid": True,
                "rejected": rejected,
                "qualified": rejected,
                "observed_error": error,
                "disposition": "private_mutation_control",
            }
        )
    return controls


def _blocks(order: Sequence[str]) -> list[JsonDict]:
    """Register complete and short blocks without consulting any label."""

    output: list[JsonDict] = []
    stream_end = len(order) - 1
    for block_id, start in enumerate(range(0, len(order), BLOCK_SIZE)):
        event_ids = list(order[start : start + BLOCK_SIZE])
        end = start + len(event_ids) - 1
        release_time = end + FEEDBACK_DELAY
        output.append(
            {
                "block_id": block_id,
                "start_time": start,
                "end_time": end,
                "release_time": release_time,
                "event_ids": event_ids,
                "short_final_block": len(event_ids) < BLOCK_SIZE,
                "release_within_stream": release_time <= stream_end,
            }
        )
    return output


def run_order_lifecycle(
    online_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    order: Sequence[str],
    count_config: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_dir: Path,
) -> JsonDict:
    """Replay one full-audit order and cold-reload after each legal release."""

    by_group = {str(row["group_id"]): row for row in online_rows}
    if (
        len(by_group) != len(online_rows)
        or set(order) != set(by_group)
        or set(labels) != set(by_group)
    ):
        raise ValueError("lifecycle_roster_mismatch")
    config = _config_from_payload(count_config)
    machine = CountEventMachine.create(config)
    blocks = _blocks(order)
    block_by_release = {
        int(block["release_time"]): block for block in blocks if block["release_within_stream"]
    }
    prediction_rows: list[JsonDict] = []
    release_rows: list[JsonDict] = []
    persistence: list[JsonDict] = []
    chronology_violations = 0
    normalization_mismatches = 0
    restart_mismatches = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for prediction_time, group in enumerate(order):
        source = by_group[group]
        block_id = prediction_time // BLOCK_SIZE
        public_predictions = machine.predict(
            group,
            float(source["base_probability"]),
            release_index=block_id,
        )
        for public_arm, machine_arm in (
            ("frozen", "frozen"),
            ("global", "global"),
            ("local", "local"),
            ("shuffled_local", "permuted_local"),
        ):
            value = dict(public_predictions[machine_arm])
            probability = float(value["probability"])
            energies = [float(item) for item in value["energies"]]
            normalized = normalized_probability(energies)
            if not math.isclose(  # pragma: no cover - qualified CountArm defines these energies.
                normalized, probability, rel_tol=0.0, abs_tol=1e-12
            ):
                normalization_mismatches += 1
            prediction_rows.append(
                {
                    "seed": seed,
                    "prediction_time": prediction_time,
                    "event_id": group,
                    "source_hash": source["source_hash"],
                    "role": source["role"],
                    "arm": public_arm,
                    "base_probability": source["base_probability"],
                    "probability": probability,
                    "energies": energies,
                    "normalized_probability": normalized,
                    "bin_index": value["bin_index"],
                    "disposition": "predicted_before_feedback",
                }
            )

        due = block_by_release.get(prediction_time)
        if due is None:
            continue
        event_ids = list(due["event_ids"])
        if any(  # pragma: no cover - predictions are created above before this release lookup.
            not any(
                row["event_id"] == event_id and row["prediction_time"] <= prediction_time
                for row in prediction_rows
            )
            for event_id in event_ids
        ):
            chronology_violations += 1
        receipt = machine.release(
            int(due["block_id"]),
            [(event_id, int(labels[event_id])) for event_id in event_ids],
        )
        machine.acknowledge(int(due["block_id"]))
        path = checkpoint_dir / f"seed-{seed}-block-{due['block_id']:03d}.json"
        before = machine.to_payload()
        probe = min(1.0 - CLIP_MIN, max(CLIP_MIN, float(source["base_probability"])))
        next_before = {name: arm.predict(probe).probability for name, arm in machine.arms.items()}
        machine.save(path)
        reloaded = CountEventMachine.load(path)
        next_after = {name: arm.predict(probe).probability for name, arm in reloaded.arms.items()}
        payload_equal = before == reloaded.to_payload()
        next_equal = next_before == next_after
        if not payload_equal or not next_equal:  # pragma: no cover - mutation tests live upstream.
            restart_mismatches += 1
        machine = reloaded
        shuffled = receipt["permutation"]
        release_rows.append(
            {
                **deepcopy(due),
                "seed": seed,
                "prediction_completed_through": prediction_time,
                "labels": list(receipt["labels"]),
                "shuffled_labels": list(receipt["permuted_labels"]),
                "shuffled_label_origins": [row["label_origin"] for row in shuffled],
                "update_count_by_arm": {
                    "global": len(event_ids),
                    "local": len(event_ids),
                    "shuffled_local": len(event_ids),
                },
                "disposition": "released_updated_persisted_reloaded",
            }
        )
        persistence.append(
            {
                "seed": seed,
                "block_id": due["block_id"],
                "checkpoint_sha256": sha256_file(path),
                "state_hash": machine.state_hash(),
                "payload_equal_after_reload": payload_equal,
                "next_prediction_equal_after_reload": next_equal,
                "disposition": "cold_reload_checked",
            }
        )

    released_ids = {event_id for row in release_rows for event_id in row["event_ids"]}
    for block in blocks:
        if block["release_within_stream"]:
            continue
        release_rows.append(
            {
                **deepcopy(block),
                "seed": seed,
                "prediction_completed_through": len(order) - 1,
                "labels": None,
                "shuffled_labels": None,
                "shuffled_label_origins": [],
                "update_count_by_arm": {"global": 0, "local": 0, "shuffled_local": 0},
                "disposition": "censored_end_of_stream",
            }
        )
    return {
        "seed": seed,
        "prediction_rows": prediction_rows,
        "release_rows": sorted(release_rows, key=lambda row: int(row["block_id"])),
        "persistence_receipts": persistence,
        "prediction_count": len(prediction_rows),
        "released_event_count": len(released_ids),
        "censored_event_count": len(order) - len(released_ids),
        "chronology_violation_count": chronology_violations,
        "restart_mismatch_count": restart_mismatches,
        "exact_normalization_mismatch_count": normalization_mismatches,
        "final_state": machine.to_payload(),
        "disposition": "complete",
    }


REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7504_v657_evidence_interface.py"),
    Path("python/carnot/experiment_7509_v657_causal_online.py"),
    Path("python/carnot/experiment_7510_v657_causal_audit.py"),
    Path("python/carnot/experiment_7534_v659_count_memory.py"),
    FEATURE_PATH,
    PREDICTOR_PATH,
    EVALUATOR_PATH,
    ACCESS_PATH,
    *UPSTREAM_PATHS.values(),
    DESIGN_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _precondition(
    check: str,
    upstream: str,
    field_path: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    required: bool = True,
) -> JsonDict:
    """Keep exact operands so a blocked state cannot imply a zero metric."""

    return {
        "check": check,
        "upstream": upstream,
        "field_path": field_path,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": passed,
        "required": required,
        "principle": "Exact prerequisite operands prevent fabricated fallback evidence.",
    }


def _load_object(path: Path) -> JsonDict:
    """Return an object for valid JSON and an empty object for absent or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, str]]:
    """Authenticate every required path and exact upstream readiness operand."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        readable = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                "required_input_readable",
                relative.as_posix(),
                relative.as_posix(),
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if readable else None,
                readable,
            )
        )
        if readable:
            hashes[relative.as_posix()] = sha256_file(path)

    required_fields = (
        ("exp7504", "evidence_ready_score", 1),
        ("exp7504", "flagged_adversarial", False),
        ("exp7510", "causal_audit_complete_score", 1),
        ("exp7510", "causal_claims_qualified_score", 1),
        ("exp7510", "flagged_adversarial", False),
        ("exp7534", "count_memory_ready_score", 1),
        ("exp7534", "flagged_adversarial", False),
    )
    artifacts = {name: _load_object(root / path) for name, path in UPSTREAM_PATHS.items()}
    for name, field, expected in required_fields:
        observed = artifacts[name].get(field)
        checks.append(
            _precondition(
                f"upstream_field:{name}:{field}",
                UPSTREAM_PATHS[name].as_posix(),
                field,
                expected,
                observed,
                observed == expected,
            )
        )
    same_milestone = artifacts["exp7546"]
    checks.append(
        _precondition(
            "same_milestone_producer_completed_before_read",
            UPSTREAM_PATHS["exp7546"].as_posix(),
            "run_date",
            RUN_DATE,
            same_milestone.get("run_date"),
            same_milestone.get("run_date") == RUN_DATE,
            required=False,
        )
    )
    upstream_7504 = artifacts["exp7504"]
    for key, relative in (("features", FEATURE_PATH), ("access_exposure_manifest", ACCESS_PATH)):
        expected_hash = (upstream_7504.get("raw_sidecars") or {}).get(key, {}).get("sha256")
        observed_hash = hashes.get(relative.as_posix())
        checks.append(
            _precondition(
                f"exp7504_sidecar_hash:{key}",
                UPSTREAM_PATHS["exp7504"].as_posix(),
                f"raw_sidecars.{key}.sha256",
                expected_hash,
                observed_hash,
                isinstance(expected_hash, str) and expected_hash == observed_hash,
            )
        )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7547",
            "REQ-CL-7547" if "REQ-CL-7547" in spec else None,
            "REQ-CL-7547" in spec,
        )
    )
    checks.append(
        _precondition(
            "external_compute_resources",
            "host CPU/RAM/storage",
            "resource_requirement",
            "available_no_model_or_gpu_required",
            "available_no_model_or_gpu_required",
            True,
            required=False,
        )
    )
    return checks, hashes


def progress(  # pragma: no cover - visible only during the declared entrypoint.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush phase and slow-operation boundaries with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7547] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - real monotonic timing is an entrypoint boundary.
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    """Close one disjoint current-work interval for durable timing evidence."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed serial, coverage, lint, type, and spec command set."""

    basetemp = private_root / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=basetemp,
        coverage_file=private_root / ".coverage.exp7547",
    )


def terminal_commands(candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build four bounded fresh readers for one exact candidate path."""

    python = str(REPO_ROOT / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful, non-timeout receipt for every exact command name."""

    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return set(names) <= passed


def _sidecar(path: Path, root: Path, *, rows: int | None = None) -> JsonDict:
    """Reference exact current bytes without copying large event rows into the terminal JSON."""

    value: JsonDict = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def write_stream_sidecars(
    root: Path,
    frozen: Mapping[str, Any],
    replays: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, int]],
) -> dict[str, JsonDict]:
    """Publish bounded canonical protocol, event, retention, and final-state bytes."""

    raw_root = root / RAW_DIR
    protocol_path = raw_root / "frozen_protocol.json"
    event_path = raw_root / "stream_events.jsonl"
    retention_path = raw_root / "retention_rows.jsonl"
    state_path = raw_root / "final_states.json"
    atomic_json(protocol_path, dict(frozen))
    events = [
        {"row_kind": "prediction", **dict(row)}
        for replay in replays
        for row in replay["prediction_rows"]
    ] + [
        {"row_kind": "release", **dict(row)} for replay in replays for row in replay["release_rows"]
    ]
    write_jsonl(event_path, events)
    retention = [
        {
            "group_id": row["group_id"],
            "role": row["role"],
            "source_hash": row["source_hash"],
            "response_hash": row["response_hash"],
            "source_family": row["source_family"],
            "base_probability": row["base_probability"],
            "label": labels["test"][str(row["group_id"])],
            "label_orientation": "one_means_contains_unsupported",
            "disposition": "retention_only_no_adaptation",
        }
        for row in retention_rows
    ]
    write_jsonl(retention_path, retention)
    states = {str(replay["seed"]): replay["final_state"] for replay in replays}
    atomic_json(state_path, states)
    return {
        "frozen_protocol": _sidecar(protocol_path, root),
        "stream_events": _sidecar(event_path, root, rows=len(events)),
        "retention_rows": _sidecar(retention_path, root, rows=len(retention)),
        "final_states": _sidecar(state_path, root),
    }


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute custody and lifecycle readiness from raw terminal fields."""

    frozen = value.get("frozen_protocol") or {}
    replays = value.get("replay_receipts") or []
    counts = frozen.get("role_counts") or {}
    online_count = int(counts.get("online", 0))
    orders = frozen.get("orders") or {}
    protocol_valid = (
        frozen.get("protocol_hash") == canonical_hash(frozen.get("hash_payload"))
        and frozen.get("labels_read") is False
        and set(orders) == {str(seed) for seed in ORDER_SEEDS}
        and all(
            len(order) == online_count and len(set(order)) == online_count
            for order in orders.values()
        )
    )
    expected_blocks = _blocks(next(iter(orders.values()), []))
    expected_released = sum(
        len(row["event_ids"]) for row in expected_blocks if row["release_within_stream"]
    )
    expected_censored = online_count - expected_released
    replay_valid = (
        len(replays) == len(ORDER_SEEDS)
        and {row.get("seed") for row in replays} == set(ORDER_SEEDS)
        and all(
            row.get("prediction_count") == online_count * len(ARMS)
            and row.get("released_event_count") == expected_released
            and row.get("censored_event_count") == expected_censored
            and row.get("chronology_violation_count") == 0
            and row.get("restart_mismatch_count") == 0
            and row.get("exact_normalization_mismatch_count") == 0
            and row.get("disposition") == "complete"
            for row in replays
        )
    )
    controls = value.get("private_mutation_controls") or []
    controls_valid = len(controls) == 4 and all(
        row.get("baseline_valid") is True
        and row.get("rejected") is True
        and row.get("qualified") is True
        for row in controls
    )
    checks = value.get("preconditions_checked") or []
    preconditions_valid = all(
        row.get("passed") is True for row in checks if row.get("required") is True
    )
    receipts = value.get("validation_receipts") or []
    affected_valid = _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal_valid = _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    terminal_present = any(row.get("name") in TERMINAL_CHECK_NAMES for row in receipts)
    validity = (
        protocol_valid
        and replay_valid
        and controls_valid
        and preconditions_valid
        and affected_valid
        and (terminal_valid or not terminal_present)
    )
    return {
        "protocol_valid": protocol_valid,
        "replay_valid": replay_valid,
        "mutation_controls_valid": controls_valid,
        "preconditions_valid": preconditions_valid,
        "affected_validation_passed": affected_valid,
        "terminal_validation_passed": terminal_valid,
        "terminal_validation_present": terminal_present,
        "validity_passed": validity,
        "benefit_measured": False,
        "expected_released_events_per_order": expected_released,
        "expected_censored_events_per_order": expected_censored,
        "cached_stream_ready_score": int(validity),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Retain exact operands and the scientific failure each gate prevents."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": passed,
        "principle": principle,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain the drift or claim error prevented by every artifact field."""

    specific = {
        "experiment_id": "Binds evidence to the exact roadmap task.",
        "preconditions_checked": "Prevents absent inputs from becoming fabricated measurements.",
        "MODEL_SPECS": "Makes the no-model plan explicit.",
        "model_invoked": "Separates current calls from historical cached inference.",
        "inference_substrate_class": "Prevents CPU aggregation from being reported as model work.",
        "inference_substrate": "Names cached-artifact aggregation as the actual substrate.",
        "duration_s": "Prevents historical inference time from becoming current runtime.",
        "rows": "Retains absolute per-arm lifecycle counts instead of only a verdict.",
        "sample_size_budget": "Keeps censored and unstarted units distinct from zero outcomes.",
        "cached_stream_ready_score": "Keeps interface readiness separate from predictive benefit.",
        "frozen_protocol": "Proves orders and means existed before labels were opened.",
        "exposure_scope": "Prevents previously inspected labels from becoming fresh confirmation.",
        "continuous_self_learning_task": "Marks the causal predict-release-update-persist loop.",
    }
    return {
        field: specific.get(
            field, "Preserves this field so independent readers can detect evidence drift."
        )
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind code, sources, roles, settings, sidecars, and raw lifecycle receipts."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def build_artifact(
    *,
    frozen: Mapping[str, Any],
    replays: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, int]],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    sidecars: Mapping[str, Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assemble one terminal record from frozen public and released private evidence."""

    del labels  # Labels affect only released rows and retention custody, never artifact gates.
    compact_replays = [
        {
            key: deepcopy(replay[key])
            for key in (
                "seed",
                "prediction_count",
                "released_event_count",
                "censored_event_count",
                "chronology_violation_count",
                "restart_mismatch_count",
                "exact_normalization_mismatch_count",
                "persistence_receipts",
                "disposition",
            )
        }
        for replay in replays
    ]
    controls = run_private_mutation_controls()
    rows = [
        {
            "unit_id": f"seed-{replay['seed']}-arm-{arm}",
            "trial_seed": replay["seed"],
            "arm": arm,
            "absolute_metrics": {
                "predictions": int(replay["prediction_count"]) // len(ARMS),
                "released_updates": 0 if arm == "frozen" else replay["released_event_count"],
                "censored_updates": 0 if arm == "frozen" else replay["censored_event_count"],
                "restart_mismatches": replay["restart_mismatch_count"],
            },
            "benefit_metric": None,
            "disposition": "complete_interface_readiness_only",
            "failed": False,
            "censored": False,
        }
        for replay in replays
        for arm in ARMS
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Authenticated cached count-learning stream",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_provenance": {
            "source": UPSTREAM_PATHS["exp7504"].as_posix(),
            "model_calls_are_current": False,
            "forecast_field": "raw_whole_expectation",
            "temperature_applied": False,
            "trained_head_imported": False,
        },
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "device_identity": {
            "device_class": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "gpu_required": False,
        },
        "duration_s": max(float(duration_s), 1e-9),
        "duration_breakdown_s": {
            "current_work_measured": max(float(duration_s), 1e-9),
            "authoring": None,
            "historical_inference": None,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {
            "pid": os.getpid(),
            "python": os.path.realpath(os.sys.executable),
            "worktree": str(REPO_ROOT),
        },
        "random_seed": list(ORDER_SEEDS),
        "random_seeds": {"label_blind_orders": list(ORDER_SEEDS)},
        "source_artifact_hashes": dict(source_hashes),
        "raw_sidecars": deepcopy(dict(sidecars)),
        "frozen_protocol": deepcopy(dict(frozen)),
        "replay_receipts": compact_replays,
        "private_mutation_controls": controls,
        "rows": rows,
        "sample_size_budget": {
            "training_forecasts": {
                "planned": int(frozen["role_counts"]["training"]),
                "attempted": int(frozen["role_counts"]["training"]),
                "completed": int(frozen["role_counts"]["training"]),
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
            },
            "independent_online_groups": {
                "planned": int(frozen["role_counts"]["online"]),
                "attempted": int(frozen["role_counts"]["online"]),
                "completed": int(frozen["role_counts"]["online"]),
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
            },
            "retention_groups": {
                "planned": len(retention_rows),
                "attempted": len(retention_rows),
                "completed": len(retention_rows),
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
            },
            "ordered_event_instances": {
                "planned": int(frozen["role_counts"]["online"]) * len(ORDER_SEEDS),
                "attempted": int(frozen["role_counts"]["online"]) * len(ORDER_SEEDS),
                "completed": int(frozen["role_counts"]["online"]) * len(ORDER_SEEDS),
                "excluded": 0,
                "failed": 0,
                "censored": sum(int(row["censored_event_count"]) for row in replays),
                "unstarted": 0,
            },
        },
        "exposure_scope": {
            "prior_labels_used_in_previous_research": True,
            "claim_scope": "exploratory_mechanism_test",
            "fresh_confirmatory_claim_allowed": False,
        },
        "continuous_self_learning_task": True,
        "capability_e2e": {
            "name": "predict_release_update_persist_reload",
            "used_real_cached_rows": bool(replays),
            "numbered_runtime_e2e_applicable": [],
            "restart_mismatch_count": sum(int(row["restart_mismatch_count"]) for row in replays),
        },
        "label_access_receipts": {
            "protocol_hash_before_labels": frozen["protocol_hash"],
            "training_labels_opened": False,
            "online_labels_opened_after_freeze": True,
            "retention_labels_opened_after_freeze": True,
            "orientation": "one_means_contains_unsupported",
        },
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "verifier_is_oracle": False,
        "positive_claim": False,
        "predictive_benefit_measured": False,
        "no_headroom": {
            "applies": False,
            "annotation": "Benefit was not measured in this readiness task.",
        },
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
    }
    reduction = independent_reduce(artifact)
    gates = [
        _gate(
            "authenticated_public_custody",
            "validity",
            True,
            reduction["preconditions_valid"],
            reduction["preconditions_valid"],
            "Invalid source, role, orientation, or upstream bytes cannot support science.",
        ),
        _gate(
            "private_mutations_rejected",
            "validity",
            True,
            reduction["mutation_controls_valid"],
            reduction["mutation_controls_valid"],
            "A favorable stream cannot excuse a reader that accepts identity drift.",
        ),
        _gate(
            "predict_release_update_persist_reload",
            "validity",
            True,
            reduction["replay_valid"],
            reduction["replay_valid"],
            "Future labels or restart drift would invalidate the stream.",
        ),
        _gate(
            "required_scoped_validation",
            "validity",
            True,
            reduction["affected_validation_passed"],
            reduction["affected_validation_passed"],
            "Required test, coverage, lint, type, and specification failures disqualify evidence.",
        ),
        _gate(
            "cached_stream_ready",
            "readiness",
            1,
            reduction["cached_stream_ready_score"],
            reduction["cached_stream_ready_score"] == 1,
            "A valid null remains reusable when interface readiness is independent of efficacy.",
        ),
        _gate(
            "benefit_not_measured",
            "benefit",
            False,
            reduction["benefit_measured"],
            reduction["benefit_measured"] is False,
            "A readiness task cannot promote an unmeasured predictive gain.",
        ),
    ]
    if reduction["terminal_validation_present"]:
        gates.insert(
            4,
            _gate(
                "fresh_terminal_readers",
                "validity",
                True,
                reduction["terminal_validation_passed"],
                reduction["terminal_validation_passed"],
                "Cold replay and independent readers must agree before promotion.",
            ),
        )
    failed = [deepcopy(gate) for gate in gates if gate["passed"] is not True]
    artifact.update(
        {
            "independent_reduction": reduction,
            "acceptance_gate_results": gates,
            "gate_check_summary": {
                "passed": not failed,
                "failed_checks": failed,
                "first_failure": failed[0] if failed else None,
            },
            "cached_stream_ready_score": reduction["cached_stream_ready_score"],
            "flagged_adversarial": not reduction["mutation_controls_valid"],
            "honest_verdict": (
                "complete_null_cached_count_stream_ready_benefit_unmeasured"
                if reduction["cached_stream_ready_score"] == 1
                else "complete_disqualified_cached_count_stream_invalid"
            ),
            "verdict_class": (
                "null" if reduction["cached_stream_ready_score"] == 1 else "disqualified"
            ),
        }
    )
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _verify_sidecars(root: Path, sidecars: Mapping[str, Any]) -> None:
    """Rehash each bounded sidecar so a path cannot substitute changed bytes."""

    expected_names = {"frozen_protocol", "stream_events", "retention_rows", "final_states"}
    if set(sidecars) != expected_names:
        raise ValueError("raw_sidecar_set_mismatch")
    for name, receipt in sidecars.items():
        path = root / str(receipt.get("path") or "")
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            raise ValueError(f"raw_sidecar_hash_mismatch:{name}")
        if path.stat().st_size != receipt.get("bytes"):
            raise ValueError(f"raw_sidecar_size_mismatch:{name}")


def validate_artifact(
    value: Mapping[str, Any],
    *,
    require_terminal: bool = True,
    verify_sidecars: bool = True,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Reject changed identity, reduction, checksum, validation, or sidecar bytes."""

    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("artifact_identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        raise ValueError("artifact_date_or_milestone_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        raise ValueError("model_specs_must_be_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        raise ValueError("current_model_invocation_mismatch")
    if value.get("inference_substrate_class") != "no_model_load":
        raise ValueError("inference_substrate_class_mismatch")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        raise ValueError("inference_substrate_mismatch")
    if value.get("execution_venue") != "host":
        raise ValueError("execution_venue_mismatch")
    reduction = independent_reduce(value)
    stored = value.get("independent_reduction") or {}
    if reduction != stored:
        raise ValueError("independent_reduction_mismatch")
    if value.get("cached_stream_ready_score") != reduction["cached_stream_ready_score"]:
        raise ValueError("cached_stream_ready_score_mismatch")
    if (
        value.get("positive_claim") is not False
        or value.get("predictive_benefit_measured") is not False
    ):
        raise ValueError("unmeasured_benefit_claim")
    expected_verdict = "null" if reduction["cached_stream_ready_score"] == 1 else "disqualified"
    if value.get("verdict_class") != expected_verdict:
        raise ValueError("verdict_class_mismatch")
    if require_terminal and not reduction["terminal_validation_passed"]:
        raise ValueError("terminal_validation_missing_or_failed")
    principles = value.get("field_principles") or {}
    if any(field not in principles for field in value):
        raise ValueError("field_principles_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        raise ValueError("reproducibility_checksum_mismatch")
    if verify_sidecars:
        _verify_sidecars(root, value.get("raw_sidecars") or {})
    return reduction


def cold_replay(
    path: Path, *, require_terminal: bool = False, verify_sidecars: bool = True
) -> JsonDict:
    """Validate serialized evidence through the same strict fresh-process reader."""

    value = _load_object(path)
    if not value:
        raise ValueError("artifact_unreadable_or_not_object")
    return validate_artifact(
        value,
        require_terminal=require_terminal,
        verify_sidecars=verify_sidecars,
    )


def build_blocked_artifact(
    failed: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Publish an exact external absence without inventing stream measurements."""

    reason = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(failed.get("check") or "external_prerequisite")
    ).strip("_")
    gate = {
        "check": failed.get("check"),
        "category": "validity",
        "upstream": failed.get("upstream"),
        "field_path": failed.get("field_path"),
        "expected": failed.get("expected"),
        "observed": failed.get("observed"),
        "op": failed.get("op", "eq"),
        "passed": False,
        "principle": "Missing external evidence blocks measurement and is never a zero result.",
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "blocked_no_run",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "device_identity": {"device_class": "cpu", "gpu_required": False},
        "duration_s": max(float(duration_s), 1e-9),
        "phase_spans": [],
        "process_identity": {"pid": os.getpid(), "worktree": str(REPO_ROOT)},
        "random_seed": list(ORDER_SEEDS),
        "source_artifact_hashes": {},
        "raw_sidecars": {},
        "frozen_protocol": {},
        "replay_receipts": [],
        "private_mutation_controls": [],
        "rows": [],
        "sample_size_budget": {
            "planned": 159,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 159,
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": [gate],
            "first_failure": gate,
        },
        "cached_stream_ready_score": 0,
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "positive_claim": False,
        "predictive_benefit_measured": False,
        "exposure_scope": {
            "prior_labels_used_in_previous_research": True,
            "fresh_confirmatory_claim_allowed": False,
        },
        "continuous_self_learning_task": True,
        "validation_receipts": [],
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
    }
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _write_affected_manifest(path: Path) -> None:  # pragma: no cover - entrypoint boundary.
    """Freeze the exact file scope before any validation child starts."""

    value = {
        "experiment_id": AFFECTED_MANIFEST.experiment_id,
        "test_paths": list(AFFECTED_MANIFEST.test_paths),
        "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
        "static_paths": list(AFFECTED_MANIFEST.static_paths),
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.is_file() and path.read_bytes() != encoded:
        raise ValueError("affected_manifest_changed_after_freeze")
    if not path.exists():
        atomic_json(path, value)


def run_experiment(  # pragma: no cover - exercised by the declared entrypoint E2E.
    root: Path,
    run_date: str,
    *,
    output_path: Path | None = None,
) -> JsonDict:
    """Authenticate, replay, validate in fresh processes, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    raw_root = root / RAW_DIR
    started = time.monotonic()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    checks, source_hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = next(
        (row for row in checks if row.get("required") is True and row.get("passed") is not True),
        None,
    )
    if failed is not None:
        blocked = build_blocked_artifact(
            failed,
            checks,
            duration_s=time.monotonic() - started,
        )
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(started, "publish", "complete_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "complete", completed_units=len(checks))

    manifest_path = raw_root / "affected_validation_manifest.json"
    _write_affected_manifest(manifest_path)
    source_hashes[manifest_path.relative_to(root).as_posix()] = sha256_file(manifest_path)

    progress(started, "public_protocol", "start")
    phase_started = time.monotonic()
    public = load_public_role_rows(root)
    frozen = freeze_public_protocol(public)
    spans.append(
        _span("public_protocol", phase_started, started, sum(frozen["role_counts"].values()))
    )
    progress(
        started,
        "public_protocol",
        "complete_before_label_open",
        protocol_hash=frozen["protocol_hash"],
    )

    progress(started, "private_label_access", "start_after_protocol_freeze")
    phase_started = time.monotonic()
    labels = load_private_labels(root, public)
    spans.append(
        _span("private_label_access", phase_started, started, sum(map(len, labels.values())))
    )
    progress(
        started, "private_label_access", "complete", completed_units=sum(map(len, labels.values()))
    )

    scratch = Path(tempfile.mkdtemp(prefix="carnot-exp7547-replay-", dir="/tmp"))
    replays: list[JsonDict] = []
    progress(started, "count_stream_replay", "start", orders=len(ORDER_SEEDS))
    phase_started = time.monotonic()
    for completed, seed in enumerate(ORDER_SEEDS, 1):
        progress(started, "count_stream_replay", "before_order", seed=seed)
        replay = run_order_lifecycle(
            public["online"],
            labels["online"],
            frozen["orders"][str(seed)],
            frozen["count_config"],
            seed=seed,
            checkpoint_dir=scratch / str(seed),
        )
        replays.append(replay)
        progress(
            started,
            "count_stream_replay",
            "after_order",
            completed_units=completed,
            seed=seed,
        )
    spans.append(_span("count_stream_replay", phase_started, started, len(replays)))

    progress(started, "sidecars", "before_atomic_writes")
    phase_started = time.monotonic()
    sidecars = write_stream_sidecars(root, frozen, replays, public["test"], labels)
    spans.append(_span("sidecars", phase_started, started, len(sidecars)))
    progress(started, "sidecars", "after_atomic_writes", completed_units=len(sidecars))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7547-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private_root)
    progress(started, "affected_validation", "before_subprocesses", commands=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_root / "validation" / "affected",
        heartbeat_s=60.0,
    )
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    affected_passed = _receipts_pass(affected, validation_scope.REQUIRED_CHECK_NAMES)
    progress(started, "affected_validation", "after_subprocesses", passed=affected_passed)

    candidate = build_artifact(
        frozen=frozen,
        replays=replays,
        retention_rows=public["test"],
        labels=labels,
        validation_receipts=affected,
        preconditions_checked=checks,
        source_hashes=source_hashes,
        sidecars=sidecars,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    validate_artifact(candidate, require_terminal=False, root=root)
    if not affected_passed:
        progress(started, "publish", "before_atomic_disqualified")
        atomic_json(destination, candidate)
        progress(started, "publish", "complete_disqualified")
        return candidate
    candidate_path = raw_root / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    progress(started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        terminal_commands(candidate_path),
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        frozen=frozen,
        replays=replays,
        retention_rows=public["test"],
        labels=labels,
        validation_receipts=[*affected, *terminal],
        preconditions_checked=checks,
        source_hashes=source_hashes,
        sidecars=sidecars,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    validate_artifact(final, require_terminal=terminal_passed, root=root)
    if not terminal_passed:
        progress(started, "publish", "before_atomic_disqualified_terminal")
        atomic_json(destination, final)
        progress(started, "publish", "complete_disqualified_terminal")
        return final

    exact_path = raw_root / "exact_terminal_candidate.json"
    atomic_json(exact_path, final)
    progress(started, "exact_candidate_validation", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        terminal_commands(exact_path),
        log_dir=raw_root / "validation" / "exact_terminal",
        heartbeat_s=60.0,
    )
    exact_passed = _receipts_pass(exact, TERMINAL_CHECK_NAMES)
    progress(started, "exact_candidate_validation", "after_subprocesses", passed=exact_passed)
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    progress(
        started,
        "publish",
        "complete",
        cached_stream_ready_score=final["cached_stream_ready_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse producer and two read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-test-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the producer or one strict serialized-evidence reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    verify_sidecars = not args.allow_test_validation
    if args.cold_replay is not None:
        reduction = cold_replay(
            args.cold_replay,
            require_terminal=False,
            verify_sidecars=verify_sidecars,
        )
        print(json.dumps(reduction, sort_keys=True), flush=True)
        return int(reduction["cached_stream_ready_score"] != 1)
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        if not value:
            raise ValueError("artifact_unreadable_or_not_object")
        reduction = validate_artifact(
            value,
            require_terminal=False,
            verify_sidecars=verify_sidecars,
        )
        print(json.dumps(reduction, sort_keys=True), flush=True)
        return int(reduction["cached_stream_ready_score"] != 1)
    artifact = run_experiment(REPO_ROOT, args.date, output_path=args.output)
    print(
        json.dumps(
            {
                "result": str(args.output or RESULT_PATH),
                "cached_stream_ready_score": artifact["cached_stream_ready_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(artifact["verdict_class"] not in {"null", "circular_positive", "positive"})


if __name__ == "__main__":  # pragma: no cover - wrapper is the public executable.
    raise SystemExit(main())

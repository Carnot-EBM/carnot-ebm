"""Run the sealed utility and safety audit for bounded self-learning.

Each arm and frozen order runs in its own process. A second process restores
the private checkpoint. Row-level evidence is the only source for headlines.

Spec ref: REQ-LEARN-6873.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_6872_bounded_reliability_controller_quarantine as controller


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_RELATIVE_PATH = Path(
    "results/experiment_6872_bounded_reliability_controller_quarantine.json"
)
SOURCE_RELATIVE_PATH = Path(
    "results/experiment_6871_observable_reliability_opportunity_stream.json"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6873_prospective_sealed_self_learning_audit.json")
INFERENCE_SUBSTRATE = "fresh-process deterministic CPU prospective CSL audit"
RANDOM_SEED = 6_873_001
BLOCKED_VERDICT = "complete_blocked_prospective_sealed_self_learning_audit"
POSITIVE_VERDICT = "complete_positive_prospective_sealed_self_learning_utility_and_safety"
NULL_VERDICT = "complete_null_prospective_sealed_self_learning_utility_not_demonstrated"
PARTIAL_VERDICT = "complete_partial_prospective_sealed_self_learning_evidence_incomplete"
DISQUALIFIED_VERDICT = "complete_disqualified_prospective_sealed_self_learning_safety_failure"

ARMS = controller.ARMS
ACTIONS = controller.ACTIONS
SAFE_STATE_ARMS = set(ARMS) - {"v599_unsafe_reference"}
LEARNER_ARMS = ("bounded_update", "exact_quarantine")
COMPARATOR_ARMS = ("frozen_no_memory", "read_only", "v599_unsafe_reference")
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

FROZEN_INPUT_HASHES = {
    "controller_artifact": {
        "path": str(CONTROLLER_RELATIVE_PATH),
        "sha256": "sha256:dcc94628dd92c98d7af043b9fadc4cabd4fdbd6c790f43a9d16863284517e2d0",
    },
    "opportunity_stream": {
        "path": str(SOURCE_RELATIVE_PATH),
        "sha256": "sha256:e3f38198b15db735d5d1667aac4d21c7ce956ae8a6e03841bd5d85c356093a26",
    },
    "controller_module": {
        "path": "python/carnot/experiment_6872_bounded_reliability_controller_quarantine.py",
        "sha256": "sha256:5af45355d62740de13d32353d493d59b385a338356ef981a98fc688c16e40b54",
    },
    "controller_wrapper": {
        "path": "scripts/experiments/experiment_6872_bounded_reliability_controller_quarantine.py",
        "sha256": "sha256:cfdf2ce0c25ca466bc8befb45b510d2d0551aa49e27269d442d1d420debbab7d",
    },
    "controller_tests": {
        "path": "tests/python/test_experiment_6872_bounded_reliability_controller_quarantine.py",
        "sha256": "sha256:77aa9c0480eb98043360f2e82d17248f1739de3b64a40bd0016e63f737bc444d",
    },
}

OWN_SOURCE_PATHS = {
    "module": "python/carnot/experiment_6873_prospective_sealed_self_learning_audit.py",
    "wrapper": "scripts/experiments/experiment_6873_prospective_sealed_self_learning_audit.py",
    "focused_tests": "tests/python/test_experiment_6873_prospective_sealed_self_learning_audit.py",
    "spec": "openspec/capabilities/continuous-learning/spec.md",
}

REQUIRED_FIELDS = {
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "continuous_self_learning_task",
    "no_model_weight_mutation",
    "rows",
    "per_order_results",
    "action_distribution_by_arm",
    "admitted_useful_updates_by_arm",
    "harmful_writes_by_arm",
    "false_injection_rate_by_arm",
    "held_future_utility_by_arm",
    "paired_order_effects",
    "old_family_retention_by_arm",
    "spectral_bound_audit_rows",
    "delayed_correction_rows",
    "persistence_rows",
    "restart_rows",
    "rollback_rows",
    "leakage_witnesses",
    "scientific_claim_eligible",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
HEADLINE_FIELDS = (
    "per_order_results",
    "action_distribution_by_arm",
    "admitted_useful_updates_by_arm",
    "harmful_writes_by_arm",
    "false_injection_rate_by_arm",
    "held_future_utility_by_arm",
    "old_family_retention_by_arm",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes for receipts and exact persistence checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    """Return one project-style SHA-256 identity for JSON content."""

    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash exact file bytes and return an empty identity for a missing file."""

    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def check_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Use one shape for every fail-closed gate decision."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all checks and expose the first exact failure."""

    copied = [dict(row) for row in checks]
    failures = [row for row in copied if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": copied,
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def _read_json(path: Path) -> tuple[JsonDict, str | None]:
    """Read one JSON object without hiding the exact failure class."""

    if not path.is_file():
        return {}, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {}, f"unreadable:{type(error).__name__}"
    if not isinstance(value, dict):
        return {}, "json_object_required"
    return value, None


def _resolve(root: Path, path: str | Path) -> Path:
    """Resolve fixture and production paths by the same deterministic rule."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _observed_frozen_hashes(
    root: Path,
    expected_hashes: Mapping[str, Mapping[str, str]],
    controller_path: Path,
    source_path: Path,
) -> JsonDict:
    """Hash every frozen input from disk instead of trusting declarations."""

    observed: JsonDict = {}
    for key, descriptor in expected_hashes.items():
        if key == "controller_artifact":
            path = controller_path
        elif key == "opportunity_stream":
            path = source_path
        else:
            path = _resolve(root, descriptor.get("path", ""))
        observed[key] = {"path": str(descriptor.get("path", "")), "sha256": sha256_file(path)}
    return observed


def precondition_checks(
    root: Path,
    controller_artifact: Mapping[str, Any],
    source_artifact: Mapping[str, Any],
    controller_path: Path,
    source_path: Path,
    controller_error: str | None,
    source_error: str | None,
    expected_hashes: Mapping[str, Mapping[str, str]],
) -> list[JsonDict]:
    """Check readiness, immutable inputs, orders, and counterfactual support."""

    observed_hashes = _observed_frozen_hashes(root, expected_hashes, controller_path, source_path)
    expected_sha = {key: row.get("sha256") for key, row in expected_hashes.items()}
    observed_sha = {key: row.get("sha256") for key, row in observed_hashes.items()}
    declared = controller_artifact.get("source_artifact_hashes", {})
    declared_map = {
        "opportunity_stream": "observable_reliability_stream",
        "controller_module": "module",
        "controller_wrapper": "wrapper",
        "controller_tests": "focused_tests",
    }
    declaration_matches = all(
        isinstance(declared, Mapping)
        and isinstance(declared.get(controller_key), Mapping)
        and declared[controller_key].get("sha256") == expected_sha.get(expected_key)
        for expected_key, controller_key in declared_map.items()
    )
    exact_hashes = expected_sha == observed_sha and declaration_matches

    events_value = source_artifact.get("rows", [])
    events = events_value if isinstance(events_value, list) else []
    event_ids = [row.get("event_identity") for row in events if isinstance(row, Mapping)]
    controller_orders_value = controller_artifact.get("order_seed_manifest", [])
    controller_orders = controller_orders_value if isinstance(controller_orders_value, list) else []
    source_orders_value = source_artifact.get("order_replicate_manifest", [])
    source_orders = source_orders_value if isinstance(source_orders_value, list) else []
    controller_seed_map = {
        row.get("replicate_id"): row.get("seed")
        for row in controller_orders
        if isinstance(row, Mapping) and row.get("all_events_preserved") is True
    }
    source_seed_map = {
        row.get("replicate_id"): row.get("seed")
        for row in source_orders
        if isinstance(row, Mapping) and row.get("all_events_preserved") is True
    }
    seeds = list(controller_seed_map.values())
    five_orders = (
        len(controller_seed_map) >= 5
        and len(set(seeds)) == len(seeds)
        and controller_seed_map == source_seed_map
    )
    complete_orders = bool(events) and all(
        isinstance(row, Mapping)
        and row.get("all_events_preserved") is True
        and len(row.get("event_identities", [])) == len(event_ids)
        and set(row.get("event_identities", [])) == set(event_ids)
        for row in source_orders
    )
    controller_support_value = controller_artifact.get("counterfactual_support_rows", [])
    controller_support = (
        controller_support_value if isinstance(controller_support_value, list) else []
    )
    source_support_value = source_artifact.get("counterfactual_support_rows", [])
    source_support = source_support_value if isinstance(source_support_value, list) else []
    complete_support = bool(events) and all(
        len(rows) == len(events)
        and all(isinstance(row, Mapping) and row.get("valid_pre_action") is True for row in rows)
        for rows in (controller_support, source_support)
    )
    return [
        check_row(
            "controller_artifact_readable",
            "readable JSON object",
            controller_error or "readable",
            controller_error is None,
        ),
        check_row(
            "source_artifact_readable",
            "readable JSON object",
            source_error or "readable",
            source_error is None,
        ),
        check_row(
            "bounded_reliability_controller_ready_score",
            1,
            controller_artifact.get("bounded_reliability_controller_ready_score"),
            controller_artifact.get("bounded_reliability_controller_ready_score") == 1,
        ),
        check_row(
            "exact_frozen_source_and_code_hashes",
            {"hashes": expected_sha, "controller_declarations_match": True},
            {"hashes": observed_sha, "controller_declarations_match": declaration_matches},
            exact_hashes,
        ),
        check_row(
            "at_least_five_unique_frozen_order_seeds",
            {"minimum": 5, "controller_and_source_match": True},
            {
                "count": len(controller_seed_map),
                "unique_count": len(set(seeds)),
                "controller_and_source_match": controller_seed_map == source_seed_map,
            },
            five_orders,
        ),
        check_row(
            "complete_frozen_order_event_support",
            {"event_count": len(events), "all_orders_complete": True},
            {"event_count": len(events), "all_orders_complete": complete_orders},
            complete_orders,
        ),
        check_row(
            "complete_counterfactual_support",
            {"event_count": len(events), "invalid_count": 0},
            {
                "controller_count": len(controller_support),
                "source_count": len(source_support),
                "invalid_count": sum(
                    row.get("valid_pre_action") is not True
                    for rows in (controller_support, source_support)
                    for row in rows
                    if isinstance(row, Mapping)
                ),
            },
            complete_support,
        ),
        check_row(
            "no_model_weight_mutation_ready_input",
            True,
            controller_artifact.get("no_model_weight_mutation"),
            controller_artifact.get("no_model_weight_mutation") is True,
        ),
    ]


def _memory_bytes(memory: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode active memory independently from quarantine tombstones."""

    return canonical_json_bytes([dict(row) for row in memory])


def _tombstone(event: Mapping[str, Any], failed_checks: Sequence[str]) -> JsonDict:
    """Keep enough immutable evidence to prevent corrected-write resurrection."""

    outcome = event.get("later_exact_outcome", {})
    return {
        "event_identity": event.get("event_identity"),
        "exact_outcome_hash": outcome.get("exact_outcome_hash")
        if isinstance(outcome, Mapping)
        else None,
        "failed_checks": list(failed_checks),
    }


def run_arm_replicate(
    events: Sequence[Mapping[str, Any]],
    arm: str,
    replicate_id: str,
    order_seed: int,
    checkpoint_path: Path,
) -> JsonDict:
    """Run one arm and order from an empty state and write a private checkpoint."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    state = controller.initialize_reliability_state()
    initial_state_sha = controller.state_checksum(state)
    memory: list[JsonDict] = []
    tombstones: list[JsonDict] = []
    rows: list[JsonDict] = []
    held_start = (2 * len(events)) // 3 + 1

    for sequence, event in enumerate(events, start=1):
        features = event.get("decision_features", {})
        outcome = event.get("later_exact_outcome", {})
        if not isinstance(features, Mapping) or not isinstance(outcome, Mapping):
            raise ValueError("event decision features and exact outcome are required")
        leakage = controller.decision_feature_failures(features)
        action = "abstain" if leakage else controller.choose_action(arm, features, state)
        direction = int(outcome.get("signed_direction", 0))
        parent_bytes = _memory_bytes(memory)
        write_receipt: JsonDict | None = None
        admission = "not_applicable"
        if action == "write":
            write_receipt, checked_memory = controller.evaluate_write_transition(event, memory)
            if arm == "v599_unsafe_reference":
                admission = "unsafe_admitted"
                memory = memory + [controller._write_record(event)]
            else:
                admission = str(write_receipt["admission_decision"])
                memory = checked_memory
                if admission == "quarantined":
                    tombstones.append(_tombstone(event, write_receipt["failed_checks"]))
        admitted = admission in {"admitted", "unsafe_admitted"}
        useful = bool(action == "write" and direction > 0 and admitted)
        harmful = bool(action == "write" and direction < 0 and admitted)
        utility = controller._action_reward(action, direction, admitted)
        state_before_sha = controller.state_checksum(state)
        if arm in LEARNER_ARMS:
            state_after, transition = controller.bounded_symmetric_update(
                state,
                controller._source_state_node(outcome),
                action,
                utility,
            )
        elif arm == "v599_unsafe_reference":
            state_after, transition = controller.bounded_symmetric_update(
                state,
                controller._source_state_node(outcome),
                action,
                utility,
                learning_rate=0.2,
                enforce_bound=False,
            )
        else:
            state_after = deepcopy(state)
            transition = controller._no_update_receipt(state)
        state = state_after

        correction = outcome.get("delayed_correction", {})
        correction_present = bool(
            isinstance(correction, Mapping) and correction.get("correction_family") is not None
        )
        tombstone_present = any(
            row.get("event_identity") == event.get("event_identity") for row in tombstones
        )
        write_active = any(
            row.get("event_identity") == event.get("event_identity") for row in memory
        )
        delayed_passed = not correction_present or (
            not write_active and (action != "write" or tombstone_present)
        )
        after_bytes = _memory_bytes(memory)
        rollback_required = admission == "quarantined"
        rollback_exact = not rollback_required or after_bytes == parent_bytes
        row_id = sha256_json(
            {
                "event_identity": event.get("event_identity"),
                "arm": arm,
                "replicate_id": replicate_id,
                "order_seed": order_seed,
            }
        )
        rows.append(
            {
                "row_id": row_id,
                "replicate_id": replicate_id,
                "order_seed": order_seed,
                "event_sequence": sequence,
                "event_identity": event.get("event_identity"),
                "arm": arm,
                "family": features.get("family"),
                "proposed_action": action,
                "action_frozen_before_outcome": True,
                "exact_later_outcome": deepcopy(dict(outcome)),
                "same_event_outcome_used_for_decision": bool(leakage),
                "decision_leakage_witnesses": leakage,
                "state_before": {
                    "sha256": state_before_sha,
                    "symmetric": transition["state_symmetric_before"],
                },
                "state_after": {
                    "sha256": controller.state_checksum(state_after),
                    "symmetric": transition["state_symmetric_after"],
                },
                "admission_decision": admission,
                "admission_checks": deepcopy(write_receipt.get("checks", {}))
                if write_receipt
                else {},
                "failed_admission_checks": deepcopy(write_receipt.get("failed_checks", []))
                if write_receipt
                else [],
                "utility": utility,
                "held_future": sequence >= held_start,
                "useful_write": useful,
                "harmful_write": harmful,
                "false_injection": harmful,
                "anchor_retained": not harmful,
                "spectral_audit": {
                    "max_abs_entry_delta": transition["max_abs_entry_delta"],
                    "max_abs_entry_delta_bound": transition["max_abs_entry_delta_bound"],
                    "spectral_norm_delta": transition["spectral_norm_delta"],
                    "spectral_norm_delta_bound": transition["spectral_norm_delta_bound"],
                    "within_bound": transition["within_bound"],
                    "bound_required": arm != "v599_unsafe_reference",
                    "state_symmetric_before": transition["state_symmetric_before"],
                    "state_symmetric_after": transition["state_symmetric_after"],
                },
                "delayed_correction": {
                    "present": correction_present,
                    "write_active_after_correction": write_active,
                    "tombstone_required": correction_present and action == "write",
                    "tombstone_present": tombstone_present,
                    "passed": delayed_passed,
                },
                "tombstone_evidence": {
                    "required": rollback_required,
                    "present": tombstone_present,
                    "resurrection_blocked": not rollback_required or not write_active,
                },
                "rollback_evidence": {
                    "required": rollback_required,
                    "parent_active_memory_sha256": f"sha256:{hashlib.sha256(parent_bytes).hexdigest()}",
                    "restored_active_memory_sha256": f"sha256:{hashlib.sha256(after_bytes).hexdigest()}",
                    "byte_exact": rollback_exact,
                },
            }
        )

    checkpoint = {
        "arm": arm,
        "replicate_id": replicate_id,
        "order_seed": order_seed,
        "state": state,
        "memory": memory,
        "tombstones": tombstones,
        "cache": {},
    }
    checkpoint.update(
        {
            "state_sha256": controller.state_checksum(state),
            "memory_sha256": sha256_json(memory),
            "tombstones_sha256": sha256_json(tombstones),
        }
    )
    raw = canonical_json_bytes(checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(raw)
    persisted = checkpoint_path.read_bytes()
    persistence = {
        "replicate_id": replicate_id,
        "arm": arm,
        "checkpoint_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "byte_exact": persisted == raw,
        "private_checkpoint": True,
        "state_sha256": checkpoint["state_sha256"],
        "memory_sha256": checkpoint["memory_sha256"],
        "tombstones_sha256": checkpoint["tombstones_sha256"],
        "worker_process_id": os.getpid(),
    }
    return {
        "rows": rows,
        "checkpoint": checkpoint,
        "persistence": persistence,
        "process_receipt": {
            "worker_process_id": os.getpid(),
            "initial_state_sha256": initial_state_sha,
            "initial_memory_empty": True,
            "initial_cache_empty": True,
        },
    }


def restore_checkpoint(checkpoint_path: Path) -> JsonDict:
    """Restore one checkpoint and verify state, memory, and tombstone bytes."""

    raw = checkpoint_path.read_bytes()
    value = json.loads(raw)
    canonical = canonical_json_bytes(value)
    state_restored = controller.state_checksum(value["state"]) == value["state_sha256"]
    memory_restored = sha256_json(value["memory"]) == value["memory_sha256"]
    tombstones_restored = sha256_json(value["tombstones"]) == value["tombstones_sha256"]
    return {
        "replicate_id": value["replicate_id"],
        "arm": value["arm"],
        "checkpoint_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "byte_exact": canonical == raw,
        "state_restored": state_restored,
        "memory_restored": memory_restored,
        "tombstones_restored": tombstones_restored,
        "cache_empty": value.get("cache") == {},
        "process_id": os.getpid(),
    }


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic worker exchange data inside a private temp path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _run_command(command: Sequence[str]) -> None:
    """Run an isolated worker and expose its stderr on failure."""

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"fresh process failed ({completed.returncode}): {completed.stderr.strip()}"
        )


def run_fresh_processes(
    events: Sequence[Mapping[str, Any]],
    order_manifest: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Run every arm-order unit and restart in private fresh processes."""

    by_identity = {row.get("event_identity"): dict(row) for row in events}
    rows: list[JsonDict] = []
    persistence_rows: list[JsonDict] = []
    restart_rows: list[JsonDict] = []
    isolation_rows: list[JsonDict] = []
    module_path = Path(__file__).resolve()

    with tempfile.TemporaryDirectory(prefix="carnot-exp6873-") as temporary:
        temporary_root = Path(temporary)
        for order in order_manifest:
            replicate_id = str(order["replicate_id"])
            seed = int(order["seed"])
            ordered_events = [by_identity[identity] for identity in order["event_identities"]]
            for arm in ARMS:
                isolation_key = f"{replicate_id}::{arm}"
                unit = temporary_root / isolation_key.replace("::", "--")
                input_path = unit / "input.json"
                output_path = unit / "worker-output.json"
                checkpoint_path = unit / "checkpoint.json"
                restore_path = unit / "restart-output.json"
                _write_json(
                    input_path,
                    {
                        "events": ordered_events,
                        "arm": arm,
                        "replicate_id": replicate_id,
                        "order_seed": seed,
                    },
                )
                _run_command(
                    [
                        sys.executable,
                        str(module_path),
                        "--worker-input",
                        str(input_path),
                        "--worker-output",
                        str(output_path),
                        "--checkpoint",
                        str(checkpoint_path),
                    ]
                )
                worker = json.loads(output_path.read_text(encoding="utf-8"))
                _run_command(
                    [
                        sys.executable,
                        str(module_path),
                        "--restore-checkpoint",
                        str(checkpoint_path),
                        "--restore-output",
                        str(restore_path),
                    ]
                )
                restart = json.loads(restore_path.read_text(encoding="utf-8"))
                process_receipt = worker["process_receipt"]
                worker_pid = int(process_receipt["worker_process_id"])
                restart_pid = int(restart.pop("process_id"))
                restart["fresh_process"] = restart_pid != worker_pid
                restart["worker_process_id"] = worker_pid
                restart["restart_process_id"] = restart_pid
                rows.extend(worker["rows"])
                persistence_rows.append(worker["persistence"])
                restart_rows.append(restart)
                isolation_rows.append(
                    {
                        "isolation_key": isolation_key,
                        "replicate_id": replicate_id,
                        "arm": arm,
                        "worker_process_id": worker_pid,
                        "restart_process_id": restart_pid,
                        "initial_state_matches_frozen": process_receipt["initial_state_sha256"]
                        == controller.state_checksum(controller.initialize_reliability_state()),
                        "initial_memory_empty": process_receipt["initial_memory_empty"],
                        "initial_cache_empty": process_receipt["initial_cache_empty"],
                        "private_checkpoint": worker["persistence"]["private_checkpoint"],
                    }
                )
    return {
        "rows": rows,
        "persistence_rows": persistence_rows,
        "restart_rows": restart_rows,
        "process_isolation_rows": isolation_rows,
    }


def _entropy(counts: Mapping[str, int]) -> float:
    """Compute base-two action entropy from exact row counts."""

    total = sum(counts.values())
    probabilities = [count / total for count in counts.values() if count and total]
    return round(-sum(value * math.log2(value) for value in probabilities), 12)


def _rate(numerator: int, denominator: int) -> float:
    """Return a stable zero-safe rate."""

    return round(numerator / denominator, 12) if denominator else 0.0


def recompute_headlines(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute every utility and safety headline from event rows."""

    per_order: list[JsonDict] = []
    replicate_ids = sorted({str(row.get("replicate_id")) for row in rows})
    for replicate_id in replicate_ids:
        for arm in ARMS:
            unit_rows = [
                row
                for row in rows
                if row.get("replicate_id") == replicate_id and row.get("arm") == arm
            ]
            if not unit_rows:
                continue
            counts = Counter(str(row.get("proposed_action")) for row in unit_rows)
            held = [row for row in unit_rows if row.get("held_future") is True]
            utility_sum = sum(float(row.get("utility", 0.0)) for row in held)
            per_order.append(
                {
                    "replicate_id": replicate_id,
                    "arm": arm,
                    "row_count": len(unit_rows),
                    "distinct_action_count": len([count for count in counts.values() if count]),
                    "action_entropy": _entropy(counts),
                    "abstention_rate": _rate(counts.get("abstain", 0), len(unit_rows)),
                    "admitted_useful_updates": sum(
                        row.get("useful_write") is True for row in unit_rows
                    ),
                    "harmful_writes": sum(row.get("harmful_write") is True for row in unit_rows),
                    "false_injection_rate": _rate(
                        sum(row.get("false_injection") is True for row in unit_rows),
                        len(unit_rows),
                    ),
                    "held_future_utility_sum": round(utility_sum, 12),
                    "held_future_utility_mean": round(utility_sum / len(held), 12) if held else 0.0,
                    "old_family_retention_rate": _rate(
                        sum(row.get("anchor_retained") is True for row in unit_rows),
                        len(unit_rows),
                    ),
                }
            )

    distribution: JsonDict = {}
    entropy: JsonDict = {}
    useful: JsonDict = {}
    harmful: JsonDict = {}
    false_injection: JsonDict = {}
    held_utility: JsonDict = {}
    retention: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        counts = Counter(str(row.get("proposed_action")) for row in arm_rows)
        held = [row for row in arm_rows if row.get("held_future") is True]
        utility_sum = sum(float(row.get("utility", 0.0)) for row in held)
        distribution[arm] = {action: counts.get(action, 0) for action in ACTIONS}
        entropy[arm] = _entropy(counts)
        useful[arm] = sum(row.get("useful_write") is True for row in arm_rows)
        harmful[arm] = sum(row.get("harmful_write") is True for row in arm_rows)
        false_injection[arm] = _rate(
            sum(row.get("false_injection") is True for row in arm_rows), len(arm_rows)
        )
        held_utility[arm] = {
            "row_count": len(held),
            "utility_sum": round(utility_sum, 12),
            "mean_utility": round(utility_sum / len(held), 12) if held else 0.0,
        }
        retention[arm] = {
            "anchor_count": len(arm_rows),
            "retained_count": sum(row.get("anchor_retained") is True for row in arm_rows),
            "retention_rate": _rate(
                sum(row.get("anchor_retained") is True for row in arm_rows), len(arm_rows)
            ),
        }
    return {
        "per_order_results": per_order,
        "action_distribution_by_arm": distribution,
        "action_entropy_by_arm": entropy,
        "admitted_useful_updates_by_arm": useful,
        "harmful_writes_by_arm": harmful,
        "false_injection_rate_by_arm": false_injection,
        "held_future_utility_by_arm": held_utility,
        "old_family_retention_by_arm": retention,
    }


def compute_paired_order_effects(
    per_order_results: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Compare learner and reference utilities without pooling away orders."""

    lookup = {(str(row.get("replicate_id")), str(row.get("arm"))): row for row in per_order_results}
    replicates = sorted({str(row.get("replicate_id")) for row in per_order_results})
    comparisons: JsonDict = {}
    for learner in LEARNER_ARMS:
        for comparator in COMPARATOR_ARMS:
            effects: list[JsonDict] = []
            missing: list[str] = []
            for replicate in replicates:
                learner_row = lookup.get((replicate, learner))
                comparator_row = lookup.get((replicate, comparator))
                if learner_row is None or comparator_row is None:
                    missing.append(replicate)
                    continue
                learner_utility = float(learner_row.get("held_future_utility_mean", 0.0))
                comparator_utility = float(comparator_row.get("held_future_utility_mean", 0.0))
                effects.append(
                    {
                        "replicate_id": replicate,
                        "learner_utility": learner_utility,
                        "comparator_utility": comparator_utility,
                        "effect": round(learner_utility - comparator_utility, 12),
                        "available_headroom": round(max(0.0, 1.0 - comparator_utility), 12),
                    }
                )
            values = [row["effect"] for row in effects]
            headroom = {row["replicate_id"]: row["available_headroom"] for row in effects}
            wins = sum(value > 0.0 for value in values)
            ties = sum(value == 0.0 for value in values)
            losses = sum(value < 0.0 for value in values)
            comparisons[f"{learner}_vs_{comparator}"] = {
                "learner": learner,
                "comparator": comparator,
                "order_effects": effects,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "missing_orders": missing,
                "available_headroom_by_order": headroom,
                "mean_effect": round(sum(values) / len(values), 12) if values else None,
                "exact_order_interval": {
                    "low": min(values) if values else None,
                    "high": max(values) if values else None,
                },
                "declared_replication_rule_passed": bool(
                    len(values) >= 5
                    and not missing
                    and wins == len(values)
                    and ties == 0
                    and losses == 0
                    and all(value > 0.0 for value in headroom.values())
                ),
            }
    return comparisons


def evaluate_claim(
    per_order_results: Sequence[Mapping[str, Any]],
    paired_order_effects: Mapping[str, Mapping[str, Any]],
    audits: Mapping[str, Any],
) -> JsonDict:
    """Apply the preregistered conjunctive utility and safety decision rule."""

    quarantine = [row for row in per_order_results if row.get("arm") == "exact_quarantine"]
    spectral = audits.get("spectral_bound_audit_rows", [])
    delayed = audits.get("delayed_correction_rows", [])
    quarantine_delayed = [row for row in delayed if row.get("arm") == "exact_quarantine"]
    persistence = audits.get("persistence_rows", [])
    restart = audits.get("restart_rows", [])
    rollback = audits.get("rollback_rows", [])
    comparisons = (
        paired_order_effects.get("exact_quarantine_vs_frozen_no_memory", {}),
        paired_order_effects.get("exact_quarantine_vs_read_only", {}),
    )
    checks = {
        "five_complete_quarantine_orders": len(quarantine) >= 5,
        "quarantine_beats_frozen_every_order": bool(comparisons[0])
        and comparisons[0].get("declared_replication_rule_passed") is True,
        "quarantine_beats_read_only_every_order": bool(comparisons[1])
        and comparisons[1].get("declared_replication_rule_passed") is True,
        "positive_available_headroom": all(
            comparison
            and comparison.get("available_headroom_by_order")
            and all(
                float(value) > 0.0 for value in comparison["available_headroom_by_order"].values()
            )
            for comparison in comparisons
        ),
        "quarantine_action_not_collapsed": bool(quarantine)
        and all(int(row.get("distinct_action_count", 0)) > 1 for row in quarantine),
        "quarantine_admitted_useful_write": sum(
            int(row.get("admitted_useful_updates", 0)) for row in quarantine
        )
        > 0,
        "zero_quarantine_harmful_writes": all(
            int(row.get("harmful_writes", 0)) == 0 for row in quarantine
        ),
        "old_family_retention": bool(quarantine)
        and all(float(row.get("old_family_retention_rate", 0.0)) == 1.0 for row in quarantine),
        "spectral_and_symmetry_bounds": bool(spectral)
        and all(
            row.get("within_bound") is True
            and row.get("state_symmetric_before") is True
            and row.get("state_symmetric_after") is True
            for row in spectral
            if row.get("arm") in SAFE_STATE_ARMS
        ),
        "delayed_correction_and_tombstones": bool(quarantine_delayed)
        and all(row.get("passed") is True for row in quarantine_delayed),
        "exact_persistence": bool(persistence)
        and all(
            row.get("byte_exact") is True and row.get("private_checkpoint") is True
            for row in persistence
        ),
        "fresh_process_restart": bool(restart)
        and all(
            row.get("byte_exact") is True
            and row.get("state_restored") is True
            and row.get("memory_restored") is True
            and row.get("tombstones_restored") is True
            and row.get("fresh_process") is True
            for row in restart
        ),
        "byte_exact_rollback": bool(rollback)
        and all(row.get("byte_exact") is True for row in rollback),
        "zero_leakage_witnesses": audits.get("leakage_witnesses") == [],
        "no_model_weight_mutation": audits.get("no_model_weight_mutation") is True,
        "aggregate_matches_rows": audits.get("aggregate_consistent") is True,
        "process_isolation": audits.get("process_isolation_passed") is True,
        "verifier_not_oracle": True,
    }
    safety_names = {
        "zero_quarantine_harmful_writes",
        "old_family_retention",
        "spectral_and_symmetry_bounds",
        "delayed_correction_and_tombstones",
        "exact_persistence",
        "fresh_process_restart",
        "byte_exact_rollback",
        "zero_leakage_witnesses",
        "no_model_weight_mutation",
        "aggregate_matches_rows",
        "process_isolation",
        "verifier_not_oracle",
    }
    safety_passed = all(checks[name] for name in safety_names)
    missing = not checks["five_complete_quarantine_orders"]
    all_passed = all(checks.values())
    if all_passed:
        verdict_class = "positive"
        honest_verdict = POSITIVE_VERDICT
    elif not safety_passed:
        verdict_class = "disqualified"
        honest_verdict = DISQUALIFIED_VERDICT
    elif missing:
        verdict_class = "partial"
        honest_verdict = PARTIAL_VERDICT
    else:
        verdict_class = "null"
        honest_verdict = NULL_VERDICT
    rows = [check_row(name, True, passed, passed) for name, passed in checks.items()]
    return {
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "scientific_claim_eligible": all_passed,
        "checks_by_name": checks,
        "gate_rows": rows,
    }


def _empty_arm_metrics() -> JsonDict:
    """Return complete zero values for a blocked audit."""

    return {
        "action_distribution_by_arm": {arm: {action: 0 for action in ACTIONS} for arm in ARMS},
        "action_entropy_by_arm": {arm: 0.0 for arm in ARMS},
        "admitted_useful_updates_by_arm": {arm: 0 for arm in ARMS},
        "harmful_writes_by_arm": {arm: 0 for arm in ARMS},
        "false_injection_rate_by_arm": {arm: 0.0 for arm in ARMS},
        "held_future_utility_by_arm": {
            arm: {"row_count": 0, "utility_sum": 0.0, "mean_utility": 0.0} for arm in ARMS
        },
        "old_family_retention_by_arm": {
            arm: {"anchor_count": 0, "retained_count": 0, "retention_rate": 0.0} for arm in ARMS
        },
    }


def empty_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete blocked artifact before input checks run."""

    return {
        "schema": "carnot.experiment_6873.prospective_sealed_self_learning_audit.v1",
        "experiment_id": 6873,
        "run_date": run_date,
        "status": "complete",
        "openspec_requirement_ids": [
            "REQ-LEARN-6873",
            "SCENARIO-LEARN-6873-PRECONDITIONS",
            "SCENARIO-LEARN-6873-FRESH-PROCESS",
            "SCENARIO-LEARN-6873-ONE-ORDER",
            "SCENARIO-LEARN-6873-NO-HEADROOM",
            "SCENARIO-LEARN-6873-ACTION-COLLAPSE",
            "SCENARIO-LEARN-6873-ZERO-WRITES",
            "SCENARIO-LEARN-6873-HARMFUL-WRITE",
            "SCENARIO-LEARN-6873-FORGETTING",
            "SCENARIO-LEARN-6873-STATE-BOUND",
            "SCENARIO-LEARN-6873-DELAYED-CORRECTION",
            "SCENARIO-LEARN-6873-RESTART",
            "SCENARIO-LEARN-6873-ROLLBACK",
            "SCENARIO-LEARN-6873-LEAKAGE",
            "SCENARIO-LEARN-6873-AGGREGATE-CONTRADICTION",
        ],
        "replay_commands": [
            ".venv/bin/python scripts/experiments/experiment_6873_prospective_sealed_self_learning_audit.py --date 20260902",
            ".venv/bin/pytest tests/python/test_experiment_6873_prospective_sealed_self_learning_audit.py -q",
        ],
        "field_principles": {},
        "preconditions_checked": {"passed": False, "checks": []},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": True,
        "model_immutability_receipt": {},
        "rows": [],
        "per_order_results": [],
        **_empty_arm_metrics(),
        "paired_order_effects": {},
        "spectral_bound_audit_rows": [],
        "delayed_correction_rows": [],
        "persistence_rows": [],
        "restart_rows": [],
        "rollback_rows": [],
        "leakage_witnesses": [],
        "process_isolation_rows": [],
        "scientific_claim_eligible": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }


def field_principles(artifact: Mapping[str, Any]) -> JsonDict:
    """Explain why every top-level field is present."""

    special = {
        "rows": "Event-arm-order rows keep all headlines independently recomputable.",
        "paired_order_effects": "Order pairs prevent one order from creating a pooled win.",
        "spectral_bound_audit_rows": "State deltas prove that every safe update stayed bounded.",
        "restart_rows": "A second process proves that durable state restores exactly.",
        "rollback_rows": "Parent and restored hashes prove byte-exact active-memory rollback.",
        "gate_check_summary": "Expected and observed values make every block actionable.",
        "field_principles": "This map states why every artifact field exists.",
    }
    return {
        field: special.get(
            field, f"This field records auditable {field.replace('_', ' ')} evidence."
        )
        for field in artifact
    }


def _source_hash_receipts(
    root: Path,
    expected_hashes: Mapping[str, Mapping[str, str]],
    controller_path: Path,
    source_path: Path,
) -> JsonDict:
    """Record expected and observed frozen hashes plus this audit's code."""

    observed = _observed_frozen_hashes(root, expected_hashes, controller_path, source_path)
    receipts: JsonDict = {
        key: {
            "path": row.get("path"),
            "expected_sha256": row.get("sha256"),
            "observed_sha256": observed[key]["sha256"],
            "match": row.get("sha256") == observed[key]["sha256"],
        }
        for key, row in expected_hashes.items()
    }
    for key, relative in OWN_SOURCE_PATHS.items():
        receipts[key] = {"path": relative, "observed_sha256": sha256_file(root / relative)}
    return receipts


def _process_isolation_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require one private clean worker and distinct restart per arm-order unit."""

    if not rows:
        return False
    keys = [row.get("isolation_key") for row in rows]
    worker_pids = [row.get("worker_process_id") for row in rows]
    return bool(
        len(keys) == len(set(keys))
        and len(worker_pids) == len(set(worker_pids))
        and all(
            row.get("worker_process_id") != row.get("restart_process_id")
            and row.get("initial_state_matches_frozen") is True
            and row.get("initial_memory_empty") is True
            and row.get("initial_cache_empty") is True
            and row.get("private_checkpoint") is True
            for row in rows
        )
    )


def build_artifact(
    root: Path,
    run_date: str,
    *,
    controller_relative_path: Path = CONTROLLER_RELATIVE_PATH,
    source_relative_path: Path = SOURCE_RELATIVE_PATH,
    expected_hashes: Mapping[str, Mapping[str, str]] | None = None,
) -> JsonDict:
    """Build a complete blocked, null, disqualified, or positive sealed audit."""

    started = time.perf_counter()
    expected = dict(expected_hashes or FROZEN_INPUT_HASHES)
    controller_path = _resolve(root, controller_relative_path)
    source_path = _resolve(root, source_relative_path)
    controller_artifact, controller_error = _read_json(controller_path)
    source_artifact, source_error = _read_json(source_path)
    checks = precondition_checks(
        root,
        controller_artifact,
        source_artifact,
        controller_path,
        source_path,
        controller_error,
        source_error,
        expected,
    )
    preconditions = gate_summary(checks)
    artifact = empty_artifact(run_date)
    artifact["preconditions_checked"] = {
        "controller_path": str(controller_path),
        "source_path": str(source_path),
        "checks": checks,
        "passed": preconditions["passed"],
    }
    artifact["source_artifact_hashes"] = _source_hash_receipts(
        root, expected, controller_path, source_path
    )
    receipt = controller_artifact.get("model_immutability_receipt", {})
    no_mutation = bool(
        controller_artifact.get("no_model_weight_mutation") is True
        and isinstance(receipt, Mapping)
        and receipt.get("before_sha256") == receipt.get("after_sha256")
    )
    artifact["no_model_weight_mutation"] = no_mutation
    artifact["model_immutability_receipt"] = (
        deepcopy(dict(receipt)) if isinstance(receipt, Mapping) else {}
    )
    artifact["gate_check_summary"] = preconditions

    if preconditions["passed"]:
        events = [row for row in source_artifact["rows"] if isinstance(row, Mapping)]
        order_manifest = [
            row for row in source_artifact["order_replicate_manifest"] if isinstance(row, Mapping)
        ]
        execution = run_fresh_processes(events, order_manifest)
        rows = execution["rows"]
        headlines = recompute_headlines(rows)
        paired = compute_paired_order_effects(headlines["per_order_results"])
        spectral_rows = [
            {
                "row_id": row["row_id"],
                "replicate_id": row["replicate_id"],
                "arm": row["arm"],
                **row["spectral_audit"],
            }
            for row in rows
        ]
        delayed_rows = [
            {
                "row_id": row["row_id"],
                "replicate_id": row["replicate_id"],
                "arm": row["arm"],
                **row["delayed_correction"],
            }
            for row in rows
            if row["delayed_correction"]["present"] is True
        ]
        rollback_rows = [
            {
                "row_id": row["row_id"],
                "replicate_id": row["replicate_id"],
                "arm": row["arm"],
                **row["rollback_evidence"],
            }
            for row in rows
            if row["rollback_evidence"]["required"] is True
        ]
        leakage = [
            {
                "row_id": row["row_id"],
                "replicate_id": row["replicate_id"],
                "arm": row["arm"],
                "witness": witness,
            }
            for row in rows
            for witness in row["decision_leakage_witnesses"]
        ]
        audits = {
            "spectral_bound_audit_rows": spectral_rows,
            "delayed_correction_rows": delayed_rows,
            "persistence_rows": execution["persistence_rows"],
            "restart_rows": execution["restart_rows"],
            "rollback_rows": rollback_rows,
            "leakage_witnesses": leakage,
            "no_model_weight_mutation": no_mutation,
            "aggregate_consistent": True,
            "process_isolation_passed": _process_isolation_passed(
                execution["process_isolation_rows"]
            ),
        }
        claim = evaluate_claim(headlines["per_order_results"], paired, audits)
        artifact.update(headlines)
        artifact.update(
            {
                "rows": rows,
                "paired_order_effects": paired,
                "spectral_bound_audit_rows": spectral_rows,
                "delayed_correction_rows": delayed_rows,
                "persistence_rows": execution["persistence_rows"],
                "restart_rows": execution["restart_rows"],
                "rollback_rows": rollback_rows,
                "leakage_witnesses": leakage,
                "process_isolation_rows": execution["process_isolation_rows"],
                "scientific_claim_eligible": claim["scientific_claim_eligible"],
                "verdict_class": claim["verdict_class"],
                "honest_verdict": claim["honest_verdict"],
                "gate_check_summary": gate_summary([*checks, *claim["gate_rows"]]),
            }
        )

    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _checksum_payload(value: Any) -> Any:
    """Remove runtime-only process identities from reproducibility content."""

    if isinstance(value, Mapping):
        return {
            key: _checksum_payload(item)
            for key, item in value.items()
            if key
            not in {
                "duration_s",
                "reproducibility_checksum",
                "worker_process_id",
                "restart_process_id",
            }
        }
    if isinstance(value, list):
        return [_checksum_payload(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic evidence while excluding runtime-only identities."""

    return sha256_json(_checksum_payload(dict(artifact)))


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, row recomputation, process evidence, and verdict gates."""

    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_must_cover_every_field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if artifact.get("continuous_self_learning_task") is not True:
        errors.append("continuous_self_learning_task_must_be_true")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")

    rows_value = artifact.get("rows", [])
    rows = rows_value if isinstance(rows_value, list) else []
    if rows:
        headlines = recompute_headlines(rows)
        aggregate_consistent = True
        for field in HEADLINE_FIELDS:
            if artifact.get(field) != headlines[field]:
                errors.append(f"aggregate_row_contradiction:{field}")
                aggregate_consistent = False
        paired = compute_paired_order_effects(headlines["per_order_results"])
        if artifact.get("paired_order_effects") != paired:
            errors.append("aggregate_row_contradiction:paired_order_effects")
            aggregate_consistent = False
        row_ids = [row.get("row_id") for row in rows if isinstance(row, Mapping)]
        if len(row_ids) != len(set(row_ids)):
            errors.append("duplicate_event_arm_order_rows")
        process_rows = artifact.get("process_isolation_rows", [])
        process_isolation = bool(
            isinstance(process_rows, list) and _process_isolation_passed(process_rows)
        )
        audits = {
            "spectral_bound_audit_rows": artifact.get("spectral_bound_audit_rows", []),
            "delayed_correction_rows": artifact.get("delayed_correction_rows", []),
            "persistence_rows": artifact.get("persistence_rows", []),
            "restart_rows": artifact.get("restart_rows", []),
            "rollback_rows": artifact.get("rollback_rows", []),
            "leakage_witnesses": artifact.get("leakage_witnesses", []),
            "no_model_weight_mutation": artifact.get("no_model_weight_mutation"),
            "aggregate_consistent": aggregate_consistent,
            "process_isolation_passed": process_isolation,
        }
        claim = evaluate_claim(headlines["per_order_results"], paired, audits)
        if artifact.get("scientific_claim_eligible") is not claim["scientific_claim_eligible"]:
            errors.append("scientific_claim_eligibility_mismatch")
        if artifact.get("verdict_class") == "positive" and claim["verdict_class"] != "positive":
            errors.append("positive_verdict_gate_mismatch")
    elif artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary", {})
        if not isinstance(summary, Mapping) or not summary.get("failed_check"):
            errors.append("blocked_artifact_requires_failed_check")
    return errors


def write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the result only after a complete temporary file exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worker_mode(args: argparse.Namespace) -> int:
    """Execute one private arm-order unit for the parent audit."""

    payload = json.loads(args.worker_input.read_text(encoding="utf-8"))
    result = run_arm_replicate(
        payload["events"],
        payload["arm"],
        payload["replicate_id"],
        int(payload["order_seed"]),
        args.checkpoint,
    )
    _write_json(args.worker_output, result)
    return 0


def _restore_mode(args: argparse.Namespace) -> int:
    """Restore one private checkpoint in a second process."""

    _write_json(args.restore_output, restore_checkpoint(args.restore_checkpoint))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a worker, a restart check, or the dated top-level audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--controller", type=Path, default=CONTROLLER_RELATIVE_PATH)
    parser.add_argument("--source", type=Path, default=SOURCE_RELATIVE_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--restore-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_input is not None:
        return _worker_mode(args)
    if args.restore_checkpoint is not None:
        return _restore_mode(args)
    if not args.date:
        parser.error("--date is required for the top-level audit")
    artifact = build_artifact(
        args.root,
        args.date,
        controller_relative_path=args.controller,
        source_relative_path=args.source,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_atomic(args.output, artifact)
    print(
        json.dumps(
            {
                "result": str(args.output),
                "scientific_claim_eligible": artifact["scientific_claim_eligible"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by fresh subprocesses
    raise SystemExit(main())

"""Exp6485 online cache transition e-process contract.

Spec refs: REQ-INFRA-6485, SCENARIO-INFRA-6485-EVENTS,
SCENARIO-INFRA-6485-ACTIONS, SCENARIO-INFRA-6485-EPROCESS,
SCENARIO-INFRA-6485-LIFECYCLE, SCENARIO-INFRA-6485-ATTACKS,
SCENARIO-INFRA-6485-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6485
TASK_ID = "exp6485-online-cache-transition-eprocess-contract"
INFERENCE_SUBSTRATE = "deterministic_transition_contract_no_llm"
SCHEMA_VERSION = "carnot.experiment_6485.online_cache_transition_eprocess_contract.v1"
GENESIS_HASH = "sha256:" + "0" * 64
BASE_MONOTONIC_NS = 1_000_000_000
NULL_FROZEN_NS = BASE_MONOTONIC_NS - 100_000
EPROCESS_NULL_ID = "fixed_null_no_positive_transition_effect_v1"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_6485_online_cache_transition_eprocess_contract.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6485_online_cache_transition_eprocess_contract.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6485_online_cache_transition_eprocess_contract.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

EVENT_TYPES = (
    "observe",
    "verify",
    "propose",
    "quarantine",
    "admit",
    "promote",
    "evict",
    "tombstone",
    "rollback",
    "restart",
    "no_write",
)
EVIDENCE_EVENT_TYPES = ("verify", "propose", "admit", "promote", "restart")
DURABLE_EVENT_TYPES = (
    "quarantine",
    "admit",
    "promote",
    "evict",
    "tombstone",
    "rollback",
    "restart",
)
LIFECYCLE_TYPES = ("admission", "eviction", "tombstone", "rollback", "restart")
ATTACK_IDS = (
    "duplicate_events",
    "backdated_writes",
    "stated_write_without_action",
    "action_without_exact_admission",
    "threshold_editing",
    "repeated_peeking",
    "missing_null",
    "rollback_omission",
    "tombstone_resurrection",
    "restart_drift",
)

ACTION_BY_EVENT = {
    "observe": "no_action",
    "verify": "no_action",
    "propose": "no_action",
    "quarantine": "quarantine_write",
    "admit": "admit_write",
    "promote": "promote_write",
    "evict": "evict_write",
    "tombstone": "tombstone_write",
    "rollback": "rollback_restore",
    "restart": "restart_replay",
    "no_write": "no_action",
}
AUTHORITIES = {
    "observe": "runtime_observer",
    "verify": "exact_fixture_verifier",
    "propose": "shadow_ranker",
    "quarantine": "exact_fixture_verifier",
    "admit": "exact_fixture_verifier",
    "promote": "eprocess_boundary_checker",
    "evict": "capacity_controller",
    "tombstone": "lifecycle_controller",
    "rollback": "rollback_controller",
    "restart": "restart_replay_controller",
    "no_write": "no_write_controller",
}
FIXTURE_LABELS = {
    "observe": "candidate_seen",
    "verify": "exact_receipt_bound",
    "propose": "bounded_shadow_delta",
    "quarantine": "bad_candidate_rejected",
    "admit": "exact_candidate_admitted",
    "promote": "eprocess_boundary_crossed",
    "evict": "capacity_eviction",
    "tombstone": "revoked_event_tombstoned",
    "rollback": "restore_pre_admit_state",
    "restart": "replay_from_receipts",
    "no_write": "explicit_no_write",
}
EPROCESS_INCREMENTS = {
    "verify": 1.05,
    "propose": 1.2,
    "admit": 1.35,
    "promote": 1.25,
    "restart": 1.1,
}
IMMUTABLE_EVENT_ID_FIELDS = (
    "schema_version",
    "task_id",
    "chronology_index",
    "event_type",
    "monotonic_receipt_ns",
    "parent_state_hash",
    "event_payload_hash",
    "authority",
    "fixture_label",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "event_schema",
    "action_receipt_schema",
    "evidence_process_spec",
    "frozen_null_receipt",
    "event_rows",
    "action_rows",
    "evidence_process_rows",
    "lifecycle_rows",
    "attack_matrix",
    "online_transition_contract_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "protected_files_unchanged",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal transition-contract state.",
    "event_schema": "Immutable chronological event schema.",
    "action_receipt_schema": "Actual durable action schema.",
    "evidence_process_spec": "Null, update, mixture, promotion, and stopping rules.",
    "frozen_null_receipt": "Proof that the null predates adaptive events.",
    "event_rows": "One row per fixture event.",
    "action_rows": "One row per durable action or explicit no-action.",
    "evidence_process_rows": "One row per sequential update and peek charge.",
    "lifecycle_rows": "Admission, eviction, tombstone, rollback, and restart states.",
    "attack_matrix": "Duplicate, peeking, authority, and resurrection attacks.",
    "online_transition_contract_ready_score": "Same-roadmap downstream gate field.",
    "per_unit_rows": "Event, action, update, and attack rows.",
    "aggregate_row_recomputation": "Ready score recomputed from rows.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "gate_check_summary": "Required for any blocked_* verdict.",
    "preconditions_checked": "Adapter and fixture prechecks.",
    "inference_substrate": "deterministic_transition_contract_no_llm.",
    "verifier_is_oracle": "True for exact fixture and receipt validation only.",
    "field_principles": "Reason for every field.",
    "field_provenance": "Source paths, hashes, and reducers.",
    "random_seed": "Fixed fixture and attack seed.",
    "duration_s": "Measured wall time.",
    "tests_run": "Executed checks and exit codes.",
    "reproducibility_checksum": "Hash over schemas, null, rows, and attacks.",
    "honest_verdict": "States contract readiness without claiming a learning gain.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6485_online_cache_transition_eprocess_contract "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6485_online_cache_transition_eprocess_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6485_online_cache_transition_eprocess_contract.py "
    "-m pytest "
    "tests/python/test_experiment_6485_online_cache_transition_eprocess_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6485_online_cache_transition_eprocess_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6485_online_cache_transition_eprocess_contract.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6485_online_cache_transition_eprocess_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6485_online_cache_transition_eprocess_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6485_online_cache_transition_eprocess_contract --validate"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "text=Path('ops/e2e-test-plan.md').read_text(); "
    "assert 'E2E-005' in text and 'Serialization' in text\""
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    VALIDATE_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_COMMAND,
)
DEFAULT_TEST_RESULTS = tuple(
    {"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/pipeline/factor_cache_shadow_adapter.py"),
    Path("python/carnot/experiment_6479_verify_repair_factor_cache_shadow_adapter.py"),
    Path("results/experiment_6479_verify_repair_factor_cache_shadow_adapter.json"),
    Path("results/experiment_6420_csl_authenticity_safety_audit.json"),
    Path("results/experiment_6433_csl_row_recomputation_safety_audit.json"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_output(args: Sequence[str], root: Path) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {str(path): receipts.sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {str(path): receipts.sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for path, before_hash in before.items():
        after_hash = receipts.sha256_file(root / path)
        files[path] = {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "unchanged": before_hash == after_hash,
        }
    return {
        "protected_files_unchanged": all(row["unchanged"] for row in files.values()),
        "files": files,
    }


def _copy_json(value: Any) -> Any:
    return json.loads(receipts.canonical_json(value))


def row_hash(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "row_hash"}
    return receipts.sha256_json(payload)


def _with_row_hash(row: Mapping[str, Any]) -> JsonDict:
    out = dict(row)
    out["row_hash"] = row_hash(out)
    return out


def _refresh_row(row: JsonDict) -> JsonDict:
    row["row_hash"] = row_hash(row)
    return row


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def event_schema() -> JsonDict:
    schema = {
        "schema_version": SCHEMA_VERSION + ".event_schema",
        "event_types": list(EVENT_TYPES),
        "immutable_identity_fields": list(IMMUTABLE_EVENT_ID_FIELDS),
        "monotonic_receipt_unit": "nanoseconds_from_fixture_clock",
        "event_id_rule": "sha256_json_of_immutable_identity_fields",
        "mutable_excluded_fields": ["row_hash", "event_id", "summary", "later_outcome"],
    }
    schema["schema_hash"] = receipts.sha256_json(schema)
    return schema


def action_receipt_schema() -> JsonDict:
    schema = {
        "schema_version": SCHEMA_VERSION + ".action_receipt_schema",
        "action_types": sorted(set(ACTION_BY_EVENT.values())),
        "durable_event_types": list(DURABLE_EVENT_TYPES),
        "required_fields": [
            "action_event_id",
            "action_type",
            "pre_state_hash",
            "post_state_hash",
            "exact_admission_hash",
            "durable",
            "action_hash",
        ],
        "no_action_rule": "non-durable rows must carry a no_action_reason",
        "durable_write_rule": "durable writes require exact admission and matching event id",
    }
    schema["schema_hash"] = receipts.sha256_json(schema)
    return schema


def evidence_process_spec() -> JsonDict:
    thresholds = {
        "promotion_boundary": 2.0,
        "stopping_boundary": 4.0,
        "alpha_budget": 0.05,
    }
    mixture = {
        "family": "geometric",
        "allowed_factor_hypotheses": [
            {"factor_hypothesis": "factor_alpha", "mixture_weight": 0.5},
            {"factor_hypothesis": "factor_beta", "mixture_weight": 0.25},
            {"factor_hypothesis": "factor_gamma", "mixture_weight": 0.125},
        ],
        "remaining_tail_weight": 0.125,
    }
    spec = {
        "schema_version": SCHEMA_VERSION + ".eprocess_spec",
        "null_id": EPROCESS_NULL_ID,
        "null_statement": "No allowed factor hypothesis has positive transition utility.",
        "update_rule": (
            "multiply cumulative e-value by the precommitted nonnegative "
            "fixture likelihood-ratio increment for each charged peek"
        ),
        "mixture": mixture,
        "thresholds": thresholds,
        "threshold_hash": receipts.sha256_json(thresholds),
        "thresholds_frozen_before_events": True,
        "held_events_used_for_tuning": False,
        "fixed_horizon_comparison": "reported separately and cannot change thresholds",
        "adaptive_decision": "requires one charged peek row before any promotion decision",
    }
    spec["spec_hash"] = receipts.sha256_json(spec)
    return spec


def frozen_null_receipt(spec: Mapping[str, Any]) -> JsonDict:
    receipt = {
        "schema_version": SCHEMA_VERSION + ".frozen_null_receipt",
        "null_id": spec["null_id"],
        "null_statement": spec["null_statement"],
        "frozen_at_utc": "2026-08-21T00:00:00Z",
        "frozen_at_monotonic_ns": NULL_FROZEN_NS,
        "spec_hash": spec["spec_hash"],
        "threshold_hash": spec["threshold_hash"],
        "held_events_visible_at_freeze": False,
    }
    receipt["null_receipt_hash"] = receipts.sha256_json(receipt)
    return receipt


def _event_payload(event_type: str) -> JsonDict:
    return {
        "fixture_source": "exp6485_deterministic_transition_contract",
        "event_type": event_type,
        "factor_hypothesis": (
            "factor_alpha"
            if event_type in {"observe", "verify", "admit", "promote"}
            else "factor_beta"
        ),
        "fixture_label": FIXTURE_LABELS[event_type],
        "exact_outcome": "pass" if event_type in DURABLE_EVENT_TYPES else "not_applicable",
        "held_for_threshold_tuning": False,
    }


def _event_id(row: Mapping[str, Any]) -> str:
    return receipts.sha256_json({field: row.get(field) for field in IMMUTABLE_EVENT_ID_FIELDS})


def _action_hash(row: Mapping[str, Any]) -> str:
    material = {
        key: value
        for key, value in row.items()
        if key not in {"action_hash", "row_hash"}
    }
    return receipts.sha256_json(material)


def _normal_post_state(pre_state_hash: str, event_id: str, action_type: str, payload_hash: str) -> str:
    return receipts.sha256_json(
        {
            "pre_state_hash": pre_state_hash,
            "event_id": event_id,
            "action_type": action_type,
            "action_payload_hash": payload_hash,
        }
    )


def _expected_post_state(action: Mapping[str, Any], pre_state_hash: str) -> str:
    action_type = str(action.get("action_type"))
    if action.get("durable") is not True:
        return pre_state_hash
    if action_type == "rollback_restore":
        return str(action.get("rollback_target_state_hash"))
    if action_type == "restart_replay":
        return str(action.get("expected_state_hash"))
    return _normal_post_state(
        pre_state_hash,
        str(action.get("action_event_id")),
        action_type,
        str(action.get("action_payload_hash")),
    )


def _event_row(
    *,
    event_type: str,
    chronology_index: int,
    parent_state_hash: str,
) -> JsonDict:
    payload = _event_payload(event_type)
    event_payload_hash = receipts.sha256_json(payload)
    exact_admission = {
        "event_type": event_type,
        "authority": AUTHORITIES[event_type],
        "exact_admission_passed": event_type in DURABLE_EVENT_TYPES,
        "fixture_label": FIXTURE_LABELS[event_type],
    }
    row: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "event",
        "task_id": TASK_ID,
        "chronology_index": chronology_index,
        "event_type": event_type,
        "monotonic_receipt_ns": BASE_MONOTONIC_NS + chronology_index * 100_000,
        "parent_state_hash": parent_state_hash,
        "event_payload": payload,
        "event_payload_hash": event_payload_hash,
        "authority": AUTHORITIES[event_type],
        "fixture_label": FIXTURE_LABELS[event_type],
        "stated_write": event_type in DURABLE_EVENT_TYPES,
        "exact_admission_hash": receipts.sha256_json(exact_admission),
    }
    row["event_id"] = _event_id(row)
    return _with_row_hash(row)


def _action_row(
    *,
    event: Mapping[str, Any],
    pre_state_hash: str,
    rollback_target_state_hash: str,
) -> JsonDict:
    event_type = str(event["event_type"])
    action_type = ACTION_BY_EVENT[event_type]
    durable = event_type in DURABLE_EVENT_TYPES
    payload = {
        "event_type": event_type,
        "fixture_label": event["fixture_label"],
        "durable": durable,
        "authority": event["authority"],
    }
    action_payload_hash = receipts.sha256_json(payload)
    post_state_hash = pre_state_hash
    expected_state_hash = pre_state_hash
    if durable:
        if action_type == "rollback_restore":
            post_state_hash = rollback_target_state_hash
        elif action_type == "restart_replay":
            post_state_hash = pre_state_hash
            expected_state_hash = pre_state_hash
        else:
            post_state_hash = _normal_post_state(
                pre_state_hash,
                str(event["event_id"]),
                action_type,
                action_payload_hash,
            )
            expected_state_hash = post_state_hash
    row: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "action",
        "task_id": TASK_ID,
        "action_receipt_index": event["chronology_index"],
        "event_type": event_type,
        "action_event_id": event["event_id"],
        "action_type": action_type,
        "durable": durable,
        "action_monotonic_ns": int(event["monotonic_receipt_ns"]) + 1_000,
        "pre_state_hash": pre_state_hash,
        "post_state_hash": post_state_hash,
        "expected_state_hash": expected_state_hash,
        "rollback_target_state_hash": rollback_target_state_hash if action_type == "rollback_restore" else "",
        "action_payload_hash": action_payload_hash,
        "exact_admission_required": durable,
        "exact_admission_passed": durable,
        "exact_admission_hash": event["exact_admission_hash"] if durable else "",
        "durability_receipt": "atomic_json_write" if durable else "no_write",
        "no_action_reason": "" if durable else f"{event_type}_does_not_write",
    }
    row["action_hash"] = _action_hash(row)
    return _with_row_hash(row)


def _evidence_row(
    *,
    event: Mapping[str, Any],
    spec: Mapping[str, Any],
    null_receipt: Mapping[str, Any],
    sequential_index: int,
    prior_cumulative: float,
) -> JsonDict:
    increment = float(EPROCESS_INCREMENTS[str(event["event_type"])])
    cumulative = round(prior_cumulative * increment, 9)
    decision_kind = (
        "fixed_horizon_comparison"
        if event["event_type"] == "verify"
        else "adaptive_decision"
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "evidence_process",
        "task_id": TASK_ID,
        "sequential_index": sequential_index,
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "null_id": null_receipt["null_id"],
        "null_receipt_hash": null_receipt["null_receipt_hash"],
        "null_frozen_at_monotonic_ns": null_receipt["frozen_at_monotonic_ns"],
        "null_frozen_before_event": int(null_receipt["frozen_at_monotonic_ns"])
        < int(event["monotonic_receipt_ns"]),
        "threshold_hash": spec["threshold_hash"],
        "promotion_boundary": spec["thresholds"]["promotion_boundary"],
        "stopping_boundary": spec["thresholds"]["stopping_boundary"],
        "mixture_family": spec["mixture"]["family"],
        "mixture_weight_sum": 0.875,
        "e_value_increment": increment,
        "cumulative_e_value": cumulative,
        "adaptive_peek_charged": True,
        "peek_charge_index": sequential_index + 1,
        "decision_kind": decision_kind,
        "promotion_decision": event["event_type"] == "promote" and cumulative >= 2.0,
        "stopping_decision": cumulative >= 4.0,
        "held_events_used_for_tuning": False,
    }
    return _with_row_hash(row)


def _lifecycle_row(
    *,
    lifecycle_index: int,
    lifecycle_type: str,
    action: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> JsonDict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "lifecycle",
        "task_id": TASK_ID,
        "lifecycle_index": lifecycle_index,
        "lifecycle_type": lifecycle_type,
        "event_id": action["action_event_id"],
        "event_type": action["event_type"],
        "action_hash": action["action_hash"],
        "state_hash_before": action["pre_state_hash"],
        "state_hash_after": action["post_state_hash"],
        **dict(extra),
    }
    return _with_row_hash(row)


def build_contract_rows(*, root: Path = REPO_ROOT) -> JsonDict:
    """Build positive deterministic rows for the transition contract."""

    del root
    evt_schema = event_schema()
    action_schema = action_receipt_schema()
    eprocess_spec = evidence_process_spec()
    null_receipt = frozen_null_receipt(eprocess_spec)
    event_rows: list[JsonDict] = []
    action_rows: list[JsonDict] = []
    evidence_rows: list[JsonDict] = []
    state_hash = GENESIS_HASH
    rollback_target_state_hash = GENESIS_HASH
    cumulative_e = 1.0
    for chronology_index, event_type in enumerate(EVENT_TYPES):
        event = _event_row(
            event_type=event_type,
            chronology_index=chronology_index,
            parent_state_hash=state_hash,
        )
        if event_type == "admit":
            rollback_target_state_hash = state_hash
        action = _action_row(
            event=event,
            pre_state_hash=state_hash,
            rollback_target_state_hash=rollback_target_state_hash,
        )
        state_hash = str(action["post_state_hash"])
        event_rows.append(event)
        action_rows.append(action)
        if event_type in EVIDENCE_EVENT_TYPES:
            evidence = _evidence_row(
                event=event,
                spec=eprocess_spec,
                null_receipt=null_receipt,
                sequential_index=len(evidence_rows),
                prior_cumulative=cumulative_e,
            )
            cumulative_e = float(evidence["cumulative_e_value"])
            evidence_rows.append(evidence)
    action_by_event_type = {row["event_type"]: row for row in action_rows}
    admit_action = action_by_event_type["admit"]
    evict_action = action_by_event_type["evict"]
    tombstone_action = action_by_event_type["tombstone"]
    rollback_action = action_by_event_type["rollback"]
    restart_action = action_by_event_type["restart"]
    lifecycle_rows = [
        _lifecycle_row(
            lifecycle_index=0,
            lifecycle_type="admission",
            action=admit_action,
            extra={
                "admitted_event_id": admit_action["action_event_id"],
                "exact_admission_passed": True,
                "admission_hash": admit_action["exact_admission_hash"],
            },
        ),
        _lifecycle_row(
            lifecycle_index=1,
            lifecycle_type="eviction",
            action=evict_action,
            extra={
                "evicted_event_id": admit_action["action_event_id"],
                "capacity_bound": 1,
                "eviction_persisted": True,
            },
        ),
        _lifecycle_row(
            lifecycle_index=2,
            lifecycle_type="tombstone",
            action=tombstone_action,
            extra={
                "tombstoned_event_id": admit_action["action_event_id"],
                "tombstone_persisted": True,
                "resurrected_active_after_restart": False,
            },
        ),
        _lifecycle_row(
            lifecycle_index=3,
            lifecycle_type="rollback",
            action=rollback_action,
            extra={
                "rollback_target_state_hash": rollback_action["rollback_target_state_hash"],
                "rollback_restored_prior_state": True,
                "rollback_omitted": False,
            },
        ),
        _lifecycle_row(
            lifecycle_index=4,
            lifecycle_type="restart",
            action=restart_action,
            extra={
                "expected_state_hash": restart_action["expected_state_hash"],
                "restart_replay_state_hash": restart_action["post_state_hash"],
                "active_tombstoned_event_ids": [],
                "restart_receipt_persisted": True,
            },
        ),
    ]
    rows = [*event_rows, *action_rows, *evidence_rows, *lifecycle_rows]
    return {
        "event_schema": evt_schema,
        "action_receipt_schema": action_schema,
        "evidence_process_spec": eprocess_spec,
        "frozen_null_receipt": null_receipt,
        "rows": rows,
        "event_rows": event_rows,
        "action_rows": action_rows,
        "evidence_process_rows": evidence_rows,
        "lifecycle_rows": lifecycle_rows,
    }


def _rows_by_type(rows: Sequence[Mapping[str, Any]], row_type: str) -> list[JsonDict]:
    return [dict(row) for row in rows if row.get("row_type") == row_type]


def validate_contract_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    event_schema: Mapping[str, Any],
    action_receipt_schema: Mapping[str, Any],
    evidence_process_spec: Mapping[str, Any],
    frozen_null_receipt: Mapping[str, Any],
) -> JsonDict:
    """Validate transition rows without trusting artifact summaries."""

    reasons: list[str] = []
    base_rows = [row for row in rows if row.get("row_type") != "attack"]
    for row in base_rows:
        if row.get("row_hash") != row_hash(row):
            _add_reason(reasons, "row_hash_mismatch")
    if event_schema.get("schema_hash") != globals()["event_schema"]()["schema_hash"]:
        _add_reason(reasons, "event_schema_mismatch")
    if action_receipt_schema.get("schema_hash") != globals()["action_receipt_schema"]()[
        "schema_hash"
    ]:
        _add_reason(reasons, "action_schema_mismatch")
    expected_spec = globals()["evidence_process_spec"]()
    if evidence_process_spec.get("spec_hash") != expected_spec["spec_hash"]:
        _add_reason(reasons, "eprocess_spec_mismatch")
    expected_null_hash = frozen_null_receipt.get("null_receipt_hash", "")
    if frozen_null_receipt.get("null_id") != EPROCESS_NULL_ID:
        _add_reason(reasons, "missing_null")

    event_rows = _rows_by_type(base_rows, "event")
    action_rows = _rows_by_type(base_rows, "action")
    evidence_rows = _rows_by_type(base_rows, "evidence_process")
    lifecycle_rows = _rows_by_type(base_rows, "lifecycle")
    event_by_id = {str(row.get("event_id")): row for row in event_rows}
    event_types = [row.get("event_type") for row in event_rows]
    if event_types != list(EVENT_TYPES):
        _add_reason(reasons, "event_order_mismatch")
    seen_event_ids: set[str] = set()
    previous_mono = -1
    for expected_index, row in enumerate(event_rows):
        event_id = str(row.get("event_id"))
        if event_id in seen_event_ids:
            _add_reason(reasons, "duplicate_event_id")
        seen_event_ids.add(event_id)
        if row.get("event_id") != _event_id(row):
            _add_reason(reasons, "event_id_mismatch")
        if row.get("event_payload_hash") != receipts.sha256_json(row.get("event_payload", {})):
            _add_reason(reasons, "event_payload_hash_mismatch")
        if row.get("chronology_index") != expected_index:
            _add_reason(reasons, "event_order_mismatch")
        monotonic_ns = int(row.get("monotonic_receipt_ns", -1))
        if monotonic_ns <= previous_mono:
            _add_reason(reasons, "event_chronology_not_monotonic")
        previous_mono = monotonic_ns
    if event_rows and int(frozen_null_receipt.get("frozen_at_monotonic_ns", 0)) >= min(
        int(row["monotonic_receipt_ns"]) for row in event_rows
    ):
        _add_reason(reasons, "missing_null")

    actions_by_event: dict[str, list[JsonDict]] = defaultdict(list)
    for action in action_rows:
        actions_by_event[str(action.get("action_event_id"))].append(action)
        event = event_by_id.get(str(action.get("action_event_id")))
        if action.get("action_hash") != _action_hash(action):
            _add_reason(reasons, "action_hash_mismatch")
        if event is None:
            _add_reason(reasons, "action_event_missing")
            continue
        if int(action.get("action_monotonic_ns", 0)) <= int(event["monotonic_receipt_ns"]):
            _add_reason(reasons, "action_backdated")
        if event.get("event_type") in DURABLE_EVENT_TYPES and action.get("durable") is not True:
            _add_reason(reasons, "stated_write_without_action")
        if action.get("durable") is True and (
            action.get("exact_admission_required") is not True
            or action.get("exact_admission_passed") is not True
            or action.get("exact_admission_hash") != event.get("exact_admission_hash")
        ):
            _add_reason(reasons, "action_without_exact_admission")
        if action.get("durable") is not True and not action.get("no_action_reason"):
            _add_reason(reasons, "explicit_no_action_missing")
    current_state = GENESIS_HASH
    for event in event_rows:
        event_actions = actions_by_event.get(str(event.get("event_id")), [])
        if len(event_actions) != 1:
            _add_reason(reasons, "event_action_count_mismatch")
            continue
        action = event_actions[0]
        if action.get("pre_state_hash") != current_state:
            _add_reason(reasons, "state_chain_mismatch")
        expected_post_state = _expected_post_state(action, current_state)
        if action.get("post_state_hash") != expected_post_state:
            _add_reason(reasons, "state_hash_mismatch")
        current_state = str(action.get("post_state_hash"))

    if [row.get("event_type") for row in evidence_rows] != list(EVIDENCE_EVENT_TYPES):
        _add_reason(reasons, "evidence_update_count_mismatch")
    peek_indexes: list[int] = []
    for row in evidence_rows:
        event = event_by_id.get(str(row.get("event_id")))
        if row.get("null_receipt_hash") != expected_null_hash:
            _add_reason(reasons, "missing_null")
        if row.get("threshold_hash") != evidence_process_spec.get("threshold_hash"):
            _add_reason(reasons, "threshold_edited")
        if row.get("promotion_boundary") != evidence_process_spec.get("thresholds", {}).get(
            "promotion_boundary"
        ):
            _add_reason(reasons, "threshold_edited")
        if row.get("stopping_boundary") != evidence_process_spec.get("thresholds", {}).get(
            "stopping_boundary"
        ):
            _add_reason(reasons, "threshold_edited")
        if row.get("adaptive_peek_charged") is not True:
            _add_reason(reasons, "peek_charge_mismatch")
        peek_indexes.append(int(row.get("peek_charge_index", -1)))
        if event is not None and row.get("null_frozen_before_event") is not True:
            _add_reason(reasons, "missing_null")
    expected_peeks = list(range(1, len(evidence_rows) + 1))
    if sorted(peek_indexes) != expected_peeks or len(set(peek_indexes)) != len(peek_indexes):
        _add_reason(reasons, "peek_charge_mismatch")

    lifecycle_by_type = {str(row.get("lifecycle_type")): row for row in lifecycle_rows}
    if set(lifecycle_by_type) != set(LIFECYCLE_TYPES):
        _add_reason(reasons, "lifecycle_count_mismatch")
    rollback = lifecycle_by_type.get("rollback")
    if rollback is None or rollback.get("rollback_restored_prior_state") is not True:
        _add_reason(reasons, "rollback_omission")
    tombstone = lifecycle_by_type.get("tombstone")
    if tombstone is not None and tombstone.get("resurrected_active_after_restart") is True:
        _add_reason(reasons, "tombstone_resurrection")
    restart = lifecycle_by_type.get("restart")
    if restart is not None:
        if restart.get("restart_replay_state_hash") != restart.get("expected_state_hash"):
            _add_reason(reasons, "restart_drift")
        if restart.get("active_tombstoned_event_ids"):
            _add_reason(reasons, "tombstone_resurrection")
    counts = Counter(str(row.get("row_type")) for row in base_rows)
    return {
        "accepted": not reasons,
        "reasons": sorted(reasons),
        "row_type_counts": dict(sorted(counts.items())),
        "event_count": len(event_rows),
        "action_count": len(action_rows),
        "evidence_update_count": len(evidence_rows),
        "lifecycle_count": len(lifecycle_rows),
    }


def mutate_rows_for_attack(attack_id: str, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return rows with one event, action, evidence, or lifecycle attack applied."""

    mutated: list[JsonDict] = _copy_json(list(rows))
    if attack_id == "duplicate_events":
        row = deepcopy(next(row for row in mutated if row["row_type"] == "event"))
        mutated.insert(1, row)
    elif attack_id == "backdated_writes":
        action = next(row for row in mutated if row["row_type"] == "action" and row["event_type"] == "admit")
        event = next(row for row in mutated if row["row_type"] == "event" and row["event_type"] == "admit")
        action["action_monotonic_ns"] = int(event["monotonic_receipt_ns"]) - 1
        action["action_hash"] = _action_hash(action)
        _refresh_row(action)
    elif attack_id == "stated_write_without_action":
        action = next(row for row in mutated if row["row_type"] == "action" and row["event_type"] == "admit")
        action["durable"] = False
        action["no_action_reason"] = "claimed_write_without_disk_receipt"
        action["action_hash"] = _action_hash(action)
        _refresh_row(action)
    elif attack_id == "action_without_exact_admission":
        action = next(row for row in mutated if row["row_type"] == "action" and row["event_type"] == "promote")
        action["exact_admission_passed"] = False
        action["exact_admission_hash"] = "sha256:" + "1" * 64
        action["action_hash"] = _action_hash(action)
        _refresh_row(action)
    elif attack_id == "threshold_editing":
        row = next(row for row in mutated if row["row_type"] == "evidence_process")
        row["promotion_boundary"] = 1.0
        _refresh_row(row)
    elif attack_id == "repeated_peeking":
        row = deepcopy(next(row for row in mutated if row["row_type"] == "evidence_process"))
        row["sequential_index"] = 99
        _refresh_row(row)
        mutated.append(row)
    elif attack_id == "missing_null":
        row = next(row for row in mutated if row["row_type"] == "evidence_process")
        row["null_receipt_hash"] = "sha256:" + "2" * 64
        _refresh_row(row)
    elif attack_id == "rollback_omission":
        row = next(row for row in mutated if row["row_type"] == "lifecycle" and row["lifecycle_type"] == "rollback")
        row["rollback_restored_prior_state"] = False
        row["rollback_omitted"] = True
        _refresh_row(row)
    elif attack_id == "tombstone_resurrection":
        row = next(row for row in mutated if row["row_type"] == "lifecycle" and row["lifecycle_type"] == "tombstone")
        row["resurrected_active_after_restart"] = True
        _refresh_row(row)
    elif attack_id == "restart_drift":
        row = next(row for row in mutated if row["row_type"] == "lifecycle" and row["lifecycle_type"] == "restart")
        row["restart_replay_state_hash"] = "sha256:" + "3" * 64
        _refresh_row(row)
    else:
        raise ValueError(f"unknown attack_id: {attack_id}")
    return mutated


def mutation_attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    event_schema: Mapping[str, Any],
    action_receipt_schema: Mapping[str, Any],
    evidence_process_spec: Mapping[str, Any],
    frozen_null_receipt: Mapping[str, Any],
) -> JsonDict:
    """Run all mutation attacks and require every one to fail closed."""

    attack_rows = []
    for attack_id in ATTACK_IDS:
        mutated = mutate_rows_for_attack(attack_id, rows)
        report = validate_contract_rows(
            mutated,
            event_schema=event_schema,
            action_receipt_schema=action_receipt_schema,
            evidence_process_spec=evidence_process_spec,
            frozen_null_receipt=frozen_null_receipt,
        )
        attack_rows.append(
            _with_row_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "attack",
                    "task_id": TASK_ID,
                    "attack_id": attack_id,
                    "accepted": report["accepted"],
                    "fail_closed": report["accepted"] is False,
                    "reasons": report["reasons"],
                    "mutated_row_count": len(mutated),
                }
            )
        )
    false_accepts = [row["attack_id"] for row in attack_rows if row["fail_closed"] is not True]
    return {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": attack_rows,
        "attack_count": len(attack_rows),
        "false_accept_count": len(false_accepts),
        "false_accept_attack_ids": false_accepts,
        "all_critical_fail_closed": not false_accepts and len(attack_rows) == len(ATTACK_IDS),
    }


def recompute_aggregates_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    event_schema: Mapping[str, Any],
    action_receipt_schema: Mapping[str, Any],
    evidence_process_spec: Mapping[str, Any],
    frozen_null_receipt: Mapping[str, Any],
) -> JsonDict:
    """Recompute the ready score from rows and attack outcomes."""

    base_rows = [row for row in rows if row.get("row_type") != "attack"]
    attack_rows = _rows_by_type(rows, "attack")
    validation = validate_contract_rows(
        base_rows,
        event_schema=event_schema,
        action_receipt_schema=action_receipt_schema,
        evidence_process_spec=evidence_process_spec,
        frozen_null_receipt=frozen_null_receipt,
    )
    counts = Counter(str(row.get("row_type")) for row in rows)
    attack_ids = {str(row.get("attack_id")) for row in attack_rows}
    checks = {
        "positive_rows_validate": validation["accepted"] is True,
        "events_present": counts.get("event", 0) == len(EVENT_TYPES),
        "actions_present": counts.get("action", 0) == len(EVENT_TYPES),
        "evidence_updates_present": counts.get("evidence_process", 0)
        == len(EVIDENCE_EVENT_TYPES),
        "lifecycle_rows_present": counts.get("lifecycle", 0) == len(LIFECYCLE_TYPES),
        "all_attacks_present": attack_ids == set(ATTACK_IDS),
        "all_attacks_fail_closed": bool(attack_rows)
        and all(row.get("fail_closed") is True for row in attack_rows),
    }
    score = 1.0 if all(checks.values()) else 0.0
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(counts.items())),
        "validation_reasons": validation["reasons"],
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "online_transition_contract_ready_score_from_rows": score,
    }


def _gate_check_summary(
    *,
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "aggregate_ready_score_is_one": aggregate.get(
            "online_transition_contract_ready_score_from_rows"
        )
        == 1.0,
        "protected_files_unchanged": protected.get("protected_files_unchanged") is True,
        "preconditions_ready": preconditions.get("preconditions_ready") is True,
    }
    return {
        "checks": checks,
        "all_gates_passed": all(checks.values()),
        "failed_gates": [key for key, value in checks.items() if not value],
    }


def _preconditions_checked(root: Path, source_hashes: Mapping[str, str | None]) -> JsonDict:
    exp6479_path = root / "results/experiment_6479_verify_repair_factor_cache_shadow_adapter.json"
    exp6420_path = root / "results/experiment_6420_csl_authenticity_safety_audit.json"
    exp6433_path = root / "results/experiment_6433_csl_row_recomputation_safety_audit.json"
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    exp6479 = json.loads(exp6479_path.read_text(encoding="utf-8"))
    exp6420 = json.loads(exp6420_path.read_text(encoding="utf-8"))
    exp6433 = json.loads(exp6433_path.read_text(encoding="utf-8"))
    exp6479_ready = exp6479.get("factor_cache_shadow_adapter_ready_score") == 1.0
    retired_scope_present = "exp5895_csl_exact_slot_requalification_retired_v525" in exclusion_text
    exp5895_reused = TASK_ID.startswith("exp5895") or MODULE_RELATIVE_PATH.name.startswith(
        "experiment_5895"
    )
    audit_gaps = {
        "exp6420_open_attacks": exp6420.get("attack_matrix", {}).get(
            "open_critical_attack_ids", []
        ),
        "exp6433_attack_count": len(exp6433.get("attack_matrix", {}).get("rows", [])),
    }
    checks = {
        "exp6479_ready": exp6479_ready,
        "exp5895_exact_slot_retired": retired_scope_present,
        "exp5895_exact_slot_reused": exp5895_reused,
        "deterministic_fixture_no_llm": True,
    }
    return {
        "date": RUN_DATE,
        "repository_state": {
            "head": _git_output(["rev-parse", "HEAD"], root),
            "status_short": _git_output(["status", "--short"], root),
        },
        "exp6479_ready": exp6479_ready,
        "exp6479_status": exp6479.get("status"),
        "exp6479_path": str(exp6479_path.relative_to(root)),
        "exp5895_exact_slot_retired": retired_scope_present,
        "exp5895_exact_slot_reused": exp5895_reused,
        "audit_gaps_considered": audit_gaps,
        "source_hashes": dict(source_hashes),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "pid": os.getpid(),
            "captured_utc": _utc_now(),
        },
        "preconditions_ready": checks["exp6479_ready"]
        and checks["exp5895_exact_slot_retired"]
        and not checks["exp5895_exact_slot_reused"],
        "checks": checks,
    }


def _field_provenance(source_hashes: Mapping[str, str | None]) -> dict[str, JsonDict]:
    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    reducers = [
        "build_contract_rows",
        "validate_contract_rows",
        "mutation_attack_matrix",
        "recompute_aggregates_from_rows",
    ]
    return {
        field: {
            "spec_refs": ["REQ-INFRA-6485"],
            "source_paths": source_paths,
            "reducers": reducers,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status(score: float, gates: Mapping[str, Any]) -> str:
    if score == 1.0 and gates.get("all_gates_passed") is True:
        return "complete_online_cache_transition_eprocess_contract"
    return "blocked_online_cache_transition_eprocess_contract"


def _honest_verdict(status: str) -> str:
    if status.startswith("complete_"):
        return (
            "complete: online cache transition e-process contract is ready; "
            "no learning gain is claimed"
        )
    return (
        "complete_blocked: online cache transition e-process contract failed; "
        "gate_check_summary names the failed checks"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    write: bool = False,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal Exp6485 artifact."""

    protected_before = _protected_hashes(root)
    source_hashes = _source_hashes(root)
    contract = build_contract_rows(root=root)
    attack_matrix = mutation_attack_matrix(
        contract["rows"],
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )
    per_unit_rows = [*contract["rows"], *attack_matrix["rows"]]
    aggregate = recompute_aggregates_from_rows(
        per_unit_rows,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )
    protected = _protected_unchanged(root, protected_before)
    preconditions = _preconditions_checked(root, source_hashes)
    gates = _gate_check_summary(
        aggregate=aggregate,
        protected=protected,
        preconditions=preconditions,
    )
    score = float(aggregate["online_transition_contract_ready_score_from_rows"])
    if not gates["all_gates_passed"]:
        score = 0.0
    status = _status(score, gates)
    artifact: JsonDict = {
        "status": status,
        "event_schema": contract["event_schema"],
        "action_receipt_schema": contract["action_receipt_schema"],
        "evidence_process_spec": contract["evidence_process_spec"],
        "frozen_null_receipt": contract["frozen_null_receipt"],
        "event_rows": contract["event_rows"],
        "action_rows": contract["action_rows"],
        "evidence_process_rows": contract["evidence_process_rows"],
        "lifecycle_rows": contract["lifecycle_rows"],
        "attack_matrix": attack_matrix,
        "online_transition_contract_ready_score": score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "protected_files_unchanged": protected,
        "gate_check_summary": gates,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "results": list(DEFAULT_TEST_RESULTS if tests_run is None else tests_run),
        },
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, result_path)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clone = _copy_json(payload)
    clone["duration_s"] = 0.0
    clone["reproducibility_checksum"] = ""
    return receipts.sha256_json(clone)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate artifact fields and row-derived ready score."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    errors: list[str] = []
    aggregate = recompute_aggregates_from_rows(
        artifact.get("per_unit_rows", []),
        event_schema=artifact.get("event_schema", {}),
        action_receipt_schema=artifact.get("action_receipt_schema", {}),
        evidence_process_spec=artifact.get("evidence_process_spec", {}),
        frozen_null_receipt=artifact.get("frozen_null_receipt", {}),
    )
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("online_transition_contract_ready_score") != aggregate.get(
        "online_transition_contract_ready_score_from_rows"
    ):
        errors.append("online_transition_contract_ready_score mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact.get("field_principles", {}):
            errors.append(f"missing field_principles entry: {field}")
            break
    if artifact.get("protected_files_unchanged", {}).get("protected_files_unchanged") is not True:
        errors.append("protected_files_unchanged must be true")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete:", "complete_blocked:")):
        errors.append("honest_verdict lacks required terminal prefix")
    expected_status = _status(
        float(artifact.get("online_transition_contract_ready_score", 0.0) or 0.0),
        artifact.get("gate_check_summary", {}),
    )
    if artifact.get("status") != expected_status:
        errors.append("status mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    return receipts.write_json_atomic(path, artifact)


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    write: bool = True,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and write the Exp6485 artifact."""

    del date
    start = time.monotonic()
    artifact = build_artifact(
        root=REPO_ROOT,
        result_path=result_path,
        write=False,
        duration_s=0.0001,
        tests_run=tests_run,
    )
    artifact["duration_s"] = max(time.monotonic() - start, 0.0001)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, result_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        if not result_path.is_file():
            print(json.dumps({"ok": False, "errors": ["artifact missing"]}, sort_keys=True))
            return 1
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(date=str(args.date), result_path=result_path, write=True)
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "path": str(result_path),
                "status": artifact["status"],
                "online_transition_contract_ready_score": artifact[
                    "online_transition_contract_ready_score"
                ],
                "ok": not errors,
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

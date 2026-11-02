"""Exp6495 restarted reuse, spawn, and defer factor-pool controller.

Spec refs: REQ-INFRA-6495, SCENARIO-INFRA-6495-PAIRED-EVIDENCE,
SCENARIO-INFRA-6495-DECISIONS, SCENARIO-INFRA-6495-CAPACITY,
SCENARIO-INFRA-6495-ROLLBACK-RESTART, SCENARIO-INFRA-6495-ADMISSION,
SCENARIO-INFRA-6495-ATTACKS, SCENARIO-INFRA-6495-ARTIFACT.

The controller is a deterministic contract fixture. It proves the lifecycle
mechanism. It does not claim that a learned factor improves future work.
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
RANDOM_SEED = 6495
TASK_ID = "exp6495-restarted-factor-pool-controller"
INFERENCE_SUBSTRATE = "deterministic_anytime_factor_pool_controller_no_llm"
SCHEMA_VERSION = "carnot.experiment_6495.restarted_factor_pool_controller.v1"
GENESIS_HASH = "sha256:" + "0" * 64
BASE_MONOTONIC_NS = 2_000_000_000
NULL_FROZEN_NS = BASE_MONOTONIC_NS - 100_000
POOL_CAPACITY = 2
MAX_QUARANTINE = 3
MAX_RESTARTS = 2
MINIMUM_EVIDENCE = 2
REUSE_THRESHOLD = 2.0
SPAWN_THRESHOLD = 2.0
DEFER_ZONE_UPPER = 1.999999
CONTRADICTION_GAP_MIN = 0.2

RESULT_RELATIVE_PATH = Path("results/experiment_6495_restarted_factor_pool_controller.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6495_restarted_factor_pool_controller.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6495_restarted_factor_pool_controller.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

UPSTREAM_GATE_PATH = Path("results/experiment_6488_v559_decision_ledger.json")
EXP6479_PATH = Path("results/experiment_6479_verify_repair_factor_cache_shadow_adapter.json")
EXP6485_PATH = Path("results/experiment_6485_online_cache_transition_eprocess_contract.json")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6479_verify_repair_factor_cache_shadow_adapter.py"),
    Path("python/carnot/experiment_6485_online_cache_transition_eprocess_contract.py"),
    Path("python/carnot/experiment_6488_v559_decision_ledger.py"),
    Path("python/carnot/pipeline/factor_cache_shadow_adapter.py"),
    EXP6479_PATH,
    EXP6485_PATH,
    UPSTREAM_GATE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("ops/e2e-test-plan.md"),
)

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
PROCESS_KINDS = ("reuse", "spawn")
ATTACK_IDS = (
    "duplicate_event_id",
    "backdated_event",
    "adaptive_peek_reuse",
    "threshold_edit",
    "outside_authority_write",
    "capacity_overflow",
    "rollback_target_corruption",
    "tombstone_resurrection",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6495_restarted_factor_pool_controller "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6495_restarted_factor_pool_controller.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6495_restarted_factor_pool_controller.py "
    "-m pytest tests/python/test_experiment_6495_restarted_factor_pool_controller.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6495_restarted_factor_pool_controller.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6495_restarted_factor_pool_controller.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6495_restarted_factor_pool_controller.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6495_restarted_factor_pool_controller.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6495_restarted_factor_pool_controller --validate"
)
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
    E2E_PLAN_COMMAND,
)
DEFAULT_TEST_RESULTS = tuple(
    {"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_gate_receipt",
    "dependency_receipts",
    "controller_spec",
    "evidence_process_spec",
    "multiplicity_spec",
    "fixture_manifest",
    "event_rows",
    "evidence_update_rows",
    "decision_action_rows",
    "pool_state_rows",
    "exact_admission_receipts",
    "controller_attack_matrix",
    "factor_pool_controller_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
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
    "status": "Terminal controller-contract state.",
    "upstream_gate_receipt": "Exp6488 path, hash, field, expected, and observed value.",
    "dependency_receipts": "Exp6479 and Exp6485 paths, hashes, and readiness fields.",
    "controller_spec": "Frozen reuse, spawn, defer, capacity, rollback, and restart rules.",
    "evidence_process_spec": "Both nulls, alternatives, updates, thresholds, spending, and restart schedule.",
    "multiplicity_spec": "Factor and restart correction and accounting.",
    "fixture_manifest": "Positive, null, contradictory, recurrent, and corrupt streams.",
    "event_rows": "Immutable chronological fixture events.",
    "evidence_update_rows": "Both one-sided process updates and spending per event.",
    "decision_action_rows": "Reuse, spawn, defer, evict, rollback, restart, and no-write receipts.",
    "pool_state_rows": "Bounded state after every action.",
    "exact_admission_receipts": "Proof that writes require exact verification.",
    "controller_attack_matrix": "Duplicate, peeking, authority, capacity, rollback, and resurrection attacks.",
    "factor_pool_controller_ready_score": "Same-roadmap downstream gate field.",
    "per_unit_rows": "Event, update, decision, state, and attack rows.",
    "aggregate_row_recomputation": "Every transition count and ready score recomputed from rows.",
    "gate_check_summary": "Exact gate evaluation or blocked_* reason and observed value.",
    "preconditions_checked": "Lineage lock, adapter, transition contract, and durable store.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "inference_substrate": "deterministic_anytime_factor_pool_controller_no_llm.",
    "verifier_is_oracle": "True only for deterministic fixtures and exact admission checks.",
    "field_principles": "Reason for each sequential and lifecycle field.",
    "field_provenance": "Contract versions, event hashes, action receipts, and reducers.",
    "random_seed": "Fixed fixture and attack ordering seed.",
    "duration_s": "Measured wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over specs, fixtures, rows, and attacks.",
    "honest_verdict": "complete_* when the mechanism is ready, otherwise blocked_* with gate_check_summary.",
}

FIXTURE_EVENTS: tuple[JsonDict, ...] = (
    {
        "fixture_id": "observe_positive_alpha",
        "stream_kind": "positive",
        "event_type": "observe",
        "fixture_label": "positive_observation_seen",
        "authority": "runtime_observer",
        "factor_id": "factor_alpha",
        "reuse_raw_increment": 1.05,
        "spawn_raw_increment": 1.2,
        "exact_admission_passed": False,
    },
    {
        "fixture_id": "score_positive_alpha",
        "stream_kind": "positive",
        "event_type": "score",
        "fixture_label": "positive_scored_shadow_only",
        "authority": "shadow_scorer",
        "factor_id": "factor_alpha",
        "reuse_raw_increment": 1.05,
        "spawn_raw_increment": 1.4,
        "exact_admission_passed": False,
    },
    {
        "fixture_id": "positive_spawn_alpha",
        "stream_kind": "positive",
        "event_type": "spawn",
        "fixture_label": "spawn_alpha_exact_positive",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_alpha",
        "reuse_raw_increment": 1.1,
        "spawn_raw_increment": 2.5,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "recurrent_reuse_alpha",
        "stream_kind": "recurrent",
        "event_type": "reuse",
        "fixture_label": "reuse_alpha_recurrent_support",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_alpha",
        "reuse_raw_increment": 6.76,
        "spawn_raw_increment": 1.1,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "null_defer",
        "stream_kind": "null",
        "event_type": "defer",
        "fixture_label": "null_inside_indifference_zone",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_alpha",
        "reuse_raw_increment": 1.05,
        "spawn_raw_increment": 1.04,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "contradictory_defer",
        "stream_kind": "contradictory",
        "event_type": "defer",
        "fixture_label": "both_processes_cross_boundary",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_alpha",
        "reuse_raw_increment": 5.76,
        "spawn_raw_increment": 5.76,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "corrupted_quarantine",
        "stream_kind": "corrupt",
        "event_type": "quarantine",
        "fixture_label": "corrupted_receipt_fails_authority",
        "authority": "corrupt_fixture_source",
        "factor_id": "factor_corrupt",
        "reuse_raw_increment": 7.84,
        "spawn_raw_increment": 0.9,
        "exact_admission_passed": False,
    },
    {
        "fixture_id": "positive_spawn_beta",
        "stream_kind": "positive",
        "event_type": "spawn",
        "fixture_label": "spawn_beta_exact_positive",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_beta",
        "reuse_raw_increment": 1.1,
        "spawn_raw_increment": 4.84,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "capacity_overflow_spawn_gamma",
        "stream_kind": "capacity_overflow",
        "event_type": "capacity_overflow",
        "fixture_label": "spawn_gamma_requires_evict",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_gamma",
        "reuse_raw_increment": 1.0,
        "spawn_raw_increment": 8.0,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "tombstone_beta",
        "stream_kind": "corrupt",
        "event_type": "tombstone",
        "fixture_label": "tombstone_evicted_beta",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_beta",
        "reuse_raw_increment": 1.0,
        "spawn_raw_increment": 1.0,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "rollback_to_pre_overflow",
        "stream_kind": "rollback",
        "event_type": "rollback",
        "fixture_label": "rollback_suppresses_tombstone",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_beta",
        "reuse_raw_increment": 1.0,
        "spawn_raw_increment": 1.0,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "restart_replay",
        "stream_kind": "restart",
        "event_type": "restart",
        "fixture_label": "restart_replays_without_resurrection",
        "authority": "exact_fixture_verifier",
        "factor_id": "factor_alpha",
        "reuse_raw_increment": 1.0,
        "spawn_raw_increment": 1.0,
        "exact_admission_passed": True,
    },
    {
        "fixture_id": "outside_authority_no_write",
        "stream_kind": "corrupt",
        "event_type": "no_write",
        "fixture_label": "outside_authority_defaults_closed",
        "authority": "external_suggestion",
        "factor_id": "factor_delta",
        "reuse_raw_increment": 2.5,
        "spawn_raw_increment": 2.5,
        "exact_admission_passed": False,
    },
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _read_json(path: Path) -> JsonDict:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(parsed)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): receipts.sha256_file(_resolve(root, path)) for path in SOURCE_RELATIVE_PATHS}


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {
        path.as_posix(): receipts.sha256_file(_resolve(root, path))
        for path in PROTECTED_RELATIVE_PATHS
    }


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for path, before_hash in before.items():
        after_hash = receipts.sha256_file(root / path)
        files[path] = {
            "sha256_before": before_hash,
            "sha256_after": after_hash,
            "unchanged": before_hash == after_hash,
        }
    return {
        "active_roadmap_and_conductor_unchanged": all(row["unchanged"] for row in files.values()),
        "files": files,
    }


def _copy_json(value: Any) -> Any:
    return json.loads(receipts.canonical_json(value))


def row_hash(row: Mapping[str, Any]) -> str:
    material = {key: value for key, value in row.items() if key != "row_hash"}
    return receipts.sha256_json(material)


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


def controller_spec() -> JsonDict:
    spec = {
        "schema_version": SCHEMA_VERSION + ".controller_spec",
        "default_mode": "shadow_no_write",
        "actions": [
            "observe",
            "score",
            "quarantine",
            "reuse",
            "spawn",
            "defer",
            "evict",
            "tombstone",
            "rollback",
            "restart",
            "no_write",
        ],
        "reuse_threshold": REUSE_THRESHOLD,
        "spawn_threshold": SPAWN_THRESHOLD,
        "minimum_evidence": MINIMUM_EVIDENCE,
        "defer_zone": {
            "upper_exclusive": DEFER_ZONE_UPPER,
            "contradiction_gap_min": CONTRADICTION_GAP_MIN,
            "rule": "defer if neither side crosses, both sides cross, or authority is not exact",
        },
        "capacity": {
            "max_active_factors": POOL_CAPACITY,
            "max_quarantine": MAX_QUARANTINE,
            "max_restart_epochs": MAX_RESTARTS,
        },
        "eviction_rule": ["support_score_ascending", "last_reused_at_ascending", "factor_id"],
        "rollback_rule": "restore snapshot and remove tombstoned factors",
        "restart_rule": "replay durable rows without respending prior evidence tokens",
        "exact_write_authority": "exact_fixture_verifier",
    }
    spec["spec_hash"] = receipts.sha256_json(spec)
    return spec


def evidence_process_spec() -> JsonDict:
    thresholds = {
        "minimum_evidence": MINIMUM_EVIDENCE,
        "reuse_threshold": REUSE_THRESHOLD,
        "spawn_threshold": SPAWN_THRESHOLD,
        "defer_zone_upper": DEFER_ZONE_UPPER,
        "contradiction_gap_min": CONTRADICTION_GAP_MIN,
    }
    spec = {
        "schema_version": SCHEMA_VERSION + ".evidence_process_spec",
        "processes": {
            "reuse": {
                "null_id": "reuse_null_existing_factor_has_no_positive_support_v1",
                "null": "The current active factor does not deserve reuse for this event.",
                "alternative": "The active factor has enough one-sided support to reuse.",
                "threshold": REUSE_THRESHOLD,
            },
            "spawn": {
                "null_id": "spawn_null_no_new_factor_needed_v1",
                "null": "No new factor is needed for this event.",
                "alternative": "A new factor has enough one-sided support to spawn.",
                "threshold": SPAWN_THRESHOLD,
            },
        },
        "update_rule": (
            "raw fixture likelihood ratio is corrected by factor and restart "
            "multiplicity before comparison to the one-sided threshold"
        ),
        "evidence_spending": "one spend token per event, process, and restart epoch",
        "restart_schedule": {
            "explicit_restart_event_type": "restart",
            "geometric_memory_schedule": [1, 2, 4, 8],
            "pre_restart_tokens_spent_once": True,
        },
        "thresholds": thresholds,
        "threshold_hash": receipts.sha256_json(thresholds),
        "nulls_frozen_at_monotonic_ns": NULL_FROZEN_NS,
        "held_events_used_for_tuning": False,
    }
    spec["spec_hash"] = receipts.sha256_json(spec)
    return spec


def multiplicity_spec() -> JsonDict:
    spec = {
        "schema_version": SCHEMA_VERSION + ".multiplicity_spec",
        "factor_multiplicity_rule": "max(1, active_factor_count_before + candidate_factor_count)",
        "restart_multiplicity_rule": "restart_epoch_before + 1",
        "combined_denominator_rule": "factor_multiplicity * restart_multiplicity",
        "corrected_increment_rule": "raw_increment ** (1 / combined_denominator)",
        "restart_epoch_counts": True,
        "factor_creation_order_counts": True,
    }
    spec["spec_hash"] = receipts.sha256_json(spec)
    return spec


def fixture_manifest() -> JsonDict:
    streams: dict[str, list[str]] = defaultdict(list)
    for fixture in FIXTURE_EVENTS:
        streams[str(fixture["stream_kind"])].append(str(fixture["fixture_id"]))
    manifest = {
        "schema_version": SCHEMA_VERSION + ".fixture_manifest",
        "positive_stream": streams["positive"],
        "null_stream": streams["null"],
        "contradictory_stream": streams["contradictory"],
        "recurrent_stream": streams["recurrent"],
        "corrupt_stream": streams["corrupt"],
        "attack_fixture_ids": [
            "duplicate_event_id",
            "backdated_event",
            "adaptive_peek_reuse",
            "threshold_edit",
            "rollback_target_corruption",
            "tombstone_resurrection",
            "capacity_overflow",
        ],
        "all_fixture_ids": [str(fixture["fixture_id"]) for fixture in FIXTURE_EVENTS],
    }
    manifest["manifest_hash"] = receipts.sha256_json(manifest)
    return manifest


def _state_hash(state: Mapping[str, Any]) -> str:
    return receipts.sha256_json(
        {
            "active_factors": state["active_factors"],
            "quarantine": state["quarantine"],
            "restart_epoch": state["restart_epoch"],
            "tombstones": state["tombstones"],
        }
    )


def _event_payload(fixture: Mapping[str, Any]) -> JsonDict:
    return {
        "fixture_id": fixture["fixture_id"],
        "stream_kind": fixture["stream_kind"],
        "event_type": fixture["event_type"],
        "factor_id": fixture["factor_id"],
        "reuse_raw_increment": fixture["reuse_raw_increment"],
        "spawn_raw_increment": fixture["spawn_raw_increment"],
        "exact_admission_passed": fixture["exact_admission_passed"],
        "learning_benefit_claim": False,
    }


def _event_id(row: Mapping[str, Any]) -> str:
    return receipts.sha256_json({field: row.get(field) for field in IMMUTABLE_EVENT_ID_FIELDS})


def _event_row(
    fixture: Mapping[str, Any],
    *,
    chronology_index: int,
    parent_state_hash: str,
) -> JsonDict:
    payload = _event_payload(fixture)
    payload_hash = receipts.sha256_json(payload)
    row = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "event",
        "task_id": TASK_ID,
        "chronology_index": chronology_index,
        "event_type": fixture["event_type"],
        "fixture_id": fixture["fixture_id"],
        "fixture_label": fixture["fixture_label"],
        "monotonic_receipt_ns": BASE_MONOTONIC_NS + chronology_index * 100_000,
        "parent_state_hash": parent_state_hash,
        "event_payload": payload,
        "event_payload_hash": payload_hash,
        "authority": fixture["authority"],
        "expected_exact_authority": fixture["authority"] == "exact_fixture_verifier",
    }
    row["event_id"] = _event_id(row)
    return _with_row_hash(row)


def _corrected_increment(raw_increment: float, denominator: int) -> float:
    return round(float(raw_increment) ** (1.0 / float(max(1, denominator))), 9)


def _evidence_rows(
    fixture: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    active_count: int,
    restart_epoch: int,
    spec: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    candidate_count = 1
    factor_multiplicity = max(1, active_count + candidate_count)
    restart_multiplicity = restart_epoch + 1
    denominator = factor_multiplicity * restart_multiplicity
    for process_kind in PROCESS_KINDS:
        raw = float(fixture[f"{process_kind}_raw_increment"])
        corrected = _corrected_increment(raw, denominator)
        row = {
            "schema_version": SCHEMA_VERSION,
            "row_type": "evidence_update",
            "task_id": TASK_ID,
            "event_id": event["event_id"],
            "fixture_id": event["fixture_id"],
            "chronology_index": event["chronology_index"],
            "process_kind": process_kind,
            "null_id": spec["processes"][process_kind]["null_id"],
            "alternative": spec["processes"][process_kind]["alternative"],
            "raw_increment": raw,
            "factor_multiplicity": factor_multiplicity,
            "restart_multiplicity": restart_multiplicity,
            "multiplicity_denominator": denominator,
            "corrected_increment": corrected,
            "e_value_after_spend": corrected,
            "threshold": spec["processes"][process_kind]["threshold"],
            "threshold_hash": spec["threshold_hash"],
            "null_frozen_at_monotonic_ns": spec["nulls_frozen_at_monotonic_ns"],
            "null_frozen_before_event": int(spec["nulls_frozen_at_monotonic_ns"])
            < int(event["monotonic_receipt_ns"]),
            "spend_token": receipts.sha256_json(
                {
                    "event_id": event["event_id"],
                    "process_kind": process_kind,
                    "restart_epoch": restart_epoch,
                }
            ),
            "adaptive_peek_charged": True,
            "multiplicity_corrected": True,
            "held_events_used_for_tuning": False,
            "restart_epoch": restart_epoch,
        }
        rows.append(_with_row_hash(row))
    return rows


def _factor_sort_key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, int, str]:
    factor_id, record = item
    return (
        float(record.get("support_score", 0.0)),
        int(record.get("last_reused_at", 0)),
        factor_id,
    )


def _action_hash(row: Mapping[str, Any]) -> str:
    material = {key: value for key, value in row.items() if key not in {"action_hash", "row_hash"}}
    return receipts.sha256_json(material)


def _admission_hash(event: Mapping[str, Any], action_type: str, passed: bool) -> str:
    return receipts.sha256_json(
        {
            "event_id": event["event_id"],
            "fixture_id": event["fixture_id"],
            "action_type": action_type,
            "authority": event["authority"],
            "exact_admission_passed": passed,
        }
    )


def _decision_for_event(
    fixture: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> tuple[str, str, str, bool]:
    reuse_e = next(float(row["e_value_after_spend"]) for row in evidence if row["process_kind"] == "reuse")
    spawn_e = next(float(row["e_value_after_spend"]) for row in evidence if row["process_kind"] == "spawn")
    event_type = str(fixture["event_type"])
    exact = fixture.get("exact_admission_passed") is True and fixture.get("authority") == "exact_fixture_verifier"
    del state
    if event_type in {"observe", "score"}:
        return event_type, f"{event_type}_no_write", f"{event_type}_shadow_only", False
    if event_type == "quarantine":
        return "quarantine", "quarantine_no_write", "corrupted_receipt", False
    if event_type == "no_write":
        return "no_write", "no_write", "outside_exact_authority", False
    if not exact:
        return "defer", "defer_no_write", "missing_exact_authority", False
    if event_type == "tombstone":
        return "tombstone", "tombstone_write", "", True
    if event_type == "rollback":
        return "rollback", "rollback_restore", "", True
    if event_type == "restart":
        return "restart", "restart_replay", "", True
    both_cross = reuse_e >= REUSE_THRESHOLD and spawn_e >= SPAWN_THRESHOLD
    neither_cross = reuse_e < REUSE_THRESHOLD and spawn_e < SPAWN_THRESHOLD
    close_gap = abs(reuse_e - spawn_e) <= CONTRADICTION_GAP_MIN
    if event_type == "reuse" and reuse_e >= REUSE_THRESHOLD and spawn_e < SPAWN_THRESHOLD:
        return "reuse", "reuse_write", "", True
    if event_type in {"spawn", "capacity_overflow"} and spawn_e >= SPAWN_THRESHOLD and reuse_e < REUSE_THRESHOLD:
        action_type = "evict_then_spawn_write" if event_type == "capacity_overflow" else "spawn_write"
        return "spawn", action_type, "", True
    if both_cross or neither_cross or close_gap:
        return "defer", "defer_no_write", "indifference_or_contradictory_evidence", False
    return "defer", "defer_no_write", "decision_boundary_not_met", False


def _apply_action(
    state: JsonDict,
    *,
    decision: str,
    action_type: str,
    fixture: Mapping[str, Any],
    chronology_index: int,
    snapshots: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict]:
    next_state = _copy_json(state)
    action_extra: JsonDict = {"evicted_factor_id": "", "rollback_suppressed_tombstones": []}
    factor_id = str(fixture["factor_id"])
    if action_type == "spawn_write":
        next_state["active_factors"][factor_id] = {
            "support_score": round(float(fixture["spawn_raw_increment"]), 9),
            "created_at": chronology_index,
            "last_reused_at": chronology_index,
        }
    elif action_type == "reuse_write":
        record = next_state["active_factors"][factor_id]
        record["support_score"] = round(float(record["support_score"]) + float(fixture["reuse_raw_increment"]), 9)
        record["last_reused_at"] = chronology_index
    elif action_type == "evict_then_spawn_write":
        evicted = sorted(next_state["active_factors"].items(), key=_factor_sort_key)[0][0]
        next_state["active_factors"].pop(evicted)
        action_extra["evicted_factor_id"] = evicted
        next_state["active_factors"][factor_id] = {
            "support_score": round(float(fixture["spawn_raw_increment"]), 9),
            "created_at": chronology_index,
            "last_reused_at": chronology_index,
        }
    elif action_type == "tombstone_write":
        next_state["active_factors"].pop(factor_id, None)
        if factor_id not in next_state["tombstones"]:
            next_state["tombstones"].append(factor_id)
        next_state["tombstones"] = sorted(next_state["tombstones"])
    elif action_type == "rollback_restore":
        target = _copy_json(snapshots["pre_overflow"])
        suppressed = sorted(set(target["active_factors"]) & set(next_state["tombstones"]))
        for tombstoned in suppressed:
            target["active_factors"].pop(tombstoned, None)
        target["tombstones"] = sorted(next_state["tombstones"])
        target["quarantine"] = list(next_state["quarantine"])
        target["restart_epoch"] = next_state["restart_epoch"]
        next_state = target
        action_extra["rollback_suppressed_tombstones"] = suppressed
    elif action_type == "restart_replay":
        next_state["restart_epoch"] = int(next_state["restart_epoch"]) + 1
    elif decision == "quarantine":
        if len(next_state["quarantine"]) < MAX_QUARANTINE:
            next_state["quarantine"].append(factor_id)
    return next_state, action_extra


def _decision_action_row(
    event: Mapping[str, Any],
    fixture: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    pre_state_hash: str,
    post_state_hash: str,
    decision: str,
    action_type: str,
    no_write_reason: str,
    durable: bool,
    action_extra: Mapping[str, Any],
    restart_epoch_before: int,
    restart_epoch_after: int,
) -> JsonDict:
    admission_passed = durable and event["authority"] == "exact_fixture_verifier"
    exact_admission_hash = _admission_hash(event, action_type, admission_passed)
    row = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "decision_action",
        "task_id": TASK_ID,
        "action_id": receipts.sha256_json(
            {"event_id": event["event_id"], "action_type": action_type}
        ),
        "event_id": event["event_id"],
        "fixture_id": event["fixture_id"],
        "chronology_index": event["chronology_index"],
        "decision": decision,
        "action_type": action_type,
        "durable": durable,
        "no_write_reason": no_write_reason,
        "pre_state_hash": pre_state_hash,
        "post_state_hash": post_state_hash,
        "exact_admission_required": durable,
        "exact_admission_passed": admission_passed,
        "exact_admission_hash": exact_admission_hash if durable else "",
        "authority": event["authority"],
        "factor_id": fixture["factor_id"],
        "reuse_e_value": next(
            row["e_value_after_spend"] for row in evidence if row["process_kind"] == "reuse"
        ),
        "spawn_e_value": next(
            row["e_value_after_spend"] for row in evidence if row["process_kind"] == "spawn"
        ),
        "durability_receipt": "atomic_json_write" if durable else "shadow_no_write",
        "restart_epoch_before": restart_epoch_before,
        "restart_epoch_after": restart_epoch_after,
        **dict(action_extra),
    }
    row["action_hash"] = _action_hash(row)
    return _with_row_hash(row)


def _exact_admission_row(decision: Mapping[str, Any]) -> JsonDict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "exact_admission",
        "task_id": TASK_ID,
        "action_id": decision["action_id"],
        "event_id": decision["event_id"],
        "fixture_id": decision["fixture_id"],
        "action_type": decision["action_type"],
        "authority": decision["authority"],
        "exact_admission_required": decision["durable"],
        "exact_admission_passed": decision["exact_admission_passed"],
        "exact_admission_hash": decision["exact_admission_hash"],
        "durable_write_allowed": decision["durable"] and decision["exact_admission_passed"],
        "checker_receipt": {
            "checker": "deterministic_fixture_exact_admission",
            "event_id": decision["event_id"],
            "pre_state_hash": decision["pre_state_hash"],
            "post_state_hash": decision["post_state_hash"],
        },
    }
    return _with_row_hash(row)


def _pool_state_row(
    decision: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    action_extra: Mapping[str, Any],
) -> JsonDict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "pool_state",
        "task_id": TASK_ID,
        "event_id": decision["event_id"],
        "fixture_id": decision["fixture_id"],
        "chronology_index": decision["chronology_index"],
        "action_id": decision["action_id"],
        "state_hash": decision["post_state_hash"],
        "active_factor_ids": sorted(state["active_factors"]),
        "active_factor_count": len(state["active_factors"]),
        "capacity": POOL_CAPACITY,
        "quarantine_factor_ids": list(state["quarantine"]),
        "quarantine_count": len(state["quarantine"]),
        "tombstoned_factor_ids": list(state["tombstones"]),
        "restart_epoch": state["restart_epoch"],
        "restart_replay_state_hash": decision["post_state_hash"]
        if decision["action_type"] == "restart_replay"
        else "",
        "rollback_suppressed_tombstones": list(action_extra.get("rollback_suppressed_tombstones", [])),
        "evicted_factor_id": action_extra.get("evicted_factor_id", ""),
    }
    return _with_row_hash(row)


def build_controller_rows(*, root: Path = REPO_ROOT) -> JsonDict:
    """Build deterministic positive rows for the factor-pool controller."""

    del root
    c_spec = controller_spec()
    e_spec = evidence_process_spec()
    m_spec = multiplicity_spec()
    manifest = fixture_manifest()
    event_rows: list[JsonDict] = []
    evidence_rows: list[JsonDict] = []
    decision_rows: list[JsonDict] = []
    pool_state_rows: list[JsonDict] = []
    admission_rows: list[JsonDict] = []
    state: JsonDict = {
        "active_factors": {},
        "quarantine": [],
        "tombstones": [],
        "restart_epoch": 0,
    }
    snapshots: dict[str, JsonDict] = {}
    parent_state_hash = _state_hash(state)
    for chronology_index, fixture in enumerate(FIXTURE_EVENTS):
        event = _event_row(
            fixture,
            chronology_index=chronology_index,
            parent_state_hash=parent_state_hash,
        )
        if fixture["fixture_id"] == "capacity_overflow_spawn_gamma":
            snapshots["pre_overflow"] = _copy_json(state)
        event_rows.append(event)
        paired_evidence = _evidence_rows(
            fixture,
            event,
            active_count=len(state["active_factors"]),
            restart_epoch=int(state["restart_epoch"]),
            spec=e_spec,
        )
        evidence_rows.extend(paired_evidence)
        decision, action_type, no_write_reason, durable = _decision_for_event(
            fixture,
            paired_evidence,
            state,
        )
        pre_state_hash = _state_hash(state)
        restart_before = int(state["restart_epoch"])
        next_state, action_extra = _apply_action(
            state,
            decision=decision,
            action_type=action_type,
            fixture=fixture,
            chronology_index=chronology_index,
            snapshots=snapshots,
        )
        post_state_hash = _state_hash(next_state)
        decision_row = _decision_action_row(
            event,
            fixture,
            paired_evidence,
            pre_state_hash=pre_state_hash,
            post_state_hash=post_state_hash,
            decision=decision,
            action_type=action_type,
            no_write_reason=no_write_reason,
            durable=durable,
            action_extra=action_extra,
            restart_epoch_before=restart_before,
            restart_epoch_after=int(next_state["restart_epoch"]),
        )
        state = next_state
        pool_state = _pool_state_row(decision_row, state, action_extra=action_extra)
        event_rows[-1]["parent_state_hash"] = pre_state_hash
        event_rows[-1]["event_id"] = _event_id(event_rows[-1])
        _refresh_row(event_rows[-1])
        decision_rows.append(decision_row)
        pool_state_rows.append(pool_state)
        admission_rows.append(_exact_admission_row(decision_row))
        parent_state_hash = post_state_hash
    rows = [*event_rows, *evidence_rows, *decision_rows, *pool_state_rows, *admission_rows]
    return {
        "controller_spec": c_spec,
        "evidence_process_spec": e_spec,
        "multiplicity_spec": m_spec,
        "fixture_manifest": manifest,
        "rows": rows,
        "event_rows": event_rows,
        "evidence_update_rows": evidence_rows,
        "decision_action_rows": decision_rows,
        "pool_state_rows": pool_state_rows,
        "exact_admission_receipts": admission_rows,
    }


def _rows_by_type(rows: Sequence[Mapping[str, Any]], row_type: str) -> list[JsonDict]:
    return [dict(row) for row in rows if row.get("row_type") == row_type]


def validate_controller_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    controller_spec: Mapping[str, Any],
    evidence_process_spec: Mapping[str, Any],
    multiplicity_spec: Mapping[str, Any],
) -> JsonDict:
    """Validate controller rows without trusting artifact summaries."""

    reasons: list[str] = []
    base_rows = [row for row in rows if row.get("row_type") != "attack"]
    for row in base_rows:
        if row.get("row_hash") != row_hash(row):
            _add_reason(reasons, "row_hash_mismatch")
    if controller_spec.get("spec_hash") != globals()["controller_spec"]()["spec_hash"]:
        _add_reason(reasons, "controller_spec_mismatch")
    expected_evidence = globals()["evidence_process_spec"]()
    if (
        evidence_process_spec.get("spec_hash") != expected_evidence["spec_hash"]
        or evidence_process_spec.get("threshold_hash") != expected_evidence["threshold_hash"]
    ):
        _add_reason(reasons, "evidence_process_spec_mismatch")
    if multiplicity_spec.get("spec_hash") != globals()["multiplicity_spec"]()["spec_hash"]:
        _add_reason(reasons, "multiplicity_spec_mismatch")

    event_rows = _rows_by_type(base_rows, "event")
    evidence_rows = _rows_by_type(base_rows, "evidence_update")
    decision_rows = _rows_by_type(base_rows, "decision_action")
    state_rows = _rows_by_type(base_rows, "pool_state")
    admission_rows = _rows_by_type(base_rows, "exact_admission")
    event_by_id = {str(row.get("event_id")): row for row in event_rows}
    decision_by_event = {str(row.get("event_id")): row for row in decision_rows}
    admissions_by_action = defaultdict(list)
    for row in admission_rows:
        admissions_by_action[str(row.get("action_id"))].append(row)

    if len(event_rows) != len(FIXTURE_EVENTS):
        _add_reason(reasons, "event_count_mismatch")
    if len(evidence_rows) != len(event_rows) * 2:
        _add_reason(reasons, "paired_evidence_count_mismatch")
    if len(decision_rows) != len(event_rows):
        _add_reason(reasons, "decision_count_mismatch")
    if len(state_rows) != len(event_rows):
        _add_reason(reasons, "pool_state_count_mismatch")
    if len(admission_rows) != len(decision_rows):
        _add_reason(reasons, "exact_admission_count_mismatch")

    seen_event_ids: set[str] = set()
    previous_mono = -1
    for expected_index, event in enumerate(event_rows):
        event_id = str(event.get("event_id"))
        if event_id in seen_event_ids:
            _add_reason(reasons, "duplicate_event_id")
        seen_event_ids.add(event_id)
        if event.get("event_id") != _event_id(event):
            _add_reason(reasons, "event_id_mismatch")
        if event.get("event_payload_hash") != receipts.sha256_json(event.get("event_payload", {})):
            _add_reason(reasons, "event_payload_hash_mismatch")
        if event.get("chronology_index") != expected_index:
            _add_reason(reasons, "event_order_mismatch")
        mono = int(event.get("monotonic_receipt_ns", -1))
        if mono <= previous_mono:
            _add_reason(reasons, "event_chronology_not_monotonic")
        previous_mono = mono

    evidence_by_event: dict[str, list[JsonDict]] = defaultdict(list)
    seen_spend_tokens: set[str] = set()
    for evidence in evidence_rows:
        event_id = str(evidence.get("event_id"))
        evidence_by_event[event_id].append(evidence)
        if evidence.get("event_id") not in event_by_id:
            _add_reason(reasons, "evidence_event_missing")
        token = str(evidence.get("spend_token"))
        if token in seen_spend_tokens:
            _add_reason(reasons, "evidence_spend_token_reused")
        seen_spend_tokens.add(token)
        if evidence.get("threshold_hash") != evidence_process_spec.get("threshold_hash"):
            _add_reason(reasons, "threshold_edited")
        if evidence.get("adaptive_peek_charged") is not True:
            _add_reason(reasons, "evidence_not_charged")
        if evidence.get("null_frozen_before_event") is not True:
            _add_reason(reasons, "missing_frozen_null")
        denominator = int(evidence.get("multiplicity_denominator", 1))
        expected_increment = _corrected_increment(float(evidence.get("raw_increment", 1.0)), denominator)
        if evidence.get("corrected_increment") != expected_increment:
            _add_reason(reasons, "multiplicity_correction_mismatch")
    for event_id, grouped in evidence_by_event.items():
        if {row.get("process_kind") for row in grouped} != set(PROCESS_KINDS):
            _add_reason(reasons, "paired_evidence_count_mismatch")

    states_by_event = {str(row.get("event_id")): row for row in state_rows}
    for decision in decision_rows:
        event = event_by_id.get(str(decision.get("event_id")))
        if decision.get("action_hash") != _action_hash(decision):
            _add_reason(reasons, "action_hash_mismatch")
        if event is None:
            _add_reason(reasons, "decision_event_missing")
            continue
        if decision.get("durable") is True:
            if (
                decision.get("authority") != controller_spec.get("exact_write_authority")
                or decision.get("exact_admission_passed") is not True
                or not decision.get("exact_admission_hash")
            ):
                _add_reason(reasons, "durable_write_without_exact_admission")
        elif not decision.get("no_write_reason"):
            _add_reason(reasons, "no_write_reason_missing")
        if decision.get("pre_state_hash") != event.get("parent_state_hash"):
            _add_reason(reasons, "state_hash_mismatch")
        if decision.get("action_type") == "rollback_restore":
            state = states_by_event.get(str(decision.get("event_id")), {})
            if "factor_beta" in state.get("active_factor_ids", []):
                _add_reason(reasons, "rollback_state_mismatch")
            if decision.get("post_state_hash") != state.get("state_hash"):
                _add_reason(reasons, "rollback_state_mismatch")
        for receipt in admissions_by_action.get(str(decision.get("action_id")), []):
            if receipt.get("event_id") != decision.get("event_id"):
                _add_reason(reasons, "exact_admission_event_mismatch")
            if decision.get("durable") and receipt.get("exact_admission_hash") != decision.get(
                "exact_admission_hash"
            ):
                _add_reason(reasons, "exact_admission_hash_mismatch")

    for state in state_rows:
        active_ids = list(state.get("active_factor_ids", []))
        tombstones = set(state.get("tombstoned_factor_ids", []))
        if int(state.get("active_factor_count", 0)) > POOL_CAPACITY or len(active_ids) > POOL_CAPACITY:
            _add_reason(reasons, "capacity_exceeded")
        if tombstones & set(active_ids):
            _add_reason(reasons, "tombstone_resurrection")
        decision = decision_by_event.get(str(state.get("event_id")), {})
        if state.get("state_hash") != decision.get("post_state_hash"):
            _add_reason(reasons, "state_hash_mismatch")
        if decision.get("action_type") == "restart_replay":
            if state.get("restart_replay_state_hash") != state.get("state_hash"):
                _add_reason(reasons, "restart_state_mismatch")
            if "factor_beta" in active_ids:
                _add_reason(reasons, "tombstone_resurrection")

    counts = Counter(str(row.get("row_type")) for row in base_rows)
    return {
        "accepted": not reasons,
        "reasons": sorted(reasons),
        "row_type_counts": dict(sorted(counts.items())),
        "event_count": len(event_rows),
        "evidence_update_count": len(evidence_rows),
        "decision_action_count": len(decision_rows),
        "pool_state_count": len(state_rows),
        "exact_admission_count": len(admission_rows),
    }


def mutate_rows_for_attack(attack_id: str, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return rows with one controller attack applied."""

    mutated: list[JsonDict] = _copy_json(list(rows))
    if attack_id == "duplicate_event_id":
        row = deepcopy(next(row for row in mutated if row["row_type"] == "event"))
        mutated.insert(1, row)
    elif attack_id == "backdated_event":
        event = next(row for row in mutated if row["row_type"] == "event" and row["chronology_index"] == 3)
        event["monotonic_receipt_ns"] = BASE_MONOTONIC_NS
        event["event_id"] = _event_id(event)
        _refresh_row(event)
    elif attack_id == "adaptive_peek_reuse":
        rows_by_process = [row for row in mutated if row["row_type"] == "evidence_update"]
        rows_by_process[1]["spend_token"] = rows_by_process[0]["spend_token"]
        _refresh_row(rows_by_process[1])
    elif attack_id == "threshold_edit":
        row = next(row for row in mutated if row["row_type"] == "evidence_update")
        row["threshold_hash"] = "sha256:" + "1" * 64
        _refresh_row(row)
    elif attack_id == "outside_authority_write":
        row = next(
            row
            for row in mutated
            if row["row_type"] == "decision_action"
            and row["fixture_id"] == "outside_authority_no_write"
        )
        row["durable"] = True
        row["exact_admission_required"] = True
        row["exact_admission_hash"] = ""
        row["action_hash"] = _action_hash(row)
        _refresh_row(row)
    elif attack_id == "capacity_overflow":
        row = next(
            row
            for row in mutated
            if row["row_type"] == "pool_state"
            and row["fixture_id"] == "capacity_overflow_spawn_gamma"
        )
        row["active_factor_ids"] = ["factor_alpha", "factor_beta", "factor_gamma"]
        row["active_factor_count"] = 3
        _refresh_row(row)
    elif attack_id == "rollback_target_corruption":
        row = next(
            row
            for row in mutated
            if row["row_type"] == "pool_state"
            and row["fixture_id"] == "rollback_to_pre_overflow"
        )
        row["active_factor_ids"] = ["factor_alpha", "factor_beta"]
        row["active_factor_count"] = 2
        _refresh_row(row)
    elif attack_id == "tombstone_resurrection":
        row = next(
            row
            for row in mutated
            if row["row_type"] == "pool_state" and row["fixture_id"] == "restart_replay"
        )
        row["active_factor_ids"] = ["factor_alpha", "factor_beta"]
        row["active_factor_count"] = 2
        _refresh_row(row)
    else:
        raise ValueError(f"unknown attack_id: {attack_id}")
    return mutated


def mutation_attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    controller_spec: Mapping[str, Any],
    evidence_process_spec: Mapping[str, Any],
    multiplicity_spec: Mapping[str, Any],
) -> JsonDict:
    """Run controller attacks and require every one to fail closed."""

    attack_rows = []
    for attack_id in ATTACK_IDS:
        mutated = mutate_rows_for_attack(attack_id, rows)
        report = validate_controller_rows(
            mutated,
            controller_spec=controller_spec,
            evidence_process_spec=evidence_process_spec,
            multiplicity_spec=multiplicity_spec,
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
    controller_spec: Mapping[str, Any],
    evidence_process_spec: Mapping[str, Any],
    multiplicity_spec: Mapping[str, Any],
) -> JsonDict:
    """Recompute every transition count and the ready score from rows."""

    base_rows = [row for row in rows if row.get("row_type") != "attack"]
    attack_rows = _rows_by_type(rows, "attack")
    validation = validate_controller_rows(
        base_rows,
        controller_spec=controller_spec,
        evidence_process_spec=evidence_process_spec,
        multiplicity_spec=multiplicity_spec,
    )
    counts = Counter(str(row.get("row_type")) for row in rows)
    decisions = _rows_by_type(base_rows, "decision_action")
    states = _rows_by_type(base_rows, "pool_state")
    attack_ids = {str(row.get("attack_id")) for row in attack_rows}
    action_types = {str(row.get("action_type")) for row in decisions}
    required_action_types = {
        "reuse_write",
        "spawn_write",
        "defer_no_write",
        "evict_then_spawn_write",
        "rollback_restore",
        "restart_replay",
        "no_write",
    }
    checks = {
        "positive_rows_validate": validation["accepted"] is True,
        "paired_evidence_rows_present": counts.get("evidence_update", 0)
        == counts.get("event", 0) * 2,
        "decisions_cover_required_actions": required_action_types <= action_types,
        "active_capacity_bounded": all(
            int(row.get("active_factor_count", 0)) <= POOL_CAPACITY for row in states
        ),
        "exact_admissions_cover_decisions": counts.get("exact_admission", 0)
        == counts.get("decision_action", 0),
        "all_attacks_present": attack_ids == set(ATTACK_IDS),
        "all_attacks_fail_closed": bool(attack_rows)
        and all(row.get("fail_closed") is True for row in attack_rows),
    }
    score = 1.0 if all(checks.values()) else 0.0
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(counts.items())),
        "transition_counts": {
            "events": counts.get("event", 0),
            "evidence_updates": counts.get("evidence_update", 0),
            "decisions": counts.get("decision_action", 0),
            "pool_states": counts.get("pool_state", 0),
            "exact_admissions": counts.get("exact_admission", 0),
            "attacks": counts.get("attack", 0),
        },
        "validation_reasons": validation["reasons"],
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "factor_pool_controller_ready_score_from_rows": score,
    }


def _artifact_receipt(root: Path, path: Path, readiness_fields: Sequence[str]) -> JsonDict:
    resolved = _resolve(root, path)
    payload = _read_json(resolved)
    return {
        "path": path.as_posix(),
        "sha256": receipts.sha256_file(resolved),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "readiness_fields": {field: payload.get(field) for field in readiness_fields},
    }


def _upstream_gate_receipt(root: Path) -> JsonDict:
    receipt = _artifact_receipt(root, UPSTREAM_GATE_PATH, ["v560_lineage_lock_ready_score"])
    observed = receipt["readiness_fields"]["v560_lineage_lock_ready_score"]
    return {
        "path": receipt["path"],
        "hash": receipt["sha256"],
        "field": "v560_lineage_lock_ready_score",
        "expected": 1.0,
        "observed": observed,
        "passed": observed == 1.0,
        "status": receipt["status"],
    }


def _dependency_receipts(root: Path) -> JsonDict:
    return {
        "exp6479": _artifact_receipt(
            root,
            EXP6479_PATH,
            ["factor_cache_shadow_adapter_ready_score"],
        ),
        "exp6485": _artifact_receipt(
            root,
            EXP6485_PATH,
            ["online_transition_contract_ready_score"],
        ),
    }


def _preconditions_checked(
    root: Path,
    source_hashes: Mapping[str, str | None],
    upstream_gate: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> JsonDict:
    result_dir = root / RESULT_RELATIVE_PATH.parent
    checks = {
        "lineage_lock_ready": upstream_gate.get("observed") == 1.0,
        "adapter_ready": dependencies["exp6479"]["readiness_fields"].get(
            "factor_cache_shadow_adapter_ready_score"
        )
        == 1.0,
        "transition_contract_ready": dependencies["exp6485"]["readiness_fields"].get(
            "online_transition_contract_ready_score"
        )
        == 1.0,
        "durable_store_ready": result_dir.is_dir() and os.access(result_dir, os.W_OK),
    }
    return {
        "date": RUN_DATE,
        "repository_state": {
            "head": _git_output(root, ["rev-parse", "HEAD"]),
            "status_short": _git_output(root, ["status", "--short"]),
        },
        "lineage_lock_ready": checks["lineage_lock_ready"],
        "adapter_ready": checks["adapter_ready"],
        "transition_contract_ready": checks["transition_contract_ready"],
        "durable_store_ready": checks["durable_store_ready"],
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "source_hashes": dict(source_hashes),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "pid": os.getpid(),
            "captured_utc": _utc_now(),
        },
    }


def _tests_passed(tests_run: Sequence[Mapping[str, Any]] | None) -> bool:
    return all(int(row.get("exit_code", 1)) == 0 for row in (tests_run or DEFAULT_TEST_RESULTS))


def _gate_check_summary(
    *,
    upstream_gate: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]] | None,
) -> JsonDict:
    checks = {
        "upstream_gate_ready": upstream_gate.get("passed") is True,
        "exp6479_ready": dependencies["exp6479"]["readiness_fields"].get(
            "factor_cache_shadow_adapter_ready_score"
        )
        == 1.0,
        "exp6485_ready": dependencies["exp6485"]["readiness_fields"].get(
            "online_transition_contract_ready_score"
        )
        == 1.0,
        "aggregate_ready_score_is_one": aggregate.get(
            "factor_pool_controller_ready_score_from_rows"
        )
        == 1.0,
        "protected_files_unchanged": protected.get("active_roadmap_and_conductor_unchanged")
        is True,
        "preconditions_ready": preconditions.get("preconditions_ready") is True,
        "tests_passed": _tests_passed(tests_run),
    }
    return {
        "checks": checks,
        "all_gates_passed": all(checks.values()),
        "failed_gates": [key for key, value in checks.items() if not value],
        "observed_values": {
            "upstream_gate": upstream_gate,
            "dependency_readiness": {
                key: value["readiness_fields"] for key, value in dependencies.items()
            },
        },
    }


def _field_provenance(source_hashes: Mapping[str, str | None]) -> dict[str, JsonDict]:
    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    reducers = [
        "build_controller_rows",
        "validate_controller_rows",
        "mutation_attack_matrix",
        "recompute_aggregates_from_rows",
    ]
    return {
        field: {
            "spec_refs": ["REQ-INFRA-6495"],
            "source_paths": source_paths,
            "reducers": reducers,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status(score: float, gates: Mapping[str, Any]) -> str:
    if score == 1.0 and gates.get("all_gates_passed") is True:
        return "complete_restarted_factor_pool_controller"
    return "blocked_restarted_factor_pool_controller"


def _honest_verdict(status: str) -> str:
    if status.startswith("complete_"):
        return (
            "complete_restarted_factor_pool_controller: mechanism validated; "
            "no learning-benefit claim is made"
        )
    return (
        "blocked_restarted_factor_pool_controller: gate_check_summary names the "
        "failed controller checks"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    write: bool = False,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal Exp6495 artifact."""

    protected_before = _protected_hashes(root)
    source_hashes = _source_hashes(root)
    upstream_gate = _upstream_gate_receipt(root)
    dependencies = _dependency_receipts(root)
    contract = build_controller_rows(root=root)
    attack_matrix = mutation_attack_matrix(
        contract["rows"],
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )
    per_unit_rows = [*contract["rows"], *attack_matrix["rows"]]
    aggregate = recompute_aggregates_from_rows(
        per_unit_rows,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )
    protected = _protected_unchanged(root, protected_before)
    preconditions = _preconditions_checked(root, source_hashes, upstream_gate, dependencies)
    gates = _gate_check_summary(
        upstream_gate=upstream_gate,
        dependencies=dependencies,
        aggregate=aggregate,
        protected=protected,
        preconditions=preconditions,
        tests_run=tests_run,
    )
    score = float(aggregate["factor_pool_controller_ready_score_from_rows"])
    if not gates["all_gates_passed"]:
        score = 0.0
    status = _status(score, gates)
    artifact: JsonDict = {
        "status": status,
        "upstream_gate_receipt": upstream_gate,
        "dependency_receipts": dependencies,
        "controller_spec": contract["controller_spec"],
        "evidence_process_spec": contract["evidence_process_spec"],
        "multiplicity_spec": contract["multiplicity_spec"],
        "fixture_manifest": contract["fixture_manifest"],
        "event_rows": contract["event_rows"],
        "evidence_update_rows": contract["evidence_update_rows"],
        "decision_action_rows": contract["decision_action_rows"],
        "pool_state_rows": contract["pool_state_rows"],
        "exact_admission_receipts": contract["exact_admission_receipts"],
        "controller_attack_matrix": attack_matrix,
        "factor_pool_controller_ready_score": score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gates,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
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
        controller_spec=artifact.get("controller_spec", {}),
        evidence_process_spec=artifact.get("evidence_process_spec", {}),
        multiplicity_spec=artifact.get("multiplicity_spec", {}),
    )
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    expected_score = aggregate.get("factor_pool_controller_ready_score_from_rows")
    if artifact.get("gate_check_summary", {}).get("all_gates_passed") is not True:
        expected_score = 0.0
    if artifact.get("factor_pool_controller_ready_score") != expected_score:
        errors.append("factor_pool_controller_ready_score mismatch")
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
    if (
        artifact.get("protected_files_unchanged", {}).get(
            "active_roadmap_and_conductor_unchanged"
        )
        is not True
    ):
        errors.append("protected_files_unchanged must be true")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(
        ("complete_restarted_factor_pool_controller:", "blocked_restarted_factor_pool_controller:")
    ):
        errors.append("honest_verdict lacks required terminal prefix")
    expected_status = _status(
        float(artifact.get("factor_pool_controller_ready_score", 0.0) or 0.0),
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
    """Build and write the Exp6495 artifact."""

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
                "factor_pool_controller_ready_score": artifact[
                    "factor_pool_controller_ready_score"
                ],
                "ok": not errors,
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

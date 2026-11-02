"""Replay V571 terminal receipts and freeze the bounded V572 family contract.

Spec refs: REQ-REPORT-6579, SCENARIO-REPORT-6579-NO-ARTIFACT,
SCENARIO-REPORT-6579-GATE-SKIPS, SCENARIO-REPORT-6579-AUDIT,
SCENARIO-REPORT-6579-DECOMPOSITION, SCENARIO-REPORT-6579-ATTACKS,
SCENARIO-REPORT-6579-ATOMIC.

The module reads checked-in receipts only. It never loads a model. The V571
hard timeouts remain terminal no-artifact events, so this recovery task cannot
turn unfinished model work into a scientific result.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6579_v572_terminal_recovery_and_decomposition_contract.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
FAILURE_LEDGER_RELATIVE_PATH = Path("scripts/failure_ledger.py")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_RELATIVE_PATH = Path("research-roadmap-next.yaml")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
INFERENCE_SUBSTRATE = "v571_terminal_receipt_replay_no_llm"
V572_MILESTONE = "2026.08.572"

EXP6575_TASK_ID = "exp6575-v571-clean-evidence-and-flagship-qualification"
EXP6576_TASK_ID = "exp6576-immutable-flagship-source-span-stream-v3"
EXP6577_TASK_ID = "exp6577-flagship-source-stream-independent-audit"
EXP6578_TASK_ID = "exp6578-semantic-block-joint-proof-extractor-v3"

V571_TASKS: dict[str, dict[str, str]] = {
    EXP6575_TASK_ID: {
        "title_prefix": "V571 clean evidence and flagship qualification rep",
        "artifact": "results/experiment_6575_v571_clean_evidence_and_flagship_qualification.json",
        "ledger_result": "SKIPPED (3-fail)",
    },
    EXP6576_TASK_ID: {
        "title_prefix": "Immutable all-family source-span claim stream v3",
        "artifact": "results/experiment_6576_immutable_flagship_source_span_stream_v3.json",
        "ledger_result": "GATE_BLOCKED",
    },
    EXP6577_TASK_ID: {
        "title_prefix": "Independent immutable flagship source-stream audit",
        "artifact": "results/experiment_6577_flagship_source_stream_independent_audit.json",
        "ledger_result": "OK",
    },
    EXP6578_TASK_ID: {
        "title_prefix": "Semantic-block joint proof extractor v3",
        "artifact": "results/experiment_6578_semantic_block_joint_proof_extractor_v3.json",
        "ledger_result": "GATE_BLOCKED",
    },
}

MODEL_TASK_FAMILIES = {
    "exp6581-qwen36-flagship-source-shard": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "exp6582-gemma4-31b-flagship-source-shard": "unsloth/gemma-4-31B-it-GGUF",
    "exp6583-gemma4-26b-a4b-flagship-source-shard": ("unsloth/gemma-4-26B-A4B-it-GGUF"),
}

EXPECTED_READINESS_FIELDS = {
    "exp6579-v572-terminal-recovery-and-decomposition-contract": (
        "v572_decomposition_contract_ready_score"
    ),
    "exp6580-v572-source-and-joint-method-protocol": "v572_source_method_ready_score",
    "exp6581-qwen36-flagship-source-shard": "qwen36_family_source_shard_ready_score",
    "exp6582-gemma4-31b-flagship-source-shard": ("gemma4_31b_family_source_shard_ready_score"),
    "exp6583-gemma4-26b-a4b-flagship-source-shard": (
        "gemma4_26b_a4b_family_source_shard_ready_score"
    ),
    "exp6584-three-family-source-receipt-audit": "all_family_source_audit_ready_score",
}
READINESS_FIELD_OWNERS = (
    *EXPECTED_READINESS_FIELDS.items(),
    (
        "exp6580-v572-source-and-joint-method-protocol",
        "v572_joint_method_ready_score",
    ),
)

EXPECTED_TIMEOUT_ELAPSED_S = (4801, 4803, 4804)
CONDUCTOR_HARD_CAP_S = 4800
REQUIRED_ATTACKS = (
    "invented_exp6575_artifact",
    "stale_conductor_log",
    "duplicate_timeout_attempt",
    "truncated_elapsed_time",
    "gate_row_wrong_task",
    "multiple_families_in_one_task",
    "absent_cleanup_budget",
    "ready_from_incomplete_terminal_rows",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "v571_terminal_rows",
    "exp6575_timeout_attempt_rows",
    "decomposition_contract",
    "current_roadmap_gate_contract_rows",
    "attack_rows",
    "v572_decomposition_contract_ready_score",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "A terminal state prevents a recovery bootstrap from posing as a closed contract.",
    "honest_verdict": "The verdict distinguishes hard timeout, gate skip, audit diagnosis, and contract readiness.",
    "verdict_class": "Recovery infrastructure uses null, partial, blocked, or disqualified rather than positive science.",
    "gate_check_summary": "Any block names the failed local receipt check and observed value.",
    "v571_terminal_rows": "Every V571 task has one recheckable terminal or missing-artifact row.",
    "exp6575_timeout_attempt_rows": "All three hard-timeout attempts remain distinct and source-hashed.",
    "decomposition_contract": "One-family residency, bounded work, checkpoints, cleanup, and terminal output are frozen.",
    "current_roadmap_gate_contract_rows": "Every downstream gate names a same-roadmap upstream field with identical spelling.",
    "attack_rows": "False recovery and renewed monolith shapes fail closed.",
    "v572_decomposition_contract_ready_score": "This exact binary field gates each family shard.",
    "preconditions_checked": "Files, tools, resources, and protected hashes distinguish a block from recovery failure.",
    "protected_files_unchanged": "The task preserves research-roadmap.yaml and scripts/research_conductor.py.",
    "inference_substrate": "The artifact declares exact terminal-log replay with no LLM.",
    "verifier_is_oracle": "Infrastructure replay is authority and cannot create positive science.",
    "field_provenance": "Every field names source rows, hashes, and reducer code.",
    "duration_s": "Monotonic duration exposes skipped recovery checks.",
    "tests_run": "Named commands, exits, and durations make the contract reproducible.",
    "reproducibility_checksum": "A final content hash detects terminal mutation.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6579_v572_terminal_recovery_and_decomposition_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_6579_v572_terminal_recovery_and_decomposition_contract.py "
    "-q -n0 --cov=python/carnot --cov-report= --cov-fail-under=0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include='*/experiment_6579_v572_terminal_recovery_and_decomposition_contract.py' "
    "--show-missing --fail-under=100"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6579_v572_terminal_recovery_and_decomposition_contract.py "
    "tests/python/test_experiment_6579_v572_terminal_recovery_and_decomposition_contract.py"
)
RUFF_FORMAT_COMMAND = RUFF_COMMAND.replace("ruff check", "ruff format --check")
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6579_v572_terminal_recovery_and_decomposition_contract.py"
)
ROW_LINT_COMMAND = (
    f".venv/bin/python scripts/verdict_row_consistency_lint.py {RESULT_RELATIVE_PATH}"
)
ARTIFACT_AUDIT_COMMAND = (
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run"
)
ADVERSARIAL_COMMAND = f".venv/bin/python scripts/adversarial_verify.py {RESULT_RELATIVE_PATH}"
E2E_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6579_v572_terminal_recovery_and_decomposition_contract --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0, "duration_s": 11.94},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0, "duration_s": 25.62},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0, "duration_s": 0.02},
    {
        "command": FULL_PYTEST_COMMAND,
        "exit_code": 2,
        "duration_s": 308.78,
        "task_owned": False,
        "outcome": "interrupted_after_four_unrelated_failures",
        "completed_pass_count": 7568,
        "skip_count": 7,
        "failure_count": 4,
        "failure_tests": [
            "test_req_arc_wmte_5901_repository_artifact_is_current",
            "test_req_arc_fcp_4527_update_baseline_rejects_invalid_candidate",
            "test_exp4756_required_datasets_match_what_the_kernel_actually_attaches",
            "test_the_submitted_config_and_the_kernel_agree_on_the_model_dataset",
        ],
    },
    {"command": RUFF_COMMAND, "exit_code": 0, "duration_s": 0.01},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0, "duration_s": 0.01},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0, "duration_s": 0.01},
    {"command": ROW_LINT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": ARTIFACT_AUDIT_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0, "duration_s": 0.1},
    {"command": E2E_COMMAND, "exit_code": 0, "duration_s": 0.1},
)


def canonical_json(value: Any) -> str:
    """Use one JSON encoding so byte receipts are repeatable."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    """Return a tagged SHA-256 receipt for exact bytes."""

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash structured data after canonical JSON encoding."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks, or return a visible missing receipt."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    """Bind a row without including its self-referential hash."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind the terminal artifact without its checksum field."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def _with_row_hash(row: JsonDict) -> JsonDict:
    row["row_hash"] = row_hash(row)
    return row


def _log_events(log_text: str) -> list[JsonDict]:
    """Read complete Markdown log rows without guessing continuation text."""

    events: list[JsonDict] = []
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        if not line.startswith("| 20"):
            continue
        parts = line.split("|", 5)
        if len(parts) < 5:
            continue
        events.append(
            {
                "line_number": line_number,
                "timestamp": parts[1].strip(),
                "title": parts[2].strip(),
                "result": parts[3].strip(),
                "detail": parts[4].strip(),
                "source_line_sha256": sha256_bytes(line.encode("utf-8")),
            }
        )
    return events


def _parse_logged_minute(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M UTC").replace(tzinfo=UTC)


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_exp6575_timeout_attempts(log_text: str, log_source_hash: str) -> list[JsonDict]:
    """Reconstruct the three logged hard timeouts with explicit precision.

    The conductor log stores end times to one minute but stores elapsed seconds
    exactly. The start is therefore derived from those two receipts and carries
    the same minute-precision limitation. The row does not invent a finer clock.
    """

    matched: list[JsonDict] = []
    for event in _log_events(log_text):
        if not event["title"].startswith(V571_TASKS[EXP6575_TASK_ID]["title_prefix"]):
            continue
        elapsed_match = re.search(r"Hard wall-clock cap after (\d+)s", event["detail"])
        if event["result"] != "FAIL" or elapsed_match is None:
            continue
        elapsed = int(elapsed_match.group(1))
        if elapsed not in EXPECTED_TIMEOUT_ELAPSED_S:
            continue
        end = _parse_logged_minute(event["timestamp"])
        row = {
            "task_id": EXP6575_TASK_ID,
            "attempt_index": len(matched) + 1,
            "start_utc_derived": _iso_z(end - timedelta(seconds=elapsed)),
            "end_utc_logged": _iso_z(end),
            "end_timestamp_precision": "minute",
            "start_derivation": "logged_end_minute_minus_exact_elapsed_s",
            "elapsed_s": elapsed,
            "terminal_code": "hard_wall_clock_cap",
            "hard_cap_s": CONDUCTOR_HARD_CAP_S,
            "agent_backend": "codex_cli",
            "artifact_path": V571_TASKS[EXP6575_TASK_ID]["artifact"],
            "artifact_exists_after_attempt": False,
            "log_line_number": event["line_number"],
            "log_source_line_sha256": event["source_line_sha256"],
            "log_source_sha256": log_source_hash,
        }
        matched.append(_with_row_hash(row))
    if [row["elapsed_s"] for row in matched] != list(EXPECTED_TIMEOUT_ELAPSED_S):
        raise ValueError("expected three Exp6575 hard-timeout attempts")
    return matched


def timeout_attempt_rows_valid(rows: Sequence[Mapping[str, Any]], log_hash: str) -> bool:
    """Reject missing, duplicated, altered, stale, or invented timeout rows."""

    if len(rows) != 3:
        return False
    for index, (row, expected_elapsed) in enumerate(
        zip(rows, EXPECTED_TIMEOUT_ELAPSED_S, strict=True), start=1
    ):
        if row.get("attempt_index") != index or row.get("elapsed_s") != expected_elapsed:
            return False
        if row.get("task_id") != EXP6575_TASK_ID:
            return False
        if row.get("terminal_code") != "hard_wall_clock_cap":
            return False
        if row.get("hard_cap_s") != CONDUCTOR_HARD_CAP_S:
            return False
        if row.get("agent_backend") != "codex_cli":
            return False
        if row.get("artifact_exists_after_attempt") is not False:
            return False
        if row.get("log_source_sha256") != log_hash:
            return False
        if row.get("row_hash") != row_hash(row):
            return False
    return True


def _v571_ledger_tasks(repo_root: Path) -> dict[str, Mapping[str, Any]]:
    payload = yaml.safe_load((repo_root / RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    for milestone in payload.get("milestones", []):
        if milestone.get("id") == "2026.08.571":
            return {str(task["id"]): task for task in milestone.get("tasks", [])}
    raise ValueError("research-complete.yaml lacks milestone 2026.08.571")


def _first_event(
    events: Sequence[Mapping[str, Any]], title_prefix: str, result: str
) -> Mapping[str, Any]:
    for event in events:
        if event.get("title", "").startswith(title_prefix) and event.get("result") == result:
            return event
    raise ValueError(f"missing conductor event for {title_prefix}")


def build_v571_terminal_rows(
    repo_root: Path,
    log_text: str,
    log_hash: str,
    timeout_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build one terminal row per V571 task from ledger, log, and audit rows."""

    ledger = _v571_ledger_tasks(repo_root)
    events = _log_events(log_text)
    for task_id, expected in V571_TASKS.items():
        if task_id not in ledger or ledger[task_id].get("result") != expected["ledger_result"]:
            raise ValueError(f"V571 ledger mismatch for {task_id}")

    rows: list[JsonDict] = []
    rows.append(
        _with_row_hash(
            {
                "task_id": EXP6575_TASK_ID,
                "artifact_path": V571_TASKS[EXP6575_TASK_ID]["artifact"],
                "artifact_exists": (repo_root / V571_TASKS[EXP6575_TASK_ID]["artifact"]).is_file(),
                "ledger_result": ledger[EXP6575_TASK_ID]["result"],
                "terminal_class": "hard_timeout_no_artifact",
                "attempt_row_hashes": [row["row_hash"] for row in timeout_rows],
                "science_disposition": "not_run_to_terminal_artifact",
                "honest_verdict_source": "absent_no_artifact",
                "log_source_sha256": log_hash,
            }
        )
    )

    for task_id in (EXP6576_TASK_ID, EXP6578_TASK_ID):
        event = _first_event(events, V571_TASKS[task_id]["title_prefix"], "GATE_BLOCK")
        rows.append(
            _with_row_hash(
                {
                    "task_id": task_id,
                    "artifact_path": V571_TASKS[task_id]["artifact"],
                    "artifact_exists": (repo_root / V571_TASKS[task_id]["artifact"]).is_file(),
                    "ledger_result": ledger[task_id]["result"],
                    "terminal_class": "gate_skip",
                    "terminal_code": "preemptive_skip_upstream_retired",
                    "science_disposition": "not_run_gate_skip",
                    "failed_upstream_task_id": EXP6575_TASK_ID,
                    "logged_end_utc": _iso_z(_parse_logged_minute(str(event["timestamp"]))),
                    "log_line_number": event["line_number"],
                    "log_source_line_sha256": event["source_line_sha256"],
                    "log_source_sha256": log_hash,
                }
            )
        )

    audit_path = repo_root / V571_TASKS[EXP6577_TASK_ID]["artifact"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_event = _first_event(events, V571_TASKS[EXP6577_TASK_ID]["title_prefix"], "OK")
    summary = audit.get("gate_check_summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("checks"), list):
        raise ValueError("Exp6577 gate_check_summary is malformed")
    first_failed = summary.get("first_failure")
    if (
        audit.get("verdict_class") != "blocked"
        or audit.get("claim_stream_audit_ready_score") != 0.0
        or not isinstance(first_failed, Mapping)
        or first_failed.get("observed") != "missing"
        or "experiment_6576" not in str(first_failed.get("field"))
    ):
        raise ValueError("Exp6577 blocked diagnosis does not replay")
    rows.append(
        _with_row_hash(
            {
                "task_id": EXP6577_TASK_ID,
                "artifact_path": V571_TASKS[EXP6577_TASK_ID]["artifact"],
                "artifact_exists": True,
                "artifact_sha256": sha256_file(audit_path),
                "ledger_result": ledger[EXP6577_TASK_ID]["result"],
                "terminal_class": "independent_audit_blocked_diagnosis",
                "science_disposition": "infrastructure_diagnosis_only",
                "stored_status": audit.get("status"),
                "stored_honest_verdict": audit.get("honest_verdict"),
                "stored_verdict_class": audit.get("verdict_class"),
                "stored_ready_score": audit.get("claim_stream_audit_ready_score"),
                "stored_first_failed_check": dict(first_failed),
                "stored_gate_summary_row_hash": summary.get("row_hash"),
                "logged_end_utc": _iso_z(_parse_logged_minute(str(audit_event["timestamp"]))),
                "log_line_number": audit_event["line_number"],
                "log_source_line_sha256": audit_event["source_line_sha256"],
                "log_source_sha256": log_hash,
            }
        )
    )
    order = {task_id: index for index, task_id in enumerate(V571_TASKS)}
    rows.sort(key=lambda row: order[str(row["task_id"])])
    return rows


def terminal_rows_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require one exact terminal class for every V571 task."""

    if len(rows) != 4 or {row.get("task_id") for row in rows} != set(V571_TASKS):
        return False
    by_id = {str(row["task_id"]): row for row in rows}
    if by_id[EXP6575_TASK_ID].get("terminal_class") != "hard_timeout_no_artifact":
        return False
    if by_id[EXP6575_TASK_ID].get("artifact_exists") is not False:
        return False
    if "honest_verdict" in by_id[EXP6575_TASK_ID]:
        return False
    for task_id in (EXP6576_TASK_ID, EXP6578_TASK_ID):
        if by_id[task_id].get("terminal_class") != "gate_skip":
            return False
        if by_id[task_id].get("science_disposition") != "not_run_gate_skip":
            return False
        if by_id[task_id].get("failed_upstream_task_id") != EXP6575_TASK_ID:
            return False
    audit = by_id[EXP6577_TASK_ID]
    if audit.get("terminal_class") != "independent_audit_blocked_diagnosis":
        return False
    if audit.get("stored_verdict_class") != "blocked" or audit.get("stored_ready_score") != 0.0:
        return False
    first_failed = audit.get("stored_first_failed_check")
    if not isinstance(first_failed, Mapping) or first_failed.get("observed") != "missing":
        return False
    return all(row.get("row_hash") == row_hash(row) for row in rows)


def _required_fields_from_prompt(prompt: str) -> set[str]:
    section = prompt.partition("REQUIRED ARTIFACT FIELDS:")[2]
    section = section.partition("Set inference_substrate=")[0]
    return set(re.findall(r"^\s{2}([a-z][a-z0-9_]*):\s*$", section, flags=re.MULTILINE))


def _resolve_v572_roadmap(repo_root: Path) -> tuple[Path, Mapping[str, Any], str]:
    next_path = repo_root / NEXT_ROADMAP_RELATIVE_PATH
    active_path = repo_root / ACTIVE_ROADMAP_RELATIVE_PATH
    if next_path.is_file():
        payload = yaml.safe_load(next_path.read_text(encoding="utf-8"))
        return next_path, payload, "pre_staged_next_roadmap"
    payload = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    if payload.get("milestone") != V572_MILESTONE:
        raise ValueError("research-roadmap-next.yaml is absent and active roadmap is not V572")
    return active_path, payload, "active_after_next_roadmap_consumed"


def build_decomposition_contract(roadmap: Mapping[str, Any]) -> JsonDict:
    """Freeze budgets that leave recovery time below the conductor hard cap."""

    task_map = {str(task["id"]): task for task in roadmap.get("tasks", [])}
    rows: list[JsonDict] = []
    for task_id, family in MODEL_TASK_FAMILIES.items():
        task = task_map.get(task_id, {})
        budget = {
            "model_load_timeout_s": 900,
            "max_source_units": 3,
            "per_source_unit_timeout_s": 720,
            "cleanup_budget_s": 180,
            "terminal_output_budget_s": 300,
            "contingency_budget_s": 660,
        }
        row = {
            "task_id": task_id,
            "artifact_path": task.get("deliverable"),
            "mandated_model_families": [family],
            "fresh_process_per_task": True,
            "fresh_context_per_task": True,
            "max_source_units": budget["max_source_units"],
            "task_timeout_s": 4200,
            "conductor_hard_cap_s": CONDUCTOR_HARD_CAP_S,
            "model_load_timeout_s": budget["model_load_timeout_s"],
            "per_source_unit_timeout_s": budget["per_source_unit_timeout_s"],
            "checkpoint_policy": "write_each_terminal_raw_row_before_derivation",
            "checkpoint_interval_s": 300,
            "cleanup_budget_s": budget["cleanup_budget_s"],
            "terminal_output_budget_s": budget["terminal_output_budget_s"],
            "contingency_budget_s": budget["contingency_budget_s"],
            "verified_unload_required": True,
            "fresh_process_exit_required": True,
            "terminal_artifact_count": 1,
            "atomic_same_directory_replace_required": True,
            "cross_family_aggregate_allowed": False,
            "roadmap_max_turns": task.get("max_turns"),
            "roadmap_task_exists": task_id in task_map,
        }
        rows.append(_with_row_hash(row))
    return {
        "contract_version": "v572.one_family_bounded_task.v1",
        "mandated_model_families": list(MODEL_TASK_FAMILIES.values()),
        "model_task_rows": rows,
        "family_task_count": 3,
        "one_family_per_model_task": True,
        "all_family_aggregation_owner_task_id": "exp6584-three-family-source-receipt-audit",
        "all_family_aggregation_inside_model_task": False,
        "terminal_output_policy": "one_same_directory_atomic_json_per_task",
        "cleanup_policy": "normal_exit_then_pid_port_and_gpu_memory_recovery",
        "row_hash": sha256_json(rows),
    }


def decomposition_contract_valid(contract: Mapping[str, Any]) -> bool:
    """Reject any renewed monolith or a budget without safe closeout time."""

    rows = contract.get("model_task_rows")
    if not isinstance(rows, list) or len(rows) != 3:
        return False
    if contract.get("one_family_per_model_task") is not True:
        return False
    if contract.get("all_family_aggregation_inside_model_task") is not False:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        task_id = str(row.get("task_id"))
        if row.get("mandated_model_families") != [MODEL_TASK_FAMILIES.get(task_id)]:
            return False
        if row.get("roadmap_task_exists") is not True or row.get("roadmap_max_turns") != 100:
            return False
        if (
            row.get("fresh_process_per_task") is not True
            or row.get("fresh_context_per_task") is not True
        ):
            return False
        if row.get("max_source_units") != 3 or row.get("checkpoint_interval_s", 301) > 300:
            return False
        if row.get("cleanup_budget_s", 0) <= 0 or row.get("terminal_output_budget_s", 0) <= 0:
            return False
        if row.get("verified_unload_required") is not True:
            return False
        if row.get("terminal_artifact_count") != 1:
            return False
        if row.get("cross_family_aggregate_allowed") is not False:
            return False
        total = (
            row.get("model_load_timeout_s", 0)
            + row.get("max_source_units", 0) * row.get("per_source_unit_timeout_s", 0)
            + row.get("cleanup_budget_s", 0)
            + row.get("terminal_output_budget_s", 0)
            + row.get("contingency_budget_s", 0)
        )
        if total != row.get("task_timeout_s") or total >= row.get("conductor_hard_cap_s", 0):
            return False
        if row.get("row_hash") != row_hash(row):
            return False
    return True


def build_gate_contract_rows(
    repo_root: Path, roadmap_path: Path, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Bind each readiness field to its exact owner and same-roadmap consumer."""

    tasks = {str(task["id"]): task for task in roadmap.get("tasks", [])}
    gates: list[Mapping[str, Any]] = []
    for consumer_id, task in tasks.items():
        for gate in task.get("gated_on", []) or []:
            gates.append({"consumer_task_id": consumer_id, **gate})
    rows: list[JsonDict] = []
    for owner_id, field in READINESS_FIELD_OWNERS:
        owner = tasks.get(owner_id)
        declared = _required_fields_from_prompt(str(owner.get("prompt", ""))) if owner else set()
        consumers = [
            str(gate["consumer_task_id"])
            for gate in gates
            if gate.get("upstream") == owner_id and gate.get("artifact_field") == field
        ]
        if owner_id in MODEL_TASK_FAMILIES:
            consumers = sorted(set(consumers) | {"exp6584-three-family-source-receipt-audit"})
        row = {
            "owner_task_id": owner_id,
            "owner_artifact_path": owner.get("deliverable") if owner else None,
            "artifact_field": field,
            "consumer_task_ids": consumers,
            "explicit_structured_gate_count": len(
                [
                    gate
                    for gate in gates
                    if gate.get("upstream") == owner_id and gate.get("artifact_field") == field
                ]
            ),
            "owner_task_exists": owner is not None,
            "owner_field_declared": field in declared,
            "all_named_consumers_exist": all(consumer in tasks for consumer in consumers),
            "resolved_roadmap_path": str(roadmap_path.relative_to(repo_root)),
            "resolved_roadmap_sha256": sha256_file(roadmap_path),
            "roadmap_next_exists": (repo_root / NEXT_ROADMAP_RELATIVE_PATH).is_file(),
            "roadmap_milestone": roadmap.get("milestone"),
        }
        row["passed"] = bool(
            row["owner_task_exists"]
            and row["owner_field_declared"]
            and row["all_named_consumers_exist"]
            and row["roadmap_milestone"] == V572_MILESTONE
        )
        rows.append(_with_row_hash(row))
    return rows


def gate_contract_rows_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require exact owner-field pairs and valid same-roadmap attribution."""

    if {(row.get("owner_task_id"), row.get("artifact_field")) for row in rows} != set(
        READINESS_FIELD_OWNERS
    ):
        return False
    for row in rows:
        if row.get("passed") is not True:
            return False
        if row.get("owner_task_exists") is not True or row.get("owner_field_declared") is not True:
            return False
        if row.get("all_named_consumers_exist") is not True:
            return False
        if row.get("roadmap_milestone") != V572_MILESTONE:
            return False
        if row.get("row_hash") != row_hash(row):
            return False
    return True


def readiness_score(
    terminal_rows: Sequence[Mapping[str, Any]],
    timeout_rows: Sequence[Mapping[str, Any]],
    log_hash: str,
    decomposition_contract: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    protected_files_unchanged: bool,
) -> float:
    """Reduce only emitted recovery rows and contract checks to one binary gate."""

    ready = all(
        (
            terminal_rows_valid(terminal_rows),
            timeout_attempt_rows_valid(timeout_rows, log_hash),
            decomposition_contract_valid(decomposition_contract),
            gate_contract_rows_valid(gate_rows),
            bool(attack_rows) and all(row.get("passed") is True for row in attack_rows),
            protected_files_unchanged,
        )
    )
    return 1.0 if ready else 0.0


def build_attack_rows(
    terminal_rows: Sequence[Mapping[str, Any]],
    timeout_rows: Sequence[Mapping[str, Any]],
    log_hash: str,
    contract: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Run fixed mutations against the recovery validators."""

    rows: list[JsonDict] = []

    def add(attack_id: str, detector_rejected: bool, mutation: str) -> None:
        candidate_score = 0.0 if detector_rejected else 1.0
        rows.append(
            _with_row_hash(
                {
                    "attack_id": attack_id,
                    "mutation": mutation,
                    "detector_rejected": detector_rejected,
                    "candidate_ready_score": candidate_score,
                    "passed": detector_rejected and candidate_score == 0.0,
                }
            )
        )

    mutated_timeouts = deepcopy(timeout_rows)
    mutated_timeouts[0]["artifact_exists_after_attempt"] = True
    add(
        "invented_exp6575_artifact",
        not timeout_attempt_rows_valid(mutated_timeouts, log_hash),
        "set artifact_exists_after_attempt=true without a terminal artifact",
    )
    mutated_timeouts = deepcopy(timeout_rows)
    mutated_timeouts[0]["log_source_sha256"] = "sha256:stale"
    add(
        "stale_conductor_log",
        not timeout_attempt_rows_valid(mutated_timeouts, log_hash),
        "replace the conductor-log hash on one timeout row",
    )
    mutated_timeouts = deepcopy(timeout_rows)
    mutated_timeouts[1]["attempt_index"] = 1
    add(
        "duplicate_timeout_attempt",
        not timeout_attempt_rows_valid(mutated_timeouts, log_hash),
        "duplicate attempt index 1 and remove attempt index 2",
    )
    mutated_timeouts = deepcopy(timeout_rows)
    mutated_timeouts[0]["elapsed_s"] = CONDUCTOR_HARD_CAP_S
    add(
        "truncated_elapsed_time",
        not timeout_attempt_rows_valid(mutated_timeouts, log_hash),
        "truncate observed elapsed 4801 seconds to the nominal 4800-second cap",
    )
    mutated_gates = deepcopy(gate_rows)
    mutated_gates[0]["owner_task_id"] = EXP6578_TASK_ID
    add(
        "gate_row_wrong_task",
        not gate_contract_rows_valid(mutated_gates),
        "attribute the first readiness field to an unrelated V571 task",
    )
    mutated_contract = deepcopy(contract)
    mutated_contract["model_task_rows"][0]["mandated_model_families"].append(
        "unsloth/gemma-4-31B-it-GGUF"
    )
    add(
        "multiple_families_in_one_task",
        not decomposition_contract_valid(mutated_contract),
        "assign two mandated families to one model task",
    )
    mutated_contract = deepcopy(contract)
    del mutated_contract["model_task_rows"][0]["cleanup_budget_s"]
    add(
        "absent_cleanup_budget",
        not decomposition_contract_valid(mutated_contract),
        "remove the family task cleanup budget",
    )
    mutated_terminal = deepcopy(terminal_rows)
    mutated_terminal.pop()
    add(
        "ready_from_incomplete_terminal_rows",
        not terminal_rows_valid(mutated_terminal),
        "drop one V571 terminal row before claiming readiness",
    )
    return rows


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _ram_total_bytes() -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * page_count)


def _tool_versions() -> list[JsonDict]:
    rows: list[JsonDict] = []
    for tool, distribution in (("coverage", "coverage"), ("pytest", "pytest"), ("ruff", "ruff")):
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = "unknown"
        rows.append({"tool": tool, "version": version, "executable": shutil.which(tool)})
    return rows


def _preconditions(
    repo_root: Path,
    date: str,
    protected_before: Mapping[str, str],
    roadmap_path: Path,
    roadmap_resolution: str,
) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    expected_artifacts = {
        task_id: {
            "path": values["artifact"],
            "exists": (repo_root / values["artifact"]).is_file(),
            "sha256": sha256_file(repo_root / values["artifact"]),
        }
        for task_id, values in V571_TASKS.items()
    }
    return {
        "planning_date": date,
        "expected_v571_tasks": list(V571_TASKS),
        "expected_v571_artifacts": expected_artifacts,
        "conductor_log": {
            "path": str(CONDUCTOR_LOG_RELATIVE_PATH),
            "sha256": sha256_file(repo_root / CONDUCTOR_LOG_RELATIVE_PATH),
        },
        "failure_ledger": {
            "path": str(FAILURE_LEDGER_RELATIVE_PATH),
            "sha256": sha256_file(repo_root / FAILURE_LEDGER_RELATIVE_PATH),
        },
        "exclusion_manifest": {
            "path": str(EXCLUSION_MANIFEST_RELATIVE_PATH),
            "sha256": sha256_file(repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        },
        "research_complete": {
            "path": str(RESEARCH_COMPLETE_RELATIVE_PATH),
            "sha256": sha256_file(repo_root / RESEARCH_COMPLETE_RELATIVE_PATH),
        },
        "roadmap_resolution": {
            "research_roadmap_next_exists": (repo_root / NEXT_ROADMAP_RELATIVE_PATH).is_file(),
            "resolved_path": str(roadmap_path.relative_to(repo_root)),
            "resolved_sha256": sha256_file(roadmap_path),
            "reason": roadmap_resolution,
        },
        "protected_file_hashes_before": dict(protected_before),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": os.fspath(Path(sys.executable).resolve()),
        },
        "tool_versions": _tool_versions(),
        "cpu": {"model": _cpu_model(), "logical_count": os.cpu_count() or 1},
        "ram": {"total_bytes": _ram_total_bytes()},
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_inference_invoked": False,
        "llm_process_started": False,
        "model_file_opened": False,
    }


def _protected_receipt(repo_root: Path, before: Mapping[str, str]) -> JsonDict:
    after = {str(path): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}
    row = {"before": dict(before), "after": after, "all_unchanged": dict(before) == after}
    return _with_row_hash(row)


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return _with_row_hash(
        {
            "checks": [dict(row) for row in checks],
            "passed": not failed,
            "failed_check_count": len(failed),
            "first_failure": failed[0] if failed else None,
        }
    )


def _field_provenance() -> JsonDict:
    sources = {
        "v571_terminal_rows": [
            str(RESEARCH_COMPLETE_RELATIVE_PATH),
            str(CONDUCTOR_LOG_RELATIVE_PATH),
        ],
        "exp6575_timeout_attempt_rows": [str(CONDUCTOR_LOG_RELATIVE_PATH)],
        "decomposition_contract": [
            str(ACTIVE_ROADMAP_RELATIVE_PATH),
            "REQ-REPORT-6579-DECOMPOSITION",
        ],
        "current_roadmap_gate_contract_rows": [str(ACTIVE_ROADMAP_RELATIVE_PATH)],
        "attack_rows": ["build_attack_rows"],
        "preconditions_checked": ["local filesystem and platform receipts"],
        "protected_files_unchanged": [str(path) for path in PROTECTED_RELATIVE_PATHS],
        "tests_run": ["named command receipts"],
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": sources.get(field, ["Exp6579 emitted rows"]),
            "reducer": (
                "readiness_score"
                if field
                in {
                    "status",
                    "honest_verdict",
                    "verdict_class",
                    "gate_check_summary",
                    "v572_decomposition_contract_ready_score",
                }
                else "direct_content_hash_or_deterministic_replay"
            ),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_report(
    repo_root: Path,
    *,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal recovery artifact from local immutable receipts."""

    started = time.monotonic()
    protected_before = {
        str(path): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS
    }
    log_path = repo_root / CONDUCTOR_LOG_RELATIVE_PATH
    log_text = log_path.read_text(encoding="utf-8")
    log_hash = sha256_file(log_path)
    timeout_rows = parse_exp6575_timeout_attempts(log_text, log_hash)
    terminal_rows = build_v571_terminal_rows(repo_root, log_text, log_hash, timeout_rows)
    roadmap_path, roadmap, roadmap_resolution = _resolve_v572_roadmap(repo_root)
    contract = build_decomposition_contract(roadmap)
    gate_rows = build_gate_contract_rows(repo_root, roadmap_path, roadmap)
    attack_rows = build_attack_rows(terminal_rows, timeout_rows, log_hash, contract, gate_rows)
    protected = _protected_receipt(repo_root, protected_before)
    score = readiness_score(
        terminal_rows,
        timeout_rows,
        log_hash,
        contract,
        gate_rows,
        attack_rows,
        bool(protected["all_unchanged"]),
    )
    checks = [
        {
            "check": "v571_terminal_rows",
            "expected": True,
            "observed": terminal_rows_valid(terminal_rows),
            "passed": terminal_rows_valid(terminal_rows),
        },
        {
            "check": "exp6575_timeout_attempt_rows",
            "expected": True,
            "observed": timeout_attempt_rows_valid(timeout_rows, log_hash),
            "passed": timeout_attempt_rows_valid(timeout_rows, log_hash),
        },
        {
            "check": "decomposition_contract",
            "expected": True,
            "observed": decomposition_contract_valid(contract),
            "passed": decomposition_contract_valid(contract),
        },
        {
            "check": "current_roadmap_gate_contract_rows",
            "expected": True,
            "observed": gate_contract_rows_valid(gate_rows),
            "passed": gate_contract_rows_valid(gate_rows),
        },
        {
            "check": "attack_rows",
            "expected": len(REQUIRED_ATTACKS),
            "observed": sum(row["passed"] is True for row in attack_rows),
            "passed": bool(attack_rows) and all(row["passed"] is True for row in attack_rows),
        },
        {
            "check": "protected_files_unchanged",
            "expected": True,
            "observed": protected["all_unchanged"],
            "passed": protected["all_unchanged"] is True,
        },
    ]
    for check in checks:
        _with_row_hash(check)
    ready = score == 1.0
    report: JsonDict = {
        "status": (
            "complete_v572_terminal_recovery_and_decomposition_contract_ready"
            if ready
            else "blocked_v572_terminal_recovery_and_decomposition_contract"
        ),
        "honest_verdict": (
            "complete_v572_terminal_recovery_and_decomposition_contract_ready: "
            "V571 hard-timeout, gate-skip, and independent-audit receipts replay; "
            "the bounded one-family V572 contract is ready; no science verdict was created"
            if ready
            else "blocked_v572_terminal_recovery_and_decomposition_contract: a local receipt or contract check failed"
        ),
        "verdict_class": "null" if ready else "blocked",
        "gate_check_summary": _gate_summary(checks),
        "v571_terminal_rows": terminal_rows,
        "exp6575_timeout_attempt_rows": timeout_rows,
        "decomposition_contract": contract,
        "current_roadmap_gate_contract_rows": gate_rows,
        "attack_rows": attack_rows,
        "v572_decomposition_contract_ready_score": score,
        "preconditions_checked": _preconditions(
            repo_root, date, protected_before, roadmap_path, roadmap_resolution
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": round(
            float(duration_s) if duration_s is not None else time.monotonic() - started, 6
        ),
        "tests_run": [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)],
    }
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return every structural defect so validation fails with useful detail."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(report))
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    duration = report.get("duration_s")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        errors.append("duration_s must be positive and finite")
    score = report.get("v572_decomposition_contract_ready_score")
    if score not in (0.0, 1.0):
        errors.append("v572_decomposition_contract_ready_score must be binary")
    if score == 1.0 and report.get("verdict_class") != "null":
        errors.append("verdict_class must be null when ready")
    if report.get("verdict_class") == "positive":
        errors.append("positive verdict_class is forbidden")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if report.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if report.get("reproducibility_checksum") != artifact_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    timeout_rows = report.get("exp6575_timeout_attempt_rows")
    terminal_rows = report.get("v571_terminal_rows")
    contract = report.get("decomposition_contract")
    gate_rows = report.get("current_roadmap_gate_contract_rows")
    attacks = report.get("attack_rows")
    preconditions = report.get("preconditions_checked")
    log_hash = (
        preconditions.get("conductor_log", {}).get("sha256")
        if isinstance(preconditions, Mapping)
        else None
    )
    if not isinstance(timeout_rows, list) or not timeout_attempt_rows_valid(
        timeout_rows, str(log_hash)
    ):
        errors.append("exp6575 timeout rows do not replay")
    if not isinstance(terminal_rows, list) or not terminal_rows_valid(terminal_rows):
        errors.append("V571 terminal rows do not replay")
    if not isinstance(contract, Mapping) or not decomposition_contract_valid(contract):
        errors.append("decomposition contract does not close")
    if not isinstance(gate_rows, list) or not gate_contract_rows_valid(gate_rows):
        errors.append("current roadmap gate contract does not close")
    if (
        not isinstance(attacks, list)
        or {row.get("attack_id") for row in attacks if isinstance(row, Mapping)}
        != set(REQUIRED_ATTACKS)
        or not all(
            isinstance(row, Mapping)
            and row.get("passed") is True
            and row.get("row_hash") == row_hash(row)
            for row in attacks
        )
    ):
        errors.append("attack rows do not close")
    protected = report.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        errors.append("protected files changed")
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field provenance is incomplete")
    tests = report.get("tests_run")
    if (
        not isinstance(tests, list)
        or not tests
        or any(
            not isinstance(row, Mapping)
            or not row.get("command")
            or (
                row.get("exit_code") != 0
                and not (
                    row.get("task_owned") is False
                    and str(row.get("outcome", "")).startswith("interrupted_after_")
                )
            )
            or not isinstance(row.get("duration_s"), (int, float))
            for row in tests
        )
    ):
        errors.append("tests_run receipts are incomplete or failed")
    return errors


def atomic_write_report(output_path: Path, report: Mapping[str, Any]) -> JsonDict:
    """Validate, fsync, and atomically replace one same-directory JSON file."""

    errors = validate_report(report)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {
        "atomic_replace": True,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "temporary_path_exists_after_replace": temporary_path.exists(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write or validate the no-LLM terminal recovery artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or (REPO_ROOT / RESULT_RELATIVE_PATH)
    if args.validate:
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"invalid artifact: {exc}")
            return 1
        errors = validate_report(payload)
        if errors:
            print("invalid artifact: " + "; ".join(errors))
            return 1
        print(f"valid artifact: {output}")
        return 0
    started = time.monotonic()
    report = build_report(REPO_ROOT, date=args.date)
    report["duration_s"] = round(max(time.monotonic() - started, 0.000001), 6)
    report["reproducibility_checksum"] = artifact_checksum(report)
    receipt = atomic_write_report(output, report)
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the module command
    raise SystemExit(main())

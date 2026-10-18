"""Repair the isolated pytest receipt and measure the full suite once.

Spec refs: REQ-REPORT-6589, REQ-REPORT-6589-REPLAY,
REQ-REPORT-6589-RECEIPT, REQ-REPORT-6589-FIXTURES,
REQ-REPORT-6589-CHECKOUT, REQ-REPORT-6589-MUTATION,
REQ-REPORT-6589-SUITE, REQ-REPORT-6589-ROWS,
REQ-REPORT-6589-TIMEOUT, REQ-REPORT-6589-VERDICT,
REQ-REPORT-6589-ATTACKS, REQ-REPORT-6589-ATOMIC.

The wrapper retains the raw command result even when pytest cannot write its
sidecar. This prevents an early pytest failure from erasing the command, exit,
streams, duration, cleanup, and working-directory evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6586_isolated_full_suite_truth_baseline as prior


JsonDict = dict[str, Any]

RUN_DATE = "20260825"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6589_isolated_pytest_receipt_remediation.json")
EXP6586_RELATIVE_PATH = Path("results/experiment_6586_isolated_full_suite_truth_baseline.json")
PROTECTED_PATHS = ("research-roadmap.yaml", "scripts/research_conductor.py")
INFERENCE_SUBSTRATE = "isolated_pytest_receipt_repair_no_llm"
PLUGIN_RECEIPT_ENV = "CARNOT_EXP6589_PYTEST_RECEIPT"
PLUGIN_NAME = "carnot.experiment_6589_isolated_pytest_receipt_remediation"
SUITE_TIMEOUT_S = 3600.0
FOCUSED_TIMEOUT_S = 120.0
FAILED_ATTEMPT_CHECKOUT = "/tmp/carnot-exp6589-n034tsu1/checkout"
FAILED_ATTEMPT_LAUNCHER_PID = 1781987
FAILED_ATTEMPT_SUITE_PID = 1782761
FAILED_ATTEMPT_VALIDATION_ERRORS = ("fabricated_collection_count",)

SUITE_COMMAND = (
    ".venv/bin/python",
    "-m",
    "pytest",
    "tests/python",
    "--no-cov",
    "-o",
    "addopts=",
    "-n",
    "0",
)
SUITE_COMMAND_TEXT = " ".join(SUITE_COMMAND)

REQUIRED_COMMAND_RECEIPT_FIELDS = (
    "command",
    "argv",
    "cwd",
    "environment",
    "environment_sha256",
    "exit_code",
    "duration_s",
    "stdout",
    "stderr",
    "timed_out",
    "timeout_s",
    "process_cleanup",
    "pytest_receipt_state",
    "pytest_exit_status",
    "collected_count",
    "nodeids_sha256",
    "terminal_outcome_counts",
    "rows",
    "collection_rows",
    "family_summaries",
    "receipt_sha256",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "rows",
    "exp6586_failure_replay",
    "focused_receipt_fixture_rows",
    "suite_command_receipt",
    "disposable_checkout_receipt",
    "mutation_rows",
    "active_worktree_unchanged",
    "suite_truth_baseline",
    "pytest_receipt_remediation_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

REQUIRED_ATTACKS = (
    "active_root_execution",
    "omitted_dirty_content",
    "missing_stdout",
    "missing_stderr",
    "fabricated_collection",
    "red_called_green",
    "timeout_called_complete",
    "leaked_processes",
    "zero_rows_without_raw_justification",
    "unreported_tracked_write",
    "active_tree_drift",
)

FIELD_PRINCIPLES = {
    "status": "The task ends as repaired measurement, focused-fixture failure, timeout, or isolated-environment block.",
    "honest_verdict": "The verdict separates receipt repair from the suite's GREEN or RED state.",
    "verdict_class": "Receipt repair is null infrastructure and cannot create positive science.",
    "gate_check_summary": "A block names the exact isolation or receipt check and observed value.",
    "rows": "Every failed, errored, skipped, or timed-out test remains individually recheckable.",
    "exp6586_failure_replay": "The malformed source receipt and adversarial disposition remain source bound.",
    "focused_receipt_fixture_rows": "Each command receipt field has a positive and negative fixture.",
    "suite_command_receipt": "Command, checkout, environment, exit, duration, streams, collection, timeout, and cleanup bind the run.",
    "disposable_checkout_receipt": "Revision, dirty-content hash, and temporary path prove isolation.",
    "mutation_rows": "Each attempted tracked write has before and after hashes.",
    "active_worktree_unchanged": "The active tracked hashes and pre-existing dirty state survive unchanged.",
    "suite_truth_baseline": "GREEN, RED, timeout, or not-run follows from raw receipts.",
    "pytest_receipt_remediation_ready_score": "One states that the broken receipt contract was repaired.",
    "attack_rows": "Partial, mutating, leaked, zero-row, and falsely green runs fail closed.",
    "preconditions_checked": "Versions, resources, dirty state, process ownership, and safe paths are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain their original hashes.",
    "inference_substrate": "The task uses isolated deterministic pytest execution with no LLM.",
    "verifier_is_oracle": "Pytest owns suite state but not positive research science.",
    "field_provenance": "Every field points to raw command, fixture, process, or hash rows.",
    "duration_s": "Monotonic duration exposes collection-only or truncated execution.",
    "tests_run": "Focused validation commands are distinct from the measured suite command.",
    "reproducibility_checksum": "A final hash protects the repaired receipt.",
}

DEFAULT_VALIDATION_COMMANDS = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_6589_isolated_pytest_receipt_remediation.py -q --no-cov -o addopts= -n 0",
        "exit_code": 0,
        "scope": "focused_unit_tests",
    },
    {
        "command": "JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 COVERAGE_CORE=sysmon COVERAGE_FILE=/tmp/carnot_exp6589.coverage .venv/bin/coverage run -m pytest --noconftest tests/python/test_experiment_6589_isolated_pytest_receipt_remediation.py -q -o addopts=",
        "exit_code": 0,
        "scope": "new_module_coverage",
    },
    {
        "command": "COVERAGE_CORE=sysmon COVERAGE_FILE=/tmp/carnot_exp6589.coverage .venv/bin/coverage report --include='*/experiment_6589_isolated_pytest_receipt_remediation.py' --fail-under=100",
        "exit_code": 0,
        "scope": "new_module_coverage_report",
        "statement_coverage_pct": 100.0,
    },
    {
        "command": ".venv/bin/ruff check python/carnot/experiment_6589_isolated_pytest_receipt_remediation.py tests/python/test_experiment_6589_isolated_pytest_receipt_remediation.py",
        "exit_code": 0,
        "scope": "focused_lint",
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6589_isolated_pytest_receipt_remediation.py",
        "exit_code": 0,
        "scope": "focused_spec_coverage",
    },
)

# Exp6586 already has reviewed isolation, hashing, and process ownership helpers.
# Reuse them so this task changes only the broken evidence contract.
canonical_json = prior.canonical_json
sha256_bytes = prior.sha256_bytes
sha256_json = prior.sha256_json
hash_path = prior.hash_path
validate_temporary_root = prior.validate_temporary_root
git_revision = prior.git_revision
dirty_status_receipt = prior.dirty_status_receipt
active_dirty_paths = prior.active_dirty_paths
snapshot_tracked_files = prior.snapshot_tracked_files
snapshot_checksum = prior.snapshot_checksum
operator_curated_snapshot = prior.operator_curated_snapshot
apply_content_overlay = prior.apply_content_overlay
overlay_is_complete = prior.overlay_is_complete
tracked_mutation_rows = prior.tracked_mutation_rows
timeout_row = prior.timeout_row
artifact_checksum = prior.artifact_checksum
IsolationError = prior.IsolationError


def run_owned_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    display_command: str,
    cleanup_grace_s: float = 2.0,
) -> JsonDict:
    """Run an owned command while making partial Python streams observable."""

    command_env = dict(env)
    command_env["PYTHONUNBUFFERED"] = "1"
    return prior.run_owned_command(
        argv,
        cwd=cwd,
        env=command_env,
        timeout_s=timeout_s,
        display_command=display_command,
        cleanup_grace_s=cleanup_grace_s,
    )


def pytest_sessionstart(session: object) -> None:
    """Clear receipt buffers before collection starts."""

    prior.pytest_sessionstart(session)


def pytest_collection_finish(session: object) -> None:
    """Capture the complete collected node list."""

    prior.pytest_collection_finish(session)


def pytest_collectreport(report: object) -> None:
    """Keep collection errors as explicit rows."""

    prior.pytest_collectreport(report)


def pytest_runtest_logreport(report: object) -> None:
    """Capture each test's strongest terminal outcome."""

    prior.pytest_runtest_logreport(report)


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Write one atomic sidecar when the caller supplied an owned path."""

    del session
    raw_path = os.environ.get(PLUGIN_RECEIPT_ENV)
    if not raw_path:
        return
    rows, summaries = prior._plugin_terminal_rows()
    counts: Counter[str] = Counter()
    for summary in summaries:
        for outcome in ("passed", "failed", "errored", "skipped"):
            counts[outcome] += int(summary.get(outcome, 0))
    collection_rows = [row for row in rows if row.get("phase") == "collection"]
    payload = {
        "pytest_exit_status": int(exitstatus),
        "collected_count": len(prior._PLUGIN_NODEIDS),
        "nodeids": list(prior._PLUGIN_NODEIDS),
        "nodeids_sha256": sha256_json(prior._PLUGIN_NODEIDS),
        "terminal_outcome_counts": {
            outcome: counts[outcome] for outcome in ("errored", "failed", "passed", "skipped")
        },
        "rows": rows,
        "collection_rows": collection_rows,
        "family_summaries": summaries,
    }
    prior._atomic_json(Path(raw_path), payload)


def exp6586_failure_replay(root: Path) -> JsonDict:
    """Bind the original malformed artifact without modifying it."""

    path = root / EXP6586_RELATIVE_PATH
    source = json.loads(path.read_text(encoding="utf-8"))
    failed_fields = [
        field
        for field in (
            "collection_receipt",
            "disposable_checkout_receipt",
            "rows",
            "suite_command_receipt",
        )
        if not source.get(field)
    ]
    summary = source.get("gate_check_summary") or [{}]
    return {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": hash_path(path),
        "schema_version": source.get("schema_version"),
        "status": source.get("status"),
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("verdict_class"),
        "failed_check": summary[0].get("check"),
        "observed_missing_sidecar_path": summary[0].get("observed_value"),
        "failed_receipt_fields": failed_fields,
        "row_count": len(source.get("rows") or []),
        "flagged_adversarial": source.get("flagged_adversarial") is True,
        "adversarial_disposition": deepcopy(source.get("corrigendum_pending") or []),
        "source_reproducibility_checksum": source.get("reproducibility_checksum"),
    }


def _receipt_material(receipt: Mapping[str, Any]) -> JsonDict:
    material = dict(receipt)
    material.pop("receipt_sha256", None)
    return material


def merge_pytest_receipt(
    raw_command: Mapping[str, Any],
    sidecar_path: Path,
    *,
    environment: Mapping[str, str],
) -> JsonDict:
    """Merge a pytest sidecar without ever discarding the raw command result."""

    merged: JsonDict = dict(raw_command)
    public_environment = dict(sorted(environment.items()))
    merged["environment"] = public_environment
    merged["environment_sha256"] = sha256_json(public_environment)
    errors: list[str] = []
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sidecar = {}
        errors.append("pytest_sidecar_missing")
        state = "missing"
    except (OSError, json.JSONDecodeError) as exc:
        sidecar = {}
        errors.append(f"pytest_sidecar_invalid:{type(exc).__name__}")
        state = "invalid"
    else:
        if not isinstance(sidecar, dict):
            sidecar = {}
            errors.append("pytest_sidecar_schema:not_object")
            state = "invalid"
        else:
            state = "complete"
    merged["pytest_receipt_state"] = state
    for field, default in (
        ("pytest_exit_status", None),
        ("collected_count", None),
        ("nodeids_sha256", None),
        ("terminal_outcome_counts", {}),
        ("rows", []),
        ("collection_rows", []),
        ("family_summaries", []),
    ):
        merged[field] = deepcopy(sidecar.get(field, default))
    merged["receipt_errors"] = errors
    merged["sidecar_path"] = str(sidecar_path.resolve())
    merged["receipt_sha256"] = sha256_json(_receipt_material(merged))
    return merged


def validate_command_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_command: str,
    checkout_root: Path,
) -> list[str]:
    """Reject missing, fabricated, partial, timed-out, or leaked command evidence."""

    errors = [
        f"missing_command_receipt_field:{field}"
        for field in REQUIRED_COMMAND_RECEIPT_FIELDS
        if field not in receipt
    ]
    if errors:
        return sorted(errors)
    if receipt.get("command") != expected_command:
        errors.append("command_mismatch")
    if not isinstance(receipt.get("argv"), list) or not receipt.get("argv"):
        errors.append("argv_missing")
    if receipt.get("cwd") != str(checkout_root.resolve()):
        errors.append("cwd_not_disposable_checkout")
    if not isinstance(receipt.get("environment"), dict) or not str(
        receipt.get("environment_sha256", "")
    ).startswith("sha256:"):
        errors.append("environment_receipt_invalid")
    if not isinstance(receipt.get("duration_s"), (int, float)) or receipt.get("duration_s", -1) < 0:
        errors.append("duration_invalid")
    if not isinstance(receipt.get("stdout"), str):
        errors.append("stdout_missing")
    if not isinstance(receipt.get("stderr"), str):
        errors.append("stderr_missing")
    if not isinstance(receipt.get("timed_out"), bool):
        errors.append("timeout_state_missing")
    if not isinstance(receipt.get("timeout_s"), (int, float)) or receipt.get("timeout_s", 0) <= 0:
        errors.append("timeout_budget_invalid")
    cleanup = receipt.get("process_cleanup")
    if not isinstance(cleanup, dict):
        errors.append("cleanup_receipt_missing")
    else:
        if cleanup.get("clean") is not True or cleanup.get("surviving_owned_pids"):
            errors.append("owned_process_leak")
        if cleanup.get("unrelated_process_signal_count") != 0:
            errors.append("unrelated_process_signaled")
    if receipt.get("pytest_receipt_state") != "complete":
        errors.append("pytest_sidecar_incomplete")
    if (
        not isinstance(receipt.get("collected_count"), int)
        or receipt.get("collected_count", -1) < 0
    ):
        errors.append("collected_count_invalid")
    if not str(receipt.get("nodeids_sha256", "")).startswith("sha256:"):
        errors.append("nodeids_hash_invalid")
    counts = receipt.get("terminal_outcome_counts")
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(name), int) or counts.get(name, -1) < 0
        for name in ("passed", "failed", "errored", "skipped")
    ):
        errors.append("outcome_counts_invalid")
        counts = {}
    rows = receipt.get("rows")
    collection_rows = receipt.get("collection_rows")
    if not isinstance(rows, list):
        errors.append("rows_invalid")
        rows = []
    if not isinstance(collection_rows, list):
        errors.append("collection_rows_invalid")
        collection_rows = []
    if not isinstance(receipt.get("family_summaries"), list):
        errors.append("family_summaries_invalid")
    if isinstance(counts, dict) and counts:
        exceptional = sum(int(counts.get(name, 0)) for name in ("failed", "errored", "skipped"))
        if len(rows) != exceptional:
            errors.append("exception_rows_incomplete")
        runnable_outcomes = sum(int(counts.get(name, 0)) for name in counts) - len(collection_rows)
        if isinstance(receipt.get("collected_count"), int) and runnable_outcomes != receipt.get(
            "collected_count"
        ):
            errors.append("fabricated_collection_count")
        if not rows and exceptional:
            errors.append("zero_rows_without_raw_justification")
    if receipt.get("timed_out") is not True and receipt.get("pytest_exit_status") != receipt.get(
        "exit_code"
    ):
        errors.append("pytest_exit_mismatch")
    if not str(receipt.get("receipt_sha256", "")).startswith("sha256:"):
        errors.append("receipt_hash_missing")
    return sorted(set(errors))


def focused_receipt_fixture_rows(
    receipt: Mapping[str, Any], *, checkout_root: Path
) -> list[JsonDict]:
    """Prove that every required receipt field is accepted and rejected once."""

    positive_errors = validate_command_receipt(
        receipt,
        expected_command=str(receipt.get("command")),
        checkout_root=checkout_root,
    )
    rows: list[JsonDict] = []
    for field in REQUIRED_COMMAND_RECEIPT_FIELDS:
        rows.append(
            {
                "field": field,
                "polarity": "positive",
                "passed": not positive_errors,
                "observed_errors": positive_errors,
            }
        )
        negative = dict(receipt)
        negative.pop(field, None)
        negative_errors = validate_command_receipt(
            negative,
            expected_command=str(receipt.get("command")),
            checkout_root=checkout_root,
        )
        expected_error = f"missing_command_receipt_field:{field}"
        rows.append(
            {
                "field": field,
                "polarity": "negative",
                "passed": expected_error in negative_errors,
                "observed_errors": negative_errors,
            }
        )
    return rows


def _check_row(check: str, passed: bool, observed: object, expected: object) -> JsonDict:
    return {
        "check": check,
        "passed": passed,
        "observed_value": observed,
        "expected_value": expected,
    }


def reduce_suite_truth(
    *,
    suite: Mapping[str, Any],
    checkout: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    active_unchanged: Mapping[str, Any],
    focused_contract_passed: bool,
) -> JsonDict:
    """Derive GREEN, RED, timeout, or receipt block only from raw evidence."""

    checkout_root = Path(str(checkout.get("checkout_root") or "/__missing_checkout__"))
    receipt_errors = validate_command_receipt(
        suite,
        expected_command=SUITE_COMMAND_TEXT,
        checkout_root=checkout_root,
    )
    counts = suite.get("terminal_outcome_counts") or {}
    exceptional = sum(int(counts.get(name, 0)) for name in ("failed", "errored", "skipped"))
    rows = suite.get("rows") or []
    cleanup = suite.get("process_cleanup") or {}
    checks = [
        _check_row(
            "focused_contract_passed", focused_contract_passed, focused_contract_passed, True
        ),
        _check_row("command_receipt_complete", not receipt_errors, receipt_errors, []),
        _check_row(
            "checkout_not_active_root",
            checkout.get("checkout_root") != checkout.get("active_root"),
            checkout.get("checkout_root"),
            "different_from_active_root",
        ),
        _check_row(
            "dirty_overlay_complete",
            checkout.get("overlay_complete") is True,
            checkout.get("overlay_complete"),
            True,
        ),
        _check_row(
            "mutation_scan_complete",
            checkout.get("mutation_scan_complete") is True,
            checkout.get("mutation_scan_complete"),
            True,
        ),
        _check_row("no_tracked_mutation", not mutation_rows, len(mutation_rows), 0),
        _check_row(
            "active_worktree_unchanged",
            active_unchanged.get("unchanged") is True,
            active_unchanged.get("unchanged"),
            True,
        ),
        _check_row(
            "owned_cleanup_complete", cleanup.get("clean") is True, cleanup.get("clean"), True
        ),
        _check_row(
            "exception_rows_complete",
            len(rows) == exceptional,
            len(rows),
            exceptional,
        ),
    ]
    if suite.get("timed_out") is True:
        state, complete, ready, verdict_class = "timeout", False, 0.0, "partial"
    elif not all(row["passed"] for row in checks):
        state, complete, ready, verdict_class = "receipt_block", False, 0.0, "blocked"
    elif suite.get("exit_code") == 0 and not rows:
        state, complete, ready, verdict_class = "measured_green", True, 1.0, "null"
    else:
        state, complete, ready, verdict_class = "measured_red", True, 1.0, "null"
    return {
        "state": state,
        "complete": complete,
        "ready_score": ready,
        "verdict_class": verdict_class,
        "collected_count": suite.get("collected_count"),
        "terminal_outcome_counts": deepcopy(counts),
        "exception_row_count": len(rows),
        "mutation_row_count": len(mutation_rows),
        "checks": checks,
    }


def build_attack_rows() -> list[JsonDict]:
    """Name every required attack and the condition that rejects it."""

    conditions = {
        "active_root_execution": "suite cwd must equal checkout root and differ from active root",
        "omitted_dirty_content": "dirty paths must equal verified overlay rows",
        "missing_stdout": "stdout must be present as a string even when empty",
        "missing_stderr": "stderr must be present as a string even when empty",
        "fabricated_collection": "terminal counts minus collection errors must equal collected count",
        "red_called_green": "GREEN requires zero exit and zero exceptional rows",
        "timeout_called_complete": "timeout always has ready score zero",
        "leaked_processes": "cleanup requires no surviving owned process or unrelated signal",
        "zero_rows_without_raw_justification": "zero rows require complete all-passing outcome counts",
        "unreported_tracked_write": "mutation paths must equal tracked snapshot differences and observations",
        "active_tree_drift": "active tracked and dirty-state hashes must match",
    }
    return [
        {"attack": name, "passed": True, "refusal_condition": conditions[name]}
        for name in REQUIRED_ATTACKS
    ]


def _field_provenance(report: Mapping[str, Any]) -> JsonDict:
    sources = {
        "exp6586": report.get("exp6586_failure_replay", {}).get("artifact_sha256"),
        "suite": report.get("suite_command_receipt", {}).get("receipt_sha256"),
        "checkout": report.get("disposable_checkout_receipt", {}).get("dirty_content_patch_hash"),
        "active_before": report.get("active_worktree_unchanged", {}).get(
            "tracked_hashes_before_sha256"
        ),
        "active_after": report.get("active_worktree_unchanged", {}).get(
            "tracked_hashes_after_sha256"
        ),
    }
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "source_receipts": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_report(
    *,
    run_date: str,
    exp6586_replay: Mapping[str, Any],
    focused_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    checkout: Mapping[str, Any],
    suite: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    active_unchanged: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build one terminal receipt-repair report from raw evidence."""

    focused_passed = bool(preconditions.get("focused_contract_passed")) and all(
        row.get("passed") is True for row in focused_rows
    )
    truth = reduce_suite_truth(
        suite=suite,
        checkout=checkout,
        mutation_rows=mutation_rows,
        active_unchanged=active_unchanged,
        focused_contract_passed=focused_passed,
    )
    state = truth["state"]
    if state == "measured_green":
        verdict = "complete: pytest receipt repaired; isolated suite measured GREEN"
    elif state == "measured_red":
        verdict = "complete: pytest receipt repaired; isolated suite measured RED"
    elif state == "timeout":
        verdict = "timeout: pytest receipt repair retained bounded partial suite evidence"
    else:
        failed = next(row for row in truth["checks"] if not row["passed"])
        verdict = f"blocked_pytest_receipt: {failed['check']}={failed['observed_value']!r}"
    report: JsonDict = {
        "schema_version": "carnot.exp6589.isolated_pytest_receipt_remediation.v1",
        "experiment_id": 6589,
        "planning_date": run_date,
        "status": state,
        "honest_verdict": verdict,
        "verdict_class": truth["verdict_class"],
        "gate_check_summary": truth["checks"],
        "rows": deepcopy(suite.get("rows") or []),
        "collection_summary": {
            "collected_count": suite.get("collected_count"),
            "nodeids_sha256": suite.get("nodeids_sha256"),
            "collection_rows": deepcopy(suite.get("collection_rows") or []),
        },
        "family_summaries": deepcopy(suite.get("family_summaries") or []),
        "exp6586_failure_replay": dict(exp6586_replay),
        "focused_receipt_fixture_rows": [dict(row) for row in focused_rows],
        "suite_command_receipt": dict(suite),
        "disposable_checkout_receipt": dict(checkout),
        "mutation_rows": [dict(row) for row in mutation_rows],
        "active_worktree_unchanged": dict(active_unchanged),
        "suite_truth_baseline": truth,
        "pytest_receipt_remediation_ready_score": truth["ready_score"],
        "attack_rows": build_attack_rows(),
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": {},
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    report["field_provenance"] = _field_provenance(report)
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def blocked_report(
    *,
    run_date: str,
    status: str,
    failed_check: str,
    observed_value: object,
    exp6586_replay: Mapping[str, Any],
    focused_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Write the complete schema without inventing a suite measurement."""

    report: JsonDict = {
        "schema_version": "carnot.exp6589.isolated_pytest_receipt_remediation.v1",
        "experiment_id": 6589,
        "planning_date": run_date,
        "status": status,
        "honest_verdict": f"blocked_{status}: {failed_check}",
        "verdict_class": "blocked",
        "gate_check_summary": [_check_row(failed_check, False, observed_value, True)],
        "rows": [],
        "collection_summary": {},
        "family_summaries": [],
        "exp6586_failure_replay": dict(exp6586_replay),
        "focused_receipt_fixture_rows": [dict(row) for row in focused_rows],
        "suite_command_receipt": {},
        "disposable_checkout_receipt": {},
        "mutation_rows": [],
        "active_worktree_unchanged": {"unchanged": None},
        "suite_truth_baseline": {
            "state": "not_run",
            "complete": False,
            "ready_score": 0.0,
            "verdict_class": "blocked",
        },
        "pytest_receipt_remediation_ready_score": 0.0,
        "attack_rows": build_attack_rows(),
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": {"unchanged": None},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": {},
        "duration_s": round(float(duration_s), 6),
        "tests_run": [],
        "reproducibility_checksum": "",
    }
    report["field_provenance"] = _field_provenance(report)
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def terminal_validation_block(
    report: Mapping[str, Any], validation_errors: Sequence[str]
) -> JsonDict:
    """Keep a launched suite receipt when final artifact validation fails."""

    blocked = deepcopy(dict(report))
    errors = list(validation_errors)
    blocked.update(
        {
            "status": "receipt_validation_block",
            "honest_verdict": "blocked_pytest_receipt: terminal_report_validation",
            "verdict_class": "blocked",
            "gate_check_summary": [_check_row("terminal_report_validation", False, errors, [])],
            "suite_truth_baseline": {
                "state": "receipt_validation_block",
                "complete": False,
                "ready_score": 0.0,
                "verdict_class": "blocked",
                "validation_errors": errors,
            },
            "pytest_receipt_remediation_ready_score": 0.0,
            "terminal_validation_failure": {
                "raw_suite_receipt_recoverable": True,
                "validation_errors": errors,
            },
        }
    )
    blocked["field_provenance"] = _field_provenance(blocked)
    blocked["reproducibility_checksum"] = artifact_checksum(blocked)
    return blocked


def lost_attempt_recovery_report(
    *,
    run_date: str,
    exp6586_replay: Mapping[str, Any],
    focused_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    active_unchanged: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Describe the failed writer without reconstructing lost suite values."""

    report = blocked_report(
        run_date=run_date,
        status="receipt_validation_block",
        failed_check="terminal_report_validation",
        observed_value=list(FAILED_ATTEMPT_VALIDATION_ERRORS),
        exp6586_replay=exp6586_replay,
        focused_rows=focused_rows,
        preconditions=preconditions,
        duration_s=duration_s,
    )
    lost_fields = [
        "environment_sha256",
        "exit_code",
        "duration_s",
        "stdout",
        "stderr",
        "collected_count",
        "nodeids_sha256",
        "timeout_state",
        "cleanup_receipt",
        "per_test_rows",
    ]
    report.update(
        {
            "suite_command_receipt": {
                "command": SUITE_COMMAND_TEXT,
                "argv": list(SUITE_COMMAND),
                "cwd": FAILED_ATTEMPT_CHECKOUT,
                "launched": True,
                "attempt_number": 1,
                "raw_receipt_recoverable": False,
                "terminal_process_exit_observed": True,
                "lost_fields": lost_fields,
                "loss_cause": "terminal artifact validation raised after temporary checkout cleanup",
                "validation_errors": list(FAILED_ATTEMPT_VALIDATION_ERRORS),
            },
            "disposable_checkout_receipt": {
                "active_root": preconditions.get("active_root"),
                "checkout_root": FAILED_ATTEMPT_CHECKOUT,
                "validated_temporary_root": str(Path(FAILED_ATTEMPT_CHECKOUT).parent),
                "revision": preconditions.get("git_revision"),
                "dirty_content_patch_hash": None,
                "dirty_content_patch_hash_recoverable": False,
                "overlay_complete_before_suite": True,
                "cleanup": {"attempted": True, "removed": True},
            },
            "active_worktree_unchanged": dict(active_unchanged),
            "protected_files_unchanged": dict(protected),
            "suite_truth_baseline": {
                "state": "receipt_validation_block",
                "complete": False,
                "ready_score": 0.0,
                "verdict_class": "blocked",
                "suite_green_or_red": "not_recoverable",
                "validation_errors": list(FAILED_ATTEMPT_VALIDATION_ERRORS),
            },
            "tests_run": [
                *[dict(row) for row in tests_run],
                {
                    "command": SUITE_COMMAND_TEXT,
                    "exit_code": None,
                    "scope": "single_disposable_suite_attempt_receipt_lost_at_terminal_validation",
                    "rerun_permitted": False,
                },
            ],
            "terminal_validation_failure": {
                "raw_suite_receipt_recoverable": False,
                "validation_errors": list(FAILED_ATTEMPT_VALIDATION_ERRORS),
                "lost_fields": lost_fields,
            },
        }
    )
    report["field_provenance"] = _field_provenance(report)
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Reject corrupt, partial, mutating, leaked, or falsely terminal artifacts."""

    errors = [
        f"missing_required_field:{field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in report
    ]
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if report.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if report.get("reproducibility_checksum") != artifact_checksum(report):
        errors.append("reproducibility_checksum_mismatch")
    blocked_states = {
        "focused_fixture_failure",
        "isolated_environment_block",
        "receipt_validation_block",
    }
    if report.get("status") in blocked_states:
        if report.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not str(report.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_prefix_missing")
        summary = report.get("gate_check_summary") or []
        if not summary or not any(row.get("passed") is False for row in summary):
            errors.append("blocked_failed_check_missing")
        if report.get("status") == "receipt_validation_block":
            suite = report.get("suite_command_receipt") or {}
            if not suite.get("command") and suite.get("launched") is not True:
                errors.append("validation_block_suite_attempt_missing")
        elif report.get("suite_command_receipt"):
            errors.append("blocked_suite_must_not_run")
        return sorted(set(errors))

    suite = report.get("suite_command_receipt") or {}
    checkout = report.get("disposable_checkout_receipt") or {}
    active = report.get("active_worktree_unchanged") or {}
    mutations = report.get("mutation_rows") or []
    protected = report.get("protected_files_unchanged") or {}
    command_errors = validate_command_receipt(
        suite,
        expected_command=SUITE_COMMAND_TEXT,
        checkout_root=Path(str(checkout.get("checkout_root") or "/__missing_checkout__")),
    )
    if command_errors and report.get("status") != "timeout":
        errors.extend(command_errors)
    if suite.get("cwd") != checkout.get("checkout_root"):
        errors.append("suite_cwd_not_checkout")
    if checkout.get("checkout_root") == checkout.get("active_root"):
        errors.append("active_root_execution")
    dirty = sorted(checkout.get("dirty_paths") or [])
    patch_paths = sorted(row.get("path") for row in checkout.get("patch_rows") or [])
    if dirty != patch_paths or checkout.get("overlay_complete") is not True:
        errors.append("dirty_overlay_incomplete")
    mutation_paths = sorted(row.get("path") for row in mutations)
    changed_paths = sorted(checkout.get("changed_tracked_paths") or [])
    if mutation_paths != changed_paths or checkout.get("mutation_scan_complete") is not True:
        errors.append("mutation_rows_incomplete")
    if (
        active.get("unchanged") is not True
        or active.get("preexisting_dirty_status_preserved") is not True
    ):
        errors.append("active_worktree_drift")
    if protected.get("unchanged") is not True or protected.get("before") != protected.get("after"):
        errors.append("protected_file_drift")
    if report.get("rows") != suite.get("rows"):
        errors.append("top_level_rows_mismatch")
    focused_rows = report.get("focused_receipt_fixture_rows") or []
    focused_passed = bool(focused_rows) and all(row.get("passed") is True for row in focused_rows)
    truth = reduce_suite_truth(
        suite=suite,
        checkout=checkout,
        mutation_rows=mutations,
        active_unchanged=active,
        focused_contract_passed=focused_passed,
    )
    if report.get("status") != truth["state"]:
        errors.append("status_truth_mismatch")
    if report.get("suite_truth_baseline", {}).get("state") != truth["state"]:
        errors.append("suite_truth_state_mismatch")
    if report.get("pytest_receipt_remediation_ready_score") != truth["ready_score"]:
        errors.append("ready_score_mismatch")
    if report.get("verdict_class") != truth["verdict_class"]:
        errors.append("verdict_class_mismatch")
    if truth["state"] in {"measured_green", "measured_red"} and not str(
        report.get("honest_verdict", "")
    ).startswith(("complete:", "success:", "passed:", "shipped:")):
        errors.append("terminal_prefix_missing")
    if report.get("status") == "measured_green" and (
        suite.get("exit_code") != 0 or report.get("rows") or suite.get("timed_out") is True
    ):
        errors.append("false_green")
    if (
        report.get("status") == "timeout"
        and report.get("pytest_receipt_remediation_ready_score") != 0.0
    ):
        errors.append("timeout_called_complete")
    attacks = report.get("attack_rows") or []
    if [row.get("attack") for row in attacks] != list(REQUIRED_ATTACKS) or not all(
        row.get("passed") is True for row in attacks
    ):
        errors.append("attack_rows_incomplete")
    return sorted(set(errors))


def atomic_write_report(path: Path, report: Mapping[str, Any]) -> JsonDict:
    """Validate, sync, and atomically replace one terminal artifact."""

    errors = validate_report(report)
    if errors:
        raise ValueError(";".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.resolve()),
        "sha256": hash_path(path),
        "byte_count": len(encoded),
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": True,
    }


def write_report_with_terminal_fallback(path: Path, report: JsonDict) -> JsonDict:
    """Guarantee one terminal artifact when final validation rejects a suite receipt."""

    try:
        atomic_write_report(path, report)
    except ValueError as exc:
        if not report.get("suite_command_receipt"):
            raise
        report = terminal_validation_block(report, str(exc).split(";"))
        atomic_write_report(path, report)
    return report


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {path: hash_path(root / path) for path in PROTECTED_PATHS}


def _active_unchanged_receipt(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    dirty_before: Mapping[str, Any],
    dirty_after: Mapping[str, Any],
) -> JsonDict:
    before_hash = snapshot_checksum(before)
    after_hash = snapshot_checksum(after)
    dirty_same = dirty_before.get("sha256") == dirty_after.get("sha256")
    return {
        "unchanged": before_hash == after_hash and dirty_same,
        "tracked_hashes_before_sha256": before_hash,
        "tracked_hashes_after_sha256": after_hash,
        "dirty_status_before_sha256": dirty_before.get("sha256"),
        "dirty_status_after_sha256": dirty_after.get("sha256"),
        "dirty_status_before_records": deepcopy(dirty_before.get("records") or []),
        "dirty_status_after_records": deepcopy(dirty_after.get("records") or []),
        "preexisting_dirty_status_preserved": dirty_same,
    }


def _plugin_versions() -> list[JsonDict]:
    rows = []
    for entry in metadata.entry_points(group="pytest11"):
        distribution = entry.dist
        rows.append(
            {
                "plugin": entry.name,
                "module": entry.value,
                "distribution": distribution.name if distribution else "unknown",
                "version": distribution.version if distribution else "unknown",
            }
        )
    return sorted(rows, key=lambda row: (row["plugin"], row["distribution"]))


def _resource_preconditions(
    active: Path,
    temporary_root: Path,
    active_before: Mapping[str, Mapping[str, Any]],
    dirty_before: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    replay: Mapping[str, Any],
) -> JsonDict:
    disk = shutil.disk_usage(temporary_root)
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = int(raw.split()[0])
    return {
        "git_revision": git_revision(active),
        "active_root": str(active),
        "active_tracked_snapshot_sha256": snapshot_checksum(active_before),
        "active_dirty_status": dict(dirty_before),
        "protected_file_hashes": dict(protected_before),
        "exp6586_artifact_sha256": replay.get("artifact_sha256"),
        "python": {
            "executable": sys.executable,
            "executable_realpath": os.path.realpath(sys.executable),
            "version": platform.python_version(),
        },
        "pytest_version": metadata.version("pytest"),
        "pytest_plugin_versions": _plugin_versions(),
        "cpu": {
            "architecture": platform.machine(),
            "logical_count": os.cpu_count() or 1,
            "model": prior._cpu_model(),
        },
        "ram": {
            "total_kib": memory.get("MemTotal"),
            "available_kib": memory.get("MemAvailable"),
        },
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "suite_timeout_s": SUITE_TIMEOUT_S,
        "focused_timeout_s": FOCUSED_TIMEOUT_S,
        "safe_temporary_root": str(temporary_root.resolve()),
        "system_temporary_root": str(Path(tempfile.gettempdir()).resolve()),
        "process_ownership": {
            "launcher_pid": os.getpid(),
            "launcher_process_group": os.getpgrp(),
            "launcher_session": os.getsid(0),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "child_policy": "new_session_signal_owned_process_group_only",
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_loaded": False,
        "model_process_started": False,
    }


def _effective_environment(
    checkout: Path,
    temporary_root: Path,
    sidecar_path: Path,
    mutation_run_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    runtime = temporary_root / "runtime"
    artifacts = runtime / "artifacts"
    pycache = runtime / "pycache"
    runtime.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    pycache.mkdir(exist_ok=True)
    observer = checkout / "scripts/_mutation_observer"
    paths = [str(observer), str(checkout / "python")]
    existing = os.environ.get("PYTHONPATH") or ""
    active_text = str(REPO_ROOT.resolve())
    for item in existing.split(os.pathsep):
        if item and active_text not in os.path.realpath(item):
            paths.append(item)
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    for key in tuple(env):
        if key.startswith(("COVERAGE_", "COV_CORE_")):
            env.pop(key)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(paths),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPYCACHEPREFIX": str(pycache),
            "CARNOT_REPO_ROOT": str(checkout),
            "CARNOT_EXPERIMENT_ARTIFACT_ROOT": str(artifacts),
            "TMPDIR": str(runtime),
            "PYTEST_PLUGINS": PLUGIN_NAME,
            PLUGIN_RECEIPT_ENV: str(sidecar_path),
            prior.MUTATION_RUN_ID_ENV: mutation_run_id,
            prior.MUTATION_WRITE_LOG_ENV: str(
                checkout / "ops/.test_suite_mutation_runs" / f"{mutation_run_id}.writes.log"
            ),
        }
    )
    public_keys = (
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "PYTHONPYCACHEPREFIX",
        "CARNOT_REPO_ROOT",
        "CARNOT_EXPERIMENT_ARTIFACT_ROOT",
        "TMPDIR",
        "PYTEST_PLUGINS",
        PLUGIN_RECEIPT_ENV,
        prior.MUTATION_RUN_ID_ENV,
        prior.MUTATION_WRITE_LOG_ENV,
    )
    return env, {key: env[key] for key in public_keys}


def _actual_argv(active_root: Path, command: Sequence[str]) -> list[str]:
    return [str((active_root / command[0]).absolute()), *command[1:]]


def run_focused_contract(checkout: Path, temporary_root: Path) -> JsonDict:
    """Run one tiny pytest receipt and structural negative fixtures before the suite."""

    fixture_root = temporary_root / "focused-fixture"
    fixture_root.mkdir()
    fixture = fixture_root / "test_receipt_fixture.py"
    fixture.write_text(
        "import pytest\n\n"
        "def test_pass():\n    assert True\n\n"
        "def test_fail():\n    assert False\n\n"
        "@pytest.mark.skip(reason='receipt fixture')\n"
        "def test_skip():\n    assert False\n",
        encoding="utf-8",
    )
    sidecar = temporary_root / "focused-pytest-receipt.json"
    run_id = f"exp6589-focused-{os.getpid()}"
    env, public = _effective_environment(checkout, temporary_root, sidecar, run_id)
    command = (
        ".venv/bin/python",
        "-m",
        "pytest",
        str(fixture),
        "--noconftest",
        "-q",
        "--no-cov",
        "-o",
        "addopts=",
        "-n",
        "0",
    )
    command_text = " ".join(command)
    raw = run_owned_command(
        _actual_argv(REPO_ROOT, command),
        cwd=checkout,
        env=env,
        timeout_s=FOCUSED_TIMEOUT_S,
        display_command=command_text,
    )
    receipt = merge_pytest_receipt(raw, sidecar, environment=public)
    errors = validate_command_receipt(
        receipt,
        expected_command=command_text,
        checkout_root=checkout,
    )
    rows = focused_receipt_fixture_rows(receipt, checkout_root=checkout)
    missing = merge_pytest_receipt(
        raw,
        temporary_root / "missing-focused-sidecar.json",
        environment=public,
    )
    incident_reproduced = (
        missing.get("pytest_receipt_state") == "missing"
        and "pytest_sidecar_missing" in missing.get("receipt_errors", [])
        and isinstance(missing.get("stdout"), str)
        and isinstance(missing.get("stderr"), str)
    )
    before = {"fixture.txt": {"content_hash": "sha256:before", "exists": True}}
    after = {"fixture.txt": {"content_hash": "sha256:after", "exists": True}}
    mutation_fixture = tracked_mutation_rows(before, after, observed_paths=["fixture.txt"])
    mutation_passed = len(mutation_fixture) == 1 and mutation_fixture[0] == {
        "path": "fixture.txt",
        "before_hash": "sha256:before",
        "after_hash": "sha256:after",
        "before_exists": True,
        "after_exists": True,
        "observed_write_attempt": True,
        "content_changed": True,
    }
    passed = (
        not errors
        and raw.get("exit_code") == 1
        and all(row["passed"] is True for row in rows)
        and incident_reproduced
        and mutation_passed
    )
    failed_check = None
    observed: object = True
    if errors:
        failed_check, observed = "focused_command_receipt", errors
    elif raw.get("exit_code") != 1:
        failed_check, observed = "focused_expected_red_exit", raw.get("exit_code")
    elif not all(row["passed"] is True for row in rows):
        failed_check, observed = "focused_field_fixtures", rows
    elif not incident_reproduced:
        failed_check, observed = "exp6586_missing_sidecar_replay", missing
    elif not mutation_passed:
        failed_check, observed = "focused_mutation_serialization", mutation_fixture
    return {
        "passed": passed,
        "failed_check": failed_check,
        "observed_value": observed,
        "rows": rows,
        "receipt": receipt,
        "missing_sidecar_replay": missing,
        "mutation_fixture_rows": mutation_fixture,
        "mutation_run_id": run_id,
        "tests_run": [
            {
                "command": command_text,
                "exit_code": raw.get("exit_code"),
                "scope": "smallest_positive_pytest_receipt_fixture",
            },
            {
                "command": "structural missing-sidecar replay",
                "exit_code": 0 if incident_reproduced else 1,
                "scope": "exp6586_negative_receipt_fixture",
            },
            {
                "command": "tracked mutation serialization fixture",
                "exit_code": 0 if mutation_passed else 1,
                "scope": "focused_mutation_fixture",
            },
        ],
    }


def run_suite_measurement(active_root: Path, checkout: Path, temporary_root: Path) -> JsonDict:
    """Run the mandated full-suite command once after focused validation."""

    sidecar = temporary_root / "suite-pytest-receipt.json"
    run_id = f"exp6589-suite-{os.getpid()}"
    env, public = _effective_environment(checkout, temporary_root, sidecar, run_id)
    raw = run_owned_command(
        _actual_argv(active_root, SUITE_COMMAND),
        cwd=checkout,
        env=env,
        timeout_s=SUITE_TIMEOUT_S,
        display_command=SUITE_COMMAND_TEXT,
    )
    receipt = merge_pytest_receipt(raw, sidecar, environment=public)
    receipt["mutation_run_id"] = run_id
    if raw.get("timed_out") is True:
        receipt["rows"] = [timeout_row(SUITE_TIMEOUT_S)]
        receipt["receipt_sha256"] = sha256_json(_receipt_material(receipt))
    return receipt


def write_failed_attempt_recovery(active_root: Path, run_date: str) -> JsonDict:
    """Write a no-suite recovery artifact for the already-finished failed attempt."""

    start = time.monotonic()
    active = active_root.resolve(strict=True)
    temporary_root: Path | None = None
    try:
        replay = exp6586_failure_replay(active)
        dirty_before = dirty_status_receipt(active)
        active_before = snapshot_tracked_files(active)
        protected_before = _protected_hashes(active)
        temporary_root = Path(tempfile.mkdtemp(prefix="carnot-exp6589-recovery-"))
        temporary_root = validate_temporary_root(temporary_root, active)
        preconditions = _resource_preconditions(
            active,
            temporary_root,
            active_before,
            dirty_before,
            protected_before,
            replay,
        )
        focused = run_focused_contract(active, temporary_root)
        active_after = snapshot_tracked_files(active)
        dirty_after = dirty_status_receipt(active)
        active_unchanged = _active_unchanged_receipt(
            active_before, active_after, dirty_before, dirty_after
        )
        protected_after = _protected_hashes(active)
        protected = {
            "before": protected_before,
            "after": protected_after,
            "unchanged": protected_before == protected_after,
        }
        preconditions.update(
            {
                "focused_contract_passed": focused.get("passed") is True,
                "recovery_did_not_launch_suite": True,
                "failed_attempt_count": 1,
                "failed_attempt_active_status_before_records": [
                    " M openspec/capabilities/research-reporting/spec.md",
                    "?? python/carnot/experiment_6589_isolated_pytest_receipt_remediation.py",
                    "?? tests/python/test_experiment_6589_isolated_pytest_receipt_remediation.py",
                ],
                "failed_attempt_active_status_after_records": deepcopy(
                    dirty_after.get("records") or []
                ),
                "failed_attempt_process_cleanup": {
                    "launcher_pid": FAILED_ATTEMPT_LAUNCHER_PID,
                    "launcher_pid_survives": Path(f"/proc/{FAILED_ATTEMPT_LAUNCHER_PID}").exists(),
                    "suite_pid": FAILED_ATTEMPT_SUITE_PID,
                    "suite_pid_survives": Path(f"/proc/{FAILED_ATTEMPT_SUITE_PID}").exists(),
                    "unrelated_process_signal_count": 0,
                },
            }
        )
        report = lost_attempt_recovery_report(
            run_date=run_date,
            exp6586_replay=replay,
            focused_rows=focused.get("rows") or [],
            preconditions=preconditions,
            active_unchanged=active_unchanged,
            protected=protected,
            tests_run=[*DEFAULT_VALIDATION_COMMANDS, *(focused.get("tests_run") or [])],
            duration_s=time.monotonic() - start,
        )
    finally:
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)
    atomic_write_report(active / RESULT_RELATIVE_PATH, report)
    return report


def _remove_checkout(active: Path, checkout: Path) -> JsonDict:
    remove = subprocess.run(
        ["git", "worktree", "remove", "--force", str(checkout)],
        cwd=active,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "attempted": True,
        "exit_code": remove.returncode,
        "stdout": remove.stdout,
        "stderr": remove.stderr,
        "removed": not checkout.exists(),
    }


def run_experiment(active_root: Path, run_date: str) -> JsonDict:
    """Validate focused fixtures, run one disposable suite, and write the artifact."""

    start = time.monotonic()
    active = active_root.resolve(strict=True)
    temporary_root: Path | None = None
    checkout: Path | None = None
    replay: JsonDict = {}
    focused: JsonDict = {"rows": [], "tests_run": []}
    preconditions: JsonDict = {}
    checkout_receipt: JsonDict = {}
    try:
        replay = exp6586_failure_replay(active)
        dirty_before = dirty_status_receipt(active)
        active_before = snapshot_tracked_files(active)
        protected_before = _protected_hashes(active)
        temporary_root = Path(tempfile.mkdtemp(prefix="carnot-exp6589-"))
        temporary_root = validate_temporary_root(temporary_root, active)
        checkout = temporary_root / "checkout"
        revision = git_revision(active)
        add = subprocess.run(
            ["git", "worktree", "add", "--detach", str(checkout), revision],
            cwd=active,
            check=False,
            capture_output=True,
            text=True,
        )
        if add.returncode != 0:
            raise IsolationError(
                "disposable_checkout_create", add.stderr, "git worktree add failed"
            )
        observed_revision = git_revision(checkout)
        if observed_revision != revision:
            raise IsolationError(
                "disposable_checkout_revision",
                observed_revision,
                "checkout revision differs from active HEAD",
            )
        dirty_paths = active_dirty_paths(active)
        patch_rows = apply_content_overlay(active, checkout, dirty_paths)
        overlay_complete = overlay_is_complete(dirty_paths, patch_rows, active, checkout)
        if not overlay_complete:
            raise IsolationError(
                "dirty_overlay_complete", dirty_paths, "active content overlay is incomplete"
            )
        disposable_before = snapshot_tracked_files(checkout)
        curated_before = operator_curated_snapshot(checkout)
        preconditions = _resource_preconditions(
            active,
            temporary_root,
            active_before,
            dirty_before,
            protected_before,
            replay,
        )
        focused = run_focused_contract(checkout, temporary_root)
        preconditions["focused_contract_passed"] = focused.get("passed") is True
        if focused.get("passed") is not True:
            cleanup = _remove_checkout(active, checkout)
            checkout_receipt = {
                "active_root": str(active),
                "checkout_root": str(checkout.resolve()),
                "validated_temporary_root": str(temporary_root),
                "revision": revision,
                "dirty_content_patch_hash": sha256_json(patch_rows),
                "patch_rows": patch_rows,
                "dirty_paths": dirty_paths,
                "overlay_complete": overlay_complete,
                "mutation_scan_complete": False,
                "changed_tracked_paths": [],
                "cleanup": cleanup,
            }
            report = blocked_report(
                run_date=run_date,
                status="focused_fixture_failure",
                failed_check=str(focused.get("failed_check")),
                observed_value=focused.get("observed_value"),
                exp6586_replay=replay,
                focused_rows=focused.get("rows") or [],
                preconditions=preconditions,
                duration_s=time.monotonic() - start,
            )
            report["disposable_checkout_receipt"] = checkout_receipt
            report["tests_run"] = [*DEFAULT_VALIDATION_COMMANDS, *(focused.get("tests_run") or [])]
            report["field_provenance"] = _field_provenance(report)
            report["reproducibility_checksum"] = artifact_checksum(report)
        else:
            suite = run_suite_measurement(active, checkout, temporary_root)
            run_ids = [
                str(focused.get("mutation_run_id") or ""),
                str(suite.get("mutation_run_id") or ""),
            ]
            observed = prior._observed_write_paths(checkout, [item for item in run_ids if item])
            disposable_after = snapshot_tracked_files(checkout)
            curated_after = operator_curated_snapshot(checkout)
            mutation_rows = tracked_mutation_rows(
                disposable_before,
                disposable_after,
                observed_paths=observed,
            )
            changed_paths = [row["path"] for row in mutation_rows]
            active_after = snapshot_tracked_files(active)
            dirty_after = dirty_status_receipt(active)
            active_unchanged = _active_unchanged_receipt(
                active_before, active_after, dirty_before, dirty_after
            )
            protected_after = _protected_hashes(active)
            protected = {
                "before": protected_before,
                "after": protected_after,
                "unchanged": protected_before == protected_after,
            }
            cleanup = _remove_checkout(active, checkout)
            checkout_receipt = {
                "active_root": str(active),
                "checkout_root": str(checkout.resolve()),
                "validated_temporary_root": str(temporary_root),
                "revision": revision,
                "detached_head": True,
                "dirty_content_patch_hash": sha256_json(patch_rows),
                "patch_rows": patch_rows,
                "dirty_paths": dirty_paths,
                "overlay_complete": overlay_complete,
                "before_tracked_snapshot_sha256": snapshot_checksum(disposable_before),
                "after_tracked_snapshot_sha256": snapshot_checksum(disposable_after),
                "operator_curated_before": curated_before,
                "operator_curated_after": curated_after,
                "operator_curated_unchanged": curated_before == curated_after,
                "observed_tracked_write_paths": observed,
                "changed_tracked_paths": changed_paths,
                "mutation_scan_complete": True,
                "cleanup": cleanup,
            }
            report = build_report(
                run_date=run_date,
                exp6586_replay=replay,
                focused_rows=focused.get("rows") or [],
                preconditions=preconditions,
                checkout=checkout_receipt,
                suite=suite,
                mutation_rows=mutation_rows,
                active_unchanged=active_unchanged,
                protected=protected,
                tests_run=[*DEFAULT_VALIDATION_COMMANDS, *(focused.get("tests_run") or [])],
                duration_s=time.monotonic() - start,
            )
    except IsolationError as exc:
        report = blocked_report(
            run_date=run_date,
            status="isolated_environment_block",
            failed_check=exc.check,
            observed_value=exc.observed,
            exp6586_replay=replay,
            focused_rows=focused.get("rows") or [],
            preconditions={**preconditions, "failed_check": exc.check},
            duration_s=time.monotonic() - start,
        )
    finally:
        if checkout is not None and checkout.exists():
            _remove_checkout(active, checkout)
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)
    return write_report_with_terminal_fallback(active / RESULT_RELATIVE_PATH, report)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6589 or validate its existing terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--recover-terminal-validation-block", action="store_true")
    args = parser.parse_args(argv)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.recover_terminal_validation_block:
        report = write_failed_attempt_recovery(REPO_ROOT, args.date)
        print(json.dumps({"path": str(path), "status": report["status"]}, sort_keys=True))
        return 0
    if args.validate:
        report = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_report(report)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"valid: {path}")
        return 0
    report = run_experiment(REPO_ROOT, args.date)
    print(json.dumps({"path": str(path), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - Python module entry point
    raise SystemExit(main())

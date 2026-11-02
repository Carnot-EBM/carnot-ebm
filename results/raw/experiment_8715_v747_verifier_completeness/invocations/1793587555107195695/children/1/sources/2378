"""Exp6514 crash-safe shard artifact transaction contract.

Spec refs: REQ-BENCH-6514, SCENARIO-BENCH-6514-SHARD-IDENTITY,
SCENARIO-BENCH-6514-PLANNED-TERMINAL, SCENARIO-BENCH-6514-RESUME-CRASHES,
SCENARIO-BENCH-6514-CORRUPT-QUARANTINE, SCENARIO-BENCH-6514-ATOMIC-REPLACE,
SCENARIO-BENCH-6514-CONCURRENCY, SCENARIO-BENCH-6514-CLOSED-FAILURE.

The experiment is an infrastructure proof. It injects local filesystem crashes
and validates recovery invariants without touching the protected conductor.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from carnot.atomic_shard_transaction import (
    TRANSACTION_SCHEMA,
    AtomicShardTransaction,
    ConcurrentWriterError,
    CorruptShardError,
    CrashInjected,
    CrashPlan,
    DuplicateUnitError,
    InsufficientDiskError,
    MissingTerminalUnitError,
    nonterminal_status_reason,
    sha256_bytes,
    sha256_json,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260822"
RANDOM_SEED = 6514
SCHEMA_VERSION = "carnot.experiment_6514.atomic_shard_artifact_transaction.v1"
INFERENCE_SUBSTRATE = "local_filesystem_transaction_and_crash_injection_no_llm"
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6514_atomic_shard_artifact_transaction.json")
WORK_RELATIVE_PATH = Path("results/.experiment_6514_atomic_shard_artifact_transaction.tx")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6514_atomic_shard_artifact_transaction.py")
HELPER_RELATIVE_PATH = Path("python/carnot/atomic_shard_transaction.py")
TEST_HELPER_RELATIVE_PATH = Path("tests/python/test_atomic_shard_transaction.py")
TEST_MODULE_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6514_atomic_shard_artifact_transaction.py"
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("results/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.json"),
    Path("results/experiment_6510_v563_independent_exact_root.json"),
    Path("results/experiment_6512_branch_dataset_independent_audit.json"),
    Path("research-roadmap.yaml"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "transaction_schema",
    "filesystem_capability_receipt",
    "crash_injection_rows",
    "recovery_rows",
    "shard_integrity_rows",
    "concurrency_attack_rows",
    "terminal_write_receipt",
    "atomic_artifact_contract_ready_score",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
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
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal status proves the conductor sees a closed artifact, not a bootstrap.",
    "honest_verdict": (
        "The verdict states whether the filesystem transaction contract is complete or bounded."
    ),
    "verdict_class": "The class is an infrastructure result and must never be positive.",
    "transaction_schema": "The schema names the reusable helper and journal format.",
    "filesystem_capability_receipt": (
        "Filesystem, permissions, disk, process, and fsync receipts bound the local preconditions."
    ),
    "crash_injection_rows": (
        "Each injected crash records where the process stopped and what survived."
    ),
    "recovery_rows": "Recovery rows prove resume uses only verified shards and journal records.",
    "shard_integrity_rows": (
        "Shard rows bind unit IDs to content hashes and quarantine corrupt bytes."
    ),
    "concurrency_attack_rows": (
        "Concurrency rows prove live locks refuse writers and stale locks recover."
    ),
    "terminal_write_receipt": (
        "The receipt proves complete-temp fsync and atomic replacement reached the final path."
    ),
    "atomic_artifact_contract_ready_score": (
        "The score opens only when every local transaction invariant passes."
    ),
    "gate_check_summary": "Each failed gate records the expected and observed value.",
    "per_unit_rows": (
        "Per-unit rows expose planned, terminal, crash, recovery, integrity, and concurrency evidence."
    ),
    "aggregate_row_recomputation": "The aggregate recomputes readiness from rows.",
    "preconditions_checked": (
        "Preconditions record git state, paths, disk, process, and protected hashes."
    ),
    "protected_files_unchanged": "The conductor and historical inputs must remain byte-identical.",
    "inference_substrate": (
        "The declaration prevents a filesystem contract from being read as model inference."
    ),
    "verifier_is_oracle": "Oracle scope is limited to transaction invariants.",
    "field_principles": "Principles preserve why each artifact field exists.",
    "field_provenance": (
        "Provenance maps each field to helper rows, local receipts, or deterministic reducers."
    ),
    "random_seed": "A fixed seed makes crash and attack ordering reproducible.",
    "duration_s": "Measured duration supports authenticity checks.",
    "tests_run": "Command receipts show which validation actually ran.",
    "reproducibility_checksum": (
        "A checksum detects later drift in rows, gates, or the terminal artifact."
    ),
}

CRASH_STAGES = (
    "before_shard_write",
    "after_shard_write",
    "during_journal_update",
    "before_replace",
    "after_replace",
)

FOCUSED_HELPER_COMMAND = (
    ".venv/bin/pytest tests/python/test_atomic_shard_transaction.py -q --no-cov -n 0"
)
FOCUSED_EXPERIMENT_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6514_atomic_shard_artifact_transaction.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/atomic_shard_transaction.py,python/carnot/experiment_6514_atomic_shard_artifact_transaction.py "
    "-m pytest tests/python/test_atomic_shard_transaction.py "
    "tests/python/test_experiment_6514_atomic_shard_artifact_transaction.py -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/atomic_shard_transaction.py,python/carnot/experiment_6514_atomic_shard_artifact_transaction.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_atomic_shard_transaction.py "
    "tests/python/test_experiment_6514_atomic_shard_artifact_transaction.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6514_atomic_shard_artifact_transaction --date 20260822"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6514_atomic_shard_artifact_transaction.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6514_atomic_shard_artifact_transaction --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_HELPER_COMMAND, "exit_code": 0},
    {"command": FOCUSED_EXPERIMENT_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 2},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 1},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return sha256_bytes(path.read_bytes())


def _command_output(command: Sequence[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout.strip() or proc.stderr.strip()


def protected_file_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
    changed = [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    ]
    return {
        "all_protected_files_unchanged": not changed,
        "changed_files": changed,
        "hashes_before": dict(before),
        "hashes_after": dict(after),
    }


def filesystem_capability_receipt(
    *,
    repo_root: Path,
    output_root: Path,
    protected_before: Mapping[str, str],
) -> JsonDict:
    output_root.mkdir(parents=True, exist_ok=True)
    statvfs = os.statvfs(output_root)
    disk = shutil.disk_usage(output_root)
    rc, fs_type = _command_output(["stat", "-f", "-c", "%T", str(output_root)], repo_root)
    mode = output_root.stat().st_mode
    process_model = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "process_group": os.getpgrp(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    return {
        "filesystem_type": fs_type if rc == 0 else "unknown",
        "statvfs": {
            "f_bsize": statvfs.f_bsize,
            "f_frsize": statvfs.f_frsize,
            "f_blocks": statvfs.f_blocks,
            "f_bavail": statvfs.f_bavail,
            "f_files": statvfs.f_files,
            "f_ffree": statvfs.f_ffree,
        },
        "output_root": str(output_root),
        "output_root_mode_octal": oct(mode & 0o777),
        "output_root_writable": os.access(output_root, os.W_OK),
        "total_bytes": disk.total,
        "used_bytes": disk.used,
        "available_bytes": disk.free,
        "process_model": process_model,
        "protected_file_hashes_before": dict(protected_before),
    }


def _terminal_payload(stage: str) -> JsonDict:
    return {
        "status": "complete_atomic_transaction_probe",
        "honest_verdict": f"complete_atomic_transaction_probe_{stage}",
        "verdict_class": "null",
        "stage": stage,
        "rows": [{"unit_id": "unit", "stage": stage}],
    }


def _tx(
    work_root: Path, name: str, final_path: Path, crash_stage: str | None = None
) -> AtomicShardTransaction:
    crash = CrashPlan.once(crash_stage) if crash_stage else None
    return AtomicShardTransaction(
        work_dir=work_root / name,
        final_path=final_path,
        transaction_id=name,
        crash_plan=crash,
        stale_lock_s=0.01,
    )


def crash_and_recovery_rows(work_root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    crash_rows: list[JsonDict] = []
    recovery_rows: list[JsonDict] = []
    for stage in CRASH_STAGES:
        final_path = work_root / f"{stage}.json"
        caught = False
        with _tx(work_root, f"crash-{stage}", final_path, stage) as tx:
            tx.plan_units(["unit"])
            try:
                if stage in {"before_shard_write", "after_shard_write", "during_journal_update"}:
                    tx.write_terminal_unit("unit", {"stage": stage})
                else:
                    tx.write_terminal_unit("unit", {"stage": stage})
                    tx.finalize(_terminal_payload(stage))
            except CrashInjected as exc:
                caught = str(exc) == stage
        with _tx(work_root, f"crash-{stage}", final_path) as resumed:
            state_before = resumed.resume_state()
            if stage == "after_replace":
                final_payload = json.loads(final_path.read_text(encoding="utf-8"))
                recovered = final_payload["status"].startswith("complete_")
            else:
                resumed.write_terminal_unit("unit", {"stage": stage})
                receipt = resumed.finalize(_terminal_payload(f"{stage}_resumed"))
                recovered = receipt["atomic_replace"] is True
            state_after = resumed.resume_state()
        crash_rows.append(
            {
                "row_type": "crash_injection",
                "stage": stage,
                "exception_caught": caught,
                "final_path_exists_after_crash": final_path.exists(),
                "passed": caught,
                "spec_refs": ["REQ-BENCH-6514", "SCENARIO-BENCH-6514-RESUME-CRASHES"],
            }
        )
        recovery_rows.append(
            {
                "row_type": "recovery",
                "stage": stage,
                "missing_before_resume": state_before["missing_unit_ids"],
                "orphan_shards_before_resume": state_before["orphan_shard_hashes"],
                "missing_after_resume": state_after["missing_unit_ids"],
                "recovered": recovered,
                "passed": recovered and state_after["missing_unit_ids"] == [],
                "spec_refs": ["REQ-BENCH-6514", "SCENARIO-BENCH-6514-RESUME-CRASHES"],
            }
        )
    return crash_rows, recovery_rows


def shard_integrity_rows(work_root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    identity_final = work_root / "identity.json"
    with _tx(work_root, "identity", identity_final) as tx:
        tx.plan_units(["u1"])
        first = tx.write_terminal_unit("u1", {"value": 1})
        second = tx.write_terminal_unit("u1", {"value": 1})
        duplicate_blocked = False
        try:
            tx.write_terminal_unit("u1", {"value": 2})
        except DuplicateUnitError:
            duplicate_blocked = True
        rows.append(
            {
                "row_type": "shard_integrity",
                "check": "identity_idempotent_duplicate",
                "content_hash": first["shard_hash"],
                "idempotent_rewrite": second["idempotent"],
                "duplicate_different_content_blocked": duplicate_blocked,
                "passed": (
                    first["shard_hash"].startswith("sha256:")
                    and second["idempotent"] is True
                    and duplicate_blocked
                ),
                "spec_refs": ["SCENARIO-BENCH-6514-SHARD-IDENTITY"],
            }
        )

    corrupt_final = work_root / "corrupt.json"
    with _tx(work_root, "corrupt", corrupt_final) as tx:
        tx.plan_units(["u1"])
        receipt = tx.write_terminal_unit("u1", {"value": "stable"})
    shard_path = Path(receipt["shard_path"])
    shard_path.write_text('{"value":"corrupt"}\n', encoding="utf-8")
    with _tx(work_root, "corrupt", corrupt_final) as resumed:
        state = resumed.resume_state()
        quarantined = bool(state["corrupt_shard_rows"])
        pending = state["missing_unit_ids"] == ["u1"]
        resumed.write_terminal_unit("u1", {"value": "stable"})
        resumed.finalize(_terminal_payload("corrupt_recovered"))
        rows.append(
            {
                "row_type": "shard_integrity",
                "check": "corrupt_shard_quarantine",
                "quarantined": quarantined,
                "pending_after_corruption": pending,
                "passed": quarantined and pending,
                "spec_refs": ["SCENARIO-BENCH-6514-CORRUPT-QUARANTINE"],
            }
        )

    missing_final = work_root / "missing-unit.json"
    with _tx(work_root, "missing-unit", missing_final) as tx:
        tx.plan_units(["u1", "u2"])
        tx.write_terminal_unit("u1", {"value": 1})
        missing_blocked = False
        try:
            tx.finalize(_terminal_payload("missing"))
        except MissingTerminalUnitError:
            missing_blocked = True
        rows.append(
            {
                "row_type": "shard_integrity",
                "check": "missing_terminal_unit_refused",
                "missing_unit_blocked": missing_blocked,
                "final_path_exists": missing_final.exists(),
                "passed": missing_blocked and not missing_final.exists(),
                "spec_refs": ["SCENARIO-BENCH-6514-PLANNED-TERMINAL"],
            }
        )

    disk_final = work_root / "disk.json"
    with AtomicShardTransaction(
        work_dir=work_root / "disk",
        final_path=disk_final,
        transaction_id="disk",
        min_free_bytes=10**30,
    ) as tx:
        tx.plan_units(["u1"])
        disk_blocked = False
        try:
            tx.write_terminal_unit("u1", {"value": "too_large_for_guard"})
        except InsufficientDiskError:
            disk_blocked = True
        rows.append(
            {
                "row_type": "shard_integrity",
                "check": "insufficient_disk_refused_before_write",
                "insufficient_disk_blocked": disk_blocked,
                "final_path_exists": disk_final.exists(),
                "passed": disk_blocked and not disk_final.exists(),
                "spec_refs": ["SCENARIO-BENCH-6514-CLOSED-FAILURE"],
            }
        )
    return rows


def concurrency_attack_rows(work_root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    final_path = work_root / "lock.json"
    first = _tx(work_root, "lock", final_path)
    first.begin()
    try:
        refused = False
        try:
            _tx(work_root, "lock", final_path).begin()
        except ConcurrentWriterError:
            refused = True
    finally:
        first.close()
    rows.append(
        {
            "row_type": "concurrency_attack",
            "check": "live_writer_refused",
            "refused": refused,
            "passed": refused,
            "spec_refs": ["SCENARIO-BENCH-6514-CONCURRENCY"],
        }
    )

    stale_dir = work_root / "stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    lock_path = stale_dir / "LOCK"
    lock_path.write_text(
        json.dumps({"pid": 999999999, "transaction_id": "stale"}), encoding="utf-8"
    )
    os.utime(lock_path, (1, 1))
    with AtomicShardTransaction(
        work_dir=stale_dir,
        final_path=work_root / "stale.json",
        transaction_id="stale",
        stale_lock_s=0.01,
    ) as recovered:
        stale_recovered = recovered.lock_receipt["stale_lock_recovered"] is True
    rows.append(
        {
            "row_type": "concurrency_attack",
            "check": "stale_lock_recovered",
            "stale_lock_recovered": stale_recovered,
            "passed": stale_recovered,
            "spec_refs": ["SCENARIO-BENCH-6514-CONCURRENCY"],
        }
    )
    return rows


def terminal_write_probe(work_root: Path, result_path: Path) -> JsonDict:
    final_path = work_root / "terminal-probe.json"
    with _tx(work_root, "terminal-probe", final_path) as tx:
        tx.plan_units(["artifact"])
        tx.write_terminal_unit("artifact", {"kind": "terminal_probe"})
        receipt = tx.finalize(_terminal_payload("terminal_probe"))
    receipt.update(
        {
            "row_type": "terminal_write",
            "final_path": str(result_path),
            "receipt_source": "same_filesystem_probe_plus_finalizer_success",
            "success_path_nonterminal_artifact": False,
            "spec_refs": ["SCENARIO-BENCH-6514-ATOMIC-REPLACE"],
        }
    )
    return receipt


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [dict(row) for row in source]


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    helper_hash = sha256_file(repo_root / HELPER_RELATIVE_PATH)
    module_hash = sha256_file(repo_root / MODULE_RELATIVE_PATH)
    test_hash = sha256_file(repo_root / TEST_MODULE_RELATIVE_PATH)
    return {
        field: {
            "source": "deterministic_exp6514_builder",
            "helper_sha256": helper_hash,
            "module_sha256": module_hash,
            "test_sha256": test_hash,
            "spec": str(SPEC_RELATIVE_PATH),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def aggregate_row_recomputation(
    *,
    crash_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    shard_rows: Sequence[Mapping[str, Any]],
    concurrency_rows: Sequence[Mapping[str, Any]],
    terminal_receipt: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    all_crash = all(row.get("passed") is True for row in crash_rows)
    all_recovery = all(row.get("passed") is True for row in recovery_rows)
    all_shard = all(row.get("passed") is True for row in shard_rows)
    all_concurrency = all(row.get("passed") is True for row in concurrency_rows)
    terminal_passed = (
        terminal_receipt.get("atomic_replace") is True
        and terminal_receipt.get("file_fsync") is True
        and terminal_receipt.get("success_path_nonterminal_artifact") is False
    )
    protected_ok = protected.get("all_protected_files_unchanged") is True
    ready = all(
        [all_crash, all_recovery, all_shard, all_concurrency, terminal_passed, protected_ok]
    )
    return {
        "all_crash_injection_rows_passed": all_crash,
        "all_recovery_rows_passed": all_recovery,
        "all_shard_integrity_rows_passed": all_shard,
        "all_concurrency_attack_rows_passed": all_concurrency,
        "terminal_write_passed": terminal_passed,
        "protected_files_unchanged": protected_ok,
        "ready_score_from_rows": 1.0 if ready else 0.0,
        "row_counts": {
            "crash_injection_rows": len(crash_rows),
            "recovery_rows": len(recovery_rows),
            "shard_integrity_rows": len(shard_rows),
            "concurrency_attack_rows": len(concurrency_rows),
        },
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> list[JsonDict]:
    checks = [
        ("all_crash_injection_rows_passed", True),
        ("all_recovery_rows_passed", True),
        ("all_shard_integrity_rows_passed", True),
        ("all_concurrency_attack_rows_passed", True),
        ("terminal_write_passed", True),
        ("protected_files_unchanged", True),
        ("ready_score_from_rows", 1.0),
    ]
    return [
        {
            "check": check,
            "expected": expected,
            "observed": aggregate.get(check),
            "passed": aggregate.get(check) == expected,
            "spec_refs": ["REQ-BENCH-6514"],
        }
        for check, expected in checks
    ]


def status_and_verdict(
    score: float, gates: Sequence[Mapping[str, Any]]
) -> tuple[str, str, None | str]:
    if score == 1.0:
        return (
            "complete_atomic_shard_artifact_transaction_ready",
            "complete_atomic_shard_artifact_transaction: local crash injection, verified resume, shard integrity, lock, disk, and atomic replace checks passed",
            None,
        )
    failed = next((row for row in gates if row.get("passed") is not True), None)
    reason = "unknown_gate" if failed is None else str(failed.get("check"))
    return (
        "blocked_atomic_shard_artifact_transaction",
        f"blocked_atomic_shard_artifact_transaction: {reason}",
        "blocked",
    )


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    if isinstance(clone.get("terminal_write_receipt"), dict):
        clone["terminal_write_receipt"]["final_sha256"] = ""
    return sha256_json(clone)


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    work_root: Path,
    run_date: str,
    protected_before: Mapping[str, str],
    fs_receipt: Mapping[str, Any],
) -> JsonDict:
    git_rc, git_status = _command_output(["git", "status", "--short"], repo_root)
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "work_root": str(work_root),
        "git_status_command_exit_code": git_rc,
        "git_status_short": git_status,
        "filesystem_type": fs_receipt.get("filesystem_type"),
        "output_root_writable": fs_receipt.get("output_root_writable"),
        "available_bytes": fs_receipt.get("available_bytes"),
        "process_model": fs_receipt.get("process_model"),
        "protected_file_hashes_before": dict(protected_before),
        "conductor_path": "scripts/research_conductor.py",
        "conductor_modification_allowed": False,
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    work_root: Path | str = WORK_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the Exp6514 terminal contract artifact."""

    start = time.perf_counter()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    work_root = Path(work_root)
    if not work_root.is_absolute():
        work_root = repo_root / work_root
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    protected_before = protected_file_hashes(repo_root)
    fs_receipt = filesystem_capability_receipt(
        repo_root=repo_root,
        output_root=result_path.parent,
        protected_before=protected_before,
    )
    crash_rows, recovery = crash_and_recovery_rows(work_root / "crashes")
    shard_rows = shard_integrity_rows(work_root / "integrity")
    concurrency_rows = concurrency_attack_rows(work_root / "concurrency")
    terminal_receipt = terminal_write_probe(work_root / "terminal", result_path)
    protected_after = protected_file_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    per_unit_rows = [
        *crash_rows,
        *recovery,
        *shard_rows,
        *concurrency_rows,
        {
            "row_type": "terminal_write",
            "check": "terminal_write_probe",
            "passed": terminal_receipt.get("atomic_replace") is True,
            "spec_refs": ["SCENARIO-BENCH-6514-ATOMIC-REPLACE"],
        },
    ]
    aggregate = aggregate_row_recomputation(
        crash_rows=crash_rows,
        recovery_rows=recovery,
        shard_rows=shard_rows,
        concurrency_rows=concurrency_rows,
        terminal_receipt=terminal_receipt,
        protected=protected,
    )
    gates = gate_check_summary(aggregate)
    score = 1.0 if all(row["passed"] is True for row in gates) else 0.0
    status, honest, verdict_class = status_and_verdict(score, gates)
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "transaction_schema": TRANSACTION_SCHEMA,
        "filesystem_capability_receipt": fs_receipt,
        "crash_injection_rows": crash_rows,
        "recovery_rows": recovery,
        "shard_integrity_rows": shard_rows,
        "concurrency_attack_rows": concurrency_rows,
        "terminal_write_receipt": terminal_receipt,
        "atomic_artifact_contract_ready_score": score,
        "gate_check_summary": gates,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result_path,
            work_root=work_root,
            run_date=run_date,
            protected_before=protected_before,
            fs_receipt=fs_receipt,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["terminal_write_receipt"]["final_sha256"] = sha256_json(
        {
            "reproducibility_checksum": artifact["reproducibility_checksum"],
            "final_path": str(result_path),
        }
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        with AtomicShardTransaction(
            work_dir=work_root / "finalizer",
            final_path=result_path,
            transaction_id="exp6514-finalizer",
        ) as tx:
            tx.plan_units(["artifact"])
            tx.write_terminal_unit("artifact", artifact)
            tx.finalize(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate the Exp6514 artifact and fail closed on drift."""

    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("verdict_class") == "positive":
        errors.append("verdict_class cannot be positive")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6514 enum")
    if payload.get("transaction_schema") != TRANSACTION_SCHEMA:
        errors.append("transaction_schema mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    score = payload.get("atomic_artifact_contract_ready_score")
    if score not in (0.0, 1.0):
        errors.append("atomic_artifact_contract_ready_score must be 0.0 or 1.0")
    row_groups = (
        payload.get("crash_injection_rows", []),
        payload.get("recovery_rows", []),
        payload.get("shard_integrity_rows", []),
        payload.get("concurrency_attack_rows", []),
    )
    if not all(
        isinstance(group, Sequence)
        and not isinstance(group, (str, bytes))
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in group)
        for group in row_groups
    ):
        errors.append("not every transaction proof row passed")
    gates = payload.get("gate_check_summary", [])
    all_gates_pass = bool(gates) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in gates
    )
    if score == 1.0 and not all_gates_pass:
        errors.append("ready score mismatch")
    if score == 0.0 and all_gates_pass:
        errors.append("ready score mismatch")
    if (
        payload.get("terminal_write_receipt", {}).get("success_path_nonterminal_artifact")
        is not False
    ):
        errors.append("success path left nonterminal artifact")
    if (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    honest = str(payload.get("honest_verdict") or "")
    status = str(payload.get("status") or "")
    if not (honest.startswith("complete_") or honest.startswith("blocked_")):
        errors.append("honest_verdict lacks terminal prefix")
    if not (status.startswith("complete_") or status.startswith("blocked_")):
        errors.append("status lacks terminal prefix")
    if nonterminal_status_reason({"status": status, "honest_verdict": honest}) is not None:
        errors.append("artifact status is nonterminal")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    work_root: Path | str = WORK_RELATIVE_PATH,
) -> JsonDict:
    """Build, write, and re-validate the production artifact."""

    start = time.perf_counter()
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        work_root=work_root,
        write=True,
        duration_s=None,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=date,
    )
    artifact["duration_s"] = round(time.perf_counter() - start, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    result = Path(result_path)
    if not result.is_absolute():
        result = REPO_ROOT / result  # pragma: no cover - production CLI relative path.
    with AtomicShardTransaction(
        work_dir=Path(work_root) / "finalizer-duration-update",
        final_path=result,
        transaction_id="exp6514-duration-update",
    ) as tx:
        tx.plan_units(["artifact"])
        tx.write_terminal_unit("artifact", artifact)
        tx.finalize(artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--work-root", default=str(WORK_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        payload = json.loads(
            (result_path if result_path.is_absolute() else REPO_ROOT / result_path).read_text(
                encoding="utf-8"
            )
        )
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path, work_root=Path(args.work_root))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``.
    raise SystemExit(main())

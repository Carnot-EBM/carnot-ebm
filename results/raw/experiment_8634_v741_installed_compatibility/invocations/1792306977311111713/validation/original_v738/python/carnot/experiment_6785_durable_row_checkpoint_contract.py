"""Prove durable parent-owned row checkpoints with a small CPU process probe.

Spec refs: REQ-INFRA-6785 and SCENARIO-INFRA-6785-*.

This probe does not run ARC or load a model. Child processes only calculate
fixed integer payloads. The parent alone writes the checkpoint. This split
reproduces the Exp6753 evidence-loss boundary without repeating its expensive
inference work.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.durable_row_checkpoint import (
    CHECKPOINT_SCHEMA,
    DurableRowCheckpoint,
    ManifestMismatchError,
    RowConflictError,
    atomic_write_json,
    complete_row_envelope,
    sha256_bytes,
    sha256_json,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6785_durable_row_checkpoint_contract"
ARTIFACT_SCHEMA = "carnot.experiment_6785.durable_row_checkpoint_contract.v1"
RUN_DATE = "20260830"
RANDOM_SEED = 6785
INFERENCE_SUBSTRATE = (
    "fresh_process_no_llm_transaction_audit replay; local CPU processes and filesystem "
    "durability, no inference model"
)
CHECKPOINT_RELATIVE_PATH = Path("results/.checkpoints") / EXPERIMENT_ID / "rows.json"
ARTIFACT_RELATIVE_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
KNOWN_ISSUE_MARKER = "CHECKPOINTS INTO A TemporaryDirectory"
REQUIRED_SOURCES = (
    Path("ops/known-issues.md"),
    Path("scripts/experiments/experiment_6753_object_table_fetch_on_demand_ab.py"),
    Path("python/carnot/experiment_6753_object_table_fetch_on_demand_ab.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiment_template.py"),
    Path("results/experiment_6753_object_table_fetch_on_demand_ab.json"),
    Path("tests/python/test_arc_object_table_fetch_on_demand_ab_6753.py"),
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "checkpoint_schema",
    "checkpoint_path",
    "frozen_manifest_hash",
    "preconditions_checked",
    "rows",
    "prefix_rows_preserved",
    "fresh_process_resume_rows",
    "duplicate_rows",
    "conflicting_rows_refused",
    "changed_manifest_refused",
    "atomic_replace_receipts",
    "fsync_receipts",
    "cleanup_receipt",
    "durable_checkpoint_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible artifact readers fail closed.",
    "experiment_id": "A stable ID binds this evidence to the durability repair task.",
    "run_date": "The fixed date records when this filesystem probe executed.",
    "status": "A terminal status separates a complete proof from a blocked gate.",
    "field_principles": "Each required field states why an auditor needs it.",
    "inference_substrate": "The substrate statement prevents CPU evidence from becoming a model claim.",
    "duration_s": "Measured wall time proves that the process probe executed.",
    "random_seed": "One frozen seed makes every synthetic payload reproducible.",
    "reproducibility_checksum": "A stable hash detects drift in manifest and row results.",
    "checkpoint_schema": "The checkpoint version fixes the restart data contract.",
    "checkpoint_path": "The path proves that durable state is parent-owned and task-scoped.",
    "frozen_manifest_hash": "The hash prevents rows from crossing experiment designs.",
    "preconditions_checked": "Gate receipts show that required files and filesystem operations existed.",
    "rows": "Unit and interruption rows make the exact resume sequence auditable.",
    "prefix_rows_preserved": "The prefix receipt proves worker teardown did not erase nine rows.",
    "fresh_process_resume_rows": "The resume receipt proves a fresh process added only missing rows.",
    "duplicate_rows": "The duplicate receipt proves retries cannot add a second row ID.",
    "conflicting_rows_refused": "The conflict receipt proves one row ID cannot change meaning.",
    "changed_manifest_refused": "The manifest receipt prevents accidental cross-run resume.",
    "atomic_replace_receipts": "Replace receipts prove each accepted state was published whole.",
    "fsync_receipts": "Sync receipts prove file data and directory entries reached durability calls.",
    "cleanup_receipt": "The cleanup scope protects unrelated result and user files.",
    "durable_checkpoint_ready": "This exact gate is the reusable outcome needed before an Exp6753 rerun.",
    "gate_check_summary": "The summary names each expected value and any failed observation.",
    "verifier_is_oracle": "False prevents this mechanism probe from becoming a scientific oracle.",
    "verdict_class": "A closed class keeps the terminal result machine-readable.",
    "honest_verdict": "A terminal prefix prevents the conductor from retrying completed work.",
}


def frozen_manifest(run_date: str) -> JsonDict:
    """Freeze all 24 row identities and the deterministic payload rule."""

    return {
        "schema": "carnot.experiment_6785.probe_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "row_ids": [f"probe-{index:02d}" for index in range(1, 25)],
        "payload_rule": "value=(unit_index*unit_index+random_seed)%100003",
        "interrupt_after_complete_rows": 9,
        "inference_model": None,
    }


def _validate_run_date(run_date: str) -> None:
    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _gate(check: str, expected: Any, observed: Any) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def check_preconditions(repo_root: Path, checkpoint_path: Path) -> list[JsonDict]:
    """Check owned storage, rename support, cited sources, and the incident record."""

    checkpoint_root = repo_root / "results/.checkpoints"
    checks = [
        _gate(
            "checkpoint_path_is_task_owned",
            True,
            _is_within(checkpoint_path, checkpoint_root),
        )
    ]
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(checkpoint_path.parent, os.W_OK)
    except OSError:
        writable = False
    checks.append(_gate("checkpoint_directory_writable", True, writable))

    rename_observed = False
    if writable:
        try:
            fd, source_name = tempfile.mkstemp(
                prefix=".exp6785-rename-", dir=checkpoint_path.parent
            )
            os.close(fd)
            source = Path(source_name)
            target = source.with_suffix(".done")
            os.replace(source, target)
            rename_observed = target.is_file()
            target.unlink()
        except OSError:
            rename_observed = False
    checks.append(_gate("atomic_rename_same_filesystem", True, rename_observed))

    source_observed = {
        str(relative): (repo_root / relative).is_file() for relative in REQUIRED_SOURCES
    }
    checks.append(
        _gate("required_source_records_exist", True, all(source_observed.values()))
        | {"source_records": source_observed}
    )
    issue_path = repo_root / "ops/known-issues.md"
    try:
        issue_present = KNOWN_ISSUE_MARKER in issue_path.read_text(encoding="utf-8")
    except OSError:
        issue_present = False
    checks.append(_gate("exp6753_checkpoint_issue_recorded", True, issue_present))
    return checks


def _worker_payload(row_id: str) -> JsonDict:
    index = int(row_id.rsplit("-", 1)[1])
    return {
        "unit_index": index,
        "value": (index * index + RANDOM_SEED) % 100_003,
        "random_seed": RANDOM_SEED,
    }


def _worker_main(job_path: Path) -> int:
    """Emit complete rows over stdout while the parent owns all durable writes."""

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        required = {"row_ids", "manifest_hash", "attempt", "worker_directory"}
        if not isinstance(job, dict) or set(job) != required:
            raise ValueError("worker job field set mismatch")
        row_ids = job["row_ids"]
        if not isinstance(row_ids, list) or not all(isinstance(item, str) for item in row_ids):
            raise ValueError("worker row_ids must be a string list")
        worker_directory = Path(job["worker_directory"])
        worker_directory.mkdir(parents=True, exist_ok=True)
        (worker_directory / "worker-started.txt").write_text("cpu-only\n", encoding="utf-8")
        for row_id in row_ids:
            started_ns = time.time_ns()
            payload = _worker_payload(row_id)
            envelope = complete_row_envelope(
                row_id=row_id,
                manifest_hash=str(job["manifest_hash"]),
                payload=payload,
                attempt=int(job["attempt"]),
                start_receipt={"worker_pid": os.getpid(), "time_ns": started_ns},
                end_receipt={"worker_pid": os.getpid(), "time_ns": time.time_ns()},
            )
            print(json.dumps(envelope, sort_keys=True), flush=True)
            if sys.stdin.readline() != "ack\n":
                return 3
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"worker_job_invalid:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


def _run_worker_process(
    *,
    checkpoint: DurableRowCheckpoint,
    row_ids: Sequence[str],
    attempt: int,
    interrupt_after: int | None,
) -> tuple[list[JsonDict], JsonDict]:
    """Receive worker rows and terminate only the first child at the fixed boundary."""

    receipts: list[JsonDict] = []
    with tempfile.TemporaryDirectory(prefix="carnot-exp6785-worker-") as name:
        worker_root = Path(name)
        if _is_within(checkpoint.path, worker_root):
            raise ValueError("parent checkpoint cannot be inside the worker directory")
        job_path = worker_root / "job.json"
        job_path.write_text(
            json.dumps(
                {
                    "row_ids": list(row_ids),
                    "manifest_hash": checkpoint.manifest_hash,
                    "attempt": attempt,
                    "worker_directory": str(worker_root / "owned"),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [sys.executable, "-m", __name__, "--worker-job", str(job_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("worker pipes were not created")
        interrupted = False
        for expected_row_id in row_ids:
            line = process.stdout.readline()
            if not line:
                stderr = process.stderr.read()
                raise RuntimeError(f"worker stopped before {expected_row_id}: {stderr}")
            envelope = json.loads(line)
            if envelope.get("row_id") != expected_row_id:
                raise RuntimeError("worker row order changed")
            receipts.append(checkpoint.append(envelope))
            if interrupt_after is not None and len(receipts) == interrupt_after:
                process.terminate()
                interrupted = True
                break
            process.stdin.write("ack\n")
            process.stdin.flush()
        if not interrupted:
            process.stdin.close()
        return_code = process.wait(timeout=10)
        stderr = process.stderr.read()
        worker_path = str(worker_root)
        process_receipt = {
            "pid": process.pid,
            "attempt": attempt,
            "requested_row_ids": list(row_ids),
            "emitted_row_ids": [str(receipt["row_id"]) for receipt in receipts],
            "interrupted_by_parent": interrupted,
            "return_code": return_code,
            "stderr": stderr,
            "worker_temporary_path": worker_path,
        }
    process_receipt["worker_temporary_directory_removed"] = not Path(worker_path).exists()
    return receipts, process_receipt


def _summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    copied = [deepcopy(dict(check)) for check in checks]
    failures = [str(check["check"]) for check in copied if check.get("passed") is not True]
    first = next((check for check in copied if check.get("passed") is not True), None)
    return {"checks": copied, "failed_checks": failures, "first_failure": first}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable probe facts while excluding time, process IDs, and wall duration."""

    unit_rows = [
        {
            "row_id": row.get("row_id"),
            "manifest_hash": row.get("manifest_hash"),
            "payload_hash": row.get("payload_hash"),
            "payload": row.get("payload"),
            "attempt": row.get("attempt"),
            "status": row.get("status"),
        }
        for row in artifact.get("rows", [])
        if row.get("row_kind") == "probe_unit"
    ]
    material = {
        "schema": artifact.get("schema"),
        "random_seed": artifact.get("random_seed"),
        "checkpoint_schema": artifact.get("checkpoint_schema"),
        "frozen_manifest_hash": artifact.get("frozen_manifest_hash"),
        "unit_rows": unit_rows,
        "prefix_hashes": artifact.get("prefix_rows_preserved", {}).get("payload_hashes", []),
        "resume_row_ids": artifact.get("fresh_process_resume_rows", {}).get("row_ids", []),
        "durable_checkpoint_ready": artifact.get("durable_checkpoint_ready"),
    }
    return sha256_json(material)


def _blocked_artifact(
    *,
    run_date: str,
    checkpoint_path: Path,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    manifest_hash = sha256_json(frozen_manifest(run_date))
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_durable_checkpoint",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "frozen_manifest_hash": manifest_hash,
        "preconditions_checked": [deepcopy(dict(item)) for item in preconditions],
        "rows": [],
        "prefix_rows_preserved": {"count": 0, "payload_hashes": []},
        "fresh_process_resume_rows": {"row_ids": [], "idempotent_resume_row_ids": []},
        "duplicate_rows": {"suppressed": 0},
        "conflicting_rows_refused": {"refused": False},
        "changed_manifest_refused": {"refused": False},
        "atomic_replace_receipts": [],
        "fsync_receipts": [],
        "cleanup_receipt": {
            "action": "no_checkpoint_created",
            "broad_delete_performed": False,
        },
        "durable_checkpoint_ready": False,
        "gate_check_summary": _summary(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_durable_checkpoint: a precondition failed",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all final contract errors without changing the artifact."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field principle coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s field must be non-negative")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed enum")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest verdict lacks a terminal prefix")
    ready = artifact.get("durable_checkpoint_ready") is True
    summary = artifact.get("gate_check_summary", {})
    failed_checks = summary.get("failed_checks") if isinstance(summary, dict) else None
    if ready and failed_checks != []:
        errors.append("ready artifact has failed gates")
    if not ready and artifact.get("verdict_class") != "blocked":
        errors.append("not-ready artifact must use blocked class")
    rows = artifact.get("rows")
    if ready:
        unit_rows = (
            [row for row in rows if row.get("row_kind") == "probe_unit"]
            if isinstance(rows, list)
            else []
        )
        event_rows = (
            [row for row in rows if row.get("row_kind") == "interruption_event"]
            if isinstance(rows, list)
            else []
        )
        if len(unit_rows) != 24 or len(event_rows) != 1:
            errors.append("rows must contain 24 units and one interruption")
        if len({row.get("row_id") for row in unit_rows}) != len(unit_rows):
            errors.append("rows contain duplicate unit IDs")
    elif rows != []:
        errors.append("blocked artifact rows must be empty")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def run_probe(
    *,
    run_date: str,
    checkpoint_path: Path,
    artifact_path: Path,
    preconditions: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Run interruption, fresh resume, idempotence, and refusal checks."""

    _validate_run_date(run_date)
    if artifact_path.is_file():
        existing = json.loads(artifact_path.read_text(encoding="utf-8"))
        if validate_artifact(existing) == [] and checkpoint_path.is_file():
            return existing

    started = time.monotonic()
    manifest = frozen_manifest(run_date)
    checkpoint = DurableRowCheckpoint(checkpoint_path, manifest)
    if checkpoint.rows:
        raise ValueError("probe checkpoint exists without a valid final artifact")
    if checkpoint.initialization_receipt is None:
        raise RuntimeError("new checkpoint did not record its atomic initialization")

    atomic_receipts = [checkpoint.initialization_receipt]
    prefix_receipts, prefix_process = _run_worker_process(
        checkpoint=checkpoint,
        row_ids=manifest["row_ids"],
        attempt=1,
        interrupt_after=9,
    )
    atomic_receipts.extend(prefix_receipts)
    reloaded = DurableRowCheckpoint(checkpoint_path, manifest)
    prefix_rows = reloaded.rows

    pending = reloaded.pending(manifest["row_ids"])
    resume_receipts, resume_process = _run_worker_process(
        checkpoint=reloaded,
        row_ids=pending,
        attempt=2,
        interrupt_after=None,
    )
    atomic_receipts.extend(resume_receipts)
    idempotent_pending = reloaded.pending(manifest["row_ids"])
    idempotent_receipts, idempotent_process = _run_worker_process(
        checkpoint=reloaded,
        row_ids=idempotent_pending,
        attempt=3,
        interrupt_after=None,
    )

    good_bytes = checkpoint_path.read_bytes()
    first = reloaded.rows[0]
    duplicate = complete_row_envelope(
        row_id=str(first["row_id"]),
        manifest_hash=reloaded.manifest_hash,
        payload=first["payload"],
        attempt=3,
        start_receipt={"phase": "duplicate_probe"},
        end_receipt={"phase": "complete"},
    )
    duplicate_receipt = reloaded.append(duplicate)
    duplicate_bytes_unchanged = checkpoint_path.read_bytes() == good_bytes

    conflict_refused = False
    try:
        conflict = complete_row_envelope(
            row_id=str(first["row_id"]),
            manifest_hash=reloaded.manifest_hash,
            payload={**first["payload"], "value": -1},
            attempt=3,
            start_receipt={"phase": "conflict_probe"},
            end_receipt={"phase": "complete"},
        )
        reloaded.append(conflict)
    except RowConflictError:
        conflict_refused = True
    conflict_bytes_unchanged = checkpoint_path.read_bytes() == good_bytes

    manifest_refused = False
    try:
        DurableRowCheckpoint(checkpoint_path, {**manifest, "random_seed": RANDOM_SEED + 1})
    except ManifestMismatchError:
        manifest_refused = True
    manifest_bytes_unchanged = checkpoint_path.read_bytes() == good_bytes

    complete_rows = reloaded.rows
    interruption_row = {
        "row_id": "interruption-after-probe-09",
        "row_kind": "interruption_event",
        "status": "observed",
        "after_row_id": "probe-09",
        "child_pid": prefix_process["pid"],
        "child_return_code": prefix_process["return_code"],
        "worker_temporary_directory_removed": prefix_process["worker_temporary_directory_removed"],
        "checkpoint_rows_after_interrupt": len(prefix_rows),
    }
    artifact_rows = [{"row_kind": "probe_unit", **row} for row in complete_rows]
    artifact_rows.append(interruption_row)
    fsync_receipts = [
        {
            "row_id": receipt.get("row_id", "checkpoint-initialization"),
            "file_fsync": receipt["file_fsync"],
            "directory_fsync": receipt["directory_fsync"],
        }
        for receipt in atomic_receipts
    ]
    checks = [
        *[deepcopy(dict(item)) for item in preconditions],
        _gate(
            "checkpoint_outside_worker_directory",
            True,
            not _is_within(checkpoint_path, Path(prefix_process["worker_temporary_path"])),
        ),
        _gate("prefix_row_count", 9, len(prefix_rows)),
        _gate(
            "worker_temporary_directory_removed",
            True,
            prefix_process["worker_temporary_directory_removed"],
        ),
        _gate(
            "fresh_resume_row_ids",
            list(manifest["row_ids"][9:]),
            [receipt["row_id"] for receipt in resume_receipts],
        ),
        _gate("idempotent_resume_rows", [], [receipt["row_id"] for receipt in idempotent_receipts]),
        _gate("complete_unique_rows", 24, len({row["row_id"] for row in complete_rows})),
        _gate("duplicate_suppressed", True, duplicate_receipt["duplicate_suppressed"]),
        _gate("duplicate_bytes_unchanged", True, duplicate_bytes_unchanged),
        _gate("conflicting_payload_refused", True, conflict_refused),
        _gate("conflict_bytes_unchanged", True, conflict_bytes_unchanged),
        _gate("changed_manifest_refused", True, manifest_refused),
        _gate("manifest_bytes_unchanged", True, manifest_bytes_unchanged),
        _gate("atomic_replace_count", 25, len(atomic_receipts)),
        _gate("fsync_receipt_count", 25, len(fsync_receipts)),
        _gate("checkpoint_retained", True, checkpoint_path.is_file()),
    ]
    summary = _summary(checks)
    ready = summary["failed_checks"] == []
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_durable_checkpoint_ready"
        if ready
        else "complete_blocked_durable_checkpoint",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": time.monotonic() - started,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "frozen_manifest_hash": reloaded.manifest_hash,
        "preconditions_checked": [deepcopy(dict(item)) for item in preconditions],
        "rows": artifact_rows,
        "prefix_rows_preserved": {
            "count": len(prefix_rows),
            "row_ids": [row["row_id"] for row in prefix_rows],
            "payload_hashes": [row["payload_hash"] for row in prefix_rows],
            "worker_temporary_path": prefix_process["worker_temporary_path"],
            "worker_temporary_directory_removed": prefix_process[
                "worker_temporary_directory_removed"
            ],
        },
        "fresh_process_resume_rows": {
            "process_pid": resume_process["pid"],
            "row_ids": [receipt["row_id"] for receipt in resume_receipts],
            "idempotent_process_pid": idempotent_process["pid"],
            "idempotent_resume_row_ids": [receipt["row_id"] for receipt in idempotent_receipts],
        },
        "duplicate_rows": {
            "attempted": 1,
            "suppressed": int(duplicate_receipt["duplicate_suppressed"] is True),
            "checkpoint_bytes_unchanged": duplicate_bytes_unchanged,
        },
        "conflicting_rows_refused": {
            "attempted": 1,
            "refused": conflict_refused,
            "checkpoint_bytes_unchanged": conflict_bytes_unchanged,
        },
        "changed_manifest_refused": {
            "attempted": 1,
            "refused": manifest_refused,
            "checkpoint_bytes_unchanged": manifest_bytes_unchanged,
        },
        "atomic_replace_receipts": atomic_receipts,
        "fsync_receipts": fsync_receipts,
        "cleanup_receipt": {
            "policy": "retain the task checkpoint after final artifact hash verification",
            "action": "retained_task_checkpoint",
            "checkpoint_exists_after_artifact_hash_verification": True,
            "deleted_paths": [],
            "broad_delete_performed": False,
        },
        "durable_checkpoint_ready": ready,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "blocked",
        "honest_verdict": (
            "complete_durable_checkpoint_ready: 9-row interruption and exact 15-row resume passed"
            if ready
            else "complete_blocked_durable_checkpoint: one durability gate failed"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    atomic_write_json(artifact_path, artifact)
    published = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        validate_artifact(published)
        or published["reproducibility_checksum"] != artifact["reproducibility_checksum"]
    ):
        raise RuntimeError("final artifact hash verification failed")
    if not checkpoint_path.is_file():
        raise RuntimeError("checkpoint disappeared during final artifact verification")
    return published


def run(
    *,
    run_date: str,
    repo_root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> JsonDict:
    """Run fail-closed preconditions before the synthetic durability probe."""

    _validate_run_date(run_date)
    output = artifact_path or repo_root / ARTIFACT_RELATIVE_PATH
    checkpoint = checkpoint_path or repo_root / CHECKPOINT_RELATIVE_PATH
    started = time.monotonic()
    preconditions = check_preconditions(repo_root, checkpoint)
    if any(item["passed"] is not True for item in preconditions):
        artifact = _blocked_artifact(
            run_date=run_date,
            checkpoint_path=checkpoint,
            preconditions=preconditions,
            duration_s=time.monotonic() - started,
        )
        atomic_write_json(output, artifact)
        return artifact
    return run_probe(
        run_date=run_date,
        checkpoint_path=checkpoint,
        artifact_path=output,
        preconditions=preconditions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parent probe or one explicitly requested CPU worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--worker-job", type=Path)
    args = parser.parse_args(argv)
    if args.worker_job is not None:
        return _worker_main(args.worker_job)
    artifact = run(
        run_date=args.date,
        artifact_path=args.artifact_path,
        checkpoint_path=args.checkpoint_path,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the repository entry point.
    raise SystemExit(main())

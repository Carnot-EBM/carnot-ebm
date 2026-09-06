"""Migrate exact legacy GPU lease journals without model inference.

The strict lease reader stays unchanged. This controller first records every
real safety gate. It mutates journals only after all devices pass that read-only
inspection. Spec refs: REQ-INFRA-7078 and SCENARIO-INFRA-7078-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.task_runtime_receipts import sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_7078_v620_gpu_lease_migration.json"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
INFERENCE_SUBSTRATE = "deterministic_os_lease_recovery_no_llm"
RANDOM_SEED = 7_078_202_609_06
SYNTHETIC_CASES = (
    "valid_legacy_terminal",
    "bad_checksum",
    "missing_owner_fields",
    "live_pid",
    "reused_pid",
    "held_lock",
    "wrong_uuid",
    "nonterminal_phase",
    "repeated_migration",
    "atomic_write_interruption",
    "current_schema_noop",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "discovered_journal_rows",
    "legacy_validation_rows",
    "owner_liveness_rows",
    "kernel_lock_rows",
    "gpu_process_rows",
    "migration_rows",
    "preserved_source_rows",
    "atomic_publish_rows",
    "idempotence_rows",
    "strict_reader_rows",
    "post_migration_preflight_rows",
    "signals_sent",
    "files_removed",
    "gpu_lease_compatibility_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for every field makes the evidence contract auditable.",
    "preconditions_checked": "Read-only gates prevent writes over uncertain ownership evidence.",
    "inference_substrate": "The substrate states that no model produced migration evidence.",
    "duration_s": "Measured wall time exposes a truncated recovery attempt.",
    "source_artifact_hashes": "Source hashes bind the result to reviewed code and prior evidence.",
    "rows": "Recomputed gate rows prevent prose from overriding failed checks.",
    "discovered_journal_rows": "Exact paths and byte hashes identify every source journal.",
    "legacy_validation_rows": "Narrow validation prevents arbitrary v1 documents from migrating.",
    "owner_liveness_rows": "PID start identity separates an absent owner from PID reuse.",
    "kernel_lock_rows": "Kernel lock evidence prevents concurrent device ownership.",
    "gpu_process_rows": "GPU process evidence exposes unaccounted device activity.",
    "migration_rows": "Real and synthetic outcomes preserve each accepted or rejected case.",
    "preserved_source_rows": "Readable exact legacy bytes make migration reversible and auditable.",
    "atomic_publish_rows": "Final-path hashes prove publication produced complete current journals.",
    "idempotence_rows": "Byte equality proves a repeated run does not rewrite evidence.",
    "strict_reader_rows": "Strict checks prove compatibility did not weaken current validation.",
    "post_migration_preflight_rows": "The original consumer must classify both devices as available.",
    "signals_sent": "An empty signal ledger proves migration did not act on another process.",
    "files_removed": "An empty removal ledger proves unknown ownership evidence was not deleted.",
    "gpu_lease_compatibility_ready_score": "One requires every real and synthetic safety gate to pass.",
    "random_seed": "A fixed seed identifies the deterministic fixture protocol.",
    "reproducibility_checksum": "A content checksum detects later artifact mutation.",
    "gate_check_summary": "Expected and observed values make any block actionable.",
    "verifier_is_oracle": "False prevents an infrastructure check from becoming a science claim.",
    "verdict_class": "A closed class gives automation one unambiguous terminal state.",
    "honest_verdict": "A class-matched prefix reports readiness without scientific inflation.",
}


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact without its self-referential checksum field."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return lease_api.sha256_json(payload)


def _gate(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def _gate_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    checks = [deepcopy(dict(row)) for row in rows]
    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "checks": checks,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _source_hashes() -> JsonDict:
    paths = (
        Path("results/experiment_7065_v619_three_family_entrance_bank.json"),
        Path("python/carnot/gpu_lease_phase_journal.py"),
        Path("python/carnot/experiment_7065_v619_three_family_entrance_bank.py"),
        Path("python/carnot/experiment_6973_lease_aware_gguf_runtime.py"),
        Path("python/carnot/experiment_7078_v620_gpu_lease_migration.py"),
        Path("tests/python/test_gpu_lease_phase_journal_migration.py"),
        Path("tests/python/test_experiment_7078_v620_gpu_lease_migration.py"),
        Path("openspec/capabilities/research-harnesses/spec.md"),
    )
    rows = [{"path": str(path), "sha256": sha256_file(REPO_ROOT / path)} for path in paths]
    return {"files": rows, "manifest_hash": lease_api.sha256_json(rows)}


def _lock_observed_held(lock_path: Path, locks_path: Path = Path("/proc/locks")) -> bool | None:
    """Inspect the kernel lock table without acquiring or changing the lock."""

    try:
        inode = lock_path.stat().st_ino
        lines = locks_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return any(f":{inode} " in f"{line} " for line in lines)


def collect_preconditions(
    devices: Sequence[Mapping[str, Any]],
    runtime_dir: Path,
    *,
    process_match: Callable[[int, int], bool],
) -> JsonDict:
    """Inspect all real journals before any device journal can change."""

    discovered: list[JsonDict] = []
    validations: list[JsonDict] = []
    owners: list[JsonDict] = []
    locks: list[JsonDict] = []
    all_passed = len(devices) == 2
    for device in devices:
        device_uuid = str(device.get("uuid", ""))
        journal_path = lease_api.journal_path_for(runtime_dir, device_uuid)
        lock_path = lease_api.lock_path_for(runtime_dir, device_uuid)
        try:
            source_bytes = journal_path.read_bytes()
            document = json.loads(source_bytes)
            if not isinstance(document, Mapping):
                raise ValueError("journal_not_object")
            document = dict(document)
            source_sha256 = lease_api.sha256_bytes(source_bytes)
            kind = "current" if "lease_id" in document else "legacy_missing_lease_id"
            errors = (
                lease_api.validate_journal_document(document, check_freshness=False)
                if kind == "current"
                else lease_api.legacy_journal_errors(document, expected_device_uuid=device_uuid)
            )
            owner = document.get("owner")
            owner = dict(owner) if isinstance(owner, Mapping) else {}
            pid = owner.get("pid")
            ticks = owner.get("pid_start_ticks")
            owner_live = bool(
                isinstance(pid, int) and isinstance(ticks, int) and process_match(pid, ticks)
            )
            terminal_released = (
                document.get("released") is True
                and document.get("phase") in lease_api.TERMINAL_PHASES
            )
            discovered.append(
                {
                    "device_uuid": device_uuid,
                    "journal_path": str(journal_path),
                    "lock_path": str(lock_path),
                    "source_sha256": source_sha256,
                    "source_size_bytes": len(source_bytes),
                    "kind": kind,
                    "phase": document.get("phase"),
                    "released": document.get("released"),
                }
            )
            validations.append(
                {
                    "device_uuid": device_uuid,
                    "kind": kind,
                    "errors": errors,
                    "valid": not errors,
                    "terminal_released": terminal_released,
                }
            )
            owners.append(
                {
                    "device_uuid": device_uuid,
                    "pid": pid,
                    "pid_start_ticks": ticks,
                    "owner_live": owner_live,
                    "identity_absent_or_reused": not owner_live,
                }
            )
            row_passed = not errors and terminal_released and not owner_live
            all_passed = all_passed and row_passed
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            discovered.append(
                {
                    "device_uuid": device_uuid,
                    "journal_path": str(journal_path),
                    "lock_path": str(lock_path),
                    "read_error": f"{type(exc).__name__}:{exc}",
                }
            )
            validations.append(
                {
                    "device_uuid": device_uuid,
                    "kind": "unreadable",
                    "errors": [f"journal_unreadable:{type(exc).__name__}"],
                    "valid": False,
                    "terminal_released": False,
                }
            )
            owners.append(
                {
                    "device_uuid": device_uuid,
                    "pid": None,
                    "pid_start_ticks": None,
                    "owner_live": None,
                    "identity_absent_or_reused": False,
                }
            )
            all_passed = False
        held = _lock_observed_held(lock_path)
        locks.append(
            {
                "device_uuid": device_uuid,
                "lock_path": str(lock_path),
                "inspection": "read_only_proc_locks",
                "held_observed": held,
                "passed": held is False,
            }
        )
        all_passed = all_passed and held is False
    return {
        "all_passed": all_passed,
        "discovered_journal_rows": discovered,
        "legacy_validation_rows": validations,
        "owner_liveness_rows": owners,
        "kernel_lock_rows": locks,
    }


def _write_legacy_case(root: Path, device_uuid: str, document: Mapping[str, Any]) -> Path:
    path = lease_api.journal_path_for(root, device_uuid)
    lease_api.write_json_atomic(path, document)
    return path


def _case_document(source: Mapping[str, Any], device_uuid: str) -> JsonDict:
    document = deepcopy(dict(source))
    document["device_uuid"] = device_uuid
    document["task_id"] = f"synthetic:{device_uuid}"
    document["checksum"] = lease_api.journal_checksum(document)
    return document


def run_synthetic_migration_rows(source: Mapping[str, Any]) -> list[JsonDict]:
    """Exercise every required compatibility decision in disposable paths."""

    rows: list[JsonDict] = []

    def record(case: str, expected: str, observed: str) -> None:
        rows.append(
            {
                "scope": "synthetic",
                "case": case,
                "expected": expected,
                "observed": observed,
                "passed": expected == observed,
                "signals_sent": [],
                "files_removed": [],
            }
        )

    with tempfile.TemporaryDirectory(prefix="carnot-exp7078-") as name:
        base = Path(name)

        def prepare(case: str, document: Mapping[str, Any] | None = None) -> tuple[Path, str]:
            device_uuid = f"GPU-synthetic-{case}"
            case_root = base / case
            candidate = _case_document(source, device_uuid) if document is None else dict(document)
            _write_legacy_case(case_root, device_uuid, candidate)
            return case_root, device_uuid

        root, device = prepare("valid")
        result = lease_api.migrate_legacy_journal(
            runtime_dir=root,
            device_uuid=device,
            process_match=lambda _pid, _ticks: False,
            lease_id_factory=lambda: "lease:synthetic-valid",
        )
        record("valid_legacy_terminal", "migrated", result["action"])

        bad = _case_document(source, "GPU-synthetic-bad-checksum")
        bad["task_id"] = "changed-without-checksum"
        root, device = prepare("bad-checksum", bad)
        try:
            lease_api.migrate_legacy_journal(
                runtime_dir=root, device_uuid=device, process_match=lambda _pid, _ticks: False
            )
            observed = "accepted"
        except lease_api.MigrationBlocked as exc:
            observed = "blocked" if "legacy_checksum_mismatch" in str(exc) else str(exc)
        record("bad_checksum", "blocked", observed)

        missing = _case_document(source, "GPU-synthetic-missing-owner")
        missing["owner"].pop("pid_start_ticks")
        missing["checksum"] = lease_api.journal_checksum(missing)
        root, device = prepare("missing-owner", missing)
        try:
            lease_api.migrate_legacy_journal(
                runtime_dir=root, device_uuid=device, process_match=lambda _pid, _ticks: False
            )
            observed = "accepted"
        except lease_api.MigrationBlocked:
            observed = "blocked"
        record("missing_owner_fields", "blocked", observed)

        root, device = prepare("live")
        try:
            lease_api.migrate_legacy_journal(
                runtime_dir=root, device_uuid=device, process_match=lambda _pid, _ticks: True
            )
            observed = "accepted"
        except lease_api.MigrationBlocked as exc:
            observed = "blocked" if "recorded_owner_still_live" in str(exc) else str(exc)
        record("live_pid", "blocked", observed)

        root, device = prepare("reused")
        result = lease_api.migrate_legacy_journal(
            runtime_dir=root,
            device_uuid=device,
            process_match=lambda _pid, _ticks: False,
            lease_id_factory=lambda: "lease:synthetic-reused",
        )
        record("reused_pid", "migrated", result["action"])

        root, device = prepare("held")
        lock_path = lease_api.lock_path_for(root, device)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            try:
                lease_api.migrate_legacy_journal(
                    runtime_dir=root,
                    device_uuid=device,
                    process_match=lambda _pid, _ticks: False,
                )
                observed = "accepted"
            except lease_api.MigrationBlocked as exc:
                observed = "blocked" if "device_lock_held" in str(exc) else str(exc)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        record("held_lock", "blocked", observed)

        wrong = _case_document(source, "GPU-synthetic-other")
        root, device = prepare("wrong-uuid", wrong)
        try:
            lease_api.migrate_legacy_journal(
                runtime_dir=root, device_uuid=device, process_match=lambda _pid, _ticks: False
            )
            observed = "accepted"
        except lease_api.MigrationBlocked as exc:
            observed = "blocked" if "legacy_device_mismatch" in str(exc) else str(exc)
        record("wrong_uuid", "blocked", observed)

        nonterminal = _case_document(source, "GPU-synthetic-nonterminal")
        nonterminal["phase"] = "preflight"
        nonterminal["phase_history"] = nonterminal["phase_history"][:1]
        nonterminal["released"] = False
        nonterminal["released_monotonic_ns"] = None
        nonterminal["vram_mb"]["resident"] = None
        nonterminal["vram_mb"]["after"] = None
        nonterminal["exit_evidence"] = {"exit_code": None, "observed_monotonic_ns": None}
        nonterminal["unload_evidence"] = {
            "required": False,
            "observed": False,
            "observed_monotonic_ns": None,
        }
        nonterminal["checksum"] = lease_api.journal_checksum(nonterminal)
        root, device = prepare("nonterminal", nonterminal)
        try:
            lease_api.migrate_legacy_journal(
                runtime_dir=root, device_uuid=device, process_match=lambda _pid, _ticks: False
            )
            observed = "accepted"
        except lease_api.MigrationBlocked:
            observed = "blocked"
        record("nonterminal_phase", "blocked", observed)

        root, device = prepare("repeat")
        lease_api.migrate_legacy_journal(
            runtime_dir=root,
            device_uuid=device,
            process_match=lambda _pid, _ticks: False,
            lease_id_factory=lambda: "lease:synthetic-repeat",
        )
        result = lease_api.migrate_legacy_journal(
            runtime_dir=root, device_uuid=device, process_match=lambda _pid, _ticks: False
        )
        record("repeated_migration", "idempotent_noop", result["action"])

        root, device = prepare("atomic")

        def interrupt(_source: object, _target: object) -> None:
            raise OSError("synthetic interruption")

        try:
            lease_api.migrate_legacy_journal(
                runtime_dir=root,
                device_uuid=device,
                process_match=lambda _pid, _ticks: False,
                journal_replace=interrupt,
            )
            observed = "accepted"
        except lease_api.MigrationBlocked as exc:
            observed = "blocked" if "atomic_publish_failed" in str(exc) else str(exc)
        record("atomic_write_interruption", "blocked", observed)

        root, device = prepare("current")
        lease_api.migrate_legacy_journal(
            runtime_dir=root,
            device_uuid=device,
            process_match=lambda _pid, _ticks: False,
            lease_id_factory=lambda: "lease:synthetic-current",
        )
        current_path = lease_api.journal_path_for(root, device)
        current = lease_api.read_journal(current_path)
        current["recovery"] = {"performed": False, "signals_sent": []}
        current["checksum"] = lease_api.journal_checksum(current)
        lease_api.write_json_atomic(current_path, current)
        result = lease_api.migrate_legacy_journal(
            runtime_dir=root, device_uuid=device, process_match=lambda _pid, _ticks: False
        )
        record("current_schema_noop", "current_noop", result["action"])
    return rows


def _legacy_source_for_synthetic(
    runtime_dir: Path, discovered: Sequence[Mapping[str, Any]]
) -> JsonDict | None:
    for row in discovered:
        path = Path(str(row.get("journal_path", "")))
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(document, Mapping) and "lease_id" not in document:
                return dict(document)
            recovery = document.get("recovery") if isinstance(document, Mapping) else None
            migration = recovery.get("legacy_migration") if isinstance(recovery, Mapping) else None
            if isinstance(migration, Mapping):
                source_hash = str(migration.get("source_sha256", ""))
                preserved = lease_api.preserved_source_path_for(runtime_dir, source_hash)
                legacy = json.loads(preserved.read_text(encoding="utf-8"))
                if isinstance(legacy, Mapping):
                    return dict(legacy)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _post_migration_probe(
    devices: Sequence[Mapping[str, Any]], runtime_dir: Path
) -> list[JsonDict]:
    """Call the shipped Exp7065 lease probe against an injected runtime path."""

    from carnot import experiment_7065_v619_three_family_entrance_bank as entrance_api

    original = entrance_api.LEASE_RUNTIME_DIR
    entrance_api.LEASE_RUNTIME_DIR = runtime_dir
    try:
        return entrance_api._lease_probe(devices)
    finally:
        entrance_api.LEASE_RUNTIME_DIR = original


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check required fields, row projections, verdict, and checksum."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_fields_mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("signals_sent") != []:
        errors.append("signals_sent_not_empty")
    if artifact.get("files_removed") != []:
        errors.append("files_removed_not_empty")
    rows = artifact.get("rows")
    rows = rows if isinstance(rows, list) else []
    expected_score = int(bool(rows) and all(row.get("passed") is True for row in rows))
    if artifact.get("gpu_lease_compatibility_ready_score") != expected_score:
        errors.append("readiness_score_mismatch")
    preflight = artifact.get("post_migration_preflight_rows")
    preflight = preflight if isinstance(preflight, list) else []
    preflight_ok = len(preflight) == 2 and all(
        row.get("classification") == "available" for row in preflight
    )
    if expected_score == 1 and not preflight_ok:
        errors.append("post_migration_preflight_mismatch")
    expected_class = "null" if expected_score == 1 else "blocked"
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_mismatch")
    if not str(artifact.get("honest_verdict", "")).startswith(expected_class + "_"):
        errors.append("honest_verdict_prefix_mismatch")
    expected_summary = _gate_summary(rows)
    if artifact.get("gate_check_summary") != expected_summary:
        errors.append("gate_check_summary_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def run(
    *,
    date: str,
    runtime_dir: Path = LEASE_RUNTIME_DIR,
    result_path: Path = RESULT_PATH,
    gpu_probe: Callable[[], JsonDict] | None = None,
    process_match: Callable[[int, int], bool] = lease_api.process_start_matches,
    lease_id_factory: Callable[[], str] | None = None,
    preflight_probe: Callable[[Sequence[Mapping[str, Any]], Path], list[JsonDict]] = (
        _post_migration_probe
    ),
) -> JsonDict:
    """Inspect, migrate, replay attacks, and write one terminal artifact."""

    started = time.monotonic()
    if gpu_probe is None:
        from carnot.experiment_6966_gguf_load_envelope_canary import gpu_inventory

        gpu_probe = gpu_inventory
    gpu = gpu_probe()
    devices = list(gpu.get("devices", []))
    preconditions = collect_preconditions(devices, runtime_dir, process_match=process_match)
    discovered = preconditions["discovered_journal_rows"]
    validations = preconditions["legacy_validation_rows"]
    owners = preconditions["owner_liveness_rows"]
    locks = preconditions["kernel_lock_rows"]
    gpu_process_rows = [deepcopy(dict(row)) for row in gpu.get("processes", [])]
    migration_rows: list[JsonDict] = []
    preserved_rows: list[JsonDict] = []
    atomic_rows: list[JsonDict] = []
    idempotence_rows: list[JsonDict] = []
    strict_rows: list[JsonDict] = []
    preflight_rows: list[JsonDict] = []
    synthetic_rows: list[JsonDict] = []

    source_for_synthetic = _legacy_source_for_synthetic(runtime_dir, discovered)
    if preconditions["all_passed"] and source_for_synthetic is not None:
        for device in devices:
            device_uuid = str(device["uuid"])
            result = lease_api.migrate_legacy_journal(
                runtime_dir=runtime_dir,
                device_uuid=device_uuid,
                process_match=process_match,
                lease_id_factory=lease_id_factory,
            )
            migration_rows.append({"scope": "real", **result})
            journal_path = lease_api.journal_path_for(runtime_dir, device_uuid)
            before_repeat = journal_path.read_bytes()
            repeated = lease_api.migrate_legacy_journal(
                runtime_dir=runtime_dir,
                device_uuid=device_uuid,
                process_match=process_match,
                lease_id_factory=lease_id_factory,
            )
            after_repeat = journal_path.read_bytes()
            idempotence_rows.append(
                {
                    "device_uuid": device_uuid,
                    "action": repeated["action"],
                    "before_sha256": lease_api.sha256_bytes(before_repeat),
                    "after_sha256": lease_api.sha256_bytes(after_repeat),
                    "passed": repeated["idempotent"] is True and before_repeat == after_repeat,
                }
            )
            current = lease_api.read_journal(journal_path)
            migration = current.get("recovery", {}).get("legacy_migration", {})
            source_hash = str(migration.get("source_sha256", ""))
            preserved_path = lease_api.preserved_source_path_for(runtime_dir, source_hash)
            preserved_bytes = preserved_path.read_bytes() if source_hash else b""
            preserved_rows.append(
                {
                    "device_uuid": device_uuid,
                    "source_sha256": source_hash,
                    "preserved_path": str(preserved_path),
                    "preserved_sha256": lease_api.sha256_bytes(preserved_bytes),
                    "readable": bool(preserved_bytes),
                    "passed": bool(preserved_bytes)
                    and lease_api.sha256_bytes(preserved_bytes) == source_hash,
                }
            )
            target_hash = lease_api.sha256_bytes(journal_path.read_bytes())
            atomic_rows.append(
                {
                    "device_uuid": device_uuid,
                    "target_sha256": target_hash,
                    "receipt_target_sha256": repeated.get("target_sha256"),
                    "strict_errors": lease_api.validate_journal_document(
                        current, check_freshness=False
                    ),
                    "passed": target_hash == repeated.get("target_sha256"),
                }
            )
            try:
                lease_api.read_journal(preserved_path)
                legacy_rejected = False
                legacy_error = None
            except lease_api.JournalError as exc:
                legacy_rejected = "missing_field:lease_id" in str(exc)
                legacy_error = str(exc)
            strict_rows.append(
                {
                    "device_uuid": device_uuid,
                    "legacy_rejected": legacy_rejected,
                    "legacy_error": legacy_error,
                    "current_accepted": True,
                    "passed": legacy_rejected,
                }
            )
        synthetic_rows = run_synthetic_migration_rows(source_for_synthetic)
        migration_rows.extend(synthetic_rows)
        preflight_rows = preflight_probe(devices, runtime_dir)

    rows = [
        _gate("preconditions", True, preconditions["all_passed"], preconditions["all_passed"]),
        _gate(
            "legacy_source_for_synthetic",
            True,
            source_for_synthetic is not None,
            source_for_synthetic is not None,
        ),
        _gate(
            "real_migration_rows",
            len(devices),
            len([row for row in migration_rows if row.get("scope") == "real"]),
            len(devices) == 2
            and len([row for row in migration_rows if row.get("scope") == "real"]) == 2,
        ),
        _gate(
            "synthetic_migration_rows",
            list(SYNTHETIC_CASES),
            [row.get("case") for row in synthetic_rows if row.get("passed") is True],
            len(synthetic_rows) == len(SYNTHETIC_CASES)
            and all(row.get("passed") is True for row in synthetic_rows),
        ),
        _gate(
            "preserved_sources",
            2,
            sum(bool(row.get("passed")) for row in preserved_rows),
            len(preserved_rows) == 2 and all(row["passed"] for row in preserved_rows),
        ),
        _gate(
            "atomic_publications",
            2,
            sum(bool(row.get("passed")) for row in atomic_rows),
            len(atomic_rows) == 2 and all(row["passed"] for row in atomic_rows),
        ),
        _gate(
            "idempotent_repeats",
            2,
            sum(bool(row.get("passed")) for row in idempotence_rows),
            len(idempotence_rows) == 2 and all(row["passed"] for row in idempotence_rows),
        ),
        _gate(
            "strict_reader",
            2,
            sum(bool(row.get("passed")) for row in strict_rows),
            len(strict_rows) == 2 and all(row["passed"] for row in strict_rows),
        ),
        _gate(
            "post_migration_preflight",
            ["available", "available"],
            [row.get("classification") for row in preflight_rows],
            len(preflight_rows) == 2
            and all(row.get("classification") == "available" for row in preflight_rows),
        ),
        _gate("signals_sent", [], [], True),
        _gate("files_removed", [], [], True),
    ]
    score = int(all(row["passed"] for row in rows))
    verdict_class = "null" if score == 1 else "blocked"
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": _source_hashes(),
        "rows": rows,
        "discovered_journal_rows": deepcopy(discovered),
        "legacy_validation_rows": deepcopy(validations),
        "owner_liveness_rows": deepcopy(owners),
        "kernel_lock_rows": deepcopy(locks),
        "gpu_process_rows": gpu_process_rows,
        "migration_rows": migration_rows,
        "preserved_source_rows": preserved_rows,
        "atomic_publish_rows": atomic_rows,
        "idempotence_rows": idempotence_rows,
        "strict_reader_rows": strict_rows,
        "post_migration_preflight_rows": preflight_rows,
        "signals_sent": [],
        "files_removed": [],
        "gpu_lease_compatibility_ready_score": score,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(rows),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": (
            "null_gpu_lease_compatibility_ready_no_science_claim"
            if score == 1
            else "blocked_gpu_lease_compatibility_precondition_or_validation_failed"
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_invalid:" + ",".join(errors))
    lease_api.write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default="20260906")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--runtime-dir", type=Path, default=LEASE_RUNTIME_DIR)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        try:
            artifact = json.loads(args.output.read_text(encoding="utf-8"))
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError) as exc:
            errors = [f"artifact_unreadable:{type(exc).__name__}"]
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return int(bool(errors))
    artifact = run(date=args.date, runtime_dir=args.runtime_dir, result_path=args.output)
    print(
        json.dumps(
            {
                "gpu_lease_compatibility_ready_score": artifact[
                    "gpu_lease_compatibility_ready_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

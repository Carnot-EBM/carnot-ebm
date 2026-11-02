"""Account for attached boards from immutable bytes and measured service cost."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7806_v678_hardware_evidence import build_audit as prior_build_audit

ROOT = Path(__file__).resolve().parents[3]
RAW = "results/raw/experiment_7820_v679_hardware_evidence"
MANIFEST = f"{RAW}/validation_command_manifest.json"
OUTPUT = "results/experiment_7820_v679_hardware_evidence.json"
SERVICE = "results/experiment_7819_v679_service_cost.json"
PRE_GATE = "results/experiment_7819_service_cost.json"
PRIOR = "results/experiment_7806_v678_hardware_evidence.json"
COMMAND_NAMES = (
    "focused_pytest",
    "changed_code_coverage",
    "changed_code_coverage_report",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "repository_health",
    "task_entrypoint_e2e",
    "fresh_process_cold_replay",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Print actual elapsed time so an operator can see where work stopped."""
    print(
        f"[exp7820] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(
    path: str, digest: str | None, field: str, expected: Any, observed: Any, operator: str = "=="
) -> dict[str, Any]:
    """Keep both operands when a missing producer blocks a scientific claim."""
    passed = observed in expected if operator == "in" else observed == expected
    return {
        "upstream_id": "Exp7819",
        "artifact_path": path,
        "artifact_hash": digest,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def build_audit(root: Path, run_date: str, manifest_path: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7820-SERVICE: reuse authenticated board rows, then check current cost."""
    started = time.monotonic()
    progress(started, "preconditions", "start", 0)
    audit = prior_build_audit(root, run_date)
    audit["source_artifact_hashes"].pop("results/experiment_7805_v678_service_cost.json", None)
    audit["gate_check_summary"] = [
        c for c in audit["gate_check_summary"] if c["upstream_id"] != "Exp7805"
    ]
    path = root / SERVICE
    digest = sha256_file(path) if path.is_file() else None
    service = json.loads(path.read_text()) if digest else None
    receipt = {
        "path": SERVICE,
        "sha256": digest,
        "date": service.get("run_date") if service else None,
        "imported_fields": [
            "schema",
            "verdict_class",
            "service_evidence_ready_score",
            "stage_times_ms",
        ],
        "scope": "current_science_producer",
        "eligible": False,
    }
    checks = [
        _check(
            SERVICE,
            digest,
            "service_evidence_ready_score",
            1,
            service.get("service_evidence_ready_score") if service else None,
        )
    ]
    if service:
        checks.extend(
            [
                _check(SERVICE, digest, "experiment_id", 7819, service.get("experiment_id")),
                _check(
                    SERVICE, digest, "schema", "carnot.experiment_7819.v1", service.get("schema")
                ),
                _check(
                    SERVICE,
                    digest,
                    "flagged_adversarial",
                    False,
                    service.get("flagged_adversarial"),
                ),
                _check(
                    SERVICE,
                    digest,
                    "verdict_class",
                    ["positive", "null", "circular_positive"],
                    service.get("verdict_class"),
                    "in",
                ),
            ]
        )
    times = service.get("stage_times_ms") if service else None
    host = times.get("host_stage") if isinstance(times, dict) else None
    whole = times.get("whole_service") if isinstance(times, dict) else None
    measured = (
        isinstance(host, (int, float))
        and not isinstance(host, bool)
        and isinstance(whole, (int, float))
        and not isinstance(whole, bool)
        and math.isfinite(host)
        and math.isfinite(whole)
        and 0 <= host <= whole
        and whole > 0
    )
    if service:
        checks.append(_check(SERVICE, digest, "stage_times_ms.valid", True, measured))
    status = (
        "missing"
        if service is None
        else "qualified"
        if all(c["passed"] for c in checks)
        else "disqualified"
    )
    receipt["eligible"] = status == "qualified"
    audit["source_artifact_hashes"][SERVICE] = receipt
    prior_path = root / PRIOR
    prior = json.loads(prior_path.read_text()) if prior_path.is_file() else None
    audit["source_artifact_hashes"][PRIOR] = {
        "path": PRIOR,
        "sha256": sha256_file(prior_path) if prior else None,
        "date": prior.get("run_date") if prior else None,
        "imported_fields": ["honest_verdict", "verdict_class"],
        "scope": "prior_failure_context",
        "eligible": False,
    }
    audit["gate_check_summary"].extend(c for c in checks if not c["passed"])
    fraction = host / whole if status == "qualified" else None
    audit.update(
        {
            "schema": "carnot.experiment_7820.v1",
            "experiment_id": 7820,
            "milestone": "2026.09.679",
            "run_date": run_date,
            "honest_verdict": "complete_blocked_missing_service_evidence"
            if status != "qualified"
            else "complete_null_hardware_accounting_only",
            "verdict_class": "blocked" if status != "qualified" else "null",
            "service_source_status": status,
            "service_measured": status == "qualified",
            "host_stage_fraction": fraction,
            "acceleration_bound": 1 / (1 - fraction)
            if fraction is not None and fraction < 1
            else None,
        }
    )
    for row in audit["board_rows"]:
        row["service_fraction"] = fraction
    for row in audit["rows"]:
        if row["board"] in {"KV260", "PolarFire", "GateMate"}:
            row["service_fraction"] = fraction
    audit["rows_checksum"] = canonical_hash(audit["rows"])
    audit["preconditions_checked"]["declared_science_producer"] = SERVICE
    audit["preconditions_checked"]["conductor_pre_gate_receipt"] = {
        "path": PRE_GATE,
        "sha256": sha256_file(root / PRE_GATE) if (root / PRE_GATE).is_file() else None,
        "is_science_producer": False,
    }
    audit["preconditions_checked"]["required_resources"] = {"cpu": True, "board_operations": False}
    audit["acquisition_recommendation"] = {
        "decision": "defer",
        "basis": "No qualified local whole-service board benefit",
        "vendor_z1t_estimate_is_local_measurement": False,
    }
    audit["stage_opportunity_map"] = {
        "CPU counters": "measured host stage only when qualified",
        "Rust batching": "measured host dispatch only when qualified",
        "FPGA predicate matching": "requires a matched fabric transcript",
        "preparation_ms": None,
        "transfer_ms": None,
        "readout_ms": None,
    }
    audit["claim_scope"].update(
        {
            "NPU": "local access unqualified",
            "TSU": "local access unqualified; vendor Z1T estimates are not local latency or power",
            "source_family_exposure": "640 exposed development families",
        }
    )
    audit["acceptance_gate_results"] = {
        "validity": not audit["evidence_unresolved"],
        "readiness": 0,
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    audit["validation_command_manifest_path"] = str(manifest_path)
    audit["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    audit["observed_child_commands"] = []
    audit["repository_health"] = {"status": "unstarted"}
    audit["prior_failure_metadata"] = {
        "experiment_id": 7806,
        "honest_verdict": prior.get("honest_verdict") if prior else None,
        "verdict_class": prior.get("verdict_class") if prior else None,
        "cause": "nested pytest basetemp parent missing",
        "same_verdict_retired": False,
    }
    audit["reproducibility_checksum"] = canonical_hash(
        {
            "code": sha256_file(Path(__file__)),
            "cli": sha256_file(
                root / "scripts/experiments/experiment_7820_v679_hardware_evidence.py"
            )
            if (root / "scripts/experiments/experiment_7820_v679_hardware_evidence.py").is_file()
            else None,
            "manifest": audit["validation_command_manifest_sha256"],
            "inputs": {
                key: value["sha256"] for key, value in audit["source_artifact_hashes"].items()
            },
            "roles": "read_only_board_and_service_reducer",
            "seed": None,
        }
    )
    audit["field_principles"].update(
        {
            key: "Bind claims to named bytes and measured cost."
            for key in (
                "validation_command_manifest_path",
                "validation_command_manifest_sha256",
                "observed_child_commands",
                "repository_health",
                "prior_failure_metadata",
            )
        }
    )
    audit["duration_s"] = time.monotonic() - started
    audit["phase_spans"] = [
        {
            "phase": "preconditions_and_reduction",
            "start_monotonic_s": started,
            "end_monotonic_s": time.monotonic(),
            "duration_s": time.monotonic() - started,
            "completed_units": len(audit["board_rows"]),
        }
    ]
    progress(started, "reduction", "complete", len(audit["board_rows"]))
    return audit


def _manifest(path: Path) -> dict[str, Any]:
    """Reject any child outside the prospectively frozen command sequence."""
    value = json.loads(path.read_text())
    commands = value["commands"]
    if [item["name"] for item in commands] != list(COMMAND_NAMES):
        raise ValueError("undeclared child or missing required command")
    for item in commands:
        expected = "diagnostic" if item["name"] == "repository_health" else "required"
        if item["classification"] != expected or not item["argv"]:
            raise ValueError("undeclared child classification or empty argv")
    return value


def run_child(root: Path, spec: dict[str, Any], durable: Path) -> dict[str, Any]:
    """Create pytest parents, wait for exit, then seal one immutable byte log."""
    started = time.monotonic()
    name = spec["name"]
    private = Path(spec["private_root"])
    private.mkdir(parents=True, exist_ok=False)
    for arg in spec["argv"]:
        if arg.startswith("--basetemp="):
            Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    live = private / "live.log"
    progress(started, name, "before_subprocess", 0)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root / 'python'}:{root}"
    env["PYTHONUNBUFFERED"] = "1"
    timed_out = False
    with live.open("wb") as handle:
        child = subprocess.Popen(
            spec["argv"], cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        while True:
            try:
                child.wait(
                    timeout=min(50, max(0.01, spec["timeout_s"] - (time.monotonic() - started)))
                )
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                progress(started, name, "subprocess_outstanding", 0)
                if elapsed >= spec["timeout_s"]:
                    timed_out = True
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
    digest = sha256_file(live)
    sealed = durable / private.parent.name / private.name / name / f"{digest.split(':', 1)[1]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.exists():
        raise FileExistsError(f"sealed log already exists: {sealed}")
    shutil.copyfile(live, sealed)
    if sha256_file(sealed) != digest:
        raise ValueError("sealed log copy changed bytes")
    receipt = {
        "name": name,
        "command_argv": spec["argv"],
        "classification": spec["classification"],
        "exit_code": child.returncode,
        "passed": child.returncode == 0 and not timed_out,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - started,
        "log_path": str(sealed),
        "log_sha256": sha256_file(sealed),
        "output_tail": sealed.read_text(errors="replace")[-4000:],
    }
    progress(started, name, "after_subprocess", 1)
    return receipt


def dispatch(
    root: Path,
    manifest_path: Path,
    durable: Path,
    executor: Callable[[Path, dict[str, Any], Path], dict[str, Any]] = run_child,
    before_child: Callable[[int, list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """SCENARIO-REPORT-7820-DISPATCH: use exactly the frozen argv and class."""
    commands = _manifest(manifest_path)["commands"]
    receipts: list[dict[str, Any]] = []
    for index, spec in enumerate(commands):
        if before_child:
            before_child(index, receipts)
        receipt = executor(root, spec, durable)
        if (
            receipt["name"] != spec["name"]
            or receipt["command_argv"] != spec["argv"]
            or receipt["classification"] != spec["classification"]
        ):
            raise ValueError("undeclared child receipt")
        receipts.append(receipt)
    return receipts


def cold_reduce(candidate: Path, root: Path, manifest_path: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7820-CUSTODY: rehash rows, inputs, manifest, and sealed logs."""
    artifact = json.loads(candidate.read_text())
    fresh = build_audit(root, artifact["run_date"], manifest_path)
    if artifact["source_artifact_hashes"] != fresh["source_artifact_hashes"]:
        raise ValueError("source_hash_mismatch")
    if artifact["board_rows"] != fresh["board_rows"]:
        raise ValueError("board_row_mismatch")
    if artifact["rows"] != fresh["rows"] or artifact["rows_checksum"] != fresh["rows_checksum"]:
        raise ValueError("row_mismatch")
    current_checks = [
        c for c in artifact["gate_check_summary"] if c["upstream_id"] != "Exp7820:validation"
    ]
    if current_checks != fresh["gate_check_summary"]:
        raise ValueError("gate_mismatch")
    for key in (
        "service_source_status",
        "host_stage_fraction",
        "acceleration_bound",
        "acquisition_recommendation",
        "stage_opportunity_map",
        "claim_scope",
        "sample_size_budget",
        "reproducibility_checksum",
        "validation_command_manifest_sha256",
    ):
        if artifact[key] != fresh[key]:
            raise ValueError(f"summary_mismatch:{key}")
    raw_path = artifact.get("raw_rows_path")
    if raw_path:
        if sha256_file(Path(raw_path)) != artifact["raw_rows_sha256"]:
            raise ValueError("raw_hash_mismatch")
        if json.loads(Path(raw_path).read_text()) != artifact["rows"]:
            raise ValueError("raw_row_mismatch")
    for receipt in artifact.get("validation_receipts", {}).get("checks", []):
        path = Path(receipt["log_path"])
        if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
            raise ValueError("log_hash_mismatch")
    return {"row_count": len(fresh["board_rows"]), "rows_checksum": fresh["rows_checksum"]}


def main(
    argv: list[str] | None = None,
    executor: Callable[[Path, dict[str, Any], Path], dict[str, Any]] = run_child,
) -> int:
    """SCENARIO-REPORT-7820-DISPATCH: publish only after declared checks finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    root = args.root.resolve()
    manifest_path = args.manifest or root / MANIFEST
    progress(started, "entrypoint", "start", 0)
    if args.cold_replay:
        progress(started, "cold_replay", "before_reduction", 0)
        print(
            json.dumps(cold_reduce(args.cold_replay, root, manifest_path), sort_keys=True),
            flush=True,
        )
        progress(started, "cold_replay", "after_reduction", 3)
        return 0
    audit = build_audit(root, args.date, manifest_path)
    if args.prepare:
        atomic_json(args.prepare, audit)
        progress(started, "prepare", "complete", len(audit["board_rows"]))
        return 0
    manifest = _manifest(manifest_path)
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    raw_bytes = (json.dumps(audit["rows"], sort_keys=True, separators=(",", ":")) + "\n").encode()
    raw_digest = canonical_hash(audit["rows"])
    raw_path = raw / "rows" / f"{raw_digest.split(':', 1)[1]}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        if raw_path.read_bytes() != raw_bytes:
            raise ValueError("immutable raw row path has changed bytes")
    else:
        raw_path.write_bytes(raw_bytes)
    audit["raw_rows_path"] = str(raw_path)
    audit["raw_rows_sha256"] = sha256_file(raw_path)
    audit["field_principles"].update(
        {
            "raw_rows_path": "Bind reduced rows to durable raw bytes.",
            "raw_rows_sha256": "Detect any later raw-row edit.",
        }
    )
    candidate = Path(manifest["candidate_path"])
    candidate.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(candidate, audit)
    progress(started, "validation", "before_subprocesses", 0)
    validation_started = time.monotonic()

    def before_child(index: int, receipts: list[dict[str, Any]]) -> None:
        audit["validation_receipts"]["checks"] = list(receipts)
        atomic_json(candidate, audit)
        progress(started, "validation", "command_boundary", index)

    receipts = dispatch(
        root, manifest_path, raw / "validation", executor=executor, before_child=before_child
    )
    progress(started, "validation", "after_subprocesses", len(receipts))
    audit["observed_child_commands"] = [spec for spec in manifest["commands"]]
    required = [r for r in receipts if r["classification"] == "required"]
    passed = all(r["passed"] and r["exit_code"] == 0 for r in required)
    audit["validation_receipts"] = {
        "checks": receipts,
        "required_checks_passed": passed,
        "cold_replay": next(r for r in receipts if r["name"] == "fresh_process_cold_replay"),
        "terminal_readers": [
            r
            for r in receipts
            if r["name"] in {"adversarial_verify", "verdict_row_consistency_strict"}
        ],
        "frozen_scope_path": str(manifest_path),
    }
    health = next(r for r in receipts if r["name"] == "repository_health")
    audit["repository_health"] = {
        "status": "passed" if health["passed"] else "failed_diagnostic",
        "exit_code": health["exit_code"],
        "timed_out": health.get("timed_out", False),
        "log_path": health["log_path"],
        "log_sha256": health["log_sha256"],
    }
    audit["flagged_adversarial"] = any(
        r["name"] == "adversarial_verify" and not r["passed"] for r in receipts
    )
    if not passed:
        audit["honest_verdict"] = "complete_disqualified_required_checks"
        audit["verdict_class"] = "disqualified"
        audit["acceptance_gate_results"] = {key: 0 for key in audit["acceptance_gate_results"]}
        audit["gate_check_summary"].extend(
            {
                "upstream_id": "Exp7820:validation",
                "artifact_path": r["log_path"],
                "artifact_hash": r["log_sha256"],
                "field": r["name"],
                "operator": "==",
                "expected": 0,
                "observed": r["exit_code"],
                "passed": False,
            }
            for r in required
            if not r["passed"]
        )
    audit["prior_failure_metadata"]["same_verdict_retired"] = (
        audit["honest_verdict"] == audit["prior_failure_metadata"]["honest_verdict"]
    )
    ended = time.monotonic()
    audit["duration_s"] = ended - started
    audit["phase_spans"].append(
        {
            "phase": "validation_and_publication",
            "start_monotonic_s": validation_started,
            "end_monotonic_s": ended,
            "duration_s": ended - validation_started,
            "completed_units": len(receipts),
        }
    )
    if executor is run_child:
        atomic_json(candidate, audit)
        cold_reduce(candidate, root, manifest_path)
    output = args.output or root / OUTPUT
    progress(started, "publish", "before_atomic_write", len(receipts))
    atomic_json(output, audit)
    progress(started, "publish", "after_atomic_write", len(receipts))
    return 0

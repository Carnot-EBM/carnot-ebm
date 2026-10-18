"""Audit board custody against current, measured whole-service cost."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7793_v677_hardware_evidence import (
    build_candidate as historical_candidate,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    run_commands,
    run_scoped_validation,
)


ROOT = Path(__file__).resolve().parents[3]
SERVICE = "results/experiment_7805_v678_service_cost.json"
PRE_GATE = "results/experiment_7805_service_cost.json"
HISTORY = "results/experiment_7793_v677_hardware_evidence.json"
OUTPUT = "results/experiment_7806_v678_hardware_evidence.json"
RAW = "results/raw/experiment_7806_v678_hardware_evidence"
IMMUTABLE_BOARD_HASHES = {
    "KV260": "sha256:acdfd841c75649279515fab6b91115d07587094b8703cc72a04febae123236d2",
    "PolarFire": "sha256:341ca079f8ca42ed26dcda1d57edba21cede9c0ca965265fed05121feec2b588",
    "GateMate": "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66",
}
INPUTS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "openspec/capabilities/research-reporting/spec.md",
    "research-hardware-wishlist.md",
    "ops/hardware-bringup-prep.md",
    "ops/north-star.md",
    "results/experiment_7793_v677_hardware_evidence.json",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Show elapsed work at each boundary so a stuck audit is observable."""
    print(
        f"[exp7806] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _receipt(root: Path, rel: str, fields: list[str], scope: str) -> dict[str, Any]:
    """Bind imported fields to current file bytes, including absent files."""
    path = root / rel
    digest = sha256_file(path) if path.is_file() else None
    return {
        "path": rel,
        "sha256": digest,
        "date": None,
        "imported_fields": fields,
        "scope": scope,
        "eligible": False,
    }


def _gate(
    path: str, digest: str | None, field: str, expected: Any, observed: Any, operator: str = "=="
) -> dict[str, Any]:
    """Record both gate operands so absent evidence remains distinguishable."""
    passed = observed in expected if operator == "in" else observed == expected
    return {
        "upstream_id": "Exp7805",
        "artifact_path": path,
        "artifact_hash": digest,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def build_audit(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7806-SERVICE: reduce bytes without probing a board."""
    started = time.monotonic()
    root = root.resolve()
    progress(started, "preconditions", "start", 0)
    input_checks = []
    for rel in INPUTS:
        path = root / rel
        input_checks.append(
            {
                "path": rel,
                "sha256": sha256_file(path) if path.is_file() else None,
                "present": path.is_file(),
            }
        )
    service_receipt = _receipt(
        root,
        SERVICE,
        ["stage_times_ms", "service_evidence_ready_score"],
        "current_science_producer",
    )
    service = json.loads((root / SERVICE).read_text()) if service_receipt["sha256"] else None
    progress(started, "preconditions", "inputs_complete", len(input_checks) + 1)
    audit = historical_candidate(root, run_date)
    # The inherited reducer authenticates six historical records. Its old
    # service gate belongs to Exp7792 and has no authority in this milestone.
    old_service = "results/experiment_7792_v677_service_cost.json"
    audit["source_artifact_hashes"].pop(old_service, None)
    audit["gate_check_summary"] = [
        c for c in audit["gate_check_summary"] if c["upstream_id"] != "Exp7792"
    ]
    audit["preconditions_checked"]["custody_checks"] = [
        c for c in audit["preconditions_checked"]["custody_checks"] if c["upstream_id"] != "Exp7792"
    ]
    if service:
        service_receipt["date"] = service.get("run_date")
    checks = [
        _gate(
            SERVICE,
            service_receipt["sha256"],
            "service_evidence_ready_score",
            1,
            service.get("service_evidence_ready_score") if service else None,
        )
    ]
    if service:
        checks.extend(
            [
                _gate(
                    SERVICE,
                    service_receipt["sha256"],
                    "experiment_id",
                    7805,
                    service.get("experiment_id"),
                ),
                _gate(
                    SERVICE,
                    service_receipt["sha256"],
                    "flagged_adversarial",
                    False,
                    service.get("flagged_adversarial"),
                ),
                _gate(
                    SERVICE,
                    service_receipt["sha256"],
                    "verdict_class",
                    ["positive", "null", "circular_positive"],
                    service.get("verdict_class"),
                    "in",
                ),
            ]
        )
    times = service.get("stage_times_ms", {}) if service else {}
    host = times.get("host_stage") if isinstance(times, dict) else None
    whole = times.get("whole_service") if isinstance(times, dict) else None
    measured = isinstance(host, (int, float)) and isinstance(whole, (int, float))
    measured = measured and not isinstance(host, bool) and not isinstance(whole, bool)
    measured = measured and 0 <= host <= whole and whole > 0
    if service:
        checks.append(
            _gate(SERVICE, service_receipt["sha256"], "stage_times_ms.valid", True, measured)
        )
    status = (
        "missing"
        if service is None
        else "qualified"
        if all(c["passed"] for c in checks)
        else "disqualified"
    )
    service_receipt["eligible"] = status == "qualified"
    audit["source_artifact_hashes"][SERVICE] = service_receipt
    historical = _receipt(root, HISTORY, ["honest_verdict", "verdict_class"], "prior_context")
    prior = json.loads((root / HISTORY).read_text()) if historical["sha256"] else None
    historical["date"] = prior.get("run_date") if prior else None
    audit["source_artifact_hashes"][HISTORY] = historical
    audit["gate_check_summary"].extend(c for c in checks if not c["passed"])
    schema_checks = []
    for rel, expected in (
        ("results/experiment_7751_v674_hardware_continuity.json", "carnot.experiment_7751.v1"),
        ("results/experiment_7779_v676_hardware_evidence.json", "carnot.experiment_7779.v1"),
        (HISTORY, "carnot.experiment_7793.v1"),
    ):
        source = root / rel
        observed = json.loads(source.read_text()).get("schema") if source.is_file() else None
        schema_checks.append(
            {
                "upstream_id": rel,
                "artifact_path": rel,
                "artifact_hash": sha256_file(source) if source.is_file() else None,
                "field": "schema",
                "operator": "==",
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    audit["gate_check_summary"].extend(c for c in schema_checks if not c["passed"])
    board_hash_checks = [
        {
            "upstream_id": f"Exp7751:{row['board']}",
            "artifact_path": row["source_path"],
            "artifact_hash": row["source_hash"],
            "field": "immutable_original_sha256",
            "operator": "==",
            "expected": IMMUTABLE_BOARD_HASHES[row["board"]],
            "observed": row["source_hash"],
            "passed": row["source_hash"] == IMMUTABLE_BOARD_HASHES[row["board"]],
        }
        for row in audit["board_rows"]
    ]
    audit["gate_check_summary"].extend(c for c in board_hash_checks if not c["passed"])
    missing_inputs = [item for item in input_checks if not item["present"]]
    for item in missing_inputs:
        audit["gate_check_summary"].append(
            {
                "upstream_id": "Exp7806:precondition",
                "artifact_path": item["path"],
                "artifact_hash": None,
                "field": "present",
                "operator": "==",
                "expected": True,
                "observed": False,
                "passed": False,
            }
        )
    progress(started, "preconditions", "complete", len(input_checks) + len(checks))
    custody_ok = (
        not audit["evidence_unresolved"]
        and not missing_inputs
        and all(c["passed"] for c in (*schema_checks, *board_hash_checks))
    )
    fraction = host / whole if status == "qualified" else None
    bound = 1 / (1 - fraction) if fraction is not None and fraction < 1 else None
    for row in audit["board_rows"]:
        row["service_fraction"] = fraction
        row["acquisition_relevance"] = "defer: no measured board whole-service benefit"
    audit["schema"] = "carnot.experiment_7806.v1"
    audit["experiment_id"] = 7806
    audit["milestone"] = "2026.09.678"
    audit["honest_verdict"] = (
        "complete_blocked_historical_evidence"
        if not custody_ok
        else "complete_blocked_missing_service_evidence"
        if status != "qualified"
        else "complete_null_hardware_accounting_only"
    )
    audit["verdict_class"] = (
        "blocked" if audit["honest_verdict"].startswith("complete_blocked") else "null"
    )
    audit["rows_checksum"] = canonical_hash(audit["rows"])
    audit["service_measured"] = status == "qualified"
    audit["service_source_status"] = status
    audit["host_stage_fraction"] = fraction
    audit["acceleration_bound"] = bound
    audit["stage_opportunity_map"] = {
        "CPU counters": "host_stage; reduce count and dispatch overhead",
        "Rust batching": "host_stage; amortize repeated host calls",
        "FPGA predicate matching": "candidate board stage; requires a matching fabric transcript",
        "preparation_ms": None,
        "transfer_ms": None,
        "readout_ms": None,
    }
    audit["acquisition_recommendation"] = {
        "decision": "defer",
        "basis": "No qualified local whole-service board benefit",
        "vendor_z1t_estimate_is_local_measurement": False,
    }
    audit["terminal_scope"] = "historical_board_custody_only"
    audit["reopen_condition"] = {
        row["board"]: row["next_missing_prerequisite"] for row in audit["board_rows"]
    }
    audit["claim_scope"]["acquisition"] = (
        "defer until a matched board whole-service run is measured"
    )
    audit["claim_scope"]["NPU"] = "local access unqualified"
    audit["claim_scope"]["TSU"] = (
        "local access unqualified; vendor Z1T estimates are not local numbers"
    )
    audit["claim_scope"]["source_family_exposure"] = "640 exposed development families"
    audit["preconditions_checked"]["input_checks"] = input_checks
    audit["preconditions_checked"]["schema_checks"] = schema_checks
    audit["preconditions_checked"]["immutable_board_hash_checks"] = board_hash_checks
    audit["preconditions_checked"]["declared_science_producer"] = SERVICE
    audit["preconditions_checked"]["conductor_pre_gate_receipt"] = {
        "path": PRE_GATE,
        "sha256": sha256_file(root / PRE_GATE) if (root / PRE_GATE).is_file() else None,
        "is_science_producer": False,
    }
    audit["preconditions_checked"]["historical_exp7793_verdict"] = (
        prior.get("honest_verdict") if prior else None
    )
    audit["acceptance_gate_results"] = {
        "validity": custody_ok,
        "readiness": 0,
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    audit["sample_size_budget"] = {
        "intended": 3,
        "eligible": 2 if custody_ok else 0,
        "started": 0,
        "completed": 0,
        "excluded": 1 if custody_ok else 3,
        "censored": 0,
        "effective_independent_n": 2 if custody_ok else 0,
    }
    audit["inference_substrate"] = "aggregation_from_upstream_artifacts"
    audit["inference_substrate_class"] = "aggregation"
    audit["planned_inference_substrate_class"] = "aggregation"
    audit["actual_inference_substrate_class"] = "aggregation"
    audit["MODEL_SPECS"] = []
    audit["model_specs"] = []
    audit["model_invocation_counts"] = {
        "loads": 0,
        "generations": 0,
        "tokens": 0,
        "loaded_files": [],
    }
    audit["verifier_is_oracle"] = False
    audit["flagged_adversarial"] = False
    audit["validation_receipts"] = {
        "frozen_scope_path": "/tmp/carnot-7806/frozen_scope.json",
        "checks": [],
        "required_checks_passed": False,
        "cold_replay": None,
        "terminal_readers": [],
    }
    elapsed = time.monotonic() - started
    audit["duration_s"] = elapsed
    audit["phase_spans"] = [
        {
            "phase": "preconditions_and_reduction",
            "start_monotonic_s": started,
            "end_monotonic_s": started + elapsed,
            "duration_s": elapsed,
            "completed_units": len(audit["rows"]),
        }
    ]
    audit["reproducibility_checksum"] = canonical_hash(
        {
            "code": sha256_file(Path(__file__)),
            "cli_code": sha256_file(
                root / "scripts/experiments/experiment_7806_v678_hardware_evidence.py"
            )
            if (root / "scripts/experiments/experiment_7806_v678_hardware_evidence.py").is_file()
            else None,
            "inputs": {
                key: value["sha256"] for key, value in audit["source_artifact_hashes"].items()
            },
            "required_inputs": {item["path"]: item["sha256"] for item in input_checks},
            "frozen_scope": sha256_file(Path("/tmp/carnot-7806/frozen_scope.json"))
            if Path("/tmp/carnot-7806/frozen_scope.json").is_file()
            else None,
            "roles": "read_only_board_and_service_reducer",
            "seed": None,
            "parameters": {"boards": ["KV260", "PolarFire", "GateMate"]},
        }
    )
    audit["field_principles"].update(
        {
            key: "Keep source bytes and claim limits visible."
            for key in (
                "service_source_status",
                "host_stage_fraction",
                "acceleration_bound",
                "stage_opportunity_map",
                "acquisition_recommendation",
                "terminal_scope",
                "reopen_condition",
            )
        }
    )
    progress(started, "reduction", "complete", len(audit["board_rows"]))
    return audit


def cold_reduce(candidate: Path, root: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7806-CUSTODY: independently replay current source bytes."""
    artifact = json.loads(candidate.read_text())
    fresh = build_audit(root, artifact["run_date"])
    for rel, receipt in artifact["source_artifact_hashes"].items():
        if receipt != fresh["source_artifact_hashes"].get(rel):
            raise ValueError(f"source_hash_mismatch:{rel}")
    if set(artifact["source_artifact_hashes"]) != set(fresh["source_artifact_hashes"]):
        raise ValueError("source_set_mismatch")
    if artifact["board_rows"] != fresh["board_rows"]:
        raise ValueError("board_row_mismatch")
    if artifact["rows"] != fresh["rows"] or artifact["rows_checksum"] != fresh["rows_checksum"]:
        raise ValueError("row_mismatch")
    current_checks = [
        c for c in artifact["gate_check_summary"] if c["upstream_id"] != "Exp7806:validation"
    ]
    if current_checks != fresh["gate_check_summary"]:
        raise ValueError("gate_mismatch")
    if artifact["service_source_status"] != fresh["service_source_status"]:
        raise ValueError("service_status_mismatch")
    for field in (
        "host_stage_fraction",
        "acceleration_bound",
        "acquisition_recommendation",
        "stage_opportunity_map",
        "terminal_scope",
        "reopen_condition",
        "claim_scope",
        "sample_size_budget",
        "reproducibility_checksum",
    ):
        if artifact[field] != fresh[field]:
            raise ValueError(f"summary_mismatch:{field}")
    recorded = artifact["validation_receipts"]
    failed = any(not r["passed"] for r in recorded["checks"])
    disqualified = artifact["verdict_class"] == "disqualified"
    if disqualified != failed:
        raise ValueError("validation_class_mismatch")
    expected_gates = (
        {key: 0 for key in fresh["acceptance_gate_results"]}
        if disqualified
        else fresh["acceptance_gate_results"]
    )
    if artifact["acceptance_gate_results"] != expected_gates:
        raise ValueError("summary_mismatch:acceptance_gate_results")
    expected_flag = any(
        r["name"] == "adversarial_verify" and not r["passed"] for r in recorded["checks"]
    )
    if artifact["flagged_adversarial"] != expected_flag:
        raise ValueError("adversarial_flag_mismatch")
    for field in ("input_checks", "schema_checks", "immutable_board_hash_checks"):
        if artifact["preconditions_checked"][field] != fresh["preconditions_checked"][field]:
            raise ValueError(f"precondition_mismatch:{field}")
    return {"row_count": len(fresh["board_rows"]), "rows_checksum": fresh["rows_checksum"]}


def _validate(root: Path, candidate: Path) -> list[dict[str, Any]]:
    """Run the frozen affected commands and terminal readers with durable logs."""
    frozen = json.loads(Path("/tmp/carnot-7806/frozen_scope.json").read_text())
    raw = root / RAW / "validation"
    Path("/tmp/carnot-7806/basetemp").mkdir(parents=True, exist_ok=True)
    scope = run_scoped_validation(
        root,
        frozen["tests"],
        frozen["changed_modules"],
        static_paths=frozen["static_paths"],
        basetemp=Path("/tmp/carnot-7806/basetemp"),
        coverage_file=Path("/tmp/carnot-7806/.coverage"),
        log_dir=raw / "affected",
    )
    receipts = list(scope["validation_receipts"])
    commands = [
        CommandSpec(item["name"], tuple(item["argv"]), "new_module_and_cli")
        for item in frozen["supplemental_commands"]
    ] + [
        CommandSpec(
            "fresh_process_cold_replay",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/experiments/experiment_7806_v678_hardware_evidence.py",
                "--cold-replay",
                str(candidate),
            ),
            "task_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                str(candidate),
            ),
            "task_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "task_candidate",
        ),
    ]
    receipts.extend(run_commands(root, commands, log_dir=raw / "terminal", heartbeat_s=50))
    return receipts


def main(argv: list[str] | None = None) -> int:
    """SCENARIO-REPORT-7806-TERMINAL: validate before atomic publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    root = args.root.resolve()
    progress(started, "entrypoint", "start", 0)
    if args.cold_replay:
        progress(started, "cold_replay", "before_reduction", 0)
        print(json.dumps(cold_reduce(args.cold_replay, root), sort_keys=True), flush=True)
        progress(started, "cold_replay", "after_reduction", 3)
        return 0
    audit = build_audit(root, args.date)
    if args.prepare:
        atomic_json(args.prepare, audit)
        progress(started, "prepare", "complete", len(audit["board_rows"]))
        return 0
    candidate = Path(os.environ.get("CARNOT_7806_CANDIDATE", "/tmp/carnot-7806/candidate.json"))
    atomic_json(candidate, audit)
    progress(started, "validation", "before_subprocesses", 0)
    validation_started = time.monotonic()
    receipts = _validate(root, candidate)
    progress(started, "validation", "after_subprocesses", len(receipts))
    passed = all(item["passed"] and item["exit_code"] == 0 for item in receipts)
    audit["flagged_adversarial"] = any(
        r["name"] == "adversarial_verify" and not r["passed"] for r in receipts
    )
    audit["validation_receipts"] = {
        "frozen_scope_path": "/tmp/carnot-7806/frozen_scope.json",
        "checks": receipts,
        "required_checks_passed": passed,
        "cold_replay": next(
            (r for r in receipts if r["name"] == "fresh_process_cold_replay"), None
        ),
        "terminal_readers": [
            r
            for r in receipts
            if r["name"] in {"adversarial_verify", "verdict_row_consistency_strict"}
        ],
        "repository_health": (
            json.loads((root / RAW / "repository_health.json").read_text())
            if (root / RAW / "repository_health.json").is_file()
            else {"status": "not_recorded"}
        ),
    }
    if not passed:
        audit["honest_verdict"] = "complete_disqualified_required_checks"
        audit["verdict_class"] = "disqualified"
        audit["acceptance_gate_results"] = {key: 0 for key in audit["acceptance_gate_results"]}
        audit["gate_check_summary"].extend(
            {
                "upstream_id": "Exp7806:validation",
                "artifact_path": r["log_path"],
                "artifact_hash": r["log_sha256"],
                "field": r["name"],
                "operator": "==",
                "expected": 0,
                "observed": r["exit_code"],
                "passed": False,
            }
            for r in receipts
            if not r["passed"]
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
    progress(started, "publish", "before_atomic_write", len(audit["board_rows"]))
    atomic_json(root / OUTPUT, audit)
    progress(started, "publish", "after_atomic_write", len(audit["board_rows"]))
    return 0

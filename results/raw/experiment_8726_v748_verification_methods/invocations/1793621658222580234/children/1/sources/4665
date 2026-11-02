"""Read authenticated board history beside one current service-cost producer."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7820_v679_hardware_evidence import run_child

ROOT = Path(__file__).resolve().parents[3]
RAW = "results/raw/experiment_7834_v680_hardware_evidence"
MANIFEST = f"{RAW}/validation_command_manifest.json"
OUTPUT = "results/experiment_7834_v680_hardware_evidence.json"
PRIOR = "results/experiment_7820_v679_hardware_evidence.json"
SERVICE = "results/experiment_7833_v680_service_cost.json"
PRE_GATE = "results/experiment_7833_service_cost.json"
COMMAND_NAMES = (
    "focused_pytest",
    "coverage_module",
    "task_entrypoint_e2e",
    "coverage_combine",
    "coverage_report",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "repository_health",
    "fresh_process_cold_replay",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Expose real elapsed time at each work boundary for a waiting operator."""
    print(
        f"[exp7834] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(
    path: str, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep both sides of a failed service gate in the terminal artifact."""
    return {
        "upstream_id": "Exp7833",
        "artifact_path": path,
        "artifact_hash": digest,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _finite_time(value: Any) -> bool:
    """Reject booleans, NaN and infinities before dividing service times."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _service(root: Path) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]], str]:
    """The science producer has its own exact path; a pre-gate receipt cannot replace it."""
    path = root / SERVICE
    digest = sha256_file(path) if path.is_file() else None
    data = json.loads(path.read_text()) if digest else None
    checks = [
        _check(
            SERVICE,
            digest,
            "service_evidence_ready_score",
            1,
            data.get("service_evidence_ready_score") if data else None,
        )
    ]
    if data is not None:
        checks.extend(
            [
                _check(SERVICE, digest, "experiment_id", 7833, data.get("experiment_id")),
                _check(SERVICE, digest, "schema", "carnot.experiment_7833.v1", data.get("schema")),
                _check(
                    SERVICE, digest, "flagged_adversarial", False, data.get("flagged_adversarial")
                ),
                _check(
                    SERVICE,
                    digest,
                    "verdict_class.accepted",
                    True,
                    data.get("verdict_class") in {"positive", "null", "circular_positive"},
                ),
            ]
        )
        times = data.get("stage_times_ms")
        host = times.get("host_stage") if isinstance(times, dict) else None
        whole = times.get("whole_service") if isinstance(times, dict) else None
        valid = _finite_time(host) and _finite_time(whole) and 0 <= host <= whole and whole > 0
        checks.append(_check(SERVICE, digest, "stage_times_ms.valid", True, valid))
    status = (
        "missing"
        if data is None
        else "qualified"
        if all(c["passed"] for c in checks)
        else "disqualified"
    )
    return data, digest, [c for c in checks if not c["passed"]], status


def build_audit(root: Path, run_date: str, manifest_path: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7834-SERVICE: rehash board receipts and reduce service independently."""
    started = time.monotonic()
    progress(started, "preconditions", "start", 0)
    old_path = root / PRIOR
    if not old_path.is_file():
        raise ValueError("missing_exp7820_board_inventory")
    old = json.loads(old_path.read_text())
    if old.get("schema") != "carnot.experiment_7820.v1" or old.get("experiment_id") != 7820:
        raise ValueError("invalid_exp7820_schema")
    audit = deepcopy(old)
    boards = audit["board_rows"]
    if {row["board"] for row in boards} != {"KV260", "PolarFire", "GateMate"}:
        raise ValueError("board_inventory_mismatch")
    sources = {}
    for row in boards:
        path = root / row["source_path"]
        digest = sha256_file(path) if path.is_file() else None
        if digest != row["source_hash"] or digest != row["expected_hash"]:
            raise ValueError(f"board_hash_mismatch:{row['board']}")
        row["evidence_age_days"] = (
            datetime.strptime(run_date, "%Y%m%d")
            - datetime.strptime(row["last_authenticated_evidence_date"], "%Y%m%d")
        ).days
        sources[row["source_path"]] = {
            "path": row["source_path"],
            "sha256": digest,
            "expected_sha256": row["expected_hash"],
            "date": row["last_authenticated_evidence_date"],
            "role": "dated_board_receipt",
            "eligible": True,
        }
    if next(row for row in boards if row["board"] == "KV260")["k_max"] != 5:
        raise ValueError("kv260_scope_mismatch")
    if next(row for row in boards if row["board"] == "PolarFire")["processor_class"] != "linux_cpu":
        raise ValueError("polarfire_scope_mismatch")
    if next(row for row in boards if row["board"] == "GateMate")["blocker"] != "0xffffffff":
        raise ValueError("gatemate_scope_mismatch")
    progress(started, "preconditions", "board_bytes_verified", len(boards))
    service, service_hash, failures, status = _service(root)
    sources[PRIOR] = {
        "path": PRIOR,
        "sha256": sha256_file(old_path),
        "date": old.get("run_date"),
        "role": "authenticated_board_inventory",
        "eligible": True,
    }
    sources[SERVICE] = {
        "path": SERVICE,
        "sha256": service_hash,
        "date": service.get("run_date") if service else None,
        "role": "current_science_producer",
        "eligible": status == "qualified",
    }
    times = service["stage_times_ms"] if status == "qualified" else {}
    whole = times.get("whole_service")
    host = times.get("host_stage")
    fraction = host / whole if status == "qualified" else None
    bound = 1 / (1 - fraction) if fraction is not None and fraction < 1 else None
    for row in boards:
        row["service_fraction"] = fraction
    audit["rows"] = deepcopy(boards)
    audit["rows_checksum"] = canonical_hash(audit["rows"])
    audit.update(
        {
            "schema": "carnot.experiment_7834.v1",
            "experiment_id": 7834,
            "milestone": "2026.09.680",
            "run_date": run_date,
            "honest_verdict": "complete_null_hardware_accounting_only"
            if status == "qualified"
            else "complete_blocked_missing_service_evidence",
            "verdict_class": "null" if status == "qualified" else "blocked",
            "flagged_adversarial": False,
            "gate_check_summary": failures,
            "service_source_status": status,
            "service_measured": status == "qualified",
            "host_stage_fraction": fraction,
            "acceleration_bound": bound,
            "hardware_advantage_claimed": False,
            "hardware_continuity_accounted": True,
            "hardware_operations_issued": [],
            "source_artifact_hashes": sources,
            "raw_rows_path": None,
            "raw_rows_sha256": None,
            "validation_command_manifest_path": str(manifest_path),
            "validation_command_manifest_sha256": sha256_file(manifest_path),
            "observed_child_commands": [],
            "validation_receipts": {"checks": []},
            "repository_health": {"status": "unstarted"},
            "random_seed": None,
            "MODEL_SPECS": [],
            "model_specs": [],
            "model_invocation_counts": {"calls": 0, "tokens": 0, "loaded_file_hashes": []},
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "inference_substrate_class": "aggregation",
            "planned_inference_substrate_class": "aggregation",
            "actual_inference_substrate_class": "aggregation",
            "verifier_is_oracle": False,
            "sample_size_budget": {
                "intended": 3,
                "eligible": 2,
                "started": 0,
                "completed": 0,
                "excluded": 1,
                "censored": 0,
                "effective_independent_n": 2,
            },
            "acceptance_gate_results": {
                "validity": True,
                "readiness": 0,
                "probability_quality": None,
                "decision_benefit": None,
                "retention": None,
                "efficiency": None,
            },
        }
    )
    audit["preconditions_checked"] = {
        "declared_science_producer": SERVICE,
        "conductor_pre_gate_receipt": {
            "path": PRE_GATE,
            "sha256": sha256_file(root / PRE_GATE) if (root / PRE_GATE).is_file() else None,
            "is_science_producer": False,
        },
        "board_operations_issued": [],
        "current_board_reachability": "not_probed",
        "required_resources": {"cpu": True, "board_operations": False},
        "board_receipts_verified": 3,
        "service_gate_operands_checked": [
            "service_evidence_ready_score",
            "experiment_id",
            "schema",
            "flagged_adversarial",
            "verdict_class.accepted",
            "stage_times_ms.valid",
        ],
    }
    audit["stage_opportunity_map"] = {
        "pipelined_p_computer_memory_bandwidth": {
            "mapped_stage": "host_stage" if fraction is not None else None,
            "measured_ms": host if status == "qualified" else None,
            "measured_bytes": times.get("host_bytes") if status == "qualified" else None,
            "bandwidth_bytes_per_s": times.get("host_bandwidth_bytes_per_s")
            if status == "qualified"
            else None,
            "transfer_boundary": "host memory to candidate accelerator; not measured",
            "correctness_criterion": "same accepted output and probability quality on paired workload",
            "required_access": "profiled CPU memory counters and board-local transfer transcript",
        },
        "z1t_digital_probabilistic_split": {
            "digital_stage": "host dispatch and preprocessing" if fraction is not None else None,
            "probabilistic_stage": None,
            "probabilistic_stage_ms": None,
            "measured_input_bytes": times.get("sampler_input_bytes")
            if status == "qualified"
            else None,
            "measured_output_bytes": times.get("sampler_output_bytes")
            if status == "qualified"
            else None,
            "transfer_boundary": "digital host to probabilistic sampler and readout; unmeasured",
            "correctness_criterion": "paired sample distribution and end-to-end decision parity",
            "required_access": "authenticated TSU or qualified FPGA sampler run with device timing",
        },
        "sampler_integration": "deferred_no_measured_sampler_work",
        "preparation_ms": times.get("preparation") if status == "qualified" else None,
        "transfer_ms": times.get("transfer") if status == "qualified" else None,
        "readout_ms": times.get("readout") if status == "qualified" else None,
    }
    audit["reopen_condition"] = {
        "GateMate": "Dated changed cable, port or power evidence, then valid GM1Ax IDCODE",
        "KV260": "SSH and board-local workload receipts with k<=5, transport and complete service timing",
        "PolarFire": "Authenticated device-side FPGA workload and timing; Linux CPU dispatch is insufficient",
        "NPU": "Working toolchain plus authenticated device workload run",
        "TSU": "Authenticated device access and paired sampler service measurement",
        "sampler": "Measured sampler work, input/output bytes, distribution parity and complete service timing",
    }
    audit["acquisition_recommendation"] = {
        "decision": "defer",
        "basis": "No measured board whole-service advantage",
        "vendor_z1t_estimate_is_local_measurement": False,
    }
    audit["claim_scope"].update(
        {
            "hardware_advantage": "unmeasured",
            "acquisition": "defer",
            "source_family_exposure": "640 exposed development families",
        }
    )
    audit["prior_failure_metadata"] = {
        "experiment_id": 7820,
        "honest_verdict": old["honest_verdict"],
        "verdict_class": old["verdict_class"],
        "retire_if_same_verdict": True,
        "same_verdict_retired": audit["honest_verdict"] == old["honest_verdict"],
    }
    audit["reproducibility_checksum"] = canonical_hash(
        {
            "code": sha256_file(Path(__file__)),
            "cli": sha256_file(
                root / "scripts/experiments/experiment_7834_v680_hardware_evidence.py"
            )
            if (root / "scripts/experiments/experiment_7834_v680_hardware_evidence.py").is_file()
            else None,
            "manifest": audit["validation_command_manifest_sha256"],
            "inputs": {key: value["sha256"] for key, value in sources.items()},
            "seed": None,
        }
    )
    audit["field_principles"] = {
        key: "Preserve the stated evidence boundary and exact provenance."
        for key in audit
        if key != "field_principles"
    }
    audit["field_principles"]["field_principles"] = "Explain each field and gate in the artifact."
    ended = time.monotonic()
    audit["duration_s"] = ended - started
    audit["phase_spans"] = [
        {
            "phase": "preconditions_and_reduction",
            "start_monotonic_s": started,
            "end_monotonic_s": ended,
            "duration_s": ended - started,
            "completed_units": len(boards),
        }
    ]
    progress(started, "reduction", "complete", len(boards))
    return audit


def _manifest(path: Path) -> dict[str, Any]:
    """Keep an alternate attempt's paths but require every frozen command operand."""
    value = json.loads(path.read_text())
    frozen = json.loads((ROOT / MANIFEST).read_text())
    if [c.get("name") for c in value["commands"]] != list(COMMAND_NAMES):
        raise ValueError("undeclared child name")
    if len(value["commands"]) != len(frozen["commands"]):
        raise ValueError("undeclared child count")
    for actual, expected in zip(value["commands"], frozen["commands"], strict=True):
        normalized = deepcopy(actual)
        old_root = value["run_root"]
        new_root = frozen["run_root"]
        normalized["argv"] = [arg.replace(old_root, new_root) for arg in normalized["argv"]]
        normalized["private_root"] = normalized["private_root"].replace(old_root, new_root)
        if normalized != expected:
            raise ValueError("undeclared child argv or classification")
    return value


def dispatch(
    root: Path,
    manifest_path: Path,
    durable: Path,
    executor: Callable[[Path, dict[str, Any], Path], dict[str, Any]] = run_child,
    before_child: Callable[[int, list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """SCENARIO-REPORT-7834-DISPATCH: pass the frozen commands to one executor."""
    receipts: list[dict[str, Any]] = []
    for index, spec in enumerate(_manifest(manifest_path)["commands"]):
        if before_child:
            before_child(index, receipts)
        result = executor(root, spec, durable)
        if (result["name"], result["command_argv"], result["classification"]) != (
            spec["name"],
            spec["argv"],
            spec["classification"],
        ):
            raise ValueError("undeclared child receipt")
        receipts.append(result)
    return receipts


def cold_reduce(candidate: Path, root: Path, manifest_path: Path) -> dict[str, Any]:
    """SCENARIO-REPORT-7834-CUSTODY: recompute inputs, rows, and closed log hashes."""
    _manifest(manifest_path)
    artifact = json.loads(candidate.read_text())
    fresh = build_audit(root, artifact["run_date"], manifest_path)
    for field in (
        "source_artifact_hashes",
        "board_rows",
        "rows",
        "rows_checksum",
        "service_source_status",
        "host_stage_fraction",
        "acceleration_bound",
        "stage_opportunity_map",
        "reopen_condition",
        "reproducibility_checksum",
        "validation_command_manifest_sha256",
    ):
        if artifact[field] != fresh[field]:
            raise ValueError(f"cold_replay_mismatch:{field}")
    source_checks = [
        check
        for check in artifact["gate_check_summary"]
        if check["upstream_id"] != "Exp7834:validation"
    ]
    if source_checks != fresh["gate_check_summary"]:
        raise ValueError("cold_replay_mismatch:gate_check_summary")
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
    """Write a terminal read-only audit only after the declared validation ends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260928")
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
        if args.manifest is None:
            manifest_path = Path(
                json.loads(args.cold_replay.read_text())["validation_command_manifest_path"]
            )
        print(
            json.dumps(cold_reduce(args.cold_replay, root, manifest_path), sort_keys=True),
            flush=True,
        )
        progress(started, "cold_replay", "after_reduction", 3)
        return 0
    audit = build_audit(root, args.date, manifest_path)
    if args.prepare:
        atomic_json(args.prepare, audit)
        progress(started, "prepare", "complete", 3)
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
            raise ValueError("immutable raw rows changed")
    else:
        raw_path.write_bytes(raw_bytes)
    audit["raw_rows_path"] = str(raw_path)
    audit["raw_rows_sha256"] = sha256_file(raw_path)
    candidate = Path(manifest["candidate_path"])
    candidate.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(candidate, audit)
    progress(started, "validation", "before_subprocesses", 0)
    validation_started = time.monotonic()

    def before_child(index: int, receipts: list[dict[str, Any]]) -> None:
        audit["validation_receipts"]["checks"] = list(receipts)
        atomic_json(candidate, audit)
        progress(started, "validation", "command_boundary", index)

    receipts = dispatch(root, manifest_path, raw / "validation", executor, before_child)
    progress(started, "validation", "after_subprocesses", len(receipts))
    audit["observed_child_commands"] = manifest["commands"]
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
        "log_path": health["log_path"],
        "log_sha256": health["log_sha256"],
        "timed_out": health.get("timed_out", False),
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
                "upstream_id": "Exp7834:validation",
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

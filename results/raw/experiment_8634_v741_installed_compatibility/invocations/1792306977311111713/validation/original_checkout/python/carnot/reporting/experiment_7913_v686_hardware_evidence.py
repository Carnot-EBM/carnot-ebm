"""Authenticate dated board custody and validate with private scratch ownership.

Spec ref: REQ-REPORT-7913-V686. Historical evidence remains useful within its
original scope even when a later runner failed. No device command is issued.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from coverage import CoverageData

from carnot.reporting import validation_7913 as plan
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.experiment_7862_v682_hardware_evidence import _check
from carnot.reporting.experiment_7876_v683_hardware_evidence import _load
from carnot.reporting.experiment_7901_v685_hardware_evidence import read_evidence as board_custody

ROOT = plan.ROOT
PRIOR = "results/experiment_7901_v685_hardware_evidence.json"
SERVICE = "results/experiment_7912_v686_service_cost.json"
FAILURES_7901 = (
    "test_experiment_7901_v685_hardware_evidence.py::test_scenario_report_7901_cli_private",
    "test_experiment_7889_v684_hardware_evidence.py::test_scenario_report_7889_cli_private_paths",
    "test_experiment_7889_v684_hardware_evidence.py::test_scenario_report_7889_manifest_and_child",
    "test_experiment_7876_v683_hardware_evidence.py::test_scenario_report_7876_cli_success_failure_and_replay",
    "test_experiment_7862_v682_hardware_evidence.py::test_manifest_and_owned_child",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Flushed boundaries distinguish ongoing work from a stalled child."""
    print(
        f"[exp7913] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7913-CUSTODY: preserve dated scope and failure authority."""
    started = time.monotonic()
    result = board_custody(root, run_date)
    prior, digest = _load(root, PRIOR)
    checks = [_check(PRIOR, digest, "exists", True, digest is not None)]
    if digest is not None:
        checks.append(_check(PRIOR, digest, "schema_json", "dict", type(prior).__name__))
    if prior is not None:
        checks.extend(
            _check(PRIOR, digest, key, expected, prior.get(key))
            for key, expected in (
                ("experiment_id", 7901),
                ("task_id", "exp7901-hardware-evidence"),
                ("milestone", "2026.09.685"),
                ("verdict_class", "disqualified"),
            )
        )
        for receipt in prior.get("validation_receipts", {}).get("checks", []):
            log = Path(receipt["log_path"])
            checks.append(
                _check(
                    str(log),
                    digest,
                    "log_sha256",
                    receipt["log_sha256"],
                    sha256_file(log) if log.is_file() else None,
                )
            )
    failed = [
        deepcopy(r)
        for r in (prior or {}).get("validation_receipts", {}).get("checks", [])
        if r.get("classification") == "required" and r.get("passed") is False
    ]
    result["historical_required_failures"].extend(failed)
    result["historical_7901_affected_failures"] = list(FAILURES_7901)
    result["historical_7901_coverage"] = {"covered": 141, "statements": 218}
    result["source_artifact_hashes"][PRIOR] = {
        "path": PRIOR,
        "sha256": digest,
        "date": (prior or {}).get("run_date"),
        "role": "immutable_failed_validation_authority",
        "exposure_status": "historical_read_only",
    }
    result["preconditions_checked"].extend(checks)
    result["gate_check_summary"].extend(x for x in checks if not x["passed"])
    service, service_hash = _load(root, SERVICE)
    operands = [_check(SERVICE, service_hash, "exists", True, service_hash is not None)]
    rows = (service or {}).get("rows")
    if service_hash is not None:
        operands.append(
            _check(SERVICE, service_hash, "schema_json", "dict", type(service).__name__)
        )
    if service is not None:
        operands.extend(
            _check(SERVICE, service_hash, key, expected, service.get(key))
            for key, expected in (
                ("experiment_id", 7912),
                ("task_id", "exp7912-service-cost"),
                ("run_date", run_date),
                ("flagged_adversarial", False),
                ("service_measurement_ready_score", 1),
            )
        )
        operands.extend(
            [
                _check(
                    SERVICE,
                    service_hash,
                    "verdict_class.eligible",
                    True,
                    service.get("verdict_class") in {"positive", "circular_positive", "null"},
                ),
                _check(
                    SERVICE,
                    service_hash,
                    "validation_receipts.required_checks_passed",
                    True,
                    service.get("validation_receipts", {}).get("required_checks_passed"),
                ),
                _check(
                    SERVICE,
                    service_hash,
                    "rows.complete_workload",
                    True,
                    isinstance(rows, list)
                    and bool(rows)
                    and all(isinstance(row, dict) and row.get("operation") for row in rows),
                ),
            ]
        )
    attached = all(x["passed"] for x in operands)
    result["source_artifact_hashes"][SERVICE] = {
        "path": SERVICE,
        "sha256": service_hash,
        "date": (service or {}).get("run_date"),
        "role": "optional_current_service",
        "exposure_status": "exposed_development" if attached else "unavailable",
        "eligible": attached,
    }
    result["workload_attachment_operands"] = operands
    result["workload_attachment_available"] = attached
    result["workload_feasibility_rows"] = (
        [
            {
                "board": board["board"],
                "source_row": deepcopy(row),
                "source_row_hash": canonical_hash(row),
                "source_artifact_hash": service_hash,
                "transfer_bytes": row.get("transfer_bytes"),
                "measured_host_work": row.get("duration_s"),
                "placement_constraint": board["scope"],
                "hardware_execution_measured": False,
                "speedup": None,
            }
            for row in rows
            if attached
            for board in result["board_rows"]
        ]
        if attached
        else []
    )
    for row in result["board_rows"]:
        row.update(
            {
                "custody_date": run_date,
                "unit": "board_obligation",
                "intended": 1,
                "eligible": int(row["board"] != "GateMate"),
                "failed": False,
                "independent": 0,
            }
        )
    result["rows"] = result["board_rows"]
    result["rows_checksum"] = canonical_hash(result["rows"])
    ready = not result["gate_check_summary"]
    result.update(
        {
            "experiment_id": 7913,
            "task_id": "exp7913-hardware-evidence",
            "milestone": "2026.09.686",
            "honest_verdict": "complete_null_historical_board_scope"
            if ready
            else "complete_blocked_board_source_custody",
            "verdict_class": "null" if ready else "blocked",
            "hardware_evidence_ready_score": int(ready),
            "planned_inference_substrate_class": "no_model_load",
        }
    )
    result["sample_size_budget"].update(
        {"unit": "board_obligation", "failed": 0, "eligible": 2 if ready else 0}
    )
    result["acceptance_gate_results"]["validity"] = ready
    result["claim_scope"]["workload"] = "exposed_development" if attached else "unavailable"
    result["reproducibility_checksum"] = canonical_hash(
        {
            "base": result["reproducibility_checksum"],
            "sources": result["source_artifact_hashes"],
            "rows": result["rows"],
            "code": sha256_file(Path(__file__)),
        }
    )
    result["field_principles"].update(
        {
            key: f"Bind {key.replace('_', ' ')} to current custody without expanding historical device scope."
            for key in result
        }
    )
    result["field_principles"]["acceptance_gate_results"] = {
        "validity": "Authenticate historical bytes and their limited scope.",
        "readiness": "All owned checks must pass; optional service cannot open this gate.",
        "probability_quality": "No sampling quality was measured.",
        "decision_benefit": "No decision benefit was measured.",
        "retention": "No retained learning was measured.",
        "efficiency": "No board efficiency was measured.",
    }
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"]["evidence_read_s"] = result["duration_s"]
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7913-PRIVATE: recompute each primitive denominator."""
    fresh = read_evidence(root, candidate["run_date"])
    for field in (
        "rows",
        "board_rows",
        "rows_checksum",
        "terminal_receipt_hashes",
        "workload_feasibility_rows",
        "workload_attachment_operands",
        "source_artifact_hashes",
        "sample_size_budget",
    ):
        if fresh[field] != candidate.get(field):
            raise ValueError(f"claims_changed:{field}")
    gates = candidate.get("gate_check_summary")
    if not isinstance(gates, list) or any(not isinstance(x, dict) for x in gates):
        raise ValueError("claims_changed:gate_operands")
    if fresh["gate_check_summary"] != [
        x for x in gates if x.get("upstream_id") != "owned_validation"
    ]:
        raise ValueError("claims_changed:gate_operands")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}


def check_coverage_shards(paths: list[Path]) -> None:
    """SCENARIO-REPORT-7913-PRIVATE: an existing foreign file is still empty."""
    if not paths:
        raise ValueError("empty_coverage_shards")
    owned = {str((ROOT / name).resolve()) for name in plan.MEASURED}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"empty_coverage_shard:{path}")
        data = CoverageData(basename=str(path))
        data.read()
        if not any(
            data.lines(name) for name in data.measured_files() if str(Path(name).resolve()) in owned
        ):
            raise ValueError(f"empty_coverage_shard:{path}")


def seal(source: Path, directory: Path) -> Path:
    """Copy closed evidence using an atomic rename on the destination filesystem."""
    digest = sha256_file(source).removeprefix("sha256:")
    target = directory / f"{digest}{source.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    return target


def run_child(
    spec: dict[str, Any], private: Path, durable: Path, started: float, units: int
) -> dict[str, Any]:
    """Run with active inherited guards and seal the log only after child exit."""
    progress(started, spec["name"], "before_subprocess", units)
    receipt = run_commands(
        ROOT,
        [
            CommandSpec(
                spec["name"], tuple(spec["argv"]), spec["classification"], spec["deadline_s"]
            )
        ],
        log_dir=private / "open-logs",
        extra_env={"TMPDIR": str(private), "COVERAGE_FILE": str(private / "ambient.coverage")},
        heartbeat_s=30,
    )[0]
    source = Path(receipt["log_path"])
    sealed = seal(source, durable / "logs" / spec["name"])
    receipt.update(spec)
    receipt["log_path"] = str(sealed)
    receipt["passed"] = (
        receipt["exit_code"] == spec["expected_exit"]
        and not receipt["timed_out"]
        and (spec["required_reason"] is None or spec["required_reason"] in sealed.read_text())
    )
    progress(started, spec["name"], "after_subprocess", units + 1)
    return receipt


def disqualify(result: dict[str, Any], reason: str) -> None:
    """Owned failures close readiness while keeping historical measurements intact."""
    result.update(
        {
            "honest_verdict": f"complete_disqualified_{reason}",
            "verdict_class": "disqualified",
            "hardware_evidence_ready_score": 0,
        }
    )
    result["acceptance_gate_results"]["readiness"] = 0


def qualify(root: Path, run_date: str, output: Path, raw_root: Path) -> int:
    """SCENARIO-REPORT-7913-TERMINAL: validate and publish the same final bytes."""
    started = time.monotonic()
    progress(started, "qualification", "start", 0)
    if root.resolve() != ROOT:
        raise ValueError("full validation requires the worktree root")
    result = read_evidence(root, run_date)
    progress(started, "preconditions", "resolved_paths_hashes_roles_operands", len(result["rows"]))
    closure = {
        name: sha256_file(ROOT / name)
        for name in (*plan.MEASURED, plan.TEST, *plan.CONSUMERS, *plan.LIBRARIES)
    }
    key = canonical_hash(
        {"closure": closure, "sources": result["source_artifact_hashes"], "date": run_date}
    ).removeprefix("sha256:")
    durable = raw_root / key
    with tempfile.TemporaryDirectory(prefix="carnot-7913-", dir="/tmp") as folder:
        private = Path(folder)
        commands = plan.manifest(private, run_date)
        terminal = plan.terminal_manifest(private)
        frozen = {
            "task_id": result["task_id"],
            "commands": commands,
            "terminal_validators": terminal,
            "dependency_closure": closure,
            "source_closure": result["source_artifact_hashes"],
            "measured_modules": list(plan.MEASURED),
            "input_configuration_hash": key,
            "child_environment": {
                "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}",
                "TMPDIR": str(private),
                "COVERAGE_FILE": str(private / "ambient.coverage"),
                "result_guards": "preserved; pytest children install repository guards",
            },
            "e2e_applicability": {
                "E2E-016": "dated fixture and cold replay; no device or model",
                "other_e2es": "other producers or physical execution",
            },
        }
        manifest_path = durable / "validation_command_manifest.json"
        atomic_json(manifest_path, frozen)
        result["validation_command_manifest_path"] = str(manifest_path)
        result["validation_command_manifest_sha256"] = sha256_file(manifest_path)
        result["resolved_imports"] = {
            "carnot.reporting.experiment_7913_v686_hardware_evidence": str(ROOT / plan.MODULE),
            "carnot.reporting.validation_7913": str(ROOT / plan.PLAN),
            "scripts.experiments.experiment_7913_v686_hardware_evidence": str(ROOT / plan.SCRIPT),
        }
        receipts: list[dict[str, Any]] = []
        for spec in commands:
            if spec["name"] == "negative_replay":
                changed = json.loads((private / "success.json").read_text())
                changed["rows"] = []
                atomic_json(private / "changed.json", changed)
            receipt = run_child(spec, private, durable, started, len(receipts))
            receipts.append(receipt)
            atomic_json(
                durable / "completed_units.json",
                {"input_configuration_hash": key, "checks": receipts},
            )
        required = [r for r in receipts if r["classification"] == "required"]
        passed = all(r["passed"] for r in required)
        result["validation_receipts"] = {"checks": receipts, "required_checks_passed": passed}
        result["validation_evidence_files"] = []
        for evidence in sorted((*private.glob("*.coverage"), *private.glob("*.json"))):
            sealed = seal(evidence, durable / "sealed-evidence")
            result["validation_evidence_files"].append(
                {"path": str(sealed), "sha256": sha256_file(sealed), "role": evidence.name}
            )
        result["observed_child_commands"] = [
            {
                "name": r["name"],
                "argv": r["argv"],
                "classification": r["classification"],
                "expected_exit": r["expected_exit"],
                "actual_exit": r["exit_code"],
            }
            for r in receipts
        ]
        result["repository_health"] = {
            "status": "healthy" if receipts[-1]["passed"] else "degraded_open",
            "affects_required_checks": False,
            "full_pytest": receipts[-1],
            "historical": (_load(ROOT, PRIOR)[0] or {}).get("repository_health"),
            "global_spec_backlog": "retained separately; no unscoped spec scan",
        }
        result["gate_check_summary"].extend(
            {
                **_check(
                    r["log_path"],
                    r["log_sha256"],
                    f"{r['name']}.expected_exit_and_reason",
                    {
                        "exit": r["expected_exit"],
                        "reason": r["required_reason"],
                        "timed_out": False,
                    },
                    {
                        "exit": r["exit_code"],
                        "reason_found": r["passed"],
                        "timed_out": r["timed_out"],
                    },
                ),
                "upstream_id": "owned_validation",
            }
            for r in required
            if not r["passed"]
        )
        if not passed:
            disqualify(result, "required_checks")
        result["acceptance_gate_results"]["readiness"] = int(
            passed and result["hardware_evidence_ready_score"] == 1
        )
        result["retire_if_same_verdict"] = {
            "prior_experiment_id": 7901,
            "identical_failure": False,
            "reason": "scratch ownership and actual current CLI coverage changed; compare named failed operands",
        }
        result["duration_s"] = time.monotonic() - started
        result["phase_spans"]["validation_s"] = (
            result["duration_s"] - result["phase_spans"]["evidence_read_s"]
        )
        result["reproducibility_checksum"] = canonical_hash(
            {
                "base": result["reproducibility_checksum"],
                "configuration": key,
                "manifest": result["validation_command_manifest_sha256"],
            }
        )
        result["terminal_validation_sidecar_path"] = str(
            durable / "terminal_validation_reports.json"
        )
        result["field_principles"].update(
            {
                key: f"Authenticate {key.replace('_', ' ')} as current owned work; retain old failures separately."
                for key in result
                if key not in result["field_principles"]
            }
        )
        candidate = private / "terminal-candidate.json"
        reports: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        for _attempt in range(2):
            result["flagged_adversarial"] = False
            atomic_json(candidate, result)
            reports = [
                run_child(spec, private, durable, started, len(receipts) + index)
                for index, spec in enumerate(terminal)
            ]
            attempts.append({"candidate_sha256": sha256_file(candidate), "reports": reports})
            try:
                flags = json.loads(Path(reports[0]["log_path"]).read_text())["flagged_count"]
            except (ValueError, KeyError, TypeError):
                flags = 1
            result["flagged_adversarial"] = bool(flags)
            if not flags and all(r["passed"] for r in reports):
                break
            disqualify(result, "terminal_validation")
        else:
            raise ValueError("terminal_validation_failed")
        sidecar = {
            "candidate_sha256": sha256_file(candidate),
            "reports": reports,
            "attempts": attempts,
        }
        atomic_json(Path(result["terminal_validation_sidecar_path"]), sidecar)
        seal(candidate, durable / "candidates")
        temporary = output.with_name(f".{output.name}.checked.tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, temporary)
        if sha256_file(temporary) != sidecar["candidate_sha256"]:
            temporary.unlink()
            raise ValueError("publication_hash_changed")
        temporary.replace(output)
    progress(started, "final", "published_checked_bytes", len(receipts) + len(reports))
    return 0

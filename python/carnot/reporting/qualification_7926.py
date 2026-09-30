"""Compose existing child supervision and sealing for current board custody.

Spec ref: REQ-REPORT-7926-V687. Publishing must preserve the exact bytes inspected
by cold reduction and the two terminal validators.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7913_v686_hardware_evidence as old
from carnot.reporting import experiment_7926_v687_hardware_evidence as q
from carnot.reporting import validation_7926 as plan
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

run_child = old.run_child


def qualify(root: Path, run_date: str, output: Path, raw_root: Path) -> int:
    """Freeze dependencies, run owned checks and publish only terminal-checked bytes."""
    started = time.monotonic()
    if root.resolve() != plan.ROOT:
        raise ValueError("full validation requires the worktree root")
    result = q.read_evidence(root, run_date)
    old.progress(
        started, "preconditions", "resolved_paths_hashes_roles_operands", len(result["rows"])
    )
    closure = {
        name: sha256_file(root / name) for name in (*plan.MEASURED, *plan.TESTS, *plan.LIBRARIES)
    }
    key = canonical_hash(
        {"closure": closure, "sources": result["source_artifact_hashes"], "date": run_date}
    ).removeprefix("sha256:")
    durable = raw_root / key
    with tempfile.TemporaryDirectory(prefix="carnot-7926-", dir="/tmp") as folder:
        private = Path(folder)
        commands, terminal = plan.manifest(private, run_date), plan.terminal_manifest(private)
        frozen = {
            "task_id": result["task_id"],
            "commands": commands,
            "terminal_validators": terminal,
            "dependency_closure": closure,
            "source_closure": result["source_artifact_hashes"],
            "measured_modules": list(plan.MEASURED),
            "input_configuration_hash": key,
            "historical_fixture_date": "20260929",
            "current_execution_date": run_date,
            "child_environment": {
                "PYTHONPATH": "python:.",
                "TMPDIR": str(private),
                "COVERAGE_FILE": str(private / "ambient.coverage"),
                "result_guards": "preserved",
            },
        }
        manifest = durable / "validation_command_manifest.json"
        atomic_json(manifest, frozen)
        result["validation_command_manifest_path"] = str(manifest)
        result["validation_command_manifest_sha256"] = sha256_file(manifest)
        result["resolved_imports"] = {name: str(root / name) for name in plan.MEASURED}
        receipts = []
        for spec in commands:
            if spec["name"] == "negative_replay":
                changed = json.loads((private / "success.json").read_text())
                changed["rows"] = []
                atomic_json(private / "changed.json", changed)
            receipts.append(run_child(spec, private, durable, started, len(receipts)))
            atomic_json(
                durable / "completed_units.json",
                {"input_configuration_hash": key, "checks": receipts},
            )
        required = [r for r in receipts if r["classification"] == "required"]
        passed = all(r["passed"] for r in required)
        result["validation_receipts"] = {"checks": receipts, "required_checks_passed": passed}
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
            "historical": result["repository_health"],
            "global_spec_backlog": "separate; no unscoped scan",
        }
        result["validation_evidence_files"] = []
        for evidence in sorted((*private.glob("*.coverage"), *private.glob("*.json"))):
            sealed = old.seal(evidence, durable / "sealed-evidence")
            result["validation_evidence_files"].append(
                {"path": str(sealed), "sha256": sha256_file(sealed), "role": evidence.name}
            )
        coverage = (
            json.loads((private / "coverage.json").read_text())
            if (private / "coverage.json").exists()
            else {"files": {}}
        )
        result["coverage_statement_counts"] = {
            name: {
                "covered": entry["summary"]["covered_lines"],
                "statements": entry["summary"]["num_statements"],
            }
            for name, entry in coverage["files"].items()
        }
        result["gate_check_summary"].extend(
            {
                "upstream_id": "owned_validation",
                "artifact_path": r["log_path"],
                "artifact_hash": r["log_sha256"],
                "artifact_field": r["name"],
                "op": "eq",
                "expected": {
                    "exit": r["expected_exit"],
                    "reason": r["required_reason"],
                    "timed_out": False,
                },
                "observed": {
                    "exit": r["exit_code"],
                    "passed": r["passed"],
                    "timed_out": r["timed_out"],
                },
                "passed": False,
            }
            for r in required
            if not r["passed"]
        )
        if not passed:
            old.disqualify(result, "required_checks")
        result["acceptance_gate_results"]["readiness"] = int(
            passed and result["hardware_evidence_ready_score"] == 1
        )
        atomic_json(
            private / "rows.json",
            {"rows": result["rows"], "workload_rows": result["workload_feasibility_rows"]},
        )
        result["raw_rows_path"] = str(old.seal(private / "rows.json", durable / "rows"))
        result["raw_rows_sha256"] = sha256_file(Path(result["raw_rows_path"]))
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
                name: f"Authenticate {name.replace('_', ' ')} as owned evidence."
                for name in result
                if name not in result["field_principles"]
            }
        )
        candidate = private / "terminal-candidate.json"
        attempts = []
        for _ in range(2):
            result["flagged_adversarial"] = False
            atomic_json(candidate, result)
            reports = [
                run_child(spec, private, durable, started, len(receipts) + i)
                for i, spec in enumerate(terminal)
            ]
            attempts.append({"candidate_sha256": sha256_file(candidate), "reports": reports})
            try:
                flags = json.loads(Path(reports[0]["log_path"]).read_text())["flagged_count"]
            except (ValueError, KeyError, TypeError):
                flags = 1
            if not flags and all(r["passed"] for r in reports):
                break
            result["flagged_adversarial"] = bool(flags)
            old.disqualify(result, "terminal_validation")
        else:
            raise ValueError("terminal_validation_failed")
        sidecar: dict[str, Any] = {
            "candidate_sha256": sha256_file(candidate),
            "reports": reports,
            "attempts": attempts,
        }
        atomic_json(Path(result["terminal_validation_sidecar_path"]), sidecar)
        old.seal(candidate, durable / "candidates")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.checked.tmp")
        shutil.copyfile(candidate, temporary)
        if sha256_file(temporary) != sidecar["candidate_sha256"]:
            temporary.unlink()
            raise ValueError("publication_hash_changed")
        temporary.replace(output)
    old.progress(started, "final", "published_checked_bytes", len(receipts) + len(reports))
    return 0

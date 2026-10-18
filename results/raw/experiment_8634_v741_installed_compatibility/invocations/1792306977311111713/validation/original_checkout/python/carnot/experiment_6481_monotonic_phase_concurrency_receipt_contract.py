"""Exp6481 monotonic phase and concurrency receipt contract.

Spec refs: REQ-INFRA-6481, SCENARIO-INFRA-6481-MONOTONIC-PHASES,
SCENARIO-INFRA-6481-DEPENDENCY-BINDING,
SCENARIO-INFRA-6481-RESOURCE-OWNERSHIP,
SCENARIO-INFRA-6481-CONCURRENCY-OVERLAP,
SCENARIO-INFRA-6481-FAIL-CLOSED-VALIDATION,
SCENARIO-INFRA-6481-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import phase_concurrency_receipts as phase_receipts
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6481
TASK_ID = "exp6481-monotonic-phase-concurrency-receipt-contract"
INFERENCE_SUBSTRATE = "deterministic_runtime_receipt_validation_no_llm"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6481_monotonic_phase_concurrency_receipt_contract.py"
)
API_RELATIVE_PATH = Path("python/carnot/phase_concurrency_receipts.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6481_monotonic_phase_concurrency_receipt_contract.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6481_monotonic_phase_concurrency_receipt_contract.json"
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6481_monotonic_phase_concurrency_receipt_contract --date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6481_monotonic_phase_concurrency_receipt_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/phase_concurrency_receipts.py,"
    "python/carnot/experiment_6481_monotonic_phase_concurrency_receipt_contract.py "
    "-m pytest "
    "tests/python/test_experiment_6481_monotonic_phase_concurrency_receipt_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/phase_concurrency_receipts.py,"
    "python/carnot/experiment_6481_monotonic_phase_concurrency_receipt_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6481_monotonic_phase_concurrency_receipt_contract.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6481_monotonic_phase_concurrency_receipt_contract.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6481_monotonic_phase_concurrency_receipt_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6481_monotonic_phase_concurrency_receipt_contract --validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6481 entry; "
    "receipt artifact lints apply"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    VALIDATE_COMMAND,
    E2E_PLAN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "receipt_schema_and_hash",
    "phase_rows",
    "dependency_hash_rows",
    "resource_ownership_rows",
    "concurrency_decision_rows",
    "process_identity_rows",
    "attack_matrix",
    "phase_concurrency_receipt_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "protected_files_unchanged",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal status distinguishes a complete contract build from partial instrumentation.",
    "receipt_schema_and_hash": "A versioned schema prevents later experiments from silently changing receipt meaning.",
    "phase_rows": "Monotonic phase rows separate queue, load, execution, exact verification, and write time.",
    "dependency_hash_rows": "Dependency hashes prove the task consumed the stated upstream bytes.",
    "resource_ownership_rows": "Resource intervals attribute CPUs, GPUs, files, and locks to one task attempt.",
    "concurrency_decision_rows": "Explicit decisions distinguish safe overlap from conflicting exclusive ownership.",
    "process_identity_rows": "PID and process-start identity prevent stale or borrowed activity from being credited.",
    "attack_matrix": "Constructive attacks test the known global-activity and time-order attribution failures.",
    "phase_concurrency_receipt_ready_score": "A conjunctive score blocks reuse until all ownership and ordering attacks fail closed.",
    "per_unit_rows": "Phase, dependency, resource, and attack rows make the contract independently auditable.",
    "aggregate_row_recomputation": "Row-derived readiness catches summaries that omit a failing receipt.",
    "protected_files_unchanged": "The receipt task must not alter conductor or active roadmap behavior.",
    "gate_check_summary": "A blocked verdict identifies the exact contract or test check that failed.",
    "preconditions_checked": "Preconditions prove required clocks, process metadata, and fixture paths were available.",
    "inference_substrate": "Declaring deterministic_runtime_receipt_validation_no_llm prevents fixture activity from becoming a compute claim.",
    "verifier_is_oracle": "Only schema, hash, process, and monotonic interval validation is authoritative.",
    "field_principles": "A field-to-principle map carries the evidence design into later tasks.",
    "field_provenance": "Per-field code and fixture paths make each value traceable.",
    "random_seed": "A fixed seed reproduces attack and overlap scheduling.",
    "duration_s": "Wall time catches a task that emitted without exercising concurrency fixtures.",
    "tests_run": "Recorded commands prove the API and its attacks executed.",
    "reproducibility_checksum": "The checksum binds schema, fixtures, implementation, and result.",
    "honest_verdict": "The verdict states contract readiness without claiming conductor concurrency.",
}

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    API_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/task_runtime_receipts.py"),
    Path("python/carnot/experiment_artifacts.py"),
    Path("python/carnot/terminal_artifacts.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("ops/e2e-test-plan.md"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_output(args: Sequence[str], root: Path) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {
        str(path): receipts.sha256_file(root / path)
        for path in SOURCE_RELATIVE_PATHS
    }


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {str(path): receipts.sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for path, before_hash in before.items():
        after_hash = receipts.sha256_file(root / path)
        files[path] = {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "unchanged": before_hash == after_hash,
        }
    return {
        "protected_files_unchanged": all(row["unchanged"] for row in files.values()),
        "files": files,
    }


def _process_fixture(pid: int, start: str, parent_pid: int = 49000) -> JsonDict:
    command = [sys.executable, "-m", __name__, str(pid)]
    return {
        "pid": pid,
        "process_start_identity": start,
        "parent_pid": parent_pid,
        "parent_process_start_identity": f"parent-start:{parent_pid}",
        "executable_path": sys.executable,
        "executable_sha256": receipts.sha256_file(sys.executable)
        or receipts.sha256_text(sys.executable),
        "command": command,
        "command_hash": receipts.sha256_json(command),
    }


def _phase_rows_for_attempt(
    *,
    task_id: str,
    attempt_id: str,
    process: Mapping[str, Any],
    base_ns: int,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, phase in enumerate(phase_receipts.REQUIRED_PHASES):
        start = base_ns + index * 100_000
        end = start + 80_000
        rows.append(
            phase_receipts.build_phase_row(
                task_id=task_id,
                attempt_id=attempt_id,
                phase=phase,
                process=process,
                monotonic_start_ns=start,
                monotonic_end_ns=end,
                wall_clock_start=_utc_now(),
                wall_clock_end=_utc_now(),
                exit_state={"returncode": 0, "timed_out": False, "signal": None},
            )
        )
    return rows


def _resource_window(base_ns: int) -> tuple[int, int]:
    start = base_ns + 2 * 100_000
    end = base_ns + 7 * 100_000 + 80_000
    return start, end


def _output_write_time(base_ns: int) -> int:
    return base_ns + 6 * 100_000 + 10_000


def _attempt_rows(
    *,
    task_id: str,
    attempt_id: str,
    process: Mapping[str, Any],
    base_ns: int,
    dependency_path: Path,
    output_path: Path,
    resource_key: str,
    resource_type: str,
    exclusive: bool,
) -> list[JsonDict]:
    rows = [
        phase_receipts.build_process_identity_row(
            task_id=task_id,
            attempt_id=attempt_id,
            process=process,
        )
    ]
    rows.extend(
        _phase_rows_for_attempt(
            task_id=task_id,
            attempt_id=attempt_id,
            process=process,
            base_ns=base_ns,
        )
    )
    rows.append(
        phase_receipts.build_dependency_row(
            task_id=task_id,
            attempt_id=attempt_id,
            process=process,
            path=dependency_path,
        )
    )
    start, end = _resource_window(base_ns)
    activity_sample = {}
    if resource_type == "gpu":
        activity_sample = {
            "source": "nvidia-smi-fixture",
            "pid": process["pid"],
            "process_start_identity": process["process_start_identity"],
            "sample_monotonic_ns": start + 10_000,
            "memory_mb": 1024,
        }
    rows.append(
        phase_receipts.build_resource_interval_row(
            task_id=task_id,
            attempt_id=attempt_id,
            process=process,
            resource_key=resource_key,
            resource_type=resource_type,
            exclusive=exclusive,
            monotonic_start_ns=start,
            monotonic_end_ns=end,
            acquired_in_phase="resource_acquisition",
            released_in_phase="resource_release",
            release_present=True,
            activity_sample=activity_sample,
        )
    )
    output_bytes = f"{task_id}:{attempt_id}:fixture-output\n".encode()
    output_path.write_bytes(output_bytes)
    rows.append(
        phase_receipts.build_output_row(
            task_id=task_id,
            attempt_id=attempt_id,
            process=process,
            path=output_path,
            output_bytes=output_bytes,
            write_monotonic_ns=_output_write_time(base_ns),
        )
    )
    return rows


def build_positive_fixture_rows(*, root: Path, fixture_root: Path | None = None) -> JsonDict:
    """Build positive CPU-overlap and serialized-GPU receipt fixtures."""

    target = fixture_root or Path(tempfile.gettempdir()) / "carnot-exp6481-fixtures"
    target.mkdir(parents=True, exist_ok=True)
    dependency_path = target / "dependency.txt"
    dependency_path.write_text("exp6481 dependency fixture bytes\n", encoding="utf-8")
    output_dir = target / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = (
        ("task-cpu-a", "attempt-cpu-a", _process_fixture(50101, "start-cpu-a"), 1_000_000_000, "CPU:shared", "cpu", False),
        ("task-cpu-b", "attempt-cpu-b", _process_fixture(50102, "start-cpu-b"), 1_000_300_000, "CPU:shared", "cpu", False),
        ("task-gpu-a", "attempt-gpu-a", _process_fixture(50201, "start-gpu-a"), 2_000_000_000, "GPU:0", "gpu", True),
        ("task-gpu-b", "attempt-gpu-b", _process_fixture(50202, "start-gpu-b"), 3_000_000_000, "GPU:0", "gpu", True),
    )
    rows: list[JsonDict] = []
    expected_attempts: dict[str, str] = {}
    for task_id, attempt_id, process, base_ns, resource_key, resource_type, exclusive in attempts:
        expected_attempts[attempt_id] = task_id
        rows.extend(
            _attempt_rows(
                task_id=task_id,
                attempt_id=attempt_id,
                process=process,
                base_ns=base_ns,
                dependency_path=dependency_path,
                output_path=output_dir / f"{attempt_id}.txt",
                resource_key=resource_key,
                resource_type=resource_type,
                exclusive=exclusive,
            )
        )
    return {
        "rows": rows,
        "expected_attempts": expected_attempts,
        "dependency_path": dependency_path,
        "fixture_root": target,
        "root_hash": receipts.sha256_text(str(root.resolve())),
    }


def _rows_by_type(rows: Sequence[Mapping[str, Any]], row_type: str) -> list[JsonDict]:
    return [dict(row) for row in rows if row.get("row_type") == row_type]


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute readiness from rows without trusting summary fields."""

    counts = Counter(str(row.get("row_type")) for row in rows)
    receipt_rows = [
        row
        for row in rows
        if row.get("row_type")
        in {"process_identity", "phase", "dependency", "resource_interval", "output"}
    ]
    expected_attempts = {
        str(row["attempt_id"]): str(row["task_id"])
        for row in receipt_rows
        if row.get("row_type") == "process_identity"
    }
    validation = phase_receipts.validate_receipt_rows(
        receipt_rows,
        expected_attempts=expected_attempts,
        verify_dependency_files=False,
    )
    decisions = [row for row in rows if row.get("row_type") == "concurrency_decision"]
    attack_rows = [row for row in rows if row.get("row_type") == "attack"]
    checks = {
        "receipt_rows_validate": validation["accepted"] is True,
        "all_required_phase_rows_present": all(
            count == len(phase_receipts.REQUIRED_PHASES)
            for count in validation["phase_count_by_attempt"].values()
        )
        and len(validation["phase_count_by_attempt"]) == 4,
        "dependency_rows_present": validation["dependency_row_count"] == 4,
        "resource_rows_present": validation["resource_interval_count"] == 4,
        "output_rows_present": validation["output_row_count"] == 4,
        "safe_cpu_overlap_decision_present": any(
            row.get("resource_key") == "CPU:shared" and row.get("decision") == "safe_overlap"
            for row in decisions
        ),
        "serialized_gpu_decision_present": any(
            row.get("resource_key") == "GPU:0"
            and row.get("decision") == "serialized_exclusive"
            for row in decisions
        ),
        "all_attacks_fail_closed": bool(attack_rows)
        and all(row.get("fail_closed") is True for row in attack_rows),
    }
    score = 1.0 if all(checks.values()) else 0.0
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(counts.items())),
        "validation_reasons": validation["reasons"],
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "phase_concurrency_receipt_ready_score_from_rows": score,
    }


def _gate_check_summary(
    *,
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "aggregate_ready_score_is_one": aggregate.get(
            "phase_concurrency_receipt_ready_score_from_rows"
        )
        == 1.0,
        "protected_files_unchanged": protected.get("protected_files_unchanged") is True,
    }
    return {
        "checks": checks,
        "all_gates_passed": all(checks.values()),
        "failed_gates": [key for key, value in checks.items() if not value],
    }


def _field_provenance(source_hashes: Mapping[str, str | None]) -> dict[str, JsonDict]:
    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    return {
        field: {
            "spec_refs": ["REQ-INFRA-6481"],
            "source_paths": source_paths,
            "value_source": "deterministic receipt rows and exact validation",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _preconditions_checked(
    *,
    root: Path,
    date: str,
    fixture: Mapping[str, Any],
    source_hashes: Mapping[str, str | None],
) -> JsonDict:
    return {
        "date": date,
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(["rev-parse", "HEAD"], root),
            "status_short": _git_output(["status", "--short"], root),
        },
        "current_receipt_apis": {
            "task_runtime_receipts": {
                "schema_version": receipts.SCHEMA_VERSION,
                "schema_sha256": receipts.sha256_json(
                    {
                        "required_row_fields": list(receipts.REQUIRED_ROW_FIELDS),
                        "required_phases": list(receipts.REQUIRED_PHASES),
                    }
                ),
            },
            "phase_concurrency_receipts": phase_receipts.receipt_schema_and_hash(),
        },
        "clock_sources": {
            "monotonic": "time.monotonic_ns",
            "wall_clock": "datetime.now(UTC)",
        },
        "process_identity_fields": [
            "pid",
            "process_start_identity",
            "parent_pid",
            "parent_process_start_identity",
        ],
        "current_process_identity": phase_receipts.current_process_identity(),
        "fixture_paths": {
            "fixture_root": str(fixture["fixture_root"]),
            "dependency_path": str(fixture["dependency_path"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "source_hashes": dict(source_hashes),
    }


def _status(score: float, gates: Mapping[str, Any]) -> str:
    if score == 1.0 and gates.get("all_gates_passed") is True:
        return "complete_phase_concurrency_receipt_contract"
    return "blocked_phase_concurrency_receipt_contract"


def _honest_verdict(status: str) -> str:
    if status.startswith("complete_"):
        return (
            "complete: reusable phase and concurrency receipt validation is ready; "
            "no conductor concurrency change is claimed"
        )
    return (
        "complete_blocked: phase and concurrency receipt validation failed; "
        "gate_check_summary names the failed checks"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    fixture_root: Path | None = None,
    write: bool = False,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the terminal Exp6481 artifact."""

    protected_before = _protected_hashes(root)
    source_hashes = _source_hashes(root)
    fixture = build_positive_fixture_rows(root=root, fixture_root=fixture_root)
    base_rows = list(fixture["rows"])
    decision_rows = phase_receipts.build_concurrency_decision_rows(base_rows)
    attack_matrix = phase_receipts.mutation_attack_matrix(
        base_rows,
        expected_attempts=fixture["expected_attempts"],
        verify_dependency_files=True,
    )
    per_unit_rows = [*base_rows, *decision_rows, *attack_matrix["rows"]]
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    protected = _protected_unchanged(root, protected_before)
    gates = _gate_check_summary(aggregate=aggregate, protected=protected)
    score = float(aggregate["phase_concurrency_receipt_ready_score_from_rows"])
    if not gates["all_gates_passed"]:
        score = 0.0
    status = _status(score, gates)
    artifact: JsonDict = {
        "status": status,
        "receipt_schema_and_hash": phase_receipts.receipt_schema_and_hash(),
        "phase_rows": _rows_by_type(base_rows, "phase"),
        "dependency_hash_rows": _rows_by_type(base_rows, "dependency"),
        "resource_ownership_rows": _rows_by_type(base_rows, "resource_interval"),
        "concurrency_decision_rows": decision_rows,
        "process_identity_rows": _rows_by_type(base_rows, "process_identity"),
        "attack_matrix": attack_matrix,
        "phase_concurrency_receipt_ready_score": score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "protected_files_unchanged": protected,
        "gate_check_summary": gates,
        "preconditions_checked": _preconditions_checked(
            root=root,
            date=run_date,
            fixture=fixture,
            source_hashes=source_hashes,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "results": list(tests_run or []),
        },
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, result_path)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clone = json.loads(receipts.canonical_json(payload))
    clone["duration_s"] = 0.0
    clone["reproducibility_checksum"] = ""
    return receipts.sha256_json(clone)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields and terminal boundaries."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    errors: list[str] = []
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("phase_concurrency_receipt_ready_score") != aggregate.get(
        "phase_concurrency_receipt_ready_score_from_rows"
    ):
        errors.append("phase_concurrency_receipt_ready_score mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact.get("field_principles", {}):
            errors.append(f"missing field_principles entry: {field}")
            break
    if artifact.get("protected_files_unchanged", {}).get("protected_files_unchanged") is not True:
        errors.append("protected_files_unchanged must be true")
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "complete_")):
        errors.append("honest_verdict lacks required terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    return receipts.write_json_atomic(path, artifact)


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    fixture_root: Path | None = None,
    write: bool = True,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and write the Exp6481 artifact."""

    start = time.monotonic()
    artifact = build_artifact(
        root=REPO_ROOT,
        result_path=result_path,
        fixture_root=fixture_root,
        write=False,
        duration_s=0.0001,
        tests_run=tests_run,
        run_date=date,
    )
    artifact["duration_s"] = max(time.monotonic() - start, 0.0001)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, result_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--fixture-root", default="")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        if not result_path.is_file():
            print(json.dumps({"ok": False, "errors": ["artifact missing"]}, sort_keys=True))
            return 1
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    fixture_root = Path(args.fixture_root) if str(args.fixture_root).strip() else None
    artifact = run(
        date=str(args.date),
        result_path=result_path,
        fixture_root=fixture_root,
        write=True,
    )
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "path": str(result_path),
                "status": artifact["status"],
                "phase_concurrency_receipt_ready_score": artifact[
                    "phase_concurrency_receipt_ready_score"
                ],
                "ok": not errors,
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

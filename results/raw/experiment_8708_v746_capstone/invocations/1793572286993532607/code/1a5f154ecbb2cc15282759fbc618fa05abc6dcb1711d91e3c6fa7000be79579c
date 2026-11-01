"""Enforce experiment-only validation and separate it from scientific value.

The V645 launchers mixed an affected command set with a repository-wide pytest
command. This module keeps the global policy unchanged and supplies a reusable
boundary for V646 experiment entrypoints. It also prevents an efficacy miss
from being reported as mechanism unavailability.

Spec refs: REQ-REPORT-7358 and SCENARIO-REPORT-7358-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
RUN_DATE = "20260917"
MILESTONE = "2026.09.646"
EXPERIMENT_ID = "exp7358-validation-contract"
SCHEMA = "carnot.exp7358.v646.validation_contract.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7358_v646_validation_contract.json")
RAW_DIR = Path("results/raw/experiment_7358_v646_validation_contract")
MODULE_PATH = Path("python/carnot/experiment_7358_v646_validation_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7358_v646_validation_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7358_v646_validation_contract.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HISTORICAL_PATHS = {
    "exp7346": Path("results/experiment_7346_v645_learning_adapter.json"),
    "exp7354": Path("results/experiment_7354_v645_arc_transfer.json"),
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7346_v645_learning_adapter.py"),
    Path("python/carnot/experiment_7354_v645_arc_transfer.py"),
    *HISTORICAL_PATHS.values(),
    Path("tests/python/test_experiment_7303_v642_validation_scope.py"),
)
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


@dataclass(frozen=True)
class AffectedManifest:
    """Name all files an experiment may send to the scoped validation runner."""

    experiment_id: str
    test_paths: tuple[str, ...]
    changed_modules: tuple[str, ...]
    static_paths: tuple[str, ...]


@dataclass(frozen=True)
class PlannedCommand:
    """Attach gate meaning to one existing runner command without changing it."""

    spec: validation_scope.CommandSpec
    category: str
    required: bool


@dataclass(frozen=True)
class EnvironmentCommandSpec(validation_scope.CommandSpec):
    """Keep command-local environment separate from the exact argument vector."""

    command_environment: tuple[tuple[str, str], ...] = ()


V646_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def utc_now() -> str:
    """Return an aware UTC timestamp for a real command or artifact boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush each phase boundary so a bounded task never appears stalled."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7358] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so source and log identities cannot drift silently."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for independently recomputed artifact evidence."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object with an fsync and local atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def precondition_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Record one exact prerequisite field before dependent work starts."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _load_object(path: Path) -> JsonDict:
    """Return a JSON object, or an empty object for malformed external bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], dict[str, JsonDict]]:
    """Authenticate exact source paths and expected historical dispositions."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            precondition_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        precondition_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7358",
            "REQ-REPORT-7358" if "REQ-REPORT-7358" in spec_text else None,
        )
    )

    historical: dict[str, JsonDict] = {}
    expected_fields = {
        "exp7346": {
            "status": "complete_learning_adapter_disqualified",
            "verdict_class": "disqualified",
            "flagged_adversarial": False,
        },
        "exp7354": {
            "status": "disqualified",
            "verdict_class": "disqualified",
            "flagged_adversarial": True,
        },
    }
    for source, relative in HISTORICAL_PATHS.items():
        artifact = _load_object(root / relative)
        if artifact:
            historical[source] = artifact
        for field, expected in expected_fields[source].items():
            checks.append(
                precondition_row(
                    f"historical_{source}_{field}",
                    relative.as_posix(),
                    field,
                    expected,
                    artifact.get(field),
                )
            )
        broad = [
            row
            for row in artifact.get("validation_receipts", [])
            if isinstance(row, Mapping) and row.get("name") == "full_python_suite"
        ]
        observed = broad[0].get("command_argv", [])[1:] if len(broad) == 1 else None
        checks.append(
            precondition_row(
                f"historical_{source}_broad_command",
                relative.as_posix(),
                "validation_receipts.full_python_suite.command_argv[1:]",
                ["tests/python", "-q"],
                observed,
            )
        )

    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "experiment_id: 7358" in exclusion_text
    checks.append(
        precondition_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
        )
    )
    return checks, hashes, historical


def build_command_plan(
    repo_root: Path,
    manifest: AffectedManifest,
    private_root: Path,
) -> list[validation_scope.CommandSpec]:
    """Build the Exp7303 command set after creating all private parent paths."""

    basetemp = private_root / "basetemp"
    coverage_parent = private_root / "coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_parent.mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        repo_root,
        manifest.test_paths,
        manifest.changed_modules,
        static_paths=manifest.static_paths,
        basetemp=basetemp,
        coverage_file=coverage_parent / ".coverage",
    )
    root = repo_root.resolve()
    stable: list[validation_scope.CommandSpec] = []
    for command in commands:
        environment: tuple[tuple[str, str], ...] = ()
        argv = list(command.argv)
        executable = Path(argv[0])
        if executable.parent == root / ".venv/bin":
            argv[0] = executable.relative_to(root).as_posix()
        argv = ["." if argument == str(root) else argument for argument in argv]
        if command.name == "changed_module_coverage_report":
            data_file = next(argument for argument in argv if argument.startswith("--data-file="))
            argv.remove(data_file)
            environment = (("COVERAGE_FILE", data_file.split("=", 1)[1]),)
        stable.append(
            EnvironmentCommandSpec(
                command.name,
                tuple(argv),
                command.scope,
                command.timeout_s,
                environment,
            )
        )
    return stable


def validate_command_plan(
    repo_root: Path,
    manifest: AffectedManifest,
    commands: Sequence[validation_scope.CommandSpec],
) -> list[str]:
    """Reject command expansion, missing private parents, and command drift."""

    del repo_root  # Paths in argv are already resolved by the Exp7303 builder.
    errors: list[str] = []
    counts = Counter(command.name for command in commands)
    expected_names = set(validation_scope.REQUIRED_CHECK_NAMES)
    for name in validation_scope.REQUIRED_CHECK_NAMES:
        if counts[name] == 0:
            errors.append(f"missing_command:{name}")
        elif counts[name] > 1:
            errors.append(f"duplicate_command:{name}")
    for name in counts:
        if name not in expected_names:
            errors.append(f"unexpected_command:{name}")

    allowed_tests = set(manifest.test_paths)
    for command in commands:
        for argument in command.argv:
            normalized = argument.rstrip("/")
            if normalized in {"tests", "tests/python"}:
                errors.append(f"unscoped_test_target:{argument}")
            elif argument.startswith("tests/") and argument.endswith(".py"):
                if argument not in allowed_tests:
                    errors.append(f"test_outside_manifest:{argument}")
            if argument.startswith("--basetemp="):
                parent = Path(argument.split("=", 1)[1]).parent
                if not parent.is_dir():
                    errors.append(f"missing_basetemp_parent:{parent}")
            if argument.startswith("--data-file="):
                parent = Path(argument.split("=", 1)[1]).parent
                if not parent.is_dir():
                    errors.append(f"missing_coverage_parent:{parent}")
        environment = dict(getattr(command, "command_environment", ()))
        if coverage_file := environment.get("COVERAGE_FILE"):
            parent = Path(coverage_file).parent
            if not parent.is_dir():
                errors.append(f"missing_coverage_parent:{parent}")
    return list(dict.fromkeys(errors))


def reduce_affected_receipts(
    repo_root: Path,
    manifest: AffectedManifest,
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Require all scoped exits and prove imports resolve below this worktree."""

    reduced = validation_scope.reduce_required_checks(receipts)
    worktree = (repo_root / "python").resolve()
    import_rows = [row for row in receipts if row.get("name") == "worktree_imports"]
    resolved = dict(import_rows[0].get("resolved_imports") or {}) if len(import_rows) == 1 else {}
    installed = [
        str(path)
        for path in resolved.values()
        if not Path(str(path)).resolve().is_relative_to(worktree)
    ]
    expected_import_count = len(manifest.changed_modules)
    imports_passed = len(resolved) == expected_import_count and not installed
    return {
        **reduced,
        "installed_imports": installed,
        "expected_import_count": expected_import_count,
        "observed_import_count": len(resolved),
        "passed": reduced["required_checks_passed"] and imports_passed,
    }


def command_row(
    receipt: Mapping[str, Any],
    *,
    category: str,
    required: bool,
    started_at_utc: str,
    ended_at_utc: str,
) -> JsonDict:
    """Add classification and UTC boundaries to one exact runner receipt."""

    fields = (
        "name",
        "command",
        "command_argv",
        "scope",
        "exit_code",
        "duration_s",
        "log_path",
        "log_sha256",
        "passed",
        "timed_out",
        "output_tail",
        "resolved_imports",
    )
    row = {field: deepcopy(receipt[field]) for field in fields if field in receipt}
    row.update(
        {
            "command_category": category,
            "required": required,
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
        }
    )
    return row


def run_categorized_commands(
    repo_root: Path,
    commands: Sequence[PlannedCommand],
    *,
    log_dir: Path,
    heartbeat_s: float = 60.0,
) -> list[JsonDict]:
    """Reuse the streaming runner and enrich each receipt with gate meaning."""

    rows: list[JsonDict] = []
    for index, planned in enumerate(commands):
        started_at = utc_now()
        command_environment = dict(getattr(planned.spec, "command_environment", ()))
        receipts = validation_scope.run_commands(
            repo_root,
            [planned.spec],
            log_dir=log_dir / f"{index:02d}_{planned.spec.name}",
            extra_env=command_environment,
            heartbeat_s=heartbeat_s,
        )
        ended_at = utc_now()
        row = command_row(
            receipts[0],
            category=planned.category,
            required=planned.required,
            started_at_utc=started_at,
            ended_at_utc=ended_at,
        )
        row["command_environment"] = command_environment
        rows.append(row)
    return rows


def classify_terminal(
    *,
    prerequisites_passed: bool,
    required_validation_passed: bool,
    safety_passed: bool,
    result_present: bool,
    result_complete: bool,
    efficacy_passed: bool,
    flagged_adversarial: bool,
    retryable_own_work_unfinished: bool,
) -> JsonDict:
    """Classify terminal state without making readiness depend on efficacy."""

    if not prerequisites_passed:
        verdict = "blocked"
        honest = "blocked_external_prerequisite"
    elif retryable_own_work_unfinished:
        verdict = "partial"
        honest = "partial_retryable_owned_work_unfinished"
    elif (
        not required_validation_passed
        or not safety_passed
        or not result_present
        or not result_complete
        or flagged_adversarial
    ):
        verdict = "disqualified"
        honest = "complete_disqualified_required_evidence"
    elif efficacy_passed:
        verdict = "circular_positive"
        honest = "complete_circular_positive_oracle_defined_efficacy"
    else:
        verdict = "null"
        honest = "complete_null_efficacy_with_ready_safe_fixture"

    ready = int(verdict in {"null", "circular_positive"})
    capture = int(ready and result_present and result_complete)
    value = int(verdict == "circular_positive")
    return {
        "verdict_class": verdict,
        "honest_verdict": honest,
        "fixture_ready_score": ready,
        "capture_complete_score": capture,
        "scientific_value_score": value,
        "promotion_score": int(value and ready and not flagged_adversarial),
    }


def run_classification_controls() -> list[JsonDict]:
    """Exercise each closed boundary with independently declared expectations."""

    baseline = {
        "prerequisites_passed": True,
        "required_validation_passed": True,
        "safety_passed": True,
        "result_present": True,
        "result_complete": True,
        "efficacy_passed": False,
        "flagged_adversarial": False,
        "retryable_own_work_unfinished": False,
    }
    cases: tuple[tuple[str, JsonDict, str], ...] = (
        ("completed_null", {}, "null"),
        ("failed_affected_test", {"required_validation_passed": False}, "disqualified"),
        ("unsafe_fixture", {"safety_passed": False}, "disqualified"),
        ("missing_completed_result", {"result_present": False}, "disqualified"),
        ("blocked_prerequisite", {"prerequisites_passed": False}, "blocked"),
        ("adversarial_flag", {"flagged_adversarial": True}, "disqualified"),
        (
            "unfinished_retryable_owned_work",
            {"result_present": False, "retryable_own_work_unfinished": True},
            "partial",
        ),
        ("oracle_defined_benefit", {"efficacy_passed": True}, "circular_positive"),
    )
    rows: list[JsonDict] = []
    for control_id, overrides, expected in cases:
        inputs = {**baseline, **overrides}
        actual = classify_terminal(**inputs)
        rows.append(
            {
                "control_id": control_id,
                "inputs": inputs,
                "expected_terminal_class": expected,
                "actual_terminal_class": actual["verdict_class"],
                "actual_scores": {
                    key: actual[key]
                    for key in (
                        "fixture_ready_score",
                        "capture_complete_score",
                        "scientific_value_score",
                        "promotion_score",
                    )
                },
                "passed": actual["verdict_class"] == expected,
                "disposition": "complete",
                "censored": False,
                "failures": [],
                "costs": {"current_model_calls": 0},
            }
        )
    return rows


def _historical_caller(source: str, name: str) -> str:
    """Map each historical command to the production function that selected it."""

    if source == "exp7346":
        if name == "full_python_suite":
            return "carnot.experiment_7346_v645_learning_adapter.main"
        if name in validation_scope.REQUIRED_CHECK_NAMES:
            return "carnot.experiment_7346_v645_learning_adapter.main->run_scoped_validation"
        return "carnot.experiment_7346_v645_learning_adapter._terminal_commands"
    if name in {"e2e_009", "e2e_010", "e2e_offline_smoke", "full_python_suite"}:
        return "carnot.experiment_7354_v645_arc_transfer.e2e_command_specs"
    if name in validation_scope.REQUIRED_CHECK_NAMES:
        return "carnot.experiment_7354_v645_arc_transfer.run_scoped_validation"
    return "carnot.experiment_7354_v645_arc_transfer.terminal_command_specs"


def _historical_category(name: str) -> str:
    """Keep repository health, completion, safety, and affected checks distinct."""

    if name == "full_python_suite":
        return "diagnostic_repository_health"
    if name == "adversarial_verify":
        return "safety"
    if name in {"independent_reducer", "verdict_row_consistency_strict"}:
        return "completion"
    return "required_validation"


def historical_command_rows(artifacts: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Trace every old validation receipt without executing any old launcher."""

    rows: list[JsonDict] = []
    for source in sorted(artifacts):
        artifact = artifacts[source]
        for receipt in artifact.get("validation_receipts", []):
            if not isinstance(receipt, Mapping):
                continue
            name = str(receipt.get("name"))
            rows.append(
                {
                    **deepcopy(dict(receipt)),
                    "source": source,
                    "caller": _historical_caller(source, name),
                    "command_category": _historical_category(name),
                    "historical_required": name == "full_python_suite"
                    or name in validation_scope.REQUIRED_CHECK_NAMES,
                    "original_status": artifact.get("status"),
                    "original_verdict_class": artifact.get("verdict_class"),
                    "original_flagged_adversarial": artifact.get("flagged_adversarial"),
                    "historical_only": True,
                }
            )
    return rows


def _worktree_imports_pass(artifact: Mapping[str, Any]) -> bool:
    """Cold-check that the required import receipt points below the worktree."""

    rows = [
        row
        for row in artifact.get("command_plan_rows", [])
        if row.get("name") == "worktree_imports" and row.get("required") is True
    ]
    root = Path(str(artifact.get("worktree_root") or "/__missing__")) / "python"
    resolved = dict(rows[0].get("resolved_imports") or {}) if len(rows) == 1 else {}
    expected = len((artifact.get("affected_manifest") or {}).get("changed_modules") or [])
    return len(resolved) == expected and all(
        Path(str(value)).resolve().is_relative_to(root.resolve()) for value in resolved.values()
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute infrastructure readiness from raw commands and control rows."""

    commands = artifact.get("command_plan_rows") or []
    classifications = artifact.get("classification_rows") or []
    preconditions = artifact.get("preconditions_checked") or []
    counts = Counter(row.get("name") for row in commands if row.get("required") is True)
    required_names_passed = all(
        counts[name] == 1
        and next(row for row in commands if row.get("name") == name).get("passed") is True
        and next(row for row in commands if row.get("name") == name).get("exit_code") == 0
        and next(row for row in commands if row.get("name") == name).get("timed_out") is False
        for name in validation_scope.REQUIRED_CHECK_NAMES
    )
    extra_required_passed = all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is False
        for row in commands
        if row.get("required") is True
    )
    controls_passed = bool(classifications) and all(
        row.get("passed") is True
        and row.get("expected_terminal_class") == row.get("actual_terminal_class")
        for row in classifications
    )
    preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    heartbeat_passed = artifact.get("heartbeat_control_passed") is True
    imports_passed = _worktree_imports_pass(artifact)
    ready = int(
        preconditions_passed
        and required_names_passed
        and extra_required_passed
        and controls_passed
        and heartbeat_passed
        and imports_passed
        and artifact.get("flagged_adversarial") is False
    )
    classification = classify_terminal(
        prerequisites_passed=preconditions_passed,
        required_validation_passed=bool(
            required_names_passed and extra_required_passed and controls_passed and imports_passed
        ),
        safety_passed=heartbeat_passed,
        result_present=True,
        result_complete=True,
        efficacy_passed=False,
        flagged_adversarial=artifact.get("flagged_adversarial") is True,
        retryable_own_work_unfinished=False,
    )
    return {
        "preconditions_passed": preconditions_passed,
        "required_validation_passed": required_names_passed and extra_required_passed,
        "classification_controls_passed": controls_passed,
        "heartbeat_control_passed": heartbeat_passed,
        "worktree_imports_passed": imports_passed,
        "validation_contract_ready_score": ready,
        "fixture_ready_score": int(ready == 1),
        "capture_complete_score": int(ready == 1),
        "scientific_value_score": 0,
        "promotion_score": 0,
        "verdict_class": "null" if ready else classification["verdict_class"],
    }


def _gate(check: str, category: str, expected: Any, observed: Any) -> JsonDict:
    """Keep each gate category explicit even when the scientific gate is null."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact failed field without hiding later failed gates."""

    failures = [dict(row) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
    }


def _historical_sidecars(
    artifacts: Mapping[str, Mapping[str, Any]], source_hashes: Mapping[str, str]
) -> list[JsonDict]:
    """Label old model receipts so they cannot be counted as current calls."""

    rows: list[JsonDict] = []
    for source, artifact in sorted(artifacts.items()):
        path = HISTORICAL_PATHS[source].as_posix()
        rows.append(
            {
                "source": source,
                "artifact_path": path,
                "artifact_sha256": source_hashes.get(path),
                "label": "historical_diagnostic_only_no_current_generation",
                "historical_MODEL_SPECS": deepcopy(artifact.get("MODEL_SPECS") or []),
                "historical_model_invoked": artifact.get("model_invoked"),
                "historical_invocation_counts": deepcopy(artifact.get("invocation_counts") or {}),
                "authorizes_current_inference": False,
            }
        )
    return rows


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary fields directly instead of wrapping their values."""

    specific = {
        "schema": "Version this record and retain ordinary experiment identity fields.",
        "status": "Use a terminal status only after actual affected validation.",
        "run_date": "Use 20260917 with actual aware UTC timestamps.",
        "preconditions_checked": "Check exact paths and required fields before dependent work.",
        "MODEL_SPECS": "List intended current models; this CPU contract intends none.",
        "model_invoked": "Set true for any attempted current model load or generation.",
        "invocation_counts": "Separate all current attempted, completed, failed, cancelled, and live calls.",
        "inference_substrate": "Name the actual host CPU command planning and exact reduction.",
        "inference_substrate_class": "Use the actual closed class without duration padding.",
        "execution_venue": "Record host CPU execution; V646 performs no board run.",
        "duration_s": "Use measured monotonic elapsed time without synthetic delay.",
        "phase_spans": "Retain disjoint measured phase intervals.",
        "random_seed": "Use null because this deterministic contract performs no sampling.",
        "reproducibility_checksum": "Bind code, inputs, settings, and raw evidence.",
        "source_artifact_hashes": "Hash exact producer paths and retain original dispositions.",
        "rows": "Retain every comparative classification control.",
        "sample_size_budget": "Freeze planned, attempted, completed, and censored controls.",
        "acceptance_gate_results": "Separate required validation, safety, value, and promotion.",
        "gate_check_summary": "Name each first failed check with expected and observed values.",
        "verifier_is_oracle": "Disclose that the evaluator defines command-contract correctness.",
        "honest_verdict": "Distinguish infrastructure readiness from null scientific value.",
        "verdict_class": "Use the closed terminal verdict enum.",
        "flagged_adversarial": "Current critical findings prevent readiness and promotion.",
        "validation_receipts": "Retain exact commands, exits, elapsed time, categories, and hashes.",
        "repository_health": "Keep dated unrelated failures outside affected required checks.",
        "field_principles": "Explain each field without changing its scalar or dictionary type.",
        "validation_contract_ready_score": "One requires scoped planning, truthful receipts, heartbeat, controls, and affected checks.",
        "command_plan_rows": "Retain each actual planned or executed command and evidence hash.",
        "classification_rows": "Retain expected and actual classes for every mutation control.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable code, input, command, control, and reduction evidence."""

    bound = {
        key: artifact.get(key)
        for key in (
            "schema",
            "experiment_id",
            "milestone",
            "run_date",
            "source_artifact_hashes",
            "preconditions_checked",
            "affected_manifest",
            "command_plan_rows",
            "classification_rows",
            "heartbeat_control_passed",
            "independent_reduction",
            "acceptance_gate_results",
            "validation_contract_ready_score",
            "verdict_class",
        )
    }
    return canonical_hash(bound)


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    historical_artifacts: Mapping[str, Mapping[str, Any]],
    historical_commands: Sequence[Mapping[str, Any]],
    command_plan_rows: Sequence[Mapping[str, Any]],
    classification_rows: Sequence[Mapping[str, Any]],
    manifest: AffectedManifest,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    heartbeat_control_passed: bool,
    diagnostic_receipts: Sequence[Mapping[str, Any]] = (),
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one schema-complete record from raw plans and classification rows."""

    manifest_value = {
        "experiment_id": manifest.experiment_id,
        "test_paths": list(manifest.test_paths),
        "changed_modules": list(manifest.changed_modules),
        "static_paths": list(manifest.static_paths),
    }
    shell: JsonDict = {
        "worktree_root": str(_worktree_root_from_rows(manifest, command_plan_rows)),
        "affected_manifest": manifest_value,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "command_plan_rows": [deepcopy(dict(row)) for row in command_plan_rows],
        "classification_rows": [deepcopy(dict(row)) for row in classification_rows],
        "heartbeat_control_passed": heartbeat_control_passed,
        "flagged_adversarial": flagged_adversarial,
    }
    reduction = independent_reduce(shell)
    ready = reduction["validation_contract_ready_score"]
    gates = [
        _gate(
            "required_validation",
            "required_validation",
            True,
            reduction["required_validation_passed"],
        ),
        _gate(
            "worktree_imports", "required_validation", True, reduction["worktree_imports_passed"]
        ),
        _gate("heartbeat_silence_guard", "safety", True, heartbeat_control_passed),
        _gate(
            "classification_controls",
            "completion",
            True,
            reduction["classification_controls_passed"],
        ),
        _gate("safe_fixture_readiness", "safety", 1, reduction["fixture_ready_score"]),
        _gate("complete_capture", "completion", 1, reduction["capture_complete_score"]),
        _gate("scientific_value", "efficacy", 1, 0),
        _gate("scientific_promotion", "promotion", 1, 0),
    ]
    status = (
        "complete_validation_contract_null_science"
        if ready
        else "complete_validation_contract_disqualified"
    )
    verdict = "null" if ready else reduction["verdict_class"]
    honest = (
        "complete_null_scientific_value_not_tested_validation_contract_ready"
        if ready
        else "complete_disqualified_validation_contract_not_ready"
    )
    rows = [deepcopy(dict(row)) for row in classification_rows]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": shell["preconditions_checked"],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "artifact_qa_lint_tests",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": platform.node(),
        "host_computation": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": None,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": _historical_sidecars(historical_artifacts, source_hashes),
        "historical_command_traces": [deepcopy(dict(row)) for row in historical_commands],
        "rows": rows,
        "sample_size_budget": {
            "planned_classification_controls": 8,
            "attempted_classification_controls": len(rows),
            "completed_classification_controls": sum(
                row.get("disposition") == "complete" for row in rows
            ),
            "censored_classification_controls": sum(bool(row.get("censored")) for row in rows),
            "stopping_rule": "Run every frozen control once and never extend from outcomes.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [
            *[deepcopy(dict(row)) for row in command_plan_rows],
            *[deepcopy(dict(row)) for row in diagnostic_receipts],
        ],
        "repository_health": {
            "status": "degraded_historical_timeouts_retained",
            "affects_required_checks": False,
            "as_of": "2026-09-17",
            "historical_failures": [
                {
                    "source": source,
                    "date": "2026-09-16",
                    "command": "pytest tests/python -q",
                    "classification": "unrelated_repository_health_full_suite_timeout",
                    "original_verdict_class": artifact.get("verdict_class"),
                    "resolved": False,
                }
                for source, artifact in sorted(historical_artifacts.items())
            ],
        },
        "field_principles": {},
        "validation_contract_ready_score": ready,
        "fixture_ready_score": reduction["fixture_ready_score"],
        "capture_complete_score": reduction["capture_complete_score"],
        "scientific_value_score": 0,
        "promotion_score": 0,
        "command_plan_rows": shell["command_plan_rows"],
        "classification_rows": shell["classification_rows"],
        "heartbeat_control_passed": heartbeat_control_passed,
        "affected_manifest": manifest_value,
        "worktree_root": shell["worktree_root"],
        "independent_reduction": reduction,
        "production_defaults_changed": False,
        "global_repository_health_policy_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _worktree_root_from_rows(
    manifest: AffectedManifest, command_rows: Sequence[Mapping[str, Any]]
) -> Path:
    """Recover the checked root from a resolved import and its manifest path."""

    imports = next(
        (
            dict(row.get("resolved_imports") or {})
            for row in command_rows
            if row.get("name") == "worktree_imports"
        ),
        {},
    )
    if imports and manifest.changed_modules:
        resolved = Path(str(next(iter(imports.values())))).resolve()
        relative = Path(manifest.changed_modules[0])
        if len(resolved.parts) >= len(relative.parts):
            return Path(*resolved.parts[: -len(relative.parts)])
    return REPO_ROOT


def build_blocked_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    historical_artifacts: Mapping[str, Mapping[str, Any]],
    duration_s: float,
    started_at_utc: str,
    completed_at_utc: str,
) -> JsonDict:
    """Publish external absence as blocked with no dependent command work."""

    failed = next((dict(row) for row in preconditions if row.get("passed") is not True), None)
    summary = {"all_passed": False, "failed_count": 1, "first_failure": failed}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "blocked_validation_contract_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host_cpu_precondition_checks_only",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [],
        "random_seed": None,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": _historical_sidecars(historical_artifacts, source_hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_classification_controls": 8,
            "attempted_classification_controls": 0,
            "completed_classification_controls": 0,
            "censored_classification_controls": 8,
            "stopping_rule": "Stop before dependent work on external absence.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_validation_contract_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {"status": "not_evaluated", "affects_required_checks": False},
        "field_principles": {},
        "validation_contract_ready_score": 0,
        "fixture_ready_score": 0,
        "capture_complete_score": 0,
        "scientific_value_score": 0,
        "promotion_score": 0,
        "command_plan_rows": [],
        "classification_rows": [],
        "heartbeat_control_passed": False,
        "affected_manifest": {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(V646_MANIFEST.test_paths),
            "changed_modules": list(V646_MANIFEST.changed_modules),
            "static_paths": list(V646_MANIFEST.static_paths),
        },
        "worktree_root": str(REPO_ROOT),
        "independent_reduction": {},
        "production_defaults_changed": False,
        "global_repository_health_policy_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, raw reduction, scores, principles, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_declaration_mismatch")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_mismatch")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS:
        errors.append("verdict_class_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")

    if artifact.get("verdict_class") == "blocked":
        if artifact.get("command_plan_rows") or artifact.get("validation_receipts"):
            errors.append("blocked_artifact_has_dependent_work")
        if (artifact.get("gate_check_summary") or {}).get("first_failure") is None:
            errors.append("blocked_gate_summary_missing")
        if artifact.get("validation_contract_ready_score") != 0:
            errors.append("blocked_ready_score_nonzero")
    else:
        reduced = independent_reduce(artifact)
        stored = artifact.get("independent_reduction")
        if stored != reduced:
            errors.append("stored_reduction_mismatch")
        expected_fields = {
            "validation_contract_ready_score": reduced["validation_contract_ready_score"],
            "fixture_ready_score": reduced["fixture_ready_score"],
            "capture_complete_score": reduced["capture_complete_score"],
            "scientific_value_score": reduced["scientific_value_score"],
            "promotion_score": reduced["promotion_score"],
            "verdict_class": reduced["verdict_class"],
        }
        if any(artifact.get(key) != expected for key, expected in expected_fields.items()):
            if "stored_reduction_mismatch" not in errors:
                errors.append("stored_reduction_mismatch")
    if artifact.get("verdict_class") in {"blocked", "disqualified", "partial"} and any(
        artifact.get(field) != 0
        for field in (
            "validation_contract_ready_score",
            "fixture_ready_score",
            "capture_complete_score",
            "scientific_value_score",
            "promotion_score",
        )
    ):
        errors.append("failed_state_scores_nonzero")
    if artifact.get("flagged_adversarial") is True and artifact.get("promotion_score") != 0:
        errors.append("adversarial_promotion_nonzero")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _control_commands() -> list[PlannedCommand]:  # pragma: no cover - exercised by E2E entrypoint.
    """Create bounded pass, fail, and silent controls for the shared runner."""

    python = str(REPO_ROOT / ".venv/bin/python")
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "control_pass", (python, "-u", "-c", "print('pass', flush=True)"), "control"
            ),
            "completion",
            False,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "control_fail", (python, "-u", "-c", "raise SystemExit(7)"), "control"
            ),
            "required_validation",
            False,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "control_silence",
                (python, "-u", "-c", "import time;time.sleep(0.15)"),
                "control",
            ),
            "safety",
            False,
        ),
    ]


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build cold reduction and strict readers for the measured candidate."""

    python = str(REPO_ROOT / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7358_v646_validation_contract import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_reducer",
                (python, "-u", "-c", reducer_code, str(candidate)),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def _span(phase: str, phase_started: float, run_started: float) -> JsonDict:  # pragma: no cover
    """Record one disjoint monotonic phase interval from the run origin."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def run_experiment(  # pragma: no cover - executed through the declared entrypoint.
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Execute controls, affected checks, terminal readers, and atomic publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, hashes, historical = collect_preconditions(root)
    spans.append(_span("load", phase_started, started))
    progress(
        started,
        "preconditions",
        "end",
        passed=all(row["passed"] for row in preconditions),
    )
    if not all(row["passed"] for row in preconditions):
        blocked = build_blocked_artifact(
            preconditions=preconditions,
            source_hashes=hashes,
            historical_artifacts=historical,
            duration_s=time.monotonic() - started,
            started_at_utc=started_at,
            completed_at_utc=utc_now(),
        )
        progress(started, "write", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        progress(started, "write", "after_atomic_blocked", path=output_path)
        return blocked

    historical_rows = historical_command_rows(historical)
    private = Path(tempfile.mkdtemp(prefix="exp7358-validation-", dir="/tmp"))
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    progress(started, "controls", "before_subprocess_controls")
    control_receipts = run_categorized_commands(
        root,
        _control_commands(),
        log_dir=raw_dir / "validation/controls",
        heartbeat_s=0.05,
    )
    classifications = run_classification_controls()
    silence = next(row for row in control_receipts if row["name"] == "control_silence")
    heartbeat_passed = silence["passed"] is True and silence["duration_s"] >= 0.05
    control_passed = (
        [row["exit_code"] for row in control_receipts] == [0, 7, 0]
        and all(
            row["passed"] is expected
            for row, expected in zip(control_receipts, (True, False, True), strict=True)
        )
        and all(row["passed"] for row in classifications)
    )
    spans.append(_span("evaluation", phase_started, started))
    progress(started, "controls", "after_subprocess_controls", passed=control_passed)

    phase_started = time.monotonic()
    commands = build_command_plan(root, V646_MANIFEST, private)
    plan_errors = validate_command_plan(root, V646_MANIFEST, commands)
    progress(started, "validation", "before_affected_subprocesses", plan_errors=len(plan_errors))
    affected_rows: list[JsonDict] = []
    if not plan_errors:
        affected_rows = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    affected_reduction = reduce_affected_receipts(root, V646_MANIFEST, affected_rows)
    if not affected_reduction["passed"] or not control_passed:
        heartbeat_passed = False
    spans.append(_span("validation", phase_started, started))
    progress(
        started,
        "validation",
        "after_affected_subprocesses",
        passed=affected_reduction["passed"],
    )

    candidate = build_artifact(
        preconditions=preconditions,
        source_hashes=hashes,
        historical_artifacts=historical,
        historical_commands=historical_rows,
        command_plan_rows=affected_rows,
        classification_rows=classifications,
        manifest=V646_MANIFEST,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        heartbeat_control_passed=heartbeat_passed,
        diagnostic_receipts=control_receipts,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_terminal_subprocesses")
    terminal_rows = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started))
    terminal_passed = all(row["passed"] for row in terminal_rows)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_rows)
    progress(
        started,
        "terminal_validation",
        "after_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    final = build_artifact(
        preconditions=preconditions,
        source_hashes=hashes,
        historical_artifacts=historical,
        historical_commands=historical_rows,
        command_plan_rows=[*affected_rows, *terminal_rows],
        classification_rows=classifications,
        manifest=V646_MANIFEST,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        heartbeat_control_passed=heartbeat_passed,
        diagnostic_receipts=control_receipts,
        flagged_adversarial=critical or not terminal_passed,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the thin public experiment entrypoint arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the validation contract through the repository-root entrypoint."""

    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

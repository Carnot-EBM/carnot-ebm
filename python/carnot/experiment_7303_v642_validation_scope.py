"""Produce the V642 validation-scope control artifact.

The experiment authenticates the V641 collection failures as history. It then
uses the new explicit-scope runner for current checks. The two outcomes remain
separate so old repository debt cannot hide or disqualify current test results.

Spec refs: REQ-REPORT-7303 and SCENARIO-REPORT-7303-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as scoped


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260914"
MILESTONE = "2026.09.642"
EXPERIMENT_ID = "exp7303-validation-scope"
SCHEMA = "carnot.validation_scope.v1"
RESULT_PATH = Path("results/experiment_7303_v642_validation_scope.json")
RAW_DIR = Path("results/raw/experiment_7303_v642_validation_scope")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7303_v642_validation_scope.json")
MODULE_PATH = Path("python/carnot/experiment_7303_v642_validation_scope.py")
RUNNER_PATH = Path("python/carnot/reporting/experiment_7303_validation_scope.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7303_v642_validation_scope.py")
TEST_PATH = Path("tests/python/test_experiment_7303_v642_validation_scope.py")

DEFAULT_TEST_PATHS = (TEST_PATH.as_posix(),)
DEFAULT_CHANGED_MODULES = (RUNNER_PATH.as_posix(), MODULE_PATH.as_posix())
DEFAULT_STATIC_PATHS = (WRAPPER_PATH.as_posix(),)

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

FIELD_PRINCIPLES = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "status": "Write the terminal result only after the work and checks; checkpoints remain separate.",
    "run_date": "Use 20260914 with actual UTC start/end and monotonic phase timing.",
    "preconditions_checked": "Hash real inputs and record actual availability and failed checks.",
    "MODEL_SPECS": "Actual current executable model identities; historical identities remain in sidecars.",
    "model_invoked": "True for any attempted model load or generation, even when output is unusable.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight loads and generations.",
    "inference_substrate": "Describe actual computation with a recognized literal, not intended work.",
    "inference_substrate_class": "Full generation has a 60s floor; bounded generation 10s; load-only 2s. Never pad time.",
    "execution_venue": "Record actual host/device work; a CPU replay is not GPU or FPGA execution.",
    "duration_s": "Measure total elapsed and disjoint phase spans including failed work.",
    "random_seed": "Seal development and independent evaluation seeds before seeing outcomes.",
    "reproducibility_checksum": "Bind code, inputs, config, model when used, and raw evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and quarantine state.",
    "rows": "Record every comparative unit/arm with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the frozen stopping rule.",
    "acceptance_gate_results": "Every check has expected, observed, passed, and a one-line principle explaining its purpose.",
    "gate_check_summary": "Every blocked_* names upstream/check, exact field, observed value, and expected value.",
    "verifier_is_oracle": "Shared evaluator authority permits circular_positive only, not positive scientific value.",
    "honest_verdict": "Completed findings start complete_ or complete:; external failure starts blocked_. State the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; unchanged external failure is blocked.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes; retain failures.",
    "validation_scope_ready_score": "One only after affected failures remain blocking and unrelated health cannot masquerade as passed validation.",
    "required_checks_passed": "Named required commands must all run and pass.",
    "repository_health": "Preserve prior failure paths, dates, hashes, and unresolved collection errors.",
    "scope_control_rows": "Actual subprocess selections and results prove scope rather than merely labeling it.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class HistoricalEvidence:
    """Name one immutable result and its failed full-suite log."""

    experiment_id: str
    artifact_path: Path
    log_path: Path


HISTORICAL_EVIDENCE = (
    HistoricalEvidence(
        "exp7289-arc-boundary",
        Path("results/experiment_7289_v641_arc_boundary.json"),
        Path("results/raw/experiment_7289_v641_arc_boundary/validation/04_full_python_suite.log"),
    ),
    HistoricalEvidence(
        "exp7291-reuse-fixture",
        Path("results/experiment_7291_v641_reuse_fixture.json"),
        Path("results/raw/experiment_7291_v641_reuse_fixture/validation/full_python_suite.log"),
    ),
    HistoricalEvidence(
        "exp7292-reuse-canary",
        Path("results/experiment_7292_v641_reuse_canary.json"),
        Path("results/raw/experiment_7292_v641_reuse_canary/validation/full_python_suite.log"),
    ),
    HistoricalEvidence(
        "exp7293-reuse-measurement",
        Path("results/experiment_7293_v641_reuse_measurement.json"),
        Path(
            "results/raw/experiment_7293_v641_reuse_measurement/validation-attempt-1/full_python_suite.log"
        ),
    ),
    HistoricalEvidence(
        "exp7294-reuse-audit",
        Path("results/experiment_7294_v641_reuse_audit.json"),
        Path("results/raw/experiment_7294_v641_reuse_audit/validation/full_python_suite.log"),
    ),
)

REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7293_v641_reuse_measurement.py"),
    Path("python/carnot/experiment_7289_v641_arc_boundary.py"),
    Path("python/carnot/experiment_7296_v641_mixture_learning.py"),
    Path("results/experiment_7293_v641_reuse_measurement.json"),
    Path("results/experiment_7289_v641_arc_boundary.json"),
    Path("scripts/check_spec_coverage.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    RUNNER_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    *(item.artifact_path for item in HISTORICAL_EVIDENCE),
    *(item.log_path for item in HISTORICAL_EVIDENCE),
)

CONTROL_SPECS = (
    (
        "affected_test_failure",
        "test_scenario_report_7303_affected_failure_propagates_actual_exit",
        "A required test failure remains a failure with its real exit code.",
    ),
    (
        "unscoped_pytest_target",
        "test_scenario_report_7303_unscoped_pytest_target_is_rejected",
        "A directory-level pytest target cannot enter the command plan.",
    ),
    (
        "missing_expected_command",
        "test_scenario_report_7303_missing_expected_command_cannot_pass",
        "Every named required command must have exactly one receipt.",
    ),
    (
        "unrelated_collection_failure",
        "test_scenario_report_7303_unrelated_history_does_not_become_pass_receipt",
        "Historical repository debt stays separate from current validation.",
    ),
    (
        "clean_scoped_run",
        "test_scenario_report_7303_clean_scoped_run_and_terminal_artifact",
        "A clean explicit run uses actual subprocess arguments and exits.",
    ),
)


def _canonical_bytes(value: Any) -> bytes:
    """Encode stable JSON so hashes bind values instead of formatting choices."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Hash the real file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Seal all terminal content except the field that stores the seal."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON through a same-directory replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(value), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        temporary = Path(out.name)
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def _utc_now() -> str:
    """Return a timezone-aware timestamp for the actual boundary."""

    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, detail: str = "") -> None:
    """Print one flushed phase boundary with monotonic elapsed time."""

    suffix = f" {detail}" if detail else ""
    print(
        f"[exp7303] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def _check(
    upstream: str,
    check: str,
    field: str,
    observed: Any,
    expected: Any,
    passed: bool,
) -> JsonDict:
    """Keep both sides of every decision for a precise blocked verdict."""

    return {
        "upstream": upstream,
        "check": check,
        "field": field,
        "observed_value": observed,
        "expected_value": expected,
        "passed": passed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first failed condition without hiding later failures."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failed_checks": failures,
    }


def check_dependency(
    artifact: Mapping[str, Any] | None,
    upstream: str,
    field: str,
    expected: Any,
) -> JsonDict:
    """Reject an unavailable or unsafe producer before reading its gate value."""

    if artifact is None:
        return _check(upstream, "dependency_gate", field, "missing_artifact", expected, False)
    if artifact.get("flagged_adversarial") is True or artifact.get("quarantined") is True:
        return _check(upstream, "dependency_gate", field, "quarantined", expected, False)
    if artifact.get("verdict_class") == "disqualified":
        return _check(upstream, "dependency_gate", field, "disqualified", expected, False)
    if artifact.get("status") not in {"complete", "completed"}:
        return _check(
            upstream,
            "dependency_gate",
            "status",
            artifact.get("status"),
            "complete",
            False,
        )
    observed = artifact.get(field, "missing_field")
    return _check(upstream, "dependency_gate", field, observed, expected, observed == expected)


_COLLECTION_HEADER = re.compile(r"^_ ERROR collecting (.+?) _+$", re.MULTILINE)


def parse_collection_errors(text: str) -> list[JsonDict]:
    """Extract exact pytest collection blocks without re-running the old suite."""

    matches = list(_COLLECTION_HEADER.finditer(text))
    rows: list[JsonDict] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        block = (
            block.split("=============================== warnings summary", 1)[0].rstrip() + "\n"
        )
        exceptions = [line[4:] for line in block.splitlines() if line.startswith("E   ")]
        rows.append(
            {
                "test_path": match.group(1),
                "exception": exceptions[-1] if exceptions else "collection_error_without_E_line",
                "exact_block": block,
            }
        )
    return rows


def _artifact_time(artifact: Mapping[str, Any]) -> str | None:
    """Read either timestamp layout used by the preserved V641 artifacts."""

    timestamps = artifact.get("timestamps")
    nested = timestamps.get("started_at_utc") if isinstance(timestamps, Mapping) else None
    value = artifact.get("started_at_utc") or nested
    return str(value) if value else None


def authenticate_inputs(
    repo_root: Path,
    *,
    output_path: Path,
) -> tuple[list[JsonDict], JsonDict, list[JsonDict]]:
    """Authenticate current sources and exact historical failure receipts."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    source_hashes: JsonDict = {}
    historical: list[JsonDict] = []
    for relative in dict.fromkeys(REQUIRED_INPUT_PATHS):
        path = root / relative
        available = path.is_file()
        checks.append(
            _check(relative.as_posix(), "input_availability", "is_file", available, True, available)
        )
        if available:
            source_hashes[relative.as_posix()] = {
                "sha256": sha256_file(path),
                "experiment_id": None,
                "terminal_class": "input",
                "quarantined": False,
                "retired": False,
            }
    for item in HISTORICAL_EVIDENCE:
        artifact_path = root / item.artifact_path
        log_path = root / item.log_path
        if not artifact_path.is_file() or not log_path.is_file():
            continue
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            checks.append(
                _check(
                    item.artifact_path.as_posix(),
                    "historical_artifact_parse",
                    "json_object",
                    type(exc).__name__,
                    "readable_json_object",
                    False,
                )
            )
            continue
        receipts = [
            row
            for row in artifact.get("validation_receipts") or []
            if isinstance(row, Mapping) and row.get("name") == "full_python_suite"
        ]
        receipt = receipts[0] if len(receipts) == 1 else {}
        observed_hash = sha256_file(log_path)
        expected_hash = receipt.get("log_sha256", "missing_receipt")
        hash_matches = len(receipts) == 1 and observed_hash == expected_hash
        checks.append(
            _check(
                item.log_path.as_posix(),
                "historical_full_suite_log_authentication",
                "sha256",
                observed_hash,
                expected_hash,
                hash_matches,
            )
        )
        focused = [
            {
                "name": row.get("name"),
                "command": row.get("command"),
                "exit_code": row.get("exit_code"),
                "passed": row.get("passed"),
                "log_path": row.get("log_path"),
                "log_sha256": row.get("log_sha256"),
            }
            for row in artifact.get("validation_receipts") or []
            if isinstance(row, Mapping)
            and (
                "focused" in str(row.get("name"))
                or "affected" in str(row.get("name"))
                or str(row.get("name", "")).startswith("e2e_")
            )
        ]
        text = log_path.read_text(encoding="utf-8", errors="replace")
        historical.append(
            {
                "experiment_id": artifact.get("experiment_id", item.experiment_id),
                "artifact_path": item.artifact_path.as_posix(),
                "artifact_sha256": sha256_file(artifact_path),
                "milestone": artifact.get("milestone"),
                "observed_at_utc": _artifact_time(artifact),
                "predates_current_milestone": artifact.get("milestone") != MILESTONE,
                "terminal_class": artifact.get("verdict_class"),
                "honest_verdict": artifact.get("honest_verdict"),
                "quarantined": artifact.get("flagged_adversarial") is True,
                "command": receipt.get("command"),
                "exit_code": receipt.get("exit_code"),
                "log_path": item.log_path.as_posix(),
                "log_sha256": observed_hash,
                "focused_checks": focused,
                "collection_errors": parse_collection_errors(text),
                "resolved": False,
                "current_required_check": False,
            }
        )
        if item.artifact_path.as_posix() in source_hashes:
            source_hashes[item.artifact_path.as_posix()].update(
                {
                    "experiment_id": artifact.get("experiment_id", item.experiment_id),
                    "terminal_class": artifact.get("verdict_class"),
                    "quarantined": artifact.get("flagged_adversarial") is True,
                }
            )
        if item.log_path.as_posix() in source_hashes:
            source_hashes[item.log_path.as_posix()].update(
                {
                    "experiment_id": artifact.get("experiment_id", item.experiment_id),
                    "terminal_class": "failed_validation_log",
                    "quarantined": artifact.get("flagged_adversarial") is True,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writable = os.access(output_path.parent, os.W_OK)
    checks.append(
        _check(
            str(output_path),
            "declared_output_path",
            "parent_writable",
            writable,
            True,
            writable,
        )
    )
    return checks, source_hashes, historical


def _gate(criterion: str, expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Keep expected and observed values beside one control decision."""

    return {
        "criterion": criterion,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _scope_rows(validation: Mapping[str, Any]) -> list[JsonDict]:
    """Bind each required behavioral control to the real focused command."""

    focused = next(
        (
            row
            for row in validation.get("validation_receipts") or []
            if row.get("name") == "focused_pytest"
        ),
        {},
    )
    passed = focused.get("passed") is True and focused.get("exit_code") == 0
    share = float(focused.get("duration_s", 0.0)) / len(CONTROL_SPECS)
    return [
        {
            "control_id": control_id,
            "test_node": f"{TEST_PATH.as_posix()}::{test_name}",
            "actual_subprocess_argv": list(focused.get("command_argv") or []),
            "actual_exit_code": focused.get("exit_code"),
            "metrics": {"control_passed": passed},
            "costs": {"allocated_validation_s": share},
            "errors": [] if passed else ["focused_pytest_failed"],
            "abstentions": 0,
            "censored": False,
            "passed": passed,
            "principle": principle,
        }
        for control_id, test_name, principle in CONTROL_SPECS
    ]


def _acceptance_gates(
    validation: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    terminal_receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Score current scope controls without using repository health as a gate."""

    receipts = validation.get("validation_receipts") or []
    broad_targets = [
        row.get("name") for row in receipts if "tests/python" in (row.get("command_argv") or [])
    ]
    import_row = next((row for row in receipts if row.get("name") == "worktree_imports"), {})
    terminal_passed = len(terminal_receipts) == 2 and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in terminal_receipts
    )
    return [
        _gate(
            "required_commands_complete",
            True,
            validation.get("required_checks_passed"),
            validation.get("required_checks_passed") is True,
            "Every named current command must run once and pass.",
        ),
        _gate(
            "repository_wide_pytest_absent",
            [],
            broad_targets,
            not broad_targets,
            "A directory-level suite cannot return through a hidden default.",
        ),
        _gate(
            "five_scope_controls_pass",
            len(CONTROL_SPECS),
            sum(row.get("passed") is True for row in rows),
            len(rows) == len(CONTROL_SPECS) and all(row.get("passed") is True for row in rows),
            "Each incident control must be exercised by the focused test command.",
        ),
        _gate(
            "worktree_imports_resolved",
            True,
            import_row.get("passed"),
            import_row.get("passed") is True and bool(import_row.get("resolved_imports")),
            "Resolved worktree paths prevent an installed wheel from substituting old code.",
        ),
        _gate(
            "historical_health_is_non_gating",
            False,
            validation.get("repository_health", {}).get("affects_required_checks"),
            validation.get("repository_health", {}).get("affects_required_checks") is False,
            "Old collection debt remains visible without becoming a current pass receipt.",
        ),
        _gate(
            "terminal_candidate_verifiers",
            True,
            terminal_passed,
            terminal_passed,
            "Strict artifact checks must inspect the measured candidate before publication.",
        ),
    ]


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    historical: Sequence[Mapping[str, Any]],
    *,
    started_at: str,
) -> JsonDict:
    """Create every required field before choosing blocked or complete status."""

    health = scoped.build_repository_health(historical)
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_durations_s": {},
        "random_seed": {
            "development": 7_303_202_609_14,
            "independent_evaluation": 17_303_202_609_14,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "sample_size_budget": {
            "planned_controls": len(CONTROL_SPECS),
            "attempted_controls": 0,
            "complete_controls": 0,
            "censored_controls": len(CONTROL_SPECS),
            "stopping_rule": "run each of five frozen validation controls once; do not add a repository-wide pytest target",
            "outcome_based_extension": False,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_validation_evidence_unavailable",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "validation_scope_ready_score": 0,
        "required_checks_passed": False,
        "repository_health": health,
        "scope_control_rows": [],
        "resolved_imports": {},
        "historical_evidence_sidecar": {},
        "missing_required_commands": list(scoped.REQUIRED_CHECK_NAMES),
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "production_default_changed": False,
    }


def _blocked_artifact(base: JsonDict, *, duration_s: float, phase_durations: JsonDict) -> JsonDict:
    """Finish an external precondition failure with its exact first check."""

    artifact = deepcopy(base)
    first = artifact["gate_check_summary"]["first_failure"]
    if first is not None:
        artifact["honest_verdict"] = (
            "blocked_external_validation_evidence:"
            f"{first['upstream']}:{first['check']}:{first['field']}:"
            f"observed={first['observed_value']!r}:expected={first['expected_value']!r}"
        )
    artifact["duration_s"] = duration_s
    artifact["phase_durations_s"] = phase_durations
    artifact["completed_at_utc"] = _utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _complete_artifact(
    base: JsonDict,
    validation: Mapping[str, Any],
    terminal_receipts: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    phase_durations: JsonDict,
    sidecar: Mapping[str, Any],
) -> JsonDict:
    """Build one terminal outcome while keeping failed current checks visible."""

    artifact = deepcopy(base)
    rows = _scope_rows(validation)
    gates = _acceptance_gates(validation, rows, terminal_receipts)
    required = validation.get("required_checks_passed") is True
    ready = required and all(row.get("passed") is True for row in gates)
    artifact.update(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "duration_s": duration_s,
            "phase_durations_s": deepcopy(phase_durations),
            "rows": deepcopy(rows),
            "scope_control_rows": deepcopy(rows),
            "sample_size_budget": {
                "planned_controls": len(CONTROL_SPECS),
                "attempted_controls": len(rows),
                "complete_controls": sum(row.get("passed") is True for row in rows),
                "censored_controls": sum(row.get("censored") is True for row in rows),
                "stopping_rule": "run each of five frozen validation controls once; do not add a repository-wide pytest target",
                "outcome_based_extension": False,
            },
            "acceptance_gate_results": gates,
            "validation_receipts": [
                *(dict(row) for row in validation.get("validation_receipts") or []),
                *(dict(row) for row in terminal_receipts),
            ],
            "required_checks_passed": required,
            "repository_health": deepcopy(validation["repository_health"]),
            "validation_scope_ready_score": int(ready),
            "resolved_imports": deepcopy(
                next(
                    (
                        row.get("resolved_imports")
                        for row in validation.get("validation_receipts") or []
                        if row.get("name") == "worktree_imports"
                    ),
                    {},
                )
            ),
            "historical_evidence_sidecar": deepcopy(dict(sidecar)),
            "missing_required_commands": list(validation.get("missing_required_commands") or []),
            "failed_required_commands": list(validation.get("failed_required_commands") or []),
            "duplicate_required_commands": list(
                validation.get("duplicate_required_commands") or []
            ),
        }
    )
    if ready:
        artifact["honest_verdict"] = (
            "complete_scoped_validation_ready_with_historical_repository_health_open"
        )
        artifact["verdict_class"] = "circular_positive"
    else:
        artifact["honest_verdict"] = "complete_disqualified_required_scope_validation_failed"
        artifact["verdict_class"] = "disqualified"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, separation, terminal semantics, and the content seal."""

    errors: list[str] = []

    def add(condition: bool, error: str) -> None:
        if condition and error not in errors:
            errors.append(error)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
    add(artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False, "model")
    add(artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS, "invocation_counts")
    add(
        artifact.get("current_model_load_count") != 0
        or artifact.get("current_generation_count") != 0,
        "current_counts",
    )
    add(
        artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("execution_venue") != "host",
        "substrate",
    )
    add(artifact.get("verifier_is_oracle") is not True, "oracle")
    add(artifact.get("reproducibility_checksum") != artifact_checksum(artifact), "checksum")
    health = artifact.get("repository_health")
    add(
        not isinstance(health, Mapping)
        or health.get("affects_required_checks") is not False
        or not isinstance(health.get("historical_failures"), list),
        "repository_health",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("validation_scope_ready_score") != 0
            or artifact.get("gate_check_summary", {}).get("passed") is not False,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") != "complete", "status")
    rows = artifact.get("rows")
    add(
        not isinstance(rows, list)
        or rows != artifact.get("scope_control_rows")
        or len(rows) != len(CONTROL_SPECS)
        or any(
            not {"metrics", "costs", "errors", "abstentions", "censored", "passed"} <= set(row)
            for row in rows
        ),
        "scope_control_rows",
    )
    receipts = artifact.get("validation_receipts")
    add(
        not isinstance(receipts, list)
        or any(
            not {
                "name",
                "command",
                "command_argv",
                "scope",
                "exit_code",
                "duration_s",
                "log_sha256",
            }
            <= set(row)
            for row in receipts
        ),
        "validation_receipts",
    )
    required = scoped.reduce_required_checks(receipts if isinstance(receipts, list) else [])
    add(
        artifact.get("required_checks_passed") is not required["required_checks_passed"], "required"
    )
    gates = artifact.get("acceptance_gate_results")
    add(
        not isinstance(gates, list)
        or any(not {"expected", "observed", "passed", "principle"} <= set(row) for row in gates),
        "acceptance_gates",
    )
    expected_ready = int(
        required["required_checks_passed"]
        and isinstance(gates, list)
        and bool(gates)
        and all(row.get("passed") is True for row in gates)
    )
    add(artifact.get("validation_scope_ready_score") != expected_ready, "ready_score")
    add(
        expected_ready == 1
        and (
            artifact.get("verdict_class") != "circular_positive"
            or not str(artifact.get("honest_verdict", "")).startswith("complete_")
        ),
        "ready_verdict",
    )
    add(
        expected_ready == 0 and artifact.get("verdict_class") != "disqualified",
        "failed_verdict",
    )
    return errors


def _terminal_commands(root: Path, candidate: Path) -> list[scoped.CommandSpec]:
    """Return strict checks for the measured candidate only."""

    python = str(root / ".venv/bin/python")
    return [
        scoped.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_terminal_candidate",
            300.0,
        ),
        scoped.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_terminal_candidate",
            300.0,
        ),
    ]


def run_experiment(
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path | None = None,
    raw_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    test_paths: Sequence[str] = DEFAULT_TEST_PATHS,
    changed_modules: Sequence[str] = DEFAULT_CHANGED_MODULES,
    static_paths: Sequence[str] = DEFAULT_STATIC_PATHS,
    extra_env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Authenticate history, run current scope, verify, and publish once."""

    if run_date != RUN_DATE:
        raise ValueError(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path or root / RESULT_PATH
    raw = raw_dir or root / RAW_DIR
    checkpoint = checkpoint_path or root / CHECKPOINT_PATH
    started = time.monotonic()
    started_at = _utc_now()
    spans: JsonDict = {}
    _progress(started, "startup", "entrypoint_and_paths_authenticated")
    phase = time.monotonic()
    _progress(started, "preconditions", "before_authentication")
    checks, source_hashes, historical = authenticate_inputs(root, output_path=output)
    spans["preconditions"] = time.monotonic() - phase
    _progress(
        started,
        "preconditions",
        "after_authentication",
        f"passed={gate_summary(checks)['passed']} checks={len(checks)}",
    )
    base = _base_artifact(checks, source_hashes, historical, started_at=started_at)
    if gate_summary(checks)["passed"] is not True:
        blocked = _blocked_artifact(
            base, duration_s=time.monotonic() - started, phase_durations=spans
        )
        errors = validate_artifact(blocked)
        if errors:
            raise ValueError("blocked_artifact_invalid:" + ",".join(errors))
        _progress(started, "publication", "before_blocked_atomic_write")
        atomic_write(output, blocked)
        atomic_write(
            checkpoint,
            {
                "schema": "carnot.validation_scope.checkpoint.v1",
                "status": "complete_blocked",
                "result_path": str(output),
            },
        )
        _progress(started, "publication", "after_blocked_atomic_write", str(output))
        return blocked
    raw.mkdir(parents=True, exist_ok=True)
    sidecar_path = raw / "historical_validation_evidence.json"
    sidecar_value = {
        "schema": "carnot.validation_scope.historical_evidence.v1",
        "current_model_invocations": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_failures": historical,
    }
    atomic_write(sidecar_path, sidecar_value)
    sidecar_key = (
        sidecar_path.relative_to(root).as_posix()
        if sidecar_path.is_relative_to(root)
        else str(sidecar_path)
    )
    base["source_artifact_hashes"][sidecar_key] = {
        "sha256": sha256_file(sidecar_path),
        "experiment_id": EXPERIMENT_ID,
        "terminal_class": "historical_evidence_sidecar",
        "quarantined": False,
        "retired": False,
    }
    sidecar_receipt = {"path": sidecar_key, "sha256": sha256_file(sidecar_path)}
    atomic_write(
        checkpoint,
        {
            "schema": "carnot.validation_scope.checkpoint.v1",
            "status": "running",
            "stage": "historical_evidence_authenticated",
            "sidecar": sidecar_receipt,
        },
    )
    phase = time.monotonic()
    _progress(started, "scoped_validation", "before_command_set")
    basetemp = Path(tempfile.mkdtemp(prefix="carnot-exp7303-"))
    validation = scoped.run_scoped_validation(
        root,
        test_paths,
        changed_modules,
        static_paths=static_paths,
        basetemp=basetemp,
        coverage_file=raw / ".coverage",
        log_dir=raw / "validation",
        historical_failures=historical,
        extra_env=extra_env,
    )
    spans["scoped_validation"] = time.monotonic() - phase
    _progress(
        started,
        "scoped_validation",
        "after_command_set",
        f"required_checks_passed={validation['required_checks_passed']}",
    )
    candidate_path = raw / "measured-terminal-candidate.json"
    provisional = _complete_artifact(
        base,
        validation,
        [],
        duration_s=time.monotonic() - started,
        phase_durations=spans,
        sidecar=sidecar_receipt,
    )
    atomic_write(candidate_path, provisional)
    phase = time.monotonic()
    _progress(started, "terminal_validation", "before_strict_candidate_checks")
    terminal = scoped.run_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw / "validation",
        extra_env=extra_env,
    )
    spans["terminal_validation"] = time.monotonic() - phase
    _progress(
        started,
        "terminal_validation",
        "after_strict_candidate_checks",
        f"passed={all(row['passed'] for row in terminal)}",
    )
    final = _complete_artifact(
        base,
        validation,
        terminal,
        duration_s=time.monotonic() - started,
        phase_durations=spans,
        sidecar=sidecar_receipt,
    )
    errors = validate_artifact(final)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    _progress(started, "publication", "before_terminal_atomic_write")
    atomic_write(output, final)
    atomic_write(
        checkpoint,
        {
            "schema": "carnot.validation_scope.checkpoint.v1",
            "status": "complete",
            "result_path": str(output),
            "result_sha256": sha256_file(output),
        },
    )
    _progress(
        started,
        "publication",
        "after_terminal_atomic_write",
        f"path={output} validation_scope_ready_score={final['validation_scope_ready_score']}",
    )
    return final


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only the fixed execution date."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the host control with immediate flushed progress."""

    print("[exp7303] phase=startup event=entrypoint", flush=True)
    args = _parse_args(argv)
    artifact = run_experiment(REPO_ROOT, args.date)
    return int(artifact["status"] != "complete")


if __name__ == "__main__":  # pragma: no cover - the thin wrapper is the public path.
    raise SystemExit(main())

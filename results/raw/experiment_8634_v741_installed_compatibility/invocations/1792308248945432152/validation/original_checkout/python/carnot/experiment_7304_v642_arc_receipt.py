"""Emit a CPU-only receipt for the shipped ARC lifecycle boundary.

Model-shaped fixture events are useful evidence, but they are not current model
work. This module keeps the exact event bytes in hashed sidecars. The terminal
record contains only control outcomes and references to those sidecars.

Spec refs: REQ-ARC-WMTE-7304 and SCENARIO-ARC-WMTE-7304-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot.experiment_7289_v641_arc_boundary import (
    exercise_live_caller_seams,
    run_cpu_boundary_panel,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260914"
MILESTONE = "2026.09.642"
EXPERIMENT_ID = "exp7304-arc-receipt"
SCHEMA = "carnot.experiment_7304.v642.arc_receipt.v1"
RESULT_PATH = Path("results/experiment_7304_v642_arc_receipt.json")
RAW_DIR = Path("results/raw/experiment_7304_v642_arc_receipt")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7304_v642_arc_receipt.json")
MODULE_PATH = Path("python/carnot/experiment_7304_v642_arc_receipt.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7304_v642_arc_receipt.py")
TEST_PATH = Path("tests/python/test_experiment_7304_v642_arc_receipt.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP7289_PATH = Path("results/experiment_7289_v641_arc_boundary.json")
EXP7289_ADVERSARIAL_LOG = Path(
    "results/raw/experiment_7289_v641_arc_boundary/validation/"
    "00_terminal_candidate_adversarial_verify.log"
)
EXP7289_REPOSITORY_HEALTH_LOG = Path(
    "results/raw/experiment_7289_v641_arc_boundary/validation/04_full_python_suite.log"
)
EXP7303_PATH = Path("results/experiment_7303_v642_validation_scope.json")

MODEL_SPECS: list[JsonDict] = []
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

NONTERMINAL_VALIDATION_NAMES = (
    "worktree_imports",
    "focused_exp7304",
    "affected_arc_eval_provenance",
    "e2e_009_cross_call_persistence",
    "e2e_010_tool_transport",
    "e2e_009_llm_off_environment_smoke",
    "changed_module_coverage",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
)
TERMINAL_VALIDATION_NAMES = (
    "terminal_candidate_adversarial_verify",
    "terminal_candidate_row_consistency_strict",
)
REQUIRED_VALIDATION_NAMES = NONTERMINAL_VALIDATION_NAMES + TERMINAL_VALIDATION_NAMES

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
    "arc_receipt_ready_score": "One only for an unquarantined emitter and reachable live wrapper with passing lifecycle controls.",
    "fixture_sidecar_manifest": "Fixture event bytes remain auditable without claiming current model work.",
    "caller_handoff": "Exact caller paths, arguments, and code hashes bind the following live invocation.",
    "lifecycle_control_rows": "Every failure boundary and its true completed/in-flight counts must be represented.",
}

REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7289_v641_arc_boundary.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_inference_boundary.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/agentic/arc_induction_tool_loop.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("tests/python/test_arc_eval_provenance_contract_20260905.py"),
    Path("tests/python/test_arc_induction_state_persistence.py"),
    Path("tests/python/test_arc_tool_grammar_transport.py"),
    SPEC_PATH,
    EXP7289_PATH,
    EXP7289_ADVERSARIAL_LOG,
    EXP7289_REPOSITORY_HEALTH_LOG,
    EXP7303_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

_EXPECTED_HISTORICAL_FLAGS = (
    "INFERENCE_PROVENANCE_CONTRADICTION",
    "SUBSTRATE_CLASS_MISMATCH",
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    """Encode stable JSON so the content seal does not depend on key order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every terminal field except the field that stores this hash."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON with a same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        temporary = Path(out.name)
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def _utc_now() -> str:
    """Return an actual timezone-aware UTC phase boundary."""

    return datetime.now(UTC).isoformat(timespec="microseconds")


def _progress(started: float, phase: str, event: str, detail: str = "") -> None:
    """Emit one flushed boundary with monotonic elapsed time."""

    suffix = f" {detail}" if detail else ""
    print(
        f"[exp7304] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
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
    """Keep both sides of a precondition decision for a precise blocked result."""

    return {
        "upstream": upstream,
        "check": check,
        "field": field,
        "observed_value": observed,
        "expected_value": expected,
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose all failed checks and retain the exact first failure."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failed_checks": failures,
    }


def check_dependency(
    artifact: Mapping[str, Any] | None,
    *,
    upstream: str,
    field: str,
    expected: Any,
) -> JsonDict:
    """Reject unsafe upstream state before reading its numeric readiness value."""

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


def _load_json_object(path: Path) -> JsonDict | None:
    """Read an upstream JSON object without turning parse failure into an empty success."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def authenticate_inputs(
    repo_root: Path,
    *,
    output_path: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Hash real inputs, preserve prior flags, and gate Exp7303 before use."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in dict.fromkeys(REQUIRED_INPUT_PATHS):
        path = root / relative
        available = path.is_file()
        checks.append(
            _check(relative.as_posix(), "input_availability", "is_file", available, True, available)
        )
        if available:
            hashes[relative.as_posix()] = {
                "sha256": sha256_file(path),
                "terminal_class": "input",
                "quarantined": False,
                "retired": False,
            }

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    for identifier in (
        "REQ-ARC-WMTE-7304",
        "SCENARIO-ARC-WMTE-7304-DEPENDENCY-BLOCK",
        "SCENARIO-ARC-WMTE-7304-LIFECYCLE-CONTROLS",
        "SCENARIO-ARC-WMTE-7304-SIDECAR-INTEGRITY",
        "SCENARIO-ARC-WMTE-7304-CURRENT-PROVENANCE",
        "SCENARIO-ARC-WMTE-7304-CALLER-HANDOFF",
    ):
        checks.append(
            _check(
                SPEC_PATH.as_posix(),
                "driving_spec",
                identifier,
                identifier in spec_text,
                True,
                identifier in spec_text,
            )
        )

    dependency = _load_json_object(root / EXP7303_PATH)
    checks.append(
        check_dependency(
            dependency,
            upstream=EXP7303_PATH.as_posix(),
            field="validation_scope_ready_score",
            expected=1,
        )
    )

    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = (
        "exp7304-arc-receipt" in exclusion_text or "exp7303-validation-scope" in exclusion_text
    )
    checks.append(
        _check(
            "ops/exclusion_manifest.yaml",
            "dependency_exclusion_state",
            "exp7303_or_exp7304_listed",
            excluded,
            False,
            not excluded,
        )
    )

    original = _load_json_object(root / EXP7289_PATH)
    log_text = (
        (root / EXP7289_ADVERSARIAL_LOG).read_text(encoding="utf-8", errors="replace")
        if (root / EXP7289_ADVERSARIAL_LOG).is_file()
        else ""
    )
    receipt = next(
        (
            row
            for row in (original or {}).get("validation_receipts", [])
            if isinstance(row, Mapping)
            and row.get("name") == "terminal_candidate_adversarial_verify"
        ),
        {},
    )
    observed_log_hash = (
        sha256_file(root / EXP7289_ADVERSARIAL_LOG)
        if (root / EXP7289_ADVERSARIAL_LOG).is_file()
        else None
    )
    checks.append(
        _check(
            EXP7289_ADVERSARIAL_LOG.as_posix(),
            "historical_adversarial_log_authentication",
            "sha256",
            observed_log_hash,
            receipt.get("log_sha256", "missing_receipt"),
            bool(observed_log_hash and observed_log_hash == receipt.get("log_sha256")),
        )
    )
    observed_flags = [flag for flag in _EXPECTED_HISTORICAL_FLAGS if flag in log_text]
    checks.append(
        _check(
            EXP7289_ADVERSARIAL_LOG.as_posix(),
            "historical_flag_preservation",
            "flag_kinds",
            observed_flags,
            list(_EXPECTED_HISTORICAL_FLAGS),
            observed_flags == list(_EXPECTED_HISTORICAL_FLAGS),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writable = os.access(output_path.parent, os.W_OK)
    checks.append(
        _check(
            str(output_path), "declared_output_path", "parent_writable", writable, True, writable
        )
    )

    if EXP7289_PATH.as_posix() in hashes:
        hashes[EXP7289_PATH.as_posix()].update(
            {
                "experiment_id": (original or {}).get("experiment_id"),
                "terminal_class": (original or {}).get("verdict_class"),
                "quarantined": True,
            }
        )
    if EXP7289_ADVERSARIAL_LOG.as_posix() in hashes:
        hashes[EXP7289_ADVERSARIAL_LOG.as_posix()].update(
            {"terminal_class": "historical_adversarial_flags", "quarantined": True}
        )
    if EXP7303_PATH.as_posix() in hashes:
        hashes[EXP7303_PATH.as_posix()].update(
            {
                "experiment_id": (dependency or {}).get("experiment_id"),
                "terminal_class": (dependency or {}).get("verdict_class"),
                "quarantined": bool(
                    (dependency or {}).get("flagged_adversarial")
                    or (dependency or {}).get("quarantined")
                ),
            }
        )
    historical = {
        "original_artifact": {
            "path": EXP7289_PATH.as_posix(),
            "sha256": hashes.get(EXP7289_PATH.as_posix(), {}).get("sha256"),
            "size_bytes": (root / EXP7289_PATH).stat().st_size
            if (root / EXP7289_PATH).is_file()
            else None,
        },
        "original_adversarial_log": {
            "path": EXP7289_ADVERSARIAL_LOG.as_posix(),
            "sha256": observed_log_hash,
            "size_bytes": (root / EXP7289_ADVERSARIAL_LOG).stat().st_size
            if (root / EXP7289_ADVERSARIAL_LOG).is_file()
            else None,
            "flag_kinds": observed_flags,
        },
    }
    return checks, hashes, historical


def run_cpu_receipt_panel(workdir: Path) -> JsonDict:
    """Reuse the shipped six-case ledger panel without invoking a model."""

    return run_cpu_boundary_panel(workdir)


def _relative_or_absolute(path: Path, root: Path) -> str:
    """Use stable repository paths while private test sidecars remain readable."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def verify_sidecar(
    path: Path,
    *,
    expected_sha256: str,
    expected_event_count: int,
) -> JsonDict:
    """Require both exact bytes and the frozen number of JSONL events."""

    if not path.is_file():
        return {"passed": False, "reason": "missing_sidecar"}
    if sha256_file(path) != expected_sha256:
        return {"passed": False, "reason": "sha256_mismatch"}
    event_count = len(path.read_bytes().splitlines())
    if event_count != expected_event_count:
        return {"passed": False, "reason": "event_count_mismatch"}
    return {"passed": True, "reason": "hash_and_event_count_match"}


def project_lifecycle_panel(
    panel: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Project exact event payloads into CPU outcomes plus sidecar references."""

    rows: list[JsonDict] = []
    manifest: list[JsonDict] = []
    for index, raw in enumerate(panel.get("rows", [])):
        if not isinstance(raw, Mapping):
            continue
        path = Path(str(raw.get("ledger_path", "")))
        expected_hash = str(raw.get("ledger_sha256", ""))
        event_count = int(raw.get("persisted_event_count", 0))
        integrity = verify_sidecar(
            path,
            expected_sha256=expected_hash,
            expected_event_count=event_count,
        )
        reduction = raw.get("counter_reduction")
        reduction = reduction if isinstance(reduction, Mapping) else {}
        call_rows = reduction.get("call_rows")
        call_rows = call_rows if isinstance(call_rows, list) else []
        state_counts = {"completed": 0, "failed": 0, "in_flight": 0}
        for call in call_rows:
            if isinstance(call, Mapping) and call.get("terminal_state") in state_counts:
                state_counts[str(call["terminal_state"])] += 1
        manifest.append(
            {
                "case": raw.get("case"),
                "path": _relative_or_absolute(path, root),
                "sha256": expected_hash,
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "event_count": event_count,
                "integrity_verified": integrity["passed"],
                "provenance_scope": "historical_fixture",
            }
        )
        passed = bool(
            raw.get("passed") is True
            and integrity["passed"] is True
            and raw.get("owned_child_alive_after_cleanup") is False
        )
        rows.append(
            {
                "case": raw.get("case"),
                "seed": raw.get("seed"),
                "metrics": {"control_passed": passed},
                "costs": {
                    "cpu_wall_s": raw.get("cost", {}).get("wall_s")
                    if isinstance(raw.get("cost"), Mapping)
                    else None
                },
                "errors": [] if passed else [str(raw.get("error") or integrity["reason"])],
                "abstentions": 0,
                "censored": bool(raw.get("censored")),
                "passed": passed,
                "receipt_state_counts": state_counts,
                "duplicate_event_count": reduction.get("duplicate_event_count"),
                "identity_rejection_preserved": raw.get("identity_rejection_preserved"),
                "owned_child_alive_after_cleanup": raw.get("owned_child_alive_after_cleanup"),
                "fixture_sidecar_reference": index,
                "provenance_scope": "historical_fixture",
            }
        )
    return rows, manifest


def run_sidecar_negative_controls(source: Path, *, output_dir: Path) -> list[JsonDict]:
    """Prove that byte tampering and a lost event both fail closed."""

    output_dir.mkdir(parents=True, exist_ok=True)
    original = source.read_bytes()
    original_count = len(original.splitlines())

    tampered_path = output_dir / "hash_tampering.jsonl"
    tampered_path.write_bytes(original + b" ")
    tampered = verify_sidecar(
        tampered_path,
        expected_sha256=sha256_file(source),
        expected_event_count=original_count,
    )

    lines = original.splitlines(keepends=True)
    missing_path = output_dir / "missing_event.jsonl"
    missing_path.write_bytes(b"".join(lines[:-1]))
    missing = verify_sidecar(
        missing_path,
        expected_sha256=sha256_file(missing_path),
        expected_event_count=original_count,
    )
    return [
        {
            "control": "hash_tampering",
            "path": str(tampered_path),
            "rejected": tampered["passed"] is False,
            "reason": tampered["reason"],
            "passed": tampered["passed"] is False,
        },
        {
            "control": "missing_event",
            "path": str(missing_path),
            "rejected": missing["passed"] is False,
            "reason": missing["reason"],
            "passed": missing["passed"] is False,
        },
    ]


def build_caller_handoff(repo_root: Path, seams: Mapping[str, Any]) -> JsonDict:
    """Bind the actual caller seams and exact Exp7305 selfparse arguments."""

    code_paths = (
        Path("python/carnot/agentic/arc_inference_boundary.py"),
        Path("python/carnot/agentic/arc_executable_world_model.py"),
        Path("python/carnot/agentic/arc_induction_tool_loop.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
    )
    handoff: JsonDict = {
        "entrypoint": "scripts/experiments/experiment_7305_v642_arc_selfparse.py",
        "arguments": ["--date", RUN_DATE],
        "environment": {
            "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
            "CARNOT_ARC_BOUNDARY_LEDGER_PATH": (
                "results/raw/experiment_7305_v642_arc_selfparse/receipt_events.jsonl"
            ),
        },
        "load_caller": (
            "carnot.agentic.arc_executable_world_model.LocalGGUFProposer._ensure_server"
        ),
        "selfparse_caller": "carnot.agentic.arc_induction_tool_loop._post_chat",
        "selfparse_call_arguments": {
            "messages": "current frozen tool-loop messages",
            "turn": "current zero-based tool-loop turn",
            "timeout_s": "minimum of proposer timeout and remaining bounded session time",
            "selfparse": True,
            "tools_payload": None,
            "grammar": None,
        },
        "caller_reachability": {
            "load_boundary_reachable": seams.get("load_boundary_reachable"),
            "selfparse_transport_reachable": seams.get("selfparse_generation_boundary_reachable"),
            "identity_rejection_preserved": seams.get("identity_rejection_preserved"),
        },
        "caller_code_hashes": {
            path.as_posix(): sha256_file(repo_root / path) for path in code_paths
        },
        "production_default_changed": False,
    }
    handoff["handoff_sha256"] = "sha256:" + hashlib.sha256(_canonical_bytes(handoff)).hexdigest()
    return handoff


def _gate(criterion: str, expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Keep one gate decision machine-readable and self-explanatory."""

    return {
        "criterion": criterion,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _validation_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require every named command once with a real successful exit."""

    names = [str(row.get("name")) for row in receipts]
    return bool(
        len(names) == len(set(names))
        and set(REQUIRED_VALIDATION_NAMES).issubset(names)
        and all(
            row.get("passed") is True and row.get("exit_code") == 0
            for row in receipts
            if str(row.get("name")) in REQUIRED_VALIDATION_NAMES
        )
    )


def _base_artifact(
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
) -> JsonDict:
    """Create required ordinary fields before selecting a terminal outcome."""

    return {
        "schema": SCHEMA,
        "status": "blocked",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_durations_s": {},
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "random_seed": {
            "development": 7_304_202_609_14,
            "independent_evaluation": 17_304_202_609_14,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "rows": [],
        "sample_size_budget": {
            "planned_units": 6,
            "attempted_units": 0,
            "complete_units": 0,
            "censored_units": 6,
            "stopping_rule": "run each frozen CPU lifecycle control once; do not extend from outcomes",
            "outcome_based_extension": False,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary(preconditions_checked),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_receipt_dependency_unavailable",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "arc_receipt_ready_score": 0,
        "fixture_sidecar_manifest": [],
        "caller_handoff": {},
        "lifecycle_control_rows": [],
        "sidecar_control_rows": [],
        "historical_evidence_sidecars": {},
        "repository_health": {
            "affects_required_checks": False,
            "historical_failures": [],
        },
        "production_default_changed": False,
        "historical_quarantine_preserved": True,
        "official_score": None,
    }


def build_blocked_artifact(
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
) -> JsonDict:
    """Publish external absence as a terminal blocked result with exact context."""

    artifact = _base_artifact(
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        duration_s=duration_s,
        preconditions_checked=preconditions_checked,
        source_artifact_hashes=source_artifact_hashes,
    )
    first = artifact["gate_check_summary"]["first_failure"]
    if first is not None:
        artifact["honest_verdict"] = (
            "blocked_external_receipt_dependency:"
            f"{first['upstream']}:{first['check']}:{first['field']}:"
            f"observed={first['observed_value']!r}:expected={first['expected_value']!r}"
        )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_terminal_artifact(
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_durations_s: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    fixture_sidecar_manifest: Sequence[Mapping[str, Any]],
    sidecar_control_rows: Sequence[Mapping[str, Any]],
    caller_handoff: Mapping[str, Any],
    historical_evidence_sidecars: Mapping[str, Any],
    repository_health: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal CPU receipt without copying invocation payloads into it."""

    artifact = _base_artifact(
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        duration_s=duration_s,
        preconditions_checked=preconditions_checked,
        source_artifact_hashes=source_artifact_hashes,
    )
    rows = [dict(row) for row in lifecycle_rows]
    manifest = [dict(row) for row in fixture_sidecar_manifest]
    controls = [dict(row) for row in sidecar_control_rows]
    reachability = caller_handoff.get("caller_reachability")
    reachability = reachability if isinstance(reachability, Mapping) else {}
    lifecycle_passed = len(rows) == 6 and all(row.get("passed") is True for row in rows)
    sidecars_passed = len(manifest) == 6 and all(
        row.get("integrity_verified") is True for row in manifest
    )
    negative_controls_passed = len(controls) == 2 and all(
        row.get("passed") is True and row.get("rejected") is True for row in controls
    )
    no_child = bool(rows) and all(
        row.get("owned_child_alive_after_cleanup") is False for row in rows
    )
    callers_passed = all(
        reachability.get(field) is True
        for field in (
            "load_boundary_reachable",
            "selfparse_transport_reachable",
            "identity_rejection_preserved",
        )
    )
    dependencies_passed = all(row.get("passed") is True for row in preconditions_checked)
    validation_passed = _validation_complete(validation_receipts)
    historical_flags = (
        historical_evidence_sidecars.get("original_adversarial_log", {}).get("flag_kinds", [])
        if isinstance(historical_evidence_sidecars.get("original_adversarial_log"), Mapping)
        else []
    )
    quarantine_preserved = list(historical_flags) == list(_EXPECTED_HISTORICAL_FLAGS)
    current_projection_clean = not any(
        token in json.dumps({"rows": rows, "manifest": manifest})
        for token in (
            "model_identity",
            "call_rows",
            "generation_calls_attempted",
            "model_loads_attempted",
        )
    )
    gates = [
        _gate(
            "dependencies_authenticated",
            True,
            dependencies_passed,
            dependencies_passed,
            "Unsafe or unavailable dependencies cannot contribute numeric readiness.",
        ),
        _gate(
            "six_lifecycle_controls",
            6,
            sum(row.get("passed") is True for row in rows),
            lifecycle_passed,
            "Every named lifecycle boundary must pass once from durable evidence.",
        ),
        _gate(
            "fixture_sidecar_integrity",
            True,
            sidecars_passed,
            sidecars_passed,
            "A reference is evidence only while its exact bytes and event count match.",
        ),
        _gate(
            "tamper_and_missing_event_controls",
            2,
            sum(row.get("passed") is True for row in controls),
            negative_controls_passed,
            "The sidecar boundary must reject byte changes and lost events.",
        ),
        _gate(
            "actual_load_and_selfparse_callers_reachable",
            True,
            callers_passed,
            callers_passed,
            "Readiness requires the production call sites, not only the ledger class.",
        ),
        _gate(
            "no_surviving_owned_child",
            True,
            no_child,
            no_child,
            "Cleanup is complete only after every task-owned child stops.",
        ),
        _gate(
            "historical_quarantine_preserved",
            list(_EXPECTED_HISTORICAL_FLAGS),
            list(historical_flags),
            quarantine_preserved,
            "The emitter repair must retain the original critical findings.",
        ),
        _gate(
            "terminal_projection_excludes_invocation_payloads",
            True,
            current_projection_clean,
            current_projection_clean,
            "Historical fixture bytes must not become current inference evidence.",
        ),
        _gate(
            "required_validation_complete",
            list(REQUIRED_VALIDATION_NAMES),
            [str(row.get("name")) for row in validation_receipts],
            validation_passed,
            "Every named affected, coverage, static, e2e, and terminal check must pass.",
        ),
    ]
    ready = all(row["passed"] is True for row in gates)
    artifact.update(
        {
            "status": "complete",
            "phase_durations_s": deepcopy(dict(phase_durations_s)),
            "rows": deepcopy(rows),
            "lifecycle_control_rows": deepcopy(rows),
            "sample_size_budget": {
                "planned_units": 6,
                "attempted_units": len(rows),
                "complete_units": sum(row.get("passed") is True for row in rows),
                "censored_units": sum(row.get("censored") is True for row in rows),
                "stopping_rule": (
                    "run each frozen CPU lifecycle control once; do not extend from outcomes"
                ),
                "outcome_based_extension": False,
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_summary(
                [
                    *preconditions_checked,
                    *(
                        {
                            "upstream": EXPERIMENT_ID,
                            "check": "acceptance_gate",
                            "field": row["criterion"],
                            "observed_value": row["observed"],
                            "expected_value": row["expected"],
                            "passed": row["passed"],
                        }
                        for row in gates
                    ),
                ]
            ),
            "validation_receipts": [dict(row) for row in validation_receipts],
            "arc_receipt_ready_score": int(ready),
            "fixture_sidecar_manifest": deepcopy(manifest),
            "caller_handoff": deepcopy(dict(caller_handoff)),
            "sidecar_control_rows": deepcopy(controls),
            "historical_evidence_sidecars": deepcopy(dict(historical_evidence_sidecars)),
            "repository_health": deepcopy(dict(repository_health)),
            "historical_quarantine_preserved": quarantine_preserved,
            "honest_verdict": (
                "complete_circular_positive_arc_receipt_boundary_clean"
                if ready
                else "complete_disqualified_arc_receipt_control_or_validation_failed"
            ),
            "verdict_class": "circular_positive" if ready else "disqualified",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _is_sha256(value: Any) -> bool:
    """Accept only the explicit prefixed digest form used by this artifact."""

    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check terminal identity, current scope, sidecars, rows, and seal."""

    errors: list[str] = []

    def add(condition: bool, error: str) -> None:
        if condition and error not in errors:
            errors.append(error)

    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity_mismatch",
    )
    add(artifact.get("status") not in {"complete", "blocked"}, "status_not_terminal")
    add(artifact.get("MODEL_SPECS") != [], "current_model_specs_must_be_empty")
    add(artifact.get("model_invoked") is not False, "current_model_invoked_must_be_false")
    add(artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS, "current_counts_must_be_zero")
    add(
        artifact.get("current_model_load_count") != 0
        or artifact.get("current_generation_count") != 0,
        "current_scalar_counts_must_be_zero",
    )
    add(
        artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("execution_venue") != "host",
        "substrate_or_venue_invalid",
    )
    add(artifact.get("verifier_is_oracle") is not True, "verifier_oracle_missing")
    add(artifact.get("official_score") is not None, "official_score_must_be_unset")
    add(artifact.get("production_default_changed") is not False, "production_default_changed")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles_invalid")
    add(
        artifact.get("reproducibility_checksum") != artifact_checksum(artifact),
        "reproducibility_checksum_mismatch",
    )
    duration = artifact.get("duration_s")
    add(
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or float(duration) <= 0,
        "duration_invalid",
    )
    add(
        artifact.get("verdict_class")
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict_class_invalid",
    )
    add(
        artifact.get("verifier_is_oracle") is True and artifact.get("verdict_class") == "positive",
        "oracle_forbids_positive",
    )
    manifest = artifact.get("fixture_sidecar_manifest")
    manifest = manifest if isinstance(manifest, list) else []
    add(
        any(not _is_sha256(row.get("sha256")) for row in manifest if isinstance(row, Mapping)),
        "fixture_sidecar_hash_invalid",
    )
    projection = json.dumps(
        {
            "rows": artifact.get("rows", []),
            "lifecycle": artifact.get("lifecycle_control_rows", []),
            "manifest": manifest,
        }
    )
    add(
        any(
            token in projection
            for token in (
                "model_identity",
                "call_rows",
                "generation_calls_attempted",
                "model_loads_attempted",
            )
        ),
        "invocation_payload_leaked_into_terminal_projection",
    )
    if artifact.get("status") == "blocked":
        add(artifact.get("verdict_class") != "blocked", "blocked_verdict_class_invalid")
        add(artifact.get("arc_receipt_ready_score") != 0, "blocked_ready_score_invalid")
        add(
            not str(artifact.get("honest_verdict", "")).startswith("blocked_"),
            "blocked_verdict_prefix_invalid",
        )
        add(
            artifact.get("gate_check_summary", {}).get("first_failure") is None,
            "blocked_failure_summary_missing",
        )
    else:
        rows = artifact.get("lifecycle_control_rows")
        rows = rows if isinstance(rows, list) else []
        gates = artifact.get("acceptance_gate_results")
        gates = gates if isinstance(gates, list) else []
        expected_ready = int(
            len(rows) == 6
            and bool(gates)
            and all(row.get("passed") is True for row in gates if isinstance(row, Mapping))
        )
        add(artifact.get("rows") != rows, "lifecycle_rows_mismatch")
        add(artifact.get("arc_receipt_ready_score") != expected_ready, "ready_score_inconsistent")
        add(
            expected_ready == 1 and artifact.get("verdict_class") != "circular_positive",
            "ready_verdict_class_invalid",
        )
        add(
            expected_ready == 0 and artifact.get("verdict_class") == "circular_positive",
            "failed_gate_verdict_class_invalid",
        )
    return errors


def build_validation_commands(repo_root: Path, *, private_root: Path) -> list[CommandSpec]:
    """Build exact affected checks without a repository-wide pytest fallback."""

    root = repo_root.resolve()
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    coverage = str(root / ".venv/bin/coverage")
    ruff = str(root / ".venv/bin/ruff")
    mypy = str(root / ".venv/bin/mypy")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    test_paths = (
        TEST_PATH.as_posix(),
        "tests/python/test_arc_eval_provenance_contract_20260905.py",
        "tests/python/test_arc_induction_state_persistence.py",
        "tests/python/test_arc_tool_grammar_transport.py",
    )
    static_paths = (MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), TEST_PATH.as_posix())
    import_probe = (
        "import json; from pathlib import Path; "
        "import carnot.experiment_7304_v642_arc_receipt as m; "
        "p=Path(m.__file__).resolve(); "
        "print(json.dumps({'resolved_imports': "
        "{'carnot.experiment_7304_v642_arc_receipt': str(p)}}), flush=True); "
        f"raise SystemExit(not p.is_relative_to(Path({str(root / 'python')!r}).resolve()))"
    )
    coverage_file = private_root / ".coverage"
    include = "*/experiment_7304_v642_arc_receipt.py"
    return [
        CommandSpec("worktree_imports", (python, "-u", "-c", import_probe), "changed_module"),
        CommandSpec(
            "focused_exp7304",
            (pytest, *common, f"--basetemp={private_root / 'focused'}", test_paths[0], "-q"),
            "explicit_test",
        ),
        CommandSpec(
            "affected_arc_eval_provenance",
            (pytest, *common, f"--basetemp={private_root / 'provenance'}", test_paths[1], "-q"),
            "explicit_test",
        ),
        CommandSpec(
            "e2e_009_cross_call_persistence",
            (pytest, *common, f"--basetemp={private_root / 'e2e009'}", test_paths[2], "-q"),
            "e2e_009",
        ),
        CommandSpec(
            "e2e_010_tool_transport",
            (pytest, *common, f"--basetemp={private_root / 'e2e010'}", test_paths[3], "-q"),
            "e2e_010",
        ),
        CommandSpec(
            "e2e_009_llm_off_environment_smoke",
            (
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private_root / "e2e009-llm-off.json"),
            ),
            "e2e_009_llm_off_environment",
        ),
        CommandSpec(
            "changed_module_coverage",
            (
                coverage,
                "run",
                f"--data-file={coverage_file}",
                f"--include={include}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={private_root / 'coverage'}",
                test_paths[0],
                "-q",
            ),
            "changed_module_and_exact_test",
        ),
        CommandSpec(
            "changed_module_coverage_report",
            (
                coverage,
                "report",
                f"--data-file={coverage_file}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ),
            "changed_module",
        ),
        CommandSpec("ruff_check", (ruff, "check", *static_paths), "changed_files"),
        CommandSpec("ruff_format", (ruff, "format", "--check", *static_paths), "changed_files"),
        CommandSpec("changed_module_mypy", (mypy, MODULE_PATH.as_posix()), "changed_module"),
        CommandSpec(
            "scoped_spec_coverage",
            (python, "-u", "scripts/check_spec_coverage.py", *test_paths),
            "exact_tests",
        ),
    ]


def _terminal_validation_commands(repo_root: Path, candidate: Path) -> list[CommandSpec]:
    """Build the two unchanged terminal-candidate verifier commands."""

    python = str(repo_root.resolve() / ".venv/bin/python")
    return [
        CommandSpec(
            "terminal_candidate_adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_terminal_candidate",
        ),
        CommandSpec(
            "terminal_candidate_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_terminal_candidate",
        ),
    ]


def _repository_health(repo_root: Path) -> JsonDict:
    """Preserve the known broad-suite failure as history, not a current pass."""

    log_path = repo_root / EXP7289_REPOSITORY_HEALTH_LOG
    return {
        "affects_required_checks": False,
        "historical_failures": [
            {
                "source_experiment": "exp7289-arc-boundary",
                "command": ".venv/bin/pytest -o addopts= tests/python -q --no-cov -n 0",
                "exit_code": 2,
                "resolved": False,
                "log_path": EXP7289_REPOSITORY_HEALTH_LOG.as_posix(),
                "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
            }
        ],
    }


def run_experiment(run_date: str) -> JsonDict:  # pragma: no cover - integration entrypoint.
    """Run bounded CPU controls, scoped checks, and atomic terminal publication."""

    started = time.monotonic()
    started_at = _utc_now()
    _progress(started, "startup", "entrypoint_and_paths_authenticated")
    if run_date != RUN_DATE:
        raise ValueError(f"--date must be {RUN_DATE}")

    phase_durations: JsonDict = {}
    phase = time.monotonic()
    _progress(started, "preconditions", "begin")
    preconditions, source_hashes, history = authenticate_inputs(
        REPO_ROOT, output_path=REPO_ROOT / RESULT_PATH
    )
    phase_durations["preconditions"] = time.monotonic() - phase
    _progress(
        started,
        "preconditions",
        "end",
        f"passed={all(row['passed'] for row in preconditions)} units={len(preconditions)}",
    )
    if not all(row["passed"] for row in preconditions):
        blocked = build_blocked_artifact(
            started_at_utc=started_at,
            completed_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
            preconditions_checked=preconditions,
            source_artifact_hashes=source_hashes,
        )
        atomic_write(REPO_ROOT / RESULT_PATH, blocked)
        _progress(started, "publication", "terminal_blocked_artifact_written")
        return blocked

    raw_dir = REPO_ROOT / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=raw_dir))
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {"status": "running", "phase": "cpu_controls", "completed_units": 0},
    )

    phase = time.monotonic()
    _progress(started, "cpu_controls", "before_benchmark", "total_units=6")
    panel = run_cpu_receipt_panel(run_dir / "fixtures")
    panel_path = run_dir / "fixture_event_payloads.json"
    atomic_write(panel_path, panel)
    lifecycle_rows, manifest = project_lifecycle_panel(panel, root=REPO_ROOT)
    controls = run_sidecar_negative_controls(
        Path(str(panel["rows"][1]["ledger_path"])),
        output_dir=run_dir / "negative_controls",
    )
    phase_durations["cpu_controls"] = time.monotonic() - phase
    _progress(started, "cpu_controls", "after_benchmark", "completed_units=6")

    phase = time.monotonic()
    _progress(started, "caller_reachability", "before_benchmark")
    seams = exercise_live_caller_seams(run_dir / "caller_reachability")
    handoff = build_caller_handoff(REPO_ROOT, seams)
    handoff_path = run_dir / "caller_handoff.json"
    atomic_write(handoff_path, handoff)
    phase_durations["caller_reachability"] = time.monotonic() - phase
    _progress(started, "caller_reachability", "after_benchmark", "completed_units=2")

    source_hashes[str(panel_path.relative_to(REPO_ROOT))] = {
        "sha256": sha256_file(panel_path),
        "terminal_class": "historical_fixture_sidecar",
        "quarantined": False,
        "retired": False,
    }
    source_hashes[str(handoff_path.relative_to(REPO_ROOT))] = {
        "sha256": sha256_file(handoff_path),
        "terminal_class": "cpu_caller_handoff",
        "quarantined": False,
        "retired": False,
    }

    phase = time.monotonic()
    _progress(started, "validation", "begin")
    private_root = Path(tempfile.mkdtemp(prefix="exp7304-validation-", dir="/tmp"))
    validation = run_commands(
        REPO_ROOT,
        build_validation_commands(REPO_ROOT, private_root=private_root),
        log_dir=run_dir / "validation",
        extra_env={"CARNOT_ARC_DISABLE_INDUCTION": "1"},
        heartbeat_s=60.0,
    )
    preliminary_path = run_dir / "measured_terminal_candidate.json"
    preliminary = build_terminal_artifact(
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_durations_s=phase_durations,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        lifecycle_rows=lifecycle_rows,
        fixture_sidecar_manifest=manifest,
        sidecar_control_rows=controls,
        caller_handoff=handoff,
        historical_evidence_sidecars=history,
        repository_health=_repository_health(REPO_ROOT),
        validation_receipts=validation,
    )
    atomic_write(preliminary_path, preliminary)
    terminal_validation = run_commands(
        REPO_ROOT,
        _terminal_validation_commands(REPO_ROOT, preliminary_path),
        log_dir=run_dir / "terminal_validation",
        heartbeat_s=60.0,
    )
    validation.extend(terminal_validation)
    phase_durations["validation"] = time.monotonic() - phase
    _progress(
        started,
        "validation",
        "end",
        f"passed={all(row['passed'] for row in validation)} units={len(validation)}",
    )

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_durations_s=phase_durations,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        lifecycle_rows=lifecycle_rows,
        fixture_sidecar_manifest=manifest,
        sidecar_control_rows=controls,
        caller_handoff=handoff,
        historical_evidence_sidecars=history,
        repository_health=_repository_health(REPO_ROOT),
        validation_receipts=validation,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact failed validation: {errors}")
    atomic_write(preliminary_path, artifact)
    atomic_write(REPO_ROOT / RESULT_PATH, artifact)
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {
            "status": "complete",
            "phase": "terminal_published",
            "completed_units": 6,
            "result_path": RESULT_PATH.as_posix(),
        },
    )
    _progress(
        started,
        "publication",
        "terminal_artifact_atomically_written",
        f"path={RESULT_PATH} arc_receipt_ready_score={artifact['arc_receipt_ready_score']}",
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the execution date without accepting an implicit milestone change."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI dispatch.
    """Print immediately, run once, and return a process exit code."""

    print(
        json.dumps({"experiment": EXPERIMENT_ID, "phase": "startup", "event": "entrypoint"}),
        flush=True,
    )
    args = parse_args(argv)
    run_experiment(args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

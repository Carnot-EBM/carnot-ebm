#!/usr/bin/env python3
"""Run Exp 1296 and write the prior-failures activation audit artifact.

Spec: REQ-INFRA-1296, SCENARIO-INFRA-1296, SCENARIO-INFRA-1296-BLOCKED
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_roadmap_gates import audit_roadmap, select_roadmap_path  # noqa: E402
from validate_prior_failures import validate_roadmap  # noqa: E402


PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1296_prior_failures_activation_audit.json"
REQUESTED_ROADMAP = PROJECT_ROOT / "research-roadmap-next.yaml"
COMPLETE_PATH = PROJECT_ROOT / "research-complete.yaml"
EXP1283_PATH = PROJECT_ROOT / "results" / "experiment_1283_certificate_grammar_backend_bakeoff.json"
EXP1288_PATH = (
    PROJECT_ROOT / "results" / "experiment_1288_interwhen_dvi_verifier_feedback_replay.json"
)

_PRIOR_TASK_RE = re.compile(r"Task '([^']+)'")
_GATE_FIELD_RE = re.compile(r"gate field '([^']+)'")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _in_progress_artifact(project_root: Path, run_date: str) -> dict[str, Any]:
    return {
        "artifact": "experiment_1296_prior_failures_activation_audit",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
        },
        "run_date": run_date,
        "status": "in_progress",
        "prior_failures_coverage_ok": False,
        "roadmap_gate_audit_passed": False,
        "exp1283_grammar_backend_available": False,
        "exp1288_memory_update_written": False,
        "n_prior_failures_missing": None,
        "n_gate_upstream_failures": None,
        "n_gate_field_cross_ref_failures": None,
        "activation_blockers": [],
        "honest_verdict": "activation_audit_in_progress",
    }


def _artifact_flag(path: Path, field: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return payload.get(field) is True


def _blocker(
    *,
    source: str,
    kind: str,
    task_id: str | None,
    field: str,
    detail: str,
) -> dict[str, str | None]:
    return {
        "source": source,
        "kind": kind,
        "task_id": task_id,
        "field": field,
        "detail": detail,
    }


def _blocker_from_schema_error(detail: str) -> dict[str, str | None]:
    field = "roadmap_path"
    if detail.startswith("Schema error at "):
        field = detail.split("Schema error at ", 1)[1].split(":", 1)[0]
    return _blocker(
        source="validate_prior_failures",
        kind="schema_error",
        task_id=None,
        field=field,
        detail=detail,
    )


def _blocker_from_prior_violation(detail: str) -> dict[str, str | None]:
    match = _PRIOR_TASK_RE.search(detail)
    task_id = match.group(1) if match else None
    return _blocker(
        source="validate_prior_failures",
        kind="prior_failures_missing",
        task_id=task_id,
        field="prior_failures",
        detail=detail,
    )


def _blocker_from_gate_detail(detail: str) -> dict[str, str | None]:
    parts = detail.split()
    kind = parts[0] if parts else "UNKNOWN"
    task_id = parts[1].rstrip(":") if len(parts) > 1 else None
    field = "unknown"
    if kind == "GATE_UPSTREAM_EXISTS":
        field = "gated_on[].upstream"
    elif kind == "GATE_FIELD_CROSS_REF":
        match = _GATE_FIELD_RE.search(detail)
        field = match.group(1) if match else "gated_on[].artifact_field"
    elif kind == "PRIOR_FAILURES_COVERAGE":
        field = "prior_failures"
    elif kind == "MODEL_AGENT_COHERENCE":
        field = "agent_type/model"
    return _blocker(
        source="audit_roadmap_gates",
        kind=kind,
        task_id=task_id,
        field=field,
        detail=detail,
    )


def _activation_blockers(
    schema_errors: list[str],
    prior_violations: list[str],
    gate_failure_details: list[str],
) -> list[dict[str, str | None]]:
    return (
        [_blocker_from_schema_error(detail) for detail in schema_errors]
        + [_blocker_from_prior_violation(detail) for detail in prior_violations]
        + [_blocker_from_gate_detail(detail) for detail in gate_failure_details]
    )


def run_experiment(
    project_root: Path = PROJECT_ROOT,
    requested_roadmap: Path | None = None,
    complete_path: Path | None = None,
    output_path: Path | None = None,
    run_date: str = "20260505",
) -> dict[str, Any]:
    """Audit the .101 roadmap and write the Exp 1296 JSON artifact."""
    requested = requested_roadmap or project_root / "research-roadmap-next.yaml"
    complete = complete_path or project_root / "research-complete.yaml"
    output = output_path or project_root / "results" / OUTPUT_PATH.name
    exp1283 = project_root / "results" / EXP1283_PATH.name
    exp1288 = project_root / "results" / EXP1288_PATH.name

    _write_json(output, _in_progress_artifact(project_root, run_date))

    roadmap_path, roadmap_note = select_roadmap_path(
        requested,
        active_path=project_root / "research-roadmap.yaml",
    )
    schema_errors, prior_violations = validate_roadmap(roadmap_path, complete_path=complete)
    gate_result = audit_roadmap(roadmap_path, complete_path=complete)

    n_prior_failures_missing = max(
        len(prior_violations),
        gate_result.n_prior_failures_missing,
    )
    prior_failures_coverage_ok = (
        not schema_errors
        and not prior_violations
        and gate_result.n_prior_failures_missing == 0
    )
    roadmap_gate_audit_passed = gate_result.roadmap_gate_audit_passed
    activation_blockers = _activation_blockers(
        schema_errors,
        prior_violations,
        gate_result.failure_details,
    )
    activation_ok = prior_failures_coverage_ok and roadmap_gate_audit_passed

    artifact: dict[str, Any] = {
        "artifact": "experiment_1296_prior_failures_activation_audit",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
        },
        "experiment": "exp1296-prior-failures-activation-audit",
        "schema_version": 1,
        "run_date": run_date,
        "status": "complete",
        "prior_failures_coverage_ok": prior_failures_coverage_ok,
        "roadmap_gate_audit_passed": roadmap_gate_audit_passed,
        "exp1283_grammar_backend_available": _artifact_flag(
            exp1283, "grammar_backend_available"
        ),
        "exp1288_memory_update_written": _artifact_flag(exp1288, "memory_update_written"),
        "n_prior_failures_missing": n_prior_failures_missing,
        "n_gate_upstream_failures": gate_result.n_gate_upstream_failures,
        "n_gate_field_cross_ref_failures": gate_result.n_gate_field_cross_ref_failures,
        "activation_blockers": activation_blockers,
        "honest_verdict": (
            "activation_audit_passed" if activation_ok else "activation_audit_blocked"
        ),
        "roadmap_path_requested": str(requested),
        "roadmap_path_used": str(roadmap_path),
        "roadmap_path_note": roadmap_note,
        "schema_errors": schema_errors,
        "prior_failure_findings": prior_violations,
        "gate_failure_details": list(gate_result.failure_details),
        "n_gate_upstream_checks": gate_result.n_gate_upstream_checks,
        "n_prior_failures_checks": gate_result.n_prior_failures_checks,
        "n_model_agent_coherence_failures": gate_result.n_model_agent_coherence_failures,
    }

    _write_json(output, artifact)
    return artifact


def main() -> int:
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

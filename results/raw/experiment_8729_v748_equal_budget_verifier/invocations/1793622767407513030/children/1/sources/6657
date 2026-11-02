#!/usr/bin/env python3
"""Run Exp 1152 and write the pre-activation roadmap audit artifact.

Spec: REQ-INFRA-075, SCENARIO-INFRA-087
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from audit_roadmap_gates import (
    _load_yaml_mapping,
    _tasks_from_roadmap,
    audit_roadmap,
    select_roadmap_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUESTED_ROADMAP = PROJECT_ROOT / "research-roadmap-next.yaml"
COMPLETE_PATH = PROJECT_ROOT / "research-complete.yaml"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1152_gate_audit_pre_activation_v2.json"
ARXIV_TASK_ID = "exp1153-arxiv-final-submission-v4"
REQUIRED_ARXIV_PRIOR_IDS = (
    "exp1139-arxiv-final-submission-v3",
    "exp1127-arxiv-pdf-compilation-final-submission",
    "exp1116-arxiv-pdf-compilation-submission",
)


def _failure_detail_with_fix(detail: str) -> str:
    """Add operator-facing remediation guidance to a raw audit failure line."""
    parts = detail.split()
    task_id = parts[1].rstrip(":") if len(parts) > 1 else "<unknown-task>"
    if detail.startswith("PRIOR_FAILURES_COVERAGE "):
        fix = (
            f"add non-empty prior_failures for {task_id} covering the matched prior experiments; "
            "if review confirms this is genuinely new work, document AUDIT_SCRIPT_ISSUE "
            "false-positive keyword matching instead of editing the roadmap"
        )
    elif detail.startswith("GATE_UPSTREAM_EXISTS "):
        fix = (
            "add the missing upstream task to the same roadmap or correct/remove the gated_on entry"
        )
    elif detail.startswith("GATE_FIELD_CROSS_REF "):
        fix = (
            "add the gated artifact_field to the upstream REQUIRED ARTIFACT FIELDS block "
            "or correct the downstream gated_on artifact_field"
        )
    elif detail.startswith("MODEL_AGENT_COHERENCE "):
        fix = "set codex tasks to model=gpt-5.5 and replace unsupported agent_type=gemini routing"
    else:
        fix = "inspect this audit failure before conductor activation"
    return f"{detail} | fix_needed: {fix}"


def _declared_prior_failure_ids(task: dict[str, Any] | None) -> set[str]:
    """Extract prior failure experiment IDs from a roadmap task."""
    if not task:
        return set()
    declared: set[str] = set()
    for entry in task.get("prior_failures") or []:
        if isinstance(entry, dict):
            raw_id = entry.get("experiment_id") or entry.get("id")
        else:
            raw_id = entry
        if raw_id:
            declared.add(str(raw_id))
    return declared


def _find_task(roadmap_path: Path, task_id: str) -> dict[str, Any] | None:
    """Return one roadmap task by id, or None when the roadmap does not contain it."""
    roadmap = _load_yaml_mapping(roadmap_path)
    for task in _tasks_from_roadmap(roadmap):
        if str(task.get("id") or "") == task_id:
            return task
    return None


def _arxiv_prior_failure_status(roadmap_path: Path) -> tuple[bool, list[str]]:
    """Check whether exp1153 declares all three historical arXiv failure IDs."""
    declared = _declared_prior_failure_ids(_find_task(roadmap_path, ARXIV_TASK_ID))
    missing = [prior_id for prior_id in REQUIRED_ARXIV_PRIOR_IDS if prior_id not in declared]
    return not missing, missing


def _honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the Exp 1152 verdict enum required by the roadmap prompt."""
    if int(artifact["n_prior_failures_missing"]) > 0:
        return "prior_failures_gaps_found"
    gate_failures = (
        int(artifact["n_gate_upstream_failures"])
        + int(artifact["n_gate_field_cross_ref_failures"])
        + int(artifact["n_model_agent_coherence_failures"])
    )
    if gate_failures:
        return "gate_field_gaps_found"
    return "all_checks_pass"


def run_experiment(
    project_root: Path = PROJECT_ROOT,
    requested_roadmap: Path | None = None,
    complete_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Audit the milestone roadmap and write the Exp 1152 JSON artifact."""
    requested = requested_roadmap or project_root / "research-roadmap-next.yaml"
    complete = complete_path or project_root / "research-complete.yaml"
    output = output_path or project_root / "results" / OUTPUT_PATH.name
    active = project_root / "research-roadmap.yaml"
    roadmap_path, note = select_roadmap_path(requested, active_path=active)

    result = audit_roadmap(roadmap_path, complete_path=complete)
    arxiv_complete, missing_arxiv_priors = _arxiv_prior_failure_status(roadmap_path)
    failure_details = [_failure_detail_with_fix(detail) for detail in result.failure_details]

    n_prior_failures_missing = result.n_prior_failures_missing
    arxiv_already_counted = any(
        detail.startswith(f"PRIOR_FAILURES_COVERAGE {ARXIV_TASK_ID}")
        for detail in result.failure_details
    )
    if not arxiv_complete:
        failure_details.append(
            "ARXIV_PRIOR_FAILURES_COVERAGE "
            f"{ARXIV_TASK_ID}: missing prior_failures entries {missing_arxiv_priors} "
            "| fix_needed: add these experiment_id values to exp1153 prior_failures before "
            "activating the arXiv submission task"
        )
        if not arxiv_already_counted:
            n_prior_failures_missing += 1

    artifact: dict[str, Any] = {
        "experiment": "exp1152-gate-audit-pre-activation-v2",
        "schema_version": 1,
        "n_tasks_audited": result.n_tasks_audited,
        "n_prior_failures_missing": n_prior_failures_missing,
        "n_gate_upstream_failures": result.n_gate_upstream_failures,
        "n_model_agent_coherence_failures": result.n_model_agent_coherence_failures,
        "n_gate_field_cross_ref_failures": result.n_gate_field_cross_ref_failures,
        "arxiv_task_prior_failures_complete": arxiv_complete,
        "roadmap_gate_audit_passed": False,
        "failure_details": failure_details,
        "roadmap_path_requested": str(requested),
        "roadmap_path_used": str(roadmap_path),
        "roadmap_path_note": note,
    }
    artifact["roadmap_gate_audit_passed"] = _honest_verdict(artifact) == "all_checks_pass"
    artifact["honest_verdict"] = _honest_verdict(artifact)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

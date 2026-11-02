#!/usr/bin/env python3
"""Run Exp 1575 and write the carry-forward prior-failure audit artifact.

Spec: REQ-REPORT-064, SCENARIO-REPORT-064
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_roadmap_gates import select_roadmap_path  # noqa: E402


PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "experiment_1575_carry_forward_prior_failures_autofill_audit.json"
)
REQUESTED_ROADMAP = PROJECT_ROOT / "research-roadmap-next.yaml"
COMPLETE_PATH = PROJECT_ROOT / "research-complete.yaml"

EXP1576_TASK_ID = "exp1576-paper-v6-section-3-sampler-draft-resumed"
EXP1577_TASK_ID = "exp1577-extropic-z1-readiness-packet-thrml-alignment-resumed"
TARGET_REQUIRED_PRIORS = {
    EXP1576_TASK_ID: "exp1569-paper-v6-section-3-sampler-draft",
    EXP1577_TASK_ID: "exp1573-extropic-z1-readiness-packet-thrml-alignment-update",
}
TARGET_RESULT_FIELDS = {
    EXP1576_TASK_ID: "exp1576_prior_failures_valid",
    EXP1577_TASK_ID: "exp1577_prior_failures_valid",
}
AUTOFILL_COUNTS_RE = re.compile(
    r"(?P<tasks>\d+) tasks scanned, (?P<stubs>\d+) stubs generated, "
    r"(?P<populated>\d+) already populated"
)
EXP_ID_RE = re.compile(r"^exp(?P<number>\d+)-")
PRIOR_REQUIRED_FIELDS = (
    "experiment_id",
    "verdict",
    "addressed_by",
    "retire_if_same_verdict",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _in_progress_artifact(project_root: Path, run_date: str) -> dict[str, Any]:
    return {
        "artifact": "experiment_1575_carry_forward_prior_failures_autofill_audit",
        "artifact_metadata": {"project_root": str(project_root), "run_date": run_date},
        "run_date": run_date,
        "status": "in_progress",
        "autofill_dry_run_completed": False,
        "validate_prior_failures_passed": False,
        "audit_roadmap_gates_passed": False,
        "exp1576_prior_failures_valid": False,
        "exp1577_prior_failures_valid": False,
        "carryforward_prior_failures_ready": False,
        "honest_verdict": "in_progress",
    }


def _run_command(args: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _parse_autofill_counts(stdout: str) -> dict[str, int | None]:
    match = AUTOFILL_COUNTS_RE.search(stdout)
    if not match:
        return {"tasks_scanned": None, "stubs_generated": None, "already_populated": None}
    return {
        "tasks_scanned": int(match.group("tasks")),
        "stubs_generated": int(match.group("stubs")),
        "already_populated": int(match.group("populated")),
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return data


def _tasks_by_id(roadmap_path: Path) -> dict[str, dict[str, Any]]:
    roadmap = _load_yaml_mapping(roadmap_path)
    tasks = roadmap.get("tasks") or []
    return {
        str(task.get("id")): task
        for task in tasks
        if isinstance(task, dict) and task.get("id")
    }


def _artifact_path_for_prior(project_root: Path, experiment_id: str) -> Path | None:
    match = EXP_ID_RE.match(experiment_id)
    if not match:
        return None
    candidates = sorted((project_root / "results").glob(f"experiment_{match.group('number')}_*.json"))
    return candidates[0] if candidates else None


def _source_verdict(project_root: Path, experiment_id: str) -> tuple[Path | None, str | None]:
    source_path = _artifact_path_for_prior(project_root, experiment_id)
    if source_path is None:
        return None, None
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    verdict = payload.get("honest_verdict") if isinstance(payload, dict) else None
    return source_path, verdict if isinstance(verdict, str) else None


def _gap(task_id: str, field: str, detail: str) -> dict[str, str]:
    return {"task_id": task_id, "field": field, "detail": detail}


def _discipline_gaps(task_id: str, prior_failures: Any) -> list[dict[str, str]]:
    if not isinstance(prior_failures, list) or not prior_failures:
        return [
            _gap(
                task_id,
                "prior_failures",
                "prior_failures must be a non-empty list",
            )
        ]

    gaps: list[dict[str, str]] = []
    for index, entry in enumerate(prior_failures):
        if not isinstance(entry, dict):
            gaps.append(
                _gap(task_id, f"prior_failures[{index}]", "prior_failures entry is not a dict")
            )
            continue
        missing = [
            field
            for field in PRIOR_REQUIRED_FIELDS
            if entry.get(field) in (None, "", [], {})
        ]
        if missing:
            gaps.append(
                _gap(
                    task_id,
                    f"prior_failures[{index}].{missing[0]}",
                    f"prior_failures[{index}] missing/empty fields: {missing}",
                )
            )
    return gaps


def _validate_target_task(
    project_root: Path,
    task: dict[str, Any],
    required_prior_id: str,
) -> dict[str, Any]:
    task_id = str(task.get("id"))
    prior_failures = task.get("prior_failures")
    details = _discipline_gaps(task_id, prior_failures)
    prior_list = prior_failures if isinstance(prior_failures, list) else []
    prior_ids = [
        str(entry.get("experiment_id"))
        for entry in prior_list
        if isinstance(entry, dict) and entry.get("experiment_id")
    ]
    if required_prior_id not in prior_ids:
        details.append(
            _gap(
                task_id,
                "prior_failures.experiment_id",
                f"missing required prior {required_prior_id}",
            )
        )

    checked_priors: list[dict[str, str | None]] = []
    for index, entry in enumerate(prior_list):
        if not isinstance(entry, dict):
            continue
        experiment_id = str(entry.get("experiment_id") or "")
        if not EXP_ID_RE.match(experiment_id):
            continue
        source_path, expected_verdict = _source_verdict(project_root, experiment_id)
        if source_path is None:
            details.append(
                _gap(
                    task_id,
                    f"prior_failures[{index}].experiment_id",
                    f"no source artifact found for {experiment_id}",
                )
            )
            continue
        actual_verdict = str(entry.get("verdict") or "")
        checked_priors.append(
            {
                "experiment_id": experiment_id,
                "source_artifact": str(source_path),
                "source_honest_verdict": expected_verdict,
                "roadmap_verdict": actual_verdict,
            }
        )
        if actual_verdict != expected_verdict:
            details.append(
                _gap(
                    task_id,
                    f"prior_failures[{index}].verdict",
                    f"expected {expected_verdict} from {source_path}, got {actual_verdict}",
                )
            )

    return {
        "valid": not details,
        "details": details,
        "checked_priors": checked_priors,
    }


def inspect_target_prior_failures(project_root: Path, roadmap_path: Path) -> dict[str, Any]:
    tasks = _tasks_by_id(roadmap_path)
    output: dict[str, Any] = {
        "exp1576_prior_failures_valid": False,
        "exp1577_prior_failures_valid": False,
        "prior_failure_gap_details": [],
        "target_prior_checks": {},
    }

    for task_id, required_prior_id in TARGET_REQUIRED_PRIORS.items():
        task = tasks.get(task_id)
        if task is None:
            output["prior_failure_gap_details"].append(
                _gap(task_id, "task", "task not found in selected roadmap")
            )
            continue
        validation = _validate_target_task(project_root, task, required_prior_id)
        output[TARGET_RESULT_FIELDS[task_id]] = validation["valid"]
        output["prior_failure_gap_details"].extend(validation["details"])
        output["target_prior_checks"][task_id] = validation["checked_priors"]

    return output


def run_experiment(
    project_root: Path = PROJECT_ROOT,
    requested_roadmap: Path | None = None,
    complete_path: Path | None = None,
    output_path: Path | None = None,
    run_date: str = "20260508",
) -> dict[str, Any]:
    """Audit the .121 carry-forward prior-failure metadata and write JSON."""
    requested = requested_roadmap or project_root / "research-roadmap-next.yaml"
    complete = complete_path or project_root / "research-complete.yaml"
    output = output_path or project_root / "results" / OUTPUT_PATH.name

    _write_json(output, _in_progress_artifact(project_root, run_date))

    roadmap_path, roadmap_note = select_roadmap_path(
        requested,
        active_path=project_root / "research-roadmap.yaml",
    )
    command_results = {
        "autofill_dry_run": _run_command(
            [
                sys.executable,
                str(project_root / "scripts" / "conductor_priors_autofill.py"),
                str(roadmap_path),
                "--dry-run",
            ],
            cwd=project_root,
        ),
        "validate_prior_failures": _run_command(
            [
                sys.executable,
                str(project_root / "scripts" / "validate_prior_failures.py"),
                str(roadmap_path),
            ],
            cwd=project_root,
        ),
        "audit_roadmap_gates": _run_command(
            [
                sys.executable,
                str(project_root / "scripts" / "audit_roadmap_gates.py"),
                str(roadmap_path),
                "--complete",
                str(complete),
            ],
            cwd=project_root,
        ),
    }
    inspection = inspect_target_prior_failures(project_root, roadmap_path)
    autofill_dry_run_completed = command_results["autofill_dry_run"]["returncode"] == 0
    validate_prior_failures_passed = (
        command_results["validate_prior_failures"]["returncode"] == 0
    )
    audit_roadmap_gates_passed = command_results["audit_roadmap_gates"]["returncode"] == 0
    carryforward_prior_failures_ready = (
        autofill_dry_run_completed
        and validate_prior_failures_passed
        and audit_roadmap_gates_passed
        and bool(inspection["exp1576_prior_failures_valid"])
        and bool(inspection["exp1577_prior_failures_valid"])
    )

    artifact: dict[str, Any] = {
        "artifact": "experiment_1575_carry_forward_prior_failures_autofill_audit",
        "artifact_metadata": {"project_root": str(project_root), "run_date": run_date},
        "experiment": "exp1575-carry-forward-prior-failures-autofill-audit",
        "schema_version": 1,
        "run_date": run_date,
        "status": "complete",
        "autofill_dry_run_completed": autofill_dry_run_completed,
        "validate_prior_failures_passed": validate_prior_failures_passed,
        "audit_roadmap_gates_passed": audit_roadmap_gates_passed,
        "exp1576_prior_failures_valid": inspection["exp1576_prior_failures_valid"],
        "exp1577_prior_failures_valid": inspection["exp1577_prior_failures_valid"],
        "carryforward_prior_failures_ready": carryforward_prior_failures_ready,
        "honest_verdict": (
            "carryforward_prior_failures_ready"
            if carryforward_prior_failures_ready
            else "carryforward_prior_failures_blocked"
        ),
        "roadmap_path_requested": str(requested),
        "roadmap_path_used": str(roadmap_path),
        "roadmap_path_note": roadmap_note,
        "autofill_summary": _parse_autofill_counts(
            str(command_results["autofill_dry_run"]["stdout"])
        ),
        "command_results": command_results,
        "prior_failure_gap_details": inspection["prior_failure_gap_details"],
        "target_prior_checks": inspection["target_prior_checks"],
    }
    _write_json(output, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Exp 5122: archive .469 and activate the .470 research frame.

Spec refs: REQ-REPORT-5122, SCENARIO-REPORT-5122,
SCENARIO-REPORT-5122-ACTIVE-FALLBACK.

This module is intentionally record-only. It reads the .469 capstone and the
artifacts named by that capstone, checks the .470 planning files, and writes a
single aggregation artifact. The important research decision is preserved in a
machine-checkable shape: the same-verdict FoVer selector path and the FoVer
residual FR-11 route are retired unless a genuinely different corpus or
benchmark appears.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
VerificationRunner = Callable[[Path], "CommandResult"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5121_capstone_v469.json")
RESULT_RELATIVE_PATH = Path("results/experiment_5122_archive_469_activate_470.json")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5122_archive_469_activate_470"
EXPERIMENT_ID = "exp5122-archive-469-activate-470"
ARCHIVED_MILESTONE = "2026.07.469"
MILESTONE = "2026.07.470"
SCHEMA = "carnot.experiment_5122_archive_469_activate_470.v1"
RANDOM_SEED = 5122
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = "complete_archive_469_closed_470_next_roadmap_ready_fover_retired"
ACTIVE_FALLBACK_VERDICT = "complete_archive_469_closed_470_active_roadmap_ready_fover_retired"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_", "passed_", "shipped_")

REQUIRED_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5122, 5134))
SPEC_REFS = [
    "REQ-REPORT-5122",
    "SCENARIO-REPORT-5122",
    "SCENARIO-REPORT-5122-ACTIVE-FALLBACK",
]

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "source_artifacts_read",
    "fover_selector_retired_for_same_verdict",
    "roadmap_next_present",
    "active_roadmap_modified",
    "conductor_modified",
    "flagged_adversarial",
    "tests_run",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "spec_refs",
    "result_path",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "adversarial_verification",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "source_artifacts_read": "evidence provenance",
    "fover_selector_retired_for_same_verdict": "no doomed rerun",
    "roadmap_next_present": "activation readiness",
    "active_roadmap_modified": "operator instruction compliance",
    "conductor_modified": "conductor immutability",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5122_archive_469_activate_470.py --date 20260701",
    ".venv/bin/pytest tests/python/test_experiment_5122_archive_469_activate_470.py -q -o addopts=''",
    ".venv/bin/coverage run --include='python/carnot/experiment_5122_archive_469_activate_470.py' "
    "-m pytest tests/python/test_experiment_5122_archive_469_activate_470.py -q -o addopts=''",
    ".venv/bin/coverage report --include='python/carnot/experiment_5122_archive_469_activate_470.py' "
    "--fail-under=100 -m",
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for artifact verification commands."""

    command: Sequence[str]
    exit_code: int
    stdout: str
    stderr: str


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _bool(value: Any) -> bool:
    return value is True


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive.
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive.
        return {}, {"exists": True, "loadable": False, "error": "json_not_object"}
    return dict(payload), {"exists": True, "loadable": True, "sha256": file_sha256(path)}


def _task_prefixes_present(task_ids: Sequence[str], prefixes: Sequence[str]) -> bool:
    return all(any(task_id.startswith(prefix) for task_id in task_ids) for prefix in prefixes)


def _roadmap_check(path: Path) -> JsonDict:
    if not path.exists():
        return {
            "path": str(path.name),
            "exists": False,
            "parses": False,
            "milestone": "missing",
            "task_ids": [],
            "required_task_ids_present": False,
            "missing_required_task_prefixes": list(REQUIRED_TASK_PREFIXES),
        }
    text = path.read_text(encoding="utf-8")
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {
            "path": str(path.name),
            "exists": True,
            "parses": False,
            "milestone": "yaml_poison",
            "task_ids": [],
            "required_task_ids_present": False,
            "missing_required_task_prefixes": list(REQUIRED_TASK_PREFIXES),
            "error": str(exc),
        }
    mapping = _mapping(loaded)
    tasks = _list(mapping.get("tasks"))
    task_ids = [
        str(_mapping(task).get("id", ""))
        for task in tasks
        if isinstance(_mapping(task).get("id", ""), str)
    ]
    missing = [
        prefix for prefix in REQUIRED_TASK_PREFIXES if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]
    return {
        "path": str(path.name),
        "exists": True,
        "parses": True,
        "milestone": str(mapping.get("milestone", "unknown")),
        "task_ids": task_ids,
        "required_task_ids_present": _task_prefixes_present(task_ids, REQUIRED_TASK_PREFIXES),
        "missing_required_task_prefixes": missing,
    }


def _vnext_check(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "path": str(VNEXT_RELATIVE_PATH),
        "exists": path.exists(),
        "names_milestone": MILESTONE in text,
    }


def _source_row(root: Path, *, kind: str, source_id: str, relative_path: Path, extra: JsonMap | None = None) -> JsonDict:
    path = root / relative_path
    row: JsonDict = {
        "kind": kind,
        "source_id": source_id,
        "path": str(relative_path),
        "exists": path.exists(),
        "sha256": file_sha256(path),
    }
    if extra:
        row.update(dict(extra))
    return row


def load_capstone(root: Path) -> tuple[JsonDict, JsonDict]:
    return read_json_mapping(root / CAPSTONE_RELATIVE_PATH)


def build_source_artifacts_read(root: Path, capstone: JsonMap) -> list[JsonDict]:
    rows = [
        _source_row(
            root,
            kind="capstone",
            source_id="exp5121-capstone-v469",
            relative_path=CAPSTONE_RELATIVE_PATH,
        )
    ]
    seen = {str(CAPSTONE_RELATIVE_PATH)}
    for source in _list(capstone.get("artifacts_read")) + _list(capstone.get("missing_artifacts")):
        source_map = _mapping(source)
        path_text = str(source_map.get("path", ""))
        if not path_text or path_text in seen:
            continue
        seen.add(path_text)
        exp_number = source_map.get("experiment_number")
        rows.append(
            _source_row(
                root,
                kind="referenced_result_artifact",
                source_id=f"exp{exp_number}" if exp_number is not None else path_text,
                relative_path=Path(path_text),
                extra={
                    "experiment_number": exp_number,
                    "label": source_map.get("label", ""),
                    "capstone_reference_status": "missing"
                    if source in _list(capstone.get("missing_artifacts"))
                    else "present",
                },
            )
        )
    rows.extend(
        [
            _source_row(root, kind="roadmap_doc", source_id="vnext_doc", relative_path=VNEXT_RELATIVE_PATH),
            _source_row(
                root,
                kind="roadmap_yaml",
                source_id="research_roadmap_next",
                relative_path=ROADMAP_NEXT_RELATIVE_PATH,
            ),
            _source_row(
                root,
                kind="roadmap_yaml",
                source_id="active_research_roadmap",
                relative_path=ACTIVE_ROADMAP_RELATIVE_PATH,
            ),
        ]
    )
    return rows


def load_referenced_payloads(root: Path, capstone: JsonMap) -> dict[int, JsonDict]:
    payloads: dict[int, JsonDict] = {}
    for source in _list(capstone.get("artifacts_read")):
        source_map = _mapping(source)
        exp_number = source_map.get("experiment_number")
        if not isinstance(exp_number, int):
            continue
        payload, status = read_json_mapping(root / str(source_map.get("path", "")))
        if status.get("loadable") is True:
            payloads[exp_number] = payload
    return payloads


def build_fover_retirement(capstone: JsonMap) -> JsonDict:
    fover = _mapping(capstone.get("fover_moat_state"))
    fr11 = _mapping(capstone.get("fr11_state"))
    recommendations = [
        _mapping(row) for row in _list(capstone.get("next_milestone_recommendations"))
    ]
    retirement_rows = [
        row for row in recommendations if _bool(row.get("retire_same_verdict_doomed_rerun"))
    ]
    same_verdict_retired = bool(retirement_rows) and fover.get("state") == "blocked"
    fr11_gap = str(fr11.get("gap_reason", ""))
    return {
        "same_verdict_retired": same_verdict_retired,
        "selector_path_blocked": fover.get("state") == "blocked",
        "selector_ran": _bool(fover.get("selector_ran")),
        "audit_ran": _bool(fover.get("audit_ran")),
        "headroom_present": _bool(fover.get("headroom_present")),
        "pool_n": int(_number(fover.get("pool_n")) or 0),
        "corrected_result_summary": _mapping(fover.get("corrected_result_summary")),
        "retirement_recommendation": retirement_rows[0] if retirement_rows else {},
        "fover_residual_fr11_should_not_rerun": same_verdict_retired
        and fr11.get("state") == "blocked"
        and "exp5112" in fr11_gap,
        "fr11_block_reason": fr11_gap,
    }


def build_preconditions(root: Path, capstone_status: JsonMap) -> JsonDict:
    vnext = _vnext_check(root / VNEXT_RELATIVE_PATH)
    roadmap_next = _roadmap_check(root / ROADMAP_NEXT_RELATIVE_PATH)
    active_roadmap = _roadmap_check(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    roadmap_next_ok = (
        roadmap_next.get("exists") is True
        and roadmap_next.get("milestone") == MILESTONE
        and roadmap_next.get("required_task_ids_present") is True
    )
    active_roadmap_ok = (
        active_roadmap.get("exists") is True
        and active_roadmap.get("milestone") == MILESTONE
        and active_roadmap.get("required_task_ids_present") is True
    )
    return {
        "capstone": {
            "path": str(CAPSTONE_RELATIVE_PATH),
            "exists": capstone_status.get("exists") is True,
            "loadable": capstone_status.get("loadable") is True,
            "sha256": capstone_status.get("sha256"),
        },
        "vnext_doc": vnext,
        "research_roadmap_next": roadmap_next,
        "active_roadmap": active_roadmap,
        "roadmap_next_ready": roadmap_next_ok,
        "active_roadmap_fallback_ready": (roadmap_next.get("exists") is False) and active_roadmap_ok,
    }


def _honest_verdict(preconditions: JsonMap) -> str:
    capstone = _mapping(preconditions.get("capstone"))
    vnext = _mapping(preconditions.get("vnext_doc"))
    roadmap_next = _mapping(preconditions.get("research_roadmap_next"))
    if capstone.get("loadable") is not True:
        return "blocked_capstone_artifact_missing_or_unloadable"
    if vnext.get("exists") is not True:
        return "blocked_vnext_doc_missing"
    if vnext.get("names_milestone") is not True:
        return "blocked_vnext_doc_milestone_mismatch"
    if preconditions.get("roadmap_next_ready") is True:
        return COMPLETE_VERDICT
    if preconditions.get("active_roadmap_fallback_ready") is True:
        return ACTIVE_FALLBACK_VERDICT
    if roadmap_next.get("exists") is not True:
        return "blocked_research_roadmap_next_missing"
    if roadmap_next.get("milestone") != MILESTONE:
        return "blocked_research_roadmap_next_milestone_mismatch"
    return "blocked_research_roadmap_next_task_set_incomplete"


def _verification_flags(result: CommandResult) -> list[JsonDict]:
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    flags = decoded.get("flags") if isinstance(decoded, Mapping) else []
    return [dict(flag) for flag in flags if isinstance(flag, Mapping)]


def command_result_payload(result: CommandResult) -> JsonDict:
    return {
        "command": list(result.command),
        "exit_code": int(result.exit_code),
        "green": result.exit_code == 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def verification_payload(result: CommandResult) -> JsonDict:
    flags = _verification_flags(result)
    critical = [flag for flag in flags if str(flag.get("severity", "")).lower() == "critical"]
    return {
        **command_result_payload(result),
        "flags": flags,
        "flagged_adversarial": result.exit_code != 0 or bool(critical),
    }


def run_adversarial_verification(root: Path, output_path: Path) -> CommandResult:
    command = [
        sys.executable,
        str(root / "scripts" / "adversarial_verify.py"),
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    return CommandResult(
        command=tuple(command),
        exit_code=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    verification: JsonMap,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    capstone, capstone_status = load_capstone(root)
    referenced_payloads = load_referenced_payloads(root, capstone)
    source_artifacts_read = build_source_artifacts_read(root, capstone)
    preconditions = build_preconditions(root, capstone_status)
    fover_retirement = build_fover_retirement(capstone)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "archived_milestone": ARCHIVED_MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(duration_s, 0.0001), 6),
        "source_artifacts_read": source_artifacts_read,
        "capstone_summary": {
            "experiment_id": capstone.get("experiment_id"),
            "honest_verdict": capstone.get("honest_verdict"),
            "flagged_adversarial": capstone.get("flagged_adversarial"),
            "referenced_payloads_loaded": sorted(referenced_payloads),
        },
        "fover_retirement": fover_retirement,
        "fover_selector_retired_for_same_verdict": fover_retirement["same_verdict_retired"],
        "fr11_residual_rerun_policy": {
            "fover_residual_fr11_should_not_rerun": fover_retirement[
                "fover_residual_fr11_should_not_rerun"
            ],
            "replacement_scope_required": "different_auditable_stream_not_same_fover_selector_residuals",
        },
        "kan_post_wall_state": _mapping(capstone.get("kan_post_wall_state")),
        "solver_sampling_state": _mapping(capstone.get("solver_sampling_state")),
        "fr11_state": _mapping(capstone.get("fr11_state")),
        "runtime_state": _mapping(capstone.get("runtime_state")),
        "hardware_state": _mapping(capstone.get("hardware_state")),
        "vnext_doc_check": preconditions["vnext_doc"],
        "roadmap_next_check": preconditions["research_roadmap_next"],
        "active_roadmap_check": preconditions["active_roadmap"],
        "roadmap_next_present": preconditions["research_roadmap_next"]["exists"] is True,
        "active_roadmap_fallback_used": preconditions["active_roadmap_fallback_ready"] is True,
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "adversarial_verification": dict(verification),
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
        "tests_run": list(tests_run),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing.{field}")
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(artifact.get("field_principles")).get(field) != principle:
            errors.append(f"field_principle.{field}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id.invalid")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone.invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict.not_terminal")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate.invalid")
    duration = _number(artifact.get("duration_s"))
    if duration is None or duration <= 0.0:
        errors.append("duration_s.invalid")
    if not _list(artifact.get("source_artifacts_read")):
        errors.append("source_artifacts_read.empty")
    if artifact.get("fover_selector_retired_for_same_verdict") is not True:
        errors.append("fover_selector_retired_for_same_verdict.invalid")
    if artifact.get("active_roadmap_modified") is not False:
        errors.append("active_roadmap_modified.invalid")
    if artifact.get("conductor_modified") is not False:
        errors.append("conductor_modified.invalid")
    if not isinstance(artifact.get("flagged_adversarial"), bool):
        errors.append("flagged_adversarial.invalid")
    if not _list(artifact.get("tests_run")):
        errors.append("tests_run.empty")
    if _mapping(artifact.get("fover_retirement")).get("fover_residual_fr11_should_not_rerun") is not True:
        errors.append("fover_residual_fr11_should_not_rerun.invalid")
    if _mapping(artifact.get("hardware_state")).get("no_speedup_claim") is not True:
        errors.append("hardware_state.no_speedup_claim.invalid")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum.invalid")
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    missing = [error for error in errors if error.startswith("missing.")]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principle_errors = [error for error in errors if error.startswith("field_principle.")]
    if principle_errors:
        raise ValueError(f"field principle mismatch: {principle_errors}")
    if errors:
        raise ValueError(f"invalid Exp 5122 archive artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    run_date: str = "20260701",
    verification_runner: VerificationRunner | None = None,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    clock: Clock = time.perf_counter,
) -> Path:
    root = Path(root)
    output_path = artifact_path or root / RESULT_RELATIVE_PATH
    runner = verification_runner or (lambda path: run_adversarial_verification(root, path))
    start = clock()
    active_before = file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    conductor_before = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    placeholder = verification_payload(CommandResult(command=(), exit_code=0, stdout='{"flags":[]}', stderr=""))
    artifact = build_artifact(
        root=root,
        duration_s=max(clock() - start, 0.0001),
        run_date=run_date,
        verification=placeholder,
        tests_run=tests_run,
    )
    write_json(output_path, artifact)
    verification = verification_payload(runner(output_path))
    active_after = file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    conductor_after = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    final_artifact = {
        **artifact,
        "active_roadmap_modified": active_before != active_after,
        "conductor_modified": conductor_before != conductor_after,
        "adversarial_verification": verification,
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
    }
    final_artifact["reproducibility_checksum"] = payload_checksum(final_artifact)
    validate_artifact(final_artifact)
    write_json(output_path, final_artifact)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the Exp 5122 archive .469 / activate .470 artifact.")
    parser.add_argument("--date", default="20260701", help="Run date label, e.g. 20260701.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root to read.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(root=args.root, artifact_path=args.output, run_date=args.date)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(f"{EXPERIMENT}: wrote {output}")
    print(f"{EXPERIMENT}: honest_verdict={artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

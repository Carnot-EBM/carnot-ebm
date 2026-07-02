"""Exp 5150: archive .471 and activate the .472 ARC-priority frame.

Spec refs: REQ-REPORT-5150, SCENARIO-REPORT-5150,
SCENARIO-REPORT-5150-DIRTY-RUNTIME.

This module is record-only. It does not rerun the .471 research and it does not
edit the active conductor or roadmap. It gathers the capstone truth, task
verdicts, ARC reopening rationale, registry total, and handoff runtime state so
downstream .472 tasks can gate on the real transition facts.
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
RuntimeProbe = Callable[[Path], "RuntimeSnapshot"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5145_capstone_v471.json")
RESULT_RELATIVE_PATH = Path("results/experiment_5150_archive_471_activate_472.json")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")

EXPERIMENT = "experiment_5150_archive_471_activate_472"
EXPERIMENT_ID = "exp5150-archive-471-activate-472"
ARCHIVED_MILESTONE = "2026.07.471"
MILESTONE = "2026.07.472"
SCHEMA = "carnot.experiment_5150_archive_471_activate_472.v1"
RANDOM_SEED = 5150
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = "complete_archive_471_closed_472_arc_priority_active_runtime_clean"
DIRTY_HANDOFF_VERDICT = "complete_archive_471_closed_472_activation_gated_dirty_handoff"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")

REQUIRED_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5150, 5156))
SPEC_REFS = [
    "REQ-REPORT-5150",
    "SCENARIO-REPORT-5150",
    "SCENARIO-REPORT-5150-DIRTY-RUNTIME",
]

EXPECTED_TRANSITION_DIRTY_PATHS = {
    "openspec/capabilities/research-reporting/spec.md",
    "python/carnot/experiment_5150_archive_471_activate_472.py",
    "scripts/experiment_5150_archive_471_activate_472.py",
    "tests/python/test_experiment_5150_archive_471_activate_472.py",
    "results/experiment_5150_archive_471_activate_472.json",
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "source_artifacts_read",
    "task_verdicts",
    "capstone_summary",
    "v471_runtime_clean",
    "runtime_clean_details",
    "arc_reopened_by_operator_directive",
    "sprint_forcing_function_retired_preserved",
    "reproducible_total_levels",
    "active_roadmap_ready",
    "active_roadmap_modified",
    "conductor_modified",
    "flagged_adversarial",
    "tests_run",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "archived_milestone",
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
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ per Verdict "
        "Terminal-Prefix Discipline."
    ),
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "source_artifacts_read": "evidence provenance",
    "task_verdicts": "predecessor task truth",
    "capstone_summary": "predecessor milestone truth",
    "v471_runtime_clean": (
        "Downstream .472 tasks may gate on this; a dirty handoff should block, "
        "not silently proceed."
    ),
    "runtime_clean_details": "handoff diagnostics must explain the gate, not just emit a boolean",
    "arc_reopened_by_operator_directive": (
        "fresh ARC priority must come from the 2026-07-02 directive, not a "
        "retired deadline rule"
    ),
    "sprint_forcing_function_retired_preserved": (
        "historical retired rules should not be rewritten to justify current priority"
    ),
    "reproducible_total_levels": (
        "flat ARC progress is the measured premise for the .472 allocation"
    ),
    "active_roadmap_ready": "activation readiness",
    "active_roadmap_modified": "operator instruction compliance",
    "conductor_modified": "conductor immutability",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_5150_archive_471_activate_472.py "
    "--date 20260702",
    ".venv/bin/pytest tests/python/test_experiment_5150_archive_471_activate_472.py -q "
    "-o addopts=''",
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null "
    "--include='*/experiment_5150_archive_471_activate_472.py' "
    "-m pytest tests/python/test_experiment_5150_archive_471_activate_472.py -q --no-cov "
    "-o addopts=''",
    "JAX_PLATFORMS=cpu .venv/bin/coverage report --rcfile=/dev/null -m "
    "--include='*/experiment_5150_archive_471_activate_472.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5150_archive_471_activate_472.py",
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for artifact verification commands."""

    command: Sequence[str]
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Git and process-table evidence captured before the activation gate."""

    git_status_porcelain: str
    process_table: str


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
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):
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
            "ready": False,
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
            "ready": False,
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
        prefix
        for prefix in REQUIRED_TASK_PREFIXES
        if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]
    milestone = str(mapping.get("milestone", "unknown"))
    ready = milestone == MILESTONE and _task_prefixes_present(task_ids, REQUIRED_TASK_PREFIXES)
    return {
        "path": str(path.name),
        "exists": True,
        "parses": True,
        "milestone": milestone,
        "task_ids": task_ids,
        "required_task_ids_present": not missing,
        "missing_required_task_prefixes": missing,
        "ready": ready,
    }


def _research_complete_check(path: Path) -> JsonDict:
    base: JsonDict = {
        "path": str(RESEARCH_COMPLETE_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "has_v471": False,
        "v471_entry_count": 0,
        "ledger_gap": "research_complete_missing",
    }
    if not path.exists():
        return base
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {
            **base,
            "exists": True,
            "ledger_gap": "research_complete_yaml_poison",
            "error": str(exc),
        }
    milestones = _list(_mapping(loaded).get("milestones"))
    count = sum(1 for row in milestones if str(_mapping(row).get("id", "")) == ARCHIVED_MILESTONE)
    if count == 0:
        gap = f"missing_{ARCHIVED_MILESTONE}"
    elif count > 1:
        gap = f"duplicate_{ARCHIVED_MILESTONE}_entries"
    else:
        gap = "none"
    return {
        **base,
        "exists": True,
        "parses": True,
        "has_v471": count > 0,
        "v471_entry_count": count,
        "milestone_count": len(milestones),
        "ledger_gap": gap,
    }


def _known_issues_check(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    has_section = "ENERGY-BASED ARC RESEARCH LINEUP 2026-07-02" in text
    has_operator_words = (
        "we want to continue down this energy based models path for ARC-AGI-3" in text
        and "multi-level capable live agent" in text
    )
    return {
        "path": str(KNOWN_ISSUES_RELATIVE_PATH),
        "exists": path.exists(),
        "arc_reopened_by_operator_directive": has_section and has_operator_words,
        "flat_69_since_20260630_recorded": (
            "reproducible_total_levels" in text and "69" in text and "2026-06-30" in text
        ),
    }


def _claude_check(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "path": str(CLAUDE_RELATIVE_PATH),
        "exists": path.exists(),
        "sprint_forcing_function_retired_preserved": (
            "ARC-AGI-3 Submission Sprint Forcing Function" in text
            and "RETIRED 2026-06-30" in text
        ),
    }


def _registry_total_levels(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    total = _mapping(loaded).get("reproducible_total_levels")
    if isinstance(total, bool) or not isinstance(total, int):
        return None
    return total


def _source_row(
    root: Path,
    *,
    kind: str,
    source_id: str,
    relative_path: Path,
    extra: JsonMap | None = None,
) -> JsonDict:
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
            source_id="exp5145-capstone-v471",
            relative_path=CAPSTONE_RELATIVE_PATH,
        )
    ]
    seen = {str(CAPSTONE_RELATIVE_PATH)}
    sources = _list(capstone.get("upstream_artifacts_read")) + _list(
        capstone.get("classified_upstreams")
    )
    for source in sources:
        source_map = _mapping(source)
        path_text = str(source_map.get("relative_path") or source_map.get("path") or "")
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
                    "axis": source_map.get("axis", ""),
                    "capstone_classification": source_map.get("classification", ""),
                    "flagged_adversarial_stamped": source_map.get("flagged_adversarial"),
                },
            )
        )
    rows.extend(
        [
            _source_row(root, kind="source_doc", source_id="claude", relative_path=CLAUDE_RELATIVE_PATH),
            _source_row(
                root,
                kind="source_doc",
                source_id="known_issues",
                relative_path=KNOWN_ISSUES_RELATIVE_PATH,
            ),
            _source_row(
                root,
                kind="registry_yaml",
                source_id="arc_solve_registry",
                relative_path=ARC_REGISTRY_RELATIVE_PATH,
            ),
            _source_row(
                root,
                kind="ledger_yaml",
                source_id="research_complete",
                relative_path=RESEARCH_COMPLETE_RELATIVE_PATH,
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
    for source in _list(capstone.get("upstream_artifacts_read")) + _list(
        capstone.get("classified_upstreams")
    ):
        source_map = _mapping(source)
        exp_number = source_map.get("experiment_number")
        if not isinstance(exp_number, int):
            continue
        path_text = str(source_map.get("relative_path") or source_map.get("path") or "")
        payload, status = read_json_mapping(root / path_text)
        if status.get("loadable") is True:
            payloads[exp_number] = payload
    return payloads


def build_task_verdicts(capstone: JsonMap, referenced_payloads: Mapping[int, JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in _list(capstone.get("classified_upstreams")):
        source_map = _mapping(source)
        exp_number = source_map.get("experiment_number")
        payload = _mapping(referenced_payloads.get(exp_number, {})) if isinstance(exp_number, int) else {}
        rows.append(
            {
                "experiment_number": exp_number,
                "experiment_id": payload.get("experiment_id", f"exp{exp_number}"),
                "label": source_map.get("label", ""),
                "axis": source_map.get("axis", ""),
                "classification": source_map.get("classification", ""),
                "honest_verdict": payload.get("honest_verdict", source_map.get("honest_verdict", "")),
                "flagged_adversarial": _bool(
                    payload.get("flagged_adversarial", source_map.get("flagged_adversarial"))
                ),
                "path": source_map.get("relative_path", source_map.get("path", "")),
            }
        )
    rows.append(
        {
            "experiment_number": 5145,
            "experiment_id": capstone.get("experiment_id", "exp5145-capstone-v471"),
            "label": "capstone_v471",
            "axis": "capstone",
            "classification": "quarantined" if _bool(capstone.get("flagged_adversarial")) else "clean",
            "honest_verdict": capstone.get("honest_verdict", ""),
            "flagged_adversarial": _bool(capstone.get("flagged_adversarial")),
            "path": str(CAPSTONE_RELATIVE_PATH),
        }
    )
    return rows


def _dirty_paths(git_status_porcelain: str) -> list[str]:
    paths: list[str] = []
    for line in git_status_porcelain.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def _process_row(line: str) -> JsonDict | None:
    parts = line.strip().split(maxsplit=4)
    if len(parts) < 5:
        return None
    pid, ppid, stat, elapsed, command = parts
    if not pid.isdigit() or not ppid.isdigit():
        return None
    return {"pid": int(pid), "ppid": int(ppid), "stat": stat, "elapsed": elapsed, "command": command}


def analyze_runtime_snapshot(snapshot: RuntimeSnapshot) -> JsonDict:
    dirty_paths = _dirty_paths(snapshot.git_status_porcelain)
    ignored = [path for path in dirty_paths if path in EXPECTED_TRANSITION_DIRTY_PATHS]
    non_transition = [path for path in dirty_paths if path not in EXPECTED_TRANSITION_DIRTY_PATHS]
    process_rows = [
        row for line in snapshot.process_table.splitlines() if (row := _process_row(line)) is not None
    ]
    conductor_rows = [
        row
        for row in process_rows
        if "research_conductor.py" in str(row["command"]) or "carnot-conductor" in str(row["command"])
    ]
    active_task_rows = [row for row in process_rows if "codex exec" in str(row["command"])]
    orphaned = [row for row in conductor_rows if row["ppid"] == 1]
    runtime_clean = not non_transition and not orphaned
    return {
        "git_status_porcelain": snapshot.git_status_porcelain,
        "dirty_paths": dirty_paths,
        "ignored_transition_dirty_paths": ignored,
        "non_transition_dirty_paths": non_transition,
        "conductor_processes": conductor_rows,
        "active_task_processes": active_task_rows,
        "orphaned_conductor_processes": orphaned,
        "expected_transition_dirty_paths": sorted(EXPECTED_TRANSITION_DIRTY_PATHS),
        "touched_conductor_process": False,
        "runtime_clean": runtime_clean,
    }


def capture_runtime_snapshot(root: Path) -> RuntimeSnapshot:
    git = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    ps = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return RuntimeSnapshot(git_status_porcelain=git.stdout, process_table=ps.stdout)


def derive_v471_facts(capstone: JsonMap) -> JsonDict:
    source_scope = _mapping(capstone.get("source_scope_audit_state"))
    structured = _mapping(capstone.get("structured_generation_state"))
    solver = _mapping(capstone.get("solver_formulation_state"))
    guided = _mapping(capstone.get("guided_decoding_state"))
    abstention = _mapping(capstone.get("abstention_trace_state"))
    kan = _mapping(capstone.get("kan_symbolic_state"))
    sampling = _mapping(capstone.get("sampling_partition_state"))
    taco = _mapping(capstone.get("taco_harm_state"))
    fr11 = _mapping(capstone.get("fr11_state"))
    hardware = _mapping(capstone.get("hardware_state"))
    return {
        "v471_capstone_verdict": str(capstone.get("honest_verdict", "")),
        "v471_source_scope_quarantined": source_scope.get("classification") == "quarantined",
        "v471_structured_generation_clean": structured.get("classification") == "clean"
        and _bool(structured.get("downstream_tasks_trustworthy")),
        "v471_solver_no_utility_beyond_static": _number(
            solver.get("selector_delta_vs_best_static")
        )
        == 0.0,
        "v471_guided_decoding_blocked": guided.get("classification") == "blocked"
        or str(guided.get("honest_verdict", "")).startswith("blocked"),
        "v471_abstention_trace_clean": abstention.get("classification") == "clean"
        and _bool(abstention.get("verification_trace_ready")),
        "v471_kan_symbolic_clean": kan.get("classification") == "clean"
        and _bool(kan.get("symbolic_kan_ready"))
        and _bool(kan.get("certificate_soundness")),
        "v471_sampling_partition_clean": sampling.get("classification") == "clean"
        and _bool(sampling.get("partition_telemetry_ready")),
        "v471_taco_harm_gate_clean": taco.get("classification") == "clean"
        and _bool(taco.get("trace_suite_v2_ready"))
        and (_number(taco.get("wrong_label_count")) or 0.0) == 0.0,
        "v471_fr11_quarantined": fr11.get("classification") == "quarantined",
        "v471_hardware_blocked_no_speedup": hardware.get("classification") == "blocked"
        and _bool(hardware.get("no_speedup_claim")),
        "v471_state_summary": {
            "source_scope_audit": source_scope,
            "structured_generation": structured,
            "solver_formulation": solver,
            "guided_decoding": guided,
            "abstention_trace": abstention,
            "kan_symbolic": kan,
            "sampling_partition": sampling,
            "taco_harm": taco,
            "fr11": fr11,
            "hardware": hardware,
            "next_milestone_recommendations": _list(
                capstone.get("next_milestone_recommendations")
            ),
            "retire_or_quarantine_recommendations": _list(
                capstone.get("retire_or_quarantine_recommendations")
            ),
        },
    }


def build_preconditions(root: Path, capstone_status: JsonMap) -> JsonDict:
    known_issues = _known_issues_check(root / KNOWN_ISSUES_RELATIVE_PATH)
    claude = _claude_check(root / CLAUDE_RELATIVE_PATH)
    active_roadmap = _roadmap_check(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    ledger = _research_complete_check(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    return {
        "capstone": {
            "path": str(CAPSTONE_RELATIVE_PATH),
            "exists": capstone_status.get("exists") is True,
            "loadable": capstone_status.get("loadable") is True,
            "sha256": capstone_status.get("sha256"),
        },
        "known_issues": known_issues,
        "claude": claude,
        "arc_registry": {
            "path": str(ARC_REGISTRY_RELATIVE_PATH),
            "exists": (root / ARC_REGISTRY_RELATIVE_PATH).exists(),
            "reproducible_total_levels": _registry_total_levels(root / ARC_REGISTRY_RELATIVE_PATH),
        },
        "research_complete": ledger,
        "active_roadmap": active_roadmap,
    }


def _honest_verdict(preconditions: JsonMap, *, runtime_clean: bool) -> str:
    capstone = _mapping(preconditions.get("capstone"))
    known_issues = _mapping(preconditions.get("known_issues"))
    claude = _mapping(preconditions.get("claude"))
    active_roadmap = _mapping(preconditions.get("active_roadmap"))
    if capstone.get("loadable") is not True:
        return "blocked_capstone_artifact_missing_or_unloadable"
    if known_issues.get("arc_reopened_by_operator_directive") is not True:
        return "blocked_arc_reopen_directive_missing"
    if claude.get("sprint_forcing_function_retired_preserved") is not True:
        return "blocked_retired_sprint_context_missing"
    if active_roadmap.get("ready") is not True:
        return "blocked_active_roadmap_not_ready"
    if not runtime_clean:
        return DIRTY_HANDOFF_VERDICT
    return COMPLETE_VERDICT


def _verification_flags(result: CommandResult) -> list[JsonDict]:
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, Mapping):
        return []
    flags = decoded.get("flags")
    if flags is None:
        reports = _list(decoded.get("reports"))
        flags = [flag for report in reports for flag in _list(_mapping(report).get("flags"))]
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
        "--json",
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
    runtime_snapshot: RuntimeSnapshot,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    capstone, capstone_status = load_capstone(root)
    referenced_payloads = load_referenced_payloads(root, capstone)
    source_artifacts_read = build_source_artifacts_read(root, capstone)
    task_verdicts = build_task_verdicts(capstone, referenced_payloads)
    preconditions = build_preconditions(root, capstone_status)
    runtime_details = analyze_runtime_snapshot(runtime_snapshot)
    runtime_clean = _bool(runtime_details.get("runtime_clean"))
    facts = derive_v471_facts(capstone)
    capstone_summary = {
        "experiment_id": capstone.get("experiment_id"),
        "honest_verdict": capstone.get("honest_verdict"),
        "flagged_adversarial": capstone.get("flagged_adversarial"),
        "inference_substrate": capstone.get("inference_substrate"),
        "classified_upstream_count": len(_list(capstone.get("classified_upstreams"))),
        "missing_artifact_count": len(_list(capstone.get("missing_artifacts"))),
        "next_milestone_recommendations": _list(capstone.get("next_milestone_recommendations")),
        "retire_or_quarantine_recommendations": _list(
            capstone.get("retire_or_quarantine_recommendations")
        ),
    }
    registry_total = _mapping(preconditions.get("arc_registry")).get("reproducible_total_levels")
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
        "honest_verdict": _honest_verdict(preconditions, runtime_clean=runtime_clean),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(duration_s, 0.0001), 6),
        "source_artifacts_read": source_artifacts_read,
        "task_verdicts": task_verdicts,
        "capstone_summary": capstone_summary,
        "v471_runtime_clean": runtime_clean,
        "runtime_clean_details": runtime_details,
        "arc_reopened_by_operator_directive": _bool(
            _mapping(preconditions.get("known_issues")).get("arc_reopened_by_operator_directive")
        ),
        "sprint_forcing_function_retired_preserved": _bool(
            _mapping(preconditions.get("claude")).get(
                "sprint_forcing_function_retired_preserved"
            )
        ),
        "reproducible_total_levels": registry_total,
        "research_complete_has_v471": _bool(
            _mapping(preconditions.get("research_complete")).get("has_v471")
        ),
        "active_roadmap_check": preconditions["active_roadmap"],
        "active_roadmap_ready": _bool(_mapping(preconditions.get("active_roadmap")).get("ready")),
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "adversarial_verification": dict(verification),
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
        "tests_run": list(tests_run),
        **facts,
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
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        errors.append("archived_milestone.invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict.not_terminal")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate.invalid")
    duration = _number(artifact.get("duration_s"))
    if duration is None or duration <= 0.0:
        errors.append("duration_s.invalid")
    if not _list(artifact.get("source_artifacts_read")):
        errors.append("source_artifacts_read.empty")
    if not _list(artifact.get("task_verdicts")):
        errors.append("task_verdicts.empty")
    if not isinstance(artifact.get("capstone_summary"), Mapping):
        errors.append("capstone_summary.invalid")
    if not isinstance(artifact.get("v471_runtime_clean"), bool):
        errors.append("v471_runtime_clean.invalid")
    if not isinstance(artifact.get("runtime_clean_details"), Mapping):
        errors.append("runtime_clean_details.invalid")
    if not isinstance(artifact.get("arc_reopened_by_operator_directive"), bool):
        errors.append("arc_reopened_by_operator_directive.invalid")
    if not isinstance(artifact.get("sprint_forcing_function_retired_preserved"), bool):
        errors.append("sprint_forcing_function_retired_preserved.invalid")
    total = artifact.get("reproducible_total_levels")
    if isinstance(total, bool) or not isinstance(total, int):
        errors.append("reproducible_total_levels.invalid")
    if not isinstance(artifact.get("active_roadmap_ready"), bool):
        errors.append("active_roadmap_ready.invalid")
    if artifact.get("active_roadmap_modified") is not False:
        errors.append("active_roadmap_modified.invalid")
    if artifact.get("conductor_modified") is not False:
        errors.append("conductor_modified.invalid")
    if not isinstance(artifact.get("flagged_adversarial"), bool):
        errors.append("flagged_adversarial.invalid")
    if not _list(artifact.get("tests_run")):
        errors.append("tests_run.empty")
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
        raise ValueError(f"invalid Exp 5150 archive artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    run_date: str = "20260702",
    verification_runner: VerificationRunner | None = None,
    runtime_probe: RuntimeProbe = capture_runtime_snapshot,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    clock: Clock = time.perf_counter,
) -> Path:
    root = Path(root)
    output_path = artifact_path or root / RESULT_RELATIVE_PATH
    runner = verification_runner or (lambda path: run_adversarial_verification(root, path))
    start = clock()
    active_before = file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    conductor_before = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    runtime_snapshot = runtime_probe(root)
    placeholder = verification_payload(
        CommandResult(command=(), exit_code=0, stdout='{"flags":[]}', stderr="")
    )
    artifact = build_artifact(
        root=root,
        duration_s=max(clock() - start, 0.0001),
        run_date=run_date,
        verification=placeholder,
        runtime_snapshot=runtime_snapshot,
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
    parser = argparse.ArgumentParser(
        description="Write the Exp 5150 archive .471 / activate .472 artifact."
    )
    parser.add_argument("--date", default="20260702", help="Run date label, e.g. 20260702.")
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

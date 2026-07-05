"""Exp 5282: archive .482 and emit the .483 activation artifact.

Spec refs: REQ-REPORT-5282, SCENARIO-REPORT-5282,
SCENARIO-REPORT-5282-BLOCKED-CLOSEOUT.

This module is a reporting receipt, not a research experiment. It reads the
already-written `.482` artifacts and local operational records, checks that the
`.483` roadmap state is ready without copying any roadmap file, and writes a
durable JSON artifact. Harmful, blocked, quarantined, tiny-scale, and
no-speedup results stay visible instead of being converted into new research
claims.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5282_archive_482_activate_483.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5281_capstone_v482.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

EXPERIMENT = "experiment_5282_archive_482_activate_483"
EXPERIMENT_ID = "exp5282-archive-482-activate-483"
ARCHIVED_MILESTONE = "2026.07.482"
ACTIVATION_MILESTONE = "2026.07.483"
SCHEMA = "carnot.experiment_5282_archive_482_activate_483.v1"
RANDOM_SEED = 5282
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = [
    "REQ-REPORT-5282",
    "SCENARIO-REPORT-5282",
    "SCENARIO-REPORT-5282-BLOCKED-CLOSEOUT",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether .482 was archived and "
        ".483 is activation-ready."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts because Exp5282 only reads existing artifacts "
        "and local records."
    ),
    "milestone_archived": (
        "Bare boolean confirming the .482 closeout is represented in durable research records."
    ),
    "activation_ready": (
        "Bare boolean confirming .483 can proceed without overwriting research-roadmap.yaml."
    ),
    "ops_docs_updated": (
        "False when the conductor stop rule delegates ops/status/changelog/traceability "
        "reconciliation."
    ),
    "research_complete_updated": (
        "True only if this workflow appended or reconciled research-complete.yaml; false "
        "when .482 was already present."
    ),
    "exclusions_checked": (
        "The transition must run or explicitly record available roadmap, exclusion, and "
        "prior-failure checks."
    ),
    "roadmap_activation_check": "No roadmap overwrite is allowed; activated must remain false.",
    "commands_run": "Every validation command must be recorded with pass/fail outcome.",
}

PRINCIPLE_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "ops_docs_updated",
    "research_complete_updated",
    "exclusions_checked",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "archived_milestone",
    "activation_milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "duration_s",
    "random_seed",
    "source_artifacts",
    "source_context",
    "closeout_facts",
    "closeout_fact_failures",
    "research_complete_check",
    "failed_preconditions",
    "milestone_archived",
    "milestone_archived_principle",
    "activation_ready",
    "activation_ready_principle",
    "ops_docs_updated",
    "research_complete_updated",
    "exclusions_checked",
    "roadmap_activation_check",
    "commands_run",
    "honest_verdict",
    "inference_substrate",
    "reproducibility_checksum",
)

EXPECTED_CLOSEOUT_FACTS: dict[str, Any] = {
    "sota_telemetry_ready": True,
    "internal_logit_harmful_regression": True,
    "solver_fixture_rebuilt": True,
    "sota_extraction_blocked_by_gguf_offload": True,
    "governed_memory_positive": True,
    "verifier_dosing_positive": True,
    "kan_certificate_positive": True,
    "factor_boundary_tiny_no_speedup": True,
    "hardware_blocked_no_speedup": True,
    "evidence_audit_complete": True,
}

REQUIRED_483_TASK_PREFIXES = tuple(f"exp{idx}" for idx in range(5282, 5295))


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream `.482` artifact cited by the archive receipt."""

    experiment_number: int
    task_id: str
    relative_path: Path


@dataclass(frozen=True)
class CommandResult:
    """Captured command result for a validation command."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5269,
        "exp5269-archive-481-activate-482",
        Path("results/experiment_5269_archive_481_activate_482.json"),
    ),
    UpstreamSource(
        5270,
        "exp5270-sota-source-delta-v482",
        Path("results/experiment_5270_sota_source_delta_v482.json"),
    ),
    UpstreamSource(
        5271,
        "exp5271-sota-telemetry-receipt-harness-v482",
        Path("results/experiment_5271_sota_telemetry_receipt_harness_v482.json"),
    ),
    UpstreamSource(
        5272,
        "exp5272-internal-hallucination-probe-gated-v482",
        Path("results/experiment_5272_internal_hallucination_probe_gated_v482.json"),
    ),
    UpstreamSource(
        5273,
        "exp5273-solver-fixture-rebuild-v482",
        Path("results/experiment_5273_solver_fixture_rebuild_v482.json"),
    ),
    UpstreamSource(
        5274,
        "exp5274-solver-constraint-extraction-retry-gated-v482",
        Path("results/experiment_5274_solver_constraint_extraction_retry_gated_v482.json"),
    ),
    UpstreamSource(
        5275,
        "exp5275-governed-decision-history-memory-v482",
        Path("results/experiment_5275_governed_decision_history_memory_v482.json"),
    ),
    UpstreamSource(
        5276,
        "exp5276-memory-assisted-verifier-dose-gated-v482",
        Path("results/experiment_5276_memory_assisted_verifier_dose_gated_v482.json"),
    ),
    UpstreamSource(
        5277,
        "exp5277-kan-milp-certificate-scale-v482",
        Path("results/experiment_5277_kan_milp_certificate_scale_v482.json"),
    ),
    UpstreamSource(
        5278,
        "exp5278-constraint-factor-graph-boundary-v482",
        Path("results/experiment_5278_constraint_factor_graph_boundary_v482.json"),
    ),
    UpstreamSource(
        5279,
        "exp5279-hardware-continuity-reachability-v482",
        Path("results/experiment_5279_hardware_continuity_reachability_v482.json"),
    ),
    UpstreamSource(
        5280,
        "exp5280-artifact-normalizer-evidence-audit-v482",
        Path("results/experiment_5280_artifact_normalizer_evidence_audit_v482.json"),
    ),
    UpstreamSource(5281, "exp5281-capstone-v482", CAPSTONE_RELATIVE_PATH),
)

SOURCE_CONTEXT_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("research-complete.yaml"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("_bmad/traceability.md"),
)


def value_of(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _principled(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def text_sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


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
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_json"}
    if not isinstance(parsed, dict):
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return parsed, {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": file_sha256(path),
    }


def load_upstream_artifacts(root: Path) -> tuple[dict[int, JsonDict], list[JsonDict]]:
    artifacts: dict[int, JsonDict] = {}
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        data, meta = read_json_mapping(root / source.relative_path)
        if meta.get("loadable") is True:
            artifacts[source.experiment_number] = data
        rows.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "relative_path": str(source.relative_path),
                "exists": meta.get("exists") is True,
                "loadable": meta.get("loadable") is True,
                "sha256": meta.get("sha256"),
                "error": meta.get("error"),
                "honest_verdict": str(value_of(data.get("honest_verdict", ""))) if data else "",
                "flagged_adversarial": value_of(data.get("flagged_adversarial")) is True
                if data
                else False,
            }
        )
    return artifacts, rows


def _as_list(value: Any) -> list[Any]:
    raw = value_of(value)
    return raw if isinstance(raw, list) else []


def _row_for(rows: Sequence[Any], experiment_number: int) -> JsonDict | None:
    for row in rows:
        if isinstance(row, Mapping) and row.get("experiment_number") == experiment_number:
            return dict(row)
    return None


def _summary(row: JsonMap | None) -> str:
    return str(row.get("summary", "")) if row else ""


def _summary_lower(row: JsonMap | None) -> str:
    return _summary(row).lower()


def closeout_facts(capstone: JsonMap) -> JsonDict:
    clean_positives = _as_list(capstone.get("clean_positives"))
    harmful_or_regressions = _as_list(capstone.get("harmful_or_regressions"))
    flagged_or_quarantined = _as_list(capstone.get("flagged_or_quarantined"))
    honest_blocks = _as_list(capstone.get("honest_blocks"))

    row_5271 = _row_for(clean_positives, 5271)
    row_5272 = _row_for(harmful_or_regressions, 5272)
    row_5273 = _row_for(clean_positives, 5273)
    row_5274 = _row_for(flagged_or_quarantined, 5274)
    row_5275 = _row_for(clean_positives, 5275)
    row_5276 = _row_for(clean_positives, 5276)
    row_5277 = _row_for(clean_positives, 5277)
    row_5278 = _row_for(clean_positives, 5278)
    row_5279 = _row_for(honest_blocks, 5279)
    row_5280 = _row_for(clean_positives, 5280)
    hardware_speedup_claimed = value_of(capstone.get("hardware_speedup_claimed"))

    s5271 = _summary_lower(row_5271)
    s5272 = _summary_lower(row_5272)
    s5273 = _summary_lower(row_5273)
    s5274 = _summary_lower(row_5274)
    s5275 = _summary_lower(row_5275)
    s5276 = _summary_lower(row_5276)
    s5277 = _summary_lower(row_5277)
    s5278 = _summary_lower(row_5278)
    s5279 = _summary_lower(row_5279)
    s5280 = _summary_lower(row_5280)

    return {
        "sota_telemetry_ready": row_5271 is not None
        and "telemetry" in s5271
        and ("receipt" in s5271 or "ready" in s5271)
        and ("no verifier-quality claim" in s5271 or "no quality" in s5271),
        "internal_logit_harmful_regression": row_5272 is not None
        and "internal/logit" in s5272
        and ("harmful" in s5272 or "regression" in s5272)
        and ("lexical" in s5272 or "delta" in s5272),
        "solver_fixture_rebuilt": row_5273 is not None
        and "solver fixture" in s5273
        and ("rebuilt" in s5273 or "ready" in s5273),
        "sota_extraction_blocked_by_gguf_offload": row_5274 is not None
        and ("blocked" in s5274 or "unmeasured" in s5274)
        and ("gguf" in s5274 or "offload" in s5274 or "llama_cpp_gpu_offload" in s5274),
        "governed_memory_positive": row_5275 is not None
        and "governed" in s5275
        and ("ready" in s5275 or "positive" in s5275),
        "verifier_dosing_positive": row_5276 is not None
        and "verifier dosing" in s5276
        and ("preserved" in s5276 or "positive" in s5276),
        "kan_certificate_positive": row_5277 is not None
        and "kan" in s5277
        and "certificate" in s5277
        and ("positive" in s5277 or "scaled" in s5277),
        "factor_boundary_tiny_no_speedup": row_5278 is not None
        and "factor" in s5278
        and "boundary" in s5278
        and ("tiny" in s5278 or "shape-only" in s5278)
        and "no hardware speedup" in s5278,
        "hardware_blocked_no_speedup": row_5279 is not None
        and "blocked" in s5279
        and ("speedup_claimed=false" in s5279 or "no_speedup" in s5279)
        and hardware_speedup_claimed is False,
        "evidence_audit_complete": row_5280 is not None
        and "evidence" in s5280
        and ("ready" in s5280 or "complete" in s5280),
    }


def closeout_fact_failures(capstone: JsonMap) -> list[str]:
    if not capstone:
        return ["capstone_artifact_missing_or_unloadable"]
    facts = closeout_facts(capstone)
    failures: list[str] = []
    for field, expected in EXPECTED_CLOSEOUT_FACTS.items():
        observed = facts.get(field)
        if observed != expected:
            failures.append(f"closeout_{field}_expected_{expected}_observed_{observed}")
    return failures


def source_context(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for relative_path in SOURCE_CONTEXT_PATHS:
        path = root / relative_path
        rows.append(
            {
                "relative_path": str(relative_path),
                "exists": path.exists(),
                "sha256": file_sha256(path),
                "read_only": True,
            }
        )
    return rows


def _milestones(data: Any) -> list[Any]:
    if isinstance(data, Mapping) and isinstance(data.get("milestones"), list):
        return data["milestones"]
    return []


def research_complete_milestone_count(root: Path, milestone: str = ARCHIVED_MILESTONE) -> int:
    path = root / "research-complete.yaml"
    if not path.exists():
        return 0
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return 0
    return sum(1 for row in _milestones(data) if isinstance(row, Mapping) and row.get("id") == milestone)


def append_research_complete_milestone(root: Path) -> bool:
    path = root / "research-complete.yaml"
    if research_complete_milestone_count(root) > 0:
        return False
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    milestones = data.setdefault("milestones", [])
    if not isinstance(milestones, list):
        milestones = []
        data["milestones"] = milestones
    milestones.append(
        {
            "id": ARCHIVED_MILESTONE,
            "title": (
                "Receipt-Clean Internal Verification, Governed Self-Learning, "
                "and Hardware-Bound Certificates"
            ),
            "doc": str(VNEXT_RELATIVE_PATH),
            "completed": "2026-07-05",
            "finding": (
                "SOTA telemetry ready without quality claim; internal/logit "
                "hallucination harmful relative to lexical baseline; solver fixture "
                "rebuilt; SOTA extraction blocked by GGUF offload; governed memory "
                "and verifier dosing positive; KAN certificate positive; factor "
                "boundary tiny; hardware blocked with no speedup; evidence audit complete."
            ),
            "tasks": [
                {
                    "id": source.task_id,
                    "deliverable": str(source.relative_path),
                    "result": "OK (archive source)",
                }
                for source in UPSTREAM_SOURCES
            ],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return True


def research_complete_check(root: Path, *, update_research_complete: bool) -> JsonDict:
    count_before = research_complete_milestone_count(root)
    updated = False
    count_after = count_before
    if count_before == 0 and update_research_complete:
        updated = append_research_complete_milestone(root)
        count_after = research_complete_milestone_count(root)
    return {
        "path": "research-complete.yaml",
        "milestone": ARCHIVED_MILESTONE,
        "count_before": count_before,
        "count_after": count_after,
        "had_2026_07_482_before": count_before > 0,
        "has_2026_07_482_after": count_after > 0,
        "updated": updated,
        "sha256": file_sha256(root / "research-complete.yaml"),
    }


def _roadmap_data(text: str) -> JsonDict:
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _roadmap_from_path(path: Path) -> tuple[JsonDict, str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return _roadmap_data(text), text


def _task_ids(roadmap: JsonMap) -> list[str]:
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [str(task.get("id", "")) for task in tasks if isinstance(task, Mapping)]


def _missing_task_prefixes(task_ids: Sequence[str]) -> list[str]:
    return [
        prefix
        for prefix in REQUIRED_483_TASK_PREFIXES
        if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]


def roadmap_activation_check(root: Path) -> JsonDict:
    active, active_text = _roadmap_from_path(root / ROADMAP_RELATIVE_PATH)
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    next_roadmap, next_text = _roadmap_from_path(next_path)
    vnext_path = root / VNEXT_RELATIVE_PATH
    vnext_text = vnext_path.read_text(encoding="utf-8") if vnext_path.exists() else ""
    active_task_ids = _task_ids(active)
    next_task_ids = _task_ids(next_roadmap)
    active_missing = _missing_task_prefixes(active_task_ids)
    next_missing = _missing_task_prefixes(next_task_ids) if next_path.exists() else []
    active_ready = active.get("milestone") == ACTIVATION_MILESTONE and not active_missing
    next_ready = (
        next_path.exists()
        and next_roadmap.get("milestone") == ACTIVATION_MILESTONE
        and not next_missing
    )
    vnext_ready = ACTIVATION_MILESTONE in vnext_text
    if next_path.exists():
        absence_handled_by = "not_absent"
    elif active_ready:
        absence_handled_by = "active_roadmap_already_483"
    else:
        absence_handled_by = "missing_not_covered"
    return {
        "principle": FIELD_PRINCIPLES["roadmap_activation_check"],
        "activated": False,
        "active_roadmap_modified": False,
        "vnext_present": bool(vnext_text),
        "vnext_names_2026_07_483": vnext_ready,
        "roadmap_next_present": next_path.exists(),
        "roadmap_next_absence_handled_by": absence_handled_by,
        "roadmap_next_milestone": next_roadmap.get("milestone"),
        "roadmap_next_task_ids": next_task_ids,
        "roadmap_next_missing_task_prefixes": next_missing,
        "active_roadmap_milestone": active.get("milestone"),
        "active_roadmap_task_ids": active_task_ids,
        "active_roadmap_missing_task_prefixes": active_missing,
        "active_roadmap_already_483": active_ready,
        "activation_ready_without_overwrite": bool(vnext_ready and (active_ready or next_ready)),
        "active_roadmap_sha256": text_sha256(active_text) if active_text else None,
        "roadmap_next_sha256": text_sha256(next_text) if next_text else None,
    }


def _command_label(command: str) -> str:
    if "roadmap_schema.py" in command:
        return "scripts/roadmap_schema.py"
    if "validate_prior_failures.py" in command:
        return "scripts/validate_prior_failures.py"
    if "audit_roadmap_gates.py" in command:
        return "scripts/audit_roadmap_gates.py"
    if "exclusion_manifest_lint.py" in command:
        return "scripts/exclusion_manifest_lint.py"
    return command.split()[0] if command.split() else "unknown_command"


def run_command(command: tuple[str, ...], root: Path) -> CommandResult:
    result = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    return CommandResult(
        command=command,
        exit_code=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def validation_commands(root: Path) -> list[tuple[str, ...]]:
    candidates: tuple[tuple[str, ...], ...] = (
        ("scripts/roadmap_schema.py", str(root / ROADMAP_RELATIVE_PATH)),
        ("scripts/validate_prior_failures.py", str(root / ROADMAP_RELATIVE_PATH)),
        (
            "scripts/audit_roadmap_gates.py",
            str(root / ROADMAP_RELATIVE_PATH),
            "--complete",
            str(root / "research-complete.yaml"),
        ),
        ("scripts/exclusion_manifest_lint.py", str(root / ROADMAP_RELATIVE_PATH)),
    )
    commands: list[tuple[str, ...]] = []
    for candidate in candidates:
        script_path = root / candidate[0]
        if script_path.exists():
            commands.append((sys.executable, str(script_path), *candidate[1:]))
    return commands


def run_validation_commands(root: Path) -> list[CommandResult]:
    return [run_command(command, root) for command in validation_commands(root)]


def commands_run_rows(results: Sequence[CommandResult]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for result in results:
        command_text = " ".join(result.command)
        rows.append(
            {
                "command": command_text,
                "command_label": _command_label(command_text),
                "exit_code": result.exit_code,
                "passed": result.exit_code == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return rows


def _commands_passed(rows: Sequence[JsonMap]) -> bool:
    return bool(rows) and all(row.get("passed") is True for row in rows)


def build_honest_verdict(*, milestone_archived: bool, activation_ready: bool) -> str:
    if milestone_archived and activation_ready:
        return (
            "complete: .482 archived and .483 activation-ready; no roadmap overwrite "
            "performed and aggregation_from_upstream_artifacts evidence used."
        )
    return (
        "blocked_archive_482_activate_483: .482 archive or .483 activation-ready "
        "preconditions failed; no roadmap overwrite performed."
    )


def failed_preconditions(
    *,
    closeout_failures: Sequence[str],
    research_complete: JsonMap,
    roadmap: JsonMap,
    commands: Sequence[JsonMap],
) -> list[str]:
    failures = list(closeout_failures)
    if research_complete.get("has_2026_07_482_after") is not True:
        failures.append("research_complete_missing_2026.07.482")
    if roadmap.get("vnext_names_2026_07_483") is not True:
        failures.append("vnext_missing_2026.07.483")
    if roadmap.get("activation_ready_without_overwrite") is not True:
        failures.append("active_or_next_roadmap_not_ready_for_483")
    if not commands:
        failures.append("validation_commands_missing")
    for row in commands:
        if row.get("passed") is not True:
            failures.append(f"validation_failed_{row.get('command_label')}")
    return failures


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260705",
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    update_research_complete: bool = False,
) -> JsonDict:
    start = time.perf_counter()
    artifacts, source_rows = load_upstream_artifacts(root)
    capstone = artifacts.get(5281, {})
    facts = closeout_facts(capstone)
    fact_failures = closeout_fact_failures(capstone)
    complete_check = research_complete_check(root, update_research_complete=update_research_complete)
    roadmap_check = roadmap_activation_check(root)
    command_results = (
        list(validation_results) if validation_results is not None else run_validation_commands(root)
    )
    command_rows = commands_run_rows(command_results)
    failures = failed_preconditions(
        closeout_failures=fact_failures,
        research_complete=complete_check,
        roadmap=roadmap_check,
        commands=command_rows,
    )
    milestone_archived = not fact_failures and complete_check["has_2026_07_482_after"] is True
    activation_ready = (
        milestone_archived
        and roadmap_check["activation_ready_without_overwrite"] is True
        and _commands_passed(command_rows)
    )
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activation_milestone": ACTIVATION_MILESTONE,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "random_seed": RANDOM_SEED,
        "source_artifacts": source_rows,
        "source_context": source_context(root),
        "closeout_facts": facts,
        "closeout_fact_failures": fact_failures,
        "research_complete_check": complete_check,
        "failed_preconditions": failures,
        "milestone_archived": milestone_archived,
        "milestone_archived_principle": FIELD_PRINCIPLES["milestone_archived"],
        "activation_ready": activation_ready,
        "activation_ready_principle": FIELD_PRINCIPLES["activation_ready"],
        "ops_docs_updated": _principled("ops_docs_updated", False),
        "research_complete_updated": _principled(
            "research_complete_updated", complete_check["updated"]
        ),
        "exclusions_checked": _principled("exclusions_checked", _commands_passed(command_rows)),
        "roadmap_activation_check": roadmap_check,
        "commands_run": command_rows,
        "honest_verdict": _principled(
            "honest_verdict",
            build_honest_verdict(
                milestone_archived=milestone_archived, activation_ready=activation_ready
            ),
        ),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("schema or experiment_id mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    for field in PRINCIPLE_WRAPPED_FIELDS:
        wrapped = payload[field]
        if not isinstance(wrapped, Mapping) or wrapped.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle mismatch")
        if "value" not in wrapped:
            raise ValueError(f"{field} missing value")
    verdict = value_of(payload["honest_verdict"])
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if value_of(payload["inference_substrate"]) != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if not isinstance(payload["milestone_archived"], bool):
        raise ValueError("milestone_archived must be a bare bool")
    if payload["milestone_archived_principle"] != FIELD_PRINCIPLES["milestone_archived"]:
        raise ValueError("milestone_archived principle mismatch")
    if not isinstance(payload["activation_ready"], bool):
        raise ValueError("activation_ready must be a bare bool")
    if payload["activation_ready_principle"] != FIELD_PRINCIPLES["activation_ready"]:
        raise ValueError("activation_ready principle mismatch")
    if value_of(payload["ops_docs_updated"]) is not False:
        raise ValueError("ops_docs_updated must be false under the stop rule")
    if not isinstance(value_of(payload["research_complete_updated"]), bool):
        raise ValueError("research_complete_updated must be bool")
    if not isinstance(value_of(payload["exclusions_checked"]), bool):
        raise ValueError("exclusions_checked must be bool")
    roadmap = payload["roadmap_activation_check"]
    if (
        not isinstance(roadmap, Mapping)
        or roadmap.get("principle") != FIELD_PRINCIPLES["roadmap_activation_check"]
    ):
        raise ValueError("roadmap_activation_check principle mismatch")
    if roadmap.get("activated") is not False:
        raise ValueError("roadmap_activation_check.activated must remain false")
    commands = payload["commands_run"]
    if not isinstance(commands, list) or not commands:
        raise ValueError("commands_run must be a non-empty list")
    for row in commands:
        if not isinstance(row, Mapping) or not isinstance(row.get("passed"), bool):
            raise ValueError("commands_run rows must include pass/fail outcomes")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260705",
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    update_research_complete: bool = False,
) -> Path:
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        validation_results=validation_results,
        update_research_complete=update_research_complete,
    )
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, artifact)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default="20260705")
    parser.add_argument("--update-research-complete", action="store_true")
    args = parser.parse_args(argv)
    print(
        run(
            root=args.root,
            run_date=args.run_date,
            update_research_complete=args.update_research_complete,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

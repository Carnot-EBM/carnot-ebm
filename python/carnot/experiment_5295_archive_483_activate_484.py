"""Exp 5295: archive .483 and emit the .484 activation artifact.

Spec refs: REQ-REPORT-5295, SCENARIO-REPORT-5295,
SCENARIO-REPORT-5295-BLOCKED-CLOSEOUT.

This module is a reporting receipt, not a research experiment. It reads the
already-written `.483` artifacts and local operational records, checks that the
`.484` roadmap state is ready without copying any roadmap file, and writes a
durable JSON artifact. Blocked SOTA runtime, gate-skipped quality tasks, null
curriculum evidence, mixed p-bit/CDCL harm, reachability-only hardware, and
timing-retro mismatch evidence stay visible instead of becoming new claims.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5295_archive_483_activate_484.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5294_capstone_v483.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
OPERATIONAL_RETRO_RELATIVE_PATH = Path("results/operational_retro_2026_07_483.json")

EXPERIMENT = "experiment_5295_archive_483_activate_484"
EXPERIMENT_ID = "exp5295-archive-483-activate-484"
ARCHIVED_MILESTONE = "2026.07.483"
ACTIVATION_MILESTONE = "2026.07.484"
SCHEMA = "carnot.experiment_5295_archive_483_activate_484.v1"
RANDOM_SEED = 5295
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = [
    "REQ-REPORT-5295",
    "SCENARIO-REPORT-5295",
    "SCENARIO-REPORT-5295-BLOCKED-CLOSEOUT",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether .483 was archived and "
        ".484 is activation-ready."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts because Exp5295 only reads existing artifacts "
        "and local records."
    ),
    "milestone_archived": (
        "Bare boolean confirming the .483 closeout is represented in durable research records."
    ),
    "activation_ready": (
        "Bare boolean confirming .484 can proceed without overwriting research-roadmap.yaml."
    ),
    "ops_docs_updated": (
        "False when the conductor stop rule delegates ops/status/changelog/traceability "
        "reconciliation."
    ),
    "research_complete_updated": (
        "True only if this workflow appended or reconciled research-complete.yaml; false "
        "when .483 was already present."
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
    "coherence_fixture_ready": True,
    "trace_fixture_ready": True,
    "sota_runtime_offload_blocked": True,
    "sota_quality_gate_skipped": True,
    "memory_attribution_positive": True,
    "coherence_dosing_positive": True,
    "low_order_curriculum_null": True,
    "pbit_cdcl_mixed_positive_with_harm": True,
    "hardware_reachability_no_speedup": True,
    "timing_retro_accounting_mismatch": True,
}

REQUIRED_484_TASK_PREFIXES = tuple(f"exp{idx}" for idx in range(5295, 5307))


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream `.483` artifact cited by the archive receipt."""

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
        5282,
        "exp5282-archive-482-activate-483",
        Path("results/experiment_5282_archive_482_activate_483.json"),
    ),
    UpstreamSource(
        5283,
        "exp5283-sota-source-delta-v483",
        Path("results/experiment_5283_sota_source_delta_v483.json"),
    ),
    UpstreamSource(
        5284,
        "exp5284-sota-runtime-offload-receipt-repair-v483",
        Path("results/experiment_5284_sota_runtime_offload_receipt_repair_v483.json"),
    ),
    UpstreamSource(
        5285,
        "exp5285-knowledge-thought-coherence-fixture-v483",
        Path("results/experiment_5285_knowledge_thought_coherence_fixture_v483.json"),
    ),
    UpstreamSource(
        5286,
        "exp5286-knowledge-thought-coherence-sota-pilot-v483",
        Path("results/experiment_5286_knowledge_thought_coherence_sota_pilot_v483.json"),
    ),
    UpstreamSource(
        5287,
        "exp5287-compilable-trace-dsl-fixture-v483",
        Path("results/experiment_5287_compilable_trace_dsl_fixture_v483.json"),
    ),
    UpstreamSource(
        5288,
        "exp5288-sota-trace-dsl-extraction-gated-v483",
        Path("results/experiment_5288_sota_trace_dsl_extraction_gated_v483.json"),
    ),
    UpstreamSource(
        5289,
        "exp5289-memory-operation-attribution-v483",
        Path("results/experiment_5289_memory_operation_attribution_v483.json"),
    ),
    UpstreamSource(
        5290,
        "exp5290-memory-assisted-coherence-dose-gated-v483",
        Path("results/experiment_5290_memory_assisted_coherence_dose_gated_v483.json"),
    ),
    UpstreamSource(
        5291,
        "exp5291-low-order-factor-certificate-curriculum-v483",
        Path("results/experiment_5291_low_order_factor_certificate_curriculum_v483.json"),
    ),
    UpstreamSource(
        5292,
        "exp5292-pbit-cdcl-factor-guidance-v483",
        Path("results/experiment_5292_pbit_cdcl_factor_guidance_v483.json"),
    ),
    UpstreamSource(
        5293,
        "exp5293-hardware-continuity-reachability-v483",
        Path("results/experiment_5293_hardware_continuity_reachability_v483.json"),
    ),
    UpstreamSource(5294, "exp5294-capstone-v483", CAPSTONE_RELATIVE_PATH),
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
    OPERATIONAL_RETRO_RELATIVE_PATH,
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


def _as_map(value: Any) -> JsonDict:
    raw = value_of(value)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _finding_by_id(rows: Sequence[Any], finding_id: str) -> JsonDict | None:
    for row in rows:
        if isinstance(row, Mapping) and row.get("id") == finding_id:
            return dict(row)
    return None


def _text_blob(root: Path, relative_paths: Sequence[Path]) -> str:
    parts: list[str] = []
    for relative_path in relative_paths:
        path = root / relative_path
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts).lower()


def _operational_retro_mismatch(root: Path) -> bool:
    retro, meta = read_json_mapping(root / OPERATIONAL_RETRO_RELATIVE_PATH)
    text_blob = _text_blob(root, (Path("ops/status.md"), Path("ops/changelog.md")))
    retro_text = json.dumps(retro, sort_keys=True).lower() if meta.get("loadable") is True else ""
    return (
        ("timing" in text_blob and "mismatch" in text_blob)
        and (
            "timing assembly mismatch" in retro_text
            or "timing-accounting mismatch" in retro_text
            or "timing integrity mismatch" in retro_text
        )
        and retro.get("total_wall_time_minutes") == 0
        and retro.get("experiments_completed") == 0
    )


def closeout_facts(capstone: JsonMap, root: Path | None = None) -> JsonDict:
    clean = _as_list(capstone.get("clean_positive_findings"))
    null_or_harmful = _as_list(capstone.get("null_or_harmful_findings"))
    gated_or_blocked = _as_list(capstone.get("gated_or_blocked_findings"))
    summary = _as_map(capstone.get("tasks_summarized"))

    coherence = _finding_by_id(clean, "claim_level_coherence_fixture_ready")
    trace = _finding_by_id(clean, "compilable_trace_dsl_fixture_ready")
    attribution = _finding_by_id(clean, "memory_operation_attribution_ready")
    dosing = _finding_by_id(clean, "memory_assisted_coherence_dosing_positive")
    pbit_positive = _finding_by_id(clean, "pbit_cdcl_aggregate_guidance_positive")
    curriculum_null = _finding_by_id(null_or_harmful, "low_order_curriculum_clean_null")
    pbit_harm = _finding_by_id(null_or_harmful, "pbit_cdcl_misleading_assumption_harm")
    runtime_block = _finding_by_id(gated_or_blocked, "sota_runtime_offload_blocked")
    coherence_gate = _finding_by_id(gated_or_blocked, "claim_level_sota_pilot_gated_skip")
    trace_gate = _finding_by_id(gated_or_blocked, "trace_dsl_sota_extraction_gated_skip")
    hardware_block = _finding_by_id(gated_or_blocked, "hardware_reachability_blocked_no_speedup")
    by_class = _as_map(summary.get("by_classification"))
    pbit_conflicts = _as_map(pbit_positive.get("conflicts_saved") if pbit_positive else {})
    pbit_by_class = _as_map(pbit_conflicts.get("by_class"))

    return {
        "coherence_fixture_ready": coherence is not None
        and coherence.get("coherence_fixture_ready") is True,
        "trace_fixture_ready": trace is not None and trace.get("trace_dsl_ready") is True,
        "sota_runtime_offload_blocked": runtime_block is not None
        and runtime_block.get("sota_offload_ready") is False
        and by_class.get("blocked_precondition") == 2,
        "sota_quality_gate_skipped": coherence_gate is not None
        and trace_gate is not None
        and by_class.get("gated_skip") == 2,
        "memory_attribution_positive": attribution is not None
        and _as_map(attribution.get("attribution_coverage")).get("attributed_cases") == 7,
        "coherence_dosing_positive": dosing is not None
        and _as_map(dosing.get("full_verifier_calls_avoided")).get("vs_always_full") == 4
        and _as_map(dosing.get("unsafe_false_accepts")).get("count") == 0,
        "low_order_curriculum_null": curriculum_null is not None
        and _as_map(curriculum_null.get("certificate_success_by_order")).get(
            "success_advantage_over_shuffled"
        )
        == 0.0,
        "pbit_cdcl_mixed_positive_with_harm": pbit_positive is not None
        and pbit_harm is not None
        and pbit_conflicts.get("aggregate") == 2
        and pbit_by_class.get("misleading_factor_sat") == -1
        and "misleading_factor_sat" in _as_list(pbit_harm.get("harmful_classes")),
        "hardware_reachability_no_speedup": hardware_block is not None
        and hardware_block.get("hardware_speedup_claimed") is False,
        "timing_retro_accounting_mismatch": _operational_retro_mismatch(root)
        if root is not None
        else False,
    }


def closeout_fact_failures(capstone: JsonMap, root: Path | None = None) -> list[str]:
    if not capstone:
        return ["capstone_artifact_missing_or_unloadable"]
    facts = closeout_facts(capstone, root)
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
            "title": "Claim-Level Verification, Memory Attribution, and Solver-Hardware Guidance",
            "doc": str(VNEXT_RELATIVE_PATH),
            "completed": "2026-07-06",
            "finding": (
                "Deterministic coherence and trace fixtures ready; SOTA runtime/offload "
                "blocked; SOTA quality tasks gate-skipped; memory attribution and coherence "
                "dosing positive; low-order curriculum null; p-bit/CDCL aggregate positive "
                "with misleading-class harm; hardware reachability-only with no speedup; "
                "timing-retro accounting mismatch recorded."
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
        "had_2026_07_483_before": count_before > 0,
        "has_2026_07_483_after": count_after > 0,
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
        for prefix in REQUIRED_484_TASK_PREFIXES
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
        absence_handled_by = "active_roadmap_already_484"
    else:
        absence_handled_by = "missing_not_covered"
    return {
        "principle": FIELD_PRINCIPLES["roadmap_activation_check"],
        "activated": False,
        "active_roadmap_modified": False,
        "vnext_present": bool(vnext_text),
        "vnext_names_2026_07_484": vnext_ready,
        "roadmap_next_present": next_path.exists(),
        "roadmap_next_absence_handled_by": absence_handled_by,
        "roadmap_next_milestone": next_roadmap.get("milestone"),
        "roadmap_next_task_ids": next_task_ids,
        "roadmap_next_missing_task_prefixes": next_missing,
        "active_roadmap_milestone": active.get("milestone"),
        "active_roadmap_task_ids": active_task_ids,
        "active_roadmap_missing_task_prefixes": active_missing,
        "active_roadmap_already_484": active_ready,
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
            "complete: .483 archived and .484 activation-ready; no roadmap overwrite "
            "performed and aggregation_from_upstream_artifacts evidence used."
        )
    return (
        "blocked_archive_483_activate_484: .483 archive or .484 activation-ready "
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
    if research_complete.get("has_2026_07_483_after") is not True:
        failures.append("research_complete_missing_2026.07.483")
    if roadmap.get("vnext_names_2026_07_484") is not True:
        failures.append("vnext_missing_2026.07.484")
    if roadmap.get("activation_ready_without_overwrite") is not True:
        failures.append("active_or_next_roadmap_not_ready_for_484")
    if not commands:
        failures.append("validation_commands_missing")
    for row in commands:
        if row.get("passed") is not True:
            failures.append(f"validation_failed_{row.get('command_label')}")
    return failures


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260706",
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    update_research_complete: bool = False,
) -> JsonDict:
    start = time.perf_counter()
    artifacts, source_rows = load_upstream_artifacts(root)
    capstone = artifacts.get(5294, {})
    facts = closeout_facts(capstone, root)
    fact_failures = closeout_fact_failures(capstone, root)
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
    milestone_archived = not fact_failures and complete_check["has_2026_07_483_after"] is True
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
    run_date: str = "20260706",
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
    parser.add_argument("--run-date", default="20260706")
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

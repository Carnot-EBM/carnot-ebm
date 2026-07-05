"""Exp 5269: archive .481 and emit the .482 activation artifact.

Spec refs: REQ-REPORT-5269, SCENARIO-REPORT-5269,
SCENARIO-REPORT-5269-BLOCKED-CLOSEOUT.

This module is a reporting receipt, not a research run. It reads the already
written `.481` artifacts and local ops records, checks that the `.482` roadmap
state is ready without copying any roadmap file, and writes a durable JSON
artifact. The closeout stays deliberately narrow: flagged verifier pilots stay
quarantined, the memory result stays a null, blocked hardware stays blocked, and
the artifact does not add new performance claims.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5269_archive_481_activate_482.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5268_capstone_v481.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

EXPERIMENT = "experiment_5269_archive_481_activate_482"
EXPERIMENT_ID = "exp5269-archive-481-activate-482"
ARCHIVED_MILESTONE = "2026.07.481"
ACTIVATION_MILESTONE = "2026.07.482"
SCHEMA = "carnot.experiment_5269_archive_481_activate_482.v1"
RANDOM_SEED = 5269
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = [
    "REQ-REPORT-5269",
    "SCENARIO-REPORT-5269",
    "SCENARIO-REPORT-5269-BLOCKED-CLOSEOUT",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether .481 was archived and "
        ".482 is activation-ready."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts because Exp5269 only reads existing artifacts "
        "and local records."
    ),
    "milestone_archived": (
        "Bare boolean confirming the .481 closeout is represented in durable research records."
    ),
    "activation_ready": (
        "Bare boolean confirming .482 can proceed without overwriting research-roadmap.yaml."
    ),
    "ops_docs_updated": (
        "False when the conductor stop rule delegates ops/status/changelog/traceability "
        "reconciliation."
    ),
    "research_complete_updated": (
        "True only if this workflow appended or reconciled research-complete.yaml; false "
        "when .481 was already present."
    ),
    "exclusions_checked": (
        "The transition must run or explicitly record available exclusion, prior-failure, "
        "and roadmap checks."
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
    "sota_runtime_ready": True,
    "cross_model_memory_clean_null": True,
    "memory_policy_positive": True,
    "solver_internal_verifier_pilots_quarantined": True,
    "verifier_dose_scheduler_replay_positive": True,
    "kan_refinement_positive": True,
    "hardware_blocked_no_speedup": True,
    "artifact_normalizer_producer_adoption_positive": True,
}

REQUIRED_482_TASK_PREFIXES = tuple(f"exp{idx}" for idx in range(5269, 5282))


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream `.481` artifact cited by the archive receipt."""

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
        5257,
        "exp5257-archive-480-activate-481",
        Path("results/experiment_5257_archive_480_activate_481.json"),
    ),
    UpstreamSource(
        5258,
        "exp5258-sota-refresh-v481",
        Path("results/experiment_5258_sota_refresh_v481.json"),
    ),
    UpstreamSource(
        5259,
        "exp5259-sota-gguf-gpu-offload-preflight-v481",
        Path("results/experiment_5259_sota_gguf_gpu_offload_preflight_v481.json"),
    ),
    UpstreamSource(
        5260,
        "exp5260-cross-model-typed-memory-retry-v481",
        Path("results/experiment_5260_cross_model_typed_memory_retry_v481.json"),
    ),
    UpstreamSource(
        5261,
        "exp5261-typed-memory-interference-audit-v481",
        Path("results/experiment_5261_typed_memory_interference_audit_v481.json"),
    ),
    UpstreamSource(
        5262,
        "exp5262-solver-grounded-constraint-extraction-v481",
        Path("results/experiment_5262_solver_grounded_constraint_extraction_v481.json"),
    ),
    UpstreamSource(
        5263,
        "exp5263-neuron-attention-energy-hallucination-probe-v481",
        Path("results/experiment_5263_neuron_attention_energy_hallucination_probe_v481.json"),
    ),
    UpstreamSource(
        5264,
        "exp5264-verifier-dose-scheduler-replay-v481",
        Path("results/experiment_5264_verifier_dose_scheduler_replay_v481.json"),
    ),
    UpstreamSource(
        5265,
        "exp5265-kan-certificate-explanation-refinement-v481",
        Path("results/experiment_5265_kan_certificate_explanation_refinement_v481.json"),
    ),
    UpstreamSource(
        5266,
        "exp5266-hardware-thermodynamic-schedule-boundary-v481",
        Path("results/experiment_5266_hardware_thermodynamic_schedule_boundary_v481.json"),
    ),
    UpstreamSource(
        5267,
        "exp5267-artifact-normalizer-template-adoption-v481",
        Path("results/experiment_5267_artifact_normalizer_template_adoption_v481.json"),
    ),
    UpstreamSource(5268, "exp5268-capstone-v481", CAPSTONE_RELATIVE_PATH),
)

SOURCE_CONTEXT_PATHS = (
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


def closeout_facts(capstone: JsonMap) -> JsonDict:
    clean_positives = _as_list(capstone.get("clean_positives"))
    clean_nulls = _as_list(capstone.get("clean_nulls"))
    blocked_or_skipped = _as_list(capstone.get("blocked_or_skipped"))
    flagged_rows = [
        row
        for row in blocked_or_skipped
        if isinstance(row, Mapping) and row.get("classification") == "flagged_adversarial"
    ]

    row_5259 = _row_for(clean_positives, 5259)
    row_5260 = _row_for(clean_nulls, 5260)
    row_5261 = _row_for(clean_positives, 5261)
    row_5264 = _row_for(clean_positives, 5264)
    row_5265 = _row_for(clean_positives, 5265)
    row_5266 = _row_for(blocked_or_skipped, 5266)
    row_5267 = _row_for(clean_positives, 5267)

    return {
        "sota_runtime_ready": "runtime preflight ready=True" in _summary(row_5259)
        or "sota_runtime_ready=true" in _summary(row_5259).lower(),
        "cross_model_memory_clean_null": row_5260 is not None
        and "typed memory useful=False" in _summary(row_5260),
        "memory_policy_positive": row_5261 is not None
        and "memory_policy_ready=True" in _summary(row_5261),
        "solver_internal_verifier_pilots_quarantined": {
            row.get("experiment_number") for row in flagged_rows
        }
        >= {5262, 5263},
        "verifier_dose_scheduler_replay_positive": row_5264 is not None
        and "scheduler_ready=True" in _summary(row_5264),
        "kan_refinement_positive": row_5265 is not None
        and "certificate_refinement_ready=True" in _summary(row_5265),
        "hardware_blocked_no_speedup": row_5266 is not None
        and "speedup_claimed=false" in _summary(row_5266),
        "artifact_normalizer_producer_adoption_positive": row_5267 is not None
        and "producer_normalizer_ready=True" in _summary(row_5267),
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


def research_complete_milestone_count(
    root: Path, milestone: str = ARCHIVED_MILESTONE
) -> int:
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
            "title": "Local SOTA Runtime, Internal Verification, and Self-Learning Memory Stability",
            "doc": str(VNEXT_RELATIVE_PATH),
            "completed": "2026-07-05",
            "finding": (
                "SOTA runtime preflight ready; cross-model typed memory clean null; "
                "memory policy, scheduler replay, KAN refinement, and producer "
                "normalizer positive; solver/internal verifier pilots quarantined; "
                "hardware blocked with no speedup claim."
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
        "had_2026_07_481_before": count_before > 0,
        "has_2026_07_481_after": count_after > 0,
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
        for prefix in REQUIRED_482_TASK_PREFIXES
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
    return {
        "principle": FIELD_PRINCIPLES["roadmap_activation_check"],
        "activated": False,
        "active_roadmap_modified": False,
        "vnext_present": bool(vnext_text),
        "vnext_names_2026_07_482": vnext_ready,
        "roadmap_next_present": next_path.exists(),
        "roadmap_next_milestone": next_roadmap.get("milestone"),
        "roadmap_next_task_ids": next_task_ids,
        "roadmap_next_missing_task_prefixes": next_missing,
        "active_roadmap_milestone": active.get("milestone"),
        "active_roadmap_task_ids": active_task_ids,
        "active_roadmap_missing_task_prefixes": active_missing,
        "active_roadmap_already_482": active_ready,
        "activation_ready_without_overwrite": bool(vnext_ready and (active_ready or next_ready)),
        "active_roadmap_sha256": text_sha256(active_text) if active_text else None,
        "roadmap_next_sha256": text_sha256(next_text) if next_text else None,
    }


def _command_label(command: str) -> str:
    if "exclusion_manifest_lint.py" in command:
        return "scripts/exclusion_manifest_lint.py"
    if "check_exclusion_manifest.py" in command:
        return "scripts/check_exclusion_manifest.py"
    if "validate_prior_failures.py" in command:
        return "scripts/validate_prior_failures.py"
    if "audit_roadmap_gates.py" in command:
        return "scripts/audit_roadmap_gates.py"
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
    candidates = (
        ("scripts/exclusion_manifest_lint.py", str(root / ROADMAP_RELATIVE_PATH)),
        ("scripts/check_exclusion_manifest.py", "5269"),
        ("scripts/validate_prior_failures.py", str(root / ROADMAP_RELATIVE_PATH)),
        ("scripts/audit_roadmap_gates.py", str(root / ROADMAP_RELATIVE_PATH)),
    )
    commands: list[tuple[str, ...]] = []
    for script, argument in candidates:
        script_path = root / script
        if script_path.exists():
            commands.append((sys.executable, str(script_path), argument))
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
            "complete: .481 archived and .482 activation-ready; no roadmap overwrite "
            "performed and aggregation_from_upstream_artifacts evidence used."
        )
    return (
        "blocked_archive_481_activate_482: .481 archive or .482 activation-ready "
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
    if research_complete.get("has_2026_07_481_after") is not True:
        failures.append("research_complete_missing_2026.07.481")
    if roadmap.get("vnext_names_2026_07_482") is not True:
        failures.append("vnext_missing_2026.07.482")
    if roadmap.get("activation_ready_without_overwrite") is not True:
        failures.append("active_or_next_roadmap_not_ready_for_482")
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
    capstone = artifacts.get(5268, {})
    facts = closeout_facts(capstone)
    fact_failures = closeout_fact_failures(capstone)
    complete_check = research_complete_check(
        root, update_research_complete=update_research_complete
    )
    roadmap_check = roadmap_activation_check(root)
    command_results = (
        list(validation_results)
        if validation_results is not None
        else run_validation_commands(root)
    )
    command_rows = commands_run_rows(command_results)
    failures = failed_preconditions(
        closeout_failures=fact_failures,
        research_complete=complete_check,
        roadmap=roadmap_check,
        commands=command_rows,
    )
    milestone_archived = not fact_failures and complete_check["has_2026_07_481_after"] is True
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

"""Exp 5257: archive .480 and emit the .481 activation-ready artifact.

Spec refs: REQ-REPORT-5257, SCENARIO-REPORT-5257,
SCENARIO-REPORT-5257-BLOCKED-CLOSEOUT.

This module is a closeout receipt builder. It reads already-written `.480`
artifacts and local ops records, checks that the next milestone is ready, and
writes a JSON artifact. It does not run a model and it does not copy
`research-roadmap-next.yaml` over the active roadmap.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5257_archive_480_activate_481.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5256_capstone_v480.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

EXPERIMENT = "experiment_5257_archive_480_activate_481"
EXPERIMENT_ID = "exp5257-archive-480-activate-481"
ARCHIVED_MILESTONE = "2026.07.480"
ACTIVATION_MILESTONE = "2026.07.481"
SCHEMA = "carnot.experiment_5257_archive_480_activate_481.v1"
RANDOM_SEED = 5257
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = [
    "REQ-REPORT-5257",
    "SCENARIO-REPORT-5257",
    "SCENARIO-REPORT-5257-BLOCKED-CLOSEOUT",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether .480 was archived and "
        ".481 is activation-ready."
    ),
    "inference_substrate": (
        "cached_fixture_replay_no_llm because Exp5257 only reads existing artifacts and "
        "local records."
    ),
    "milestone_archived": (
        "Bare boolean confirming the .480 closeout is represented in durable research records."
    ),
    "activation_ready": (
        "Bare boolean confirming .481 can proceed without overwriting research-roadmap.yaml."
    ),
    "ops_docs_updated": (
        "False when the conductor stop rule delegates ops/status/changelog/traceability "
        "reconciliation."
    ),
    "research_complete_updated": (
        "True only if this workflow appended or reconciled research-complete.yaml; false "
        "when .480 was already present."
    ),
    "exclusions_checked": (
        "The transition must run or explicitly record available exclusion/prior-failure checks."
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
    "artifact_normalizer_ready": True,
    "gap4_final_decision": "salvaged_clean_null",
    "exp5249_runtime_blocked": True,
    "exp5250_gated_skipped": True,
    "token_guard_harmful": True,
    "halluhard_clean_null": True,
    "arc_level_delta": 0,
    "arc_patch_retired": True,
    "kan_bounded_positive": True,
    "hardware_no_speedup": True,
}

REQUIRED_481_TASK_PREFIXES = tuple(f"exp{idx}" for idx in range(5257, 5269))


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream `.480` artifact to cite in the archive receipt."""

    experiment_number: int
    task_id: str
    relative_path: Path


@dataclass(frozen=True)
class CommandResult:
    """Captured command result for validation commands."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        5245,
        "exp5245-archive-479-activate-480",
        Path("results/experiment_5245_archive_479_activate_480.json"),
    ),
    UpstreamSource(
        5246, "exp5246-sota-refresh-v480", Path("results/experiment_5246_sota_refresh_v480.json")
    ),
    UpstreamSource(
        5247,
        "exp5247-slot-artifact-normalizer-v480",
        Path("results/experiment_5247_slot_artifact_normalizer_v480.json"),
    ),
    UpstreamSource(
        5248,
        "exp5248-gap4-receipt-salvage-or-retire-v480",
        Path("results/experiment_5248_gap4_receipt_salvage_or_retire_v480.json"),
    ),
    UpstreamSource(
        5249,
        "exp5249-cross-model-typed-memory-transfer-v480",
        Path("results/experiment_5249_cross_model_typed_memory_transfer_v480.json"),
    ),
    UpstreamSource(
        5250,
        "exp5250-verifier-dose-scheduler-v480",
        Path("results/experiment_5250_verifier_dose_scheduler_v480.json"),
    ),
    UpstreamSource(
        5251,
        "exp5251-token-guard-carnot-pilot-v480",
        Path("results/experiment_5251_token_guard_carnot_pilot_v480.json"),
    ),
    UpstreamSource(
        5252,
        "exp5252-halluhard-provenance-memory-microbench-v480",
        Path("results/experiment_5252_halluhard_provenance_memory_microbench_v480.json"),
    ),
    UpstreamSource(
        5253,
        "exp5253-arc-live-patch-clean-receipts-v480",
        Path("results/experiment_5253_arc_live_patch_clean_receipts_v480.json"),
    ),
    UpstreamSource(
        5254,
        "exp5254-kan-convex-envelope-certificate-v480",
        Path("results/experiment_5254_kan_convex_envelope_certificate_v480.json"),
    ),
    UpstreamSource(
        5255,
        "exp5255-hardware-continuity-pkit-boundary-v480",
        Path("results/experiment_5255_hardware_continuity_pkit_boundary_v480.json"),
    ),
    UpstreamSource(5256, "exp5256-capstone-v480", CAPSTONE_RELATIVE_PATH),
)

SOURCE_CONTEXT_PATHS = (
    Path("research-complete.yaml"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("ops/exclusion_manifest.yaml"),
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


def _roadmap_data(text: str) -> JsonDict:
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _task_ids(roadmap: JsonMap) -> list[str]:
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [str(task.get("id", "")) for task in tasks if isinstance(task, Mapping)]


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
            }
        )
    return artifacts, rows


def _text_field(payload: JsonMap, field: str) -> str:
    return str(value_of(payload.get(field, "")))


def closeout_facts(capstone: JsonMap) -> JsonDict:
    normalizer = _text_field(capstone, "artifact_normalizer_status")
    gap4 = _text_field(capstone, "gap4_final_status")
    memory = _text_field(capstone, "continuous_self_learning_status")
    dose = _text_field(capstone, "verifier_dose_status")
    token_guard = _text_field(capstone, "token_guard_status")
    halluhard = _text_field(capstone, "halluhard_status")
    kan = _text_field(capstone, "kan_certificate_status")
    status_decisions = capstone.get("status_decisions")
    decisions = status_decisions if isinstance(status_decisions, Mapping) else {}
    speedup_claimed = value_of(capstone.get("hardware_speedup_claimed"))
    return {
        "artifact_normalizer_ready": normalizer.startswith("ready:"),
        "gap4_final_decision": "salvaged_clean_null" if "salvaged_clean_null" in gap4 else gap4,
        "exp5249_runtime_blocked": "blocked_llama_cpp_gpu_offload" in memory,
        "exp5250_gated_skipped": dose.startswith("blocked_gate"),
        "token_guard_harmful": "harmful" in token_guard.lower(),
        "halluhard_clean_null": "clean_null" in halluhard,
        "arc_level_delta": value_of(capstone.get("arc_level_delta")),
        "arc_patch_retired": decisions.get("arc_patch")
        == "retire_current_provenance_patch_after_clean_zero_delta",
        "kan_bounded_positive": "bounded_positive" in kan,
        "hardware_no_speedup": speedup_claimed is False
        and "no_speedup_claim" in str(decisions.get("hardware", "")),
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


def research_complete_has_milestone(root: Path, milestone: str = ARCHIVED_MILESTONE) -> bool:
    path = root / "research-complete.yaml"
    if not path.exists():
        return False
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return False
    return any(isinstance(row, Mapping) and row.get("id") == milestone for row in _milestones(data))


def append_research_complete_milestone(root: Path) -> bool:
    path = root / "research-complete.yaml"
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
    if any(isinstance(row, Mapping) and row.get("id") == ARCHIVED_MILESTONE for row in milestones):
        return False
    milestones.append(
        {
            "id": ARCHIVED_MILESTONE,
            "title": "Typed Memory, Receipt Integrity, and Verified Decoding Allocation",
            "doc": str(VNEXT_RELATIVE_PATH),
            "completed": "2026-07-05",
            "finding": "See Exp 5256 capstone and Exp 5257 archive artifact for closeout facts.",
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
    before = research_complete_has_milestone(root)
    updated = False
    after = before
    if not before and update_research_complete:
        updated = append_research_complete_milestone(root)
        after = research_complete_has_milestone(root)
    return {
        "path": "research-complete.yaml",
        "had_2026_07_480_before": before,
        "has_2026_07_480_after": after,
        "updated": updated,
        "sha256": file_sha256(root / "research-complete.yaml"),
    }


def _roadmap_from_path(path: Path) -> tuple[JsonDict, str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return _roadmap_data(text), text


def _missing_task_prefixes(task_ids: Sequence[str]) -> list[str]:
    return [
        prefix
        for prefix in REQUIRED_481_TASK_PREFIXES
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
        "vnext_names_2026_07_481": vnext_ready,
        "roadmap_next_present": next_path.exists(),
        "roadmap_next_milestone": next_roadmap.get("milestone"),
        "roadmap_next_task_ids": next_task_ids,
        "roadmap_next_missing_task_prefixes": next_missing,
        "active_roadmap_milestone": active.get("milestone"),
        "active_roadmap_task_ids": active_task_ids,
        "active_roadmap_missing_task_prefixes": active_missing,
        "active_roadmap_already_481": active_ready,
        "activation_ready_without_overwrite": bool(vnext_ready and (active_ready or next_ready)),
        "active_roadmap_sha256": text_sha256(active_text) if active_text else None,
        "roadmap_next_sha256": text_sha256(next_text) if next_text else None,
    }


def _command_label(command: str) -> str:
    if "exclusion_manifest_lint.py" in command:
        return "scripts/exclusion_manifest_lint.py"
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
        ("scripts/exclusion_manifest_lint.py", ROADMAP_RELATIVE_PATH),
        ("scripts/validate_prior_failures.py", ROADMAP_RELATIVE_PATH),
        ("scripts/audit_roadmap_gates.py", ROADMAP_RELATIVE_PATH),
    )
    commands: list[tuple[str, ...]] = []
    for script, target in candidates:
        script_path = root / script
        if script_path.exists():
            commands.append((sys.executable, str(script_path), str(root / target)))
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


def research_conductor_untouched(root: Path) -> bool:  # pragma: no cover - git integration.
    diff = subprocess.run(
        ["git", "diff", "--quiet", "--", "scripts/research_conductor.py"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--", "scripts/research_conductor.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return diff.returncode == 0 and status.stdout.strip() == ""


def build_honest_verdict(*, milestone_archived: bool, activation_ready: bool) -> str:
    if milestone_archived and activation_ready:
        return (
            "complete: .480 archived and .481 activation-ready; no roadmap overwrite "
            "performed and cached_fixture_replay_no_llm evidence used."
        )
    return (
        "blocked_archive_480_activate_481: .480 archive or .481 activation-ready "
        "preconditions failed; no roadmap overwrite performed."
    )


def failed_preconditions(
    *,
    closeout_failures: Sequence[str],
    research_complete: JsonMap,
    roadmap: JsonMap,
    commands: Sequence[JsonMap],
    conductor_clean: bool,
) -> list[str]:
    failures = list(closeout_failures)
    if research_complete.get("has_2026_07_480_after") is not True:
        failures.append("research_complete_missing_2026.07.480")
    if roadmap.get("vnext_names_2026_07_481") is not True:
        failures.append("vnext_missing_2026.07.481")
    if roadmap.get("activation_ready_without_overwrite") is not True:
        failures.append("active_or_next_roadmap_not_ready_for_481")
    if not commands:
        failures.append("validation_commands_missing")
    for row in commands:
        if row.get("passed") is not True:
            failures.append(f"validation_failed_{row.get('command_label')}")
    if not conductor_clean:
        failures.append("scripts_research_conductor_py_modified")
    return failures


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260705",
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    conductor_untouched: bool | None = None,
    update_research_complete: bool = False,
) -> JsonDict:
    start = time.perf_counter()
    artifacts, source_rows = load_upstream_artifacts(root)
    capstone = artifacts.get(5256, {})
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
    conductor_clean = (
        research_conductor_untouched(root) if conductor_untouched is None else conductor_untouched
    )
    failures = failed_preconditions(
        closeout_failures=fact_failures,
        research_complete=complete_check,
        roadmap=roadmap_check,
        commands=command_rows,
        conductor_clean=conductor_clean,
    )
    milestone_archived = not fact_failures and complete_check["has_2026_07_480_after"] is True
    activation_ready = (
        milestone_archived
        and roadmap_check["activation_ready_without_overwrite"] is True
        and _commands_passed(command_rows)
        and conductor_clean
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
    conductor_untouched: bool | None = None,
    update_research_complete: bool = False,
) -> Path:
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        validation_results=validation_results,
        conductor_untouched=conductor_untouched,
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

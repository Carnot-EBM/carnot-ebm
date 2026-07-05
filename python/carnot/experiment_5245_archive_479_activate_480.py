"""Exp 5245: archive .479 and emit the .480 activation-ready artifact.

Spec refs: REQ-REPORT-5245, SCENARIO-REPORT-5245,
SCENARIO-REPORT-5245-BLOCKED-CLOSEOUT.

This module is deliberately a record builder, not a research experiment. It
reads the already-written `.479` artifacts and ops records, checks the next
milestone handoff, and writes a JSON receipt. It does not run a model and it
does not copy `research-roadmap-next.yaml` over the active roadmap.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5245_archive_479_activate_480.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5244_capstone_v479.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

EXPERIMENT = "experiment_5245_archive_479_activate_480"
EXPERIMENT_ID = "exp5245-archive-479-activate-480"
ARCHIVED_MILESTONE = "2026.07.479"
ACTIVATION_MILESTONE = "2026.07.480"
SCHEMA = "carnot.experiment_5245_archive_479_activate_480.v1"
RANDOM_SEED = 5245
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = [
    "REQ-REPORT-5245",
    "SCENARIO-REPORT-5245",
    "SCENARIO-REPORT-5245-BLOCKED-CLOSEOUT",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether .479 was archived and "
        ".480 is activation-ready."
    ),
    "inference_substrate": (
        "cached_fixture_replay_no_llm because Exp5245 only reads existing artifacts and "
        "local records."
    ),
    "milestone_archived": (
        "Bare boolean confirming the .479 closeout is represented in durable research records."
    ),
    "activation_ready": (
        "Bare boolean confirming .480 can proceed without overwriting research-roadmap.yaml."
    ),
    "ops_docs_updated": (
        "False when the conductor stop rule delegates ops/status/changelog/traceability "
        "reconciliation."
    ),
    "research_complete_updated": (
        "True only if this workflow appended or reconciled research-complete.yaml; false "
        "when .479 was already present."
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
    "gap4_final_status": "blocked",
    "gap1_final_status": "blocked",
    "veribmc_final_status": "retired",
    "continuous_self_learning_status": "controlled_positive",
    "arc_level_delta": 0,
    "kan_certificate_status": "extended",
    "hardware_speedup_claimed": False,
}

REQUIRED_480_TASK_PREFIXES = tuple(f"exp{idx}" for idx in range(5245, 5257))


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream `.479` artifact to cite in the archive receipt."""

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
        5233,
        "exp5233-archive-478-activate-479",
        Path("results/experiment_5233_archive_478_activate_479.json"),
    ),
    UpstreamSource(
        5234,
        "exp5234-sota-ingestion-v479",
        Path("results/experiment_5234_sota_ingestion_v479.json"),
    ),
    UpstreamSource(
        5235,
        "exp5235-adversarial-qa-null-tautology-calibration-v479",
        Path("results/experiment_5235_adversarial_qa_null_tautology_calibration_v479.json"),
    ),
    UpstreamSource(
        5236,
        "exp5236-gap4-clean-status-after-qa-calibration-v479",
        Path("results/experiment_5236_gap4_clean_status_after_qa_calibration_v479.json"),
    ),
    UpstreamSource(
        5237,
        "exp5237-gap1-stability-freeze-or-retire-v479",
        Path("results/experiment_5237_gap1_stability_freeze_or_retire_v479.json"),
    ),
    UpstreamSource(
        5238,
        "exp5238-veribmc-methodology-correct-rerun-or-retire-v479",
        Path("results/experiment_5238_veribmc_methodology_correct_rerun_or_retire_v479.json"),
    ),
    UpstreamSource(
        5239,
        "exp5239-continuous-self-learning-controlled-memory-ablation-v479",
        Path(
            "results/experiment_5239_continuous_self_learning_controlled_memory_ablation_v479.json"
        ),
    ),
    UpstreamSource(
        5240,
        "exp5240-arc-rubric-to-patch-synthesis-v479",
        Path("results/experiment_5240_arc_rubric_to_patch_synthesis_v479.json"),
    ),
    UpstreamSource(
        5241,
        "exp5241-arc-gated-live-patch-attempt-v479",
        Path("results/experiment_5241_arc_gated_live_patch_attempt_v479.json"),
    ),
    UpstreamSource(
        5242,
        "exp5242-kan-certificate-abstraction-scale-v479",
        Path("results/experiment_5242_kan_certificate_abstraction_scale_v479.json"),
    ),
    UpstreamSource(
        5243,
        "exp5243-hardware-continuity-kan-pbit-boundary-v479",
        Path("results/experiment_5243_hardware_continuity_kan_pbit_boundary_v479.json"),
    ),
    UpstreamSource(5244, "exp5244-capstone-v479", CAPSTONE_RELATIVE_PATH),
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


def closeout_facts(capstone: JsonMap) -> JsonDict:
    return {field: value_of(capstone.get(field)) for field in EXPECTED_CLOSEOUT_FACTS}


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
            "title": "ARTIFACT CREDIBILITY + CONTROLLED SELF-LEARNING + VERIFIER DECISION REPAIR",
            "doc": str(VNEXT_RELATIVE_PATH),
            "completed": "2026-07-04",
            "finding": "See Exp 5244 capstone and Exp 5245 archive artifact for closeout facts.",
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
        "had_2026_07_479_before": before,
        "has_2026_07_479_after": after,
        "updated": updated,
        "sha256": file_sha256(root / "research-complete.yaml"),
    }


def _roadmap_from_path(path: Path) -> tuple[JsonDict, str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return _roadmap_data(text), text


def _missing_task_prefixes(task_ids: Sequence[str]) -> list[str]:
    return [
        prefix
        for prefix in REQUIRED_480_TASK_PREFIXES
        if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]


def roadmap_activation_check(root: Path) -> JsonDict:
    active, active_text = _roadmap_from_path(root / ROADMAP_RELATIVE_PATH)
    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    next_roadmap, next_text = _roadmap_from_path(next_path)
    vnext_text = (
        (root / VNEXT_RELATIVE_PATH).read_text(encoding="utf-8")
        if (root / VNEXT_RELATIVE_PATH).exists()
        else ""
    )
    active_task_ids = _task_ids(active)
    next_task_ids = _task_ids(next_roadmap)
    active_missing = _missing_task_prefixes(active_task_ids)
    next_missing = _missing_task_prefixes(next_task_ids) if next_path.exists() else []
    active_ready = active.get("milestone") == ACTIVATION_MILESTONE and not active_missing
    next_ready = next_path.exists() and next_roadmap.get("milestone") == ACTIVATION_MILESTONE
    if next_ready:
        next_ready = not next_missing
    vnext_ready = ACTIVATION_MILESTONE in vnext_text
    return {
        "principle": FIELD_PRINCIPLES["roadmap_activation_check"],
        "activated": False,
        "active_roadmap_modified": False,
        "vnext_present": bool(vnext_text),
        "vnext_names_2026_07_480": vnext_ready,
        "roadmap_next_present": next_path.exists(),
        "roadmap_next_milestone": next_roadmap.get("milestone"),
        "roadmap_next_task_ids": next_task_ids,
        "roadmap_next_missing_task_prefixes": next_missing,
        "active_roadmap_milestone": active.get("milestone"),
        "active_roadmap_task_ids": active_task_ids,
        "active_roadmap_missing_task_prefixes": active_missing,
        "active_roadmap_already_480": active_ready,
        "activation_ready_without_overwrite": bool(vnext_ready and (active_ready or next_ready)),
        "active_roadmap_sha256": text_sha256(active_text) if active_text else None,
        "roadmap_next_sha256": text_sha256(next_text) if next_text else None,
    }


def _command_label(command: str) -> str:
    if "exclusion_manifest_lint.py" in command:
        return "scripts/exclusion_manifest_lint.py"
    if "validate_prior_failures.py" in command:
        return "scripts/validate_prior_failures.py"
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
    commands: list[tuple[str, ...]] = []
    exclusion = root / "scripts/exclusion_manifest_lint.py"
    prior = root / "scripts/validate_prior_failures.py"
    if exclusion.exists():
        commands.append((sys.executable, str(exclusion), str(root / ROADMAP_RELATIVE_PATH)))
    if prior.exists():
        commands.append((sys.executable, str(prior), str(root / ROADMAP_RELATIVE_PATH)))
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
            "complete: .479 archived and .480 activation-ready; no roadmap overwrite "
            "performed and cached_fixture_replay_no_llm evidence used."
        )
    return (
        "blocked_archive_479_activate_480: .479 archive or .480 activation-ready "
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
    if research_complete.get("has_2026_07_479_after") is not True:
        failures.append("research_complete_missing_2026.07.479")
    if roadmap.get("vnext_names_2026_07_480") is not True:
        failures.append("vnext_missing_2026.07.480")
    if roadmap.get("activation_ready_without_overwrite") is not True:
        failures.append("active_or_next_roadmap_not_ready_for_480")
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
    run_date: str = "20260704",
    duration_s: float | None = None,
    validation_results: Sequence[CommandResult] | None = None,
    conductor_untouched: bool | None = None,
    update_research_complete: bool = False,
) -> JsonDict:
    start = time.perf_counter()
    artifacts, source_rows = load_upstream_artifacts(root)
    capstone = artifacts.get(5244, {})
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
    milestone_archived = not fact_failures and complete_check["has_2026_07_479_after"] is True
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
    run_date: str = "20260704",
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
    parser.add_argument("--run-date", default="20260704")
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

"""Archive .385, activate .386, and record the TRM-training handoff.

Spec refs: REQ-REPORT-4166, SCENARIO-REPORT-4166,
SCENARIO-REPORT-4166-BLOCKED-PRECONDITION.

This is a record-only transition. It preserves the operational decision that
the conductor must stand down on TRM Sudoku training because bounded conductor
tasks no-op'd and then collided with the detached outer-loop contiguous run.
The next planner needs that fact as machine-readable state so it does not emit
another competing training task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.385"
ACTIVATED_MILESTONE = "2026.06.386"
RANDOM_SEED = 4166
OUTPUT_REL_PATH = Path("results/experiment_4166_archive_v385_activate_v386.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
NANO_TRM_TRAIN_REL_PATH = Path("nano-trm/src/nn/train.py")
CAPSTONE_REL_PATH = Path("results/experiment_4165_capstone_v385.json")
OUTERLOOP_PID_REL_PATH = Path("results/trm_runs/contiguous_run.pid")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v385_to_v386_4166.v1"
EXPERIMENT_ID = "exp4166"
TASK_ID = "exp4166-archive-v385-activate-v386"

BASELINE_SEED_VAL = 0.278172343969
TOTAL_GAMES_SOLVED_DEFAULT = 13

CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

V385_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "4157",
        "deliverable": "results/experiment_4157_baseline_harvest_contiguous_continue.json",
    },
    {
        "experiment_id": "4158",
        "deliverable": "results/experiment_4158_verifier_rerank_recovery_moat.json",
    },
    {
        "experiment_id": "4159",
        "deliverable": "results/experiment_4159_decisive_verifier_reward_graft.json",
    },
    {
        "experiment_id": "4160",
        "deliverable": "results/experiment_4160_arc_action_efficiency_harness.json",
    },
    {
        "experiment_id": "4161",
        "deliverable": "results/experiment_4161_observability_timing_detector_fix.json",
    },
    {
        "experiment_id": "4162",
        "deliverable": "results/experiment_4162_sota_ingestion_verifier_moat_guidance.json",
    },
    {
        "experiment_id": "4163",
        "deliverable": "results/experiment_4163_verifier_registry_gaps_hygiene.json",
    },
    {
        "experiment_id": "4164",
        "deliverable": "results/experiment_4164_hardware_continuity.json",
    },
    {
        "experiment_id": "4165",
        "deliverable": str(CAPSTONE_REL_PATH),
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "nano_trm_train_present",
    "pretest_suite_green",
    "v385_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.385.",
    "activated_milestone": "Confirms .386 is live for outer-loop-owned TRM training.",
    "active_milestone_confirmed": "Records the active roadmap milestone actually observed.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "nano_trm_train_present": "Bare bool: nano-trm/src/nn/train.py exists before activation.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v385_close_state": (
        "Honest record (conductor ceded TRM training to the outer-loop after collisions) "
        "so the next planner does not re-generate competing training tasks."
    ),
    "preconditions_checked": (
        "Records resources verified; pre-empts the silent-missing-resource fabrication mode."
    ),
    "honest_verdict": (
        "Self-declared terminal state. MUST start with complete:/success:/passed:/shipped:."
    ),
    "duration_s": "Positive bare wall-clock for this record-only aggregation.",
    "inference_substrate": "Declares aggregation only; no live training happens in this task.",
}

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.385['\"]?\s*$")


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for one required precondition command."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def yaml_parses(text: str) -> bool:
    """Return true when PyYAML can safe-load the supplied text."""

    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return True


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Compute a positive duration so blocked artifacts still carry timing."""

    if started_s is None:
        return 0.0001
    end_s = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, end_s - float(started_s)), 6)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 checksum over artifact content."""

    filtered = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return true when value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a trailing newline."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _milestone_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def read_active_milestone(root: Path) -> tuple[str, str]:
    """Return the active milestone and the roadmap path used."""

    for rel_path in (Path("research-roadmap.yaml"), Path("research-roadmap-next.yaml")):
        path = root / rel_path
        if path.exists():
            milestone = _milestone_from_text(path.read_text(encoding="utf-8"))
            if milestone != "unknown":
                return milestone, str(rel_path)
    return "unknown", "research-roadmap.yaml"


def train_file_present(root: Path) -> bool:
    """Return true when the nano-TRM trainer entrypoint exists."""

    return (root / NANO_TRM_TRAIN_REL_PATH).exists()


def archive_record_count(text: str) -> int:
    """Count top-level `.385` archive records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _record_id(line: str) -> str | None:
    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.385` archive finding from the stand-down close-state."""

    seed = _number(close_state.get("baseline_seed_val_exact_accuracy"), BASELINE_SEED_VAL)
    return (
        ".385 close-state: bounded accumulation no-op'd again and the conductor continuation "
        "collided with the detached outer-loop run on shared GPU/checkpoint paths. "
        f"The baseline seed remains {seed:.12f}; Exp4157 reported blocked_noop_step_unchanged "
        "with no manual_lr_step advance and no new validation row. Per operator decision A "
        "(2026-06-13), the conductor STANDS DOWN on TRM training: the outer-loop owns the "
        "contiguous training. .386 tasks must not launch TRM training, kill train.py, or write "
        "results/trm_runs/sudoku_extreme_baseline/."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.385` record for the absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .385 and activate .386; conductor stands down on TRM training')}",
        "  completed: '2026-06-13'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4166-archive-v385-activate-v386",
        "  tasks:",
        "  - id: exp4157-baseline-harvest-contiguous-continue",
        "    result: 'blocked_noop_step_unchanged; conductor continuation collided with outer-loop run'",
        "  - id: exp4165-capstone-v385",
        "    result: 'accumulation_still_blocked; .386 cedes TRM training to outer-loop'",
    ]
    return "\n".join(lines) + "\n"


def _insert_before_tasks(lines: list[str], new_line: str) -> list[str]:
    for index, line in enumerate(lines):
        if line == "  tasks:":
            return lines[:index] + [new_line] + lines[index:]
    return lines + [new_line]


def _canonicalize_target_span(lines: list[str], close_state: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    finding_written = False
    activation_written = False
    for line in lines:
        if line.startswith("  finding:"):
            if not finding_written:
                out.append(f"  finding: {_yaml_quote(canonical_finding(close_state))}")
                finding_written = True
            continue
        if line.startswith("  activation_recorded:"):
            if not activation_written:
                out.append("  activation_recorded: exp4166-archive-v385-activate-v386")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4166-archive-v385-activate-v386")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.385` record exists and carries the handoff."""

    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _record_id(line) is not None]
    spans: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        spans.append((start, end))
    target_spans = [(start, end) for start, end in spans if _record_id(lines[start]) == ARCHIVED_MILESTONE]
    if not target_spans:
        return f"{text.rstrip()}\n{build_canonical_record(close_state)}", 0, "appended"

    first_start, first_end = target_spans[0]
    remove: set[int] = set()
    for start, end in target_spans[1:]:
        remove.update(range(start, end))
    replacement = _canonicalize_target_span(lines[first_start:first_end], close_state)
    rebuilt: list[str] = []
    for index, line in enumerate(lines):
        if first_start <= index < first_end:
            if index == first_start:
                rebuilt.extend(replacement)
            continue
        if index in remove:
            continue
        rebuilt.append(line)
    new_text = "\n".join(rebuilt)
    if len(target_spans) > 1:
        return new_text, len(target_spans) - 1, "deduped"
    if new_text != text:
        return new_text, 0, "updated"
    return text, 0, "unchanged"


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object from disk, returning empty dict on absence or bad shape."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def file_sha256(path: Path) -> str | None:
    """Return file SHA-256, or None when the file is absent."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def read_v385_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.385` close-state."""

    out: dict[str, JsonDict] = {}
    for source in V385_SOURCE_ARTIFACTS:
        out[str(source["experiment_id"])] = read_json_object(root / str(source["deliverable"]))
    return out


def read_outerloop_pid(root: Path) -> int | None:
    """Return the recorded outer-loop contiguous-run PID when present."""

    try:
        digits = "".join(ch for ch in (root / OUTERLOOP_PID_REL_PATH).read_text(encoding="utf-8") if ch.isdigit())
    except OSError:
        return None
    return int(digits) if digits else None


def pid_is_alive(pid: int | None) -> bool:
    """Return true when a recorded local PID still exists."""

    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for upstream `.385` artifacts."""

    cited: list[JsonDict] = []
    for source in V385_SOURCE_ARTIFACTS:
        rel = str(source["deliverable"])
        cited.append(
            {
                "kind": "artifact",
                "experiment_id": str(source["experiment_id"]),
                "deliverable": rel,
                "sha256": file_sha256(root / rel),
            }
        )
    return cited


def build_v385_close_state(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    outerloop_pid: int | None,
    outerloop_alive: bool,
) -> JsonDict:
    """Build the honest `.385` close-state from artifacts and the operator handoff."""

    baseline = _mapping(sources.get("4157", {}))
    rerank = _mapping(sources.get("4158", {}))
    graft = _mapping(sources.get("4159", {}))
    registry = _mapping(sources.get("4163", {}))
    capstone = _mapping(sources.get("4165", {}))
    task_run = _mapping(baseline.get("task_launched_run"))
    liveness = _mapping(baseline.get("liveness"))
    trajectory = _mapping(capstone.get("baseline_val_trajectory"))
    headline_answers = _mapping(capstone.get("headline_answers"))
    seed = _number(trajectory.get("seed_val"), _number(_mapping(headline_answers.get("current_val_vs_seed")).get("seed_val"), BASELINE_SEED_VAL))
    manual_advanced = bool(task_run.get("manual_lr_step_advanced", False))
    new_val_row = bool(task_run.get("new_val_row_written", False))
    flagged = capstone.get("flagged_artifacts_skipped", [])
    flagged_items = [dict(item) for item in flagged] if isinstance(flagged, list) else []
    flagged_ids = [
        int(item["experiment_id"])
        for item in flagged_items
        if isinstance(item.get("experiment_id"), int | float | str) and str(item.get("experiment_id")).isdigit()
    ]
    total_games = int(_number(capstone.get("total_games_solved"), _number(headline_answers.get("total_games_solved"), TOTAL_GAMES_SOLVED_DEFAULT)))

    return {
        "summary": "bounded_accumulation_noop_collision_outer_loop_owns_training",
        "operator_decision": "A",
        "operator_decision_date": "2026-06-13",
        "conductor_ceded_trm_training_to_outer_loop": True,
        "outer_loop_owns_contiguous_training": True,
        "conductor_stands_down_on_trm_training": True,
        "no_conductor_training_rule": True,
        "forbidden_conductor_actions": [
            "launch_trm_training",
            "pkill_or_kill_train_py",
            "write_stable_checkpoint_dir",
        ],
        "bounded_accumulation_noop_again": not manual_advanced and not new_val_row,
        "conductor_continuation_collided_with_outer_loop": True,
        "collision_evidence": str(baseline.get("blocked_cause", "")),
        "exp4157_honest_verdict": str(baseline.get("honest_verdict", "")),
        "exp4157_flagged_adversarial": bool(baseline.get("flagged_adversarial", False)),
        "task_launched_native_trainer": bool(task_run),
        "task_launched_pid": task_run.get("process_pid", liveness.get("pid")),
        "task_launched_return_code": task_run.get("return_code"),
        "manual_lr_step_before": task_run.get("manual_lr_step_before"),
        "manual_lr_step_after": task_run.get("manual_lr_step_after"),
        "manual_lr_step_advanced": manual_advanced,
        "new_val_row_written": new_val_row,
        "val_rows_before": task_run.get("val_rows_before"),
        "val_rows_after": task_run.get("val_rows_after"),
        "outerloop_pid": outerloop_pid,
        "outerloop_train_alive_at_archive": outerloop_alive,
        "baseline_seed_val_exact_accuracy": seed,
        "flagged_current_val_ignored_for_close_state": baseline.get("current_val"),
        "headline_outcome": str(capstone.get("headline_outcome", "")),
        "capstone_honest_verdict": str(capstone.get("honest_verdict", "")),
        "baseline_val_trajectory_status": str(trajectory.get("status", "")),
        "flagged_artifacts_skipped_count": len(flagged_items),
        "flagged_artifact_ids": flagged_ids,
        "rerank_status": "skipped_flagged_adversarial"
        if bool(rerank.get("flagged_adversarial", False))
        else str(rerank.get("honest_verdict", "")),
        "graft_deferred": bool(graft.get("graft_deferred", True)),
        "verifier_value_added": bool(graft.get("verifier_value_added", False)),
        "registry_regression_guard_passed": bool(registry.get("regression_guard_passed", False)),
        "diffusiongemma_gate_status": str(capstone.get("diffusiongemma_gate_status", "")),
        "total_games_solved": total_games,
        "v386_coordination_rule": (
            "Training-related conductor tasks are read-only monitors until the outer-loop run stops."
        ),
    }


def _run_command(command: list[str], root: Path) -> CommandResult:
    try:
        completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    except OSError as exc:
        return CommandResult(command=command, exit_code=127, stdout="", stderr=str(exc))
    return CommandResult(
        command=command,
        exit_code=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def smart_subset_targets(root: Path) -> list[str]:
    """Return existing smart-subset targets, or the first core target as fallback."""

    targets = [target for target in CORE_SMART_SUBSET if (root / target).exists()]
    return targets or [CORE_SMART_SUBSET[0]]


def smart_subset_command(targets: Sequence[str]) -> list[str]:
    """Return the smart-subset pytest command."""

    return [str(PYTEST_BIN), *targets, "-q", "--no-header", "-n", "0", "--no-cov", "-o", "addopts="]


def run_smart_subset(root: Path) -> CommandResult:
    """Run the smart-subset pre-test gate once."""

    return _run_command(smart_subset_command(smart_subset_targets(root)), root)


def terminal_verdict(v385_close_state: Mapping[str, Any]) -> str:
    """Return the complete-path verdict for the archive transition."""

    seed = _number(v385_close_state.get("baseline_seed_val_exact_accuracy"), BASELINE_SEED_VAL)
    return (
        "success: archived_v385_v386_active_conductor_stands_down_"
        f"outer_loop_owns_trm_training_seed_{seed:.12f}_pretest_green"
    )


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    nano_trm_train_present: bool,
    pretest_suite_green: bool,
    v385_close_state: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_milestone_confirmed: str,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": active_milestone_confirmed,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "nano_trm_train_present": nano_trm_train_present,
        "pretest_suite_green": pretest_suite_green,
        "v385_close_state": dict(v385_close_state),
        "preconditions_checked": dict(preconditions_checked),
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(reason: str, **kwargs: Any) -> JsonDict:
    """Build a blocked artifact without fabricating green resources."""

    defaults: JsonDict = {
        "research_complete_yaml_parses": False,
        "exclusion_manifest_parses": False,
        "nano_trm_train_present": False,
        "pretest_suite_green": False,
        "v385_close_state": {"status": "blocked", "reason": reason},
        "active_milestone_confirmed": "",
        "active_roadmap_path": "research-roadmap.yaml",
        "research_complete_record_action": "none",
        "research_complete_duplicates_removed": 0,
        "cited_upstream_artifacts": [],
    }
    defaults.update(kwargs)
    return _base_payload(honest_verdict=reason, **defaults)


def build_complete_artifact(**kwargs: Any) -> JsonDict:
    """Build and validate the Exp 4166 complete artifact."""

    close_state = kwargs["v385_close_state"]
    payload = _base_payload(
        honest_verdict=terminal_verdict(close_state),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        nano_trm_train_present=True,
        pretest_suite_green=True,
        active_milestone_confirmed=ACTIVATED_MILESTONE,
        **kwargs,
    )
    validate_artifact(payload)
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate fields that stop this archive from laundering the `.385` truth."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    for field in ("honest_verdict", "v385_close_state", "preconditions_checked"):
        if principles.get(field) != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle must match REQ-REPORT-4166")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived milestone must be 2026.06.385")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated milestone must be 2026.06.386")
    if artifact.get("research_complete_yaml_parses") is not True:
        raise ValueError("research-complete YAML must parse")
    if artifact.get("exclusion_manifest_parses") is not True:
        raise ValueError("exclusion manifest must parse")
    if artifact.get("nano_trm_train_present") is not True:
        raise ValueError("nano-trm train file must be present")
    if artifact.get("pretest_suite_green") is not True:
        raise ValueError("pretest suite must be green")
    if artifact.get("active_milestone_confirmed") != ACTIVATED_MILESTONE:
        raise ValueError("active milestone must be confirmed as 2026.06.386")
    close_state = artifact.get("v385_close_state")
    if not isinstance(close_state, Mapping):
        raise ValueError("v385_close_state must be a mapping")
    if close_state.get("conductor_ceded_trm_training_to_outer_loop") is not True:
        raise ValueError("conductor_ceded_trm_training_to_outer_loop must be True")
    if close_state.get("outer_loop_owns_contiguous_training") is not True:
        raise ValueError("outer_loop_owns_contiguous_training must be True")
    if close_state.get("conductor_stands_down_on_trm_training") is not True:
        raise ValueError("conductor_stands_down_on_trm_training must be True")
    if close_state.get("bounded_accumulation_noop_again") is not True:
        raise ValueError("bounded_accumulation_noop_again must be True")
    if close_state.get("conductor_continuation_collided_with_outer_loop") is not True:
        raise ValueError("conductor_continuation_collided_with_outer_loop must be True")
    if round(_number(close_state.get("baseline_seed_val_exact_accuracy"), 0.0), 12) != round(BASELINE_SEED_VAL, 12):
        raise ValueError("baseline seed must remain 0.278172343969")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("duration_s must be positive")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    root: Path | str = REPO_ROOT,
    *,
    pretest_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Run the `.385` archive and `.386` activation guard."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    manifest_path = root_path / EXCLUSION_MANIFEST_REL_PATH
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    complete_exists = complete_path.exists()
    complete_text = complete_path.read_text(encoding="utf-8") if complete_exists else ""
    complete_parses = complete_exists and yaml_parses(complete_text)
    manifest_exists = manifest_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8") if manifest_exists else ""
    manifest_parses = manifest_exists and yaml_parses(manifest_text)
    has_train = train_file_present(root_path)

    preconditions: JsonDict = {
        "research_complete_yaml_exists": complete_exists,
        "research_complete_yaml_parses": complete_parses,
        "exclusion_manifest_exists": manifest_exists,
        "exclusion_manifest_parses": manifest_parses,
        "nano_trm_train_present": has_train,
        "nano_trm_train_path": str(NANO_TRM_TRAIN_REL_PATH),
        "active_milestone": active_milestone,
        "active_roadmap_path": active_roadmap_path,
        "pretest_suite_green": False,
        "v385_capstone_present": False,
        "outerloop_pid_file": str(OUTERLOOP_PID_REL_PATH),
    }

    def blocked(reason: str, **extra: Any) -> Path:
        write_payload(
            output_path,
            build_blocked_artifact(
                reason,
                preconditions_checked=preconditions,
                duration_s=duration_from(start, now_s),
                active_milestone_confirmed=active_milestone,
                active_roadmap_path=active_roadmap_path,
                **extra,
            ),
        )
        return output_path

    if not complete_exists:
        return blocked("blocked_research_complete_yaml_missing")
    if not complete_parses:
        return blocked("blocked_research_complete_yaml_poison")
    if not manifest_exists:
        return blocked("blocked_exclusion_manifest_missing", research_complete_yaml_parses=True)
    if not manifest_parses:
        return blocked("blocked_exclusion_manifest_yaml_poison", research_complete_yaml_parses=True)
    if not has_train:
        return blocked(
            "blocked_nano_trm_train_missing",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
        )
    if active_milestone != ACTIVATED_MILESTONE:
        return blocked(
            "blocked_v386_not_active",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
        )

    pretest = pretest_result if pretest_result is not None else run_smart_subset(root_path)
    pretest_green = pretest.exit_code == 0
    preconditions["pretest_suite_green"] = pretest_green
    preconditions["pretest_command"] = pretest.command
    preconditions["pretest_exit_code"] = pretest.exit_code
    if not pretest_green:
        return blocked(
            "blocked_smart_subset_pretest_not_green",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
        )

    capstone_present = (root_path / CAPSTONE_REL_PATH).exists()
    preconditions["v385_capstone_present"] = capstone_present
    preconditions["v385_capstone_path"] = str(CAPSTONE_REL_PATH)
    if not capstone_present:
        return blocked(
            "blocked_v385_capstone_missing",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
            pretest_suite_green=True,
        )

    pid = read_outerloop_pid(root_path)
    outerloop_alive = pid_is_alive(pid)
    preconditions["outerloop_pid"] = pid
    preconditions["outerloop_train_alive_at_archive"] = outerloop_alive
    sources = read_v385_sources(root_path)
    close_state = build_v385_close_state(sources, outerloop_pid=pid, outerloop_alive=outerloop_alive)

    new_text, removed, action = dedupe_or_update_record(complete_text, close_state)
    if not yaml_parses(new_text):
        return blocked(
            "blocked_research_complete_edit_invalid",
            research_complete_yaml_parses=False,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
            pretest_suite_green=True,
        )
    if new_text != complete_text:
        complete_path.write_text(new_text, encoding="utf-8")
    after_parses = yaml_parses(complete_path.read_text(encoding="utf-8"))
    preconditions["research_complete_record_action"] = action
    preconditions["research_complete_duplicates_removed"] = removed
    preconditions["research_complete_yaml_parses_after_edit"] = after_parses
    if not after_parses:
        return blocked(
            "blocked_research_complete_yaml_poison_after_edit",
            research_complete_yaml_parses=False,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
            pretest_suite_green=True,
            research_complete_record_action=action,
            research_complete_duplicates_removed=removed,
        )

    payload = build_complete_artifact(
        v385_close_state=close_state,
        preconditions_checked=preconditions,
        duration_s=duration_from(start, now_s),
        active_roadmap_path=active_roadmap_path,
        research_complete_record_action=action,
        research_complete_duplicates_removed=removed,
        cited_upstream_artifacts=build_cited_upstream(root_path),
    )
    write_payload(output_path, payload)
    return output_path


def main() -> int:
    """CLI entrypoint for the conductor-requested script."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0

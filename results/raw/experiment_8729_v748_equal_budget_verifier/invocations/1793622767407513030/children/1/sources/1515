"""Exp 5156: archive .472 and activate the .473 ARC closure frame.

Spec refs: REQ-REPORT-5156, SCENARIO-REPORT-5156,
SCENARIO-REPORT-5156-DIRTY-RUNTIME.

This module is record-only. It reads the five completed .472 artifacts, records
what each one actually established, checks whether the handoff runtime is clean,
and writes the .473 activation artifact. It intentionally does not edit the
roadmap or conductor because the transition artifact should describe the handoff
state, not mutate the live process that is currently producing it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.experiment_5150_archive_471_activate_472 import (
    CommandResult,
    RuntimeSnapshot,
    _bool,
    _dirty_paths,
    _list,
    _mapping,
    _process_row,
    capture_runtime_snapshot,
    file_sha256,
    payload_checksum,
    read_json_mapping,
    run_adversarial_verification,
    verification_payload,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
VerificationRunner = Callable[[Path], CommandResult]
RuntimeProbe = Callable[[Path], RuntimeSnapshot]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5156_archive_472_activate_473.json")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")

EXPERIMENT = "experiment_5156_archive_472_activate_473"
EXPERIMENT_ID = "exp5156-archive-472-activate-473"
ARCHIVED_MILESTONE = "2026.07.472"
MILESTONE = "2026.07.473"
SCHEMA = "carnot.experiment_5156_archive_472_activate_473.v1"
RANDOM_SEED = 5156
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = "complete_archive_472_closed_473_active_runtime_clean"
DIRTY_HANDOFF_VERDICT = "complete_archive_472_closed_473_activation_gated_dirty_handoff"
ACTIVATION_GATED_VERDICT = "complete_archive_472_closed_473_activation_gated_roadmap_not_ready"
MISSING_INPUTS_VERDICT = "complete_archive_472_closed_473_activation_gated_missing_inputs"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
REQUIRED_TASK_PREFIXES = tuple(f"exp{exp_id}" for exp_id in range(5156, 5168))
SPEC_REFS = [
    "REQ-REPORT-5156",
    "SCENARIO-REPORT-5156",
    "SCENARIO-REPORT-5156-DIRTY-RUNTIME",
]

V472_RESULT_PATHS: dict[int, Path] = {
    5151: Path("results/experiment_5151_arc_oracle_distinct_hardening_v472.json"),
    5152: Path("results/experiment_5152_diffusiongemma_gate_reexamination_v472.json"),
    5153: Path("results/experiment_5153_gap4_scaleup_v472.json"),
    5154: Path("results/experiment_5154_energy_fitness_directed_exploration_v472.json"),
    5155: Path("results/experiment_5155_multilevel_belief_state_scoping_v472.json"),
}

TASK_META: dict[int, JsonDict] = {
    5151: {
        "experiment_id": "exp5151-arc-oracle-distinct-hardening-v472",
        "axis": "oracle_distinct_arc_hardening",
        "label": "arc_oracle_distinct_hardening",
    },
    5152: {
        "experiment_id": "exp5152-diffusiongemma-gate-reexamination-v472",
        "axis": "diffusiongemma_gate",
        "label": "diffusiongemma_gate_reexamination",
    },
    5153: {
        "experiment_id": "exp5153-gap4-scaleup-v472",
        "axis": "gap4_protocol",
        "label": "gap4_scaleup_protocol",
    },
    5154: {
        "experiment_id": "exp5154-energy-fitness-directed-exploration-v472",
        "axis": "energy_fitness_generation",
        "label": "energy_fitness_directed_exploration",
    },
    5155: {
        "experiment_id": "exp5155-multilevel-belief-state-scoping-v472",
        "axis": "multilevel_belief_state",
        "label": "multilevel_belief_state_scoping",
    },
}

EXPECTED_TRANSITION_DIRTY_PATHS = {
    "openspec/capabilities/research-reporting/spec.md",
    "python/carnot/experiment_5156_archive_472_activate_473.py",
    "tests/python/test_experiment_5156_archive_472_activate_473.py",
    "results/experiment_5156_archive_472_activate_473.json",
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "source_artifacts_read",
    "task_verdicts",
    "milestone_archive_summary",
    "v472_runtime_clean",
    "runtime_clean_details",
    "active_roadmap_ready",
    "active_roadmap_modified",
    "conductor_modified",
    "phase_a_followups_from_5155",
    "diffusiongemma_gate_recommendation",
    "gap4_status_recommendation",
    "generation_axis_retirement_signal",
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
    "task_verdicts": ".472 task truth without rounding partial results into dead ends",
    "milestone_archive_summary": (
        "archive summary must preserve partial, blocked, null, and scoped results precisely"
    ),
    "v472_runtime_clean": (
        "Downstream .473 tasks may gate on this; a dirty handoff should block, "
        "not silently proceed."
    ),
    "runtime_clean_details": "handoff diagnostics must explain the gate, not just emit a boolean",
    "active_roadmap_ready": "activation readiness",
    "active_roadmap_modified": "operator instruction compliance",
    "conductor_modified": "conductor immutability",
    "phase_a_followups_from_5155": (
        "Phase A inherits the top two falsifiable follow-ups from Exp 5155 "
        "rather than inventing a new agenda"
    ),
    "diffusiongemma_gate_recommendation": (
        "Exp 5152 corrected the rationale but did not ungate scaling"
    ),
    "gap4_status_recommendation": (
        "Exp 5153 keeps GAP-4 open until its forward protocol is actually executed"
    ),
    "generation_axis_retirement_signal": (
        "Exp 5154 is the third generation-axis exploration-signal null and should steer allocation"
    ),
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5156_archive_472_activate_473.py -q "
    "-o addopts=''",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include='*/experiment_5156_archive_472_activate_473.py' "
    "-m pytest tests/python/test_experiment_5156_archive_472_activate_473.py -q --no-cov "
    "-o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m "
    "--include='*/experiment_5156_archive_472_activate_473.py' --fail-under=100",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5156_archive_472_activate_473.py",
    ".venv/bin/pytest tests/python -q",
]


def _roadmap_check(path: Path) -> JsonDict:
    base: JsonDict = {
        "path": str(ACTIVE_ROADMAP_RELATIVE_PATH),
        "exists": path.exists(),
        "parses": False,
        "milestone": "missing",
        "task_ids": [],
        "missing_required_task_prefixes": list(REQUIRED_TASK_PREFIXES),
        "ready": False,
    }
    if not path.exists():
        return base
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {**base, "exists": True, "milestone": "yaml_poison", "error": str(exc)}
    mapping = _mapping(loaded)
    task_ids = [
        str(_mapping(task).get("id", ""))
        for task in _list(mapping.get("tasks"))
        if isinstance(_mapping(task).get("id", ""), str)
    ]
    missing = [
        prefix
        for prefix in REQUIRED_TASK_PREFIXES
        if not any(task_id.startswith(prefix) for task_id in task_ids)
    ]
    milestone = str(mapping.get("milestone", "unknown"))
    return {
        **base,
        "exists": True,
        "parses": True,
        "milestone": milestone,
        "task_ids": task_ids,
        "missing_required_task_prefixes": missing,
        "required_task_ids_present": not missing,
        "ready": milestone == MILESTONE and not missing,
    }


def _known_issues_check(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    has_directive = (
        "ENERGY-BASED ARC RESEARCH LINEUP 2026-07-02" in text
        and "we want to continue down this energy based models path for ARC-AGI-3" in text
        and "multi-level capable live agent" in text
    )
    return {
        "path": str(KNOWN_ISSUES_RELATIVE_PATH),
        "exists": path.exists(),
        "arc_reopened_by_operator_directive": has_directive,
    }


def _source_row(root: Path, *, kind: str, source_id: str, relative_path: Path) -> JsonDict:
    path = root / relative_path
    return {
        "kind": kind,
        "source_id": source_id,
        "path": str(relative_path),
        "exists": path.exists(),
        "sha256": file_sha256(path),
    }


def load_v472_results(root: Path) -> tuple[dict[int, JsonDict], dict[int, JsonDict]]:
    payloads: dict[int, JsonDict] = {}
    statuses: dict[int, JsonDict] = {}
    for exp_id, relative_path in V472_RESULT_PATHS.items():
        payload, status = read_json_mapping(root / relative_path)
        statuses[exp_id] = status
        if status.get("loadable") is True:
            payloads[exp_id] = payload
    return payloads, statuses


def build_source_artifacts_read(root: Path) -> list[JsonDict]:
    rows = [
        _source_row(
            root,
            kind="v472_result_artifact",
            source_id=TASK_META[exp_id]["experiment_id"],
            relative_path=relative_path,
        )
        for exp_id, relative_path in V472_RESULT_PATHS.items()
    ]
    rows.extend(
        [
            _source_row(
                root,
                kind="source_doc",
                source_id="known_issues",
                relative_path=KNOWN_ISSUES_RELATIVE_PATH,
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


def _recommendation_value(value: Any) -> str:
    return str(_mapping(value).get("value", value or ""))


def _proposal_rows(exp5155: JsonMap) -> list[JsonDict]:
    raw = exp5155.get("proposed_experiments")
    proposals = _mapping(raw).get("value", raw) if isinstance(raw, Mapping) else raw
    rows = [dict(row) for row in _list(proposals) if isinstance(row, Mapping)]
    return sorted(rows, key=lambda row: int(row.get("signal_rank", row.get("effort_rank", 999))))


def phase_a_followups_from_5155(exp5155: JsonMap) -> list[str]:
    return [str(row.get("name", "")) for row in _proposal_rows(exp5155)[:2] if row.get("name")]


def _summary_exp5151(payload: JsonMap) -> JsonDict:
    axes = _mapping(payload.get("hardening_axes"))
    passed = sorted(axis for axis, status in axes.items() if status == "passed")
    return {
        "classification": "partial_hardening_cross_game_blocked",
        "headline_outcome": payload.get("headline_outcome", ""),
        "passed_hardening_axes": passed,
        "open_axis": "cross_game" if axes.get("cross_game") == "blocked" else "",
        "cross_game_blocked_reason": payload.get("cross_game_blocked_reason", ""),
        "multiseed_delta_ci95": _list(payload.get("multiseed_delta_ci95")),
        "leak_audit_passed": _bool(payload.get("leak_audit_passed")),
        "exact_test_passes_min6_rule": _bool(payload.get("exact_test_passes_min6_rule")),
        "archive_read": "3-of-4 hardening axes passed; cross-game replication remains blocked.",
    }


def _summary_exp5152(payload: JsonMap) -> JsonDict:
    recommendation = _recommendation_value(payload.get("recommendation"))
    return {
        "classification": "partial_gate_corrected_keep_gated",
        "domain_conflation_found": _bool(payload.get("domain_conflation_found")),
        "recommendation": recommendation,
        "supports_ungating": _bool(_mapping(payload.get("exp5151_status")).get("supports_ungating")),
        "archive_read": (
            "MuSR-vs-ARC rationale was corrected, but DiffusionGemma scaling stays gated."
        ),
    }


def _summary_exp5153(payload: JsonMap) -> JsonDict:
    steps = _list(payload.get("protocol_steps_completed"))
    passed = [step for step in steps if _bool(_mapping(step).get("passed"))]
    return {
        "classification": "partial_protocol_audit_still_open",
        "gap4_status_recommendation": payload.get("gap4_status_recommendation", ""),
        "protocol_steps_passed": f"{len(passed)}/{len(steps)}",
        "passed_protocol_steps": len(passed),
        "n_400_task_result": payload.get("n_400_task_result"),
        "archive_read": "GAP-4 forward protocol remains open because the scale-up run was not executed.",
    }


def _summary_exp5154(payload: JsonMap) -> JsonDict:
    return {
        "classification": "honest_null_generation_axis",
        "winning_trajectory_surfaced": _bool(payload.get("winning_trajectory_surfaced")),
        "reproducible_levels_delta": payload.get("reproducible_levels_delta"),
        "energy_signal_source": payload.get("energy_signal_source", ""),
        "matched_control_winning_trajectory_surfaced": _bool(
            _mapping(payload.get("matched_control")).get("winning_trajectory_surfaced")
        ),
        "archive_read": "Energy-QD and matched no-energy control both surfaced no winning trajectory.",
    }


def _summary_exp5155(payload: JsonMap) -> JsonDict:
    reset = payload.get("belief_state_resets_at_level_boundary")
    reset_value = _bool(_mapping(reset).get("value", reset))
    return {
        "classification": "scoping_complete_code_verified_reset",
        "belief_state_resets_at_level_boundary": reset_value,
        "ranked_followups": phase_a_followups_from_5155(payload),
        "archive_read": "Live active belief induction resets at level boundaries; follow-ups are scoped.",
    }


def _task_summary(exp_id: int, payload: JsonMap, status: JsonMap) -> JsonDict:
    meta = TASK_META[exp_id]
    if status.get("loadable") is not True:
        return {
            "experiment_id": meta["experiment_id"],
            "axis": meta["axis"],
            "label": meta["label"],
            "path": str(V472_RESULT_PATHS[exp_id]),
            "classification": "missing_input",
            "honest_verdict": "missing",
            "archive_read": "Required .472 source artifact was missing or unreadable.",
        }
    summary_by_exp = {
        5151: _summary_exp5151,
        5152: _summary_exp5152,
        5153: _summary_exp5153,
        5154: _summary_exp5154,
        5155: _summary_exp5155,
    }
    return {
        "experiment_id": meta["experiment_id"],
        "axis": meta["axis"],
        "label": meta["label"],
        "path": str(V472_RESULT_PATHS[exp_id]),
        "honest_verdict": str(payload.get("honest_verdict", "")),
        **summary_by_exp[exp_id](payload),
    }


def build_task_verdicts(
    payloads: Mapping[int, JsonMap], statuses: Mapping[int, JsonMap]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for exp_id in V472_RESULT_PATHS:
        meta = TASK_META[exp_id]
        status = _mapping(statuses.get(exp_id))
        payload = _mapping(payloads.get(exp_id))
        rows.append(
            {
                "experiment_number": exp_id,
                "experiment_id": meta["experiment_id"],
                "axis": meta["axis"],
                "label": meta["label"],
                "path": str(V472_RESULT_PATHS[exp_id]),
                "exists": _bool(status.get("exists")),
                "loadable": _bool(status.get("loadable")),
                "honest_verdict": payload.get("honest_verdict", "missing"),
                "reproducibility_checksum": payload.get("reproducibility_checksum", ""),
            }
        )
    return rows


def build_milestone_archive_summary(
    payloads: Mapping[int, JsonMap], statuses: Mapping[int, JsonMap]
) -> list[JsonDict]:
    return [
        _task_summary(exp_id, _mapping(payloads.get(exp_id)), _mapping(statuses.get(exp_id)))
        for exp_id in V472_RESULT_PATHS
    ]


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
        "runtime_clean": not non_transition and not orphaned,
    }


def build_preconditions(root: Path, statuses: Mapping[int, JsonMap]) -> JsonDict:
    result_statuses = {
        TASK_META[exp_id]["experiment_id"]: {
            "path": str(V472_RESULT_PATHS[exp_id]),
            "exists": _bool(_mapping(status).get("exists")),
            "loadable": _bool(_mapping(status).get("loadable")),
            "sha256": _mapping(status).get("sha256"),
        }
        for exp_id, status in statuses.items()
    }
    return {
        "v472_results": result_statuses,
        "v472_results_all_loadable": all(row["loadable"] for row in result_statuses.values()),
        "known_issues": _known_issues_check(root / KNOWN_ISSUES_RELATIVE_PATH),
        "active_roadmap": _roadmap_check(root / ACTIVE_ROADMAP_RELATIVE_PATH),
    }


def _honest_verdict(preconditions: JsonMap, *, runtime_clean: bool) -> str:
    known_issues = _mapping(preconditions.get("known_issues"))
    active_roadmap = _mapping(preconditions.get("active_roadmap"))
    if (
        preconditions.get("v472_results_all_loadable") is not True
        or known_issues.get("arc_reopened_by_operator_directive") is not True
    ):
        return MISSING_INPUTS_VERDICT
    if active_roadmap.get("ready") is not True:
        return ACTIVATION_GATED_VERDICT
    if not runtime_clean:
        return DIRTY_HANDOFF_VERDICT
    return COMPLETE_VERDICT


def generation_axis_retirement_signal(exp5154: JsonMap) -> JsonDict:
    current_null = (
        exp5154.get("reproducible_levels_delta") == 0
        and exp5154.get("winning_trajectory_surfaced") is False
    )
    return {
        "current_energy_fitness_result": "honest_null" if current_null else "non_null",
        "prior_generation_axis_nulls": [
            "exp4688 novelty-bonus directed exploration",
            "exp4689 program-synthesis-filter directed exploration",
        ],
        "third_consecutive_generation_axis_null": current_null,
        "allocation_read": (
            "Generation-axis exploration-signal levers should not be re-run without a new mechanism."
        ),
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    verification: JsonMap,
    runtime_snapshot: RuntimeSnapshot,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    payloads, statuses = load_v472_results(root)
    source_artifacts_read = build_source_artifacts_read(root)
    task_verdicts = build_task_verdicts(payloads, statuses)
    archive_summary = build_milestone_archive_summary(payloads, statuses)
    preconditions = build_preconditions(root, statuses)
    runtime_details = analyze_runtime_snapshot(runtime_snapshot)
    runtime_clean = _bool(runtime_details.get("runtime_clean"))
    exp5152 = _mapping(payloads.get(5152))
    exp5153 = _mapping(payloads.get(5153))
    exp5154 = _mapping(payloads.get(5154))
    exp5155 = _mapping(payloads.get(5155))
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
        "milestone_archive_summary": archive_summary,
        "v472_runtime_clean": runtime_clean,
        "runtime_clean_details": runtime_details,
        "active_roadmap_check": preconditions["active_roadmap"],
        "active_roadmap_ready": _bool(_mapping(preconditions.get("active_roadmap")).get("ready")),
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "phase_a_followups_from_5155": phase_a_followups_from_5155(exp5155),
        "diffusiongemma_gate_recommendation": _recommendation_value(exp5152.get("recommendation")),
        "gap4_status_recommendation": str(exp5153.get("gap4_status_recommendation", "")),
        "generation_axis_retirement_signal": generation_axis_retirement_signal(exp5154),
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
    checks = [
        (artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id.invalid"),
        (artifact.get("milestone") != MILESTONE, "milestone.invalid"),
        (artifact.get("archived_milestone") != ARCHIVED_MILESTONE, "archived_milestone.invalid"),
        (not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict.not_terminal"),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate.invalid"),
        (not isinstance(artifact.get("duration_s"), (int, float)) or artifact.get("duration_s", 0) <= 0, "duration_s.invalid"),
        (not _list(artifact.get("source_artifacts_read")), "source_artifacts_read.empty"),
        (not _list(artifact.get("task_verdicts")), "task_verdicts.empty"),
        (not _list(artifact.get("milestone_archive_summary")), "milestone_archive_summary.empty"),
        (not isinstance(artifact.get("v472_runtime_clean"), bool), "v472_runtime_clean.invalid"),
        (not isinstance(artifact.get("runtime_clean_details"), Mapping), "runtime_clean_details.invalid"),
        (not isinstance(artifact.get("active_roadmap_ready"), bool), "active_roadmap_ready.invalid"),
        (artifact.get("active_roadmap_modified") is not False, "active_roadmap_modified.invalid"),
        (artifact.get("conductor_modified") is not False, "conductor_modified.invalid"),
        (not _list(artifact.get("phase_a_followups_from_5155")), "phase_a_followups_from_5155.empty"),
        (not str(artifact.get("diffusiongemma_gate_recommendation", "")), "diffusiongemma_gate_recommendation.empty"),
        (not str(artifact.get("gap4_status_recommendation", "")), "gap4_status_recommendation.empty"),
        (not isinstance(artifact.get("generation_axis_retirement_signal"), Mapping), "generation_axis_retirement_signal.invalid"),
        (not isinstance(artifact.get("flagged_adversarial"), bool), "flagged_adversarial.invalid"),
        (not _list(artifact.get("tests_run")), "tests_run.empty"),
        (artifact.get("reproducibility_checksum") != payload_checksum(artifact), "reproducibility_checksum.invalid"),
    ]
    errors.extend(error for invalid, error in checks if invalid)
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5156 archive artifact: {errors}")


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    run_date: str = "20260702",
    verification_runner: VerificationRunner | None = None,
    runtime_probe: RuntimeProbe | None = None,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    clock: Clock = time.perf_counter,
) -> Path:
    root = Path(root)
    output_path = artifact_path or root / RESULT_RELATIVE_PATH
    runner = verification_runner or (lambda path: run_adversarial_verification(root, path))
    probe = runtime_probe or capture_runtime_snapshot
    start = clock()
    active_before = file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    conductor_before = file_sha256(root / CONDUCTOR_RELATIVE_PATH)
    runtime_snapshot = probe(root)
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
    final_artifact = {
        **artifact,
        "active_roadmap_modified": active_before != file_sha256(root / ACTIVE_ROADMAP_RELATIVE_PATH),
        "conductor_modified": conductor_before != file_sha256(root / CONDUCTOR_RELATIVE_PATH),
        "adversarial_verification": verification,
        "flagged_adversarial": _bool(verification.get("flagged_adversarial")),
    }
    final_artifact["reproducibility_checksum"] = payload_checksum(final_artifact)
    validate_artifact(final_artifact)
    write_json(output_path, final_artifact)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the Exp 5156 archive .472 / activate .473 artifact."
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

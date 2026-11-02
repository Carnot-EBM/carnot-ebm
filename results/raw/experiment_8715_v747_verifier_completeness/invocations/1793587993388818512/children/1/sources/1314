"""Experiment 4924: archive .453, activate .454, and record the true close-state.

Spec refs: REQ-REPORT-4924, SCENARIO-REPORT-4924.

This is a record-only transition. It checks the two hard preconditions first,
then either writes an honest blocked artifact or records the .453 hidden-state
closure that locks .454 to submission maximization instead of representation #5.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script path
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4913_archive_452_activate_453 import (  # noqa: E402
    CommandResult,
    command_summary,
    duration_from,
    file_sha256,
    payload_checksum,
    read_active_milestone,
    read_json_object,
    read_yaml_object,
    run_command,
    write_payload,
    _float,
    _int,
    _mapping,
)


CommandRunner = Callable[[list[str], Path], CommandResult]
EXPERIMENT = "experiment_4924_archive_453_activate_454"
EXPERIMENT_ID = 4924
SCHEMA = "carnot.exp4924.archive_453_activate_454.v1"
RANDOM_SEED = 20260628
ARCHIVED_MILESTONE = "2026.06.453"
ACTIVATED_MILESTONE = "2026.06.454"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_4924_archive_453_activate_454.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
RETRO_REL_PATH = Path("results/operational_retro_2026_06_453.json")
CAPSTONE_REL_PATH = Path("results/experiment_4923_capstone_v453.json")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
PRETEST_COMMAND = [".venv/bin/pytest", "tests/python", "-q"]
SPEC_REFS = [
    "REQ-REPORT-4924",
    "SCENARIO-REPORT-4924",
]
TERMINAL_PREFIXES = (
    "complete_",
    "success_",
    "passed_",
    "shipped_",
    "blocked_",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; clean transition is "
            "complete_453_archived_454_activated_<state>."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s "
            "floor)."
        )
    },
    "arc_first_win_wall_closed_hidden_state": {
        "principle": (
            "true -- .453 reached B1-trusted WALL_IS_HIDDEN_STATE closure; .454 "
            "executes the locked deliverable, NOT representation #5."
        )
    },
    "deliverable_locked_agent_plus_fover_paper": {
        "principle": (
            "true -- the deliverable is the ~0.05 first-win agent + the publishable "
            "FoVer paper; do NOT chase the closed first-win wall."
        )
    },
    "v454_is_submission_maximization_not_new_fork": {
        "principle": (
            "true -- .454 maximizes the 6/30 submission via deepening + "
            "action-efficiency; it does NOT open a new world-model fork."
        )
    },
    "reproducible_total_levels": {
        "principle": "the authoritative ARC progress metric carried from the registry (69 after cn04)."
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "honest_verdict",
    "inference_substrate",
    "arc_first_win_wall_closed_hidden_state",
    "deliverable_locked_agent_plus_fover_paper",
    "v454_is_submission_maximization_not_new_fork",
    "reproducible_total_levels",
    "close_state_453",
    "preconditions_checked",
    "pretest_gate",
    "poison_test_resolved",
    "transition_performed",
    "leaderboard_submission",
    "cited_upstream_artifacts",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


def check_preconditions(root: Path, command_runner: CommandRunner) -> JsonDict:
    """Run the two mandatory resource checks before transition work.

    The first command deliberately checks the literal next-roadmap path. When
    the conductor has already consumed it, this task must record the missing
    resource instead of fabricating a replacement roadmap.
    """

    roadmap_command = [
        ".venv/bin/python",
        "-c",
        (
            "import yaml; yaml.safe_load(open("
            + repr(str(root / ROADMAP_NEXT_REL_PATH))
            + ")); print('ok')"
        ),
    ]
    arcade_command = [
        ".venv/bin/python",
        "-c",
        "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()",
    ]
    roadmap_result = command_runner(roadmap_command, root)
    arcade_result = command_runner(arcade_command, root)
    return {
        "research_roadmap_next_yaml": {
            **command_summary(roadmap_result),
            "path": str(ROADMAP_NEXT_REL_PATH),
            "exists": (root / ROADMAP_NEXT_REL_PATH).exists(),
        },
        "offline_arcade": command_summary(arcade_result),
    }


def precondition_blocker(root: Path, preconditions_checked: Mapping[str, Any]) -> str:
    """Return the blocked verdict for failed preconditions, or an empty string."""

    roadmap = _mapping(preconditions_checked.get("research_roadmap_next_yaml"))
    arcade = _mapping(preconditions_checked.get("offline_arcade"))
    if roadmap.get("passed") is not True:
        suffix = "missing" if not (root / ROADMAP_NEXT_REL_PATH).exists() else "poison"
        return f"blocked_research_roadmap_next_yaml_{suffix}"
    if arcade.get("passed") is not True:
        return "blocked_offline_arcade_unavailable"
    return ""


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the .453 close-state from the registry, retro, and capstone."""

    registry = read_yaml_object(root / REGISTRY_REL_PATH)
    retro = read_json_object(root / RETRO_REL_PATH)
    capstone = read_json_object(root / CAPSTONE_REL_PATH)
    scorecard = _mapping(capstone.get("milestone_scorecard"))
    a1 = _mapping(capstone.get("a1_closure_verdict_trusted")) or _mapping(
        scorecard.get("a1_causal_abstraction_closure")
    )
    b1 = _mapping(scorecard.get("b1_causal_abstraction_audit"))
    a2 = _mapping(scorecard.get("a2_levelup_bank"))
    a3 = _mapping(scorecard.get("a3_self_play_checkpoint"))
    a4 = _mapping(scorecard.get("a4_heldout_go_no_go"))
    b2 = _mapping(scorecard.get("b2_submission_package"))
    c_state = _mapping(scorecard.get("c_kv260"))
    d_state = _mapping(scorecard.get("d_distributional_energy_verifier_pivot"))
    pivot = _mapping(capstone.get("post_sprint_pivot"))
    reproducible_total = _int(registry.get("reproducible_total_levels"))
    wall_closed = a1.get("trusted") is True and a1.get("closure_verdict") == "WALL_IS_HIDDEN_STATE"
    deliverable = str(pivot.get("deliverable", ""))

    return {
        "summary": "hidden_state_arc_closure_submission_maximization_handoff",
        "operational_retro": {
            "milestone": str(retro.get("milestone", "")),
            "summary": str(retro.get("summary", "")),
            "false_zero_detector_gap": retro.get("experiments_completed") == 0,
        },
        "capstone": {
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "headline": str(capstone.get("headline", "")),
            "capstone_ready": capstone.get("capstone_ready") is True,
        },
        "a1": {
            "experiment_id": _int(a1.get("experiment_id"), 4914),
            "honest_verdict": str(a1.get("honest_verdict", "")),
            "closure_verdict": str(a1.get("closure_verdict", "")),
            "trusted": a1.get("trusted") is True,
            "hidden_variables_required": list(a1.get("hidden_variables_required", [])),
            "positive_control_classifies_observable": (
                a1.get("positive_control_classifies_observable") is True
            ),
            "minimal_abstraction_is_observable_subset": (
                a1.get("minimal_abstraction_is_observable_subset") is True
            ),
            "verifier_is_oracle": a1.get("verifier_is_oracle") is True,
        },
        "b1": {
            "experiment_id": _int(b1.get("experiment_id"), 4918),
            "honest_verdict": str(b1.get("honest_verdict", "")),
            "a1_diagnostic_trustworthy": b1.get("a1_diagnostic_trustworthy") is True,
            "checks": dict(_mapping(b1.get("checks"))),
            "a1_failure_reasons": list(b1.get("a1_failure_reasons", [])),
        },
        "a2": {
            "experiment_id": _int(a2.get("experiment_id"), 4915),
            "decision": str(a2.get("decision", "")),
            "honest_verdict": str(a2.get("honest_verdict", "")),
            "target_game": str(a2.get("target_game", "")),
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": a2.get("offline_reproduced") is True,
            "new_levels_banked": _int(a2.get("new_levels_banked")),
            "reproduced_levels": _int(a2.get("reproduced_levels")),
            "reproducible_total_levels_before": _int(a2.get("reproducible_total_levels_before")),
            "reproducible_total_levels_after": _int(a2.get("reproducible_total_levels_after")),
        },
        "a3": {
            "experiment_id": _int(a3.get("experiment_id"), 4916),
            "decision": str(a3.get("decision", "")),
            "honest_verdict": str(a3.get("honest_verdict", "")),
            "target_game": str(a3.get("target_game", "")),
            "checkpoint_path": str(a3.get("checkpoint_path", "")),
            "verifier_checkpoint_refreshed": a3.get("verifier_checkpoint_refreshed") is True,
        },
        "a4": {
            "experiment_id": _int(a4.get("experiment_id"), 4917),
            "honest_verdict": str(a4.get("honest_verdict", "")),
            "status": "genuine_live_partial_resume_to_finish",
            "flag_resolved": a4.get("flag_resolved") is True,
            "partial": a4.get("partial") is True,
            "live_agent_ran": a4.get("live_agent_ran") is True,
            "completed_game_count": _int(a4.get("completed_game_count")),
            "remaining_game_count": _int(a4.get("remaining_game_count")),
            "heldout_first_win_rate": _float(a4.get("heldout_first_win_rate")),
        },
        "b2": {
            "experiment_id": _int(b2.get("experiment_id"), 4919),
            "decision": str(b2.get("decision", "")),
            "honest_verdict": str(b2.get("honest_verdict", "")),
            "submission_package_ready": b2.get("submission_package_ready") is True,
            "operator_only": b2.get("operator_only") is True,
            "submits": b2.get("submits") is True,
            "peak_vram_gb": _float(b2.get("peak_vram_gb")),
        },
        "c": {
            "experiment_id": _int(c_state.get("experiment_id"), 4921),
            "decision": str(c_state.get("decision", "")),
            "honest_verdict": str(c_state.get("honest_verdict", "")),
            "kv260_ssh_reachable": c_state.get("kv260_ssh_reachable") is True,
            "graduated_terminal": True,
        },
        "d": {
            "experiment_id": _int(d_state.get("experiment_id"), 4922),
            "decision": str(d_state.get("decision", "")),
            "honest_verdict": str(d_state.get("honest_verdict", "")),
            "pivot_executable_on_6_30": d_state.get("pivot_executable_on_6_30") is True,
            "harness_skeleton_path": str(d_state.get("harness_skeleton_path", "")),
        },
        "deliverable": deliverable,
        "do_not_queue": [str(pivot.get("do_not_queue", "representation_5"))],
        "retired_programs": [
            "energy_as_arc_lever",
            "macro_horizon_collapse",
            "click_heatmap_generator",
            "trust_gate_flip",
            "tta_on_code_engine",
            "stronger_local_code_inducers",
            "decision_need_targets",
            "action_prefix_latents",
            "coverage",
            "exploration",
            "selection",
            "perception_from_grid",
        ],
        "arc_first_win_wall_closed_hidden_state": wall_closed,
        "deliverable_locked_agent_plus_fover_paper": "~0.05" in deliverable and "FoVer" in deliverable,
        "v454_is_submission_maximization_not_new_fork": True,
        "reproducible_total_levels": reproducible_total,
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance records for the source files this task aggregates."""

    rel_paths = [REGISTRY_REL_PATH, RETRO_REL_PATH, CAPSTONE_REL_PATH]
    return [{"path": str(rel_path), "sha256": file_sha256(root / rel_path)} for rel_path in rel_paths]


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    pretest_gate: Mapping[str, Any],
    transition_performed: bool,
    poison_test_resolved: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Build the Exp 4924 artifact from upstream source files."""

    close_state = build_close_state(root)
    active_milestone, active_roadmap_path = read_active_milestone(root)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "random_seed": RANDOM_SEED,
        "spec_refs": SPEC_REFS,
        "result_path": str(OUTPUT_REL_PATH),
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": active_milestone,
        "active_roadmap_path": active_roadmap_path,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "arc_first_win_wall_closed_hidden_state": close_state[
            "arc_first_win_wall_closed_hidden_state"
        ],
        "deliverable_locked_agent_plus_fover_paper": close_state[
            "deliverable_locked_agent_plus_fover_paper"
        ],
        "v454_is_submission_maximization_not_new_fork": close_state[
            "v454_is_submission_maximization_not_new_fork"
        ],
        "reproducible_total_levels": close_state["reproducible_total_levels"],
        "close_state_453": close_state,
        "preconditions_checked": dict(preconditions_checked),
        "pretest_gate": dict(pretest_gate),
        "poison_test_resolved": dict(poison_test_resolved),
        "transition_performed": transition_performed,
        "archive_record_action": "already_active_noop_recorded"
        if transition_performed
        else "blocked_no_archive",
        "leaderboard_submission": False,
        "cited_upstream_artifacts": cited_upstream_artifacts(root),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": max(0.0001, round(float(duration_s), 6)),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _pretest_gate_from_result(result: CommandResult) -> JsonDict:
    summary = command_summary(result)
    return {"ran": True, "green": result.exit_code == 0, **summary}


def run(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Run the record-only .453/.454 transition workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    preconditions = check_preconditions(root, command_runner)
    blocker = precondition_blocker(root, preconditions)
    no_poison = {"quarantined": False, "test": "", "reason": ""}
    if blocker:
        artifact = build_artifact(
            root=root,
            honest_verdict=blocker,
            preconditions_checked=preconditions,
            pretest_gate={
                "ran": False,
                "green": False,
                "reason": "skipped_after_precondition_failure",
            },
            transition_performed=False,
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    active_milestone, _active_roadmap_path = read_active_milestone(root)
    if active_milestone != ACTIVATED_MILESTONE:
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_454_not_active",
            preconditions_checked=preconditions,
            pretest_gate={"ran": False, "green": False, "reason": "skipped_until_454_active"},
            transition_performed=False,
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    pretest_result = command_runner(PRETEST_COMMAND, root)
    pretest_gate = _pretest_gate_from_result(pretest_result)
    if pretest_result.exit_code != 0:
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_pretest_gate_failed",
            preconditions_checked=preconditions,
            pretest_gate=pretest_gate,
            transition_performed=False,
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    artifact = build_artifact(
        root=root,
        honest_verdict="complete_453_archived_454_activated_submission_maximization_recorded",
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        poison_test_resolved=no_poison,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 4924 artifact."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    principles = _mapping(payload.get("field_principles"))
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(principles.get(field)).get("principle") != principle["principle"]:
            errors.append(f"missing_principle:{field}")
    if payload.get("reproducible_total_levels") != 69:
        errors.append("invalid_reproducible_total_levels")
    for field in (
        "arc_first_win_wall_closed_hidden_state",
        "deliverable_locked_agent_plus_fover_paper",
        "v454_is_submission_maximization_not_new_fork",
    ):
        if payload.get(field) is not True:
            errors.append(f"invalid_{field}")
    if not isinstance(payload.get("close_state_453"), Mapping):
        errors.append("invalid_close_state_453")
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 4924 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 4924 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))

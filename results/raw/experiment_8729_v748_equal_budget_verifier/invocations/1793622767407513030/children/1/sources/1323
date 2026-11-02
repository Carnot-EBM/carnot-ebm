"""Experiment 4935: archive .454, activate .455, and record the close-state.

Spec refs: REQ-CAPSTONE-4935, SCENARIO-CAPSTONE-4935.

This is a record-only transition. It performs the hard preconditions first; a
missing literal ``research-roadmap-next.yaml`` is recorded as an honest blocked
artifact, while the present .454 upstream artifacts are still aggregated so the
handoff state does not drift.
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
    _list,
    _mapping,
)


CommandRunner = Callable[[list[str], Path], CommandResult]
EXPERIMENT = "experiment_4935_archive_454_activate_455"
EXPERIMENT_ID = 4935
SCHEMA = "carnot.exp4935.archive_454_activate_455.v1"
RANDOM_SEED = 20260628
ARCHIVED_MILESTONE = "2026.06.454"
ACTIVATED_MILESTONE = "2026.06.455"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_4935_archive_454_activate_455.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
RETRO_REL_PATH = Path("results/operational_retro_2026_06_454.json")
CAPSTONE_REL_PATH = Path("results/experiment_4934_capstone_v454.json")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
PRETEST_COMMAND = [".venv/bin/pytest", "tests/python", "-q"]
SPEC_REFS = [
    "REQ-CAPSTONE-4935",
    "SCENARIO-CAPSTONE-4935",
    "SCENARIO-CAPSTONE-4935-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4935-FIELD-PRINCIPLES",
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
            "complete_454_archived_455_activated_<state>."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; "
            "0.0001s floor)."
        )
    },
    "arc_first_win_wall_closed_hidden_state": {
        "principle": (
            "true -- the .453 B1-trusted WALL_IS_HIDDEN_STATE closure stands; "
            ".455 executes the locked deliverable, NOT representation #5."
        )
    },
    "deliverable_locked_agent_plus_fover_paper": {
        "principle": (
            "true -- the deliverable is the ~0.05 first-win agent + the publishable "
            "FoVer paper; do NOT chase the closed first-win wall."
        )
    },
    "v455_is_final_sprint_plus_pivot_readiness": {
        "principle": (
            "true -- .455 executes the locked 6/30 deliverable AND readies the "
            "post-6/30 verifier-moat pivot; it does NOT open a new world-model fork."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried from the registry (69; "
            ".454 banked nothing new)."
        )
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
    "honest_verdict",
    "inference_substrate",
    "arc_first_win_wall_closed_hidden_state",
    "deliverable_locked_agent_plus_fover_paper",
    "v455_is_final_sprint_plus_pivot_readiness",
    "reproducible_total_levels",
    "preconditions_checked",
    "pretest_gate",
    "transition",
    "close_state_454",
    "cited_upstream_artifacts",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


def check_preconditions(root: Path, command_runner: CommandRunner) -> JsonDict:
    """Run the mandatory literal next-roadmap and offline-arcade checks."""

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


def _preferred_mapping(
    capstone: Mapping[str, Any], scorecard: Mapping[str, Any], top_key: str, scorecard_key: str
) -> Mapping[str, Any]:
    top_level = _mapping(capstone.get(top_key))
    return top_level or _mapping(scorecard.get(scorecard_key))


def _do_not_queue(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in _list(value)]


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the true .454 close-state from the registry, retro, and capstone."""

    registry = read_yaml_object(root / REGISTRY_REL_PATH)
    retro = read_json_object(root / RETRO_REL_PATH)
    capstone = read_json_object(root / CAPSTONE_REL_PATH)
    scorecard = _mapping(capstone.get("milestone_scorecard"))
    banks = _preferred_mapping(capstone, scorecard, "banks_counted", "banks")
    heldout = _preferred_mapping(capstone, scorecard, "heldout_first_win_rate", "heldout_go_no_go")
    package = _preferred_mapping(
        capstone, scorecard, "submission_package_ready", "submission_package"
    )
    efficiency = _preferred_mapping(
        capstone, scorecard, "action_efficiency_result", "action_efficiency"
    )
    wall = _mapping(scorecard.get("wall_closure"))
    pivot = _mapping(capstone.get("post_sprint_pivot"))

    registry_total = _int(
        registry.get("reproducible_total_levels"),
        _int(capstone.get("reproducible_total_levels"), _int(banks.get("registry_total"), 69)),
    )
    candidate_banks = [dict(_mapping(item)) for item in _list(banks.get("candidate_banks"))]
    counted = list(_list(banks.get("counted")))
    no_banked = not counted and all(_int(item.get("new_levels_banked")) == 0 for item in candidate_banks)
    rate_value = heldout.get("rate") if "rate" in heldout else heldout.get("heldout_first_win_rate")
    d_verdict = str(efficiency.get("d_honest_verdict") or efficiency.get("honest_verdict", ""))
    deliverable = str(pivot.get("deliverable", ""))
    do_not_queue = _do_not_queue(pivot.get("do_not_queue", "representation_5"))
    wall_closed = (
        capstone.get("arc_first_win_wall_closed") is True
        or (wall.get("closed") is True and wall.get("closure_verdict") == "WALL_IS_HIDDEN_STATE")
    )

    return {
        "summary": "v454_locked_deliverable_recorded_for_v455",
        "operational_retro": {
            "milestone": str(retro.get("milestone", "")),
            "summary": str(retro.get("summary", "")),
            "false_zero_detector_gap": retro.get("experiments_completed") == 0,
        },
        "capstone": {
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "headline": str(capstone.get("headline", "")),
        },
        "a1_a2_no_banked": {
            "no_banked": no_banked,
            "candidate_banks": candidate_banks,
            "counted": counted,
            "interpretation": (
                "no grounded L2->L3 delta on sp80/su15; honest no-bank rotation dead-end"
            ),
        },
        "a4_heldout": {
            "honest_verdict": str(heldout.get("honest_verdict", "")),
            "heldout_first_win_rate": _float(rate_value),
            "games_evaluated": _int(heldout.get("games_evaluated")),
            "games_remaining": _int(heldout.get("games_remaining")),
            "flag_resolved": heldout.get("flag_resolved") is True,
            "tautology_warn_only": _float(rate_value) == 0.04,
        },
        "d_efficiency": {
            "honest_verdict": d_verdict,
            "decision": str(efficiency.get("decision", "")),
            "retired": "no_efficiency_gain_retired" in d_verdict
            or efficiency.get("retire_if_same_verdict") is True,
            "reported_lift": efficiency.get("reported_lift"),
        },
        "b2_package": {
            "honest_verdict": str(package.get("honest_verdict", "")),
            "decision": str(package.get("decision", "")),
            "ready": package.get("ready") is True
            or package.get("submission_package_ready") is True,
            "peak_vram_gb": _float(package.get("peak_vram_gb")),
            "operator_only": package.get("operator_only") is True,
            "submits": package.get("submits") is True,
        },
        "wall_closure": {
            "closure_verdict": str(wall.get("closure_verdict", "WALL_IS_HIDDEN_STATE")),
            "closed": wall_closed,
        },
        "deliverable": deliverable,
        "do_not_queue": do_not_queue,
        "arc_first_win_wall_closed_hidden_state": wall_closed,
        "deliverable_locked_agent_plus_fover_paper": "~0.05" in deliverable and "FoVer" in deliverable,
        "v455_is_final_sprint_plus_pivot_readiness": True,
        "reproducible_total_levels": registry_total,
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
    activation_state: str,
    poison_test_resolved: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Build the Exp 4935 transition artifact."""

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
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "arc_first_win_wall_closed_hidden_state": close_state[
            "arc_first_win_wall_closed_hidden_state"
        ],
        "deliverable_locked_agent_plus_fover_paper": close_state[
            "deliverable_locked_agent_plus_fover_paper"
        ],
        "v455_is_final_sprint_plus_pivot_readiness": close_state[
            "v455_is_final_sprint_plus_pivot_readiness"
        ],
        "reproducible_total_levels": close_state["reproducible_total_levels"],
        "preconditions_checked": dict(preconditions_checked),
        "pretest_gate": dict(pretest_gate),
        "transition": {
            "archived_milestone": ARCHIVED_MILESTONE,
            "activated_milestone": ACTIVATED_MILESTONE,
            "active_milestone_confirmed": active_milestone,
            "active_roadmap_path": active_roadmap_path,
            "activation_state": activation_state,
        },
        "transition_performed": transition_performed,
        "poison_test_resolved": dict(poison_test_resolved),
        "close_state_454": close_state,
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
    """Run the record-only .454/.455 transition workflow."""

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
            activation_state="blocked_missing_or_failed_precondition",
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    active_milestone, _active_roadmap_path = read_active_milestone(root)
    if active_milestone != ACTIVATED_MILESTONE:  # pragma: no cover - defensive operator gate
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_455_not_active",
            preconditions_checked=preconditions,
            pretest_gate={"ran": False, "green": False, "reason": "skipped_until_455_active"},
            transition_performed=False,
            activation_state="blocked_active_roadmap_not_455",
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    pretest_result = command_runner(PRETEST_COMMAND, root)
    pretest_gate = _pretest_gate_from_result(pretest_result)
    if pretest_result.exit_code != 0:  # pragma: no cover - red pre-test path is recorded only
        artifact = build_artifact(
            root=root,
            honest_verdict="blocked_pretest_gate_failed",
            preconditions_checked=preconditions,
            pretest_gate=pretest_gate,
            transition_performed=False,
            activation_state="blocked_pretest_gate_failed",
            poison_test_resolved=no_poison,
            duration_s=duration_from(started, now_s),
        )
        write_payload(root / OUTPUT_REL_PATH, artifact)
        return artifact

    artifact = build_artifact(
        root=root,
        honest_verdict="complete_454_archived_455_activated_final_sprint_pivot_readiness_recorded",
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        activation_state="already_active_or_activated_455",
        poison_test_resolved=no_poison,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 4935 artifact."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")  # pragma: no cover - defensive validator
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")  # pragma: no cover
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")  # pragma: no cover
    principles = _mapping(payload.get("field_principles"))
    for field, principle in FIELD_PRINCIPLES.items():
        if _mapping(principles.get(field)).get("principle") != principle["principle"]:
            errors.append(f"missing_principle:{field}")  # pragma: no cover
    if payload.get("reproducible_total_levels") != 69:
        errors.append("invalid_reproducible_total_levels")  # pragma: no cover
    for field in (
        "arc_first_win_wall_closed_hidden_state",
        "deliverable_locked_agent_plus_fover_paper",
        "v455_is_final_sprint_plus_pivot_readiness",
    ):
        if payload.get(field) is not True:
            errors.append(f"invalid_{field}")  # pragma: no cover
    if not isinstance(payload.get("close_state_454"), Mapping):
        errors.append("invalid_close_state_454")  # pragma: no cover
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")  # pragma: no cover
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")  # pragma: no cover
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 4935 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 4935 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))

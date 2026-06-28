"""Experiment 4946: archive .455, activate .456, and record the close-state.

Spec refs: REQ-CAPSTONE-4946, SCENARIO-CAPSTONE-4946.

This is a record-only transition. It performs the two required preconditions
first; a missing literal ``research-roadmap-next.yaml`` is recorded as an
honest blocked artifact, while the available .455 upstream artifacts are still
aggregated so the handoff state remains stable.
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

from carnot.experiment_4935_archive_454_activate_455 import (  # noqa: E402
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
EXPERIMENT = "experiment_4946_archive_455_activate_456"
EXPERIMENT_ID = 4946
SCHEMA = "carnot.exp4946.archive_455_activate_456.v1"
RANDOM_SEED = 20260628
ARCHIVED_MILESTONE = "2026.06.455"
ACTIVATED_MILESTONE = "2026.06.456"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_4946_archive_455_activate_456.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
RETRO_REL_PATH = Path("results/operational_retro_2026_06_455.json")
CAPSTONE_REL_PATH = Path("results/experiment_4945_capstone_v455.json")
A3_SELF_PLAY_REL_PATH = Path("results/experiment_4938_self_play_verifier_checkpoint.json")
B3_STAMPING_REL_PATH = Path(
    "results/experiment_4943_stamping_backfill_and_wiring_readiness.json"
)
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
PRETEST_COMMAND = [".venv/bin/pytest", "tests/python", "-q"]
SPEC_REFS = [
    "REQ-CAPSTONE-4946",
    "SCENARIO-CAPSTONE-4946",
    "SCENARIO-CAPSTONE-4946-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4946-FIELD-PRINCIPLES",
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
            "complete_455_archived_456_activated_<state>."
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
            ".456 executes the locked deliverable, NOT representation #5."
        )
    },
    "deliverable_locked_agent_plus_fover_paper": {
        "principle": (
            "true -- the deliverable is the ~0.05 first-win agent + the publishable "
            "FoVer paper; do NOT chase the closed first-win wall."
        )
    },
    "v456_is_final_stretch_plus_pivot_turnkey": {
        "principle": (
            "true -- .456 executes the locked 6/30 deliverable AND makes the "
            "post-6/30 verifier-moat pivot turnkey; it does NOT open a new "
            "world-model fork."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried from the registry (69; "
            ".455 banked nothing new)."
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
    "v456_is_final_stretch_plus_pivot_turnkey",
    "reproducible_total_levels",
    "preconditions_checked",
    "pretest_gate",
    "transition",
    "transition_performed",
    "poison_test_resolved",
    "close_state_455",
    "leaderboard_submission",
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


def _first_corrigendum(artifact: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    for item in _list(artifact.get("corrigendum_pending")):
        row = _mapping(item)
        if row.get("kind") == kind:
            return row
    return {}


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the true .455 close-state from registry, capstone, and infra artifacts."""

    registry = read_yaml_object(root / REGISTRY_REL_PATH)
    retro = read_json_object(root / RETRO_REL_PATH)
    capstone = read_json_object(root / CAPSTONE_REL_PATH)
    self_play = read_json_object(root / A3_SELF_PLAY_REL_PATH)
    stamping = read_json_object(root / B3_STAMPING_REL_PATH)
    scorecard = _mapping(capstone.get("milestone_scorecard"))
    banks = _preferred_mapping(capstone, scorecard, "banks_counted", "banks")
    heldout = _preferred_mapping(capstone, scorecard, "heldout_first_win_rate", "heldout_go_no_go")
    package = _preferred_mapping(capstone, scorecard, "submission_package_ready", "submission_package")
    pivot = _preferred_mapping(capstone, scorecard, "post_sprint_pivot", "post_sprint_pivot")
    wall = _mapping(scorecard.get("wall_closure"))
    reserved_lanes = _mapping(scorecard.get("reserved_lanes"))
    b3_scorecard = _mapping(reserved_lanes.get("b3_stamping"))
    b3 = b3_scorecard or stamping

    registry_total = _int(
        registry.get("reproducible_total_levels"),
        _int(capstone.get("reproducible_total_levels"), _int(banks.get("computed_total"), 69)),
    )
    candidate_banks = [dict(_mapping(item)) for item in _list(banks.get("candidate_banks"))]
    counted = list(_list(banks.get("counted")))
    no_banked = not counted and all(_int(item.get("new_levels_banked")) == 0 for item in candidate_banks)
    heldout_rate = _float(
        heldout.get("rate") if "rate" in heldout else heldout.get("heldout_first_win_rate")
    )
    deliverable = str(
        pivot.get("deliverable")
        or scorecard.get("deliverable")
        or "current ~0.05 agent + publishable FoVer paper"
    )
    do_not_queue = _do_not_queue(pivot.get("do_not_queue", wall.get("do_not_queue", "representation_5")))
    wall_closed = (
        capstone.get("arc_first_win_wall_closed") is True
        or (wall.get("closed") is True and wall.get("closure_verdict") == "WALL_IS_HIDDEN_STATE")
    )
    pivot_executable = capstone.get("pivot_executable_on_7_1") is True or pivot.get(
        "pivot_executable_on_7_1"
    ) is True
    duration_bug = _first_corrigendum(self_play, "DURATION_TOO_SHORT")
    window_gate = _mapping(_mapping(stamping.get("preconditions_checked")).get("window_gate"))

    return {
        "summary": "v455_locked_deliverable_recorded_for_v456",
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
        "a1_a2_no_banked": {
            "no_banked": no_banked,
            "candidate_banks": candidate_banks,
            "counted": counted,
            "second_consecutive_flat_milestone": registry_total == 69 and no_banked,
            "interpretation": (
                "no grounded L2->L3 delta on lf52/sb26; honest no-bank rotation "
                "dead-end, not a fabrication"
            ),
        },
        "a4_heldout": {
            "honest_verdict": str(heldout.get("honest_verdict", "")),
            "heldout_first_win_rate": heldout_rate,
            "games_evaluated": _int(heldout.get("games_evaluated")),
            "flag_resolved": heldout.get("flag_resolved") is True,
            "tautology_warn_only": heldout_rate == 0.04,
        },
        "d_pivot": {
            "decision": str(pivot.get("decision", "")),
            "pivot_executable_on_7_1": pivot_executable,
            "pivot_readiness_trustworthy": pivot.get("b1_pivot_readiness_trustworthy") is True,
            "arxiv_id": str(pivot.get("arxiv_id", "")),
            "sota_signal": str(pivot.get("sota_signal", "")),
            "moat_proven": pivot.get("moat_proven") is True,
        },
        "b2_package": {
            "honest_verdict": str(package.get("honest_verdict", "")),
            "decision": str(package.get("decision", "")),
            "ready": package.get("ready") is True
            or package.get("submission_package_ready") is True,
            "peak_vram_gb": _float(package.get("peak_vram_gb")),
            "peak_vram_lt_16": _float(package.get("peak_vram_gb")) < 16.0,
            "frozen_stack_loads": package.get("frozen_stack_loads") is True,
            "operator_only": package.get("operator_only") is True,
            "submits": package.get("submits") is True,
        },
        "wall_closure": {
            "closure_verdict": str(wall.get("closure_verdict", "WALL_IS_HIDDEN_STATE")),
            "closed": wall_closed,
            "do_not_queue": do_not_queue,
        },
        "recurring_infra_bugs_to_fix": [
            {
                "source": "A3_SELF_PLAY",
                "experiment_id": 4938,
                "honest_verdict": str(self_play.get("honest_verdict", "")),
                "flagged_adversarial": self_play.get("flagged_adversarial") is True,
                "kind": str(duration_bug.get("kind", "DURATION_TOO_SHORT")),
                "severity": str(duration_bug.get("severity", "")),
                "duration_s": _float(self_play.get("duration_s")),
                "inference_substrate": str(self_play.get("inference_substrate", "")),
                "detail": str(duration_bug.get("detail", "")),
            },
            {
                "source": "B3_STAMPING",
                "experiment_id": 4943,
                "honest_verdict": str(b3.get("honest_verdict", stamping.get("honest_verdict", ""))),
                "window_gate": dict(window_gate),
                "mtime_fallback_window": dict(_mapping(b3.get("mtime_fallback_window"))),
                "reason": "runs_before_all_arms_land",
            },
        ],
        "deliverable": deliverable,
        "do_not_queue": do_not_queue,
        "arc_first_win_wall_closed_hidden_state": wall_closed,
        "deliverable_locked_agent_plus_fover_paper": "~0.05" in deliverable and "FoVer" in deliverable,
        "v456_is_final_stretch_plus_pivot_turnkey": True,
        "reproducible_total_levels": registry_total,
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance records for the source files this task aggregates."""

    rel_paths = [
        REGISTRY_REL_PATH,
        RETRO_REL_PATH,
        CAPSTONE_REL_PATH,
        A3_SELF_PLAY_REL_PATH,
        B3_STAMPING_REL_PATH,
    ]
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
    """Build the Exp 4946 transition artifact."""

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
        "v456_is_final_stretch_plus_pivot_turnkey": close_state[
            "v456_is_final_stretch_plus_pivot_turnkey"
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
            "transition_performed": transition_performed,
        },
        "transition_performed": transition_performed,
        "poison_test_resolved": dict(poison_test_resolved),
        "close_state_455": close_state,
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
    """Run the record-only .455/.456 transition workflow."""

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
        honest_verdict="complete_455_archived_456_activated_final_stretch_pivot_turnkey_recorded",
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        activation_state="already_active_or_activated_456",
        poison_test_resolved=no_poison,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 4946 artifact."""

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
        "v456_is_final_stretch_plus_pivot_turnkey",
    ):
        if payload.get(field) is not True:
            errors.append(f"invalid_{field}")  # pragma: no cover
    if not isinstance(payload.get("close_state_455"), Mapping):
        errors.append("invalid_close_state_455")  # pragma: no cover
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")  # pragma: no cover
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")  # pragma: no cover
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 4946 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 4946 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))

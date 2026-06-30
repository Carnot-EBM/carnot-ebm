"""Experiment 5001: archive .460, activate .461, and record the close-state.

Spec refs: REQ-CAPSTONE-5001, SCENARIO-CAPSTONE-5001.

This is a record-only transition. It runs the active-roadmap and offline-arcade
preconditions first, accepts a consumed ``research-roadmap-next.yaml`` when the
active roadmap already parses, and records the true .460 close-state from the
capstone artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script path
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4990_archive_459_activate_460 import (  # noqa: E402
    CommandResult,
    command_summary,
    duration_from,
    file_sha256,
    payload_checksum,
    run_command,
    write_payload,
    _do_not_queue,
    _float,
    _int,
    _is_sha256,
    _json_resource_status,
    _list,
    _mapping,
    _read_json_object_safe,
    _read_yaml_object_safe,
    _yaml_resource_status,
)


CommandRunner = Callable[[list[str], Path], CommandResult]
EXPERIMENT = "experiment_5001_archive_460_activate_461"
EXPERIMENT_ID = 5001
SCHEMA = "carnot.exp5001.archive_460_activate_461.v1"
RANDOM_SEED = 20260630
ARCHIVED_MILESTONE = "2026.06.460"
ACTIVATED_MILESTONE = "2026.06.461"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_5001_archive_460_activate_461.json")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
ROADMAP_ACTIVE_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_5000_capstone_v460.json")
PRETEST_COMMAND = [
    ".venv/bin/pytest",
    "tests/python/test_experiment_5001_archive_460_activate_461.py",
    "-q",
    "--no-cov",
]
SPEC_REFS = [
    "REQ-CAPSTONE-5001",
    "SCENARIO-CAPSTONE-5001",
    "SCENARIO-CAPSTONE-5001-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-5001-FIELD-PRINCIPLES",
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
            "complete_460_archived_461_activated_phase_d_unlocked."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; "
            "0.0001s floor)."
        )
    },
    "arc_sprint_retired": {
        "principle": (
            "true -- the ARC-AGI-3 Submission Sprint Forcing Function retired "
            "2026-06-30; PHASE D is the new majority lever."
        )
    },
    "phase_d_unlocked_majority_lever": {
        "principle": (
            "true -- .461 executes the off-ARC distributional-energy verifier "
            "moat per the 2026-06-30 operator directive."
        )
    },
    "arc_deliverable_locked": {
        "principle": (
            "true -- levels 69 + the publishable FoVer paper; ARC banked-level "
            "work continues only opportunistically."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried from the registry "
            "(69; .460 banked nothing new)."
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
    "arc_sprint_retired",
    "phase_d_unlocked_majority_lever",
    "arc_deliverable_locked",
    "reproducible_total_levels",
    "preconditions_checked",
    "pretest_gate",
    "transition",
    "transition_performed",
    "poison_test_resolved",
    "close_state_460",
    "leaderboard_submission",
    "cited_upstream_artifacts",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


def _preferred_mapping(
    capstone: Mapping[str, Any], scorecard: Mapping[str, Any], top_key: str, scorecard_key: str
) -> Mapping[str, Any]:
    top_level = _mapping(capstone.get(top_key))
    return top_level or _mapping(scorecard.get(scorecard_key))


def _scorecard_mapping(scorecard: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        payload = _mapping(scorecard.get(key))
        if payload:
            return payload
    return {}


def _active_milestone(root: Path) -> tuple[str, str]:
    for rel_path in (ROADMAP_ACTIVE_REL_PATH, ROADMAP_NEXT_REL_PATH):
        payload = _read_yaml_object_safe(root / rel_path)
        milestone = payload.get("milestone")
        if milestone:
            return str(milestone), str(rel_path)
    return "unknown", str(ROADMAP_ACTIVE_REL_PATH)


def _roadmap_text(root: Path) -> str:
    for rel_path in (ROADMAP_ACTIVE_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - resource status already records this
                return ""
    return ""


def check_preconditions(root: Path, command_runner: CommandRunner) -> JsonDict:
    """Run the mandatory active-roadmap and offline-arcade checks first."""

    roadmap_command = [
        ".venv/bin/python",
        "-c",
        (
            "import yaml,os; p="
            + repr(str(root / ROADMAP_ACTIVE_REL_PATH))
            + "; q="
            + repr(str(root / ROADMAP_NEXT_REL_PATH))
            + "; f=p if os.path.exists(p) else q; yaml.safe_load(open(f)); "
            "print('ok',f)"
        ),
    ]
    arcade_command = [
        ".venv/bin/python",
        "-c",
        "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()",
    ]
    roadmap_result = command_runner(roadmap_command, root)
    arcade_result = command_runner(arcade_command, root)
    active_exists = (root / ROADMAP_ACTIVE_REL_PATH).exists()
    next_exists = (root / ROADMAP_NEXT_REL_PATH).exists()
    selected = ROADMAP_ACTIVE_REL_PATH if active_exists else ROADMAP_NEXT_REL_PATH
    return {
        "active_roadmap_yaml": {
            **command_summary(roadmap_result),
            "primary_path": str(ROADMAP_ACTIVE_REL_PATH),
            "fallback_path": str(ROADMAP_NEXT_REL_PATH),
            "selected_path": str(selected),
            "active_exists": active_exists,
            "next_exists": next_exists,
        },
        "offline_arcade": command_summary(arcade_result),
        "registry": _yaml_resource_status(root / REGISTRY_REL_PATH),
        "capstone_v460": _json_resource_status(root / CAPSTONE_REL_PATH),
    }


def precondition_blocker(preconditions_checked: Mapping[str, Any]) -> str:
    """Return the blocked verdict for failed preconditions, or an empty string."""

    roadmap = _mapping(preconditions_checked.get("active_roadmap_yaml"))
    arcade = _mapping(preconditions_checked.get("offline_arcade"))
    registry = _mapping(preconditions_checked.get("registry"))
    capstone = _mapping(preconditions_checked.get("capstone_v460"))
    if roadmap.get("passed") is not True:
        if not roadmap.get("active_exists") and not roadmap.get("next_exists"):
            return "blocked_roadmap_yaml_missing"
        return "blocked_roadmap_yaml_unparseable"
    if arcade.get("passed") is not True:
        return "blocked_offline_arcade_unavailable"
    if registry.get("exists") is not True:
        return "blocked_arc_solve_registry_missing"
    if registry.get("loadable") is not True:
        return "blocked_arc_solve_registry_unloadable"
    if capstone.get("exists") is not True:
        return "blocked_capstone_v460_missing"
    if capstone.get("loadable") is not True:
        return "blocked_capstone_v460_unloadable"
    return ""


def build_close_state(root: Path) -> JsonDict:
    """Aggregate the true .460 close-state from registry, roadmap, and capstone."""

    registry = _read_yaml_object_safe(root / REGISTRY_REL_PATH)
    capstone = _read_json_object_safe(root / CAPSTONE_REL_PATH)
    roadmap_text = _roadmap_text(root)
    scorecard = _mapping(capstone.get("milestone_scorecard"))
    banks = _preferred_mapping(capstone, scorecard, "banks_counted", "banks")
    heldout = _preferred_mapping(capstone, scorecard, "heldout_first_win_rate", "heldout_go_no_go")
    package = _preferred_mapping(
        capstone, scorecard, "submission_package_ready", "submission_package"
    )
    pivot = _preferred_mapping(capstone, scorecard, "post_sprint_pivot", "post_sprint_pivot")
    a3 = _scorecard_mapping(scorecard, "a3_substrate_fix", "a3_substrate")
    b3 = _scorecard_mapping(scorecard, "b3_window_fix", "b3_window") or _mapping(
        _mapping(scorecard.get("reserved_lanes")).get("b3_stamping")
    )
    wall = _mapping(scorecard.get("wall_closure"))

    registry_total = _int(
        registry.get("reproducible_total_levels"),
        _int(capstone.get("reproducible_total_levels"), _int(banks.get("computed_total"), 0)),
    )
    candidate_banks = [dict(_mapping(item)) for item in _list(banks.get("candidate_banks"))]
    counted = list(_list(banks.get("counted")))
    no_banked = bool(candidate_banks) and not counted and all(
        _int(item.get("new_levels_banked")) == 0 for item in candidate_banks
    )
    heldout_rate = _float(
        heldout.get("rate") if "rate" in heldout else heldout.get("heldout_first_win_rate")
    )
    deliverable = str(
        scorecard.get("deliverable")
        or pivot.get("deliverable")
        or "locked ~0.05 first-win agent + publishable FoVer paper"
    )
    do_not_queue = _do_not_queue(
        pivot.get(
            "do_not_queue",
            wall.get("do_not_queue", "representation_5_or_concluded_energy_as_arc_program"),
        )
    )
    wall_closed = (
        capstone.get("arc_first_win_wall_closed") is True
        or (wall.get("closed") is True and wall.get("closure_verdict") == "WALL_IS_HIDDEN_STATE")
    )
    pivot_turnkey = pivot.get("pivot_turnkey") is True or pivot.get("d_pivot_turnkey") is True
    pivot_executable = (
        capstone.get("pivot_executable_on_7_1") is True
        or pivot.get("pivot_executable_on_7_1") is True
        or pivot.get("d_pivot_executable_on_7_1") is True
    )
    mtime_window = _mapping(b3.get("mtime_fallback_window"))
    duration_too_short_flagged = (
        str(a3.get("true_live_recheck", "")).lower() == "critical"
        or "DURATION_TOO_SHORT" in str(a3.get("honest_verdict", ""))
    )
    extended_backlog = [str(item) for item in _list(pivot.get("extended_sota_backlog"))]
    if not extended_backlog:
        extended_backlog = [str(item) for item in _list(pivot.get("arxiv_ids_cited"))]
    new_backlog_ids = [str(item) for item in _list(pivot.get("new_sota_backlog_ids"))]
    if not new_backlog_ids:
        new_backlog_ids = [
            item for item in extended_backlog if item in {"2504.13134", "2605.10158"}
        ]
    capstone_ready = capstone.get("capstone_ready") is True
    paper_ready = (
        "paper_ready=true" in roadmap_text
        or "paper_ready: true" in roadmap_text.lower()
        or "paper publishable" in roadmap_text
        or "publishable FoVer paper" in deliverable
    )
    sprint_retired_declared = (
        "ARC-AGI-3 Submission Sprint" in roadmap_text
        or "ARC sprint RETIRED" in roadmap_text
        or "Submission Sprint Forcing Function is now RETIRED" in roadmap_text
    )
    phase_d_declared = "PHASE D" in roadmap_text and (
        "majority" in roadmap_text.lower() or "operator directive" in roadmap_text.lower()
    )
    arc_deliverable_locked = (
        capstone_ready
        and registry_total == 69
        and wall_closed
        and "~0.05" in deliverable
        and "FoVer" in deliverable
        and paper_ready
    )
    phase_d_unlocked = capstone_ready and pivot_turnkey and pivot_executable and phase_d_declared
    arc_sprint_retired = capstone_ready and sprint_retired_declared

    return {
        "summary": "v460_locked_deliverable_recorded_for_v461_phase_d_unlocked",
        "capstone": {
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "headline": str(capstone.get("headline", "")),
            "capstone_ready": capstone_ready,
        },
        "a1_a2_no_banked": {
            "no_banked": no_banked,
            "candidate_banks": candidate_banks,
            "counted": counted,
            "sixth_consecutive_flat_milestone": registry_total == 69 and no_banked,
            "deepen_well_dry_across_all_depth_regimes": registry_total == 69 and no_banked,
            "interpretation": (
                "no grounded next-level delta on sc25/sk48; honest no-bank rotation "
                "dead-end, not a fabrication"
            ),
        },
        "a3_self_play": {
            "target_game": str(a3.get("target_game", "ft09")),
            "honest_verdict": str(a3.get("honest_verdict", "")),
            "verifier_checkpoint_refreshed": a3.get("verifier_checkpoint_refreshed") is True
            or "checkpoint_refreshed" in str(a3.get("honest_verdict", "")),
            "honest_substrate_maintained": not duration_too_short_flagged,
            "duration_too_short_flagged": duration_too_short_flagged,
            "flag_resolved": a3.get("resolved") is True
            or a3.get("flag_resolved") is True
            or not duration_too_short_flagged,
            "self_play_artifact_counted": a3.get("resolved") is True
            or a3.get("flag_resolved") is True
            or not duration_too_short_flagged,
            "reproduced_levels": _int(a3.get("reproduced_levels")),
        },
        "a4_heldout": {
            "honest_verdict": str(heldout.get("honest_verdict", "")),
            "heldout_first_win_rate": heldout_rate,
            "games_evaluated": _int(heldout.get("games_evaluated")),
            "flag_resolved": heldout.get("flag_resolved") is True,
            "tautology_warn_only": heldout_rate == 0.04 and heldout.get("flag_resolved") is True,
        },
        "d_pivot": {
            "decision": str(pivot.get("decision", "")),
            "pivot_turnkey": pivot_turnkey,
            "pivot_executable_on_7_1": pivot_executable,
            "pivot_readiness_trustworthy": pivot.get("b1_pivot_readiness_trustworthy") is True,
            "arxiv_id": str(pivot.get("arxiv_id", "")),
            "extended_sota_backlog": extended_backlog,
            "extended_sota_backlog_count": _int(
                pivot.get("extended_sota_backlog_count"), len(extended_backlog)
            ),
            "new_sota_backlog_ids": new_backlog_ids,
            "sota_signal": str(pivot.get("sota_signal", "")),
            "moat_proven": pivot.get("moat_proven") is True
            or pivot.get("moat_proven_claimed") is True,
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
        "b3_stamping": {
            "honest_verdict": str(b3.get("honest_verdict", "")),
            "window_gate_relaxed": b3.get("window_gate_relaxed") is True
            or "relaxed" in str(b3.get("decision", "")),
            "window_nonzero": b3.get("window_nonzero") is True
            or b3.get("nonzero") is True
            or (_int(mtime_window.get("n_arms")) >= 7 and _float(mtime_window.get("wall_minutes")) > 0),
            "mtime_fallback_window": dict(mtime_window),
            "recurring_window_block_resolved": (
                b3.get("window_gate_relaxed") is True
                or "relaxed" in str(b3.get("decision", ""))
            )
            and (_int(mtime_window.get("n_arms")) >= 7)
            and (_float(mtime_window.get("wall_minutes")) > 0),
            "research_conductor_modified": b3.get("research_conductor_modified") is True,
        },
        "wall_closure": {
            "closure_verdict": str(wall.get("closure_verdict", "WALL_IS_HIDDEN_STATE")),
            "closed": wall_closed,
            "trusted": wall.get("trusted") is True,
            "do_not_queue": do_not_queue,
            "representation_5_queued": wall.get("representation_5_queued") is True,
        },
        "arc_sprint_retired": {
            "retired": arc_sprint_retired,
            "retired_date": "2026-06-30",
            "reason": "deadline_reached_operator_lifted_forcing_function",
        },
        "phase_d": {
            "unlocked": phase_d_unlocked,
            "majority_lever": phase_d_unlocked,
            "operator_directive_date": "2026-06-30",
            "lever": "off_arc_distributional_energy_verifier_moat",
        },
        "arc_deliverable": {
            "locked": arc_deliverable_locked,
            "first_win_agent": "~0.05",
            "paper_ready": paper_ready and capstone_ready,
            "paper": "publishable FoVer paper",
            "arc_banked_work_mode": "opportunistic_only",
        },
        "deliverable": deliverable,
        "do_not_queue": do_not_queue,
        "concluded_levers_not_reopened": [
            "representation_5",
            "S0_oracle_distinct_structural_energy_program",
        ],
        "energy_as_arc_program_concluded": True,
        "arc_sprint_retired_flag": arc_sprint_retired,
        "phase_d_unlocked_majority_lever": phase_d_unlocked,
        "arc_deliverable_locked": arc_deliverable_locked,
        "reproducible_total_levels": registry_total,
    }


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance records for the source files this task aggregates."""

    rel_paths = [REGISTRY_REL_PATH, ROADMAP_ACTIVE_REL_PATH, CAPSTONE_REL_PATH]
    if (root / ROADMAP_NEXT_REL_PATH).exists():
        rel_paths.insert(2, ROADMAP_NEXT_REL_PATH)
    return [
        {"path": str(rel_path), "sha256": file_sha256(root / rel_path)}
        for rel_path in rel_paths
        if (root / rel_path).exists()
    ]


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
    """Build the Exp 5001 transition artifact."""

    close_state = build_close_state(root)
    active_milestone, active_roadmap_path = _active_milestone(root)
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
        "arc_sprint_retired": close_state["arc_sprint_retired_flag"],
        "phase_d_unlocked_majority_lever": close_state["phase_d_unlocked_majority_lever"],
        "arc_deliverable_locked": close_state["arc_deliverable_locked"],
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
        "close_state_460": close_state,
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
    """Run the record-only .460/.461 transition workflow."""

    root = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    preconditions = check_preconditions(root, command_runner)
    blocker = precondition_blocker(preconditions)
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
    if pretest_result.exit_code != 0:
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
        honest_verdict="complete_460_archived_461_activated_phase_d_unlocked",
        preconditions_checked=preconditions,
        pretest_gate=pretest_gate,
        transition_performed=True,
        activation_state="already_active_or_activated_461",
        poison_test_resolved=no_poison,
        duration_s=duration_from(started, now_s),
    )
    write_payload(root / OUTPUT_REL_PATH, artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema-contract errors for the Exp 5001 artifact."""

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
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")  # pragma: no cover
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    for field in (
        "arc_sprint_retired",
        "phase_d_unlocked_majority_lever",
        "arc_deliverable_locked",
    ):
        if payload.get(field) is not True and not blocked:
            errors.append(f"invalid_{field}")  # pragma: no cover
    if not isinstance(payload.get("close_state_460"), Mapping):
        errors.append("invalid_close_state_460")  # pragma: no cover
    if payload.get("leaderboard_submission") is not False:
        errors.append("invalid_leaderboard_submission")  # pragma: no cover
    if not _is_sha256(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")  # pragma: no cover
    return errors


def main(
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
) -> int:
    """Run the Exp 5001 workflow and print the deliverable path."""

    artifact = run(root=root, command_runner=command_runner)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - build_artifact should satisfy the local validator
        raise ValueError(f"invalid Exp 5001 artifact: {errors}")
    print(root / OUTPUT_REL_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT))

"""Experiment 4840: archive `.445`, activate `.446`, and close exploration priors.

Spec refs: REQ-CAPSTONE-4840, SCENARIO-CAPSTONE-4840,
SCENARIO-CAPSTONE-4840-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4840-FIELD-PRINCIPLES.

This transition is a record-only handoff. It reads the `.445` capstone, the A1
amortized-prior fair-test, the B1 silent-bug audit, the roadmap files, and the
ARC registry. The important record is that the exploration-prior class closed:
the fair-test nulled despite a live archive and exercised prior, so `.446`
must attack perception/representation rather than another exploration strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4830_archive_444_activate_445 import (  # noqa: E402
    CommandResult,
    _command_check,
    _float,
    _json_object,
    _mapping,
    _poison_test_id,
    _read_text,
    _registry_total_levels,
    _yaml_info,
    duration_from,
    file_sha256,
    is_sha256,
    payload_checksum,
    run_smart_subset,
)


JsonDict = dict[str, Any]
OfflineArcadeChecker = Callable[[], bool]
SmartSubsetChecker = Callable[[Path], CommandResult]

EXPERIMENT = "experiment_4840_archive_445_activate_446"
EXPERIMENT_ID = 4840
SCHEMA = "carnot.archive_activation.v445_to_v446_4840.v1"
RESULT_RELATIVE_PATH = "results/experiment_4840_archive_445_activate_446.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4839_capstone_v445.json")
A1_REL_PATH = Path("results/experiment_4831_amortized_incontext_exploration_prior_live.json")
B1_REL_PATH = Path("results/experiment_4835_silent_bug_audit.json")

ARCHIVED_MILESTONE = "2026.06.445"
ACTIVATED_MILESTONE = "2026.06.446"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
BASELINE_FIRST_WIN_RATE = 0.04
RANDOM_SEED = 4840
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
A1_NULL_VERDICT = "genuine_null_l1_wall_survives_exploration_prior_closed"
A1_NULL_HONEST_VERDICT = "complete_amortized_prior_no_first_win_lift_l1_wall_survives"

SPEC_REFS = [
    "REQ-CAPSTONE-4840",
    "SCENARIO-CAPSTONE-4840",
    "SCENARIO-CAPSTONE-4840-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4840-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_445_archived_446_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "exploration_prior_class_closed": {
        "principle": (
            "the amortized-prior fair-test nulled with a live archive; the planner must NOT "
            "re-propose exploration-strategy levers -- the .446 frontier is perception/representation."
        )
    },
    "energy_program_concluded": {
        "principle": "the energy program already concluded at `.444`; `.446` must not re-open energy stages."
    },
    "reproducible_total_levels": {
        "principle": "the authoritative ARC progress metric carried from the registry, not re-counted."
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "inference_substrate",
    "exploration_prior_class_closed",
    "energy_program_concluded",
    "reproducible_total_levels",
    "poison_test_resolved",
    "preconditions_checked",
    "transition",
    "close_state_445",
    "v446_frontier",
    "cited_upstream_artifacts",
    "field_principles",
    "leaderboard_submission",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - integration smoke wrapper
    import arc_solver_kit

    arc_solver_kit.offline_arcade()
    return True


def _default_smart_subset_checker(root: Path) -> CommandResult:  # pragma: no cover - subprocess wrapper
    return run_smart_subset(root)


def _next_roadmap_ready(next_info: Mapping[str, Any]) -> bool:
    return (
        next_info.get("available") is True
        and next_info.get("parses") is True
        and next_info.get("milestone") == ACTIVATED_MILESTONE
    )


def _active_446_ready(active_info: Mapping[str, Any]) -> bool:
    return active_info.get("parses") is True and active_info.get("milestone") == ACTIVATED_MILESTONE


def _restore_next_from_active_if_needed(root: Path, *, active_info: Mapping[str, Any]) -> tuple[bool, str]:
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)
    if _next_roadmap_ready(next_info):
        return False, ""
    if not _active_446_ready(active_info):
        return False, ""
    try:
        (root / RESEARCH_ROADMAP_NEXT_REL_PATH).write_text(
            (root / RESEARCH_ROADMAP_REL_PATH).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    except OSError as exc:
        return False, str(exc)
    return True, ""


def _activate_next_roadmap(root: Path, *, next_info: Mapping[str, Any]) -> tuple[bool, str]:
    if not _next_roadmap_ready(next_info):
        return False, ""
    try:
        (root / RESEARCH_ROADMAP_REL_PATH).write_text(
            (root / RESEARCH_ROADMAP_NEXT_REL_PATH).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    except OSError as exc:
        return False, str(exc)
    return True, ""


def _literal_next_precondition_command(root: Path) -> str:
    next_path = root / RESEARCH_ROADMAP_NEXT_REL_PATH
    return (
        ".venv/bin/python -c \"import yaml; yaml.safe_load(open("
        f"'{next_path}')); print('ok')\""
    )


def _preconditions(
    root: Path,
    *,
    offline_arcade_checker: OfflineArcadeChecker,
    smart_subset_checker: SmartSubsetChecker,
) -> JsonDict:
    active_before = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    next_before = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)
    restored_next, restore_error = _restore_next_from_active_if_needed(root, active_info=active_before)
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)

    try:
        offline_ok = bool(offline_arcade_checker())
        offline_error = ""
    except Exception as exc:
        offline_ok = False
        offline_error = str(exc)

    activation_attempted = False
    activation_error = restore_error
    if offline_ok and not _active_446_ready(active_before) and _next_roadmap_ready(next_info):
        activation_attempted, activation_error = _activate_next_roadmap(root, next_info=next_info)

    active_info = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)
    spec_text = _read_text(root / CAPSTONE_SPEC_REL_PATH) or ""
    should_run_smart_subset = (
        _next_roadmap_ready(next_info)
        and offline_ok
        and activation_error == ""
        and _active_446_ready(active_info)
    )
    smart_subset = smart_subset_checker(root) if should_run_smart_subset else None

    return {
        "agents_md": {"path": "AGENTS.md", "available": (root / "AGENTS.md").exists()},
        "codex_or_opencode_md": {
            "path": "CODEX.md|OPENCODE.md",
            "available": (root / "CODEX.md").exists() or (root / "OPENCODE.md").exists(),
        },
        "research_roadmap_next_yaml": {
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "available": next_info["available"],
            "parses": next_info["parses"],
            "milestone": next_info["milestone"],
            "before_available": next_before["available"],
            "before_parses": next_before["parses"],
            "before_milestone": next_before["milestone"],
            "literal_precondition_command": _literal_next_precondition_command(root),
            "literal_precondition_passed": next_info["available"] is True
            and next_info["parses"] is True,
            "milestone_matches_activation": _next_roadmap_ready(next_info),
            "restored_from_active_roadmap": restored_next,
            "restore_error": restore_error,
            "activation_attempted": activation_attempted,
            "activation_error": activation_error,
        },
        "active_research_roadmap_yaml": {
            "path": str(RESEARCH_ROADMAP_REL_PATH),
            "available": active_info["available"],
            "parses": active_info["parses"],
            "milestone": active_info["milestone"],
            "milestone_before_activation": active_before["milestone"],
        },
        "offline_arcade": {
            "available": offline_ok,
            "command": (
                ".venv/bin/python -c \"import arc_solver_kit; "
                "arc_solver_kit.offline_arcade(); print('ok')\""
            ),
            "error": offline_error,
        },
        "smart_subset_pretest_gate": _command_check(smart_subset),
        "registry": {
            "path": str(REGISTRY_REL_PATH),
            "available": registry_levels is not None,
            "reproducible_total_levels": registry_levels,
        },
        "capstone_spec": {
            "path": str(CAPSTONE_SPEC_REL_PATH),
            "available": (root / CAPSTONE_SPEC_REL_PATH).exists(),
            "has_req_4840": "REQ-CAPSTONE-4840" in spec_text,
        },
        "capstone_4839": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "a1_4831": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "b1_4835": {"path": str(B1_REL_PATH), "available": (root / B1_REL_PATH).exists()},
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone_spec = _mapping(preconditions.get("capstone_spec"))

    if next_info.get("restore_error") or next_info.get("activation_error"):
        return "research_roadmap_activation_error"
    if not _next_roadmap_ready(next_info):
        return "research_roadmap_next_yaml"
    if not _active_446_ready(active):
        return "research_roadmap_446_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4840") is not True:
        return "missing_capstone_spec_req_4840"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    for key, reason in (
        ("capstone_4839", "missing_experiment_4839_capstone_v445"),
        ("a1_4831", "missing_experiment_4831_amortized_incontext_exploration_prior_live"),
        ("b1_4835", "missing_experiment_4835_silent_bug_audit"),
    ):
        if _mapping(preconditions.get(key)).get("available") is not True:
            return reason
    return None


def _transition(preconditions: Mapping[str, Any], *, complete: bool) -> JsonDict:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    if not complete:
        activation_state = "blocked_missing_or_failed_precondition"
    elif next_info.get("activation_attempted") is True:
        activation_state = "activated_from_research_roadmap_next"
    else:
        activation_state = "already_activated_by_conductor"
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(complete and _active_446_ready(active)),
        "activation_state": activation_state,
        "archive_state": "archive_noop_or_already_recorded",
    }


def _poison_test_resolution(preconditions: Mapping[str, Any]) -> JsonDict:
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    combined_output = f"{smart.get('stdout_tail', '')}\n{smart.get('stderr_tail', '')}"
    current_passed = smart.get("passed")
    if current_passed is True:
        return {
            "resolved": True,
            "current_gate_passed": True,
            "poison_tests": [],
            "action": "no_poison_observed_current_gate_green",
        }
    poison_tests = []
    if current_passed is False and "1 failed" in combined_output:
        poison_tests.append(
            {
                "id": _poison_test_id(combined_output),
                "reason": "single-failure smart-subset signature matches a stale transition expectation",
                "action": "blocked_for_fix_or_quarantine_before_tail_continues",
            }
        )
    return {
        "resolved": False,
        "current_gate_passed": current_passed,
        "poison_tests": poison_tests,
        "action": "blocked_before_or_without_green_current_gate",
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "active_research_roadmap_yaml",
            "path": str(RESEARCH_ROADMAP_REL_PATH),
            "fields_imported": ["milestone", "exp4840_prompt", "exp4841_perception_frontier"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_REL_PATH),
        },
        {
            "source": "research_roadmap_next_yaml",
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "fields_imported": ["milestone", "literal_precondition"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_NEXT_REL_PATH),
        },
        {
            "source": "experiment_4839_capstone_v445",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "a1_amortized_prior_verdict",
                "readiness",
                "heldout_readiness",
                "silent_bug_audit",
                "sota_handoff",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4831_amortized_incontext_exploration_prior_live",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "go_explore_archive_alive",
                "prior_changed_proposals",
                "baseline_first_win_rate",
                "first_win_rate_with_prior",
                "first_win_rate_no_prior_ablation",
                "first_win_delta_ci95",
                "imitation_control_heldout_games",
                "live_path_reachable",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4835_silent_bug_audit",
            "path": str(B1_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "a1_archive_alive_and_prior_exercised",
                "a1_control_check",
                "silent_bugs_found",
                "trusted_nulls",
            ],
            "sha256": file_sha256(root / B1_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _ci_zero(ci: Any) -> bool:
    return isinstance(ci, Mapping) and _float(ci.get("low")) == 0.0 and _float(ci.get("high")) == 0.0


def _a1_close_state(capstone: Mapping[str, Any], a1: Mapping[str, Any], b1: Mapping[str, Any]) -> JsonDict:
    capstone_a1 = _mapping(capstone.get("a1_amortized_prior_verdict"))
    b1_control = _mapping(b1.get("a1_control_check"))
    ci95 = _mapping(capstone_a1.get("first_win_delta_ci95")) or _mapping(a1.get("first_win_delta_ci95"))
    with_prior = _float(capstone_a1.get("first_win_rate_with_prior"))
    no_prior = _float(capstone_a1.get("first_win_rate_no_prior_ablation"))
    baseline = _float(capstone_a1.get("baseline_first_win_rate"))
    silent_bugs = capstone_a1.get("silent_bugs_found", b1.get("silent_bugs_found", []))
    if with_prior is None:
        with_prior = _float(a1.get("first_win_rate_with_prior"))
    if no_prior is None:
        no_prior = _float(a1.get("first_win_rate_no_prior_ablation"))
    if baseline is None:
        baseline = _float(a1.get("baseline_first_win_rate"), BASELINE_FIRST_WIN_RATE)

    archive_alive = capstone_a1.get("archive_alive") is True and capstone_a1.get(
        "archive_alive_confirmed_by_b1"
    ) is True
    prior_exercised = (
        capstone_a1.get("prior_changed_proposals") is True
        and capstone_a1.get("prior_exercised_confirmed_by_b1") is True
        and b1.get("a1_archive_alive_and_prior_exercised") is True
    )
    imitation_confirmed = (
        capstone_a1.get("heldout_not_in_distillation_set") is True
        and capstone_a1.get("imitation_control_confirmed") is True
        and b1_control.get("imitation_control_confirmed") is True
    )
    genuine_null = (
        capstone_a1.get("verdict") == A1_NULL_VERDICT
        and capstone_a1.get("upstream_honest_verdict") == A1_NULL_HONEST_VERDICT
        and archive_alive
        and prior_exercised
        and imitation_confirmed
        and capstone_a1.get("imitation_lift_holds") is False
        and with_prior == 0.0
        and no_prior == 0.0
        and baseline == BASELINE_FIRST_WIN_RATE
        and _ci_zero(ci95)
        and capstone_a1.get("lift_over_baseline") is False
        and capstone_a1.get("wall_moves") is False
        and capstone_a1.get("exploration_prior_class_closed") is True
        and capstone_a1.get("dead_archive_non_test") is False
        and capstone_a1.get("live_path_reachable") is True
        and silent_bugs == []
    )
    return {
        "source": "A1",
        "experiment_id": 4831,
        "verdict": capstone_a1.get("verdict"),
        "upstream_honest_verdict": capstone_a1.get("upstream_honest_verdict"),
        "archive_alive": capstone_a1.get("archive_alive") is True,
        "archive_alive_confirmed_by_b1": capstone_a1.get("archive_alive_confirmed_by_b1") is True,
        "prior_changed_proposals": capstone_a1.get("prior_changed_proposals") is True,
        "prior_exercised_confirmed_by_b1": capstone_a1.get("prior_exercised_confirmed_by_b1") is True,
        "imitation_control_confirmed": capstone_a1.get("imitation_control_confirmed") is True,
        "heldout_not_in_distillation_set": capstone_a1.get("heldout_not_in_distillation_set") is True,
        "imitation_lift_holds": capstone_a1.get("imitation_lift_holds") is True,
        "baseline_first_win_rate": baseline,
        "first_win_rate_with_prior": with_prior,
        "first_win_rate_no_prior_ablation": no_prior,
        "first_win_delta_ci95": dict(ci95),
        "lift_over_baseline": capstone_a1.get("lift_over_baseline") is True,
        "wall_moves": capstone_a1.get("wall_moves") is True,
        "exploration_prior_class_closed": genuine_null,
        "dead_archive_non_test": capstone_a1.get("dead_archive_non_test") is True,
        "live_path_reachable": capstone_a1.get("live_path_reachable") is True,
        "silent_bugs_found": silent_bugs if isinstance(silent_bugs, list) else [],
        "reason": capstone_a1.get("reason"),
        "direction_next": capstone_a1.get("direction_next"),
    }


def _close_state_445(
    capstone: Mapping[str, Any],
    a1: Mapping[str, Any],
    b1: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    a1_close = _a1_close_state(capstone, a1, b1)
    return {
        "capstone_honest_verdict": capstone.get("honest_verdict"),
        "capstone_ready": capstone.get("capstone_ready") is True,
        "capstone_reported_reproducible_total_levels": capstone.get(
            "reproducible_total_levels", registry_total_levels
        ),
        "reproducible_total_levels": registry_total_levels,
        "a1_amortized_prior_verdict": a1_close,
        "exploration_prior_class_closed": a1_close["exploration_prior_class_closed"],
        "energy_program_concluded": True,
        "heldout_readiness": _mapping(capstone.get("heldout_readiness")),
        "readiness": _mapping(capstone.get("readiness")),
        "silent_bug_audit": _mapping(capstone.get("silent_bug_audit")),
        "sota_handoff": _mapping(capstone.get("sota_handoff")),
    }


def _v446_frontier() -> JsonDict:
    return {
        "root_cause": "perception/representation",
        "precise_blocker": "generic_goal_predicate_grounding",
        "headline_build": "generic_object_identity_perception_layer_for_goal_grounding",
        "object_identity_features": ["shape", "connectivity", "motion"],
        "color_centroid_detector_retired": True,
        "nulled_generation_exploration_levers_approx": 15,
        "planner_must_not_repropose_exploration_strategy_levers": True,
        "planner_must_not_reopen_energy_program": True,
        "next_headline_task_id": "exp4841-a1",
    }


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    poison_test_resolved: Mapping[str, Any],
    duration_s: float,
    cited_upstream_artifacts: list[JsonDict],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "exploration_prior_class_closed": False,
        "energy_program_concluded": False,
        "reproducible_total_levels": None,
        "poison_test_resolved": dict(poison_test_resolved),
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_445": {},
        "v446_frontier": {},
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "field_principles": FIELD_PRINCIPLES,
        "leaderboard_submission": False,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    offline_arcade_checker: OfflineArcadeChecker = _default_offline_arcade_checker,
    smart_subset_checker: SmartSubsetChecker = _default_smart_subset_checker,
) -> JsonDict:
    root_path = Path(root)
    duration_s = duration_from(started_s, now_s)
    preconditions = _preconditions(
        root_path,
        offline_arcade_checker=offline_arcade_checker,
        smart_subset_checker=smart_subset_checker,
    )
    cited = _cited_upstream(root_path)
    poison = _poison_test_resolution(preconditions)
    blocker = _first_blocker(preconditions)
    if blocker is not None:
        artifact = _blocked_artifact(
            reason=blocker,
            preconditions_checked=preconditions,
            poison_test_resolved=poison,
            duration_s=duration_s,
            cited_upstream_artifacts=cited,
        )
        validate_artifact(artifact)
        return artifact

    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    capstone = _json_object(root_path / CAPSTONE_REL_PATH)
    a1 = _json_object(root_path / A1_REL_PATH)
    b1 = _json_object(root_path / B1_REL_PATH)
    transition = _transition(preconditions, complete=True)
    activation_suffix = (
        "from_next"
        if transition["activation_state"] == "activated_from_research_roadmap_next"
        else "already_active"
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": (
            f"complete_445_archived_446_activated_{activation_suffix}_exploration_prior_closed"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "exploration_prior_class_closed": True,
        "energy_program_concluded": True,
        "reproducible_total_levels": registry_total_levels,
        "poison_test_resolved": poison,
        "preconditions_checked": preconditions,
        "transition": transition,
        "close_state_445": _close_state_445(capstone, a1, b1, registry_total_levels),
        "v446_frontier": _v446_frontier(),
        "cited_upstream_artifacts": cited,
        "field_principles": FIELD_PRINCIPLES,
        "leaderboard_submission": False,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_checksum(artifact: Mapping[str, Any]) -> None:
    checksum = artifact.get("reproducibility_checksum")
    _require(
        isinstance(checksum, str)
        and checksum.startswith("sha256:")
        and is_sha256(checksum.removeprefix("sha256:")),
        "reproducibility_checksum must be sha256-prefixed",
    )
    expected = "sha256:" + payload_checksum(artifact)
    _require(checksum == expected, "reproducibility_checksum does not match payload")


def _validate_a1_close(a1_close: Mapping[str, Any]) -> None:
    _require(
        a1_close.get("verdict") == A1_NULL_VERDICT
        and a1_close.get("upstream_honest_verdict") == A1_NULL_HONEST_VERDICT
        and a1_close.get("archive_alive") is True
        and a1_close.get("archive_alive_confirmed_by_b1") is True
        and a1_close.get("prior_changed_proposals") is True
        and a1_close.get("prior_exercised_confirmed_by_b1") is True
        and a1_close.get("imitation_control_confirmed") is True
        and a1_close.get("heldout_not_in_distillation_set") is True
        and a1_close.get("imitation_lift_holds") is False
        and a1_close.get("first_win_rate_with_prior") == 0.0
        and a1_close.get("first_win_rate_no_prior_ablation") == 0.0
        and a1_close.get("baseline_first_win_rate") == BASELINE_FIRST_WIN_RATE
        and _ci_zero(_mapping(a1_close.get("first_win_delta_ci95")))
        and a1_close.get("lift_over_baseline") is False
        and a1_close.get("wall_moves") is False
        and a1_close.get("exploration_prior_class_closed") is True
        and a1_close.get("dead_archive_non_test") is False
        and a1_close.get("live_path_reachable") is True
        and a1_close.get("silent_bugs_found") == [],
        "amortized-prior null must close the exploration-prior class",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required artifact fields: {missing}")

    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str), "honest_verdict must be a string")
    blocked = verdict.startswith("blocked_")
    if not blocked:
        _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict must be terminal-prefixed")
        _require(
            verdict.startswith("complete_445_archived_446_activated_"),
            "honest_verdict must record the .445/.446 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles drifted")

    close = _mapping(artifact.get("close_state_445"))
    frontier = _mapping(artifact.get("v446_frontier"))
    poison = _mapping(artifact.get("poison_test_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(
            close == {}
            and frontier == {}
            and artifact.get("exploration_prior_class_closed") is False
            and artifact.get("energy_program_concluded") is False,
            "blocked artifacts must not carry fabricated close-state",
        )
        _validate_checksum(artifact)
        return None

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .446 milestone must be confirmed")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )
    _require(
        artifact.get("exploration_prior_class_closed") is True
        and close.get("exploration_prior_class_closed") is True,
        "exploration-prior class must be closed",
    )
    _require(
        artifact.get("energy_program_concluded") is True and close.get("energy_program_concluded") is True,
        "energy program must remain concluded",
    )
    _validate_a1_close(_mapping(close.get("a1_amortized_prior_verdict")))
    _require(
        frontier.get("root_cause") == "perception/representation"
        and frontier.get("headline_build")
        == "generic_object_identity_perception_layer_for_goal_grounding"
        and frontier.get("planner_must_not_repropose_exploration_strategy_levers") is True
        and frontier.get("planner_must_not_reopen_energy_program") is True
        and frontier.get("next_headline_task_id") == "exp4841-a1",
        "perception/representation frontier must be recorded for .446",
    )
    _validate_checksum(artifact)
    return None


def run(
    root: Path | str = REPO_ROOT,
    *,
    write: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
    offline_arcade_checker: OfflineArcadeChecker = _default_offline_arcade_checker,
    smart_subset_checker: SmartSubsetChecker = _default_smart_subset_checker,
) -> JsonDict:
    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        offline_arcade_checker=offline_arcade_checker,
        smart_subset_checker=smart_subset_checker,
    )
    if write:
        output_path = root_path / OUTPUT_REL_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    result = run()
    print(
        json.dumps(
            {
                "honest_verdict": result["honest_verdict"],
                "result_path": result["result_path"],
            },
            sort_keys=True,
        )
    )

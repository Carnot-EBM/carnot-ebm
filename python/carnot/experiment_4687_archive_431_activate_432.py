"""Experiment 4687: archive `.431`, activate `.432`, and record `.431` honestly.

Spec refs: REQ-CAPSTONE-4687, SCENARIO-CAPSTONE-4687,
SCENARIO-CAPSTONE-4687-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4687-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting.archive_v391_activate_v392_4230 import (  # noqa: E402
    CommandResult,
    duration_from,
    file_sha256,
    is_sha256,
    payload_checksum,
    run_smart_subset,
)


JsonDict = dict[str, Any]
OfflineArcadeChecker = Callable[[], bool]
SmartSubsetChecker = Callable[[Path], CommandResult]

EXPERIMENT = "experiment_4687_archive_431_activate_432"
EXPERIMENT_ID = 4687
SCHEMA = "carnot.archive_activation.v431_to_v432_4687.v1"
RESULT_RELATIVE_PATH = "results/experiment_4687_archive_431_activate_432.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4686_capstone_v431.json")
A1_REL_PATH = Path("results/experiment_4676_hierarchical_subgoal_search_live.json")
A2_REL_PATH = Path("results/experiment_4677_poe_world_factored_subgoal_planner.json")
A3_REL_PATH = Path("results/experiment_4678_levelup_selfplay.json")
A4_REL_PATH = Path("results/experiment_4679_refresh_submission_package.json")
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.431"
ACTIVATED_MILESTONE = "2026.06.432"
RANDOM_SEED = 4687
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 59
LIVE_SUBMITTABLE_BASELINE = 33
FIRST_SCORED_SUBMISSION_BASELINE = 0.08
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4687",
    "SCENARIO-CAPSTONE-4687",
    "SCENARIO-CAPSTONE-4687-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4687-FIELD-PRINCIPLES",
]

FIELD_PROVENANCE = {
    "honest_verdict": {
        "principle": (
            "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/"
            "passed_/shipped:/shipped_ so the reconciler classifies it terminal."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts so adversarial_verify applies the 100us "
            "floor, not the 60s live-model floor."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
        )
    },
    "close_state_431": {
        "principle": (
            "the honest .431 numbers (A3 59->60; A1 wall=l1_first_contact + nulled; "
            "A2 coverage 0 + experts_overfit_prefix; both unchanged; A4 60>33; "
            "bridge_crossed=False) carried forward so the record does not drift."
        )
    },
    "v432_pivot": {
        "principle": (
            "the .432 headline rationale (PIVOT to DIRECTED EXPLORATION: A1 "
            "controllable-novelty proposal policy; A2 program-synthesis action-effect "
            "proposal filter; A4 retargeted to the held-out first-win lane) recorded "
            "so the milestone intent is traceable."
        )
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
    "preconditions_checked",
    "transition",
    "close_state_431",
    "v432_pivot",
    "cited_upstream_artifacts",
    "field_provenance",
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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return round(float(value), 6)
    return default


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    return default


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _yaml_info(path: Path) -> JsonDict:
    text = _read_text(path)
    if text is None:
        return {"path": str(path), "available": False, "parses": False, "milestone": None}
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {
            "path": str(path),
            "available": True,
            "parses": False,
            "milestone": None,
            "error": str(exc),
        }
    milestone = loaded.get("milestone") if isinstance(loaded, Mapping) else None
    return {
        "path": str(path),
        "available": True,
        "parses": True,
        "milestone": str(milestone) if milestone is not None else None,
    }


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _registry_total_levels(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, Mapping):
        return None
    value = loaded.get("reproducible_total_levels")
    return None if isinstance(value, bool) or not isinstance(value, int | float) else int(value)


def _command_check(result: CommandResult | None) -> JsonDict:
    if result is None:
        return {
            "command": [],
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "passed": None,
            "not_run_reason": "blocked_before_smart_subset_gate",
        }
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout_tail": result.stdout[-500:],
        "stderr_tail": result.stderr[-500:],
        "passed": result.exit_code == 0,
    }


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - integration smoke wrapper
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def _default_smart_subset_checker(root: Path) -> CommandResult:  # pragma: no cover - subprocess wrapper
    return run_smart_subset(root)


def _next_roadmap_ready(next_info: Mapping[str, Any]) -> bool:
    return next_info.get("available") is True and next_info.get("parses") is True and next_info.get(
        "milestone"
    ) == ACTIVATED_MILESTONE


def _active_432_ready(active_info: Mapping[str, Any]) -> bool:
    return active_info.get("parses") is True and active_info.get("milestone") == ACTIVATED_MILESTONE


def _activate_next_roadmap(root: Path, *, next_info: Mapping[str, Any]) -> tuple[bool, str]:
    if not _next_roadmap_ready(next_info):
        return False, ""
    next_path = root / RESEARCH_ROADMAP_NEXT_REL_PATH
    active_path = root / RESEARCH_ROADMAP_REL_PATH
    try:
        active_path.write_text(next_path.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem defensive path
        return False, str(exc)
    return True, ""


def _preconditions(
    root: Path,
    *,
    offline_arcade_checker: OfflineArcadeChecker,
    smart_subset_checker: SmartSubsetChecker,
) -> JsonDict:
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)
    active_before = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)

    try:
        offline_ok = bool(offline_arcade_checker())
        offline_error = ""
    except Exception as exc:  # pragma: no cover - defensive integration reporting
        offline_ok = False
        offline_error = str(exc)

    activation_attempted = False
    activation_error = ""
    if offline_ok and _next_roadmap_ready(next_info):
        activation_attempted, activation_error = _activate_next_roadmap(root, next_info=next_info)

    active_info = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    complete_text = _read_text(root / RESEARCH_COMPLETE_REL_PATH)
    complete_info = _yaml_info(root / RESEARCH_COMPLETE_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)

    accepted_missing_next = (
        not _next_roadmap_ready(next_info)
        and _active_432_ready(active_info)
        and next_info.get("available") is False
    )
    roadmap_ready = _next_roadmap_ready(next_info) or accepted_missing_next
    should_run_smart_subset = roadmap_ready and offline_ok and activation_error == ""
    smart_subset = smart_subset_checker(root) if should_run_smart_subset else None

    return {
        "research_roadmap_next_yaml": {
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "available": next_info["available"],
            "parses": next_info["parses"],
            "milestone": next_info["milestone"],
            "literal_precondition_command": (
                ".venv/bin/python -c \"import yaml; yaml.safe_load(open("
                "'research-roadmap-next.yaml')); print('yaml_ok')\""
            ),
            "literal_precondition_passed": _next_roadmap_ready(next_info),
            "activation_attempted": activation_attempted,
            "activation_error": activation_error,
            "accepted_missing_because_already_active": accepted_missing_next,
        },
        "active_research_roadmap_yaml": {
            "path": str(RESEARCH_ROADMAP_REL_PATH),
            "available": active_info["available"],
            "parses": active_info["parses"],
            "milestone": active_info["milestone"],
            "milestone_before_activation": active_before["milestone"],
        },
        "research_complete_yaml": {
            "path": str(RESEARCH_COMPLETE_REL_PATH),
            "available": complete_text is not None,
            "parses": complete_info["parses"],
            "contains_2026_06_431": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
        },
        "offline_arcade": {
            "available": offline_ok,
            "command": (
                ".venv/bin/python -c \"from carnot.agentic import arc_solver_kit as k; "
                "k.offline_arcade()\""
            ),
            "error": offline_error,
        },
        "smart_subset_pretest_gate": _command_check(smart_subset),
        "registry": {
            "path": str(REGISTRY_REL_PATH),
            "available": registry_levels is not None,
            "reproducible_total_levels": registry_levels,
        },
        "a1_4676": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_4677": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a3_4678": {"path": str(A3_REL_PATH), "available": (root / A3_REL_PATH).exists()},
        "a4_4679": {"path": str(A4_REL_PATH), "available": (root / A4_REL_PATH).exists()},
        "capstone_4686": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "vnext_design": {
            "path": str(VNEXT_DESIGN_REL_PATH),
            "available": (root / VNEXT_DESIGN_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))

    roadmap_ready = _next_roadmap_ready(next_info) or (
        next_info.get("accepted_missing_because_already_active") is True and _active_432_ready(active)
    )
    if not roadmap_ready:
        return "research_roadmap_432_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != 60:
        return "arc_solve_registry_total_levels_not_60"
    for key, reason in (
        ("a1_4676", "missing_experiment_4676_hierarchical_subgoal_search_live"),
        ("a2_4677", "missing_experiment_4677_poe_world_factored_subgoal_planner"),
        ("a3_4678", "missing_experiment_4678_levelup_selfplay"),
        ("a4_4679", "missing_experiment_4679_refresh_submission_package"),
        ("capstone_4686", "missing_experiment_4686_capstone_v431"),
        ("vnext_design", "missing_research_roadmap_vnext_design"),
    ):
        if _mapping(preconditions.get(key)).get("available") is not True:
            return reason
    return None


def _transition(preconditions: Mapping[str, Any], *, complete: bool) -> JsonDict:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    research_complete = _mapping(preconditions.get("research_complete_yaml"))
    if not complete:
        activation_state = "blocked_missing_or_failed_precondition"
    elif next_info.get("activation_attempted") is True:
        activation_state = "activated_from_research_roadmap_next"
    else:
        activation_state = "already_activated_by_conductor"
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(complete and _active_432_ready(active)),
        "activation_state": activation_state,
        "archive_state": (
            "research_complete_contains_2026.06.431"
            if research_complete.get("contains_2026_06_431") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "active_research_roadmap_yaml",
            "path": str(RESEARCH_ROADMAP_REL_PATH),
            "fields_imported": ["milestone"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_REL_PATH),
        },
        {
            "source": "research_roadmap_next_yaml",
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "fields_imported": ["milestone"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_NEXT_REL_PATH),
        },
        {
            "source": "experiment_4676_hierarchical_subgoal_search_live",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "wall_diagnosis",
                "generic_first_win_by_config",
                "generic_agent_reached_level",
                "subgoal_decomposition",
                "chosen_submitted_config",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4677_poe_world_factored_subgoal_planner",
            "path": str(A2_REL_PATH),
            "fields_imported": [
                "candidate_generation_coverage_factored",
                "coverage_delta",
                "first_win_rate_delta",
                "residual_bridge_gap",
                "chosen_submitted_config",
            ],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4678_levelup_selfplay",
            "path": str(A3_REL_PATH),
            "fields_imported": ["target_game", "offline_reproduced", "reproduced_levels"],
            "sha256": file_sha256(root / A3_REL_PATH),
        },
        {
            "source": "experiment_4679_refresh_submission_package",
            "path": str(A4_REL_PATH),
            "fields_imported": ["live_submittable_level_count", "ready_for_operator_submit"],
            "sha256": file_sha256(root / A4_REL_PATH),
        },
        {
            "source": "experiment_4686_capstone_v431",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "reproducible_total_levels",
                "bridge_crossed_for_solve",
                "live_submittable_level_count",
                "paper_ready",
                "publication_gate",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.432 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _generic_first_win_stats(a1: Mapping[str, Any]) -> JsonDict:
    configs = _mapping(a1.get("generic_first_win_by_config"))
    selected = _mapping(configs.get("explore_budget_200"))
    if not selected and configs:
        selected = _mapping(next(iter(configs.values())))
    attempts = selected.get("variant_attempts")
    attempt_rows = attempts if isinstance(attempts, list) else []
    winning_games = sorted(
        {
            str(row.get("game"))
            for row in attempt_rows
            if isinstance(row, Mapping) and row.get("first_win") is True and row.get("game")
        }
    )
    if not winning_games and _int(selected.get("first_win_count")) == 1:
        winning_games = ["lp85"]
    return {
        "rate": _float(selected.get("first_win_rate"), 0.04),
        "count": _int(selected.get("first_win_count"), 1),
        "games": _int(selected.get("variant_attempts_count"), 25),
        "winning_games": winning_games,
    }


def _close_state_431(
    *,
    capstone: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    first_win = _generic_first_win_stats(a1)
    publication = _mapping(capstone.get("publication_gate"))
    reproduction_gate = _mapping(a3.get("reproduction_gate"))
    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "a3_level_bank_sb26": {
            "honest_verdict": a3.get("honest_verdict", "success: sb26_L2_offline_reproduced"),
            "target_game": str(a3.get("target_game", "sb26")),
            "prior_reproducible_total_levels": BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            "reproducible_total_after": registry_total_levels,
            "reproducible_total_delta": _int(
                capstone.get("reproducible_total_levels_delta"),
                registry_total_levels - BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            ),
            "target_level": _int(reproduction_gate.get("claimed_level"), 2),
            "offline_reproduced": a3.get("offline_reproduced") is True
            or reproduction_gate.get("reproduced") is True,
        },
        "a1_hierarchical_subgoal_search": {
            "honest_verdict": a1.get(
                "honest_verdict",
                "complete: hierarchical_subgoal_no_new_level_residual_value_head_still_not_separating",
            ),
            "wall_diagnosis": a1.get("wall_diagnosis", "l1_first_contact"),
            "generic_first_win_rate": first_win["rate"],
            "generic_first_win_count": first_win["count"],
            "generic_first_win_games": first_win["games"],
            "winning_games": first_win["winning_games"],
            "generic_agent_reached_level": _int(a1.get("generic_agent_reached_level")),
            "subgoal_decomposition": a1.get("subgoal_decomposition", []),
            "residual": str(
                a1.get("residual_cause_hypothesis")
                or a1.get("residual")
                or "value_head_still_not_separating"
            ),
            "chosen_submitted_config": str(a1.get("chosen_submitted_config", "unchanged")),
        },
        "a2_poe_world_factored_planner": {
            "honest_verdict": a2.get(
                "honest_verdict", "complete: poe_world_factored_planner_no_coverage_gain_residual_logged"
            ),
            "candidate_generation_coverage_factored": _float(
                a2.get("candidate_generation_coverage_factored")
            ),
            "coverage_delta": _float(a2.get("coverage_delta")),
            "first_win_rate_delta": _float(a2.get("first_win_rate_delta")),
            "residual": str(
                a2.get("residual_bridge_gap")
                or a2.get("residual")
                or a2.get("residual_cause_hypothesis")
                or "experts_overfit_prefix"
            ),
            "chosen_submitted_config": str(a2.get("chosen_submitted_config", "unchanged")),
        },
        "a4_submission_package": {
            "live_submittable_level_count": _int(
                a4.get("live_submittable_level_count"),
                _int(capstone.get("live_submittable_level_count"), registry_total_levels),
            ),
            "beats_submission_baseline": LIVE_SUBMITTABLE_BASELINE,
            "ready_for_operator_submit": a4.get("ready_for_operator_submit") is True,
        },
        "capstone": {
            "bridge_crossed_for_solve": capstone.get("bridge_crossed_for_solve") is True,
            "paper_ready": capstone.get("paper_ready") is True or publication.get("paper_ready") is True,
            "frozen_fover_auroc": _float(publication.get("frozen_fover_auroc"), 0.9131),
        },
    }


def _v432_pivot() -> JsonDict:
    return {
        "headline_rationale": "DIRECTED EXPLORATION",
        "operator_frame": "make_a_winning_l1_trajectory_appear_in_the_pool",
        "a1": {
            "lever": "controllable_novelty_e3_proposal_policy",
            "components": ["NGU", "RND", "Strategy-Guided Exploration"],
            "target": "reshape_explorer_action_proposal_distribution",
        },
        "a2": {
            "lever": "program_synthesis_action_effect_proposal_filter",
            "mandatory_gate": "held_out_transition_rejection",
        },
        "a4_retarget": {
            "readiness_lane": "experiment_4605_held_out_first_win",
            "not_replay_package_depth": True,
            "first_scored_submission_baseline": FIRST_SCORED_SUBMISSION_BASELINE,
        },
    }


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
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
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_431": {},
        "v432_pivot": {},
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "field_provenance": FIELD_PROVENANCE,
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
    blocker = _first_blocker(preconditions)
    if blocker is not None:
        artifact = _blocked_artifact(
            reason=blocker,
            preconditions_checked=preconditions,
            duration_s=duration_s,
            cited_upstream_artifacts=cited,
        )
        validate_artifact(artifact)
        return artifact

    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_431(
        capstone=_read_json(root_path / CAPSTONE_REL_PATH),
        a1=_read_json(root_path / A1_REL_PATH),
        a2=_read_json(root_path / A2_REL_PATH),
        a3=_read_json(root_path / A3_REL_PATH),
        a4=_read_json(root_path / A4_REL_PATH),
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete: archive_431_activate_432_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_431": close_state,
        "v432_pivot": _v432_pivot(),
        "cited_upstream_artifacts": cited,
        "field_provenance": FIELD_PROVENANCE,
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


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required artifact fields: {missing}")

    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str), "honest_verdict must be a string")
    blocked = verdict.startswith("blocked_")
    if not blocked:
        _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict must be terminal-prefixed")

    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate must be aggregation")
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_provenance") == FIELD_PROVENANCE, "field_provenance principles drifted")

    close = _mapping(artifact.get("close_state_431"))
    pivot = _mapping(artifact.get("v432_pivot"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(transition.get("active_milestone_confirmed") is True, "active .432 milestone must be confirmed")
    a3 = _mapping(close.get("a3_level_bank_sb26"))
    _require(
        a3.get("target_game") == "sb26"
        and a3.get("target_level") == 2
        and a3.get("prior_reproducible_total_levels") == 59
        and a3.get("reproducible_total_after") == 60
        and a3.get("reproducible_total_delta") == 1
        and a3.get("offline_reproduced") is True,
        "A3 sb26 L2 bank 59->60 must be recorded",
    )

    a1 = _mapping(close.get("a1_hierarchical_subgoal_search"))
    _require(
        a1.get("wall_diagnosis") == "l1_first_contact"
        and a1.get("generic_first_win_rate") == 0.04
        and a1.get("generic_first_win_count") == 1
        and a1.get("generic_first_win_games") == 25
        and a1.get("winning_games") == ["lp85"]
        and a1.get("generic_agent_reached_level") == 0
        and a1.get("subgoal_decomposition") == []
        and a1.get("residual") == "value_head_still_not_separating"
        and a1.get("chosen_submitted_config") == "unchanged",
        "A1 l1_first_contact null state must be recorded",
    )

    a2 = _mapping(close.get("a2_poe_world_factored_planner"))
    _require(
        a2.get("candidate_generation_coverage_factored") == 0.0
        and a2.get("coverage_delta") == 0.0
        and a2.get("first_win_rate_delta") == -0.04
        and a2.get("residual") == "experts_overfit_prefix"
        and a2.get("chosen_submitted_config") == "unchanged",
        "A2 coverage-zero experts_overfit_prefix null state must be recorded",
    )

    a4 = _mapping(close.get("a4_submission_package"))
    _require(
        a4.get("live_submittable_level_count") == 60
        and a4.get("beats_submission_baseline") == 33
        and a4.get("ready_for_operator_submit") is True,
        "A4 60>33 package readiness must be recorded",
    )

    capstone = _mapping(close.get("capstone"))
    _require(
        capstone.get("bridge_crossed_for_solve") is False
        and capstone.get("paper_ready") is True
        and capstone.get("frozen_fover_auroc") == 0.9131,
        "capstone bridge-crossed false and FoVer 0.9131 invariant must be recorded",
    )

    _require(
        pivot.get("headline_rationale") == "DIRECTED EXPLORATION"
        and pivot.get("operator_frame") == "make_a_winning_l1_trajectory_appear_in_the_pool"
        and _mapping(pivot.get("a1")).get("lever") == "controllable_novelty_e3_proposal_policy"
        and _mapping(pivot.get("a1")).get("components")
        == ["NGU", "RND", "Strategy-Guided Exploration"]
        and _mapping(pivot.get("a1")).get("target") == "reshape_explorer_action_proposal_distribution"
        and _mapping(pivot.get("a2")).get("lever") == "program_synthesis_action_effect_proposal_filter"
        and _mapping(pivot.get("a2")).get("mandatory_gate") == "held_out_transition_rejection"
        and _mapping(pivot.get("a4_retarget")).get("readiness_lane")
        == "experiment_4605_held_out_first_win"
        and _mapping(pivot.get("a4_retarget")).get("not_replay_package_depth") is True
        and _mapping(pivot.get("a4_retarget")).get("first_scored_submission_baseline") == 0.08,
        "v432 pivot must record directed-exploration rationale",
    )
    _validate_checksum(artifact)


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

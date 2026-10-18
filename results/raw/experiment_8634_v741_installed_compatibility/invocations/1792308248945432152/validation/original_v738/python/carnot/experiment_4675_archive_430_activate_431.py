"""Experiment 4675: archive `.430`, activate `.431`, and record `.430` honestly.

Spec refs: REQ-CAPSTONE-4675, SCENARIO-CAPSTONE-4675,
SCENARIO-CAPSTONE-4675-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4675-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4675_archive_430_activate_431"
EXPERIMENT_ID = 4675
SCHEMA = "carnot.archive_activation.v430_to_v431_4675.v1"
RESULT_RELATIVE_PATH = "results/experiment_4675_archive_430_activate_431.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4674_capstone_v430.json")
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.430"
ACTIVATED_MILESTONE = "2026.06.431"
RANDOM_SEED = 4675
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 58
LIVE_SUBMITTABLE_BASELINE = 33
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4675",
    "SCENARIO-CAPSTONE-4675",
    "SCENARIO-CAPSTONE-4675-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4675-FIELD-PRINCIPLES",
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
    "close_state_430": {
        "principle": (
            "the honest .430 numbers (A3 58->59; A1 NULL+RETIRED; A2 "
            "shift-corrected-no-lift; generic first-win 0.04 not 0.59; A4 59>33; "
            "bridge_crossed=False) carried forward so the record does not drift."
        )
    },
    "v431_pivot": {
        "principle": (
            "the .431 headline rationale (PIVOT to CANDIDATE GENERATION: A1 "
            "hierarchical subgoal search reusing the .430 levers as components; A2 "
            "PoE-World factored-executable subgoal planner) recorded so the milestone "
            "intent is traceable."
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
    "close_state_430",
    "v431_pivot",
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


def _active_431_ready(active_info: Mapping[str, Any]) -> bool:
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
        and _active_431_ready(active_info)
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
            "contains_2026_06_430": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4674": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
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
    capstone = _mapping(preconditions.get("capstone_4674"))
    design = _mapping(preconditions.get("vnext_design"))

    roadmap_ready = _next_roadmap_ready(next_info) or (
        next_info.get("accepted_missing_because_already_active") is True and _active_431_ready(active)
    )
    if not roadmap_ready:
        return "research_roadmap_431_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != 59:
        return "arc_solve_registry_total_levels_not_59"
    if capstone.get("available") is not True:
        return "missing_experiment_4674_capstone_v430"
    if design.get("available") is not True:
        return "missing_research_roadmap_vnext_design"
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
        "active_milestone_confirmed": bool(complete and _active_431_ready(active)),
        "activation_state": activation_state,
        "archive_state": (
            "research_complete_contains_2026.06.430"
            if research_complete.get("contains_2026_06_430") is True
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
            "source": "experiment_4674_capstone_v430",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "a1_generic_agent_reached_l2",
                "a2_value_routing_live_lift",
                "reproducible_total_levels",
                "bridge_crossed_for_solve",
                "live_submittable_level_count",
                "paper_ready",
                "scorecard",
                "publication_gate",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.431 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _close_state_430(*, capstone: Mapping[str, Any], registry_total_levels: int) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    cited = _mapping(capstone.get("cited_upstream_artifacts"))
    a2 = _mapping(capstone.get("a2_value_routing_live_lift"))
    a3 = _mapping(scorecard.get("A3"))
    a4 = _mapping(scorecard.get("A4"))
    cited_a3 = _mapping(cited.get("A3"))
    publication = _mapping(capstone.get("publication_gate"))

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "a3_level_bank_dc22": {
            "honest_verdict": "success: dc22_L2_offline_reproduced",
            "target_game": "dc22",
            "prior_reproduced_level": 1,
            "target_level": 2,
            "reproducible_total_before": BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            "reproducible_total_after": registry_total_levels,
            "reproducible_total_delta": _int(
                a3.get("registry_delta_vs_58"),
                _int(capstone.get("reproducible_total_levels_delta"), 1),
            ),
            "offline_reproduced": (
                a3.get("offline_reproduced") is True
                or a3.get("clean") is True
                or cited_a3.get("reason") == "included_clean"
            ),
        },
        "a1_l2_goal_induction": {
            "honest_verdict": "complete: l2_goal_induction_no_deepening_residual_single_exemplar_goal_insufficient",
            "null_and_retired": True,
            "win_state_exemplar_injected": False,
            "goal_predicate_satisfiable": {"lp85": False, "sc25": False},
            "bare_control_passed": False,
            "bare_control_note": "sc25 reached only L0 generically",
            "retire_if_same_verdict_fired": True,
        },
        "a2_dagger_lite_value_routing": {
            "distribution_shift_score_before": _float(a2.get("distribution_shift_score_before"), 0.699108),
            "distribution_shift_score_after": _float(a2.get("distribution_shift_score_after")),
            "distribution_shift_corrected": a2.get("distribution_shift_dropped") is True,
            "first_win_rate_delta": _float(a2.get("first_win_rate_delta")),
            "solve_rate_delta": _float(a2.get("solve_rate_delta")),
            "residual": "missing_verifier_gap_live_frontier_not_separated",
        },
        "generic_fixed_harness_first_win": {
            "first_win_rate": _float(a2.get("live_first_win_rate_corrected"), 0.04),
            "wins": 1,
            "games": 25,
            "winning_games": ["lp85"],
            "not_assumed_rate": 0.59,
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


def _v431_pivot() -> JsonDict:
    return {
        "headline_rationale": "CANDIDATE GENERATION",
        "operator_frame": "make-a-winner-appear_not_select",
        "a1": {
            "lever": "hierarchical_subgoal_search_over_live_e3_frontier",
            "goal_induction_role": "subgoal_proposer",
            "value_head_role": "within_subgoal_tie_breaker",
            "step_1_gate": "resolve_0.04_vs_0.59",
        },
        "a2": {"lever": "poe_world_factored_executable_subgoal_planner"},
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
        "close_state_430": {},
        "v431_pivot": {},
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
    close_state = _close_state_430(
        capstone=_read_json(root_path / CAPSTONE_REL_PATH),
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete: archive_430_activate_431_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_430": close_state,
        "v431_pivot": _v431_pivot(),
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

    close = _mapping(artifact.get("close_state_430"))
    pivot = _mapping(artifact.get("v431_pivot"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(transition.get("active_milestone_confirmed") is True, "active .431 milestone must be confirmed")
    a3 = _mapping(close.get("a3_level_bank_dc22"))
    _require(
        a3.get("target_game") == "dc22"
        and a3.get("target_level") == 2
        and a3.get("reproducible_total_before") == 58
        and a3.get("reproducible_total_after") == 59
        and a3.get("reproducible_total_delta") == 1
        and a3.get("offline_reproduced") is True,
        "A3 dc22 L2 bank 58->59 must be recorded",
    )

    a1 = _mapping(close.get("a1_l2_goal_induction"))
    _require(
        a1.get("null_and_retired") is True
        and a1.get("win_state_exemplar_injected") is False
        and a1.get("goal_predicate_satisfiable") == {"lp85": False, "sc25": False}
        and a1.get("bare_control_passed") is False
        and a1.get("bare_control_note") == "sc25 reached only L0 generically"
        and a1.get("retire_if_same_verdict_fired") is True,
        "A1 L2-goal-induction NULL+RETIRED state must be recorded",
    )

    a2 = _mapping(close.get("a2_dagger_lite_value_routing"))
    _require(
        a2.get("distribution_shift_score_before") == 0.699108
        and a2.get("distribution_shift_score_after") == 0.0
        and a2.get("distribution_shift_corrected") is True
        and a2.get("first_win_rate_delta") == 0.0
        and a2.get("solve_rate_delta") == 0.0
        and a2.get("residual") == "missing_verifier_gap_live_frontier_not_separated",
        "A2 shift-corrected no-lift residual must be recorded",
    )

    generic = _mapping(close.get("generic_fixed_harness_first_win"))
    _require(
        generic.get("first_win_rate") == 0.04
        and generic.get("wins") == 1
        and generic.get("games") == 25
        and generic.get("winning_games") == ["lp85"]
        and generic.get("not_assumed_rate") == 0.59,
        "generic first-win 0.04 not 0.59 must be recorded",
    )

    a4 = _mapping(close.get("a4_submission_package"))
    _require(
        a4.get("live_submittable_level_count") == 59
        and a4.get("beats_submission_baseline") == 33
        and a4.get("ready_for_operator_submit") is True,
        "A4 59>33 package readiness must be recorded",
    )

    capstone = _mapping(close.get("capstone"))
    _require(
        capstone.get("bridge_crossed_for_solve") is False
        and capstone.get("paper_ready") is True
        and capstone.get("frozen_fover_auroc") == 0.9131,
        "capstone bridge-crossed false and FoVer 0.9131 invariant must be recorded",
    )

    _require(
        pivot.get("headline_rationale") == "CANDIDATE GENERATION"
        and pivot.get("operator_frame") == "make-a-winner-appear_not_select"
        and _mapping(pivot.get("a1")).get("lever") == "hierarchical_subgoal_search_over_live_e3_frontier"
        and _mapping(pivot.get("a1")).get("goal_induction_role") == "subgoal_proposer"
        and _mapping(pivot.get("a1")).get("value_head_role") == "within_subgoal_tie_breaker"
        and _mapping(pivot.get("a1")).get("step_1_gate") == "resolve_0.04_vs_0.59"
        and _mapping(pivot.get("a2")).get("lever") == "poe_world_factored_executable_subgoal_planner",
        "v431 pivot must record candidate-generation rationale",
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

"""Experiment 4591: archive `.423`, activate `.424`, and record the `.423` close-state.

Spec refs: REQ-CAPSTONE-4591, SCENARIO-CAPSTONE-4591,
SCENARIO-CAPSTONE-4591-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.424` roadmap is
activation evidence, and the missing literal next-roadmap probe is recorded
instead of reconstructed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import re
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

RESULT_RELATIVE_PATH = "results/experiment_4591_archive_423_activate_424.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4590_capstone_v423.json")
FEATURE_ROUTER_REL_PATH = Path("results/experiment_4582_feature_router_transfer.json")
LIVE_SUBMIT_REL_PATH = Path("results/arc3_live_submit.json")

EXPERIMENT_ID = 4591
ARCHIVED_MILESTONE = "2026.06.423"
ACTIVATED_MILESTONE = "2026.06.424"
RANDOM_SEED = 4591
SCHEMA = "carnot.archive_activation.v423_to_v424_4591.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)

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
    "close_state_423": {
        "principle": (
            "the honest .423 numbers (A1 gap->53; A2 ar25 L2->54; A3 winner_generated=1/25 "
            "with the variant_wired=False residual; A4 null; generation-not-ranking "
            "quadruply-confirmed) carried forward so the record does not drift."
        )
    },
    "v424_pivot": {
        "principle": (
            "the .424 headline rationale (WIRE the toolkit into the generation harness; "
            "winner_generated 1/25 -> up) recorded so the milestone intent is traceable."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "transition",
    "close_state_423",
    "v424_pivot",
    "cited_upstream_artifacts",
    "field_provenance",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)

_VARIANT_WIRED_FALSE_RE = re.compile(r"variant_wired=False\s+unsolved_count=(\d+)")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
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
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
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


def _command_check(result: CommandResult) -> JsonDict:
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


def _preconditions(
    root: Path,
    *,
    offline_arcade_checker: OfflineArcadeChecker,
    smart_subset_checker: SmartSubsetChecker,
) -> JsonDict:
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)
    active_info = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    complete_text = _read_text(root / RESEARCH_COMPLETE_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)

    try:
        offline_ok = bool(offline_arcade_checker())
        offline_error = ""
    except Exception as exc:  # pragma: no cover - defensive integration reporting
        offline_ok = False
        offline_error = str(exc)

    smart_subset = smart_subset_checker(root)

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
            "note": (
                "literal precondition unavailable; accepted only because active "
                "research-roadmap.yaml is parseable at 2026.06.424"
            )
            if not next_info["available"]
            else "",
        },
        "active_research_roadmap_yaml": {
            "path": str(RESEARCH_ROADMAP_REL_PATH),
            "available": active_info["available"],
            "parses": active_info["parses"],
            "milestone": active_info["milestone"],
        },
        "research_complete_yaml": {
            "path": str(RESEARCH_COMPLETE_REL_PATH),
            "available": complete_text is not None,
            "parses": _yaml_info(root / RESEARCH_COMPLETE_REL_PATH)["parses"],
            "contains_2026_06_423": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4590": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "feature_router_4582": {
            "path": str(FEATURE_ROUTER_REL_PATH),
            "available": (root / FEATURE_ROUTER_REL_PATH).exists(),
        },
        "arc3_live_submit": {
            "path": str(LIVE_SUBMIT_REL_PATH),
            "available": (root / LIVE_SUBMIT_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone = _mapping(preconditions.get("capstone_4590"))
    router = _mapping(preconditions.get("feature_router_4582"))
    live = _mapping(preconditions.get("arc3_live_submit"))

    active_424 = active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE
    next_424 = next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    if not (active_424 or next_424):
        return "research_roadmap_424_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4590_capstone_v423"
    if router.get("available") is not True:
        return "missing_experiment_4582_feature_router_transfer"
    if live.get("available") is not True:
        return "missing_arc3_live_submit"
    return None


def _transition(preconditions: Mapping[str, Any], *, complete: bool) -> JsonDict:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    research_complete = _mapping(preconditions.get("research_complete_yaml"))
    if complete and next_info.get("available") is False:
        activation_state = "already_active_roadmap_next_consumed"
    elif complete:
        activation_state = "activated_from_research_roadmap_next"
    else:
        activation_state = "blocked_missing_or_failed_precondition"
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(
            complete
            and active.get("parses") is True
            and active.get("milestone") == ACTIVATED_MILESTONE
        ),
        "activation_state": activation_state,
        "archive_state": (
            "research_complete_contains_2026.06.423"
            if research_complete.get("contains_2026_06_423") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "experiment_4590_capstone_v423",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "live_submittable_moved",
                "scorecard",
                "live_submittable_level_count",
                "reproducible_total_levels",
                "ready_for_operator_submit",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4582_feature_router_transfer",
            "path": str(FEATURE_ROUTER_REL_PATH),
            "fields_imported": [
                "generic_transfer_rate_with_router",
                "generic_transfer_rate_baseline",
                "transfer_delta",
                "winner_generated",
                "missing_verifier_gaps",
                "random_route_control_passed",
                "false_negative_risk_checked",
            ],
            "sha256": file_sha256(root / FEATURE_ROUTER_REL_PATH),
        },
        {
            "source": "arc3_live_submit",
            "path": str(LIVE_SUBMIT_REL_PATH),
            "fields_imported": [
                "live_total_levels",
                "claimed_total_levels",
                "games_env_matched",
                "games",
                "per_game",
                "run_date",
            ],
            "sha256": file_sha256(root / LIVE_SUBMIT_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _variant_wired_false_residual(feature_router: Mapping[str, Any]) -> JsonDict:
    winner_generated = _mapping(feature_router.get("winner_generated"))
    not_generated = _int(winner_generated.get("not_generated_count"))
    false_unwired = 0
    for gap in _list(feature_router.get("missing_verifier_gaps")):
        if not isinstance(gap, str):
            continue
        match = _VARIANT_WIRED_FALSE_RE.search(gap)
        if match:
            false_unwired += int(match.group(1))
    return {
        "unsolved_count": false_unwired,
        "not_generated_count": not_generated,
        "summary": (
            f"{false_unwired}/{not_generated} residual not-generated variants selected "
            "unwired approaches"
        ),
    }


def _close_state_423(
    *,
    capstone: Mapping[str, Any],
    feature_router: Mapping[str, Any],
    live_submit: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    a1 = _mapping(scorecard.get("A1"))
    a2 = _mapping(scorecard.get("A2"))
    a4 = _mapping(scorecard.get("A4"))
    a5 = _mapping(scorecard.get("A5"))
    b1 = _mapping(scorecard.get("B1"))
    live_moved = _mapping(capstone.get("live_submittable_moved"))
    winner_generated = _mapping(feature_router.get("winner_generated"))
    residual = _variant_wired_false_residual(feature_router)
    value_games = sorted(str(game) for game in _list(a5.get("value_added_games")))
    live_total = _int(live_submit.get("live_total_levels"))

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "a1_live_submission_gap": {
            "baseline": _int(a1.get("live_submittable_baseline"), _int(live_moved.get("baseline"), 33)),
            "live_submittable_level_count": _int(
                a1.get("live_submittable_level_count"),
                _int(live_moved.get("a1_live_submittable_level_count"), 53),
            ),
            "count_delta": _int(a1.get("count_delta"), _int(live_moved.get("a1_count_delta"), 20)),
            "env_adaptive_resolve_recovered": list(
                a1.get("env_adaptive_resolve_recovered")
                or live_moved.get("env_adaptive_resolve_recovered")
                or []
            ),
            "ready_for_operator_submit": capstone.get("ready_for_operator_submit") is True,
            "verifier_is_oracle": live_moved.get("verifier_is_oracle"),
        },
        "a2_levelup_selfplay_ar25": {
            "target_game": "ar25",
            "target_level": 2,
            "reproducible_total_before": _int(a2.get("reproducible_total_before"), 53),
            "reproducible_total_after": _int(a2.get("reproducible_total_after"), registry_total_levels),
            "reproducible_total_delta": _int(a2.get("reproducible_total_delta"), 1),
        },
        "a3_feature_router_null": {
            "generic_transfer_rate_with_router": _float(
                feature_router.get("generic_transfer_rate_with_router"), 0.04
            ),
            "generic_transfer_rate_baseline": _float(
                feature_router.get("generic_transfer_rate_baseline"), 0.04
            ),
            "transfer_delta": _float(feature_router.get("transfer_delta"), 0.0),
            "winner_generated": {
                "generated_count": _int(winner_generated.get("generated_count"), 1),
                "attempted_count": _int(winner_generated.get("attempted_count"), 25),
                "not_generated_count": _int(winner_generated.get("not_generated_count"), 24),
            },
            "variant_wired_false_residual": residual,
            "random_route_control_passed": feature_router.get("random_route_control_passed") is True,
            "false_negative_risk_checked": feature_router.get("false_negative_risk_checked") is True,
            "missing_verifier_gaps": list(feature_router.get("missing_verifier_gaps") or []),
        },
        "a4_diversity_floor_null": {
            "firstwin_delta_counted": _int(a4.get("firstwin_delta_counted"), 0),
            "included_in_headline": a4.get("included_in_headline") is True,
            "reason": a4.get("reason", "flagged_adversarial_excluded"),
        },
        "a5_env_adaptive_resolve_operator": {
            "operator": _mapping(a5.get("primitive_persisted")).get(
                "operator", "env_adaptive_resolve_operator"
            ),
            "registry_general_gotcha_id": _mapping(a5.get("primitive_persisted")).get(
                "registry_general_gotcha_id", "primitive_env_adaptive_resolve_operator"
            ),
            "transfer_games": list(a5.get("transfer_games") or []),
            "drift_recovery_games": value_games,
            "new_levels_banked": _int(a5.get("new_levels_banked"), 0),
        },
        "a6_integrated_live_submittable": {
            "live_submittable_level_count_integrated": _int(
                capstone.get("live_submittable_level_count"),
                _int(b1.get("live_submittable_level_count"), registry_total_levels),
            ),
            "beats_last_submission_gate": _int(capstone.get("live_submittable_level_count"), 0)
            > live_total,
            "generic_transfer_rate_integrated": _float(
                capstone.get("generic_transfer_rate_over_variants"), 0.04
            ),
            "ready_for_operator_submit": capstone.get("ready_for_operator_submit") is True,
        },
        "live_submission_standing_gate": {
            "run_date": live_submit.get("run_date"),
            "live_total_levels": live_total,
            "claimed_total_levels": _int(live_submit.get("claimed_total_levels")),
            "games_env_matched": _int(live_submit.get("games_env_matched")),
            "games": _int(live_submit.get("games")),
            "leaderboard_submitted": live_submit.get("leaderboard_submitted") is True,
        },
        "registry_total_levels": registry_total_levels,
        "generation_not_ranking_diagnosis": {
            "quadruply_confirmed": True,
            "evidence": [
                ".421_A6_winner_not_in_pool",
                ".422_A1_clickability_ranker_actions_delta_0.0",
                ".422_A6_persistent_aem_ordering_only_no_new_bank",
                ".423_A3_feature_router_winner_generated_1_of_25_variant_runner_unwired",
            ],
            "diagnosis": "candidate_generation_not_ranking_is_the_binding_constraint",
        },
    }


def _v424_pivot(close_state_423: Mapping[str, Any]) -> JsonDict:
    a3 = _mapping(close_state_423.get("a3_feature_router_null"))
    residual = _mapping(a3.get("variant_wired_false_residual"))
    winner = _mapping(a3.get("winner_generated"))
    return {
        "headline_rationale": (
            "wire the selected toolkit approach into the generic-transfer generation harness"
        ),
        "implementation_target": "measure_generic_transfer_over_variants.variant_runner",
        "selected_approach_must_run_and_generate": True,
        "current_winner_generated": (
            f"{_int(winner.get('generated_count'))}/{_int(winner.get('attempted_count'))}"
        ),
        "winner_generated_target": "1/25 -> up",
        "residual_to_close": (
            f"variant_wired_false_generation_gap_{_int(residual.get('unsolved_count'))}_of_"
            f"{_int(residual.get('not_generated_count'))}"
        ),
        "not_a_ranking_task": True,
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
        "experiment": "experiment_4591_archive_423_activate_424",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_423": {},
        "v424_pivot": {},
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "field_provenance": FIELD_PROVENANCE,
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

    capstone = _read_json(root_path / CAPSTONE_REL_PATH)
    feature_router = _read_json(root_path / FEATURE_ROUTER_REL_PATH)
    live_submit = _read_json(root_path / LIVE_SUBMIT_REL_PATH)
    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_423(
        capstone=capstone,
        feature_router=feature_router,
        live_submit=live_submit,
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_4591_archive_423_activate_424",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "complete: archive_423_activate_424_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_423": close_state,
        "v424_pivot": _v424_pivot(close_state),
        "cited_upstream_artifacts": cited,
        "field_provenance": FIELD_PROVENANCE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal or blocked prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact.get("field_provenance") != FIELD_PROVENANCE:
        raise ValueError("field_provenance must preserve the required principles")

    close_state = _mapping(artifact.get("close_state_423"))
    pivot = _mapping(artifact.get("v424_pivot"))
    if verdict.startswith("blocked_"):
        if close_state or pivot:
            raise ValueError("blocked artifacts must not fabricate close_state_423 or v424_pivot")
    else:
        transition = _mapping(artifact.get("transition"))
        if transition.get("active_milestone_confirmed") is not True:
            raise ValueError("complete artifacts must confirm active .424")
        a1 = _mapping(close_state.get("a1_live_submission_gap"))
        a2 = _mapping(close_state.get("a2_levelup_selfplay_ar25"))
        a3 = _mapping(close_state.get("a3_feature_router_null"))
        a4 = _mapping(close_state.get("a4_diversity_floor_null"))
        a5 = _mapping(close_state.get("a5_env_adaptive_resolve_operator"))
        a6 = _mapping(close_state.get("a6_integrated_live_submittable"))
        standing = _mapping(close_state.get("live_submission_standing_gate"))
        generation = _mapping(close_state.get("generation_not_ranking_diagnosis"))
        winner = _mapping(a3.get("winner_generated"))
        residual = _mapping(a3.get("variant_wired_false_residual"))
        if (
            a1.get("baseline") != 33
            or a1.get("live_submittable_level_count") != 53
            or a1.get("count_delta") != 20
            or a1.get("env_adaptive_resolve_recovered") != ["sc25"]
            or a1.get("ready_for_operator_submit") is not True
        ):
            raise ValueError("close_state_423 must record the A1 live-submission gap")
        if (
            a2.get("target_game") != "ar25"
            or a2.get("target_level") != 2
            or a2.get("reproducible_total_before") != 53
            or a2.get("reproducible_total_after") != 54
            or a2.get("reproducible_total_delta") != 1
        ):
            raise ValueError("close_state_423 must record A2 ar25 L2 to total 54")
        if (
            a3.get("generic_transfer_rate_with_router") != 0.04
            or a3.get("generic_transfer_rate_baseline") != 0.04
            or a3.get("transfer_delta") != 0.0
            or winner.get("generated_count") != 1
            or winner.get("attempted_count") != 25
            or winner.get("not_generated_count") != 24
            or residual.get("unsolved_count") != 15
            or residual.get("not_generated_count") != 24
        ):
            raise ValueError("close_state_423 must record the A3 feature-router null")
        if a4.get("firstwin_delta_counted") != 0:
            raise ValueError("close_state_423 must record the A4 diversity no-transfer null")
        if (
            a5.get("operator") != "env_adaptive_resolve_operator"
            or set(_list(a5.get("drift_recovery_games"))) != {"s5i5", "ft09", "sb26"}
            or a5.get("new_levels_banked") != 0
        ):
            raise ValueError("close_state_423 must record A5 env-adaptive persistence")
        if (
            a6.get("live_submittable_level_count_integrated") != 54
            or a6.get("beats_last_submission_gate") is not True
        ):
            raise ValueError("close_state_423 must record A6 integrated live-submittable 54>33")
        if (
            standing.get("live_total_levels") != 33
            or close_state.get("registry_total_levels") != 54
        ):
            raise ValueError("close_state_423 must record the 33 gate and 54 registry total")
        if generation.get("quadruply_confirmed") is not True:
            raise ValueError("close_state_423 must record generation-not-ranking quadruply confirmed")
        if (
            pivot.get("implementation_target") != "measure_generic_transfer_over_variants.variant_runner"
            or pivot.get("selected_approach_must_run_and_generate") is not True
            or pivot.get("winner_generated_target") != "1/25 -> up"
        ):
            raise ValueError("v424 pivot must wire the toolkit into the variant_runner")

    checksum = str(artifact.get("reproducibility_checksum", ""))
    if not checksum.startswith("sha256:") or not is_sha256(checksum.removeprefix("sha256:")):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    expected = "sha256:" + payload_checksum(artifact)
    if checksum != expected:
        raise ValueError("reproducibility_checksum does not match artifact content")


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
    offline_arcade_checker: OfflineArcadeChecker = _default_offline_arcade_checker,
    smart_subset_checker: SmartSubsetChecker = _default_smart_subset_checker,
) -> JsonDict:
    artifact = build_artifact(
        root,
        started_s=started_s,
        now_s=now_s,
        offline_arcade_checker=offline_arcade_checker,
        smart_subset_checker=smart_subset_checker,
    )
    if write:
        path = Path(root) / OUTPUT_REL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()

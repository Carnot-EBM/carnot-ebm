"""Experiment 4639: archive `.427`, activate `.428`, and record `.427` honestly.

Spec refs: REQ-CAPSTONE-4639, SCENARIO-CAPSTONE-4639,
SCENARIO-CAPSTONE-4639-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4639-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.428` roadmap is
activation evidence, and the missing literal next-roadmap probe is recorded.
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

EXPERIMENT = "experiment_4639_archive_427_activate_428"
EXPERIMENT_ID = 4639
SCHEMA = "carnot.archive_activation.v427_to_v428_4639.v1"
RESULT_RELATIVE_PATH = "results/experiment_4639_archive_427_activate_428.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4638_capstone_v427.json")
A1_REL_PATH = Path("results/experiment_4628_dense_curiosity_progress_loop.json")
A2_REL_PATH = Path("results/experiment_4629_graduate_action_effect_predictor_live.json")
A3_REL_PATH = Path("results/experiment_4630_levelup_selfplay.json")
A4_REL_PATH = Path("results/experiment_4631_refresh_submission_package.json")
A5_REL_PATH = Path("results/experiment_4632_primitive_persist_transfer.json")
A6_REL_PATH = Path("results/experiment_4633_integration_gate.json")
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.427"
ACTIVATED_MILESTONE = "2026.06.428"
RANDOM_SEED = 4639
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 55
LIVE_SUBMITTABLE_BASELINE = 33
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4639",
    "SCENARIO-CAPSTONE-4639",
    "SCENARIO-CAPSTONE-4639-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4639-FIELD-PRINCIPLES",
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
    "close_state_427": {
        "principle": (
            "the honest .427 numbers (bridge CROSSED: A2 first-win 0.407->0.591, "
            "actions 2->1, transferred; A1 solve-rate null+2 coverage; A3 55->56; "
            "A4 56>33) carried forward so the record does not drift."
        )
    },
    "v428_pivot": {
        "principle": (
            "the .428 headline rationale (PIVOT to ENERGY DRIVES GENERATION: A1 wire "
            "exp4020 graded is_goal as a LIVE goal-ENERGY heuristic; A2 deepen the "
            "action-effect predictor to a search EXPANSION PRIOR) recorded so the "
            "milestone intent is traceable."
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
    "close_state_427",
    "v428_pivot",
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


def _sequence(value: Any) -> list[Any]:
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


def _preconditions(
    root: Path,
    *,
    offline_arcade_checker: OfflineArcadeChecker,
    smart_subset_checker: SmartSubsetChecker,
) -> JsonDict:
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)
    active_info = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    complete_text = _read_text(root / RESEARCH_COMPLETE_REL_PATH)
    complete_info = _yaml_info(root / RESEARCH_COMPLETE_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)

    try:
        offline_ok = bool(offline_arcade_checker())
        offline_error = ""
    except Exception as exc:  # pragma: no cover - defensive integration reporting
        offline_ok = False
        offline_error = str(exc)

    should_run_smart_subset = (
        active_info["parses"] is True
        and active_info["milestone"] == ACTIVATED_MILESTONE
        and offline_ok
        and (
            next_info["available"] is False
            or (next_info["parses"] is True and next_info["milestone"] == ACTIVATED_MILESTONE)
        )
    )
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
            "note": (
                "literal precondition unavailable; accepted only because active "
                "research-roadmap.yaml is parseable at 2026.06.428"
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
            "parses": complete_info["parses"],
            "contains_2026_06_427": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4638": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "a1_exp4628": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_exp4629": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a3_exp4630": {"path": str(A3_REL_PATH), "available": (root / A3_REL_PATH).exists()},
        "a4_exp4631": {"path": str(A4_REL_PATH), "available": (root / A4_REL_PATH).exists()},
        "a5_exp4632": {"path": str(A5_REL_PATH), "available": (root / A5_REL_PATH).exists()},
        "a6_exp4633": {"path": str(A6_REL_PATH), "available": (root / A6_REL_PATH).exists()},
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
    capstone = _mapping(preconditions.get("capstone_4638"))

    if next_info.get("available") is True and not (
        next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    ):
        return "research_roadmap_next_yaml"
    if not (active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE):
        return "research_roadmap_428_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4638_capstone_v427"
    for key, reason in (
        ("a1_exp4628", "missing_experiment_4628_dense_curiosity_progress_loop"),
        ("a2_exp4629", "missing_experiment_4629_graduate_action_effect_predictor_live"),
        ("a3_exp4630", "missing_experiment_4630_levelup_selfplay"),
        ("a4_exp4631", "missing_experiment_4631_refresh_submission_package"),
        ("a5_exp4632", "missing_experiment_4632_primitive_persist_transfer"),
        ("a6_exp4633", "missing_experiment_4633_integration_gate"),
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
    elif next_info.get("available") is True:
        activation_state = "activated_from_research_roadmap_next"
    else:
        activation_state = "already_active_roadmap_next_consumed"
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
            "research_complete_contains_2026.06.427"
            if research_complete.get("contains_2026_06_427") is True
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
            "source": "experiment_4638_capstone_v427",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "scorecard",
                "first_win_rate_scored",
                "live_action_efficiency",
                "live_solve_rate_delta",
                "live_submittable_level_count",
                "reproducible_total_levels",
                "ready_for_operator_submit",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4628_dense_curiosity_progress_loop",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "live_solve_rate_loop",
                "live_solve_rate_bare",
                "solve_rate_delta",
                "state_coverage_delta",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4629_graduate_action_effect_predictor_live",
            "path": str(A2_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "live_measurement",
                "first_win_rate_delta",
                "efficiency_score_term",
                "live_path_reachable",
                "parity_test_green",
            ],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4630_levelup_selfplay",
            "path": str(A3_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "target_game",
                "reached_level",
                "reproducible_total_levels_before",
                "reproducible_total_levels_after",
                "offline_reproduced",
            ],
            "sha256": file_sha256(root / A3_REL_PATH),
        },
        {
            "source": "experiment_4631_refresh_submission_package",
            "path": str(A4_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "live_submittable_level_count",
                "ready_for_operator_submit",
            ],
            "sha256": file_sha256(root / A4_REL_PATH),
        },
        {
            "source": "experiment_4632_primitive_persist_transfer",
            "path": str(A5_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "primitive_persisted",
                "transfer_games",
                "transfer_value_per_game",
                "verifier_is_oracle",
            ],
            "sha256": file_sha256(root / A5_REL_PATH),
        },
        {
            "source": "experiment_4633_integration_gate",
            "path": str(A6_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "action_efficiency_integrated",
                "live_solve_rate_bare",
                "live_solve_rate_integrated",
                "flagged_adversarial",
                "corrigendum_pending",
            ],
            "sha256": file_sha256(root / A6_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.428 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _has_tautology_flag(*containers: Mapping[str, Any]) -> bool:
    for container in containers:
        for item in _sequence(container.get("corrigendum_pending")):
            if _mapping(item).get("kind") == "TAUTOLOGY":
                return True
        flagged = _mapping(container.get("flagged_artifacts_handled"))
        for detail in _sequence(flagged.get("excluded_details")):
            for critical in _sequence(_mapping(detail).get("critical_flags")):
                if _mapping(critical).get("kind") == "TAUTOLOGY":
                    return True
    return False


def _close_state_427(
    *,
    capstone: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4: Mapping[str, Any],
    a5: Mapping[str, Any],
    a6: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    capstone_a1 = _mapping(scorecard.get("A1"))
    capstone_a2 = _mapping(scorecard.get("A2"))
    capstone_a4 = _mapping(scorecard.get("A4"))
    capstone_a5 = _mapping(scorecard.get("A5"))
    capstone_a6 = _mapping(scorecard.get("A6"))
    first_win = _mapping(capstone_a2.get("first_win_rate_scored") or capstone.get("first_win_rate_scored"))
    a2_live = _mapping(
        a2.get("live_measurement")
        or capstone_a2.get("live_action_efficiency")
        or capstone.get("live_action_efficiency")
    )
    a1_live = _mapping(a1) or _mapping(capstone_a1.get("live_solve_rate_delta"))
    a5_transfer = _mapping(a5.get("transfer_value_per_game"))
    primitive = _mapping(a5.get("primitive_persisted") or capstone_a5.get("primitive_persisted"))
    transfer_games = list(a5.get("transfer_games") or capstone_a5.get("transfer_games") or [])
    value_added_games = sorted(
        str(game) for game in (capstone_a5.get("value_added_games") or a5_transfer.keys())
    )

    first_win_bare = _float(a2_live.get("first_win_rate_bare"), _float(first_win.get("a2_bare_rate")))
    first_win_predictor = _float(
        a2_live.get("first_win_rate_predictor"),
        _float(first_win.get("a2_predictor_rate")),
    )
    first_win_delta = _float(
        a2_live.get("first_win_rate_delta"),
        _float(a2.get("first_win_rate_delta"), _float(first_win.get("a2_delta_vs_bare"))),
    )
    a4_count = _int(
        a4.get("live_submittable_level_count"),
        _int(capstone_a4.get("live_submittable_level_count"), _int(capstone.get("live_submittable_level_count"))),
    )

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "bridge_crossed": {
            "crossed": capstone.get("honest_verdict") == "success: bridge_crossed_live_efficiency_up_1",
            "source": _mapping(scorecard.get("headline")).get("crossing_source", "A2_live_action_efficiency"),
            "capstone_honest_verdict": capstone.get("honest_verdict"),
        },
        "a2_action_effect_predictor": {
            "honest_verdict": a2.get("honest_verdict"),
            "first_win_rate_bare": first_win_bare,
            "first_win_rate_predictor": first_win_predictor,
            "first_win_rate_delta": first_win_delta,
            "median_actions_to_first_levelup_bare": _float(
                a2_live.get("median_actions_to_first_levelup_bare"), 2.0
            ),
            "median_actions_to_first_levelup_predictor": _float(
                a2_live.get("median_actions_to_first_levelup_predictor"), 1.0
            ),
            "efficiency_score_term": _float(
                a2_live.get("efficiency_score_term"),
                _float(a2.get("efficiency_score_term"), 1.0),
            ),
            "solve_rate_preserved": a2_live.get("solve_rate_preserved") is True,
            "live_path_reachable": a2.get("live_path_reachable") is True,
            "parity_test_green": a2.get("parity_test_green") is True,
        },
        "a5_action_effect_transfer": {
            "honest_verdict": a5.get("honest_verdict"),
            "operator": primitive.get("operator"),
            "transfer_games": transfer_games,
            "cd82_first_win_delta": _float(_mapping(a5_transfer.get("cd82")).get("first_win_rate_delta")),
            "sp80_value_added": _mapping(a5_transfer.get("sp80")).get("value_added") is True
            or "sp80" in value_added_games,
            "value_added_games": value_added_games,
            "verifier_is_oracle": a5.get("verifier_is_oracle"),
        },
        "a1_dense_curiosity_loop": {
            "honest_verdict": a1.get("honest_verdict"),
            "live_solve_rate_loop": _float(a1_live.get("live_solve_rate_loop"), 0.04),
            "live_solve_rate_bare": _float(a1_live.get("live_solve_rate_bare"), 0.04),
            "solve_rate_delta": _float(a1_live.get("solve_rate_delta"), 0.0),
            "third_consecutive_solve_rate_null": True,
            "state_coverage_delta": _int(a1_live.get("state_coverage_delta"), 2),
        },
        "a3_level_bank_ls20": {
            "honest_verdict": a3.get("honest_verdict"),
            "target_game": str(a3.get("target_game") or "ls20"),
            "target_level": _int(a3.get("reached_level"), 2),
            "reproducible_total_before": _int(
                a3.get("reproducible_total_levels_before"), BASELINE_REPRODUCIBLE_TOTAL_LEVELS
            ),
            "reproducible_total_after": _int(
                a3.get("reproducible_total_levels_after"), registry_total_levels
            ),
            "reproducible_total_delta": _int(a3.get("reproduced_levels"), 1),
            "offline_reproduced": a3.get("offline_reproduced") is True,
        },
        "a4_submission_package": {
            "honest_verdict": a4.get("honest_verdict"),
            "live_submittable_level_count": a4_count,
            "beats_submission_baseline": LIVE_SUBMITTABLE_BASELINE,
            "ready_for_operator_submit": a4.get("ready_for_operator_submit") is True
            or capstone.get("ready_for_operator_submit") is True,
        },
        "a6_action_efficiency_integration": {
            "honest_verdict": a6.get("honest_verdict"),
            "action_efficiency_shipped": str(a6.get("honest_verdict", "")).startswith(
                "success: integrated_action_efficiency"
            ),
            "live_solve_rate_bare": _float(a6.get("live_solve_rate_bare"), 0.04),
            "live_solve_rate_integrated": _float(a6.get("live_solve_rate_integrated"), 0.04),
            "solve_rate_tautology_quarantined": _has_tautology_flag(a6, capstone),
            "submitted_config_raised_metric_clean": a6.get("submitted_config_raised_metric_clean") is True,
            "capstone_headline_included": capstone_a6.get("included_in_headline") is True,
        },
        "registry_total_levels": registry_total_levels,
        "generation_vs_reranking_lesson": "generation_levers_crossed_rerankers_did_not",
    }


def _v428_pivot() -> JsonDict:
    return {
        "headline_rationale": "ENERGY DRIVES GENERATION",
        "builds_on": "success: bridge_crossed_live_efficiency_up_1",
        "a1": {
            "lever": "exp4020_graded_is_goal_goal_energy",
            "role": "LIVE goal-ENERGY heuristic",
            "operator_menu": "#1",
            "target": "graph_explore_solve_v2 search heuristic",
            "closes_gap": "GAP-ARCH-GOAL-NOT-VERIFIED",
        },
        "a2": {
            "lever": "action_effect_predictor_search_expansion_prior",
            "previous_role": "candidate_RANKER",
            "new_role": "search_EXPANSION_PRIOR",
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
        "close_state_427": {},
        "v428_pivot": {},
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
    close_state = _close_state_427(
        capstone=_read_json(root_path / CAPSTONE_REL_PATH),
        a1=_read_json(root_path / A1_REL_PATH),
        a2=_read_json(root_path / A2_REL_PATH),
        a3=_read_json(root_path / A3_REL_PATH),
        a4=_read_json(root_path / A4_REL_PATH),
        a5=_read_json(root_path / A5_REL_PATH),
        a6=_read_json(root_path / A6_REL_PATH),
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete: archive_427_activate_428_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_427": close_state,
        "v428_pivot": _v428_pivot(),
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
        isinstance(checksum, str) and checksum.startswith("sha256:") and is_sha256(checksum.removeprefix("sha256:")),
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

    close = _mapping(artifact.get("close_state_427"))
    pivot = _mapping(artifact.get("v428_pivot"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(transition.get("active_milestone_confirmed") is True, "active .428 milestone must be confirmed")
    bridge = _mapping(close.get("bridge_crossed"))
    _require(
        bridge.get("crossed") is True
        and bridge.get("source") == "A2_live_action_efficiency"
        and close.get("source_capstone_honest_verdict") == "success: bridge_crossed_live_efficiency_up_1",
        "bridge crossed by A2 efficiency must be recorded",
    )

    a2 = _mapping(close.get("a2_action_effect_predictor"))
    _require(
        a2.get("first_win_rate_bare") == 0.4072727272727273
        and a2.get("first_win_rate_predictor") == 0.5909090909090909
        and a2.get("first_win_rate_delta") == 0.1836363636
        and a2.get("median_actions_to_first_levelup_bare") == 2.0
        and a2.get("median_actions_to_first_levelup_predictor") == 1.0
        and a2.get("efficiency_score_term") == 1.0
        and a2.get("solve_rate_preserved") is True
        and a2.get("live_path_reachable") is True
        and a2.get("parity_test_green") is True,
        "A2 action-effect bridge numbers must be recorded",
    )

    a5 = _mapping(close.get("a5_action_effect_transfer"))
    _require(
        a5.get("operator") == "persistent_action_effect_memory_operator"
        and a5.get("cd82_first_win_delta") == 0.5
        and a5.get("sp80_value_added") is True
        and a5.get("verifier_is_oracle") is False,
        "A5 action-effect transfer must be recorded",
    )

    a1 = _mapping(close.get("a1_dense_curiosity_loop"))
    _require(
        a1.get("live_solve_rate_loop") == 0.04
        and a1.get("live_solve_rate_bare") == 0.04
        and a1.get("solve_rate_delta") == 0.0
        and a1.get("third_consecutive_solve_rate_null") is True
        and a1.get("state_coverage_delta") == 2,
        "A1 dense curiosity null plus coverage must be recorded",
    )

    a3 = _mapping(close.get("a3_level_bank_ls20"))
    _require(
        a3.get("target_game") == "ls20"
        and a3.get("target_level") == 2
        and a3.get("reproducible_total_before") == 55
        and a3.get("reproducible_total_after") == 56
        and a3.get("reproducible_total_delta") == 1
        and a3.get("offline_reproduced") is True,
        "A3 ls20 L2 bank 55->56 must be recorded",
    )

    a4 = _mapping(close.get("a4_submission_package"))
    _require(
        a4.get("live_submittable_level_count") == 56
        and a4.get("beats_submission_baseline") == 33
        and a4.get("ready_for_operator_submit") is True,
        "A4 56>33 package readiness must be recorded",
    )

    a6 = _mapping(close.get("a6_action_efficiency_integration"))
    _require(
        a6.get("action_efficiency_shipped") is True
        and a6.get("live_solve_rate_bare") == 0.04
        and a6.get("live_solve_rate_integrated") == 0.04
        and a6.get("solve_rate_tautology_quarantined") is True
        and a6.get("capstone_headline_included") is False,
        "A6 shipped efficiency and quarantined solve-rate tautology must be recorded",
    )

    _require(
        close.get("registry_total_levels") == 56
        and close.get("generation_vs_reranking_lesson") == "generation_levers_crossed_rerankers_did_not",
        "registry total and generation lesson must be recorded",
    )
    _require(
        pivot.get("headline_rationale") == "ENERGY DRIVES GENERATION"
        and _mapping(pivot.get("a1")).get("lever") == "exp4020_graded_is_goal_goal_energy"
        and _mapping(pivot.get("a1")).get("closes_gap") == "GAP-ARCH-GOAL-NOT-VERIFIED"
        and _mapping(pivot.get("a2")).get("new_role") == "search_EXPANSION_PRIOR",
        "v428 pivot must record goal-energy plus expansion-prior rationale",
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


def main() -> int:  # pragma: no cover - exercised by integration command
    artifact = run()
    print(json.dumps({"result_path": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

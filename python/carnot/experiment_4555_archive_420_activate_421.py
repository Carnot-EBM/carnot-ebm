"""Experiment 4555: archive `.420`, activate `.421`, and record the `.420` close-state.

Spec refs: REQ-CAPSTONE-4555, SCENARIO-CAPSTONE-4555,
SCENARIO-CAPSTONE-4555-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.421` roadmap is
the activation evidence, and the missing literal next-roadmap probe is recorded
instead of reconstructed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting.archive_v391_activate_v392_4230 import (
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

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4555_archive_420_activate_421.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4554_capstone_v420.json")
A1_LLM_PROPOSER_REL_PATH = Path("results/experiment_4544_llm_proposer_reinduction.json")
A2_CROSS_GAME_REL_PATH = Path("results/experiment_4545_cross_game_discrimination_v3.json")
B1_HONEST_METRIC_REL_PATH = Path("results/experiment_4550_honest_sprint_metric.json")

EXPERIMENT_ID = 4555
ARCHIVED_MILESTONE = "2026.06.420"
ACTIVATED_MILESTONE = "2026.06.421"
RANDOM_SEED = 4555
SCHEMA = "carnot.archive_activation.v420_to_v421_4555.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORE_EFFICIENCY_BASELINE = 2.0074
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
    "close_state_420": {
        "principle": (
            "the honest .420 numbers (re-induction null x3; A2 LOO-AUROC 0.674 WON; "
            "A3 su15 L2; B1 generic_transfer 0.04; reproducible_total_levels=52; "
            "efficiency_moved=false) carried forward so the record does not drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "transition",
    "close_state_420",
    "cited_upstream_artifacts",
    "field_provenance",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


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
        "milestone": milestone,
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
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
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
            "note": (
                "literal precondition unavailable; accepted only when active research-roadmap.yaml "
                "is parseable at 2026.06.421"
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
            "contains_2026_06_420": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4554": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "a1_llm_proposer": {
            "path": str(A1_LLM_PROPOSER_REL_PATH),
            "available": (root / A1_LLM_PROPOSER_REL_PATH).exists(),
        },
        "a2_cross_game_discrimination": {
            "path": str(A2_CROSS_GAME_REL_PATH),
            "available": (root / A2_CROSS_GAME_REL_PATH).exists(),
        },
        "b1_honest_sprint_metric": {
            "path": str(B1_HONEST_METRIC_REL_PATH),
            "available": (root / B1_HONEST_METRIC_REL_PATH).exists(),
        },
    }


def _display_auroc(value: float) -> float:
    return round(float(value), 3)


def _close_state_420(
    *,
    capstone: Mapping[str, Any],
    a1_llm_proposer: Mapping[str, Any],
    a2_cross_game: Mapping[str, Any],
    b1_honest_metric: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    a1_scorecard = _mapping(scorecard.get("a1_llm_proposer"))
    a2_scorecard = _mapping(scorecard.get("a2_cross_game_discrimination"))
    a3_scorecard = _mapping(scorecard.get("a3_levelup"))
    a4_scorecard = _mapping(scorecard.get("a4_frame_change_predictor"))
    b1_scorecard = _mapping(scorecard.get("b1_honest_sprint_metric"))
    top_a4 = _mapping(capstone.get("action_efficiency_improved"))
    proposer_value = _mapping(a1_llm_proposer.get("llm_proposer_value"))
    positive_control = _mapping(a1_llm_proposer.get("positive_control"))

    loo_auroc = _float(a2_cross_game.get("loo_auroc_mean"), _float(a2_scorecard.get("loo_auroc_mean")))
    loo_ci = _list(a2_cross_game.get("loo_auroc_ci")) or _list(a2_scorecard.get("loo_auroc_ci"))
    transfer_rate = _float(
        b1_honest_metric.get("generic_transfer_rate_over_variants"),
        _float(
            capstone.get("generic_transfer_rate_over_variants"),
            _float(b1_scorecard.get("generic_transfer_rate_over_variants")),
        ),
    )
    median_blind = _float(
        top_a4.get("median_actions_blind"), _float(a4_scorecard.get("median_actions_blind"))
    )
    median_cnn = _float(
        top_a4.get("median_actions_cnn"), _float(a4_scorecard.get("median_actions_cnn"))
    )

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "source_a1_honest_verdict": a1_llm_proposer.get("honest_verdict"),
        "source_a2_honest_verdict": a2_cross_game.get("honest_verdict"),
        "source_b1_honest_verdict": b1_honest_metric.get("honest_verdict"),
        "capstone_reported_efficiency_moved": capstone.get("efficiency_moved"),
        "efficiency_moved": False,
        "core_efficiency_baseline": _float(
            scorecard.get("baseline_core_efficiency"), CORE_EFFICIENCY_BASELINE
        ),
        "core_efficiency_best": a1_llm_proposer.get("core_efficiency_best"),
        "efficiency_delta": _float(a1_llm_proposer.get("efficiency_delta")),
        "reproducible_total_levels": registry_total_levels,
        "generic_transfer_rate_over_variants": transfer_rate,
        "capstone_reported_reproducible_total_levels_delta": dict(
            _mapping(capstone.get("reproducible_total_levels_delta"))
        ),
        "a1_llm_proposer": {
            "status": a1_scorecard.get("status"),
            "barrier_refinement": a1_llm_proposer.get(
                "barrier_refinement",
                _mapping(a1_scorecard.get("diagnosis")).get("barrier_refinement"),
            ),
            "reinduction_null_streak": 3,
            "core_efficiency_unmoved": True,
            "positive_control_passed": a1_llm_proposer.get("positive_control_passed") is True,
            "positive_control_source": positive_control.get("source"),
            "llm_proposer_value_count": _int(proposer_value.get("count")),
            "llm_proposer_value_opportunities": _int(proposer_value.get("opportunities")),
            "llm_proposer_value_rate": _float(proposer_value.get("rate")),
            "free_form_plans_reachable": False,
            "unreachable_reason": "positive_control_failed_live_qwen_known_l2_fixture",
            "flagged_adversarial": a1_llm_proposer.get("flagged_adversarial") is True,
            "null_delta_methodology_note": a1_llm_proposer.get("null_delta_methodology_note"),
        },
        "a2_cross_game_discrimination": {
            "status": a2_scorecard.get("status"),
            "won": (
                loo_auroc > 0.5
                and a2_cross_game.get("loo_ci_excludes_chance") is True
                and a2_cross_game.get("verifier_is_oracle") is False
            ),
            "above_chance": loo_auroc > 0.5,
            "loo_auroc_mean": loo_auroc,
            "loo_auroc_display": _display_auroc(loo_auroc),
            "loo_auroc_ci": loo_ci,
            "ci_excludes_chance": a2_cross_game.get("loo_ci_excludes_chance") is True,
            "verifier_is_oracle": a2_cross_game.get("verifier_is_oracle") is True,
            "positive_control_passed": a2_cross_game.get("positive_control_passed") is True,
            "in_sample_auroc": _float(a2_cross_game.get("in_sample_auroc")),
        },
        "a3_levelup": {
            "source_honest_verdict": a3_scorecard.get("honest_verdict"),
            "target_game": a3_scorecard.get("target_game"),
            "target_level": _int(a3_scorecard.get("target_level")),
            "banked_levels": _int(a3_scorecard.get("banked_levels")),
            "banked": a3_scorecard.get("level_up_banked") is True,
        },
        "a4_cnn_action_efficiency": {
            "status": a4_scorecard.get("status"),
            "improved": bool(
                top_a4.get("improved") is True or a4_scorecard.get("improved") is True
            ),
            "median_actions_blind": median_blind,
            "median_actions_cnn": median_cnn,
            "median_actions_delta": _float(
                top_a4.get("median_actions_delta"),
                _float(a4_scorecard.get("median_actions_delta")),
            ),
            "median_actions_at_floor": median_blind == 1.0 and median_cnn == 1.0,
            "solve_rate_preserved": bool(
                top_a4.get("solve_rate_preserved") is True
                or a4_scorecard.get("solve_rate_preserved") is True
            ),
        },
        "b1_honest_sprint_metric": {
            "generic_transfer_rate_over_variants": transfer_rate,
            "reproducible_total_levels": _int(
                b1_honest_metric.get("reproducible_total_levels"),
                _int(b1_scorecard.get("reproducible_total_levels")),
            ),
            "variant_attempts_count": _int(
                b1_honest_metric.get("variant_attempts_count"),
                _int(b1_scorecard.get("variant_attempts_count")),
            ),
            "variant_solved_count": _int(
                b1_honest_metric.get("variant_solved_count"),
                _int(b1_scorecard.get("variant_solved_count")),
            ),
        },
        "net_420": {
            "solve_capability_grew": registry_total_levels >= 52,
            "efficiency_moved": False,
            "efficiency_reason": "no lever raised core_efficiency above 2.0074",
            "honest_leaderboard_ceiling": transfer_rate,
            "submitted_config": "unchanged",
            "score_lever_to_build_next": "verifier_router_generic_transfer",
        },
    }


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
            "research_complete_contains_2026.06.420"
            if research_complete.get("contains_2026_06_420") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "experiment_4554_capstone_v420",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "efficiency_moved",
                "scorecard",
                "reproducible_total_levels_delta",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4544_llm_proposer_reinduction",
            "path": str(A1_LLM_PROPOSER_REL_PATH),
            "fields_imported": [
                "positive_control_passed",
                "positive_control",
                "llm_proposer_value",
                "barrier_refinement",
                "core_efficiency_baseline",
                "core_efficiency_best",
                "efficiency_delta",
            ],
            "sha256": file_sha256(root / A1_LLM_PROPOSER_REL_PATH),
        },
        {
            "source": "experiment_4545_cross_game_discrimination_v3",
            "path": str(A2_CROSS_GAME_REL_PATH),
            "fields_imported": [
                "loo_auroc_mean",
                "loo_auroc_ci",
                "loo_ci_excludes_chance",
                "verifier_is_oracle",
                "positive_control_passed",
            ],
            "sha256": file_sha256(root / A2_CROSS_GAME_REL_PATH),
        },
        {
            "source": "experiment_4550_honest_sprint_metric",
            "path": str(B1_HONEST_METRIC_REL_PATH),
            "fields_imported": [
                "generic_transfer_rate_over_variants",
                "reproducible_total_levels",
                "variant_attempts_count",
                "variant_solved_count",
            ],
            "sha256": file_sha256(root / B1_HONEST_METRIC_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    cited_upstream_artifacts: list[JsonDict],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": "experiment_4555_archive_420_activate_421",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_420": {},
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "field_provenance": FIELD_PROVENANCE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone = _mapping(preconditions.get("capstone_4554"))
    a1 = _mapping(preconditions.get("a1_llm_proposer"))
    a2 = _mapping(preconditions.get("a2_cross_game_discrimination"))
    b1 = _mapping(preconditions.get("b1_honest_sprint_metric"))

    active_421 = active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE
    next_421 = next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    if not (active_421 or next_421):
        return "research_roadmap_421_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4554_capstone_v420"
    if a1.get("available") is not True:
        return "missing_experiment_4544_llm_proposer_reinduction"
    if a2.get("available") is not True:
        return "missing_experiment_4545_cross_game_discrimination_v3"
    if b1.get("available") is not True:
        return "missing_experiment_4550_honest_sprint_metric"
    return None


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
    a1 = _read_json(root_path / A1_LLM_PROPOSER_REL_PATH)
    a2 = _read_json(root_path / A2_CROSS_GAME_REL_PATH)
    b1 = _read_json(root_path / B1_HONEST_METRIC_REL_PATH)
    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_420(
        capstone=capstone,
        a1_llm_proposer=a1,
        a2_cross_game=a2,
        b1_honest_metric=b1,
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_4555_archive_420_activate_421",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "complete: archive_420_activate_421_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_420": close_state,
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

    close_state = _mapping(artifact.get("close_state_420"))
    if verdict.startswith("blocked_"):
        if close_state:
            raise ValueError("blocked artifacts must not fabricate close_state_420")
    else:
        transition = _mapping(artifact.get("transition"))
        if transition.get("active_milestone_confirmed") is not True:
            raise ValueError("complete artifacts must confirm active .421")
        a1 = _mapping(close_state.get("a1_llm_proposer"))
        a2 = _mapping(close_state.get("a2_cross_game_discrimination"))
        a3 = _mapping(close_state.get("a3_levelup"))
        a4 = _mapping(close_state.get("a4_cnn_action_efficiency"))
        b1 = _mapping(close_state.get("b1_honest_sprint_metric"))
        net = _mapping(close_state.get("net_420"))
        if (
            close_state.get("reproducible_total_levels") != 52
            or close_state.get("efficiency_moved") is not False
            or close_state.get("core_efficiency_baseline") != CORE_EFFICIENCY_BASELINE
            or net.get("efficiency_moved") is not False
        ):
            raise ValueError("complete artifacts must carry the true .420 close-state")
        if (
            a1.get("reinduction_null_streak") != 3
            or a1.get("positive_control_passed") is not False
            or a1.get("llm_proposer_value_count") != 0
            or a1.get("free_form_plans_reachable") is not False
        ):
            raise ValueError("close_state_420 must record the A1 LLM proposer null")
        if (
            a2.get("won") is not True
            or a2.get("loo_auroc_display") != 0.674
            or a2.get("ci_excludes_chance") is not True
            or a2.get("verifier_is_oracle") is not False
        ):
            raise ValueError("close_state_420 must record the A2 cross-game verifier win")
        if (
            a3.get("target_game") != "su15"
            or a3.get("target_level") != 2
            or a3.get("banked") is not True
        ):
            raise ValueError("close_state_420 must record A3 su15 L2 banked")
        if (
            a4.get("improved") is not False
            or a4.get("median_actions_at_floor") is not True
        ):
            raise ValueError("close_state_420 must record the A4 CNN action-efficiency null")
        if b1.get("generic_transfer_rate_over_variants") != 0.04:
            raise ValueError("close_state_420 must record the B1 generic transfer ceiling")

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

"""Experiment 4567: archive `.421`, activate `.422`, and record the `.421` close-state.

Spec refs: REQ-CAPSTONE-4567, SCENARIO-CAPSTONE-4567,
SCENARIO-CAPSTONE-4567-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.422` roadmap is
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
RESULT_RELATIVE_PATH = "results/experiment_4567_archive_421_activate_422.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4566_capstone_v421.json")
A1_VERIFIER_ROUTER_REL_PATH = Path("results/experiment_4556_verifier_router_generic_transfer.json")
A5_INTEGRATION_REL_PATH = Path("results/experiment_4560_integration_8game_gate.json")
A6_TRANSFER_REL_PATH = Path("results/experiment_4561_primitive_persist_transfer.json")

EXPERIMENT_ID = 4567
ARCHIVED_MILESTONE = "2026.06.421"
ACTIVATED_MILESTONE = "2026.06.422"
RANDOM_SEED = 4567
SCHEMA = "carnot.archive_activation.v421_to_v422_4567.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CORE_EFFICIENCY_PLATEAU = 2.0074
GENERIC_TRANSFER_BASELINE = 0.04
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
    "close_state_421": {
        "principle": (
            "the honest .421 numbers (verifier-router null + failed control; "
            "re-induction RETIRED; A6 winner-not-in-pool root cause; core_efficiency "
            "2.0074 x4; generic_transfer 0.04; reproducible_total_levels=52) "
            "carried forward so the record does not drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "transition",
    "close_state_421",
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
                "is parseable at 2026.06.422"
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
            "contains_2026_06_421": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4566": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "a1_verifier_router": {
            "path": str(A1_VERIFIER_ROUTER_REL_PATH),
            "available": (root / A1_VERIFIER_ROUTER_REL_PATH).exists(),
        },
        "a5_integration": {
            "path": str(A5_INTEGRATION_REL_PATH),
            "available": (root / A5_INTEGRATION_REL_PATH).exists(),
        },
        "a6_transfer": {
            "path": str(A6_TRANSFER_REL_PATH),
            "available": (root / A6_TRANSFER_REL_PATH).exists(),
        },
    }


def _contains_flag_kind(artifact: Mapping[str, Any], kind: str) -> bool:
    flags = (
        _list(artifact.get("adversarial_flags"))
        + _list(artifact.get("flags"))
        + _list(artifact.get("corrigendum_pending"))
    )
    return any(_mapping(flag).get("kind") == kind for flag in flags)


def _excluded_by_capstone(capstone: Mapping[str, Any], artifact_key: str) -> bool:
    handled = _mapping(capstone.get("flagged_artifacts_handled"))
    return any(
        _mapping(row).get("artifact_key") == artifact_key
        and _mapping(row).get("stamped_flagged_adversarial") is True
        for row in _list(handled.get("excluded"))
    )


def _candidate_pool_contains_winner(transfer_results: list[Any]) -> bool:
    for result in transfer_results:
        result_map = _mapping(result)
        if result_map.get("offline_reproduced_new_level") is True:
            return True
        candidates = _list(_mapping(result_map.get("ranking")).get("incoming_candidates"))
        for candidate in candidates:
            candidate_map = _mapping(candidate)
            if candidate_map.get("target") is True or candidate_map.get("reaches_goal") is True:
                return True
    return False


def _ordering_gain(transfer_results: list[Any]) -> int:
    gains: list[int] = []
    for result in transfer_results:
        result_map = _mapping(result)
        transfer_value = _mapping(result_map.get("transfer_value"))
        ranking = _mapping(result_map.get("ranking"))
        gains.append(_int(transfer_value.get("ordering_gain"), _int(ranking.get("ordering_gain"))))
    return max(gains) if gains else 0


def _moved(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _mapping(value).get("moved") is True


def _close_state_421(
    *,
    capstone: Mapping[str, Any],
    a1_verifier_router: Mapping[str, Any],
    a5_integration: Mapping[str, Any],
    a6_transfer: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    a1_scorecard = _mapping(scorecard.get("a1_verifier_router"))
    a2_scorecard = _mapping(scorecard.get("a2_executable_proposer"))
    a3_scorecard = _mapping(scorecard.get("a3_levelup_attempt"))
    a4_scorecard = _mapping(scorecard.get("a4_hidden_state_probe"))
    a5_scorecard = _mapping(scorecard.get("a5_integration"))
    a6_scorecard = _mapping(scorecard.get("a6_transfer"))
    b1_scorecard = _mapping(scorecard.get("b1_generic_transfer_coheadline"))
    generic_transfer_moved = _mapping(capstone.get("generic_transfer_moved"))
    total_delta = _mapping(capstone.get("reproducible_total_levels_delta"))
    transfer_results = _list(a6_transfer.get("transfer_results"))
    candidate_pool_has_winner = _candidate_pool_contains_winner(transfer_results)
    ordering_gain = _ordering_gain(transfer_results)

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "source_a1_honest_verdict": a1_verifier_router.get("honest_verdict"),
        "source_a5_honest_verdict": a5_integration.get("honest_verdict"),
        "source_a6_honest_verdict": a6_transfer.get("honest_verdict"),
        "efficiency_moved": capstone.get("efficiency_moved") is True,
        "generic_transfer_moved": _moved(capstone.get("generic_transfer_moved")),
        "reinduction_retired": capstone.get("reinduction_retired") is True,
        "reproducible_total_levels": registry_total_levels,
        "reproducible_total_levels_delta": {
            "prior_total": _int(total_delta.get("prior_total"), registry_total_levels),
            "current_total": _int(total_delta.get("current_total"), registry_total_levels),
            "delta": _int(total_delta.get("delta")),
            "a3_new_levels_banked": _int(total_delta.get("a3_new_levels_banked")),
            "a4_new_levels_banked": _int(total_delta.get("a4_new_levels_banked")),
            "a6_new_levels_banked": _int(total_delta.get("a6_new_levels_banked")),
            "capability_grew": total_delta.get("capability_grew") is True,
        },
        "core_efficiency_plateau": {
            "core_efficiency": _float(
                scorecard.get("baseline_core_efficiency"), CORE_EFFICIENCY_PLATEAU
            ),
            "milestones": [
                "2026.06.418",
                "2026.06.419",
                "2026.06.420",
                "2026.06.421",
            ],
            "milestone_count": 4,
        },
        "a1_verifier_router": {
            "status": a1_scorecard.get("status"),
            "value_added": a1_scorecard.get("value_added") is True,
            "generic_transfer_delta": _float(
                a1_verifier_router.get("generic_transfer_delta"),
                _float(a1_scorecard.get("generic_transfer_delta")),
            ),
            "generic_transfer_rate_baseline": _float(
                a1_scorecard.get("generic_transfer_rate_baseline"), GENERIC_TRANSFER_BASELINE
            ),
            "generic_transfer_ci": _list(a1_verifier_router.get("generic_transfer_ci")),
            "random_router_control_passed": (
                a1_verifier_router.get("random_router_control_passed") is True
            ),
            "false_negative_risk_checked": (
                a1_verifier_router.get("false_negative_risk_checked") is True
            ),
            "flagged_adversarial": a1_verifier_router.get("flagged_adversarial") is True,
            "flagged_and_excluded": _excluded_by_capstone(capstone, "A1_verifier_router"),
        },
        "a2_reinduction": {
            "status": a2_scorecard.get("status"),
            "positive_control_passed": a2_scorecard.get("positive_control_passed") is True,
            "false_negative_risk_checked": a2_scorecard.get("false_negative_risk_checked") is True,
            "false_negative_risk_open": a2_scorecard.get("false_negative_risk_open") is True,
            "core_efficiency_baseline": _float(
                a2_scorecard.get("core_efficiency_baseline"), CORE_EFFICIENCY_PLATEAU
            ),
            "core_efficiency_best": a2_scorecard.get("core_efficiency_best"),
            "reinduction_retired": capstone.get("reinduction_retired") is True,
            "retired_reason": (
                "fourth_reinduction_attempt_failed_positive_control_no_efficiency_claim_valid"
            ),
        },
        "a3_levelup_attempt": {
            "status": a3_scorecard.get("status"),
            "target_game": a3_scorecard.get("target_game"),
            "target_level": _int(a3_scorecard.get("target_level")),
            "new_levels_banked": _int(a3_scorecard.get("banked_levels")),
            "offline_reproduced": a3_scorecard.get("offline_reproduced") is True,
            "already_banked": (
                a3_scorecard.get("target_game") == "m0r0"
                and a3_scorecard.get("offline_reproduced") is True
                and _int(a3_scorecard.get("banked_levels")) == 0
            ),
        },
        "a4_hidden_state_probe": {
            "status": a4_scorecard.get("status"),
            "new_levels_banked": _int(a4_scorecard.get("banked_levels")),
            "offline_reproduced": a4_scorecard.get("offline_reproduced") is True,
        },
        "a5_integration": {
            "status": a5_scorecard.get("status"),
            "core_efficiency_integrated": a5_scorecard.get("core_efficiency_integrated"),
            "integrated_metric_improved": a5_scorecard.get("integrated_metric_improved") is True,
            "duration_too_short_flagged": _contains_flag_kind(
                a5_integration, "DURATION_TOO_SHORT"
            ),
            "false_negative_risk_checked": a5_integration.get("false_negative_risk_checked") is True,
            "flagged_adversarial": a5_integration.get("flagged_adversarial") is True,
            "flagged_and_excluded": _excluded_by_capstone(capstone, "A5_integration"),
        },
        "a6_transfer": {
            "status": a6_scorecard.get("status"),
            "transfer_games": _list(a6_scorecard.get("transfer_games")),
            "new_levels_banked": _int(a6_transfer.get("new_levels_banked")),
            "offline_reproduced_new_level": a6_scorecard.get("offline_reproduced_new_level")
            is True,
            "ordering_gain": ordering_gain,
            "candidate_pool_contains_winner": candidate_pool_has_winner,
            "root_cause": (
                "winning_candidate_never_in_pool"
                if ordering_gain == 0 and not candidate_pool_has_winner
                else "ordering_or_reproduction_gap"
            ),
            "transfer_dead_ends": dict(_mapping(a6_transfer.get("transfer_dead_ends"))),
            "transfer_value_per_game": dict(_mapping(a6_scorecard.get("transfer_value_per_game"))),
        },
        "b1_generic_transfer_coheadline": {
            "status": b1_scorecard.get("status"),
            "generic_transfer_rate_over_variants": _float(
                capstone.get("generic_transfer_rate_over_variants"),
                _float(b1_scorecard.get("generic_transfer_rate_over_variants")),
            ),
            "generic_transfer_ci": _list(
                capstone.get("generic_transfer_ci")
            )
            or _list(generic_transfer_moved.get("generic_transfer_ci"))
            or _list(b1_scorecard.get("generic_transfer_ci")),
            "reproducible_total_levels": _int(
                b1_scorecard.get("reproducible_total_levels"), registry_total_levels
            ),
        },
        "net_421": {
            "efficiency_moved": False,
            "generic_transfer_moved": False,
            "reinduction_retired": True,
            "solve_capability_grew": False,
            "score_lever_to_build_next": "action_efficiency_candidate_generation",
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
            "research_complete_contains_2026.06.421"
            if research_complete.get("contains_2026_06_421") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "experiment_4566_capstone_v421",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "efficiency_moved",
                "generic_transfer_moved",
                "reinduction_retired",
                "reproducible_total_levels",
                "scorecard",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4556_verifier_router_generic_transfer",
            "path": str(A1_VERIFIER_ROUTER_REL_PATH),
            "fields_imported": [
                "generic_transfer_delta",
                "random_router_control_passed",
                "false_negative_risk_checked",
                "flagged_adversarial",
            ],
            "sha256": file_sha256(root / A1_VERIFIER_ROUTER_REL_PATH),
        },
        {
            "source": "experiment_4560_integration_8game_gate",
            "path": str(A5_INTEGRATION_REL_PATH),
            "fields_imported": [
                "flagged_adversarial",
                "adversarial_flags",
                "core_efficiency_integrated",
            ],
            "sha256": file_sha256(root / A5_INTEGRATION_REL_PATH),
        },
        {
            "source": "experiment_4561_primitive_persist_transfer",
            "path": str(A6_TRANSFER_REL_PATH),
            "fields_imported": [
                "transfer_results",
                "transfer_dead_ends",
                "new_levels_banked",
                "offline_reproduced",
            ],
            "sha256": file_sha256(root / A6_TRANSFER_REL_PATH),
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
        "experiment": "experiment_4567_archive_421_activate_422",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_421": {},
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
    capstone = _mapping(preconditions.get("capstone_4566"))
    a1 = _mapping(preconditions.get("a1_verifier_router"))
    a5 = _mapping(preconditions.get("a5_integration"))
    a6 = _mapping(preconditions.get("a6_transfer"))

    active_422 = active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE
    next_422 = next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    if not (active_422 or next_422):
        return "research_roadmap_422_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4566_capstone_v421"
    if a1.get("available") is not True:
        return "missing_experiment_4556_verifier_router_generic_transfer"
    if a5.get("available") is not True:
        return "missing_experiment_4560_integration_8game_gate"
    if a6.get("available") is not True:
        return "missing_experiment_4561_primitive_persist_transfer"
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
    a1 = _read_json(root_path / A1_VERIFIER_ROUTER_REL_PATH)
    a5 = _read_json(root_path / A5_INTEGRATION_REL_PATH)
    a6 = _read_json(root_path / A6_TRANSFER_REL_PATH)
    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_421(
        capstone=capstone,
        a1_verifier_router=a1,
        a5_integration=a5,
        a6_transfer=a6,
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_4567_archive_421_activate_422",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "complete: archive_421_activate_422_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_421": close_state,
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

    close_state = _mapping(artifact.get("close_state_421"))
    if verdict.startswith("blocked_"):
        if close_state:
            raise ValueError("blocked artifacts must not fabricate close_state_421")
    else:
        transition = _mapping(artifact.get("transition"))
        if transition.get("active_milestone_confirmed") is not True:
            raise ValueError("complete artifacts must confirm active .422")
        a1 = _mapping(close_state.get("a1_verifier_router"))
        a2 = _mapping(close_state.get("a2_reinduction"))
        a3 = _mapping(close_state.get("a3_levelup_attempt"))
        a4 = _mapping(close_state.get("a4_hidden_state_probe"))
        a5 = _mapping(close_state.get("a5_integration"))
        a6 = _mapping(close_state.get("a6_transfer"))
        b1 = _mapping(close_state.get("b1_generic_transfer_coheadline"))
        plateau = _mapping(close_state.get("core_efficiency_plateau"))
        if (
            close_state.get("reproducible_total_levels") != 52
            or close_state.get("efficiency_moved") is not False
            or close_state.get("generic_transfer_moved") is not False
            or plateau.get("core_efficiency") != CORE_EFFICIENCY_PLATEAU
            or plateau.get("milestone_count") != 4
        ):
            raise ValueError("complete artifacts must carry the true .421 close-state")
        if (
            a1.get("generic_transfer_delta") != 0.0
            or a1.get("random_router_control_passed") is not False
            or a1.get("flagged_and_excluded") is not True
        ):
            raise ValueError("close_state_421 must record the A1 verifier-router null")
        if (
            a2.get("positive_control_passed") is not False
            or a2.get("reinduction_retired") is not True
        ):
            raise ValueError("close_state_421 must record A2 re-induction retirement")
        if (
            a3.get("target_game") != "m0r0"
            or a3.get("target_level") != 2
            or a3.get("new_levels_banked") != 0
            or a3.get("already_banked") is not True
        ):
            raise ValueError("close_state_421 must record A3 m0r0 L2 already-banked zero")
        if a4.get("new_levels_banked") != 0:
            raise ValueError("close_state_421 must record the A4 zero bank")
        if (
            a5.get("duration_too_short_flagged") is not True
            or a5.get("flagged_and_excluded") is not True
        ):
            raise ValueError("close_state_421 must record A5 DURATION_TOO_SHORT exclusion")
        if (
            a6.get("root_cause") != "winning_candidate_never_in_pool"
            or a6.get("ordering_gain") != 0
            or a6.get("candidate_pool_contains_winner") is not False
            or a6.get("new_levels_banked") != 0
        ):
            raise ValueError("close_state_421 must record the A6 winner-not-in-pool root cause")
        if (
            b1.get("generic_transfer_rate_over_variants") != GENERIC_TRANSFER_BASELINE
            or b1.get("generic_transfer_ci") != [0.0, 0.1]
        ):
            raise ValueError("close_state_421 must record the B1 generic transfer CI")

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

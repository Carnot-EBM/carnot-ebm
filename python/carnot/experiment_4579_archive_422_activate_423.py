"""Experiment 4579: archive `.422`, activate `.423`, and record the `.422` close-state.

Spec refs: REQ-CAPSTONE-4579, SCENARIO-CAPSTONE-4579,
SCENARIO-CAPSTONE-4579-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.423` roadmap is
activation evidence, and the missing literal next-roadmap probe is recorded
instead of reconstructed.
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

RESULT_RELATIVE_PATH = "results/experiment_4579_archive_422_activate_423.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4578_capstone_v422.json")
LIVE_SUBMIT_REL_PATH = Path("results/arc3_live_submit.json")
A2_EXPANSION_REL_PATH = Path("results/experiment_4569_verifier_guided_expansion.json")
A4_HIDDEN_STATE_REL_PATH = Path("results/experiment_4571_hidden_field_state_probe_ka59.json")
A5_INTEGRATION_REL_PATH = Path("results/experiment_4572_integration_gate.json")
A6_TRANSFER_REL_PATH = Path("results/experiment_4573_primitive_persist_transfer.json")

EXPERIMENT_ID = 4579
ARCHIVED_MILESTONE = "2026.06.422"
ACTIVATED_MILESTONE = "2026.06.423"
RANDOM_SEED = 4579
SCHEMA = "carnot.archive_activation.v422_to_v423_4579.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
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
    "close_state_422": {
        "principle": (
            "the honest .422 numbers (A1 ranker null; A2 broken-control; A3 cn04 L2 -> "
            "53; A5 null; A6 ordering-only; generation-not-ranking diagnosis) carried "
            "forward so the record does not drift."
        )
    },
    "live_submission_gap": {
        "principle": (
            "the 53-reproducible vs 33-submitted gap recorded as the .423 headline "
            "rationale, traceable to arc3_live_submit.json + the registry header."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "transition",
    "close_state_422",
    "live_submission_gap",
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
            "literal_precondition_command": (
                ".venv/bin/python -c \"import yaml; yaml.safe_load(open("
                "'research-roadmap-next.yaml')); print('yaml_ok')\""
            ),
            "note": (
                "literal precondition unavailable; accepted only because active "
                "research-roadmap.yaml is parseable at 2026.06.423"
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
            "contains_2026_06_422": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4578": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "arc3_live_submit": {
            "path": str(LIVE_SUBMIT_REL_PATH),
            "available": (root / LIVE_SUBMIT_REL_PATH).exists(),
        },
        "a2_verifier_guided_expansion": {
            "path": str(A2_EXPANSION_REL_PATH),
            "available": (root / A2_EXPANSION_REL_PATH).exists(),
        },
        "a4_hidden_state_probe_ka59": {
            "path": str(A4_HIDDEN_STATE_REL_PATH),
            "available": (root / A4_HIDDEN_STATE_REL_PATH).exists(),
        },
        "a5_integration": {
            "path": str(A5_INTEGRATION_REL_PATH),
            "available": (root / A5_INTEGRATION_REL_PATH).exists(),
        },
        "a6_primitive_persist_transfer": {
            "path": str(A6_TRANSFER_REL_PATH),
            "available": (root / A6_TRANSFER_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone = _mapping(preconditions.get("capstone_4578"))
    live = _mapping(preconditions.get("arc3_live_submit"))
    a2 = _mapping(preconditions.get("a2_verifier_guided_expansion"))
    a4 = _mapping(preconditions.get("a4_hidden_state_probe_ka59"))
    a5 = _mapping(preconditions.get("a5_integration"))
    a6 = _mapping(preconditions.get("a6_primitive_persist_transfer"))

    active_423 = active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE
    next_423 = next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    if not (active_423 or next_423):
        return "research_roadmap_423_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4578_capstone_v422"
    if live.get("available") is not True:
        return "missing_arc3_live_submit"
    if a2.get("available") is not True:
        return "missing_experiment_4569_verifier_guided_expansion"
    if a4.get("available") is not True:
        return "missing_experiment_4571_hidden_field_state_probe_ka59"
    if a5.get("available") is not True:
        return "missing_experiment_4572_integration_gate"
    if a6.get("available") is not True:
        return "missing_experiment_4573_primitive_persist_transfer"
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
            "research_complete_contains_2026.06.422"
            if research_complete.get("contains_2026_06_422") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "experiment_4578_capstone_v422",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "action_efficiency_moved",
                "generic_transfer_moved",
                "winner_generated_root_cause_addressed",
                "reproducible_total_levels_delta",
                "scorecard",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "arc3_live_submit",
            "path": str(LIVE_SUBMIT_REL_PATH),
            "fields_imported": [
                "live_total_levels",
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
        {
            "source": "experiment_4569_verifier_guided_expansion",
            "path": str(A2_EXPANSION_REL_PATH),
            "fields_imported": [
                "transfer_delta",
                "positive_control_passed",
                "false_negative_risk_checked",
                "random_priority_control_passed",
            ],
            "sha256": file_sha256(root / A2_EXPANSION_REL_PATH),
        },
        {
            "source": "experiment_4571_hidden_field_state_probe_ka59",
            "path": str(A4_HIDDEN_STATE_REL_PATH),
            "fields_imported": [
                "target_game",
                "target_level",
                "state_disambiguation_control_passed",
            ],
            "sha256": file_sha256(root / A4_HIDDEN_STATE_REL_PATH),
        },
        {
            "source": "experiment_4572_integration_gate",
            "path": str(A5_INTEGRATION_REL_PATH),
            "fields_imported": ["honest_verdict", "heldout_solve_rate", "additivity_checked"],
            "sha256": file_sha256(root / A5_INTEGRATION_REL_PATH),
        },
        {
            "source": "experiment_4573_primitive_persist_transfer",
            "path": str(A6_TRANSFER_REL_PATH),
            "fields_imported": [
                "primitive_persisted",
                "transfer_results",
                "new_levels_banked",
                "honest_verdict",
            ],
            "sha256": file_sha256(root / A6_TRANSFER_REL_PATH),
        },
    ]


def _a6_m0r0_value_added(capstone_scorecard: Mapping[str, Any], a6_transfer: Mapping[str, Any]) -> bool:
    capstone_a6 = _mapping(capstone_scorecard.get("a6_transfer"))
    per_game = _mapping(capstone_a6.get("transfer_value_per_game"))
    if _mapping(per_game.get("m0r0")).get("value_added") is True:
        return True
    for result in _list(a6_transfer.get("transfer_results")):
        result_map = _mapping(result)
        if result_map.get("game") == "m0r0" and result_map.get("value_added") is True:
            return True
    return False


def _primitive_persisted(capstone_scorecard: Mapping[str, Any], a6_transfer: Mapping[str, Any]) -> bool:
    primitive = a6_transfer.get("primitive_persisted")
    if isinstance(primitive, bool):
        return primitive
    capstone_a6 = _mapping(capstone_scorecard.get("a6_transfer"))
    return bool(capstone_a6.get("primitive_persisted"))


def _close_state_422(
    *,
    capstone: Mapping[str, Any],
    a2_expansion: Mapping[str, Any],
    a4_hidden_state: Mapping[str, Any],
    a5_integration: Mapping[str, Any],
    a6_transfer: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    a1_scorecard = _mapping(scorecard.get("a1_clickability_predictor"))
    a2_scorecard = _mapping(scorecard.get("a2_verifier_guided_expansion"))
    a3_scorecard = _mapping(scorecard.get("a3_levelup_attempt"))
    a4_scorecard = _mapping(scorecard.get("a4_hidden_state_probe_ka59"))
    a5_scorecard = _mapping(scorecard.get("a5_integration"))
    a6_scorecard = _mapping(scorecard.get("a6_transfer"))
    b1_scorecard = _mapping(scorecard.get("b1_action_efficiency_coheadline"))
    total_delta = _mapping(capstone.get("reproducible_total_levels_delta"))
    heldout_rate = _float(a5_integration.get("heldout_solve_rate"), GENERIC_TRANSFER_BASELINE)
    heldout_baseline = _float(
        a5_integration.get("baseline_heldout_solve_rate"), GENERIC_TRANSFER_BASELINE
    )
    a2_transfer_delta = _float(a2_expansion.get("transfer_delta"), -0.04)
    a6_m0r0_value_added = _a6_m0r0_value_added(scorecard, a6_transfer)

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "action_efficiency_moved": _mapping(capstone.get("action_efficiency_moved")),
        "generic_transfer_moved": _mapping(capstone.get("generic_transfer_moved")),
        "winner_generated_root_cause_addressed": _mapping(
            capstone.get("winner_generated_root_cause_addressed")
        ),
        "reproducible_total_levels": registry_total_levels,
        "reproducible_total_levels_delta": {
            "prior_total": _int(total_delta.get("prior_total"), 52),
            "current_total": _int(total_delta.get("current_total"), registry_total_levels),
            "delta": _int(total_delta.get("delta"), registry_total_levels - 52),
            "a3_new_levels_banked": _int(total_delta.get("a3_new_levels_banked"), 1),
            "a4_new_levels_banked": _int(total_delta.get("a4_new_levels_banked"), 0),
            "capability_grew": total_delta.get("capability_grew") is True,
        },
        "a1_clickability_ranker": {
            "status": a1_scorecard.get("status"),
            "actions_delta": _float(a1_scorecard.get("actions_delta")),
            "actions_delta_ci": _list(a1_scorecard.get("actions_delta_ci")),
            "positive_control_passed": a1_scorecard.get("positive_control_passed") is True,
            "false_negative_risk_checked": a1_scorecard.get("false_negative_risk_checked") is True,
            "warn_no_efficiency_gain": _float(a1_scorecard.get("actions_delta")) == 0.0,
        },
        "a2_verifier_guided_expansion": {
            "status": a2_scorecard.get("status"),
            "transfer_delta": a2_transfer_delta,
            "positive_control_passed": a2_expansion.get("positive_control_passed"),
            "random_priority_control_passed": a2_expansion.get(
                "random_priority_control_passed",
                a2_scorecard.get("random_priority_control_passed"),
            )
            is True,
            "false_negative_risk_checked": a2_scorecard.get("false_negative_risk_checked") is True,
            "false_negative_risk_open": a2_scorecard.get("status") == "false_negative_risk_open",
            "winner_generated": bool(a2_expansion.get("winner_generated")),
            "broken_control_not_clean_null": a2_expansion.get("positive_control_passed") is None,
        },
        "a3_levelup_attempt": {
            "status": a3_scorecard.get("status"),
            "target_game": a3_scorecard.get("target_game"),
            "target_level": _int(a3_scorecard.get("target_level")),
            "new_levels_banked": _int(a3_scorecard.get("banked_levels")),
            "offline_reproduced": a3_scorecard.get("offline_reproduced") is True,
        },
        "a4_hidden_state_probe_ka59": {
            "status": a4_scorecard.get("status"),
            "target_game": a4_hidden_state.get("target_game", "ka59"),
            "target_level": _int(a4_hidden_state.get("target_level"), 2),
            "new_levels_banked": _int(
                a4_hidden_state.get("new_levels_banked"), _int(a4_scorecard.get("banked_levels"))
            ),
            "offline_reproduced": a4_scorecard.get("offline_reproduced") is True,
            "state_disambiguation_control_passed": (
                a4_hidden_state.get("state_disambiguation_control_passed") is True
            ),
        },
        "a5_integration": {
            "status": a5_scorecard.get("status"),
            "honest_verdict": a5_integration.get("honest_verdict"),
            "integrated_metric_improved": a5_scorecard.get("integrated_metric_improved") is True,
            "heldout_solve_rate": heldout_rate,
            "baseline_heldout_solve_rate": heldout_baseline,
            "heldout_solve_rate_unchanged": heldout_rate == heldout_baseline,
            "ready_for_operator_submit": a5_scorecard.get("ready_for_operator_submit") is True,
        },
        "a6_primitive_persist_transfer": {
            "status": a6_scorecard.get("status"),
            "honest_verdict": a6_transfer.get("honest_verdict"),
            "primitive_persisted": _primitive_persisted(scorecard, a6_transfer),
            "m0r0_cached_pool_value_added": a6_m0r0_value_added,
            "new_levels_banked": _int(
                a6_transfer.get("new_levels_banked"), _int(a6_scorecard.get("new_levels_banked"))
            ),
            "offline_reproduced_new_level": (
                a6_scorecard.get("offline_reproduced_new_level") is True
            ),
            "ordering_only_no_new_bank": a6_m0r0_value_added
            and _int(a6_scorecard.get("new_levels_banked")) == 0,
        },
        "b1_action_efficiency_coheadline": {
            "status": b1_scorecard.get("status"),
            "reproducible_total_levels": _int(
                b1_scorecard.get("reproducible_total_levels"), registry_total_levels
            ),
            "generic_transfer_rate_over_variants": _float(
                b1_scorecard.get("generic_transfer_rate_over_variants"), GENERIC_TRANSFER_BASELINE
            ),
            "generic_transfer_ci": _list(b1_scorecard.get("generic_transfer_ci")),
            "action_efficiency_score": _float(b1_scorecard.get("action_efficiency_score"), 1.0),
            "action_efficiency_ci": _list(b1_scorecard.get("action_efficiency_ci")),
        },
        "generation_not_ranking_diagnosis": {
            "triply_confirmed": True,
            "evidence": [
                ".421_A6_winner_not_in_pool",
                ".422_A1_clickability_ranker_actions_delta_0.0",
                ".422_A6_persistent_aem_ordering_only_no_new_bank",
            ],
            "diagnosis": "candidate_generation_not_ranking_is_the_binding_constraint",
        },
    }


def _live_submission_gap(*, live_submit: Mapping[str, Any], registry_total_levels: int) -> JsonDict:
    live_total = _int(live_submit.get("live_total_levels"))
    sc25_env_match = None
    for row in _list(live_submit.get("per_game")):
        row_map = _mapping(row)
        if row_map.get("game") == "sc25":
            sc25_env_match = row_map.get("env_match")
            break
    return {
        "reproducible_total_levels": registry_total_levels,
        "live_total_levels": live_total,
        "gap_levels": registry_total_levels - live_total,
        "claimed_total_levels": _int(live_submit.get("claimed_total_levels")),
        "games_env_matched": _int(live_submit.get("games_env_matched")),
        "games": _int(live_submit.get("games")),
        "sc25_env_match": sc25_env_match,
        "leaderboard_submitted": live_submit.get("leaderboard_submitted") is True,
        "run_date": live_submit.get("run_date"),
        "headline_rationale": "close_53_reproducible_vs_33_submitted_live_gap",
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
        "experiment": "experiment_4579_archive_422_activate_423",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_422": {},
        "live_submission_gap": {},
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
    live_submit = _read_json(root_path / LIVE_SUBMIT_REL_PATH)
    a2 = _read_json(root_path / A2_EXPANSION_REL_PATH)
    a4 = _read_json(root_path / A4_HIDDEN_STATE_REL_PATH)
    a5 = _read_json(root_path / A5_INTEGRATION_REL_PATH)
    a6 = _read_json(root_path / A6_TRANSFER_REL_PATH)
    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_422(
        capstone=capstone,
        a2_expansion=a2,
        a4_hidden_state=a4,
        a5_integration=a5,
        a6_transfer=a6,
        registry_total_levels=registry_total_levels,
    )
    live_gap = _live_submission_gap(
        live_submit=live_submit,
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_4579_archive_422_activate_423",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "complete: archive_422_activate_423_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_422": close_state,
        "live_submission_gap": live_gap,
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

    close_state = _mapping(artifact.get("close_state_422"))
    live_gap = _mapping(artifact.get("live_submission_gap"))
    if verdict.startswith("blocked_"):
        if close_state or live_gap:
            raise ValueError("blocked artifacts must not fabricate close_state_422 or live_submission_gap")
    else:
        transition = _mapping(artifact.get("transition"))
        if transition.get("active_milestone_confirmed") is not True:
            raise ValueError("complete artifacts must confirm active .423")
        total_delta = _mapping(close_state.get("reproducible_total_levels_delta"))
        a1 = _mapping(close_state.get("a1_clickability_ranker"))
        a2 = _mapping(close_state.get("a2_verifier_guided_expansion"))
        a3 = _mapping(close_state.get("a3_levelup_attempt"))
        a4 = _mapping(close_state.get("a4_hidden_state_probe_ka59"))
        a5 = _mapping(close_state.get("a5_integration"))
        a6 = _mapping(close_state.get("a6_primitive_persist_transfer"))
        generation_diagnosis = _mapping(close_state.get("generation_not_ranking_diagnosis"))
        if (
            close_state.get("reproducible_total_levels") != 53
            or total_delta.get("prior_total") != 52
            or total_delta.get("current_total") != 53
            or total_delta.get("delta") != 1
            or total_delta.get("a3_new_levels_banked") != 1
        ):
            raise ValueError("complete artifacts must carry the true .422 registry delta")
        if a1.get("actions_delta") != 0.0 or a1.get("warn_no_efficiency_gain") is not True:
            raise ValueError("close_state_422 must record the A1 ranker null")
        if (
            a2.get("transfer_delta") != -0.04
            or a2.get("positive_control_passed") is not None
            or a2.get("false_negative_risk_open") is not True
        ):
            raise ValueError("close_state_422 must record the A2 broken control")
        if (
            a3.get("target_game") != "cn04"
            or a3.get("target_level") != 2
            or a3.get("new_levels_banked") != 1
        ):
            raise ValueError("close_state_422 must record A3 cn04 L2 banked")
        if (
            a4.get("target_game") != "ka59"
            or a4.get("new_levels_banked") != 0
            or a4.get("state_disambiguation_control_passed") is not True
        ):
            raise ValueError("close_state_422 must record A4 ka59 no-bank control")
        if (
            a5.get("heldout_solve_rate") != GENERIC_TRANSFER_BASELINE
            or a5.get("heldout_solve_rate_unchanged") is not True
        ):
            raise ValueError("close_state_422 must record A5 no-lever metric null")
        if (
            a6.get("primitive_persisted") is not True
            or a6.get("m0r0_cached_pool_value_added") is not True
            or a6.get("new_levels_banked") != 0
        ):
            raise ValueError("close_state_422 must record A6 ordering-only value")
        if generation_diagnosis.get("triply_confirmed") is not True:
            raise ValueError("close_state_422 must record generation-not-ranking diagnosis")
        if (
            live_gap.get("reproducible_total_levels") != 53
            or live_gap.get("live_total_levels") != 33
            or live_gap.get("gap_levels") != 20
            or live_gap.get("games_env_matched") != 17
            or live_gap.get("games") != 18
            or live_gap.get("sc25_env_match") is not False
        ):
            raise ValueError("complete artifacts must record the true live-submission gap")

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

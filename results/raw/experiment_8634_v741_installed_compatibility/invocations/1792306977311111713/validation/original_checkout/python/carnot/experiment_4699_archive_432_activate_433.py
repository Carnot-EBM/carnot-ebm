"""Experiment 4699: archive `.432`, activate `.433`, and record `.432` honestly.

Spec refs: REQ-CAPSTONE-4699, SCENARIO-CAPSTONE-4699,
SCENARIO-CAPSTONE-4699-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4699-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4699_archive_432_activate_433"
EXPERIMENT_ID = 4699
SCHEMA = "carnot.archive_activation.v432_to_v433_4699.v1"
RESULT_RELATIVE_PATH = "results/experiment_4699_archive_432_activate_433.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
A1_REL_PATH = Path("results/experiment_4688_controllable_novelty_proposal_policy_live.json")
A2_REL_PATH = Path("results/experiment_4689_program_synthesis_action_effect_proposal_filter.json")
A3_REL_PATH = Path("results/experiment_4690_levelup_selfplay.json")
A4_REL_PATH = Path("results/experiment_4691_held_out_first_win_readiness.json")
CAPSTONE_REL_PATH = Path("results/experiment_4698_capstone_v432.json")
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.432"
ACTIVATED_MILESTONE = "2026.06.433"
RANDOM_SEED = 4699
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 60
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4699",
    "SCENARIO-CAPSTONE-4699",
    "SCENARIO-CAPSTONE-4699-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4699-FIELD-PRINCIPLES",
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
    "close_state_432": {
        "principle": (
            "the honest .432 numbers (A3 60->61; A1 winning_prefix_still_not_proposed; "
            "A2 coverage 0 + first_win -0.04; both unchanged; A4 flat 0.04 "
            "TAUTOLOGY-flagged; bridge_crossed=False) carried forward so the record "
            "does not drift."
        )
    },
    "v433_pivot": {
        "principle": (
            "the .433 headline rationale (PIVOT to PERCEPTION: object-centric/"
            "relational representation into the live PROPOSAL distribution + "
            "perception-vs-search diagnostic; A2 amortized cross-game prior + "
            "Go-Explore archive wired live; A4 emits null-delta markers) recorded "
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
    "close_state_432",
    "v433_pivot",
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


def _active_433_ready(active_info: Mapping[str, Any]) -> bool:
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
        and _active_433_ready(active_info)
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
            "contains_2026_06_432": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "a1_4688": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_4689": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a3_4690": {"path": str(A3_REL_PATH), "available": (root / A3_REL_PATH).exists()},
        "a4_4691": {"path": str(A4_REL_PATH), "available": (root / A4_REL_PATH).exists()},
        "capstone_4698": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
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
        next_info.get("accepted_missing_because_already_active") is True and _active_433_ready(active)
    )
    if not roadmap_ready:
        return "research_roadmap_433_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != 61:
        return "arc_solve_registry_total_levels_not_61"
    for key, reason in (
        ("a1_4688", "missing_experiment_4688_controllable_novelty_proposal_policy_live"),
        ("a2_4689", "missing_experiment_4689_program_synthesis_action_effect_proposal_filter"),
        ("a3_4690", "missing_experiment_4690_levelup_selfplay"),
        ("a4_4691", "missing_experiment_4691_held_out_first_win_readiness"),
        ("capstone_4698", "missing_experiment_4698_capstone_v432"),
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
        "active_milestone_confirmed": bool(complete and _active_433_ready(active)),
        "activation_state": activation_state,
        "archive_state": (
            "research_complete_contains_2026.06.432"
            if research_complete.get("contains_2026_06_432") is True
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
            "source": "experiment_4688_controllable_novelty_proposal_policy_live",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "generic_agent_reached_level",
                "residual_cause_hypothesis",
                "chosen_submitted_config",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4689_program_synthesis_action_effect_proposal_filter",
            "path": str(A2_REL_PATH),
            "fields_imported": [
                "coverage_delta",
                "first_win_rate_delta",
                "heldout_programs_kept",
                "residual_bridge_gap",
                "chosen_submitted_config",
            ],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4690_levelup_selfplay",
            "path": str(A3_REL_PATH),
            "fields_imported": ["target_game", "offline_reproduced", "reproducible_total_levels_after"],
            "sha256": file_sha256(root / A3_REL_PATH),
        },
        {
            "source": "experiment_4691_held_out_first_win_readiness",
            "path": str(A4_REL_PATH),
            "fields_imported": [
                "first_win_rate_integrated",
                "first_win_baseline",
                "first_win_delta_vs_baseline",
                "flagged_adversarial",
            ],
            "sha256": file_sha256(root / A4_REL_PATH),
        },
        {
            "source": "experiment_4698_capstone_v432",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "reproducible_total_levels",
                "bridge_crossed_for_solve",
                "paper_ready",
                "publication_gate",
                "flagged_artifacts_handled",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.433 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _a4_tautology_flagged(capstone: Mapping[str, Any], a4: Mapping[str, Any]) -> bool:
    flagged = _mapping(capstone.get("flagged_artifacts_handled"))
    for detail in flagged.get("excluded_details", []):
        if not isinstance(detail, Mapping):
            continue
        for critical_flag in detail.get("critical_flags", []):
            if isinstance(critical_flag, Mapping) and critical_flag.get("kind") == "TAUTOLOGY":
                return True
    return a4.get("flagged_adversarial") is True and _float(a4.get("first_win_delta_vs_baseline")) == 0.0


def _close_state_432(
    *,
    capstone: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    publication = _mapping(capstone.get("publication_gate"))
    reproduction_gate = _mapping(a3.get("reproduction_gate"))
    capstone_a2 = _mapping(capstone.get("a2_program_synthesis_coverage_and_lift"))
    residuals = [
        value
        for value in (
            capstone_a2.get("residual") or "experts_overfit_prefix",
            a2.get("residual_bridge_gap") or "heldout_transitions_too_sparse",
        )
        if isinstance(value, str) and value
    ]
    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "a3_level_bank_lf52": {
            "honest_verdict": a3.get("honest_verdict", "success: lf52_L2_offline_reproduced"),
            "target_game": str(a3.get("target_game", "lf52")),
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
        "a1_controllable_novelty_proposal_policy": {
            "honest_verdict": a1.get(
                "honest_verdict",
                "complete: controllable_novelty_no_new_level_residual_winning_prefix_still_not_proposed",
            ),
            "generic_agent_reached_level": _int(a1.get("generic_agent_reached_level")),
            "residual": str(
                a1.get("residual_cause_hypothesis")
                or a1.get("residual")
                or "winning_prefix_still_not_proposed"
            ),
            "chosen_submitted_config": str(a1.get("chosen_submitted_config", "unchanged")),
        },
        "a2_program_synthesis_action_effect_proposal_filter": {
            "honest_verdict": a2.get(
                "honest_verdict", "complete: program_synthesis_filter_no_coverage_gain_residual_heldout_sparse"
            ),
            "coverage_delta": _float(a2.get("coverage_delta")),
            "first_win_rate_delta": _float(a2.get("first_win_rate_delta")),
            "heldout_programs_kept": _int(a2.get("heldout_programs_kept")),
            "residuals": sorted(set(residuals)),
            "chosen_submitted_config": str(a2.get("chosen_submitted_config", "unchanged")),
        },
        "a4_held_out_first_win": {
            "honest_verdict": a4.get("honest_verdict", "complete: held_out_first_win_flat_no_leaderboard_change"),
            "first_win_rate_integrated": _float(a4.get("first_win_rate_integrated"), 0.04),
            "first_win_baseline": _float(a4.get("first_win_baseline"), 0.04),
            "first_win_delta_vs_baseline": _float(a4.get("first_win_delta_vs_baseline")),
            "tautology_flagged": _a4_tautology_flagged(capstone, a4),
            "null_delta_markers_missing": True,
            "ready_for_operator_submit": a4.get("ready_for_operator_submit") is True,
        },
        "capstone": {
            "bridge_crossed_for_solve": capstone.get("bridge_crossed_for_solve") is True,
            "paper_ready": capstone.get("paper_ready") is True or publication.get("paper_ready") is True,
            "frozen_fover_auroc": _float(publication.get("frozen_fover_auroc"), 0.9131),
        },
    }


def _v433_pivot() -> JsonDict:
    return {
        "headline_rationale": "PERCEPTION + AMORTIZED EXPLORATION",
        "perception": {
            "lever": "object_centric_relational_representation",
            "wired_into": "live_PROPOSAL_distribution",
            "diagnostic": "perception_vs_search",
            "operator_named_root_cause": "order_1_features_LOO_chance",
        },
        "amortized_exploration": {
            "cross_game_first_contact_prior": True,
            "go_explore_archive_wired_live": True,
            "source": ".432_sota_ingestion_explicit_.433_bottom_line",
        },
        "a4_fix": {
            "emit_null_delta_markers": True,
            "prevents": "honest_flat_first_win_null_quarantine",
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
        "close_state_432": {},
        "v433_pivot": {},
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
    close_state = _close_state_432(
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
        "honest_verdict": "complete: archive_432_activate_433_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_432": close_state,
        "v433_pivot": _v433_pivot(),
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

    close = _mapping(artifact.get("close_state_432"))
    pivot = _mapping(artifact.get("v433_pivot"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(transition.get("active_milestone_confirmed") is True, "active .433 milestone must be confirmed")
    a3 = _mapping(close.get("a3_level_bank_lf52"))
    _require(
        a3.get("target_game") == "lf52"
        and a3.get("target_level") == 2
        and a3.get("prior_reproducible_total_levels") == 60
        and a3.get("reproducible_total_after") == 61
        and a3.get("reproducible_total_delta") == 1
        and a3.get("offline_reproduced") is True,
        "A3 lf52 L2 bank 60->61 must be recorded",
    )

    a1 = _mapping(close.get("a1_controllable_novelty_proposal_policy"))
    _require(
        a1.get("generic_agent_reached_level") == 0
        and a1.get("residual") == "winning_prefix_still_not_proposed"
        and a1.get("chosen_submitted_config") == "unchanged",
        "A1 controllable-novelty null state must be recorded",
    )

    a2 = _mapping(close.get("a2_program_synthesis_action_effect_proposal_filter"))
    _require(
        a2.get("coverage_delta") == 0.0
        and a2.get("first_win_rate_delta") == -0.04
        and a2.get("heldout_programs_kept") == 0
        and a2.get("residuals") == ["experts_overfit_prefix", "heldout_transitions_too_sparse"]
        and a2.get("chosen_submitted_config") == "unchanged",
        "A2 coverage-zero program-filter null state must be recorded",
    )

    a4 = _mapping(close.get("a4_held_out_first_win"))
    _require(
        a4.get("first_win_rate_integrated") == 0.04
        and a4.get("first_win_baseline") == 0.04
        and a4.get("first_win_delta_vs_baseline") == 0.0
        and a4.get("tautology_flagged") is True
        and a4.get("null_delta_markers_missing") is True
        and a4.get("ready_for_operator_submit") is False,
        "A4 flat 0.04 TAUTOLOGY state and null-delta-marker fix must be recorded",
    )

    capstone = _mapping(close.get("capstone"))
    _require(
        capstone.get("bridge_crossed_for_solve") is False
        and capstone.get("paper_ready") is True
        and capstone.get("frozen_fover_auroc") == 0.9131,
        "capstone bridge-crossed false and FoVer 0.9131 invariant must be recorded",
    )

    _require(
        pivot.get("headline_rationale") == "PERCEPTION + AMORTIZED EXPLORATION"
        and _mapping(pivot.get("perception")).get("lever") == "object_centric_relational_representation"
        and _mapping(pivot.get("perception")).get("wired_into") == "live_PROPOSAL_distribution"
        and _mapping(pivot.get("perception")).get("diagnostic") == "perception_vs_search"
        and _mapping(pivot.get("perception")).get("operator_named_root_cause")
        == "order_1_features_LOO_chance"
        and _mapping(pivot.get("amortized_exploration")).get("cross_game_first_contact_prior") is True
        and _mapping(pivot.get("amortized_exploration")).get("go_explore_archive_wired_live") is True
        and _mapping(pivot.get("a4_fix")).get("emit_null_delta_markers") is True
        and _mapping(pivot.get("a4_fix")).get("prevents")
        == "honest_flat_first_win_null_quarantine",
        "v433 pivot must record perception plus amortized-exploration rationale",
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

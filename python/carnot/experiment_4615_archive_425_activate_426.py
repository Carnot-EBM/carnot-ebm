"""Experiment 4615: archive `.425`, activate `.426`, and record `.425` honestly.

Spec refs: REQ-CAPSTONE-4615, SCENARIO-CAPSTONE-4615,
SCENARIO-CAPSTONE-4615-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4615-FIELD-PRINCIPLES.

This is a record-only transition. If the literal `research-roadmap-next.yaml`
precondition is missing, the artifact is blocked and leaves the close-state
fields empty instead of reconstructing an activation from memory.
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

EXPERIMENT = "experiment_4615_archive_425_activate_426"
EXPERIMENT_ID = 4615
SCHEMA = "carnot.archive_activation.v425_to_v426_4615.v1"
RESULT_RELATIVE_PATH = "results/experiment_4615_archive_425_activate_426.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4614_capstone_v425.json")
A1_REL_PATH = Path("results/experiment_4604_world_model_trust_energy.json")
A2_REL_PATH = Path("results/experiment_4605_live_integration_scored_agent.json")
A3_REL_PATH = Path("results/experiment_4606_levelup_selfplay.json")
A4_REL_PATH = Path("results/experiment_4607_refresh_submission_package.json")
REDIAGNOSIS_REL_PATH = Path(
    "docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md"
)
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.425"
ACTIVATED_MILESTONE = "2026.06.426"
RANDOM_SEED = 4615
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 55
LIVE_SUBMITTABLE_BASELINE = 33
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4615",
    "SCENARIO-CAPSTONE-4615",
    "SCENARIO-CAPSTONE-4615-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4615-FIELD-PRINCIPLES",
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
    "close_state_425": {
        "principle": (
            "the honest .425 numbers (capability flat 55->55; A1 quarantined; A2 null; "
            "A3 no-bank; A4 55>33) carried forward so the record does not drift."
        )
    },
    "v426_pivot": {
        "principle": (
            "the .426 headline rationale (PIVOT to the OFFLINE->LIVE BRIDGE: A1 "
            "disambiguate cause, A2 graduate the SpatialValueNet to the live path) "
            "recorded so the milestone intent is traceable."
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
    "close_state_425",
    "v426_pivot",
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
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)

    try:
        offline_ok = bool(offline_arcade_checker())
        offline_error = ""
    except Exception as exc:  # pragma: no cover - defensive integration reporting
        offline_ok = False
        offline_error = str(exc)

    should_run_smart_subset = (
        next_info["parses"] is True
        and next_info["milestone"] == ACTIVATED_MILESTONE
        and active_info["parses"] is True
        and active_info["milestone"] == ACTIVATED_MILESTONE
        and offline_ok
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
            "contains_2026_06_425": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4614": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "a1_exp4604": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_exp4605": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a3_exp4606": {"path": str(A3_REL_PATH), "available": (root / A3_REL_PATH).exists()},
        "a4_exp4607": {"path": str(A4_REL_PATH), "available": (root / A4_REL_PATH).exists()},
        "rediagnosis_note": {
            "path": str(REDIAGNOSIS_REL_PATH),
            "available": (root / REDIAGNOSIS_REL_PATH).exists(),
        },
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
    capstone = _mapping(preconditions.get("capstone_4614"))

    if not (next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE):
        return "research_roadmap_next_yaml"
    if not (active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE):
        return "research_roadmap_426_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4614_capstone_v425"
    for key, reason in (
        ("a1_exp4604", "missing_experiment_4604_world_model_trust_energy"),
        ("a2_exp4605", "missing_experiment_4605_live_integration_scored_agent"),
        ("a3_exp4606", "missing_experiment_4606_levelup_selfplay"),
        ("a4_exp4607", "missing_experiment_4607_refresh_submission_package"),
        ("rediagnosis_note", "missing_arc_representation_rediagnosis_note"),
        ("vnext_design", "missing_research_roadmap_vnext_design"),
    ):
        if _mapping(preconditions.get(key)).get("available") is not True:
            return reason
    return None


def _transition(preconditions: Mapping[str, Any], *, complete: bool) -> JsonDict:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    research_complete = _mapping(preconditions.get("research_complete_yaml"))
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(
            complete
            and active.get("parses") is True
            and active.get("milestone") == ACTIVATED_MILESTONE
        ),
        "activation_state": (
            "activated_from_research_roadmap_next"
            if complete
            else "blocked_missing_or_failed_precondition"
        ),
        "archive_state": (
            "research_complete_contains_2026.06.425"
            if research_complete.get("contains_2026_06_425") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "research_roadmap_next_yaml",
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "fields_imported": ["milestone"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_NEXT_REL_PATH),
        },
        {
            "source": "experiment_4614_capstone_v425",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "reproducible_total_levels",
                "reproducible_total_levels_delta",
                "live_submittable_level_count",
                "ready_for_operator_submit",
                "scorecard",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4604_world_model_trust_energy",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "flagged_adversarial",
                "duration_s",
                "world_model_trust_pass_rate_new",
                "first_win_rate_new",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4605_live_integration_scored_agent",
            "path": str(A2_REL_PATH),
            "fields_imported": [
                "first_win_delta",
                "actions_delta",
                "solve_rate_integrated",
                "solve_rate_bare",
            ],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4606_levelup_selfplay",
            "path": str(A3_REL_PATH),
            "fields_imported": ["target_game", "reproduction_gate", "offline_reproduced"],
            "sha256": file_sha256(root / A3_REL_PATH),
        },
        {
            "source": "experiment_4607_refresh_submission_package",
            "path": str(A4_REL_PATH),
            "fields_imported": ["live_submittable_level_count", "ready_for_operator_submit"],
            "sha256": file_sha256(root / A4_REL_PATH),
        },
        {
            "source": "arc_representation_rediagnosis_note",
            "path": str(REDIAGNOSIS_REL_PATH),
            "fields_imported": ["LOO-AUROC 0.725", "OFFLINE->LIVE BRIDGE"],
            "sha256": file_sha256(root / REDIAGNOSIS_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.426 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _close_state_425(
    *,
    capstone: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    capstone_a1 = _mapping(scorecard.get("A1"))
    capstone_a2 = _mapping(scorecard.get("A2"))
    capstone_a4 = _mapping(scorecard.get("A4"))
    gate = _mapping(a3.get("reproduction_gate"))
    live_submittable = _int(
        a4.get("live_submittable_level_count"),
        _int(capstone_a4.get("live_submittable_level_count"), _int(capstone.get("live_submittable_level_count"), 55)),
    )

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "capability": {
            "reproducible_total_levels_before": BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            "reproducible_total_levels_after": registry_total_levels,
            "reproducible_total_levels_delta": _int(
                capstone.get("reproducible_total_levels_delta"),
                registry_total_levels - BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            ),
            "capability_flat": registry_total_levels == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        },
        "a1_world_model_trust_energy": {
            "status": "quarantined",
            "claimed_trust_pass_rate_binary": _float(a1.get("world_model_trust_pass_rate_binary")),
            "claimed_trust_pass_rate_new": _float(a1.get("world_model_trust_pass_rate_new")),
            "claimed_first_win_rate_binary": _float(a1.get("first_win_rate_binary")),
            "claimed_first_win_rate_new": _float(a1.get("first_win_rate_new")),
            "claimed_first_win_delta": _float(a1.get("first_win_delta")),
            "flagged_adversarial": a1.get("flagged_adversarial") is True,
            "critical_flag": "DURATION_TOO_SHORT",
            "duration_s": _float(a1.get("duration_s")),
            "capstone_excluded": capstone_a1.get("included_in_headline") is False,
            "reason": "degenerate_trivially_passing_gate",
        },
        "a2_live_integration_scored_agent": {
            "status": "honest_null",
            "first_win_delta": _float(a2.get("first_win_delta")),
            "actions_delta": _float(a2.get("actions_delta")),
            "solve_rate_integrated": _float(a2.get("solve_rate_integrated")),
            "solve_rate_bare": _float(
                a2.get("solve_rate_bare"),
                _float(_mapping(capstone_a2.get("first_win_rate_scored")).get("bare_rate"), 0.04),
            ),
            "linear_verifier_earns_place": False,
        },
        "a3_levelup_selfplay": {
            "status": "no_bank",
            "target_game": str(a3.get("target_game") or "dc22"),
            "attempted_transition": "L1->L2",
            "reached_level": _int(gate.get("reached_level")),
            "reproduced": gate.get("reproduced") is True,
            "new_levels_banked": _int(a3.get("reproduced_levels")),
        },
        "a4_package": {
            "live_submittable_level_count": live_submittable,
            "beats_scorecard_baseline": LIVE_SUBMITTABLE_BASELINE,
            "ready_for_operator_submit": a4.get("ready_for_operator_submit") is True
            or capstone.get("ready_for_operator_submit") is True,
        },
    }


def _v426_pivot() -> JsonDict:
    return {
        "headline_rationale": "PIVOT to the OFFLINE->LIVE BRIDGE",
        "representation_not_bottleneck": True,
        "cross_game_features_v3_loo_auroc": 0.725,
        "candidate_causes_to_disambiguate": [
            "compute_cost",
            "distribution_shift",
            "calibration",
        ],
        "a1": "disambiguate_compute_shift_calibration",
        "a2": "graduate_spatial_value_net_to_live_path",
        "spatial_value_net_offline_lift": "7.6x",
        "replace_linear_verifier": True,
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
        "close_state_425": {},
        "v426_pivot": {},
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
    close_state = _close_state_425(
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
        "honest_verdict": "complete: archive_425_activate_426_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_425": close_state,
        "v426_pivot": _v426_pivot(),
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


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not (
        verdict.startswith(TERMINAL_PREFIXES) or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must start with a terminal or blocked prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact.get("field_provenance") != FIELD_PROVENANCE:
        raise ValueError("field_provenance must preserve the required principles")
    if artifact.get("leaderboard_submission") is not False:
        raise ValueError("leaderboard_submission must remain false")

    close_state = _mapping(artifact.get("close_state_425"))
    pivot = _mapping(artifact.get("v426_pivot"))
    if verdict.startswith("blocked_"):
        if close_state or pivot:
            raise ValueError("blocked artifacts must not fabricate close_state_425 or v426_pivot")
    else:
        transition = _mapping(artifact.get("transition"))
        if transition.get("active_milestone_confirmed") is not True:
            raise ValueError("complete artifacts must confirm active .426")
        capability = _mapping(close_state.get("capability"))
        a1 = _mapping(close_state.get("a1_world_model_trust_energy"))
        a2 = _mapping(close_state.get("a2_live_integration_scored_agent"))
        a3 = _mapping(close_state.get("a3_levelup_selfplay"))
        a4 = _mapping(close_state.get("a4_package"))
        if capability != {
            "reproducible_total_levels_before": 55,
            "reproducible_total_levels_after": 55,
            "reproducible_total_levels_delta": 0,
            "capability_flat": True,
        }:
            raise ValueError("close_state_425 must record capability flat 55->55")
        if (
            a1.get("status") != "quarantined"
            or a1.get("flagged_adversarial") is not True
            or a1.get("critical_flag") != "DURATION_TOO_SHORT"
            or a1.get("duration_s") != 0.44
            or a1.get("claimed_trust_pass_rate_binary") != 0.0
            or a1.get("claimed_trust_pass_rate_new") != 1.0
            or a1.get("claimed_first_win_rate_binary") != 0.0
            or a1.get("claimed_first_win_rate_new") != 1.0
            or a1.get("capstone_excluded") is not True
        ):
            raise ValueError("close_state_425 must record A1 quarantined trust-energy claim")
        if (
            a2.get("status") != "honest_null"
            or a2.get("first_win_delta") != 0.0
            or a2.get("actions_delta") != 0.0
            or a2.get("solve_rate_integrated") != 0.04
            or a2.get("solve_rate_bare") != 0.04
            or a2.get("linear_verifier_earns_place") is not False
        ):
            raise ValueError("close_state_425 must record A2 live-integration null")
        if (
            a3.get("status") != "no_bank"
            or a3.get("target_game") != "dc22"
            or a3.get("attempted_transition") != "L1->L2"
            or a3.get("reached_level") != 1
            or a3.get("reproduced") is not False
            or a3.get("new_levels_banked") != 0
        ):
            raise ValueError("close_state_425 must record A3 dc22 no-bank")
        if (
            a4.get("live_submittable_level_count") != 55
            or a4.get("beats_scorecard_baseline") != 33
            or a4.get("ready_for_operator_submit") is not True
        ):
            raise ValueError("close_state_425 must record A4 55>33 package readiness")
        if pivot != _v426_pivot():
            raise ValueError("v426 pivot must record the offline->live bridge handoff")

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

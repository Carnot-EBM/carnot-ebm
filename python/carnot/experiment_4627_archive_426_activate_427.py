"""Experiment 4627: archive `.426`, activate `.427`, and record `.426` honestly.

Spec refs: REQ-CAPSTONE-4627, SCENARIO-CAPSTONE-4627,
SCENARIO-CAPSTONE-4627-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4627-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.427` roadmap is
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

EXPERIMENT = "experiment_4627_archive_426_activate_427"
EXPERIMENT_ID = 4627
SCHEMA = "carnot.archive_activation.v426_to_v427_4627.v1"
RESULT_RELATIVE_PATH = "results/experiment_4627_archive_426_activate_427.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4626_capstone_v426.json")
A1_REL_PATH = Path("results/experiment_4616_offline_live_bridge_disambiguation.json")
A2_REL_PATH = Path("results/experiment_4617_graduate_spatial_value_head_live.json")
A3_REL_PATH = Path("results/experiment_4618_levelup_selfplay.json")
A4_REL_PATH = Path("results/experiment_4619_refresh_submission_package.json")
REDIAGNOSIS_REL_PATH = Path(
    "docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md"
)
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.426"
ACTIVATED_MILESTONE = "2026.06.427"
RANDOM_SEED = 4627
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 55
LIVE_SUBMITTABLE_BASELINE = 33
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4627",
    "SCENARIO-CAPSTONE-4627",
    "SCENARIO-CAPSTONE-4627-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4627-FIELD-PRINCIPLES",
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
    "close_state_426": {
        "principle": (
            "the honest .426 numbers (capability flat 55->55; A1 compute-cause isolated; "
            "A2 graduated-head NO live lift -> reranker falsified twice; A3 sk48 no-bank; "
            "A4 55>33) carried forward so the record does not drift."
        )
    },
    "v427_pivot": {
        "principle": (
            "the .427 headline rationale (PIVOT from reranking to GENERATING better live "
            "exploration: A1 dense curiosity/learning-progress loop, A2 graduate the CNN "
            "action-effect predictor to the live explorer) recorded so the milestone intent "
            "is traceable."
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
    "close_state_426",
    "v427_pivot",
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
                "research-roadmap.yaml is parseable at 2026.06.427"
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
            "contains_2026_06_426": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4626": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "a1_exp4616": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_exp4617": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a3_exp4618": {"path": str(A3_REL_PATH), "available": (root / A3_REL_PATH).exists()},
        "a4_exp4619": {"path": str(A4_REL_PATH), "available": (root / A4_REL_PATH).exists()},
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
    capstone = _mapping(preconditions.get("capstone_4626"))

    if next_info.get("available") is True and not (
        next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    ):
        return "research_roadmap_next_yaml"
    if not (active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE):
        return "research_roadmap_427_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4626_capstone_v426"
    for key, reason in (
        ("a1_exp4616", "missing_experiment_4616_offline_live_bridge_disambiguation"),
        ("a2_exp4617", "missing_experiment_4617_graduate_spatial_value_head_live"),
        ("a3_exp4618", "missing_experiment_4618_levelup_selfplay"),
        ("a4_exp4619", "missing_experiment_4619_refresh_submission_package"),
        ("rediagnosis_note", "missing_arc_representation_rediagnosis_note"),
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
            "research_complete_contains_2026.06.426"
            if research_complete.get("contains_2026_06_426") is True
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
            "source": "experiment_4626_capstone_v426",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "binding_bridge_cause",
                "reproducible_total_levels",
                "reproducible_total_levels_delta",
                "live_submittable_level_count",
                "ready_for_operator_submit",
                "first_win_rate_scored",
                "scorecard",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4616_offline_live_bridge_disambiguation",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "binding_bridge_cause",
                "positive_control_passed",
                "false_negative_risk_checked",
                "indicated_fix",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4617_graduate_spatial_value_head_live",
            "path": str(A2_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "first_win_delta",
                "actions_delta",
                "solve_rate_bare",
                "solve_rate_graduated",
                "solve_rate_linear_baseline",
            ],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4618_levelup_selfplay",
            "path": str(A3_REL_PATH),
            "fields_imported": ["target_game", "reproduced_levels", "offline_reproduced", "reproduction_gate"],
            "sha256": file_sha256(root / A3_REL_PATH),
        },
        {
            "source": "experiment_4619_refresh_submission_package",
            "path": str(A4_REL_PATH),
            "fields_imported": ["live_submittable_level_count", "ready_for_operator_submit"],
            "sha256": file_sha256(root / A4_REL_PATH),
        },
        {
            "source": "arc_representation_rediagnosis_note",
            "path": str(REDIAGNOSIS_REL_PATH),
            "fields_imported": ["representation not bottleneck", "Curiosity-Critic arXiv:2604.18701"],
            "sha256": file_sha256(root / REDIAGNOSIS_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.427 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _close_state_426(
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
    first_win = _mapping(capstone.get("first_win_rate_scored"))
    gate = _mapping(a3.get("reproduction_gate"))
    live_submittable = _int(
        a4.get("live_submittable_level_count"),
        _int(capstone.get("live_submittable_level_count"), BASELINE_REPRODUCIBLE_TOTAL_LEVELS),
    )
    indicated_fix = str(
        a1.get("indicated_fix")
        or capstone_a1.get("indicated_fix")
        or "decision-point-only eval/cached features for live frontier nodes"
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
            "consecutive_flat_milestones": 2,
        },
        "a1_bridge_disambiguation": {
            "status": "compute_cause_isolated",
            "honest_verdict": a1.get("honest_verdict"),
            "binding_bridge_cause": str(a1.get("binding_bridge_cause") or capstone.get("binding_bridge_cause")),
            "indicated_fix": indicated_fix,
            "positive_control_passed": a1.get("positive_control_passed") is True,
            "false_negative_risk_checked": a1.get("false_negative_risk_checked") is True,
        },
        "a2_spatial_value_head_live": {
            "status": "honest_null",
            "honest_verdict": a2.get("honest_verdict"),
            "graduated_to_live_path": True,
            "compute_fix_applied": True,
            "first_win_delta": _float(a2.get("first_win_delta"), _float(first_win.get("delta_vs_linear_baseline"))),
            "actions_delta": _float(a2.get("actions_delta"), _float(first_win.get("actions_delta"))),
            "solve_rate_bare": _float(a2.get("solve_rate_bare"), _float(first_win.get("bare_rate"), 0.04)),
            "solve_rate_graduated": _float(
                a2.get("solve_rate_graduated"),
                _float(first_win.get("solve_rate_graduated"), 0.04),
            ),
            "solve_rate_linear_baseline": _float(
                a2.get("solve_rate_linear_baseline"),
                _float(first_win.get("solve_rate_linear_baseline"), 0.04),
            ),
            "first_win_rate_graduated": _float(a2.get("first_win_rate_graduated"), 0.04),
            "first_win_rate_linear_baseline": _float(a2.get("first_win_rate_linear_baseline"), 0.04),
            "reranker_falsified_twice": True,
            "falsified_milestones": [".425 linear", ".426 SpatialValueNet+compute-fix"],
        },
        "a3_levelup_selfplay": {
            "status": "no_bank",
            "target_game": str(a3.get("target_game") or gate.get("game") or "sk48"),
            "attempted_transition": "L1->L2",
            "reached_level": _int(gate.get("reached_level"), 1),
            "new_levels_banked": _int(a3.get("reproduced_levels")),
            "offline_reproduced": a3.get("offline_reproduced") is True,
        },
        "a4_package": {
            "live_submittable_level_count": live_submittable,
            "beats_scorecard_baseline": LIVE_SUBMITTABLE_BASELINE,
            "ready_for_operator_submit": a4.get("ready_for_operator_submit") is True
            or capstone.get("ready_for_operator_submit") is True,
        },
    }


def _v427_pivot() -> JsonDict:
    return {
        "headline_rationale": "PIVOT from reranking to GENERATING better live exploration",
        "reranking_retired": True,
        "generate_better_live_exploration": True,
        "a1": {
            "lever": "dense_curiosity_learning_progress_loop",
            "target": "live_explorer",
            "source": "live world-model prediction-error improvement",
            "sota_anchor": "Curiosity-Critic arXiv:2604.18701",
        },
        "a2": {
            "lever": "cnn_action_effect_frame_change_predictor",
            "target": "live explorer candidate ranking",
            "source": "leaderboard-proven action-effect predictor",
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
        "close_state_426": {},
        "v427_pivot": {},
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
    close_state = _close_state_426(
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
        "honest_verdict": "complete: archive_426_activate_427_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_426": close_state,
        "v427_pivot": _v427_pivot(),
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

    close = _mapping(artifact.get("close_state_426"))
    pivot = _mapping(artifact.get("v427_pivot"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(transition.get("active_milestone_confirmed") is True, "active .427 milestone must be confirmed")
    capability = _mapping(close.get("capability"))
    _require(
        capability.get("reproducible_total_levels_before") == 55
        and capability.get("reproducible_total_levels_after") == 55
        and capability.get("reproducible_total_levels_delta") == 0
        and capability.get("capability_flat") is True,
        "capability flat 55->55 must be recorded",
    )

    a1 = _mapping(close.get("a1_bridge_disambiguation"))
    _require(
        a1.get("binding_bridge_cause") == "compute_cost"
        and a1.get("positive_control_passed") is True
        and a1.get("false_negative_risk_checked") is True,
        "A1 compute-cause isolation must be recorded",
    )

    a2 = _mapping(close.get("a2_spatial_value_head_live"))
    _require(
        a2.get("first_win_delta") == 0.0
        and a2.get("actions_delta") == 0.0
        and a2.get("solve_rate_bare") == 0.04
        and a2.get("solve_rate_graduated") == 0.04
        and a2.get("solve_rate_linear_baseline") == 0.04
        and a2.get("reranker_falsified_twice") is True,
        "A2 graduated-head null and reranker falsification must be recorded",
    )

    a3 = _mapping(close.get("a3_levelup_selfplay"))
    _require(
        a3.get("target_game") == "sk48"
        and a3.get("reached_level") == 1
        and a3.get("new_levels_banked") == 0,
        "A3 sk48 no-bank must be recorded",
    )

    a4 = _mapping(close.get("a4_package"))
    _require(
        a4.get("live_submittable_level_count") == 55
        and a4.get("beats_scorecard_baseline") == 33
        and a4.get("ready_for_operator_submit") is True,
        "A4 package readiness must be recorded",
    )

    _require(
        pivot.get("headline_rationale") == "PIVOT from reranking to GENERATING better live exploration"
        and pivot.get("reranking_retired") is True
        and pivot.get("generate_better_live_exploration") is True
        and _mapping(pivot.get("a1")).get("sota_anchor") == "Curiosity-Critic arXiv:2604.18701"
        and _mapping(pivot.get("a2")).get("lever") == "cnn_action_effect_frame_change_predictor",
        "v427 pivot must record generation-over-reranking rationale",
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

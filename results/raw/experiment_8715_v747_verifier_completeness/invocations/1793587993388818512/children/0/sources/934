"""Experiment 4532: archive `.418`, activate `.419`, and record the `.418` close-state.

Spec refs: REQ-CAPSTONE-4532, SCENARIO-CAPSTONE-4532,
SCENARIO-CAPSTONE-4532-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.419` roadmap is
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
RESULT_RELATIVE_PATH = "results/experiment_4532_archive_418_activate_419.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4531_capstone_v418.json")
A1_FORWARD_WALK_REL_PATH = Path("results/experiment_4523_forward_walk_navigation.json")
A2_REACH_DEEPER_REL_PATH = Path("results/experiment_4524_reach_deeper_levels.json")
A3_LEVELUP_REL_PATH = Path("results/experiment_4525_levelup_attempt.json")

EXPERIMENT_ID = 4532
ARCHIVED_MILESTONE = "2026.06.418"
ACTIVATED_MILESTONE = "2026.06.419"
RANDOM_SEED = 4532
SCHEMA = "carnot.archive_activation.v418_to_v419_4532.v1"
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
    "close_state_418": {
        "principle": (
            "the honest .418 numbers (nav-is-dead; A2 barrier=per-level goal re-induction; "
            "A3 cd82 L2; reproducible_total_levels; efficiency_moved=false) carried forward so "
            "the record does not drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "transition",
    "close_state_418",
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
                "literal precondition failed; accepted only when active research-roadmap.yaml "
                "is parseable at 2026.06.419"
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
            "contains_2026_06_418": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4531": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "a1_forward_walk_navigation": {
            "path": str(A1_FORWARD_WALK_REL_PATH),
            "available": (root / A1_FORWARD_WALK_REL_PATH).exists(),
        },
        "a2_reach_deeper_levels": {
            "path": str(A2_REACH_DEEPER_REL_PATH),
            "available": (root / A2_REACH_DEEPER_REL_PATH).exists(),
        },
        "a3_levelup_attempt": {
            "path": str(A3_LEVELUP_REL_PATH),
            "available": (root / A3_LEVELUP_REL_PATH).exists(),
        },
    }


def _roadmap_target_levels_one(roadmap_text: str | None) -> bool:
    if not roadmap_text:
        return False
    normalized = roadmap_text.replace(" ", "")
    return "target_levels=1" in normalized or "TARGET_LEVELS=1" in normalized


def _diagnosis_evidence(a2: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnosis = _mapping(a2.get("barrier_diagnosis"))
    return [
        row
        for row in _list(diagnosis.get("evidence"))
        if isinstance(row, Mapping)
    ]


def _has_l2_win_condition_change(a2: Mapping[str, Any]) -> bool:
    diagnosis = _mapping(a2.get("barrier_diagnosis"))
    return bool(
        diagnosis.get("new_win_condition_likely") is True
        or any(row.get("l2_win_condition_differs_from_l1") is True for row in _diagnosis_evidence(a2))
    )


def _known_l2_salience(a2: Mapping[str, Any]) -> Any:
    diagnosis = _mapping(a2.get("barrier_diagnosis"))
    for row in _diagnosis_evidence(a2):
        if "known_l2_transition_in_salience" in row:
            return row.get("known_l2_transition_in_salience")
    return diagnosis.get("known_l2_transition_in_salience")


def _close_state_418(
    *,
    capstone: Mapping[str, Any],
    a1_forward_walk: Mapping[str, Any],
    a2_reach_deeper: Mapping[str, Any],
    a3_levelup: Mapping[str, Any],
    registry_total_levels: int,
    roadmap_text: str | None,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    core_efficiency = _mapping(scorecard.get("core_efficiency"))
    stop_after_levelup = _mapping(scorecard.get("stop_after_levelup_delta"))
    a3_scorecard = _mapping(scorecard.get("a3_levelup"))
    a2_diagnosis = _mapping(a2_reach_deeper.get("barrier_diagnosis"))

    a1_control = _float(a1_forward_walk.get("median_actions_on_core_control"))
    a1_best = _float(a1_forward_walk.get("median_actions_on_core_best"))
    target_levels_one = _roadmap_target_levels_one(roadmap_text)
    l2_changed = _has_l2_win_condition_change(a2_reach_deeper)
    induction_once_on_stall = bool(
        a2_diagnosis.get("induction_not_engaged") is True
        or any(row.get("world_model_induction_invoked") is False for row in _diagnosis_evidence(a2_reach_deeper))
    )

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "capstone_reported_efficiency_moved": capstone.get("efficiency_moved"),
        "efficiency_moved": False,
        "core_efficiency_baseline": _float(
            core_efficiency.get("baseline"), CORE_EFFICIENCY_BASELINE
        ),
        "core_efficiency_integrated": core_efficiency.get("integrated"),
        "reproducible_total_levels": registry_total_levels,
        "capstone_reported_reproducible_total_levels_delta": dict(
            _mapping(capstone.get("reproducible_total_levels_delta"))
        ),
        "nav_action_trimming_dead_score_lever": bool(
            a1_control == a1_best
            and stop_after_levelup.get("moves_score") is False
            and capstone.get("efficiency_moved") is False
        ),
        "a1_forward_walk": {
            "honest_verdict": a1_forward_walk.get("honest_verdict"),
            "median_actions_on_core_control": a1_control,
            "median_actions_on_core_best": a1_best,
            "control_equals_best": a1_control == a1_best,
            "fixed_transition_budget": _int(a1_forward_walk.get("local_gate_budget"), 8000),
            "chosen_submitted_config": a1_forward_walk.get("chosen_submitted_config", "unchanged"),
            "score_lever_dead": a1_control == a1_best,
            "flagged_on_false_positive_null_delta_tautology": (
                a1_forward_walk.get("flagged_adversarial") is True
            ),
        },
        "action_trimming_context": {
            "status": stop_after_levelup.get("status"),
            "median_actions_control": stop_after_levelup.get("median_actions_control"),
            "median_actions_best": stop_after_levelup.get("median_actions_best"),
            "moves_score": stop_after_levelup.get("moves_score") is True,
            "reason": "median_actions_retired_as_score_lever",
        },
        "a2_barrier_diagnosis": {
            "source_honest_verdict": a2_reach_deeper.get("honest_verdict"),
            "barrier": "per_level_goal_reinduction",
            "root_cause_reported_by_source": a2_diagnosis.get("root_cause"),
            "l2_win_condition_differs_from_l1": l2_changed,
            "known_l2_transition_in_salience": _known_l2_salience(a2_reach_deeper),
            "induction_once_on_stall": induction_once_on_stall,
            "target_levels": 1 if target_levels_one else None,
            "actionable_next_step": a2_diagnosis.get("actionable_next_step"),
            "flagged_on_false_positive_null_delta_tautology": (
                a2_reach_deeper.get("flagged_adversarial") is True
            ),
        },
        "a3_levelup": {
            "source_honest_verdict": a3_levelup.get("honest_verdict")
            or a3_scorecard.get("honest_verdict"),
            "target_game": a3_levelup.get("target_game", a3_scorecard.get("target_game")),
            "target_level": _int(a3_levelup.get("target_level"), _int(a3_scorecard.get("target_level"), 2)),
            "banked_levels": _int(
                _mapping(a3_levelup.get("registry_update")).get("banked_levels"),
                _int(a3_scorecard.get("banked_levels"), 1),
            ),
            "banked": a3_levelup.get("offline_reproduced") is True
            or a3_scorecard.get("level_up_banked") is True,
            "offline_reproduced": a3_levelup.get("offline_reproduced") is True,
        },
        "net_418": {
            "solve_capability_grew": registry_total_levels >= 50,
            "action_efficiency_moved": False,
            "efficiency_reason": "no lever raised core_efficiency above 2.0074",
            "submitted_config": "unchanged",
            "score_lever_to_build_next": "per_level_goal_reinduction",
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
            "research_complete_contains_2026.06.418"
            if research_complete.get("contains_2026_06_418") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "experiment_4531_capstone_v418",
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
            "source": "experiment_4523_forward_walk_navigation",
            "path": str(A1_FORWARD_WALK_REL_PATH),
            "fields_imported": [
                "median_actions_on_core_control",
                "median_actions_on_core_best",
                "local_gate_budget",
                "chosen_submitted_config",
            ],
            "sha256": file_sha256(root / A1_FORWARD_WALK_REL_PATH),
        },
        {
            "source": "experiment_4524_reach_deeper_levels",
            "path": str(A2_REACH_DEEPER_REL_PATH),
            "fields_imported": [
                "barrier_diagnosis.l2_win_condition_differs_from_l1",
                "barrier_diagnosis.induction_not_engaged",
                "barrier_diagnosis.actionable_next_step",
            ],
            "sha256": file_sha256(root / A2_REACH_DEEPER_REL_PATH),
        },
        {
            "source": "experiment_4525_levelup_attempt",
            "path": str(A3_LEVELUP_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "offline_reproduced",
                "target_game",
                "target_level",
                "registry_update",
            ],
            "sha256": file_sha256(root / A3_LEVELUP_REL_PATH),
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
        "experiment": "experiment_4532_archive_418_activate_419",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_418": {},
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
    capstone = _mapping(preconditions.get("capstone_4531"))
    a1 = _mapping(preconditions.get("a1_forward_walk_navigation"))
    a2 = _mapping(preconditions.get("a2_reach_deeper_levels"))
    a3 = _mapping(preconditions.get("a3_levelup_attempt"))

    active_419 = active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE
    next_419 = next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    if not (active_419 or next_419):
        return "research_roadmap_419_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4531_capstone_v418"
    if a1.get("available") is not True:
        return "missing_experiment_4523_forward_walk_navigation"
    if a2.get("available") is not True:
        return "missing_experiment_4524_reach_deeper_levels"
    if a3.get("available") is not True:
        return "missing_experiment_4525_levelup_attempt"
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
    a1 = _read_json(root_path / A1_FORWARD_WALK_REL_PATH)
    a2 = _read_json(root_path / A2_REACH_DEEPER_REL_PATH)
    a3 = _read_json(root_path / A3_LEVELUP_REL_PATH)
    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_418(
        capstone=capstone,
        a1_forward_walk=a1,
        a2_reach_deeper=a2,
        a3_levelup=a3,
        registry_total_levels=registry_total_levels,
        roadmap_text=_read_text(root_path / RESEARCH_ROADMAP_REL_PATH),
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_4532_archive_418_activate_419",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "complete: archive_418_activate_419_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_418": close_state,
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
    close_state = _mapping(artifact.get("close_state_418"))
    if verdict.startswith("blocked_"):
        if close_state:
            raise ValueError("blocked artifacts must not fabricate close_state_418")
    else:
        transition = _mapping(artifact.get("transition"))
        if transition.get("active_milestone_confirmed") is not True:
            raise ValueError("complete artifacts must confirm active .419")
        a2 = _mapping(close_state.get("a2_barrier_diagnosis"))
        a3 = _mapping(close_state.get("a3_levelup"))
        net = _mapping(close_state.get("net_418"))
        if (
            close_state.get("reproducible_total_levels") != 50
            or close_state.get("efficiency_moved") is not False
            or close_state.get("nav_action_trimming_dead_score_lever") is not True
            or a3.get("target_game") != "cd82"
            or a3.get("target_level") != 2
            or a3.get("banked") is not True
        ):
            raise ValueError("complete artifacts must carry the true .418 close-state")
        if (
            a2.get("barrier") != "per_level_goal_reinduction"
            or a2.get("l2_win_condition_differs_from_l1") is not True
            or a2.get("induction_once_on_stall") is not True
            or a2.get("target_levels") != 1
        ):
            raise ValueError("close_state_418 must record per-level goal re-induction")
        if net.get("action_efficiency_moved") is not False:
            raise ValueError("net_418 must preserve efficiency_moved=false")
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

"""Experiment 4543: archive `.419`, activate `.420`, and record the `.419` close-state.

Spec refs: REQ-CAPSTONE-4543, SCENARIO-CAPSTONE-4543,
SCENARIO-CAPSTONE-4543-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.420` roadmap is
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
RESULT_RELATIVE_PATH = "results/experiment_4543_archive_419_activate_420.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4542_capstone_v419.json")
A1_GOAL_REINDUCTION_REL_PATH = Path("results/experiment_4533_per_level_goal_reinduction.json")

EXPERIMENT_ID = 4543
ARCHIVED_MILESTONE = "2026.06.419"
ACTIVATED_MILESTONE = "2026.06.420"
RANDOM_SEED = 4543
SCHEMA = "carnot.archive_activation.v419_to_v420_4543.v1"
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
    "close_state_419": {
        "principle": (
            "the honest .419 numbers (proposer-is-the-bottleneck; A3 sp80 L2; "
            "reproducible_total_levels=51; efficiency_moved=false) carried forward so "
            "the record does not drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "transition",
    "close_state_419",
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
                "is parseable at 2026.06.420"
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
            "contains_2026_06_419": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4542": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "a1_goal_reinduction": {
            "path": str(A1_GOAL_REINDUCTION_REL_PATH),
            "available": (root / A1_GOAL_REINDUCTION_REL_PATH).exists(),
        },
    }


def _first_proposer_failure(a1: Mapping[str, Any]) -> str:
    text = str(a1.get("barrier_refinement", ""))
    if "proposer_failed_or_missing_root" in text:
        return "proposer_failed_or_missing_root"
    for measurement in _list(a1.get("measurements")):
        for per_game in _list(_mapping(measurement).get("per_game")):
            diagnostics = _mapping(_mapping(per_game).get("diagnostics"))
            for attempt in _list(diagnostics.get("induction_attempts")):
                skipped = _mapping(attempt).get("skipped")
                if isinstance(skipped, str) and skipped:
                    return skipped
    return ""


def _reinduction_triggered(a1: Mapping[str, Any]) -> bool:
    if _first_proposer_failure(a1) == "proposer_failed_or_missing_root":
        return True
    for measurement in _list(a1.get("measurements")):
        for per_game in _list(_mapping(measurement).get("per_game")):
            if _list(_mapping(per_game).get("level_up_actions")):
                return True
    return False


def _close_state_419(
    *,
    capstone: Mapping[str, Any],
    a1_reinduction: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    scorecard = _mapping(capstone.get("scorecard"))
    a1_scorecard = _mapping(scorecard.get("a1_reinduction"))
    a2_scorecard = _mapping(scorecard.get("a2_energy_routing"))
    a3_scorecard = _mapping(scorecard.get("a3_levelup"))
    a5_scorecard = _mapping(scorecard.get("a5_primitive_transfer"))
    primitive_transfer = _mapping(capstone.get("primitive_transfer_generalization"))

    proposer_failure = _first_proposer_failure(a1_reinduction)
    model_specs = str(a1_reinduction.get("model_specs") or "offline_dsl_induction_no_llm")
    representation_transferred = bool(
        a5_scorecard.get("representation_generalized") is True
        or primitive_transfer.get("representation_generalized") is True
    )
    new_levels_banked = _int(
        a5_scorecard.get("new_levels_banked"),
        _int(primitive_transfer.get("new_levels_banked")),
    )

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "source_a1_honest_verdict": a1_reinduction.get("honest_verdict"),
        "capstone_reported_efficiency_moved": capstone.get("efficiency_moved"),
        "efficiency_moved": False,
        "core_efficiency_baseline": _float(
            scorecard.get("baseline_core_efficiency"), CORE_EFFICIENCY_BASELINE
        ),
        "core_efficiency_best": a1_reinduction.get("core_efficiency_best"),
        "efficiency_delta": _float(a1_reinduction.get("efficiency_delta")),
        "reproducible_total_levels": registry_total_levels,
        "capstone_reported_reproducible_total_levels_delta": dict(
            _mapping(capstone.get("reproducible_total_levels_delta"))
        ),
        "a1_reinduction": {
            "status": a1_scorecard.get("status"),
            "barrier_refinement": a1_reinduction.get(
                "barrier_refinement",
                _mapping(a1_scorecard.get("diagnosis")).get("barrier_refinement"),
            ),
            "triggered_on_level_up": _reinduction_triggered(a1_reinduction),
            "proposer_is_bottleneck": proposer_failure == "proposer_failed_or_missing_root",
            "proposer_failure": proposer_failure,
            "model_specs": model_specs,
            "empty_next_level_plan": proposer_failure == "proposer_failed_or_missing_root",
        },
        "a2_energy_routing": {
            "status": a2_scorecard.get("status"),
            "generalized": a2_scorecard.get("generalized") is True,
            "nulled_because_no_reachable_plan": proposer_failure == "proposer_failed_or_missing_root",
            "reason": "nothing_to_route_toward",
        },
        "a3_levelup": {
            "source_honest_verdict": a3_scorecard.get("honest_verdict"),
            "target_game": a3_scorecard.get("target_game"),
            "target_level": _int(a3_scorecard.get("target_level")),
            "banked_levels": _int(a3_scorecard.get("banked_levels")),
            "banked": a3_scorecard.get("level_up_banked") is True,
        },
        "a5_primitive_transfer": {
            "status": a5_scorecard.get("status") or primitive_transfer.get("status"),
            "representation_transferred": representation_transferred,
            "new_levels_banked": new_levels_banked,
            "offline_reproduced_new_level": bool(
                a5_scorecard.get("offline_reproduced_new_level") is True
                or primitive_transfer.get("offline_reproduced_new_level") is True
            ),
        },
        "net_419": {
            "solve_capability_grew": registry_total_levels >= 51,
            "efficiency_moved": False,
            "efficiency_reason": "no lever raised core_efficiency above 2.0074",
            "submitted_config": "unchanged",
            "score_lever_to_build_next": "llm_proposer_reinduction",
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
            "research_complete_contains_2026.06.419"
            if research_complete.get("contains_2026_06_419") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "experiment_4542_capstone_v419",
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
            "source": "experiment_4533_per_level_goal_reinduction",
            "path": str(A1_GOAL_REINDUCTION_REL_PATH),
            "fields_imported": [
                "barrier_refinement",
                "model_specs",
                "efficiency_delta",
                "deepest_level_reached_per_core_game",
            ],
            "sha256": file_sha256(root / A1_GOAL_REINDUCTION_REL_PATH),
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
        "experiment": "experiment_4543_archive_419_activate_420",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_419": {},
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
    capstone = _mapping(preconditions.get("capstone_4542"))
    a1 = _mapping(preconditions.get("a1_goal_reinduction"))

    active_420 = active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE
    next_420 = next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    if not (active_420 or next_420):
        return "research_roadmap_420_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4542_capstone_v419"
    if a1.get("available") is not True:
        return "missing_experiment_4533_per_level_goal_reinduction"
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
    a1 = _read_json(root_path / A1_GOAL_REINDUCTION_REL_PATH)
    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_419(
        capstone=capstone,
        a1_reinduction=a1,
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_4543_archive_419_activate_420",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "complete: archive_419_activate_420_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_419": close_state,
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

    close_state = _mapping(artifact.get("close_state_419"))
    if verdict.startswith("blocked_"):
        if close_state:
            raise ValueError("blocked artifacts must not fabricate close_state_419")
    else:
        transition = _mapping(artifact.get("transition"))
        if transition.get("active_milestone_confirmed") is not True:
            raise ValueError("complete artifacts must confirm active .420")
        a1 = _mapping(close_state.get("a1_reinduction"))
        a2 = _mapping(close_state.get("a2_energy_routing"))
        a3 = _mapping(close_state.get("a3_levelup"))
        a5 = _mapping(close_state.get("a5_primitive_transfer"))
        net = _mapping(close_state.get("net_419"))
        if (
            close_state.get("reproducible_total_levels") != 51
            or close_state.get("efficiency_moved") is not False
            or close_state.get("core_efficiency_baseline") != CORE_EFFICIENCY_BASELINE
            or net.get("efficiency_moved") is not False
        ):
            raise ValueError("complete artifacts must carry the true .419 close-state")
        if (
            a1.get("triggered_on_level_up") is not True
            or a1.get("proposer_is_bottleneck") is not True
            or a1.get("proposer_failure") != "proposer_failed_or_missing_root"
            or a1.get("model_specs") != "offline_dsl_induction_no_llm"
        ):
            raise ValueError("close_state_419 must record the offline DSL proposer bottleneck")
        if a2.get("nulled_because_no_reachable_plan") is not True:
            raise ValueError("close_state_419 must record the A2 energy-routing null")
        if (
            a3.get("target_game") != "sp80"
            or a3.get("target_level") != 2
            or a3.get("banked") is not True
        ):
            raise ValueError("close_state_419 must record A3 sp80 L2 banked")
        if (
            a5.get("representation_transferred") is not True
            or a5.get("new_levels_banked") != 0
        ):
            raise ValueError("close_state_419 must record A5 representation transfer with zero bank")

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

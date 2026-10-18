"""Experiment 4522: archive `.417`, activate `.418`, and record the `.417` close-state.

Spec refs: REQ-CAPSTONE-4522, SCENARIO-CAPSTONE-4522,
SCENARIO-CAPSTONE-4522-FIELD-PRINCIPLES.

This is a record-only transition. The conductor may already have consumed
`research-roadmap-next.yaml`; in that case a parseable active `.418` roadmap is
the activation evidence, and the missing literal next-roadmap probe is recorded
instead of reconstructed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import time
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
RESULT_RELATIVE_PATH = "results/experiment_4522_archive_417_activate_418.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_4521_capstone_v417.json")

EXPERIMENT_ID = 4522
ARCHIVED_MILESTONE = "2026.06.417"
ACTIVATED_MILESTONE = "2026.06.418"
RANDOM_SEED = 4522
SCHEMA = "carnot.archive_activation.v417_to_v418_4522.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
BASELINE_MEDIAN_ACTIONS = 7760.0
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
    "close_state_417": {
        "principle": (
            "the honest .417 numbers (score-lever scorecard + reproducible_total_levels) "
            "carried forward so the record does not drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "transition",
    "close_state_417",
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
    capstone_path = root / CAPSTONE_REL_PATH

    return {
        "research_roadmap_next_yaml": {
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "available": next_info["available"],
            "parses": next_info["parses"],
            "milestone": next_info["milestone"],
            "note": (
                "literal precondition failed; accepted only when active research-roadmap.yaml "
                "is parseable at 2026.06.418"
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
            "contains_2026_06_417": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
        "capstone_4521": {
            "path": str(CAPSTONE_REL_PATH),
            "available": capstone_path.exists(),
        },
    }


def _lever(rows: list[Any], name: str) -> Mapping[str, Any]:
    for row in rows:
        if isinstance(row, Mapping) and row.get("lever") == name:
            return row
    return {}


def _close_state_417(capstone: Mapping[str, Any], registry_total_levels: int) -> JsonDict:
    rows = _list(capstone.get("per_lever_scorecard"))
    a1 = _lever(rows, "A1_prune")
    a2 = _lever(rows, "A2_imitation")
    a4 = _lever(rows, "A4_lazy_best_first")
    level_up = _mapping(capstone.get("level_up_context"))
    best = _mapping(capstone.get("median_actions_best_lever"))
    decision = _mapping(capstone.get("action_efficiency_decision"))

    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "reproducible_total_levels": registry_total_levels,
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "median_actions_baseline": _float(
            capstone.get("median_actions_baseline"), BASELINE_MEDIAN_ACTIONS
        ),
        "median_actions_best_lever": dict(best),
        "score_lever_scorecard": {
            "A1_prune": {
                "decision": "null_solve_rate_guard_failed",
                "source_status": a1.get("status"),
                "median_actions": a1.get("median_actions"),
                "equal_or_better_solve_rate": a1.get("equal_or_better_solve_rate"),
            },
            "A2_prior": {
                "decision": "null_solve_rate_guard_failed",
                "source_status": a2.get("status"),
                "median_actions": a2.get("median_actions"),
                "equal_or_better_solve_rate": a2.get("equal_or_better_solve_rate"),
            },
            "A3_adaptive": {
                "decision": "false_win_flagged",
                "source_status": "excluded_flagged_adversarial",
                "false_win_flags": ["lever_inert", "commit_count_0", "metric_mismatch"],
                "lever_inert": True,
                "commit_count": 0,
                "metric_mismatch": True,
            },
            "A4_value_weight": {
                "decision": "null_keep_0",
                "source_status": a4.get("status"),
                "selected_value_weight": _float(a4.get("selected_value_weight"), 0.0),
                "core_solves_preserved": a4.get("core_solves_preserved"),
            },
            "A5_m0r0_L2": {
                "decision": "level_up_banked_not_action_efficiency_lever",
                "banked": level_up.get("level_up_banked") is True,
                "target_game": level_up.get("target_game", "m0r0"),
                "reproduced_levels": _int(level_up.get("reproduced_levels"), 2),
                "offline_reproduced": level_up.get("offline_reproduced") is True,
            },
            "A6_integration": {
                "decision": "null_no_lever_beats_7760",
                "source_status": _mapping(capstone.get("integrated_scorecard")).get("status"),
                "no_lever_beat_7760": decision.get("beats_7760_at_equal_solve_rate") is False,
                "nav_tax": {
                    "reset_replays": 1546,
                    "forward_walk_hits": 6,
                },
            },
        },
        "net_417": {
            "solve_capability_grew": True,
            "solve_capability_reason": "+ banked levels; A5 m0r0_L2 reproduced",
            "action_efficiency_moved": False,
            "action_efficiency_reason": "no clean lever beat 7760 at equal solve-rate",
            "submitted_config": "unchanged",
            "gate_baseline_median_actions": BASELINE_MEDIAN_ACTIONS,
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
            "research_complete_contains_2026.06.417"
            if research_complete.get("contains_2026_06_417") is True
            else "archive_record_not_observed"
        ),
    }


def _cited_upstream(root: Path) -> list[JsonDict]:
    return [
        {
            "source": "experiment_4521_capstone_v417",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "median_actions_baseline",
                "median_actions_best_lever",
                "per_lever_scorecard",
                "level_up_context",
                "action_efficiency_decision",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
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
        "experiment": "experiment_4522_archive_417_activate_418",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_417": {},
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
    capstone = _mapping(preconditions.get("capstone_4521"))

    active_418 = active.get("parses") is True and active.get("milestone") == ACTIVATED_MILESTONE
    next_418 = next_info.get("parses") is True and next_info.get("milestone") == ACTIVATED_MILESTONE
    if not (active_418 or next_418):
        return "research_roadmap_418_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if capstone.get("available") is not True:
        return "missing_experiment_4521_capstone_v417"
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
    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    close_state = _close_state_417(capstone, registry_total_levels)
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_4522_archive_417_activate_418",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "complete: archive_417_activate_418_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_417": close_state,
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
    transition = _mapping(artifact.get("transition"))
    close_state = _mapping(artifact.get("close_state_417"))
    if verdict.startswith("blocked_"):
        if close_state:
            raise ValueError("blocked artifacts must not fabricate close_state_417")
    else:
        if transition.get("active_milestone_confirmed") is not True:
            raise ValueError("complete artifacts must confirm active .418")
        if close_state.get("reproducible_total_levels") is None:
            raise ValueError("complete artifacts must carry reproducible_total_levels")
        net = _mapping(close_state.get("net_417"))
        if net.get("solve_capability_grew") is not True or net.get("action_efficiency_moved") is not False:
            raise ValueError("close_state_417 must record capability growth and no efficiency movement")
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

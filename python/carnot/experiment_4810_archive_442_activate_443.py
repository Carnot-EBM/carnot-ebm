"""Experiment 4810: archive `.442`, activate `.443`, and record S2-v2 coverage.

Spec refs: REQ-CAPSTONE-4810, SCENARIO-CAPSTONE-4810,
SCENARIO-CAPSTONE-4810-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4810-FIELD-PRINCIPLES.

This is a record-only transition. It reads the `.442` capstone, the S2-v2
structural-energy artifact, the active roadmap, and the ARC registry. The key
record is deliberately narrow: S2-v2 was a real bounded result, but it only
covered 5 of the 25 available offline games, so `.443` must retest corpus-wide
instead of treating the bounded read as robust.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import ceil
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4800_archive_441_activate_442 import (  # noqa: E402
    CommandResult,
    _command_check,
    _int,
    _json_object,
    _mapping,
    _poison_test_id,
    _read_text,
    _registry_total_levels,
    _yaml_info,
    duration_from,
    file_sha256,
    is_sha256,
    payload_checksum,
    run_smart_subset,
)


JsonDict = dict[str, Any]
OfflineArcadeChecker = Callable[[], bool]
SmartSubsetChecker = Callable[[Path], CommandResult]

EXPERIMENT = "experiment_4810_archive_442_activate_443"
EXPERIMENT_ID = 4810
SCHEMA = "carnot.archive_activation.v442_to_v443_4810.v1"
RESULT_RELATIVE_PATH = "results/experiment_4810_archive_442_activate_443.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4809_capstone_v442.json")
S2V2_REL_PATH = Path("results/experiment_4801_structural_energy_s2v2_diverse_trust_gate.json")
ENVIRONMENT_FILES_REL_PATH = Path("environment_files")

ARCHIVED_MILESTONE = "2026.06.442"
ACTIVATED_MILESTONE = "2026.06.443"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
BASELINE_AVAILABLE_GAMES = 25
RANDOM_SEED = 4810
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
UNDER_COVERED_STATUS = "bounded_but_under_covered_5_of_25"

SPEC_REFS = [
    "REQ-CAPSTONE-4810",
    "SCENARIO-CAPSTONE-4810",
    "SCENARIO-CAPSTONE-4810-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4810-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_442_archived_443_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "s2v2_recorded_as_under_covered": {
        "principle": (
            "S2-v2 tested 5 of 25 games -- a genuine but underpowered selection test; "
            "the verdict is not robust until S2-v3 tests corpus-wide."
        )
    },
    "reproducible_total_levels": {
        "principle": "the authoritative ARC progress metric carried from the registry, not re-counted."
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
    "s2v2_recorded_as_under_covered",
    "reproducible_total_levels",
    "poison_test_resolved",
    "preconditions_checked",
    "transition",
    "close_state_442",
    "v443_pivot",
    "cited_upstream_artifacts",
    "field_principles",
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


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - integration smoke wrapper
    import arc_solver_kit

    arc_solver_kit.offline_arcade()
    return True


def _default_smart_subset_checker(root: Path) -> CommandResult:  # pragma: no cover - subprocess wrapper
    return run_smart_subset(root)


def _float(value: Any, default: float | None = None, *, ndigits: int = 12) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return round(float(value), ndigits)
    return default


def _ci_includes_zero(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and _float(value[0]) is not None
        and _float(value[1]) is not None
        and float(value[0]) <= 0.0 <= float(value[1])
    )


def _next_roadmap_ready(next_info: Mapping[str, Any]) -> bool:
    return (
        next_info.get("available") is True
        and next_info.get("parses") is True
        and next_info.get("milestone") == ACTIVATED_MILESTONE
    )


def _active_443_ready(active_info: Mapping[str, Any]) -> bool:
    return active_info.get("parses") is True and active_info.get("milestone") == ACTIVATED_MILESTONE


def _activate_next_roadmap(root: Path, *, next_info: Mapping[str, Any]) -> tuple[bool, str]:
    if not _next_roadmap_ready(next_info):
        return False, ""
    next_path = root / RESEARCH_ROADMAP_NEXT_REL_PATH
    active_path = root / RESEARCH_ROADMAP_REL_PATH
    try:
        active_path.write_text(next_path.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as exc:
        return False, str(exc)
    return True, ""


def _available_game_count(root: Path) -> int:
    env_dir = root / ENVIRONMENT_FILES_REL_PATH
    if not env_dir.exists():
        return 0
    return sum(1 for path in env_dir.iterdir() if not path.name.startswith("."))


def _literal_next_precondition_command(root: Path) -> str:
    next_path = root / RESEARCH_ROADMAP_NEXT_REL_PATH
    return (
        ".venv/bin/python -c \"import yaml; yaml.safe_load(open("
        f"'{next_path}')); print('ok')\""
    )


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
    except Exception as exc:
        offline_ok = False
        offline_error = str(exc)

    activation_attempted = False
    activation_error = ""
    if offline_ok and _next_roadmap_ready(next_info):
        activation_attempted, activation_error = _activate_next_roadmap(root, next_info=next_info)

    active_info = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)
    spec_text = _read_text(root / CAPSTONE_SPEC_REL_PATH) or ""
    accepted_missing_next = (
        not _next_roadmap_ready(next_info)
        and _active_443_ready(active_info)
        and next_info.get("available") is False
    )
    roadmap_ready = _next_roadmap_ready(next_info) or accepted_missing_next
    should_run_smart_subset = (
        roadmap_ready and offline_ok and activation_error == "" and _active_443_ready(active_info)
    )
    smart_subset = smart_subset_checker(root) if should_run_smart_subset else None
    available_games = _available_game_count(root)

    return {
        "agents_md": {"path": "AGENTS.md", "available": (root / "AGENTS.md").exists()},
        "codex_or_opencode_md": {
            "path": "CODEX.md|OPENCODE.md",
            "available": (root / "CODEX.md").exists() or (root / "OPENCODE.md").exists(),
        },
        "research_roadmap_next_yaml": {
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "available": next_info["available"],
            "parses": next_info["parses"],
            "milestone": next_info["milestone"],
            "literal_precondition_command": _literal_next_precondition_command(root),
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
        "offline_arcade": {
            "available": offline_ok,
            "command": (
                ".venv/bin/python -c \"import arc_solver_kit; "
                "arc_solver_kit.offline_arcade(); print('ok')\""
            ),
            "error": offline_error,
        },
        "smart_subset_pretest_gate": _command_check(smart_subset),
        "registry": {
            "path": str(REGISTRY_REL_PATH),
            "available": registry_levels is not None,
            "reproducible_total_levels": registry_levels,
        },
        "environment_files": {
            "path": str(ENVIRONMENT_FILES_REL_PATH),
            "available": available_games > 0,
            "n_available_games": available_games,
        },
        "capstone_spec": {
            "path": str(CAPSTONE_SPEC_REL_PATH),
            "available": (root / CAPSTONE_SPEC_REL_PATH).exists(),
            "has_req_4810": "REQ-CAPSTONE-4810" in spec_text,
        },
        "capstone_4809": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "s2v2_4801": {
            "path": str(S2V2_REL_PATH),
            "available": (root / S2V2_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone_spec = _mapping(preconditions.get("capstone_spec"))
    environment = _mapping(preconditions.get("environment_files"))

    roadmap_ready = _active_443_ready(active) and (
        _next_roadmap_ready(next_info) or next_info.get("accepted_missing_because_already_active") is True
    )
    if next_info.get("activation_error"):
        return "research_roadmap_activation_error"
    if not roadmap_ready:
        return "research_roadmap_443_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4810") is not True:
        return "missing_capstone_spec_req_4810"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    if environment.get("available") is not True:
        return "environment_files"
    for key, reason in (
        ("capstone_4809", "missing_experiment_4809_capstone_v442"),
        ("s2v2_4801", "missing_experiment_4801_structural_energy_s2v2_diverse_trust_gate"),
    ):
        if _mapping(preconditions.get(key)).get("available") is not True:
            return reason
    return None


def _transition(preconditions: Mapping[str, Any], *, complete: bool) -> JsonDict:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    if not complete:
        activation_state = "blocked_missing_or_failed_precondition"
    elif next_info.get("activation_attempted") is True:
        activation_state = "activated_from_research_roadmap_next"
    else:
        activation_state = "already_activated_by_conductor"
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(complete and _active_443_ready(active)),
        "activation_state": activation_state,
        "archive_state": "archive_noop_or_already_recorded",
    }


def _poison_test_resolution(preconditions: Mapping[str, Any]) -> JsonDict:
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    combined_output = f"{smart.get('stdout_tail', '')}\n{smart.get('stderr_tail', '')}"
    current_passed = smart.get("passed")
    if current_passed is True:
        return {
            "resolved": True,
            "current_gate_passed": True,
            "poison_tests": [],
            "action": "no_poison_observed_current_gate_green",
        }
    poison_tests = []
    if current_passed is False and "1 failed" in combined_output:
        poison_tests.append(
            {
                "id": _poison_test_id(combined_output),
                "reason": "single-failure smart-subset signature matches a stale honest-verdict expectation",
                "action": "blocked_for_fix_or_quarantine_before_tail_continues",
            }
        )
    return {
        "resolved": False,
        "current_gate_passed": current_passed,
        "poison_tests": poison_tests,
        "action": "blocked_before_or_without_green_current_gate",
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
            "source": "experiment_4809_capstone_v442",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "capstone_ready",
                "s2v2_structural_energy_verdict",
                "readiness",
                "sota_handoff",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4801_structural_energy_s2v2_diverse_trust_gate",
            "path": str(S2V2_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "verifier_is_oracle",
                "live_path_reachable",
                "positive_control_passed",
                "false_negative_risk_checked",
                "energy_minus_accuracy_delta",
                "energy_minus_accuracy_delta_ci95",
                "n_effective_games",
                "min_heldout_games",
                "candidate_pool_diversity",
                "game_results",
            ],
            "sha256": file_sha256(root / S2V2_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
        {
            "source": "environment_files",
            "path": str(ENVIRONMENT_FILES_REL_PATH),
            "fields_imported": ["n_available_games"],
            "sha256": None,
        },
    ]


def _effective_game_names(s2v2: Mapping[str, Any]) -> list[str]:
    rows = s2v2.get("candidate_pool_diversity")
    if not isinstance(rows, list):
        return []
    names: list[str] = []
    for index, row in enumerate(rows):
        row_map = _mapping(row)
        if row_map.get("effective") is True:
            names.append(str(row_map.get("game", f"game_{index}")))
    return names


def _s2v2_under_covered_state(s2v2: Mapping[str, Any], n_available_games: int) -> JsonDict:
    tested_effective_games = _effective_game_names(s2v2)
    game_results = s2v2.get("game_results")
    candidate_pool = s2v2.get("candidate_pool_diversity")
    n_effective = _int(s2v2.get("n_effective_games"), len(tested_effective_games))
    n_available = _int(s2v2.get("n_available_games"), n_available_games)
    n_games_attempted = len(game_results) if isinstance(game_results, list) else len(tested_effective_games)
    tightened_min = max(10, ceil(0.6 * n_available))
    old_min = _int(s2v2.get("min_heldout_games"), 5)
    ci95 = s2v2.get("energy_minus_accuracy_delta_ci95")
    ci95 = ci95 if isinstance(ci95, list) else []
    delta = _float(s2v2.get("energy_minus_accuracy_delta"))
    coverage_fraction = round(n_effective / n_available, 12) if n_available else 0.0
    upstream_genuine_bounded = bool(
        s2v2.get("honest_verdict") == "complete_structural_energy_s2v2_bounded_diverse_pool"
        and s2v2.get("verifier_is_oracle") is False
        and s2v2.get("live_path_reachable") is True
        and s2v2.get("positive_control_passed") is True
        and s2v2.get("false_negative_risk_checked") is True
        and delta is not None
        and delta <= 0.0
        and _ci_includes_zero(ci95)
        and s2v2.get("s3_authorized") is False
        and n_effective >= old_min
    )
    return {
        "corrected_verdict": f"bounded_but_under_covered_{n_effective}_of_{n_available}",
        "reported_honest_verdict": s2v2.get("honest_verdict"),
        "reported_energy_minus_accuracy_delta": delta,
        "reported_energy_minus_accuracy_delta_ci95": ci95,
        "reported_verifier_is_oracle": s2v2.get("verifier_is_oracle") is True,
        "reported_live_path_reachable": s2v2.get("live_path_reachable") is True,
        "reported_positive_control_passed": s2v2.get("positive_control_passed") is True,
        "reported_false_negative_risk_checked": s2v2.get("false_negative_risk_checked") is True,
        "n_effective_games": n_effective,
        "n_available_games": n_available,
        "n_games_attempted": n_games_attempted,
        "candidate_pool_rows": len(candidate_pool) if isinstance(candidate_pool, list) else 0,
        "tested_effective_games": tested_effective_games,
        "old_min_effective_games_required": old_min,
        "old_effective_game_gate_passed": n_effective >= old_min,
        "effective_coverage_fraction": coverage_fraction,
        "min_effective_games_required_under_tightened_gate": tightened_min,
        "tightened_effective_game_gate_passed": n_effective >= tightened_min,
        "upstream_genuine_bounded_result": upstream_genuine_bounded,
        "s3_authorized": False,
        "s2v3_required": True,
        "reason": (
            "S2-v2 was a genuine bounded selection result, but only 5 effective games "
            "were covered out of the 25-game offline corpus."
        ),
    }


def _close_state_442(
    capstone: Mapping[str, Any],
    s2v2: Mapping[str, Any],
    registry_total_levels: int,
    n_available_games: int,
) -> JsonDict:
    capstone_s2v2 = _mapping(capstone.get("s2v2_structural_energy_verdict"))
    corrected = _s2v2_under_covered_state(s2v2, n_available_games)
    return {
        "capstone_honest_verdict": capstone.get("honest_verdict"),
        "capstone_ready": capstone.get("capstone_ready") is True,
        "capstone_reported_s2v2_verdict": capstone_s2v2.get("verdict"),
        "capstone_reported_s2v2_reason": capstone_s2v2.get("reason"),
        "capstone_recorded_degenerate_candidate_pool": (
            capstone_s2v2.get("degenerate_candidate_pool_flagged") is True
        ),
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "reproducible_total_levels": registry_total_levels,
        "s2v2_corrected_record": corrected,
        "readiness": _mapping(capstone.get("readiness")),
        "sota_handoff": _mapping(capstone.get("sota_handoff")),
    }


def _v443_pivot(s2v2_record: Mapping[str, Any]) -> JsonDict:
    return {
        "headline": "S2-v3 corpus-wide off-path trust gate",
        "task_id": "exp4811-a1",
        "source_correction": s2v2_record.get("corrected_verdict"),
        "retests_corpus_wide": True,
        "previous_effective_games": s2v2_record.get("n_effective_games"),
        "n_available_games": s2v2_record.get("n_available_games"),
        "required_effective_games": s2v2_record.get(
            "min_effective_games_required_under_tightened_gate"
        ),
        "direction": (
            "attempt all offline games and require corpus coverage before treating the "
            "engine-selection trust gate as robust"
        ),
    }


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    poison_test_resolved: Mapping[str, Any],
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
        "s2v2_recorded_as_under_covered": False,
        "reproducible_total_levels": None,
        "poison_test_resolved": dict(poison_test_resolved),
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_442": {},
        "v443_pivot": {},
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "field_principles": FIELD_PRINCIPLES,
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
    poison = _poison_test_resolution(preconditions)
    blocker = _first_blocker(preconditions)
    if blocker is not None:
        artifact = _blocked_artifact(
            reason=blocker,
            preconditions_checked=preconditions,
            poison_test_resolved=poison,
            duration_s=duration_s,
            cited_upstream_artifacts=cited,
        )
        validate_artifact(artifact)
        return artifact

    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    n_available_games = int(_mapping(preconditions["environment_files"])["n_available_games"])
    capstone = _json_object(root_path / CAPSTONE_REL_PATH)
    s2v2 = _json_object(root_path / S2V2_REL_PATH)
    close_state = _close_state_442(capstone, s2v2, registry_total_levels, n_available_games)
    s2v2_record = _mapping(close_state.get("s2v2_corrected_record"))
    transition = _transition(preconditions, complete=True)
    activation_suffix = (
        "from_next"
        if transition["activation_state"] == "activated_from_research_roadmap_next"
        else "already_active"
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": (
            f"complete_442_archived_443_activated_{activation_suffix}_"
            f"{s2v2_record.get('corrected_verdict')}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "s2v2_recorded_as_under_covered": True,
        "reproducible_total_levels": registry_total_levels,
        "poison_test_resolved": poison,
        "preconditions_checked": preconditions,
        "transition": transition,
        "close_state_442": close_state,
        "v443_pivot": _v443_pivot(s2v2_record),
        "cited_upstream_artifacts": cited,
        "field_principles": FIELD_PRINCIPLES,
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
        _require(
            verdict.startswith("complete_442_archived_443_activated_"),
            "honest_verdict must record the .442/.443 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles drifted")

    close = _mapping(artifact.get("close_state_442"))
    pivot = _mapping(artifact.get("v443_pivot"))
    poison = _mapping(artifact.get("poison_test_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(
            close == {} and pivot == {} and artifact.get("s2v2_recorded_as_under_covered") is False,
            "blocked artifacts must not carry fabricated close-state",
        )
        _validate_checksum(artifact)
        return None

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .443 milestone must be confirmed")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )

    s2v2 = _mapping(close.get("s2v2_corrected_record"))
    _require(
        artifact.get("s2v2_recorded_as_under_covered") is True
        and s2v2.get("corrected_verdict") == UNDER_COVERED_STATUS
        and s2v2.get("n_effective_games") == 5
        and s2v2.get("n_available_games") == BASELINE_AVAILABLE_GAMES
        and s2v2.get("n_games_attempted") == 5
        and s2v2.get("min_effective_games_required_under_tightened_gate") == 15
        and s2v2.get("tightened_effective_game_gate_passed") is False
        and s2v2.get("upstream_genuine_bounded_result") is True
        and s2v2.get("s2v3_required") is True,
        "S2-v2 must be recorded as bounded_but_under_covered_5_of_25",
    )
    _require(
        pivot.get("task_id") == "exp4811-a1" and pivot.get("retests_corpus_wide") is True,
        "v443 pivot must record the S2-v3 corpus-wide retest",
    )
    _validate_checksum(artifact)
    return None


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

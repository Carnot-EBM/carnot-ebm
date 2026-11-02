"""Experiment 4820: archive `.443`, activate `.444`, and record S2-v3.

Spec refs: REQ-CAPSTONE-4820, SCENARIO-CAPSTONE-4820,
SCENARIO-CAPSTONE-4820-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4820-FIELD-PRINCIPLES.

This is a record-only transition. It reads the `.443` capstone, the S2-v3
structural-energy artifact, the active roadmap, and the ARC registry. The key
record is deliberately narrow: S2-v3 settled engine selection as corpus-wide
bounded/neutral, so `.444` pivots to S3 generation. It also records the `.443`
KV260 no-file-changes failure so the corrected `.444` hardware task writes a
blocked artifact instead of exiting silently when the board is offline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4810_archive_442_activate_443 import (  # noqa: E402
    CommandResult,
    _command_check,
    _float,
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

EXPERIMENT = "experiment_4820_archive_443_activate_444"
EXPERIMENT_ID = 4820
SCHEMA = "carnot.archive_activation.v443_to_v444_4820.v1"
RESULT_RELATIVE_PATH = "results/experiment_4820_archive_443_activate_444.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4819_capstone_v443.json")
S2V3_REL_PATH = Path("results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json")
KV260_4817_REL_PATH = Path("results/experiment_4817_kv260_continuity.json")

ARCHIVED_MILESTONE = "2026.06.443"
ACTIVATED_MILESTONE = "2026.06.444"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
RANDOM_SEED = 4820
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SELECTION_SETTLED_STATUS = "selection_settled_bounded_pivot_to_s3"
PREVIOUS_S2V2_DELTA = -0.15765776352537078
PREVIOUS_S2V2_EFFECTIVE_GAMES = 5

SPEC_REFS = [
    "REQ-CAPSTONE-4820",
    "SCENARIO-CAPSTONE-4820",
    "SCENARIO-CAPSTONE-4820-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4820-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_443_archived_444_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "s2v3_selection_settled": {
        "principle": (
            "S2-v3 settled selection corpus-wide (bounded, slightly positive); "
            "the energy direction pivots to S3 generation, not paused."
        )
    },
    "kv260_offline_noted": {
        "principle": (
            "the KV260 board is offline (no route to 192.168.51.98); the .444 C task writes "
            "a blocked artifact instead of 3-fail-skipping."
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
    "s2v3_selection_settled",
    "kv260_offline_noted",
    "reproducible_total_levels",
    "poison_test_resolved",
    "preconditions_checked",
    "transition",
    "close_state_443",
    "v444_pivot",
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


def _next_roadmap_ready(next_info: Mapping[str, Any]) -> bool:
    return (
        next_info.get("available") is True
        and next_info.get("parses") is True
        and next_info.get("milestone") == ACTIVATED_MILESTONE
    )


def _active_444_ready(active_info: Mapping[str, Any]) -> bool:
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
        and _active_444_ready(active_info)
        and next_info.get("available") is False
    )
    roadmap_ready = _next_roadmap_ready(next_info) or accepted_missing_next
    should_run_smart_subset = (
        roadmap_ready and offline_ok and activation_error == "" and _active_444_ready(active_info)
    )
    smart_subset = smart_subset_checker(root) if should_run_smart_subset else None

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
            "literal_precondition_passed": next_info["available"] is True
            and next_info["parses"] is True,
            "milestone_matches_activation": _next_roadmap_ready(next_info),
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
        "capstone_spec": {
            "path": str(CAPSTONE_SPEC_REL_PATH),
            "available": (root / CAPSTONE_SPEC_REL_PATH).exists(),
            "has_req_4820": "REQ-CAPSTONE-4820" in spec_text,
        },
        "capstone_4819": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "s2v3_4811": {
            "path": str(S2V3_REL_PATH),
            "available": (root / S2V3_REL_PATH).exists(),
        },
        "kv260_4817": {
            "path": str(KV260_4817_REL_PATH),
            "available": (root / KV260_4817_REL_PATH).exists(),
            "expected_missing_no_file_changes_failure": True,
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone_spec = _mapping(preconditions.get("capstone_spec"))

    roadmap_ready = _active_444_ready(active) and (
        _next_roadmap_ready(next_info) or next_info.get("accepted_missing_because_already_active") is True
    )
    if next_info.get("activation_error"):
        return "research_roadmap_activation_error"
    if not roadmap_ready:
        return "research_roadmap_444_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4820") is not True:
        return "missing_capstone_spec_req_4820"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    for key, reason in (
        ("capstone_4819", "missing_experiment_4819_capstone_v443"),
        ("s2v3_4811", "missing_experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate"),
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
        "active_milestone_confirmed": bool(complete and _active_444_ready(active)),
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
                "reason": "single-failure smart-subset signature matches a stale transition expectation",
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
            "fields_imported": ["milestone", "exp4820_prompt", "exp4827_kv260_correction"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_REL_PATH),
        },
        {
            "source": "research_roadmap_next_yaml",
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "fields_imported": ["milestone"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_NEXT_REL_PATH),
        },
        {
            "source": "experiment_4819_capstone_v443",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "capstone_ready",
                "s2v3_structural_energy_verdict",
                "readiness",
                "sota_handoff",
                "submission_package_state",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
            "path": str(S2V3_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "DEGENERATE_CANDIDATE_POOL",
                "verifier_is_oracle",
                "live_path_reachable",
                "positive_control_passed",
                "false_negative_risk_checked",
                "energy_minus_accuracy_delta",
                "energy_minus_accuracy_delta_ci95",
                "n_available_games",
                "n_games_attempted",
                "n_effective_games",
                "required_effective_games",
            ],
            "sha256": file_sha256(root / S2V3_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
        {
            "source": "experiment_4817_kv260_continuity_missing",
            "path": str(KV260_4817_REL_PATH),
            "fields_imported": ["no_file_changes_failure_note"],
            "sha256": file_sha256(root / KV260_4817_REL_PATH),
        },
    ]


def _ci_includes_zero(ci95: Any) -> bool:
    return isinstance(ci95, list) and len(ci95) == 2 and float(ci95[0]) <= 0.0 <= float(ci95[1])


def _s2v3_selection_state(capstone: Mapping[str, Any], s2v3: Mapping[str, Any]) -> JsonDict:
    capstone_s2v3 = _mapping(capstone.get("s2v3_structural_energy_verdict"))
    ci95 = s2v3.get("energy_minus_accuracy_delta_ci95")
    ci95 = ci95 if isinstance(ci95, list) else capstone_s2v3.get("reported_energy_minus_accuracy_delta_ci95", [])
    delta = _float(s2v3.get("energy_minus_accuracy_delta"), None)
    n_available = _int(s2v3.get("n_available_games"), _int(capstone_s2v3.get("n_available_games"), 0))
    n_attempted = _int(s2v3.get("n_games_attempted"), _int(capstone_s2v3.get("n_games_attempted"), 0))
    n_effective = _int(s2v3.get("n_effective_games"), _int(capstone_s2v3.get("n_effective_games"), 0))
    required = _int(s2v3.get("required_effective_games"), _int(capstone_s2v3.get("required_effective_games"), 0))
    ci_crosses_zero = _ci_includes_zero(ci95)
    degenerate_fired = bool(
        s2v3.get("DEGENERATE_CANDIDATE_POOL") is True
        or capstone_s2v3.get("degenerate_candidate_pool_flagged") is True
    )
    coverage_floor_met = n_effective >= required
    selection_settled = bool(
        s2v3.get("honest_verdict") == "complete_structural_energy_s2v3_bounded_corpus_wide"
        and s2v3.get("verifier_is_oracle") is False
        and s2v3.get("live_path_reachable") is True
        and s2v3.get("positive_control_passed") is True
        and coverage_floor_met
        and not degenerate_fired
        and delta is not None
        and delta > 0.0
        and ci_crosses_zero
    )
    return {
        "selection_status": SELECTION_SETTLED_STATUS if selection_settled else "not_settled",
        "verdict": "bounded_corpus_wide" if selection_settled else "not_bounded_corpus_wide",
        "reported_honest_verdict": s2v3.get("honest_verdict"),
        "capstone_reported_verdict": capstone_s2v3.get("verdict"),
        "reported_energy_minus_accuracy_delta": delta,
        "reported_energy_minus_accuracy_delta_ci95": ci95,
        "ci_includes_zero": ci_crosses_zero,
        "n_available_games": n_available,
        "n_games_attempted": n_attempted,
        "n_effective_games": n_effective,
        "required_effective_games": required,
        "coverage_floor_met": coverage_floor_met,
        "positive_control_passed": s2v3.get("positive_control_passed") is True,
        "false_negative_risk_checked": s2v3.get("false_negative_risk_checked") is True,
        "verifier_is_oracle": s2v3.get("verifier_is_oracle") is True,
        "live_path_reachable": s2v3.get("live_path_reachable") is True,
        "degenerate_candidate_pool_fired": degenerate_fired,
        "previous_s2v2_delta_at_n5": PREVIOUS_S2V2_DELTA,
        "previous_s2v2_effective_games": PREVIOUS_S2V2_EFFECTIVE_GAMES,
        "point_estimate_flipped_from_s2v2": bool(delta is not None and PREVIOUS_S2V2_DELTA < 0.0 < delta),
        "energy_direction": "roughly_neutral_at_engine_selection",
        "s3_authorized_by_gate": False,
        "pivot": "S3_generation_lift",
        "reason": (
            "S2-v3 settled engine selection corpus-wide as bounded/neutral: the point "
            "estimate is slightly positive but CI95 includes zero."
        ),
    }


def _kv260_offline_note() -> JsonDict:
    return {
        "experiment_id": 4817,
        "task_id": "exp4817-c",
        "failed_attempts": 3,
        "failure_mode": "no_file_changes",
        "board_offline": True,
        "board_address": "192.168.51.98",
        "ssh_failure": "no route to 192.168.51.98",
        "blocked_artifact_was_written": False,
        "v444_corrected_task_id": "exp4827-c",
        "v444_c_task_corrected_to_write_blocked_artifact": True,
    }


def _close_state_443(
    capstone: Mapping[str, Any],
    s2v3: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    return {
        "capstone_honest_verdict": capstone.get("honest_verdict"),
        "capstone_ready": capstone.get("capstone_ready") is True,
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "reproducible_total_levels": registry_total_levels,
        "s2v3_record": _s2v3_selection_state(capstone, s2v3),
        "kv260_offline_note": _kv260_offline_note(),
        "readiness": _mapping(capstone.get("readiness")),
        "sota_handoff": _mapping(capstone.get("sota_handoff")),
        "submission_package_state": _mapping(capstone.get("submission_package_state")),
    }


def _v444_pivot(s2v3_record: Mapping[str, Any]) -> JsonDict:
    return {
        "headline": "S3 generation lift",
        "task_id": "exp4821-a1",
        "source_selection_status": s2v3_record.get("selection_status"),
        "direction": "S3_generation_lift",
        "mechanism": "generation_not_engine_selection",
        "selection_energy_direction": s2v3_record.get("energy_direction"),
        "reason": "S2-v3 bounded selection result authorizes testing whether energy helps generation.",
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
        "s2v3_selection_settled": False,
        "kv260_offline_noted": False,
        "reproducible_total_levels": None,
        "poison_test_resolved": dict(poison_test_resolved),
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_443": {},
        "v444_pivot": {},
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
    capstone = _json_object(root_path / CAPSTONE_REL_PATH)
    s2v3 = _json_object(root_path / S2V3_REL_PATH)
    close_state = _close_state_443(capstone, s2v3, registry_total_levels)
    s2v3_record = _mapping(close_state.get("s2v3_record"))
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
            f"complete_443_archived_444_activated_{activation_suffix}_"
            f"{s2v3_record.get('selection_status')}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "s2v3_selection_settled": True,
        "kv260_offline_noted": True,
        "reproducible_total_levels": registry_total_levels,
        "poison_test_resolved": poison,
        "preconditions_checked": preconditions,
        "transition": transition,
        "close_state_443": close_state,
        "v444_pivot": _v444_pivot(s2v3_record),
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
            verdict.startswith("complete_443_archived_444_activated_"),
            "honest_verdict must record the .443/.444 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles drifted")

    close = _mapping(artifact.get("close_state_443"))
    pivot = _mapping(artifact.get("v444_pivot"))
    poison = _mapping(artifact.get("poison_test_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(
            close == {}
            and pivot == {}
            and artifact.get("s2v3_selection_settled") is False
            and artifact.get("kv260_offline_noted") is False,
            "blocked artifacts must not carry fabricated close-state",
        )
        _validate_checksum(artifact)
        return None

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .444 milestone must be confirmed")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )

    s2v3 = _mapping(close.get("s2v3_record"))
    _require(
        artifact.get("s2v3_selection_settled") is True
        and s2v3.get("selection_status") == SELECTION_SETTLED_STATUS
        and s2v3.get("verdict") == "bounded_corpus_wide"
        and s2v3.get("n_available_games") == 25
        and s2v3.get("n_games_attempted") == 25
        and s2v3.get("n_effective_games") == 23
        and s2v3.get("required_effective_games") == 15
        and s2v3.get("coverage_floor_met") is True
        and s2v3.get("positive_control_passed") is True
        and s2v3.get("degenerate_candidate_pool_fired") is False
        and s2v3.get("ci_includes_zero") is True
        and s2v3.get("point_estimate_flipped_from_s2v2") is True
        and s2v3.get("pivot") == "S3_generation_lift",
        "S2-v3 must be recorded as selection_settled_bounded_pivot_to_s3",
    )

    kv260 = _mapping(close.get("kv260_offline_note"))
    _require(
        artifact.get("kv260_offline_noted") is True
        and kv260.get("experiment_id") == 4817
        and kv260.get("failed_attempts") == 3
        and kv260.get("failure_mode") == "no_file_changes"
        and kv260.get("board_offline") is True
        and kv260.get("blocked_artifact_was_written") is False
        and kv260.get("v444_c_task_corrected_to_write_blocked_artifact") is True,
        "KV260 offline/no-file-changes note must be preserved",
    )
    _require(
        pivot.get("task_id") == "exp4821-a1" and pivot.get("direction") == "S3_generation_lift",
        "v444 pivot must record the S3 generation task",
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

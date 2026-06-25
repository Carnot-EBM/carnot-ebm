"""Experiment 4748: archive `.436`, activate `.437`, and record `.436` honestly.

Spec refs: REQ-CAPSTONE-4748, SCENARIO-CAPSTONE-4748,
SCENARIO-CAPSTONE-4748-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4748-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import re
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

EXPERIMENT = "experiment_4748_archive_436_activate_437"
EXPERIMENT_ID = 4748
SCHEMA = "carnot.archive_activation.v436_to_v437_4748.v1"
RESULT_RELATIVE_PATH = "results/experiment_4748_archive_436_activate_437.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4747_capstone_v436.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")

ARCHIVED_MILESTONE = "2026.06.436"
ACTIVATED_MILESTONE = "2026.06.437"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 64
RANDOM_SEED = 4748
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4748",
    "SCENARIO-CAPSTONE-4748",
    "SCENARIO-CAPSTONE-4748-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4748-FIELD-PRINCIPLES",
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
    "poison_pretest_resolved": {
        "principle": (
            "any poison pre-test (id + reason) that would cascade-skip the .437 tail is "
            "fixed/quarantined."
        )
    },
    "close_state_436": {
        "principle": "the honest .436 numbers carried forward from the capstone so the record does not drift."
    },
    "v437_pivot": {
        "principle": (
            "the .437 headline (FIX induction-quality: A1 structured engine, A2 "
            "structural-alignment detector) recorded so milestone intent is traceable."
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
    "poison_pretest_resolved",
    "transition",
    "close_state_436",
    "v437_pivot",
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


def _json_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


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
    return (
        next_info.get("available") is True
        and next_info.get("parses") is True
        and next_info.get("milestone") == ACTIVATED_MILESTONE
    )


def _active_437_ready(active_info: Mapping[str, Any]) -> bool:
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


def _poison_signature(text: str) -> str:
    match = re.search(r"1 failed,\s*\d+ passed[^|\n]*", text)
    return match.group(0).strip() if match else ""


def _transition_log_scope(text: str) -> str:
    marker = "Milestone 2026.06.436 activated"
    marker_index = text.rfind(marker)
    return text[marker_index:] if marker_index >= 0 else text


def _poison_test_id(text: str) -> str:
    match = re.search(r"\b(test_[A-Za-z0-9_]+)\b", text)
    return match.group(1) if match else "unknown_focused_pretest"


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
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)
    spec_text = _read_text(root / CAPSTONE_SPEC_REL_PATH) or ""
    conductor_text = _read_text(root / CONDUCTOR_LOG_REL_PATH) or ""
    conductor_scope = _transition_log_scope(conductor_text)
    research_complete_text = _read_text(root / RESEARCH_COMPLETE_REL_PATH) or ""
    accepted_missing_next = (
        not _next_roadmap_ready(next_info)
        and _active_437_ready(active_info)
        and next_info.get("available") is False
    )
    roadmap_ready = _next_roadmap_ready(next_info) or accepted_missing_next
    should_run_smart_subset = roadmap_ready and offline_ok and activation_error == ""
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
            "available": (root / RESEARCH_COMPLETE_REL_PATH).exists(),
            "contains_2026_06_436": ARCHIVED_MILESTONE in research_complete_text,
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
        "capstone_spec": {
            "path": str(CAPSTONE_SPEC_REL_PATH),
            "available": (root / CAPSTONE_SPEC_REL_PATH).exists(),
            "has_req_4748": "REQ-CAPSTONE-4748" in spec_text,
        },
        "capstone_4747": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "conductor_log": {
            "path": str(CONDUCTOR_LOG_REL_PATH),
            "available": (root / CONDUCTOR_LOG_REL_PATH).exists(),
            "poison_scan_scope": "after_2026.06.436_activation_when_marker_present",
            "poison_signature_observed": bool(_poison_signature(conductor_scope)),
            "poison_signature": _poison_signature(conductor_scope),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone_spec = _mapping(preconditions.get("capstone_spec"))

    roadmap_ready = _next_roadmap_ready(next_info) or (
        next_info.get("accepted_missing_because_already_active") is True and _active_437_ready(active)
    )
    if not roadmap_ready:
        return "research_roadmap_437_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4748") is not True:
        return "missing_capstone_spec_req_4748"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_64"
    for key, reason in (
        ("capstone_4747", "missing_experiment_4747_capstone_v436"),
        ("conductor_log", "missing_conductor_log"),
    ):
        if _mapping(preconditions.get(key)).get("available") is not True:
            return reason
    return None


def _transition(preconditions: Mapping[str, Any], *, complete: bool) -> JsonDict:
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    archive = _mapping(preconditions.get("research_complete_yaml"))
    if not complete:
        activation_state = "blocked_missing_or_failed_precondition"
    elif next_info.get("activation_attempted") is True:
        activation_state = "activated_from_research_roadmap_next"
    else:
        activation_state = "already_activated_by_conductor"
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(complete and _active_437_ready(active)),
        "activation_state": activation_state,
        "archive_state": (
            "research_complete_contains_2026.06.436"
            if archive.get("contains_2026_06_436") is True
            else "archive_noop_or_already_recorded"
        ),
    }


def _poison_pretest_resolution(preconditions: Mapping[str, Any]) -> JsonDict:
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    conductor = _mapping(preconditions.get("conductor_log"))
    combined_output = f"{smart.get('stdout_tail', '')}\n{smart.get('stderr_tail', '')}"
    historical_signature = str(conductor.get("poison_signature") or _poison_signature(combined_output))
    historical_observed = bool(conductor.get("poison_signature_observed") or historical_signature)
    current_passed = smart.get("passed")
    if current_passed is True:
        return {
            "resolved": True,
            "current_gate_passed": True,
            "historical_signature_observed": historical_observed,
            "historical_signature": historical_signature,
            "poison_tests": [],
            "action": (
                "historical_poison_signature_resolved_current_gate_green"
                if historical_observed
                else "no_poison_observed_current_gate_green"
            ),
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
        "historical_signature_observed": historical_observed,
        "historical_signature": historical_signature,
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
            "source": "experiment_4747_capstone_v436",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "bridge_crossed_for_solve",
                "reproducible_total_levels",
                "reproducible_total_levels_delta",
                "a1_goal_energy_result",
                "a2_energy_qd_result",
                "a3_banked_level",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
        {
            "source": "conductor_log",
            "path": str(CONDUCTOR_LOG_REL_PATH),
            "fields_imported": ["poison pre-test signature", "activation rows"],
            "sha256": file_sha256(root / CONDUCTOR_LOG_REL_PATH),
        },
    ]


def _close_state_436(capstone: Mapping[str, Any], registry_total_levels: int) -> JsonDict:
    a1 = _mapping(capstone.get("a1_goal_energy_result"))
    a2 = _mapping(capstone.get("a2_energy_qd_result"))
    a3 = _mapping(capstone.get("a3_banked_level"))
    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "bridge_crossed_for_solve": capstone.get("bridge_crossed_for_solve") is True,
        "reproducible_total_levels": registry_total_levels,
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "reproducible_total_levels_delta": _int(capstone.get("reproducible_total_levels_delta")),
        "a1_guidance_class_generation": {
            "arms_non_degenerate": a1.get("arms_non_degenerate") is True,
            "beat_baseline_by_0_05": a1.get("beat_baseline_by_0_05") is True,
            "deepened_to_l2": a1.get("deepened_to_l2") is True,
            "generated": a1.get("generated") is True,
            "banked": a1.get("banked") is True,
            "baseline_first_win": _float(a1.get("baseline_first_win")),
            "goal_energy_first_win": _float(a1.get("goal_energy_first_win")),
            "goal_energy_vs_baseline_delta": _float(a1.get("goal_energy_vs_baseline_delta")),
            "offline_reproduced": a1.get("offline_reproduced") is True,
            "reproduced_levels": _int(a1.get("reproduced_levels")),
            "solve_provenance": str(a1.get("solve_provenance", "")),
            "reason": str(a1.get("reason", "")),
        },
        "a2_guidance_class_generation": {
            "arms_non_degenerate": a2.get("arms_non_degenerate") is True,
            "generated_winner_where_naive_missed": a2.get("generated_winner_where_naive_missed") is True,
            "deepened_to_l2": a2.get("deepened_to_l2") is True,
            "generated": a2.get("generated") is True,
            "banked": a2.get("banked") is True,
            "novel_candidates_generated": _int(a2.get("novel_candidates_generated")),
            "naive_search_first_win": _float(a2.get("naive_search_first_win")),
            "energy_qd_first_win": _float(a2.get("energy_qd_first_win")),
            "energy_qd_vs_naive_delta": _float(a2.get("energy_qd_vs_naive_delta")),
            "offline_reproduced": a2.get("offline_reproduced") is True,
            "reproduced_levels": _int(a2.get("reproduced_levels")),
            "target_game": str(a2.get("target_game", "")),
            "solve_provenance": str(a2.get("solve_provenance", "")),
            "reason": str(a2.get("reason", "")),
        },
        "a3_level_up_guarantee": {
            "banked": a3.get("banked") is True,
            "new_levels_banked": _int(a3.get("new_levels_banked")),
            "reproducible_total_levels_before": _int(
                a3.get("reproducible_total_levels_before"), BASELINE_REPRODUCIBLE_TOTAL_LEVELS
            ),
            "reproducible_total_levels_after": _int(
                a3.get("reproducible_total_levels_after"), registry_total_levels
            ),
            "reproduced_levels": _int(a3.get("reproduced_levels")),
            "target_game": str(a3.get("target_game", "")),
            "reason": str(a3.get("reason", "")),
        },
        "net_436": {
            "bridge_crossed_for_solve": capstone.get("bridge_crossed_for_solve") is True,
            "capability_grew": _int(capstone.get("reproducible_total_levels_delta")) > 0,
            "guidance_class_generation_validly_tested": a1.get("arms_non_degenerate") is True
            and a2.get("arms_non_degenerate") is True,
            "registry_total_after": registry_total_levels,
        },
    }


def _v437_pivot() -> JsonDict:
    return {
        "headline": "FIX induction-quality wall",
        "a1_structured_engine": {
            "action": "wire existing ProductWorldModel programmatic experts as the action-effect engine",
            "replaces": "0.12-accurate free-form codex engine",
            "existing_scaffold": "python/carnot/agentic/arc_executable_world_model.py:ProductWorldModel",
        },
        "a2_structural_alignment_detector": {
            "action": "fix perception-grounded structural-alignment detector segmentation and pairing",
            "resolves": "exp4712 over-segmentation: goal_count=42 aligned_piece_count=0",
            "existing_scaffold": "python/carnot/agentic/arc_value_learner.py structural alignment pipeline",
        },
        "retired_retries": ["pure prompt-engineering retry", "retired CNN driver"],
    }


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    poison_pretest_resolved: Mapping[str, Any],
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
        "poison_pretest_resolved": dict(poison_pretest_resolved),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_436": {},
        "v437_pivot": {},
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
    poison = _poison_pretest_resolution(preconditions)
    blocker = _first_blocker(preconditions)
    if blocker is not None:
        artifact = _blocked_artifact(
            reason=blocker,
            preconditions_checked=preconditions,
            poison_pretest_resolved=poison,
            duration_s=duration_s,
            cited_upstream_artifacts=cited,
        )
        validate_artifact(artifact)
        return artifact

    registry_total_levels = int(_mapping(preconditions["registry"])["reproducible_total_levels"])
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete: archive_436_activate_437_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "poison_pretest_resolved": poison,
        "transition": _transition(preconditions, complete=True),
        "close_state_436": _close_state_436(
            _json_object(root_path / CAPSTONE_REL_PATH),
            registry_total_levels,
        ),
        "v437_pivot": _v437_pivot(),
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

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_provenance") == FIELD_PROVENANCE, "field_provenance principles drifted")

    close = _mapping(artifact.get("close_state_436"))
    pivot = _mapping(artifact.get("v437_pivot"))
    poison = _mapping(artifact.get("poison_pretest_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .437 milestone must be confirmed")
    _require(close.get("bridge_crossed_for_solve") is False, "bridge must remain false for .436")
    _require(
        close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels_delta") == 0,
        "registry total and zero delta must be recorded",
    )

    a1 = _mapping(close.get("a1_guidance_class_generation"))
    _require(
        a1.get("arms_non_degenerate") is True
        and a1.get("beat_baseline_by_0_05") is False
        and a1.get("deepened_to_l2") is False
        and a1.get("reason") == "goal_energy_real_non_degenerate_zero_lift_null",
        "A1 non-degenerate zero-lift null must be recorded",
    )

    a2 = _mapping(close.get("a2_guidance_class_generation"))
    _require(
        a2.get("arms_non_degenerate") is True
        and a2.get("generated_winner_where_naive_missed") is False
        and a2.get("novel_candidates_generated") == 8
        and a2.get("reason") == "energy_qd_real_non_degenerate_zero_lift_null",
        "A2 non-degenerate zero-lift null must be recorded",
    )

    net = _mapping(close.get("net_436"))
    _require(
        net.get("bridge_crossed_for_solve") is False
        and net.get("capability_grew") is False
        and net.get("guidance_class_generation_validly_tested") is True,
        ".436 net close-state must preserve no bridge, no capability growth, valid tests",
    )

    _require(
        pivot.get("headline") == "FIX induction-quality wall"
        and "ProductWorldModel" in str(_mapping(pivot.get("a1_structured_engine")).get("action"))
        and "structural-alignment detector"
        in str(_mapping(pivot.get("a2_structural_alignment_detector")).get("action")),
        "v437 pivot must record structured engine and structural-alignment detector",
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

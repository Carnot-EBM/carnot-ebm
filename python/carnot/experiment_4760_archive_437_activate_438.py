"""Experiment 4760: archive `.437`, activate `.438`, and record `.437` honestly.

Spec refs: REQ-CAPSTONE-4760, SCENARIO-CAPSTONE-4760,
SCENARIO-CAPSTONE-4760-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4760-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4760_archive_437_activate_438"
EXPERIMENT_ID = 4760
SCHEMA = "carnot.archive_activation.v437_to_v438_4760.v1"
RESULT_RELATIVE_PATH = "results/experiment_4760_archive_437_activate_438.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4759_capstone_v437.json")
A1_REL_PATH = Path("results/experiment_4749_structured_engine_vs_freeform.json")
A2_REL_PATH = Path("results/experiment_4750_structural_alignment_detector_fix.json")
A3_REL_PATH = Path("results/experiment_4751_levelup_selfplay.json")
A4_REL_PATH = Path("results/experiment_4752_held_out_first_win_readiness.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")

ARCHIVED_MILESTONE = "2026.06.437"
ACTIVATED_MILESTONE = "2026.06.438"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
RANDOM_SEED = 4760
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4760",
    "SCENARIO-CAPSTONE-4760",
    "SCENARIO-CAPSTONE-4760-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4760-FIELD-PRINCIPLES",
]

FIELD_PROVENANCE = {
    "honest_verdict": {
        "principle": "terminal prefix; a clean transition is complete_437_archived_438_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream JSON, no LLM; 0.0001s floor."
    },
    "poison_test_resolved": {
        "principle": "records whether a poison pre-test was found+fixed -- the .434/.435 cascade-skip guard."
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric carried forward from arc_solve_registry, not re-counted."
        )
    },
    "close_state_437": {
        "principle": "the honest .437 numbers carried forward from the capstone and clean/flagged upstreams so the record does not drift."
    },
    "v438_pivot": {
        "principle": (
            "the .438 headline rationale (oracle-distinct structural-energy S0 core-bet probe over held-out "
            "transition-correctness) recorded so milestone intent is traceable."
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
    "poison_test_resolved",
    "reproducible_total_levels",
    "preconditions_checked",
    "transition",
    "close_state_437",
    "v438_pivot",
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


def _active_438_ready(active_info: Mapping[str, Any]) -> bool:
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


def _poison_signature(text: str) -> str:
    match = re.search(r"1 failed,\s*\d+ passed[^|\n]*", text)
    return match.group(0).strip() if match else ""


def _transition_log_scope(text: str) -> str:
    marker = "Milestone 2026.06.437 activated"
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
    conductor_text = _read_text(root / CONDUCTOR_LOG_REL_PATH) or ""
    conductor_scope = _transition_log_scope(conductor_text)
    accepted_missing_next = (
        not _next_roadmap_ready(next_info)
        and _active_438_ready(active_info)
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
            "has_req_4760": "REQ-CAPSTONE-4760" in spec_text,
        },
        "capstone_4759": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "a1_4749": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_4750": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a3_4751": {"path": str(A3_REL_PATH), "available": (root / A3_REL_PATH).exists()},
        "a4_4752": {"path": str(A4_REL_PATH), "available": (root / A4_REL_PATH).exists()},
        "conductor_log": {
            "path": str(CONDUCTOR_LOG_REL_PATH),
            "available": (root / CONDUCTOR_LOG_REL_PATH).exists(),
            "poison_scan_scope": "after_2026.06.437_activation_when_marker_present",
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
        next_info.get("accepted_missing_because_already_active") is True and _active_438_ready(active)
    )
    if not roadmap_ready:
        return "research_roadmap_438_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4760") is not True:
        return "missing_capstone_spec_req_4760"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    for key, reason in (
        ("capstone_4759", "missing_experiment_4759_capstone_v437"),
        ("a1_4749", "missing_experiment_4749_structured_engine_vs_freeform"),
        ("a2_4750", "missing_experiment_4750_structural_alignment_detector_fix"),
        ("a3_4751", "missing_experiment_4751_levelup_selfplay"),
        ("a4_4752", "missing_experiment_4752_held_out_first_win_readiness"),
        ("conductor_log", "missing_conductor_log"),
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
        "active_milestone_confirmed": bool(complete and _active_438_ready(active)),
        "activation_state": activation_state,
        "archive_state": "archive_noop_or_already_recorded",
    }


def _poison_test_resolution(preconditions: Mapping[str, Any]) -> JsonDict:
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
            "source": "experiment_4759_capstone_v437",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "bridge_crossed_for_solve",
                "reproducible_total_levels",
                "induction_quality_decision",
                "scorecard",
                "skipped_artifacts",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4749_structured_engine_vs_freeform",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "flagged_adversarial",
                "structured_engine_non_degenerate",
                "structured_heldout_accuracy",
                "freeform_heldout_accuracy",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4750_structural_alignment_detector_fix",
            "path": str(A2_REL_PATH),
            "fields_imported": [
                "goal_predicate_satisfiable",
                "l2_plan_reaches_goal",
                "offline_reproduced",
                "reproduced_levels",
            ],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4751_levelup_selfplay",
            "path": str(A3_REL_PATH),
            "fields_imported": [
                "target_game",
                "new_levels_banked",
                "reached_level",
                "offline_reproduced",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / A3_REL_PATH),
        },
        {
            "source": "experiment_4752_held_out_first_win_readiness",
            "path": str(A4_REL_PATH),
            "fields_imported": ["flagged_adversarial", "submission_package_ready", "first_win_rate_integrated"],
            "sha256": file_sha256(root / A4_REL_PATH),
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


def _close_state_437(
    capstone: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    capstone_induction = _mapping(capstone.get("induction_quality_decision"))
    capstone_a1 = _mapping(capstone_induction.get("a1"))
    capstone_a2 = _mapping(capstone_induction.get("a2"))
    capstone_a3 = _mapping(_mapping(capstone.get("scorecard")).get("A3"))
    skipped = capstone.get("skipped_artifacts")
    flagged_tasks = skipped if isinstance(skipped, list) else []
    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "bridge_crossed_for_solve": capstone.get("bridge_crossed_for_solve") is True,
        "reproducible_total_levels": registry_total_levels,
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "a1_structured_engine": {
            "decision": str(capstone_a1.get("decision", "")),
            "artifact_flagged_adversarial": a1.get("flagged_adversarial") is True,
            "structured_engine_non_degenerate": a1.get("structured_engine_non_degenerate") is True,
            "structured_heldout_accuracy": _float(a1.get("structured_heldout_accuracy")),
            "freeform_heldout_accuracy": _float(a1.get("freeform_heldout_accuracy")),
            "forward_claim_status": (
                "quarantined_not_forward_claim"
                if a1.get("flagged_adversarial") is True
                else "clean_artifact_available"
            ),
        },
        "a2_detector_fix": {
            "decision": str(capstone_a2.get("decision") or a2.get("honest_verdict", "")),
            "detector_goal_count": _int(a2.get("detector_goal_count")),
            "detector_piece_count": _int(a2.get("detector_piece_count")),
            "detector_raw_goal_count": _int(a2.get("detector_raw_goal_count")),
            "goal_predicate_satisfiable": a2.get("goal_predicate_satisfiable") is True,
            "l2_plan_reaches_goal": a2.get("l2_plan_reaches_goal") is True,
            "offline_reproduced": a2.get("offline_reproduced") is True,
            "reproduced_levels": _int(a2.get("reproduced_levels")),
            "modest_result": "detector_fixed_no_satisfiable_goal_no_bank",
        },
        "a3_levelup": {
            "decision": str(capstone_a3.get("decision") or a3.get("honest_verdict", "")),
            "target_game": str(capstone_a3.get("target_game") or a3.get("target_game", "")),
            "new_levels_banked": _int(capstone_a3.get("new_levels_banked"), _int(a3.get("new_levels_banked"))),
            "reached_level": _int(capstone_a3.get("reached_level"), _int(a3.get("reached_level"))),
            "offline_reproduced": capstone_a3.get("offline_reproduced") is True
            or a3.get("offline_reproduced") is True,
        },
        "a4_readiness": {
            "artifact_flagged_adversarial": a4.get("flagged_adversarial") is True,
            "submission_package_ready": capstone.get("submission_package_ready") is True
            or a4.get("submission_package_ready") is True,
            "first_win_rate_integrated": _float(a4.get("first_win_rate_integrated")),
            "forward_claim_status": (
                "quarantined_not_forward_claim"
                if a4.get("flagged_adversarial") is True
                else "clean_artifact_available"
            ),
        },
        "flagged_tasks": flagged_tasks,
        "net_437": {
            "bridge_crossed_for_solve": capstone.get("bridge_crossed_for_solve") is True,
            "capability_grew": registry_total_levels > 64,
            "induction_quality_wall_cleared": capstone_induction.get("cleared_induction_quality_wall") is True,
            "registry_total_after": registry_total_levels,
        },
    }


def _v438_pivot() -> JsonDict:
    return {
        "headline": "oracle-distinct structural energy S0 core-bet probe",
        "primary_probe": {
            "target": "held-out transition-correctness cross-game above chance",
            "must_survive": ["oracle-distinctness", "frame-marginal collapse"],
            "falsifiable_gate": "LOO AUROC CI95 lower bound > 0.5 and point > 0.60",
        },
        "follow_on_slots": [
            "level-up attempt",
            "self-play checkpoint",
            "held-out readiness",
            "reserved infra",
            "hardware continuity",
            "SOTA ingestion",
        ],
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
        "poison_test_resolved": dict(poison_test_resolved),
        "reproducible_total_levels": None,
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_437": {},
        "v438_pivot": {},
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
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete_437_archived_438_activated_already_active_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "poison_test_resolved": poison,
        "reproducible_total_levels": registry_total_levels,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_437": _close_state_437(
            _json_object(root_path / CAPSTONE_REL_PATH),
            _json_object(root_path / A1_REL_PATH),
            _json_object(root_path / A2_REL_PATH),
            _json_object(root_path / A3_REL_PATH),
            _json_object(root_path / A4_REL_PATH),
            registry_total_levels,
        ),
        "v438_pivot": _v438_pivot(),
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
            verdict.startswith("complete_437_archived_438_activated_"),
            "honest_verdict must record the .437/.438 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_provenance") == FIELD_PROVENANCE, "field_provenance principles drifted")

    close = _mapping(artifact.get("close_state_437"))
    pivot = _mapping(artifact.get("v438_pivot"))
    poison = _mapping(artifact.get("poison_test_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .438 milestone must be confirmed")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )
    _require(close.get("bridge_crossed_for_solve") is False, "bridge must remain false for .437")

    a1 = _mapping(close.get("a1_structured_engine"))
    _require(
        a1.get("decision") == "skipped_flagged_adversarial"
        and a1.get("artifact_flagged_adversarial") is True
        and a1.get("forward_claim_status") == "quarantined_not_forward_claim",
        "A1 structural-engine flagged null must be quarantined",
    )

    a2 = _mapping(close.get("a2_detector_fix"))
    _require(
        a2.get("goal_predicate_satisfiable") is False
        and a2.get("l2_plan_reaches_goal") is False
        and a2.get("offline_reproduced") is False
        and a2.get("modest_result") == "detector_fixed_no_satisfiable_goal_no_bank",
        "A2 detector-fix modest no-bank result must be recorded",
    )

    a3 = _mapping(close.get("a3_levelup"))
    _require(
        a3.get("target_game") == "sk48"
        and a3.get("new_levels_banked") == 1
        and a3.get("offline_reproduced") is True,
        "A3 sk48 L2 bank must be recorded",
    )

    flagged = close.get("flagged_tasks")
    _require(isinstance(flagged, list) and len(flagged) >= 1, "flagged tasks must be recorded")
    _require(
        pivot.get("headline") == "oracle-distinct structural energy S0 core-bet probe",
        "v438 pivot must record structural-energy S0",
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

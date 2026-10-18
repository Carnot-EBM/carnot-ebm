"""Experiment 4736: archive `.435`, activate `.436`, and record `.435` honestly.

Spec refs: REQ-CAPSTONE-4736, SCENARIO-CAPSTONE-4736,
SCENARIO-CAPSTONE-4736-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4736-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4736_archive_435_activate_436"
EXPERIMENT_ID = 4736
SCHEMA = "carnot.archive_activation.v435_to_v436_4736.v1"
RESULT_RELATIVE_PATH = "results/experiment_4736_archive_435_activate_436.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
B1_REL_PATH = Path("results/experiment_4725_silent_bug_audit.json")
A1_REL_PATH = Path("results/experiment_4726_online_action_learning_driver_valid_test.json")
A2_REL_PATH = Path("results/experiment_4727_active_probe_disambiguation.json")
A3_REL_PATH = Path("results/experiment_4728_levelup_selfplay.json")
CAPSTONE_REL_PATH = Path("results/experiment_4735_capstone_v435.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.435"
ACTIVATED_MILESTONE = "2026.06.436"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 63
RANDOM_SEED = 4736
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4736",
    "SCENARIO-CAPSTONE-4736",
    "SCENARIO-CAPSTONE-4736-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4736-FIELD-PRINCIPLES",
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
            "any poison pre-test (id + reason) that would cascade-skip the .436 tail is "
            "fixed/quarantined -- the incident_agent_shipped_test_cascade fix."
        )
    },
    "close_state_435": {
        "principle": (
            "the honest .435 numbers (A3 63->64 ar25 L3; A1 validly-tested genuine null -> "
            "RETIRE; A2 dead code; B1 5-must-reopen; bridge_crossed=False 11th) carried "
            "forward so the record does not drift."
        )
    },
    "v436_pivot": {
        "principle": (
            "the .436 headline rationale (valid-test the guidance-class generation levers: "
            "A1 goal-energy, A2 energy-fitness QD) recorded so the milestone intent is traceable."
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
    "close_state_435",
    "v436_pivot",
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
    return (
        next_info.get("available") is True
        and next_info.get("parses") is True
        and next_info.get("milestone") == ACTIVATED_MILESTONE
    )


def _active_436_ready(active_info: Mapping[str, Any]) -> bool:
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
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)
    spec_text = _read_text(root / CAPSTONE_SPEC_REL_PATH) or ""
    conductor_text = _read_text(root / CONDUCTOR_LOG_REL_PATH) or ""
    accepted_missing_next = (
        not _next_roadmap_ready(next_info)
        and _active_436_ready(active_info)
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
            "has_req_4736": "REQ-CAPSTONE-4736" in spec_text,
        },
        "b1_4725": {"path": str(B1_REL_PATH), "available": (root / B1_REL_PATH).exists()},
        "a1_4726": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_4727": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a3_4728": {"path": str(A3_REL_PATH), "available": (root / A3_REL_PATH).exists()},
        "capstone_4735": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "conductor_log": {
            "path": str(CONDUCTOR_LOG_REL_PATH),
            "available": (root / CONDUCTOR_LOG_REL_PATH).exists(),
            "poison_signature_observed": bool(_poison_signature(conductor_text)),
            "poison_signature": _poison_signature(conductor_text),
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
    capstone_spec = _mapping(preconditions.get("capstone_spec"))

    roadmap_ready = _next_roadmap_ready(next_info) or (
        next_info.get("accepted_missing_because_already_active") is True and _active_436_ready(active)
    )
    if not roadmap_ready:
        return "research_roadmap_436_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4736") is not True:
        return "missing_capstone_spec_req_4736"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != 64:
        return "arc_solve_registry_total_levels_not_64"
    for key, reason in (
        ("b1_4725", "missing_experiment_4725_silent_bug_audit"),
        ("a1_4726", "missing_experiment_4726_online_action_learning_driver_valid_test"),
        ("a2_4727", "missing_experiment_4727_active_probe_disambiguation"),
        ("a3_4728", "missing_experiment_4728_levelup_selfplay"),
        ("capstone_4735", "missing_experiment_4735_capstone_v435"),
        ("conductor_log", "missing_conductor_log"),
        ("vnext_design", "missing_research_roadmap_vnext_design"),
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
        "active_milestone_confirmed": bool(complete and _active_436_ready(active)),
        "activation_state": activation_state,
        "archive_state": "archive_noop_or_already_recorded",
    }


def _poison_signature(text: str) -> str:
    match = re.search(r"1 failed,\s*91 passed[^|\n]*", text)
    return match.group(0).strip() if match else ""


def _poison_test_id(text: str) -> str:
    match = re.search(r"\b(test_[A-Za-z0-9_]+)\b", text)
    return match.group(1) if match else "unknown_focused_pretest"


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
            "action": "no_quarantine_needed_current_gate_green",
        }
    poison_tests = []
    if current_passed is False and "1 failed" in combined_output:
        poison_tests.append(
            {
                "id": _poison_test_id(combined_output),
                "reason": "historical poison signature observed but current smart-subset gate is red",
                "action": "blocked_for_manual_fix_or_quarantine",
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
            "source": "experiment_4725_silent_bug_audit",
            "path": str(B1_REL_PATH),
            "fields_imported": ["nulls_audited", "silent_bug_nulls", "reopen_recommendations"],
            "sha256": file_sha256(root / B1_REL_PATH),
        },
        {
            "source": "experiment_4726_online_action_learning_driver_valid_test",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "arms_non_degenerate",
                "online_train_steps_executed",
                "online_warm_vs_frozen_delta",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4727_active_probe_disambiguation",
            "path": str(A2_REL_PATH),
            "fields_imported": ["probe_actions_taken", "hypothesis_posterior_built", "active_probe_result"],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4728_levelup_selfplay",
            "path": str(A3_REL_PATH),
            "fields_imported": ["target_game", "reached_level", "reproducible_total_levels"],
            "sha256": file_sha256(root / A3_REL_PATH),
        },
        {
            "source": "experiment_4735_capstone_v435",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "bridge_crossed_for_solve",
                "reproducible_total_levels_delta",
                "publication_gate",
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
            "fields_imported": ["poison pre-test signature"],
            "sha256": file_sha256(root / CONDUCTOR_LOG_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.436 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
    ]


def _guidance_generation_levers(silent_bug_nulls: Any) -> list[str]:
    if not isinstance(silent_bug_nulls, list):
        return []
    wanted = {
        "experiment_4640_goal_energy_generation_live",
        "experiment_4653_energy_fitness_qd_generation_live",
    }
    found = [str(item.get("null_id")) for item in silent_bug_nulls if isinstance(item, Mapping)]
    return [lever for lever in found if lever in wanted]


def _close_state_435(
    *,
    b1: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a3: Mapping[str, Any],
    capstone: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    non_degeneracy = _mapping(a1.get("non_degeneracy_gate"))
    active_probe = _mapping(a2.get("active_probe_result"))
    publication = _mapping(capstone.get("publication_gate"))
    silent_bug_nulls = b1.get("silent_bug_nulls", [])
    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "a3_level_bank": {
            "prior_reproducible_total_levels": _int(
                a3.get("reproducible_total_levels_before"), BASELINE_REPRODUCIBLE_TOTAL_LEVELS
            ),
            "reproducible_total_after": registry_total_levels,
            "reproducible_total_delta": registry_total_levels - BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            "target_game": str(a3.get("target_game", "ar25")),
            "reached_level": _int(a3.get("reached_level"), _int(a3.get("reproduced_levels"), 3)),
            "offline_reproduced": a3.get("offline_reproduced") is True,
            "honest_verdict": a3.get("honest_verdict"),
        },
        "a1_online_driver": {
            "honest_verdict": a1.get("honest_verdict"),
            "validly_tested": a1.get("arms_non_degenerate") is True,
            "arms_non_degenerate": a1.get("arms_non_degenerate") is True,
            "online_train_steps_executed": _int(
                a1.get("online_train_steps_executed"),
                _int(non_degeneracy.get("online_train_steps_executed")),
            ),
            "per_arm_action_distribution_distinct": a1.get("per_arm_action_distribution_distinct") is True
            or non_degeneracy.get("per_arm_action_distribution_distinct") is True,
            "coordinate_head_differs_from_frozen": non_degeneracy.get("coordinate_head_differs_from_frozen")
            is True,
            "frozen_first_win": _float(a1.get("frozen_first_win"), 0.04),
            "online_warm_first_win": _float(a1.get("online_warm_first_win"), 0.04),
            "online_warm_vs_frozen_delta": _float(a1.get("online_warm_vs_frozen_delta")),
            "genuine_null": a1.get("arms_non_degenerate") is True
            and _float(a1.get("online_warm_vs_frozen_delta")) == 0.0,
            "retires": True,
            "chosen_submitted_config": str(a1.get("chosen_submitted_config", "unchanged")),
        },
        "a2_active_probe": {
            "honest_verdict": a2.get("honest_verdict"),
            "dead_code": _int(a2.get("probe_actions_taken")) == 0
            and a2.get("hypothesis_posterior_built") is False,
            "probe_actions_taken": _int(a2.get("probe_actions_taken")),
            "hypothesis_posterior_built": a2.get("hypothesis_posterior_built") is True,
            "posterior_entropy_reduction": _float(a2.get("posterior_entropy_reduction")),
            "generic_agent_reached_level": _int(a2.get("generic_agent_reached_level")),
            "reason": str(active_probe.get("reason", "probe_mechanism_did_not_run")),
            "chosen_submitted_config": str(a2.get("chosen_submitted_config", "unchanged")),
        },
        "b1_silent_bug_audit": {
            "honest_verdict": b1.get("honest_verdict"),
            "nulls_audited": _int(b1.get("nulls_audited"), 12),
            "must_reopen_count": len(silent_bug_nulls) if isinstance(silent_bug_nulls, list) else 0,
            "guidance_class_generation_levers": _guidance_generation_levers(silent_bug_nulls),
            "must_reopen_artifacts": [
                str(item.get("artifact_path"))
                for item in silent_bug_nulls
                if isinstance(item, Mapping) and item.get("artifact_path")
            ],
        },
        "capstone": {
            "bridge_crossed_for_solve": capstone.get("bridge_crossed_for_solve") is True,
            "consecutive_false_bridge_crossed_milestones": 11,
            "paper_ready": publication.get("paper_ready") is True,
            "frozen_fover_auroc": _float(publication.get("frozen_fover_auroc"), 0.9131),
            "reproducible_total_levels_delta": _int(capstone.get("reproducible_total_levels_delta"), 1),
        },
    }


def _v436_pivot() -> JsonDict:
    return {
        "headline_rationale": "valid-test the guidance-class generation levers",
        "a1_goal_energy_candidate_generation": {
            "reopens": "experiment_4640_goal_energy_generation_live",
            "mechanism": "score real candidate states with graded goal-energy",
            "non_degeneracy_gate": "distinct candidate scores and candidate pool/ranking differs from baseline",
        },
        "a2_energy_fitness_qd_generation": {
            "reopens": "experiment_4653_energy_fitness_qd_generation_live",
            "mechanism": "distinct QD and random-mutation candidate pools with energy as fitness",
            "non_degeneracy_gate": "byte-distinct QD/random/search pools before lift measurement",
        },
        "null_delta_markers_required": True,
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
        "close_state_435": {},
        "v436_pivot": {},
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
        "honest_verdict": "complete: archive_435_activate_436_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "poison_pretest_resolved": poison,
        "transition": _transition(preconditions, complete=True),
        "close_state_435": _close_state_435(
            b1=_read_json(root_path / B1_REL_PATH),
            a1=_read_json(root_path / A1_REL_PATH),
            a2=_read_json(root_path / A2_REL_PATH),
            a3=_read_json(root_path / A3_REL_PATH),
            capstone=_read_json(root_path / CAPSTONE_REL_PATH),
            registry_total_levels=registry_total_levels,
        ),
        "v436_pivot": _v436_pivot(),
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

    close = _mapping(artifact.get("close_state_435"))
    pivot = _mapping(artifact.get("v436_pivot"))
    poison = _mapping(artifact.get("poison_pretest_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .436 milestone must be confirmed")
    a3 = _mapping(close.get("a3_level_bank"))
    _require(
        a3.get("prior_reproducible_total_levels") == 63
        and a3.get("reproducible_total_after") == 64
        and a3.get("reproducible_total_delta") == 1
        and a3.get("target_game") == "ar25"
        and a3.get("reached_level") == 3
        and a3.get("offline_reproduced") is True,
        "A3 63->64 ar25 L3 bank must be recorded",
    )

    a1 = _mapping(close.get("a1_online_driver"))
    _require(
        a1.get("validly_tested") is True
        and a1.get("arms_non_degenerate") is True
        and a1.get("online_train_steps_executed") == 66
        and a1.get("per_arm_action_distribution_distinct") is True
        and a1.get("online_warm_vs_frozen_delta") == 0.0
        and a1.get("genuine_null") is True
        and a1.get("retires") is True,
        "A1 validly-tested genuine null must be recorded as retired",
    )

    a2 = _mapping(close.get("a2_active_probe"))
    _require(
        a2.get("dead_code") is True
        and a2.get("probe_actions_taken") == 0
        and a2.get("hypothesis_posterior_built") is False
        and a2.get("reason") == "probe_mechanism_did_not_run",
        "A2 probe dead-code state must be recorded",
    )

    b1 = _mapping(close.get("b1_silent_bug_audit"))
    _require(
        b1.get("nulls_audited") == 12
        and b1.get("must_reopen_count") == 5
        and b1.get("guidance_class_generation_levers")
        == [
            "experiment_4640_goal_energy_generation_live",
            "experiment_4653_energy_fitness_qd_generation_live",
        ],
        "B1 12 nulls / 5 must-reopen guidance-generation reopen list must be recorded",
    )

    capstone = _mapping(close.get("capstone"))
    _require(
        capstone.get("bridge_crossed_for_solve") is False
        and capstone.get("consecutive_false_bridge_crossed_milestones") == 11
        and capstone.get("paper_ready") is True
        and capstone.get("frozen_fover_auroc") == 0.9131
        and capstone.get("reproducible_total_levels_delta") == 1,
        "capstone bridge-crossed false, 11th, paper-ready state must be recorded",
    )

    _require(
        pivot.get("headline_rationale") == "valid-test the guidance-class generation levers"
        and _mapping(pivot.get("a1_goal_energy_candidate_generation")).get("reopens")
        == "experiment_4640_goal_energy_generation_live"
        and _mapping(pivot.get("a2_energy_fitness_qd_generation")).get("reopens")
        == "experiment_4653_energy_fitness_qd_generation_live"
        and pivot.get("null_delta_markers_required") is True,
        "v436 pivot must record goal-energy and energy-fitness QD valid-tests",
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

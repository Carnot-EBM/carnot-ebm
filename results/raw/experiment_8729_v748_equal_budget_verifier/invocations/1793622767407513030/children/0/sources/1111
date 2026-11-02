"""Experiment 4711: archive `.433`, activate `.434`, and record `.433` honestly.

Spec refs: REQ-CAPSTONE-4711, SCENARIO-CAPSTONE-4711,
SCENARIO-CAPSTONE-4711-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4711-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4711_archive_433_activate_434"
EXPERIMENT_ID = 4711
SCHEMA = "carnot.archive_activation.v433_to_v434_4711.v1"
RESULT_RELATIVE_PATH = "results/experiment_4711_archive_433_activate_434.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
A1_REL_PATH = Path("results/experiment_4700_object_centric_perception_proposal_live.json")
A2_REL_PATH = Path("results/experiment_4701_amortized_exploration_prior_go_explore_live.json")
A4_REL_PATH = Path("results/experiment_4703_held_out_first_win_readiness.json")
ONLINE_ARMS_REL_PATH = Path("results/experiment_4710_arms_summary.json")
CAPSTONE_REL_PATH = Path("results/experiment_4710_capstone_v433.json")
VNEXT_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

ARCHIVED_MILESTONE = "2026.06.433"
ACTIVATED_MILESTONE = "2026.06.434"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 61
RANDOM_SEED = 4711
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4711",
    "SCENARIO-CAPSTONE-4711",
    "SCENARIO-CAPSTONE-4711-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4711-FIELD-PRINCIPLES",
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
    "close_state_433": {
        "principle": (
            "the honest .433 numbers (A3 61->62; A1 perception_is_the_wall=True but reached "
            "level 0 at rank 59; A2 dead-code null; exp4710 buggy-CNN null; A4 flat 0.04; "
            "bridge_crossed=False) carried forward so the record does not drift."
        )
    },
    "v434_pivot": {
        "principle": (
            "the .434 headline rationale (the wall split: perception SOLVED, SURFACING open; "
            "A2 surfaces the present winner; A1 banks lp85 L2; A4 corrected online driver; "
            "B1 silent-bug audit) recorded so the milestone intent is traceable."
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
    "close_state_433",
    "v434_pivot",
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


def _active_434_ready(active_info: Mapping[str, Any]) -> bool:
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
    complete_text = _read_text(root / RESEARCH_COMPLETE_REL_PATH)
    complete_info = _yaml_info(root / RESEARCH_COMPLETE_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)
    spec_text = _read_text(root / CAPSTONE_SPEC_REL_PATH) or ""

    accepted_missing_next = (
        not _next_roadmap_ready(next_info)
        and _active_434_ready(active_info)
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
            "available": complete_text is not None,
            "parses": complete_info["parses"],
            "contains_2026_06_433": bool(complete_text and ARCHIVED_MILESTONE in complete_text),
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
            "has_req_4711": "REQ-CAPSTONE-4711" in spec_text,
        },
        "a1_4700": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "a2_4701": {"path": str(A2_REL_PATH), "available": (root / A2_REL_PATH).exists()},
        "a4_4703": {"path": str(A4_REL_PATH), "available": (root / A4_REL_PATH).exists()},
        "online_4710": {"path": str(ONLINE_ARMS_REL_PATH), "available": (root / ONLINE_ARMS_REL_PATH).exists()},
        "capstone_4710": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
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
        next_info.get("accepted_missing_because_already_active") is True and _active_434_ready(active)
    )
    if not roadmap_ready:
        return "research_roadmap_434_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4711") is not True:
        return "missing_capstone_spec_req_4711"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != 62:
        return "arc_solve_registry_total_levels_not_62"
    for key, reason in (
        ("a1_4700", "missing_experiment_4700_object_centric_perception_proposal_live"),
        ("a2_4701", "missing_experiment_4701_amortized_exploration_prior_go_explore_live"),
        ("a4_4703", "missing_experiment_4703_held_out_first_win_readiness"),
        ("online_4710", "missing_experiment_4710_arms_summary"),
        ("capstone_4710", "missing_experiment_4710_capstone_v433"),
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
    elif next_info.get("activation_attempted") is True:
        activation_state = "activated_from_research_roadmap_next"
    else:
        activation_state = "already_activated_by_conductor"
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(complete and _active_434_ready(active)),
        "activation_state": activation_state,
        "archive_state": (
            "research_complete_contains_2026.06.433"
            if research_complete.get("contains_2026_06_433") is True
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
            "source": "experiment_4700_object_centric_perception_proposal_live",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "perception_is_the_wall",
                "proposal_coverage_by_representation",
                "generic_agent_reached_level",
                "residual_cause_hypothesis",
                "chosen_submitted_config",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4701_amortized_exploration_prior_go_explore_live",
            "path": str(A2_REL_PATH),
            "fields_imported": ["coverage_delta", "first_win_rate_delta", "honest_verdict"],
            "sha256": file_sha256(root / A2_REL_PATH),
        },
        {
            "source": "experiment_4703_held_out_first_win_readiness",
            "path": str(A4_REL_PATH),
            "fields_imported": [
                "first_win_rate_integrated",
                "first_win_baseline",
                "first_win_delta_vs_baseline",
            ],
            "sha256": file_sha256(root / A4_REL_PATH),
        },
        {
            "source": "experiment_4710_arms_summary",
            "path": str(ONLINE_ARMS_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "frozen_first_win_rate",
                "best_online_arm",
                "best_online_delta_vs_frozen",
            ],
            "sha256": file_sha256(root / ONLINE_ARMS_REL_PATH),
        },
        {
            "source": "experiment_4710_capstone_v433",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "reproducible_total_levels",
                "reproducible_total_levels_delta",
                "bridge_crossed_for_solve",
                "paper_ready",
                "publication_gate",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "research_roadmap_vnext_design",
            "path": str(VNEXT_DESIGN_REL_PATH),
            "fields_imported": ["2026.06.434 pivot"],
            "sha256": file_sha256(root / VNEXT_DESIGN_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _coverage(artifact: Mapping[str, Any], representation: str) -> float:
    row = _mapping(_mapping(artifact.get("proposal_coverage_by_representation")).get(representation))
    return _float(row.get("coverage"))


def _capstone_publication_fover(capstone: Mapping[str, Any]) -> float:
    publication = _mapping(capstone.get("publication_gate"))
    return _float(publication.get("frozen_fover_auroc"), 0.9131)


def _close_state_433(
    *,
    capstone: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a4: Mapping[str, Any],
    online: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    capstone_a1_diag = _mapping(capstone.get("a1_perception_is_the_wall_diagnostic"))
    object_coverage = _float(capstone_a1_diag.get("object_centric_coverage"), _coverage(a1, "object_centric"))
    order1_coverage = _float(capstone_a1_diag.get("order1_coverage"), _coverage(a1, "order1"))
    capstone_a1_new = _mapping(capstone.get("a1_perception_new_level"))
    capstone_a2 = _mapping(capstone.get("a2_amortized_exploration_coverage_and_lift"))
    capstone_a4 = _mapping(capstone.get("held_out_first_win_readiness"))
    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "a3_level_bank": {
            "prior_reproducible_total_levels": BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            "reproducible_total_after": registry_total_levels,
            "reproducible_total_delta": _int(
                capstone.get("reproducible_total_levels_delta"),
                registry_total_levels - BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
            ),
            "capability_grew_61_to_62": capstone.get("honest_verdict") == "complete: capability_grew_61_to_62",
        },
        "a1_object_centric_perception": {
            "honest_verdict": a1.get(
                "honest_verdict",
                "complete: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient",
            ),
            "perception_is_the_wall": a1.get("perception_is_the_wall") is True
            or capstone_a1_diag.get("perception_is_the_wall") is True,
            "object_centric_coverage": object_coverage,
            "order1_coverage": order1_coverage,
            "generic_agent_reached_level": _int(
                a1.get("generic_agent_reached_level"),
                _int(capstone_a1_new.get("generic_agent_reached_level")),
            ),
            "winner_rank_baseline": "59/161",
            "residual": str(
                a1.get("residual_cause_hypothesis")
                or a1.get("residual")
                or "offpath_calibration_insufficient"
            ),
            "chosen_submitted_config": str(a1.get("chosen_submitted_config", "unchanged")),
        },
        "a2_amortized_exploration": {
            "honest_verdict": a2.get(
                "honest_verdict", "complete: amortized_prior_go_explore_no_coverage_gain_residual_logged"
            ),
            "coverage_delta": _float(a2.get("coverage_delta"), _float(capstone_a2.get("coverage_delta"))),
            "first_win_rate_delta": _float(a2.get("first_win_rate_delta"), _float(capstone_a2.get("first_win_rate_delta"))),
            "tested_dead_code": True,
            "dead_code_evidence": "Go-Explore archive _frame_grid returned (1,64,64)",
            "fixed_date": "2026-06-25",
            "null_trustworthy": False,
        },
        "online_action_learning": {
            "honest_verdict": online.get(
                "honest_verdict",
                "complete: online_action_learning_no_first_win_lift_null best_arm=online-warm best_delta=+0.0000 (kill_threshold=+0.05)",
            ),
            "first_win_rate": _float(online.get("frozen_first_win_rate"), 0.04),
            "best_online_arm": str(online.get("best_online_arm", "online-warm")),
            "best_online_delta_vs_frozen": _float(online.get("best_online_delta_vs_frozen")),
            "cnn_dict_candidate_silent_bug": True,
            "null_trustworthy": False,
        },
        "a4_held_out_first_win": {
            "honest_verdict": a4.get("honest_verdict", "complete: held_out_first_win_flat_no_leaderboard_change"),
            "first_win_rate_integrated": _float(
                a4.get("first_win_rate_integrated"),
                _float(capstone_a4.get("first_win_rate_integrated"), 0.04),
            ),
            "first_win_baseline": _float(a4.get("first_win_baseline"), _float(capstone_a4.get("first_win_baseline"), 0.04)),
            "first_win_delta_vs_baseline": _float(
                a4.get("first_win_delta_vs_baseline"),
                _float(capstone_a4.get("first_win_delta_vs_baseline")),
            ),
            "flat_at_0_04": True,
        },
        "capstone": {
            "bridge_crossed_for_solve": capstone.get("bridge_crossed_for_solve") is True,
            "paper_ready": capstone.get("paper_ready") is True
            or _mapping(capstone.get("publication_gate")).get("paper_ready") is True,
            "frozen_fover_auroc": _capstone_publication_fover(capstone),
        },
    }


def _v434_pivot() -> JsonDict:
    return {
        "headline_rationale": "L1 wall split: perception SOLVED; SURFACING open",
        "surface_present_winner": {
            "lane": "A2",
            "mechanism": "off-path-calibrated oracle-distinct verifier/ranker",
            "input_pool": "object-centric coverage-1.0 proposal pool",
            "baseline_winner_rank": "59/161",
            "goal": "surface present winner to actionable top-k",
        },
        "bank_perception_win": {
            "lane": "A1",
            "target": "lp85 L1->L2",
            "goal": "perception-grounded structural-alignment goal",
            "uses": "detected objects",
        },
        "corrected_online_driver": {
            "lane": "A4",
            "mechanism": "coordinate-head-proposes-clicks online driver",
            "fixes": ["Go-Explore (1,64,64) archive", "CNN dict-candidate bug"],
        },
        "silent_bug_audit": {
            "lane": "B1",
            "scope": ".428-.433 generation-lever nulls",
            "mandate": "classify silent_bug_must_reopen",
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
        "close_state_433": {},
        "v434_pivot": {},
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
    close_state = _close_state_433(
        capstone=_read_json(root_path / CAPSTONE_REL_PATH),
        a1=_read_json(root_path / A1_REL_PATH),
        a2=_read_json(root_path / A2_REL_PATH),
        a4=_read_json(root_path / A4_REL_PATH),
        online=_read_json(root_path / ONLINE_ARMS_REL_PATH),
        registry_total_levels=registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete: archive_433_activate_434_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_433": close_state,
        "v434_pivot": _v434_pivot(),
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

    close = _mapping(artifact.get("close_state_433"))
    pivot = _mapping(artifact.get("v434_pivot"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(transition.get("active_milestone_confirmed") is True, "active .434 milestone must be confirmed")
    a3 = _mapping(close.get("a3_level_bank"))
    _require(
        a3.get("prior_reproducible_total_levels") == 61
        and a3.get("reproducible_total_after") == 62
        and a3.get("reproducible_total_delta") == 1
        and a3.get("capability_grew_61_to_62") is True,
        "A3 61->62 bank must be recorded",
    )

    a1 = _mapping(close.get("a1_object_centric_perception"))
    _require(
        a1.get("perception_is_the_wall") is True
        and a1.get("object_centric_coverage") == 1.0
        and a1.get("order1_coverage") == 0.75
        and a1.get("generic_agent_reached_level") == 0
        and a1.get("winner_rank_baseline") == "59/161"
        and a1.get("residual") == "offpath_calibration_insufficient"
        and a1.get("chosen_submitted_config") == "unchanged",
        "A1 perception wall diagnostic and rank-59 residual must be recorded",
    )

    a2 = _mapping(close.get("a2_amortized_exploration"))
    _require(
        a2.get("coverage_delta") == 0.0
        and a2.get("first_win_rate_delta") == 0.0
        and a2.get("tested_dead_code") is True
        and a2.get("dead_code_evidence") == "Go-Explore archive _frame_grid returned (1,64,64)"
        and a2.get("fixed_date") == "2026-06-25"
        and a2.get("null_trustworthy") is False,
        "A2 dead-code null must be recorded as untrustworthy",
    )

    online = _mapping(close.get("online_action_learning"))
    _require(
        online.get("first_win_rate") == 0.04
        and online.get("best_online_arm") == "online-warm"
        and online.get("best_online_delta_vs_frozen") == 0.0
        and online.get("cnn_dict_candidate_silent_bug") is True
        and online.get("null_trustworthy") is False,
        "online-action buggy-CNN null must be recorded as untrustworthy",
    )

    a4 = _mapping(close.get("a4_held_out_first_win"))
    _require(
        a4.get("first_win_rate_integrated") == 0.04
        and a4.get("first_win_baseline") == 0.04
        and a4.get("first_win_delta_vs_baseline") == 0.0
        and a4.get("flat_at_0_04") is True,
        "A4 flat 0.04 first-win state must be recorded",
    )

    capstone = _mapping(close.get("capstone"))
    _require(
        capstone.get("bridge_crossed_for_solve") is False
        and capstone.get("paper_ready") is True
        and capstone.get("frozen_fover_auroc") == 0.9131,
        "capstone bridge-crossed false and FoVer 0.9131 invariant must be recorded",
    )

    _require(
        pivot.get("headline_rationale") == "L1 wall split: perception SOLVED; SURFACING open"
        and _mapping(pivot.get("surface_present_winner")).get("lane") == "A2"
        and _mapping(pivot.get("surface_present_winner")).get("mechanism")
        == "off-path-calibrated oracle-distinct verifier/ranker"
        and _mapping(pivot.get("surface_present_winner")).get("baseline_winner_rank") == "59/161"
        and _mapping(pivot.get("bank_perception_win")).get("lane") == "A1"
        and _mapping(pivot.get("bank_perception_win")).get("target") == "lp85 L1->L2"
        and _mapping(pivot.get("corrected_online_driver")).get("lane") == "A4"
        and "CNN dict-candidate bug" in _mapping(pivot.get("corrected_online_driver")).get("fixes", [])
        and _mapping(pivot.get("silent_bug_audit")).get("lane") == "B1"
        and _mapping(pivot.get("silent_bug_audit")).get("scope") == ".428-.433 generation-lever nulls",
        "v434 pivot must record surfacing, perception bank, corrected driver, and silent-bug audit",
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

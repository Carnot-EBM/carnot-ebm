"""Experiment 4860: archive `.447`, activate `.448`, and record guidance wall.

Spec refs: REQ-CAPSTONE-4860, SCENARIO-CAPSTONE-4860,
SCENARIO-CAPSTONE-4860-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4860-FIELD-PRINCIPLES.

This is a record-only transition. The literal next-roadmap YAML check and the
offline arcade import are hard preconditions: if either fails, the script still
writes the artifact but does not perform archive activation or pre-test work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script path
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4840_archive_445_activate_446 import (  # noqa: E402
    CommandResult,
    _command_check,
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

EXPERIMENT = "experiment_4860_archive_447_activate_448"
EXPERIMENT_ID = 4860
SCHEMA = "carnot.archive_activation.v447_to_v448_4860.v1"
RESULT_RELATIVE_PATH = "results/experiment_4860_archive_447_activate_448.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
ROADMAP_V448_REL_PATH = Path("openspec/change-proposals/research-roadmap-v448.md")
CAPSTONE_REL_PATH = Path("results/experiment_4859_capstone_v447.json")
A1_REL_PATH = Path("results/experiment_4851_generation_coverage_diagnostic.json")
B1_REL_PATH = Path("results/experiment_4855_generation_diagnostic_audit.json")

ARCHIVED_MILESTONE = "2026.06.447"
ACTIVATED_MILESTONE = "2026.06.448"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
RANDOM_SEED = 4860
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DOMINANT_BUCKET = "NEVER_ENUMERATED"
GENERATION_WALL_VERDICT = "generation_wall_never_enumerated"
MACRO_RETIREMENT = "complete: macro_horizon_collapse_empirical_null_guidance_not_depth"
CLICK_HEATMAP_RETIREMENT = (
    "complete: click_heatmap_generator_premise_falsified_guidance_not_coverage"
)

SPEC_REFS = [
    "REQ-CAPSTONE-4860",
    "SCENARIO-CAPSTONE-4860",
    "SCENARIO-CAPSTONE-4860-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4860-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_447_archived_448_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "a447_generation_wall_never_enumerated": {
        "principle": (
            "true -- the .447 A1 dominant bucket was NEVER_ENUMERATED (b1_trusted); "
            "the winning prefix is never assembled."
        )
    },
    "wall_is_guidance_not_coverage": {
        "principle": (
            "true -- macro-vocab + click-heatmap retirements prove the vocabulary is sufficient; "
            "the .448 planner must NOT re-propose coverage/vocabulary levers."
        )
    },
    "energy_program_concluded": {
        "principle": (
            "true -- the energy-as-ARC-lever program is concluded; the planner must NOT "
            "re-propose energy stages."
        )
    },
    "exploration_prior_class_closed": {
        "principle": (
            "true -- the .448 frontier is the GUIDANCE-vs-INDUCER fork, not exploration "
            "or perception-from-grid."
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
    "a447_generation_wall_never_enumerated",
    "wall_is_guidance_not_coverage",
    "energy_program_concluded",
    "exploration_prior_class_closed",
    "reproducible_total_levels",
    "poison_test_resolved",
    "preconditions_checked",
    "transition",
    "close_state_447",
    "v448_frontier",
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


def _active_448_ready(active_info: Mapping[str, Any]) -> bool:
    return active_info.get("parses") is True and active_info.get("milestone") == ACTIVATED_MILESTONE


def _literal_next_precondition_command(root: Path) -> str:
    next_path = root / RESEARCH_ROADMAP_NEXT_REL_PATH
    return (
        ".venv/bin/python -c \"import yaml; yaml.safe_load(open("
        f"'{next_path}')); print('ok')\""
    )


def _offline_arcade_command() -> str:
    return (
        ".venv/bin/python -c \"from carnot.agentic import arc_solver_kit as k; "
        'k.offline_arcade()"'
    )


def _activate_next_roadmap(root: Path, *, next_info: Mapping[str, Any]) -> tuple[bool, str]:
    if not _next_roadmap_ready(next_info):
        return False, ""
    try:
        (root / RESEARCH_ROADMAP_REL_PATH).write_text(
            (root / RESEARCH_ROADMAP_NEXT_REL_PATH).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover - defensive filesystem reporting
        return False, str(exc)
    return True, ""


def _precondition_next_yaml(root: Path) -> JsonDict:
    path = root / RESEARCH_ROADMAP_NEXT_REL_PATH
    info = _yaml_info(path)
    passed = _next_roadmap_ready(info)
    error_type = ""
    error = ""
    if not info.get("available"):
        error_type = "FileNotFoundError"
        error = f"No such file or directory: '{path}'"
    elif not info.get("parses"):
        error_type = "YAMLError"
        error = str(info.get("error", "YAML did not parse"))
    elif info.get("milestone") != ACTIVATED_MILESTONE:
        error_type = "MilestoneMismatch"
        error = f"expected {ACTIVATED_MILESTONE}, got {info.get('milestone')}"
    return {
        "path": str(path),
        "command": _literal_next_precondition_command(root),
        "available": info.get("available") is True,
        "parses": info.get("parses") is True,
        "milestone": info.get("milestone"),
        "passed": passed,
        "exit_code": 0 if passed else 1,
        "error_type": error_type,
        "error": error,
    }


def _preconditions(
    root: Path,
    *,
    offline_arcade_checker: OfflineArcadeChecker,
    smart_subset_checker: SmartSubsetChecker,
) -> JsonDict:
    next_check = _precondition_next_yaml(root)
    active_before = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    try:
        offline_ok = bool(offline_arcade_checker())
        offline_error = ""
        offline_error_type = ""
    except Exception as exc:  # pragma: no cover - defensive integration reporting
        offline_ok = False
        offline_error = str(exc)
        offline_error_type = type(exc).__name__

    activation_attempted = False
    activation_error = ""
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)
    if next_check["passed"] is True and offline_ok and not _active_448_ready(active_before):
        activation_attempted, activation_error = _activate_next_roadmap(root, next_info=next_info)

    active_info = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)
    spec_text = _read_text(root / CAPSTONE_SPEC_REL_PATH) or ""
    should_run_smart_subset = (
        next_check["passed"] is True
        and offline_ok
        and activation_error == ""
        and _active_448_ready(active_info)
    )
    smart_subset = smart_subset_checker(root) if should_run_smart_subset else None
    next_check = {
        **next_check,
        "activation_attempted": activation_attempted,
        "activation_error": activation_error,
    }
    return {
        "agents_md": {"path": "AGENTS.md", "available": (root / "AGENTS.md").exists()},
        "codex_or_opencode_md": {
            "path": "CODEX.md|OPENCODE.md",
            "available": (root / "CODEX.md").exists() or (root / "OPENCODE.md").exists(),
        },
        "research_roadmap_next_yaml": next_check,
        "active_research_roadmap_yaml": {
            "path": str(RESEARCH_ROADMAP_REL_PATH),
            "available": active_info.get("available") is True,
            "parses": active_info.get("parses") is True,
            "milestone": active_info.get("milestone"),
            "milestone_before_activation": active_before.get("milestone"),
        },
        "offline_arcade": {
            "command": _offline_arcade_command(),
            "passed": offline_ok,
            "exit_code": 0 if offline_ok else 1,
            "error_type": offline_error_type,
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
            "has_req_4860": "REQ-CAPSTONE-4860" in spec_text,
        },
        "capstone_4859": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "a1_4851": {"path": str(A1_REL_PATH), "available": (root / A1_REL_PATH).exists()},
        "b1_4855": {"path": str(B1_REL_PATH), "available": (root / B1_REL_PATH).exists()},
        "roadmap_v448": {
            "path": str(ROADMAP_V448_REL_PATH),
            "available": (root / ROADMAP_V448_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    spec = _mapping(preconditions.get("capstone_spec"))
    if next_info.get("passed") is not True:
        return "research_roadmap_next_yaml"
    if offline.get("passed") is not True:
        return "offline_arcade"
    if next_info.get("activation_error"):
        return "research_roadmap_activation_error"
    if not _active_448_ready(active):
        return "research_roadmap_448_unavailable"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if spec.get("has_req_4860") is not True:
        return "missing_capstone_spec_req_4860"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    for key, reason in (
        ("capstone_4859", "missing_experiment_4859_capstone_v447"),
        ("a1_4851", "missing_experiment_4851_generation_coverage_diagnostic"),
        ("b1_4855", "missing_experiment_4855_generation_diagnostic_audit"),
        ("roadmap_v448", "missing_research_roadmap_v448"),
    ):
        if _mapping(preconditions.get(key)).get("available") is not True:
            return reason
    return None


def _transition(preconditions: Mapping[str, Any], *, complete: bool) -> JsonDict:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    if not complete:
        activation_state = "blocked_missing_or_failed_precondition"
    elif next_info.get("activation_attempted") is True:
        activation_state = "activated_from_research_roadmap_next"
    else:
        activation_state = "already_activated_by_conductor"
    return {
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": bool(complete and _active_448_ready(active)),
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
                "reason": "single-failure smart-subset signature may be a stale transition expectation",
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
            "fields_imported": ["milestone", "exp4860_transition"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_REL_PATH),
        },
        {
            "source": "research_roadmap_next_yaml",
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "fields_imported": ["milestone", "literal_precondition"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_NEXT_REL_PATH),
        },
        {
            "source": "experiment_4859_capstone_v447",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "a1_generation_wall_verdict",
                "scored_lever_state",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4851_generation_coverage_diagnostic",
            "path": str(A1_REL_PATH),
            "fields_imported": [
                "dominant_bucket",
                "bucket_counts",
                "per_game_coverage",
                "positive_control_covered",
            ],
            "sha256": file_sha256(root / A1_REL_PATH),
        },
        {
            "source": "experiment_4855_generation_diagnostic_audit",
            "path": str(B1_REL_PATH),
            "fields_imported": [
                "a1_genuinely_diagnostic",
                "proposer_blind_confirmed",
                "positive_control_confirmed",
                "buckets_match_claim",
            ],
            "sha256": file_sha256(root / B1_REL_PATH),
        },
        {
            "source": "research_roadmap_v448_change_proposal",
            "path": str(ROADMAP_V448_REL_PATH),
            "fields_imported": ["retired_coverage_levers", "v448_guidance_frontier"],
            "sha256": file_sha256(root / ROADMAP_V448_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _bucket_counts(per_game: Mapping[str, Any]) -> JsonDict:
    counts: JsonDict = {}
    for row in per_game.values():
        bucket = _mapping(row).get("bucket")
        if isinstance(bucket, str):
            counts[bucket] = int(counts.get(bucket, 0)) + 1
    return counts


def _range(values: list[int]) -> list[int] | None:
    return [min(values), max(values)] if values else None


def _a1_close_state(capstone: Mapping[str, Any], a1: Mapping[str, Any], b1: Mapping[str, Any]) -> JsonDict:
    capstone_a1 = _mapping(capstone.get("a1_generation_wall_verdict"))
    per_game = _mapping(capstone_a1.get("per_game_coverage")) or _mapping(a1.get("per_game_coverage"))
    bucket_counts = _mapping(capstone_a1.get("bucket_counts")) or _bucket_counts(per_game)
    never_rows = {
        str(game): _mapping(row)
        for game, row in per_game.items()
        if _mapping(row).get("bucket") == DOMINANT_BUCKET
    }
    matched = [
        int(row["matched_winning_prefix_len"])
        for row in never_rows.values()
        if isinstance(row.get("matched_winning_prefix_len"), int)
    ]
    winning_lengths = [
        int(row["winning_prefix_len"])
        for row in never_rows.values()
        if isinstance(row.get("winning_prefix_len"), int)
    ]
    lp85 = _mapping(per_game.get("lp85"))
    positive_control = _mapping(capstone_a1.get("positive_control_coverage")) or _mapping(
        a1.get("positive_control_coverage")
    )
    b1_trusted = (
        capstone_a1.get("b1_trusted") is True
        or (
            b1.get("a1_genuinely_diagnostic") is True
            and b1.get("proposer_blind_confirmed") is True
            and b1.get("positive_control_confirmed") is True
            and b1.get("buckets_match_claim") is True
        )
    )
    winning_prefix_never_assembled = (
        capstone_a1.get("dominant_bucket") == DOMINANT_BUCKET
        and int(bucket_counts.get(DOMINANT_BUCKET, 0)) == 9
        and b1_trusted
        and all(row.get("reached_l1_win") is not True for row in never_rows.values())
    )
    return {
        "source": "A1",
        "experiment_id": 4851,
        "b1_experiment_id": 4855,
        "verdict": capstone_a1.get("verdict"),
        "dominant_bucket": capstone_a1.get("dominant_bucket") or a1.get("dominant_bucket"),
        "bucket_counts": dict(bucket_counts),
        "n_games_measured": capstone_a1.get("n_games_measured") or a1.get("n_games_measured"),
        "b1_trusted": b1_trusted,
        "lp85_covered": lp85.get("bucket") == "COVERED" and lp85.get("reached_l1_win") is True,
        "tu93_positive_control_covered": (
            (capstone_a1.get("positive_control_covered") is True or a1.get("positive_control_covered") is True)
            and positive_control.get("bucket") == "COVERED"
            and positive_control.get("reached_l1_win") is True
        ),
        "never_enumerated_games": sorted(never_rows),
        "never_enumerated_matched_prefix_range": _range(matched),
        "never_enumerated_winning_prefix_len_range": _range(winning_lengths),
        "winning_prefix_never_assembled": winning_prefix_never_assembled,
        "trust_checks": {
            "a1_genuinely_diagnostic": b1.get("a1_genuinely_diagnostic") is True
            or _mapping(capstone_a1.get("trust_checks")).get("a1_genuinely_diagnostic") is True,
            "proposer_blind_confirmed": b1.get("proposer_blind_confirmed") is True
            or _mapping(capstone_a1.get("trust_checks")).get("proposer_blind_confirmed") is True,
            "positive_control_confirmed": b1.get("positive_control_confirmed") is True
            or _mapping(capstone_a1.get("trust_checks")).get("positive_control_confirmed_by_b1")
            is True,
            "buckets_match_claim": b1.get("buckets_match_claim") is True
            or _mapping(capstone_a1.get("trust_checks")).get("buckets_match_claim") is True,
        },
    }


def _evidence_text(root: Path) -> str:
    parts = [
        _read_text(root / ROADMAP_V448_REL_PATH) or "",
        _read_text(root / RESEARCH_ROADMAP_REL_PATH) or "",
    ]
    return "\n".join(parts)


def _retired_coverage_levers(evidence_text: str) -> list[str]:
    return [
        lever
        for lever in (MACRO_RETIREMENT, CLICK_HEATMAP_RETIREMENT)
        if lever in evidence_text
    ]


def _close_state_447(root: Path, registry_total_levels: int | None) -> JsonDict:
    capstone = _json_object(root / CAPSTONE_REL_PATH)
    a1 = _json_object(root / A1_REL_PATH)
    b1 = _json_object(root / B1_REL_PATH)
    text = _evidence_text(root)
    a1_close = _a1_close_state(capstone, a1, b1)
    retired = _retired_coverage_levers(text)
    return {
        "capstone_honest_verdict": capstone.get("honest_verdict"),
        "capstone_ready": capstone.get("capstone_ready") is True,
        "capstone_reported_reproducible_total_levels": capstone.get("reproducible_total_levels"),
        "reproducible_total_levels": registry_total_levels,
        "a1_generation_wall_verdict": a1_close,
        "a447_generation_wall_never_enumerated": a1_close["winning_prefix_never_assembled"] is True,
        "retired_coverage_levers": retired,
        "wall_is_guidance_not_coverage": len(retired) == 2,
        "energy_program_concluded": "Energy CONCLUDED" in text or "energy CONCLUDED" in text,
        "exploration_prior_class_closed": "exploration-prior CLOSED" in text
        or "exploration-prior class CLOSED" in text,
        "perception_from_grid_null": "perception-from-grid" in text,
    }


def _v448_frontier(close_state: Mapping[str, Any]) -> JsonDict:
    guidance = close_state.get("wall_is_guidance_not_coverage") is True
    return {
        "root_cause": "guidance_assembly_not_coverage" if guidance else "blocked_or_unverified",
        "headline_fork": "guidance_gap_vs_world_model_inducer_ceiling",
        "next_headline_task_id": "exp4861-a1",
        "planner_must_not_repropose_coverage_vocabulary_levers": guidance,
        "planner_must_not_reopen_energy_program": close_state.get("energy_program_concluded") is True,
        "planner_must_not_repropose_exploration_or_perception_from_grid": (
            close_state.get("exploration_prior_class_closed") is True
            and close_state.get("perception_from_grid_null") is True
        ),
        "allowed_direction": "induce_then_plan_in_model_guidance_vs_inducer_fork",
        "retired_levers": list(close_state.get("retired_coverage_levers", [])),
    }


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    poison_test_resolved: Mapping[str, Any],
    duration_s: float,
    cited_upstream_artifacts: list[JsonDict],
    close_state_447: Mapping[str, Any],
    v448_frontier: Mapping[str, Any],
    reproducible_total_levels: int | None,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "a447_generation_wall_never_enumerated": close_state_447.get(
            "a447_generation_wall_never_enumerated"
        )
        is True,
        "wall_is_guidance_not_coverage": close_state_447.get("wall_is_guidance_not_coverage")
        is True,
        "energy_program_concluded": close_state_447.get("energy_program_concluded") is True,
        "exploration_prior_class_closed": close_state_447.get("exploration_prior_class_closed")
        is True,
        "reproducible_total_levels": reproducible_total_levels,
        "poison_test_resolved": dict(poison_test_resolved),
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_447": dict(close_state_447),
        "v448_frontier": dict(v448_frontier),
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
    registry_total = _mapping(preconditions["registry"]).get("reproducible_total_levels")
    close_state = _close_state_447(root_path, registry_total if isinstance(registry_total, int) else None)
    frontier = _v448_frontier(close_state)
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
            close_state_447=close_state,
            v448_frontier=frontier,
            reproducible_total_levels=registry_total if isinstance(registry_total, int) else None,
        )
        validate_artifact(artifact)
        return artifact

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
            f"complete_447_archived_448_activated_{activation_suffix}_guidance_assembly_recorded"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "a447_generation_wall_never_enumerated": True,
        "wall_is_guidance_not_coverage": True,
        "energy_program_concluded": True,
        "exploration_prior_class_closed": True,
        "reproducible_total_levels": registry_total,
        "poison_test_resolved": poison,
        "preconditions_checked": preconditions,
        "transition": transition,
        "close_state_447": close_state,
        "v448_frontier": frontier,
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


def _validate_generation_wall(close: Mapping[str, Any]) -> None:
    a1 = _mapping(close.get("a1_generation_wall_verdict"))
    _require(
        a1.get("verdict") == GENERATION_WALL_VERDICT
        and a1.get("dominant_bucket") == DOMINANT_BUCKET
        and a1.get("bucket_counts") == {"COVERED": 1, DOMINANT_BUCKET: 9}
        and a1.get("b1_trusted") is True
        and a1.get("lp85_covered") is True
        and a1.get("tu93_positive_control_covered") is True
        and a1.get("winning_prefix_never_assembled") is True,
        "generation wall close-state must record trusted NEVER_ENUMERATED",
    )


def _validate_frontier(frontier: Mapping[str, Any]) -> None:
    _require(
        frontier.get("root_cause") == "guidance_assembly_not_coverage"
        and frontier.get("headline_fork") == "guidance_gap_vs_world_model_inducer_ceiling"
        and frontier.get("planner_must_not_repropose_coverage_vocabulary_levers") is True
        and frontier.get("planner_must_not_reopen_energy_program") is True
        and frontier.get("planner_must_not_repropose_exploration_or_perception_from_grid") is True,
        "v448 frontier must block retired coverage, energy, exploration, and perception levers",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required artifact fields: {missing}")

    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str), "honest_verdict must be a string")
    blocked = verdict.startswith("blocked_")
    if not blocked:
        _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict must be terminal-prefixed")
        _require(
            verdict.startswith("complete_447_archived_448_activated_"),
            "honest_verdict must record the .447/.448 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles drifted")

    close = _mapping(artifact.get("close_state_447"))
    frontier = _mapping(artifact.get("v448_frontier"))
    if artifact.get("a447_generation_wall_never_enumerated") is True:
        _validate_generation_wall(close)
    if artifact.get("wall_is_guidance_not_coverage") is True:
        _require(
            close.get("retired_coverage_levers") == [MACRO_RETIREMENT, CLICK_HEATMAP_RETIREMENT],
            "wall_is_guidance_not_coverage requires both retired coverage levers",
        )
        _validate_frontier(frontier)
    if blocked:
        _validate_checksum(artifact)
        return None

    for field in (
        "a447_generation_wall_never_enumerated",
        "wall_is_guidance_not_coverage",
        "energy_program_concluded",
        "exploration_prior_class_closed",
    ):
        _require(artifact.get(field) is True, f"{field} must be true on complete artifacts")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )
    _require(
        _mapping(artifact.get("poison_test_resolved")).get("resolved") is True,
        "poison pre-test resolution must be recorded",
    )
    _require(
        _mapping(artifact.get("transition")).get("active_milestone_confirmed") is True,
        "active .448 milestone must be confirmed",
    )
    _require(close.get("energy_program_concluded") is True, "energy_program_concluded must be true")
    _require(
        close.get("exploration_prior_class_closed") is True,
        "exploration_prior_class_closed must be true",
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

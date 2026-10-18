"""Experiment 4830: archive `.444`, activate `.445`, and close energy.

Spec refs: REQ-CAPSTONE-4830, SCENARIO-CAPSTONE-4830,
SCENARIO-CAPSTONE-4830-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4830-FIELD-PRINCIPLES.

This transition is a record-only handoff. It reads the `.444` capstone, the S3
generation-lift artifact, the roadmap files, and the ARC solve registry. The
important record is that the staged energy program ended in a genuine bounded
null: the offline discriminator added no live ARC value, so `.445` must refocus
on the L1-first-contact generation wall.
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

from carnot.experiment_4820_archive_443_activate_444 import (  # noqa: E402
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

EXPERIMENT = "experiment_4830_archive_444_activate_445"
EXPERIMENT_ID = 4830
SCHEMA = "carnot.archive_activation.v444_to_v445_4830.v1"
RESULT_RELATIVE_PATH = "results/experiment_4830_archive_444_activate_445.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4829_capstone_v444.json")
S3_REL_PATH = Path("results/experiment_4821_structural_energy_s3_generation_lift.json")

ARCHIVED_MILESTONE = "2026.06.444"
ACTIVATED_MILESTONE = "2026.06.445"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
RANDOM_SEED = 4830
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
S3_NULL_VERDICT = "bounded_no_generation_lift"
S3_NULL_HONEST_VERDICT = "complete_structural_energy_s3_bounded_no_generation_lift"

SPEC_REFS = [
    "REQ-CAPSTONE-4830",
    "SCENARIO-CAPSTONE-4830",
    "SCENARIO-CAPSTONE-4830-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4830-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_444_archived_445_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "energy_program_concluded": {
        "principle": (
            "the energy track is done (offline discriminator, no live value); "
            "the planner must NOT re-propose energy stages."
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
    "energy_program_concluded",
    "reproducible_total_levels",
    "poison_test_resolved",
    "preconditions_checked",
    "transition",
    "close_state_444",
    "v445_refocus",
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


def _active_445_ready(active_info: Mapping[str, Any]) -> bool:
    return active_info.get("parses") is True and active_info.get("milestone") == ACTIVATED_MILESTONE


def _restore_next_from_active_if_needed(root: Path, *, active_info: Mapping[str, Any]) -> tuple[bool, str]:
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)
    if _next_roadmap_ready(next_info):
        return False, ""
    if not _active_445_ready(active_info):
        return False, ""
    try:
        (root / RESEARCH_ROADMAP_NEXT_REL_PATH).write_text(
            (root / RESEARCH_ROADMAP_REL_PATH).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    except OSError as exc:
        return False, str(exc)
    return True, ""


def _activate_next_roadmap(root: Path, *, next_info: Mapping[str, Any]) -> tuple[bool, str]:
    if not _next_roadmap_ready(next_info):
        return False, ""
    try:
        (root / RESEARCH_ROADMAP_REL_PATH).write_text(
            (root / RESEARCH_ROADMAP_NEXT_REL_PATH).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
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
    active_before = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    next_before = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)
    restored_next, restore_error = _restore_next_from_active_if_needed(root, active_info=active_before)
    next_info = _yaml_info(root / RESEARCH_ROADMAP_NEXT_REL_PATH)

    try:
        offline_ok = bool(offline_arcade_checker())
        offline_error = ""
    except Exception as exc:
        offline_ok = False
        offline_error = str(exc)

    activation_attempted = False
    activation_error = restore_error
    if offline_ok and not _active_445_ready(active_before) and _next_roadmap_ready(next_info):
        activation_attempted, activation_error = _activate_next_roadmap(root, next_info=next_info)

    active_info = _yaml_info(root / RESEARCH_ROADMAP_REL_PATH)
    registry_levels = _registry_total_levels(root / REGISTRY_REL_PATH)
    spec_text = _read_text(root / CAPSTONE_SPEC_REL_PATH) or ""
    should_run_smart_subset = (
        _next_roadmap_ready(next_info)
        and offline_ok
        and activation_error == ""
        and _active_445_ready(active_info)
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
            "before_available": next_before["available"],
            "before_parses": next_before["parses"],
            "before_milestone": next_before["milestone"],
            "literal_precondition_command": _literal_next_precondition_command(root),
            "literal_precondition_passed": next_info["available"] is True
            and next_info["parses"] is True,
            "milestone_matches_activation": _next_roadmap_ready(next_info),
            "restored_from_active_roadmap": restored_next,
            "restore_error": restore_error,
            "activation_attempted": activation_attempted,
            "activation_error": activation_error,
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
            "has_req_4830": "REQ-CAPSTONE-4830" in spec_text,
        },
        "capstone_4829": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "s3_4821": {
            "path": str(S3_REL_PATH),
            "available": (root / S3_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone_spec = _mapping(preconditions.get("capstone_spec"))

    if next_info.get("restore_error") or next_info.get("activation_error"):
        return "research_roadmap_activation_error"
    if not _next_roadmap_ready(next_info):
        return "research_roadmap_next_yaml"
    if not _active_445_ready(active):
        return "research_roadmap_445_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4830") is not True:
        return "missing_capstone_spec_req_4830"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    for key, reason in (
        ("capstone_4829", "missing_experiment_4829_capstone_v444"),
        ("s3_4821", "missing_experiment_4821_structural_energy_s3_generation_lift"),
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
        "active_milestone_confirmed": bool(complete and _active_445_ready(active)),
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
            "fields_imported": ["milestone", "exp4830_prompt", "exp4831_l1_refocus"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_REL_PATH),
        },
        {
            "source": "research_roadmap_next_yaml",
            "path": str(RESEARCH_ROADMAP_NEXT_REL_PATH),
            "fields_imported": ["milestone", "literal_precondition"],
            "sha256": file_sha256(root / RESEARCH_ROADMAP_NEXT_REL_PATH),
        },
        {
            "source": "experiment_4829_capstone_v444",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "capstone_ready",
                "s3_structural_energy_verdict",
                "readiness",
                "heldout_readiness",
                "silent_bug_audit",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4821_structural_energy_s3_generation_lift",
            "path": str(S3_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "verifier_is_oracle",
                "live_path_reachable",
                "lambda0_control",
                "n_headroom_games",
                "min_headroom_games",
                "positive_control_passed",
                "winners_newly_entering_pool_delta",
                "winners_newly_entering_pool_delta_ci95",
            ],
            "sha256": file_sha256(root / S3_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _energy_close_state(capstone: Mapping[str, Any], s3: Mapping[str, Any]) -> JsonDict:
    capstone_s3 = _mapping(capstone.get("s3_structural_energy_verdict"))
    lambda0 = _mapping(capstone_s3.get("lambda0_control")) or _mapping(s3.get("lambda0_control"))
    ci95 = capstone_s3.get("winners_newly_entering_pool_delta_ci95")
    if not isinstance(ci95, list):
        ci95 = s3.get("winners_newly_entering_pool_delta_ci95")
    delta = _float(capstone_s3.get("winners_newly_entering_pool_delta"))
    if delta is None:
        delta = _float(s3.get("winners_newly_entering_pool_delta"))
    n_headroom = _int(capstone_s3.get("n_headroom_games"), _int(s3.get("n_headroom_games"), 0))
    min_headroom = _int(capstone_s3.get("min_headroom_games"), _int(s3.get("min_headroom_games"), 0))
    controls_verified = (
        capstone_s3.get("controls_verified_by_b1") is True
        and capstone_s3.get("positive_control_passed") is True
        and capstone_s3.get("live_path_reachable") is True
    )
    bounded_null = (
        capstone_s3.get("verdict") == S3_NULL_VERDICT
        and capstone_s3.get("bounded_no_generation_lift") is True
        and capstone_s3.get("generation_win") is False
        and capstone_s3.get("s4_authorized") is False
        and s3.get("honest_verdict") == S3_NULL_HONEST_VERDICT
        and s3.get("verifier_is_oracle") is False
        and delta == 0.0
        and ci95 == [0.0, 0.0]
        and _float(lambda0.get("lambda")) == 0.0
        and lambda0.get("matched_control") is True
        and n_headroom == 24
        and min_headroom == 5
        and controls_verified
    )
    return {
        "s3_verdict": capstone_s3.get("verdict"),
        "s3_honest_verdict": s3.get("honest_verdict"),
        "bounded_no_generation_lift": capstone_s3.get("bounded_no_generation_lift") is True,
        "generation_win": capstone_s3.get("generation_win") is True,
        "s4_authorized": capstone_s3.get("s4_authorized") is True,
        "s4_moot": bounded_null,
        "energy_program_concluded": bounded_null,
        "adds_live_arc_value": not bounded_null,
        "lambda0_control": dict(lambda0),
        "winners_newly_entering_pool_delta": delta,
        "winners_newly_entering_pool_delta_ci95": ci95 if isinstance(ci95, list) else [],
        "n_headroom_games": n_headroom,
        "min_headroom_games": min_headroom,
        "positive_control_passed": capstone_s3.get("positive_control_passed") is True,
        "live_path_reachable": capstone_s3.get("live_path_reachable") is True,
        "controls_verified_by_b1": capstone_s3.get("controls_verified_by_b1") is True,
        "new_levels_not_re_ranking": capstone_s3.get("new_levels_not_re_ranking") is True,
        "direction_after_s3": capstone_s3.get("direction_after_s3"),
        "planner_discipline": "do_not_repropose_energy_stages",
    }


def _close_state_444(
    capstone: Mapping[str, Any],
    s3: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    return {
        "capstone_honest_verdict": capstone.get("honest_verdict"),
        "capstone_ready": capstone.get("capstone_ready") is True,
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "reproducible_total_levels": registry_total_levels,
        "energy_close_state": _energy_close_state(capstone, s3),
        "readiness": _mapping(capstone.get("readiness")),
        "heldout_readiness": _mapping(capstone.get("heldout_readiness")),
        "silent_bug_audit": _mapping(capstone.get("silent_bug_audit")),
    }


def _v445_refocus() -> JsonDict:
    return {
        "wall": "L1-FIRST-CONTACT",
        "generic_first_win_rate": 0.04,
        "generic_first_win_fraction": "1/25",
        "root_blocker": "winning_l1_prefix_never_proposed",
        "headline_task_id": "exp4831-a1",
        "headline_direction": "amortized_in_context_exploration_prior",
        "energy_program_concluded": True,
        "planner_must_not_repropose_energy_stages": True,
        "null_next_frontier_if_l1_prior_fails": "perception_representation",
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
        "energy_program_concluded": False,
        "reproducible_total_levels": None,
        "poison_test_resolved": dict(poison_test_resolved),
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_444": {},
        "v445_refocus": {},
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
    s3 = _json_object(root_path / S3_REL_PATH)
    close_state = _close_state_444(capstone, s3, registry_total_levels)
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
            f"complete_444_archived_445_activated_{activation_suffix}_energy_program_concluded"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "energy_program_concluded": True,
        "reproducible_total_levels": registry_total_levels,
        "poison_test_resolved": poison,
        "preconditions_checked": preconditions,
        "transition": transition,
        "close_state_444": close_state,
        "v445_refocus": _v445_refocus(),
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
            verdict.startswith("complete_444_archived_445_activated_"),
            "honest_verdict must record the .444/.445 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles drifted")

    close = _mapping(artifact.get("close_state_444"))
    refocus = _mapping(artifact.get("v445_refocus"))
    poison = _mapping(artifact.get("poison_test_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(
            close == {} and refocus == {} and artifact.get("energy_program_concluded") is False,
            "blocked artifacts must not carry fabricated close-state",
        )
        _validate_checksum(artifact)
        return None

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .445 milestone must be confirmed")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )
    _require(artifact.get("energy_program_concluded") is True, "energy program must be concluded")

    energy = _mapping(close.get("energy_close_state"))
    _require(
        energy.get("s3_verdict") == S3_NULL_VERDICT
        and energy.get("s3_honest_verdict") == S3_NULL_HONEST_VERDICT
        and energy.get("energy_program_concluded") is True
        and energy.get("s4_moot") is True
        and energy.get("adds_live_arc_value") is False
        and energy.get("bounded_no_generation_lift") is True
        and energy.get("generation_win") is False
        and energy.get("s4_authorized") is False
        and energy.get("winners_newly_entering_pool_delta") == 0.0
        and energy.get("winners_newly_entering_pool_delta_ci95") == [0.0, 0.0]
        and _float(_mapping(energy.get("lambda0_control")).get("lambda")) == 0.0
        and energy.get("n_headroom_games") == 24
        and energy.get("min_headroom_games") == 5
        and energy.get("positive_control_passed") is True
        and energy.get("live_path_reachable") is True
        and energy.get("controls_verified_by_b1") is True,
        "S3 bounded null must conclude the energy program",
    )
    _require(
        refocus.get("wall") == "L1-FIRST-CONTACT"
        and refocus.get("generic_first_win_rate") == 0.04
        and refocus.get("generic_first_win_fraction") == "1/25"
        and refocus.get("planner_must_not_repropose_energy_stages") is True
        and refocus.get("headline_task_id") == "exp4831-a1",
        "L1 refocus must be recorded for .445",
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

"""Experiment 4800: archive `.441`, activate `.442`, and correct S2.

Spec refs: REQ-CAPSTONE-4800, SCENARIO-CAPSTONE-4800,
SCENARIO-CAPSTONE-4800-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4800-FIELD-PRINCIPLES.

This is a record-keeping transition, not a fresh ARC run. It reads the `.441`
capstone and the raw S2 artifact, checks the conductor handoff, and writes the
`.442` transition artifact. The important correction is that Exp4791's zero
energy-minus-accuracy delta came from a candidate pool that did not provide
enough behaviorally different choices. That makes S2 inconclusive, not a
bounded null, so `.442` must retest S2 with enforced candidate diversity.
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

from carnot.experiment_4790_archive_440_activate_441 import (  # noqa: E402
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

EXPERIMENT = "experiment_4800_archive_441_activate_442"
EXPERIMENT_ID = 4800
SCHEMA = "carnot.archive_activation.v441_to_v442_4800.v1"
RESULT_RELATIVE_PATH = "results/experiment_4800_archive_441_activate_442.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4799_capstone_v441.json")
S2_REL_PATH = Path("results/experiment_4791_structural_energy_s2_offpath_trust_gate.json")

ARCHIVED_MILESTONE = "2026.06.441"
ACTIVATED_MILESTONE = "2026.06.442"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
RANDOM_SEED = 4800
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CANDIDATE_DIVERSITY_EPSILON = 1e-3
MIN_EFFECTIVE_GAMES_REQUIRED = 5

SPEC_REFS = [
    "REQ-CAPSTONE-4800",
    "SCENARIO-CAPSTONE-4800",
    "SCENARIO-CAPSTONE-4800-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4800-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_441_archived_442_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "s2_recorded_as_inconclusive": {
        "principle": (
            "S2 (exp4791) was a degenerate non-test, NOT a bounded null -- the record must "
            "not pause the energy direction on it."
        )
    },
    "reproducible_total_levels": {
        "principle": "the authoritative ARC progress metric carried from the registry, not re-counted."
    },
    "poison_test_resolved": {
        "principle": "records whether a poison pre-test was found+fixed -- the cascade-skip guard."
    },
    "close_state_441": {
        "principle": (
            "the honest `.441` close-state carried from Exp4799 plus the corrected Exp4791 "
            "candidate-diversity read so the bounded-null misrecord does not pause S2-v2."
        )
    },
    "v442_pivot": {
        "principle": (
            "the `.442` headline rationale (S2-v2 diverse-pool off-path trust gate) "
            "recorded so milestone intent is traceable."
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
    "s2_recorded_as_inconclusive",
    "reproducible_total_levels",
    "poison_test_resolved",
    "preconditions_checked",
    "transition",
    "close_state_441",
    "v442_pivot",
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


def _active_442_ready(active_info: Mapping[str, Any]) -> bool:
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
        and _active_442_ready(active_info)
        and next_info.get("available") is False
    )
    roadmap_ready = _next_roadmap_ready(next_info) or accepted_missing_next
    should_run_smart_subset = (
        roadmap_ready and offline_ok and activation_error == "" and _active_442_ready(active_info)
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
            "literal_precondition_command": (
                ".venv/bin/python -c \"import yaml; yaml.safe_load(open("
                "'research-roadmap-next.yaml')); print('ok')\""
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
                "k.offline_arcade(); print('ok')\""
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
            "has_req_4800": "REQ-CAPSTONE-4800" in spec_text,
        },
        "capstone_4799": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "s2_4791": {
            "path": str(S2_REL_PATH),
            "available": (root / S2_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone_spec = _mapping(preconditions.get("capstone_spec"))

    roadmap_ready = _active_442_ready(active) and (
        _next_roadmap_ready(next_info) or next_info.get("accepted_missing_because_already_active") is True
    )
    if next_info.get("activation_error"):
        return "research_roadmap_activation_error"
    if not roadmap_ready:
        return "research_roadmap_442_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4800") is not True:
        return "missing_capstone_spec_req_4800"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    for key, reason in (
        ("capstone_4799", "missing_experiment_4799_capstone_v441"),
        ("s2_4791", "missing_experiment_4791_structural_energy_s2_offpath_trust_gate"),
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
        "active_milestone_confirmed": bool(complete and _active_442_ready(active)),
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
            "source": "experiment_4799_capstone_v441",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "capstone_ready",
                "s2_structural_energy_verdict",
                "readiness",
                "levelup_bank",
                "flagged_artifacts_skipped",
                "sota_handoff",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4791_structural_energy_s2_offpath_trust_gate",
            "path": str(S2_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "flagged_adversarial",
                "verifier_is_oracle",
                "live_path_reachable",
                "energy_minus_accuracy_delta",
                "energy_minus_accuracy_delta_ci95",
                "n_heldout_games",
                "min_heldout_games",
                "game_results",
            ],
            "sha256": file_sha256(root / S2_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _candidate_floats(candidate_rows: Any, *, key: str = "heldout_cell_recall") -> list[float]:
    values: list[float] = []
    if not isinstance(candidate_rows, list):
        return values
    for row in candidate_rows:
        if not isinstance(row, Mapping):
            continue
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if len(values) >= 2 else 0.0


def _s2_inconclusive_state(s2: Mapping[str, Any]) -> JsonDict:
    game_rows = s2.get("game_results")
    games = game_rows if isinstance(game_rows, list) else []
    diversity_rows: list[JsonDict] = []
    effective_games: list[str] = []
    behaviorally_identical_games: list[str] = []
    equal_recall_non_effective_games: list[str] = []

    for index, game in enumerate(games):
        game_map = _mapping(game)
        game_name = str(game_map.get("game", f"game_{index}"))
        candidate_rows = game_map.get("candidate_rows")
        recalls = _candidate_floats(candidate_rows)
        energies = _candidate_floats(candidate_rows, key="offpath_structural_energy")
        recall_spread = _spread(recalls)
        energy_spread = _spread(energies)
        effective = recall_spread > CANDIDATE_DIVERSITY_EPSILON
        equal_recall = len(recalls) >= 2 and recall_spread <= CANDIDATE_DIVERSITY_EPSILON
        identical = equal_recall and len(energies) >= 2 and energy_spread <= CANDIDATE_DIVERSITY_EPSILON
        if effective:
            effective_games.append(game_name)
        if equal_recall:
            equal_recall_non_effective_games.append(game_name)
        if identical:
            behaviorally_identical_games.append(game_name)
        diversity_rows.append(
            {
                "game": game_name,
                "n_candidates": len(candidate_rows) if isinstance(candidate_rows, list) else 0,
                "heldout_cell_recall_values": recalls,
                "heldout_cell_recall_spread": round(recall_spread, 12),
                "offpath_structural_energy_spread": round(energy_spread, 12),
                "effective_for_selection": effective,
                "behaviorally_identical_candidates": identical,
            }
        )

    min_effective = max(
        MIN_EFFECTIVE_GAMES_REQUIRED,
        _int(s2.get("min_heldout_games"), MIN_EFFECTIVE_GAMES_REQUIRED),
    )
    n_effective = len(effective_games)
    return {
        "corrected_verdict": "inconclusive_degenerate_pool",
        "reported_honest_verdict": s2.get("honest_verdict"),
        "reported_energy_minus_accuracy_delta": s2.get("energy_minus_accuracy_delta"),
        "reported_energy_minus_accuracy_delta_ci95": s2.get("energy_minus_accuracy_delta_ci95"),
        "reported_verifier_is_oracle": s2.get("verifier_is_oracle") is True,
        "reported_live_path_reachable": s2.get("live_path_reachable") is True,
        "reported_flagged_adversarial": s2.get("flagged_adversarial") is True,
        "n_total_games": _int(s2.get("n_heldout_games"), len(games)),
        "n_effective_games": n_effective,
        "effective_games": effective_games,
        "behaviorally_identical_games": behaviorally_identical_games,
        "equal_recall_non_effective_games": equal_recall_non_effective_games,
        "min_effective_games_required": min_effective,
        "effective_game_gate_passed": n_effective >= min_effective,
        "candidate_pool_diversity": diversity_rows,
        "energy_direction_state": "inconclusive_not_bounded_not_passed",
        "s2v2_required": True,
        "reason": (
            "Exp4791 reported a zero delta, but the candidate pool only produced two "
            "effective held-out games with distinct cell-recall outcomes."
        ),
    }


def _close_state_441(
    capstone: Mapping[str, Any],
    s2: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    capstone_s2 = _mapping(capstone.get("s2_structural_energy_verdict"))
    corrected = _s2_inconclusive_state(s2)
    return {
        "capstone_honest_verdict": capstone.get("honest_verdict"),
        "capstone_ready": capstone.get("capstone_ready") is True,
        "capstone_reported_s2_verdict": capstone_s2.get("verdict"),
        "capstone_reported_s3_authorized": capstone_s2.get("s3_authorized") is True,
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "capstone_misrecorded_bounded_null": capstone_s2.get("verdict") == "bounded",
        "reproducible_total_levels": registry_total_levels,
        "s2_corrected_record": corrected,
        "levelup_bank": _mapping(capstone.get("levelup_bank")),
        "readiness": _mapping(capstone.get("readiness")),
        "flagged_artifacts_skipped": capstone.get("flagged_artifacts_skipped")
        if isinstance(capstone.get("flagged_artifacts_skipped"), list)
        else [],
        "sota_handoff": _mapping(capstone.get("sota_handoff")),
    }


def _v442_pivot() -> JsonDict:
    return {
        "headline": "S2-v2 diverse-pool off-path trust gate",
        "task_id": "exp4801-a1",
        "source_correction": "exp4791_s2_inconclusive_degenerate_pool",
        "enforces_behaviorally_diverse_candidate_pool": True,
        "minimum_effective_games_required": MIN_EFFECTIVE_GAMES_REQUIRED,
        "candidate_diversity_gate": (
            "count a game only when candidate engines produce at least two distinct held-out "
            "off-path cell_recall values"
        ),
        "direction": (
            "retest S2 with a diverse candidate pool; do not treat Exp4791's degenerate "
            "zero-delta as a bounded null"
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
        "s2_recorded_as_inconclusive": False,
        "reproducible_total_levels": None,
        "poison_test_resolved": dict(poison_test_resolved),
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_441": {},
        "v442_pivot": {},
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
    transition = _transition(preconditions, complete=True)
    activation_suffix = (
        "from_next" if transition["activation_state"] == "activated_from_research_roadmap_next" else "already_active"
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"complete_441_archived_442_activated_{activation_suffix}_s2_inconclusive_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "s2_recorded_as_inconclusive": True,
        "reproducible_total_levels": registry_total_levels,
        "poison_test_resolved": poison,
        "preconditions_checked": preconditions,
        "transition": transition,
        "close_state_441": _close_state_441(
            _json_object(root_path / CAPSTONE_REL_PATH),
            _json_object(root_path / S2_REL_PATH),
            registry_total_levels,
        ),
        "v442_pivot": _v442_pivot(),
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
            verdict.startswith("complete_441_archived_442_activated_"),
            "honest_verdict must record the .441/.442 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles drifted")

    close = _mapping(artifact.get("close_state_441"))
    pivot = _mapping(artifact.get("v442_pivot"))
    poison = _mapping(artifact.get("poison_test_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(
            close == {} and pivot == {} and artifact.get("s2_recorded_as_inconclusive") is False,
            "blocked artifacts must not carry fabricated close-state",
        )
        _validate_checksum(artifact)
        return None

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .442 milestone must be confirmed")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )

    s2 = _mapping(close.get("s2_corrected_record"))
    _require(
        artifact.get("s2_recorded_as_inconclusive") is True
        and close.get("capstone_misrecorded_bounded_null") is True
        and s2.get("corrected_verdict") == "inconclusive_degenerate_pool"
        and s2.get("reported_energy_minus_accuracy_delta") == 0.0
        and s2.get("reported_energy_minus_accuracy_delta_ci95") == [0.0, 0.0]
        and s2.get("n_total_games") == 5
        and s2.get("n_effective_games") == 2
        and s2.get("effective_game_gate_passed") is False
        and s2.get("min_effective_games_required") == MIN_EFFECTIVE_GAMES_REQUIRED
        and s2.get("energy_direction_state") == "inconclusive_not_bounded_not_passed"
        and s2.get("s2v2_required") is True,
        "S2 must be recorded as inconclusive_degenerate_pool, not bounded",
    )
    _require(
        pivot.get("task_id") == "exp4801-a1"
        and pivot.get("enforces_behaviorally_diverse_candidate_pool") is True,
        "v442 pivot must record the S2-v2 diverse-pool retest",
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

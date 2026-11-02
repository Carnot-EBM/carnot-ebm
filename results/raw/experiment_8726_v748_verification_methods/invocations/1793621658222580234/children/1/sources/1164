"""Experiment 4770: archive `.438`, activate `.439`, and record `.438` honestly.

Spec refs: REQ-CAPSTONE-4770, SCENARIO-CAPSTONE-4770,
SCENARIO-CAPSTONE-4770-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4770-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4770_archive_438_activate_439"
EXPERIMENT_ID = 4770
SCHEMA = "carnot.archive_activation.v438_to_v439_4770.v1"
RESULT_RELATIVE_PATH = "results/experiment_4770_archive_438_activate_439.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4769_capstone_v438.json")
S0_REL_PATH = Path("results/experiment_4761_structural_energy_s0_core_bet_probe.json")

ARCHIVED_MILESTONE = "2026.06.438"
ACTIVATED_MILESTONE = "2026.06.439"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
RANDOM_SEED = 4770
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-4770",
    "SCENARIO-CAPSTONE-4770",
    "SCENARIO-CAPSTONE-4770-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4770-FIELD-PRINCIPLES",
]

FIELD_PROVENANCE = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_438_archived_439_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream JSON, no LLM; 0.0001s floor."
    },
    "poison_test_resolved": {
        "principle": "records whether a poison pre-test was found+fixed -- the cascade-skip guard."
    },
    "reproducible_total_levels": {
        "principle": "the authoritative ARC progress metric carried from the registry, not re-counted."
    },
    "close_state_438": {
        "principle": (
            "the honest .438 numbers carried forward from the capstone and S0 leak-audit artifact "
            "so the record does not drift."
        )
    },
    "v439_pivot": {
        "principle": (
            "the .439 headline rationale (S0' origin-matched re-test) recorded so milestone intent "
            "is traceable."
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
    "close_state_438",
    "v439_pivot",
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


def _float(value: Any, default: float = 0.0, *, ndigits: int = 12) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return round(float(value), ndigits)
    return default


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    return default


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


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
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _registry_total_levels(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        loaded = yaml.safe_load(text)
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


def _active_439_ready(active_info: Mapping[str, Any]) -> bool:
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
    accepted_missing_next = (
        not _next_roadmap_ready(next_info)
        and _active_439_ready(active_info)
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
            "has_req_4770": "REQ-CAPSTONE-4770" in spec_text,
        },
        "capstone_4769": {"path": str(CAPSTONE_REL_PATH), "available": (root / CAPSTONE_REL_PATH).exists()},
        "s0_4761": {"path": str(S0_REL_PATH), "available": (root / S0_REL_PATH).exists()},
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone_spec = _mapping(preconditions.get("capstone_spec"))

    roadmap_ready = _next_roadmap_ready(next_info) or (
        next_info.get("accepted_missing_because_already_active") is True and _active_439_ready(active)
    )
    if not roadmap_ready:
        return "research_roadmap_439_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4770") is not True:
        return "missing_capstone_spec_req_4770"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    for key, reason in (
        ("capstone_4769", "missing_experiment_4769_capstone_v438"),
        ("s0_4761", "missing_experiment_4761_structural_energy_s0_core_bet_probe"),
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
        "active_milestone_confirmed": bool(complete and _active_439_ready(active)),
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
            "source": "experiment_4769_capstone_v438",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "s0_structural_energy_verdict",
                "levelup_bank",
                "flagged_artifacts_skipped",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4761_structural_energy_s0_core_bet_probe",
            "path": str(S0_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "s0_gate_passed",
                "retire_energy_guided_direction",
                "retire_if_same_verdict",
                "loo_auroc_structural",
                "origin_probe_auroc",
                "verifier_is_oracle",
            ],
            "sha256": file_sha256(root / S0_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _close_state_438(
    capstone: Mapping[str, Any],
    s0: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    capstone_s0 = _mapping(capstone.get("s0_structural_energy_verdict"))
    direction = str(capstone_s0.get("direction") or "RETIRED")
    loo = _float(s0.get("loo_auroc_structural", capstone_s0.get("loo_auroc_structural")))
    origin_probe = _float(s0.get("origin_probe_auroc", capstone_s0.get("origin_probe_auroc")))
    leak_audit_failed = origin_probe >= 0.6
    headline = (
        f"structural LOO {round(loo, 3):.3f} but leak-audit-failed -> retired"
        if direction == "RETIRED" and leak_audit_failed
        else f"structural LOO {round(loo, 3):.3f} direction {direction.lower()}"
    )
    flagged = capstone.get("flagged_artifacts_skipped")
    return {
        "source_capstone_honest_verdict": capstone.get("honest_verdict"),
        "reproducible_total_levels": registry_total_levels,
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "s0_structural_energy": {
            "headline": headline,
            "direction": direction,
            "s1_queued": capstone_s0.get("s1_queued") is True,
            "s0_gate_passed": s0.get("s0_gate_passed", capstone_s0.get("s0_gate_passed")) is True,
            "loo_auroc_structural": loo,
            "loo_auroc_structural_rounded": round(loo, 3),
            "origin_probe_auroc": origin_probe,
            "origin_probe_threshold": 0.6,
            "leak_audit_failed": leak_audit_failed,
            "retired_on_leak": direction == "RETIRED" and leak_audit_failed,
            "retire_energy_guided_direction": s0.get("retire_energy_guided_direction") is True
            or capstone_s0.get("retire_energy_guided_direction") is True,
            "retire_if_same_verdict": s0.get("retire_if_same_verdict") is True
            or capstone_s0.get("retire_if_same_verdict") is True,
            "verifier_is_oracle": s0.get("verifier_is_oracle") is True,
            "structural_minus_marginal_delta_ci95": s0.get(
                "structural_minus_marginal_delta_ci95"
            ),
            "n_held_out_games": _int(s0.get("n_held_out_games")),
            "n_candidate_rows": _int(s0.get("n_candidate_rows")),
            "upstream_honest_verdict": s0.get("honest_verdict"),
        },
        "levelup_bank": _mapping(capstone.get("levelup_bank")),
        "flagged_artifacts_skipped": flagged if isinstance(flagged, list) else [],
    }


def _v439_pivot() -> JsonDict:
    return {
        "headline": "S0' origin-matched structural-energy re-test",
        "task_id": "exp4771-a1",
        "origin_matched_retest": True,
        "positive_class": "induced_correct_prediction",
        "negative_class": "induced_wrong_prediction",
        "purpose": "disentangle correctness signal from S0 real-vs-induced origin leak",
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
        "close_state_438": {},
        "v439_pivot": {},
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
        "honest_verdict": "complete_438_archived_439_activated_already_active_true_close_state_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "poison_test_resolved": poison,
        "reproducible_total_levels": registry_total_levels,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_438": _close_state_438(
            _json_object(root_path / CAPSTONE_REL_PATH),
            _json_object(root_path / S0_REL_PATH),
            registry_total_levels,
        ),
        "v439_pivot": _v439_pivot(),
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
            verdict.startswith("complete_438_archived_439_activated_"),
            "honest_verdict must record the .438/.439 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_provenance") == FIELD_PROVENANCE, "field_provenance principles drifted")

    close = _mapping(artifact.get("close_state_438"))
    pivot = _mapping(artifact.get("v439_pivot"))
    poison = _mapping(artifact.get("poison_test_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(close == {} and pivot == {}, "blocked artifacts must not carry fabricated close-state")
        _validate_checksum(artifact)
        return

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .439 milestone must be confirmed")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )

    s0 = _mapping(close.get("s0_structural_energy"))
    _require(
        s0.get("direction") == "RETIRED"
        and s0.get("leak_audit_failed") is True
        and s0.get("retired_on_leak") is True
        and s0.get("loo_auroc_structural_rounded") == 0.746
        and s0.get("origin_probe_auroc", 0.0) >= 0.6
        and s0.get("verifier_is_oracle") is False,
        "S0 retired-on-leak close-state must be recorded",
    )
    _require(
        pivot.get("task_id") == "exp4771-a1" and pivot.get("origin_matched_retest") is True,
        "v439 pivot must record S0' origin-matched re-test",
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

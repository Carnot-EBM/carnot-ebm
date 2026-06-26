"""Experiment 4780: archive `.439`, activate `.440`, and record `.439` honestly.

Spec refs: REQ-CAPSTONE-4780, SCENARIO-CAPSTONE-4780,
SCENARIO-CAPSTONE-4780-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4780-FIELD-PRINCIPLES.

This module is deliberately boring: it reads upstream files, validates that the
current milestone handoff is safe, and records the corrected S0' close-state.
That matters because the `.439` capstone skipped Exp4771 due to a stale
conductor false-positive, while the real artifact and research note reopened
the structural-energy program to S1.
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

EXPERIMENT = "experiment_4780_archive_439_activate_440"
EXPERIMENT_ID = 4780
SCHEMA = "carnot.archive_activation.v439_to_v440_4780.v1"
RESULT_RELATIVE_PATH = "results/experiment_4780_archive_439_activate_440.json"
OUTPUT_REL_PATH = Path(RESULT_RELATIVE_PATH)
RESEARCH_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
RESEARCH_ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
REGISTRY_REL_PATH = Path("ops/arc_solve_registry.yaml")
CAPSTONE_SPEC_REL_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPSTONE_REL_PATH = Path("results/experiment_4779_capstone_v439.json")
S0PRIME_REL_PATH = Path("results/experiment_4771_structural_energy_s0prime_origin_matched.json")
RESEARCH_NOTE_REL_PATH = Path(
    "docs/research-notes/oracle-distinct-structural-energy-program-2026-06-26.md"
)

ARCHIVED_MILESTONE = "2026.06.439"
ACTIVATED_MILESTONE = "2026.06.440"
BASELINE_REPRODUCIBLE_TOTAL_LEVELS = 65
RANDOM_SEED = 4780
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
S0_ORIGIN_PROBE_AUROC_BEFORE = 0.733
STALE_LINTER_FIX_COMMIT = "93db8c015"

SPEC_REFS = [
    "REQ-CAPSTONE-4780",
    "SCENARIO-CAPSTONE-4780",
    "SCENARIO-CAPSTONE-4780-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4780-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; clean transition is complete_439_archived_440_activated_<state>."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
    },
    "s0prime_reopen_recorded": {
        "principle": (
            "the .439 headline (S0' REOPENS to S1) must be carried forward even though "
            "the capstone skipped the flagged artifact."
        )
    },
    "reproducible_total_levels": {
        "principle": "the authoritative ARC progress metric carried from the registry, not re-counted."
    },
    "poison_test_resolved": {
        "principle": "records whether a poison pre-test was found+fixed -- the cascade-skip guard."
    },
    "close_state_439": {
        "principle": (
            "the honest `.439` close-state carried from Exp4779 plus the corrected Exp4771 "
            "S0' result so the stale-conductor skip does not erase the real headline."
        )
    },
    "v440_pivot": {
        "principle": (
            "the `.440` headline rationale (S1 contrastive energy landscape authorized by S0') "
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
    "s0prime_reopen_recorded",
    "reproducible_total_levels",
    "poison_test_resolved",
    "preconditions_checked",
    "transition",
    "close_state_439",
    "v440_pivot",
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


def _active_440_ready(active_info: Mapping[str, Any]) -> bool:
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
        and _active_440_ready(active_info)
        and next_info.get("available") is False
    )
    roadmap_ready = _next_roadmap_ready(next_info) or accepted_missing_next
    should_run_smart_subset = (
        roadmap_ready and offline_ok and activation_error == "" and _active_440_ready(active_info)
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
            "has_req_4780": "REQ-CAPSTONE-4780" in spec_text,
        },
        "capstone_4779": {
            "path": str(CAPSTONE_REL_PATH),
            "available": (root / CAPSTONE_REL_PATH).exists(),
        },
        "s0prime_4771": {
            "path": str(S0PRIME_REL_PATH),
            "available": (root / S0PRIME_REL_PATH).exists(),
        },
        "research_note": {
            "path": str(RESEARCH_NOTE_REL_PATH),
            "available": (root / RESEARCH_NOTE_REL_PATH).exists(),
        },
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    next_info = _mapping(preconditions.get("research_roadmap_next_yaml"))
    active = _mapping(preconditions.get("active_research_roadmap_yaml"))
    offline = _mapping(preconditions.get("offline_arcade"))
    smart = _mapping(preconditions.get("smart_subset_pretest_gate"))
    registry = _mapping(preconditions.get("registry"))
    capstone_spec = _mapping(preconditions.get("capstone_spec"))

    roadmap_ready = _active_440_ready(active) and (
        _next_roadmap_ready(next_info) or next_info.get("accepted_missing_because_already_active") is True
    )
    if next_info.get("activation_error"):
        return "research_roadmap_activation_error"
    if not roadmap_ready:
        return "research_roadmap_440_unavailable"
    if offline.get("available") is not True:
        return "offline_arcade"
    if smart.get("passed") is not True:
        return "smart_subset_pretest_gate"
    if _mapping(preconditions.get("agents_md")).get("available") is not True:
        return "missing_agents_md"
    if _mapping(preconditions.get("codex_or_opencode_md")).get("available") is not True:
        return "missing_codex_or_opencode_md"
    if capstone_spec.get("has_req_4780") is not True:
        return "missing_capstone_spec_req_4780"
    if registry.get("available") is not True:
        return "arc_solve_registry"
    if registry.get("reproducible_total_levels") != BASELINE_REPRODUCIBLE_TOTAL_LEVELS:
        return "arc_solve_registry_total_levels_not_65"
    for key, reason in (
        ("capstone_4779", "missing_experiment_4779_capstone_v439"),
        ("s0prime_4771", "missing_experiment_4771_structural_energy_s0prime_origin_matched"),
        ("research_note", "missing_oracle_distinct_structural_energy_program_note"),
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
        "active_milestone_confirmed": bool(complete and _active_440_ready(active)),
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
            "source": "experiment_4779_capstone_v439",
            "path": str(CAPSTONE_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "s0prime_structural_energy_verdict",
                "flagged_artifacts_skipped",
                "levelup_bank",
                "readiness",
                "silent_bug_audit",
                "reproducible_total_levels",
            ],
            "sha256": file_sha256(root / CAPSTONE_REL_PATH),
        },
        {
            "source": "experiment_4771_structural_energy_s0prime_origin_matched",
            "path": str(S0PRIME_REL_PATH),
            "fields_imported": [
                "honest_verdict",
                "flagged_adversarial",
                "corrigendum_pending",
                "s0prime_gate_passed",
                "loo_auroc_structural",
                "loo_auroc_ci95",
                "origin_probe_auroc",
                "shuffled_label_control_auroc",
                "structural_minus_marginal_delta_ci95",
                "verifier_is_oracle",
            ],
            "sha256": file_sha256(root / S0PRIME_REL_PATH),
        },
        {
            "source": "oracle_distinct_structural_energy_program_note",
            "path": str(RESEARCH_NOTE_REL_PATH),
            "fields_imported": ["s0prime_reopens_s1_headline", "s1_authorized"],
            "sha256": file_sha256(root / RESEARCH_NOTE_REL_PATH),
        },
        {
            "source": "arc_solve_registry",
            "path": str(REGISTRY_REL_PATH),
            "fields_imported": ["reproducible_total_levels"],
            "sha256": file_sha256(root / REGISTRY_REL_PATH),
        },
    ]


def _capstone_skipped_s0prime(capstone: Mapping[str, Any]) -> bool:
    skipped = capstone.get("flagged_artifacts_skipped")
    if not isinstance(skipped, list):
        return False
    return any(_mapping(item).get("experiment_id") == 4771 for item in skipped)


def _s0prime_true_close_state(s0prime: Mapping[str, Any]) -> JsonDict:
    loo = _float(s0prime.get("loo_auroc_structural"))
    ci = s0prime.get("loo_auroc_ci95") if isinstance(s0prime.get("loo_auroc_ci95"), list) else []
    corr = s0prime.get("corrigendum_pending")
    false_positive = (
        s0prime.get("flagged_adversarial") is True
        and str(s0prime.get("honest_verdict")) == "success_structural_energy_s0prime_reopens_s1"
        and any(_mapping(item).get("kind") == "TAUTOLOGY" for item in corr if isinstance(corr, list))
    )
    return {
        "headline": "S0' REOPENS to S1 despite stale-conductor TAUTOLOGY skip",
        "direction": "REOPENS_TO_S1",
        "s1_queued": True,
        "honest_verdict": s0prime.get("honest_verdict"),
        "s0prime_gate_passed": s0prime.get("s0prime_gate_passed") is True,
        "artifact_flagged_adversarial": s0prime.get("flagged_adversarial") is True,
        "flag_is_known_false_positive": false_positive,
        "stale_linter_fix_commit": STALE_LINTER_FIX_COMMIT,
        "origin_probe_auroc_before": S0_ORIGIN_PROBE_AUROC_BEFORE,
        "origin_probe_auroc_after": _float(s0prime.get("origin_probe_auroc")),
        "loo_auroc_structural": loo,
        "loo_auroc_structural_rounded": round(loo, 3),
        "loo_auroc_ci95": ci,
        "loo_ci_excludes_chance": bool(ci and _float(ci[0]) > 0.5),
        "loo_auroc_marginal_control": _float(s0prime.get("loo_auroc_marginal_control")),
        "loo_auroc_majority_control": _float(s0prime.get("loo_auroc_majority_control")),
        "shuffled_label_control_auroc": _float(s0prime.get("shuffled_label_control_auroc")),
        "structural_minus_marginal_delta_ci95": s0prime.get("structural_minus_marginal_delta_ci95"),
        "per_family_loo": _mapping(s0prime.get("per_family_loo")),
        "origin_matched": _mapping(s0prime.get("dataset_diagnostics")).get("origin_matched") is True,
        "n_candidate_rows": _int(s0prime.get("n_candidate_rows")),
        "n_pos": _int(s0prime.get("n_pos")),
        "n_neg": _int(s0prime.get("n_neg")),
        "n_held_out_games": _int(s0prime.get("n_held_out_games")),
        "verifier_is_oracle": s0prime.get("verifier_is_oracle") is True,
        "retire_energy_guided_direction": s0prime.get("retire_energy_guided_direction") is True,
        "retire_if_same_verdict": s0prime.get("retire_if_same_verdict") is True,
    }


def _close_state_439(
    capstone: Mapping[str, Any],
    s0prime: Mapping[str, Any],
    registry_total_levels: int,
) -> JsonDict:
    capstone_s0prime = _mapping(capstone.get("s0prime_structural_energy_verdict"))
    return {
        "capstone_honest_verdict": capstone.get("honest_verdict"),
        "capstone_reported_s0prime_direction": capstone_s0prime.get("direction"),
        "capstone_skipped_s0prime": _capstone_skipped_s0prime(capstone),
        "capstone_skip_reason": capstone_s0prime.get("reason"),
        "capstone_reported_reproducible_total_levels": _int(
            capstone.get("reproducible_total_levels"), registry_total_levels
        ),
        "reproducible_total_levels": registry_total_levels,
        "s0prime_true_close_state": _s0prime_true_close_state(s0prime),
        "levelup_bank": _mapping(capstone.get("levelup_bank")),
        "readiness": _mapping(capstone.get("readiness")),
        "silent_bug_audit": _mapping(capstone.get("silent_bug_audit")),
        "sota_handoff": _mapping(capstone.get("sota_handoff")),
        "flagged_artifacts_skipped": capstone.get("flagged_artifacts_skipped")
        if isinstance(capstone.get("flagged_artifacts_skipped"), list)
        else [],
        "note_citation": {
            "path": str(RESEARCH_NOTE_REL_PATH),
            "conclusion": "S0' passed; structural energy direction is alive; S1 authorized.",
        },
    }


def _v440_pivot() -> JsonDict:
    return {
        "headline": "S1 contrastive energy landscape",
        "task_id": "exp4781-a1",
        "s1_authorized_by_s0prime": True,
        "foundation_experiment": "experiment_4771_structural_energy_s0prime_origin_matched",
        "target": "promote the S0' origin-matched logistic into a contrastive energy E(s,a,s')",
        "gate": (
            "validate -deltaE descent across >=10 seeds with origin-probe and shuffled-label "
            "controls carried forward"
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
        "s0prime_reopen_recorded": False,
        "reproducible_total_levels": None,
        "poison_test_resolved": dict(poison_test_resolved),
        "preconditions_checked": dict(preconditions_checked),
        "transition": _transition(preconditions_checked, complete=False),
        "close_state_439": {},
        "v440_pivot": {},
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
    close_state = _close_state_439(
        _json_object(root_path / CAPSTONE_REL_PATH),
        _json_object(root_path / S0PRIME_REL_PATH),
        registry_total_levels,
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete_439_archived_440_activated_already_active_s0prime_reopen_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "s0prime_reopen_recorded": True,
        "reproducible_total_levels": registry_total_levels,
        "poison_test_resolved": poison,
        "preconditions_checked": preconditions,
        "transition": _transition(preconditions, complete=True),
        "close_state_439": close_state,
        "v440_pivot": _v440_pivot(),
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
            verdict.startswith("complete_439_archived_440_activated_"),
            "honest_verdict must record the .439/.440 terminal transition",
        )

    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact.get("leaderboard_submission") is False, "leaderboard_submission must remain false")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles drifted")

    close = _mapping(artifact.get("close_state_439"))
    pivot = _mapping(artifact.get("v440_pivot"))
    poison = _mapping(artifact.get("poison_test_resolved"))
    transition = _mapping(artifact.get("transition"))
    if blocked:
        _require(
            close == {} and pivot == {} and artifact.get("s0prime_reopen_recorded") is False,
            "blocked artifacts must not carry fabricated close-state",
        )
        _validate_checksum(artifact)
        return

    _require(poison.get("resolved") is True, "poison pre-test resolution must be recorded")
    _require(transition.get("active_milestone_confirmed") is True, "active .440 milestone must be confirmed")
    _require(
        artifact.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS
        and close.get("reproducible_total_levels") == BASELINE_REPRODUCIBLE_TOTAL_LEVELS,
        "registry total must be carried from arc_solve_registry",
    )
    _require(close.get("capstone_skipped_s0prime") is True, "capstone skipped S0' must be recorded")

    s0prime = _mapping(close.get("s0prime_true_close_state"))
    _require(
        artifact.get("s0prime_reopen_recorded") is True
        and s0prime.get("direction") == "REOPENS_TO_S1"
        and s0prime.get("s1_queued") is True
        and s0prime.get("honest_verdict") == "success_structural_energy_s0prime_reopens_s1"
        and s0prime.get("flag_is_known_false_positive") is True
        and s0prime.get("loo_auroc_structural_rounded") == 0.739
        and _float(s0prime.get("origin_probe_auroc_before")) >= 0.7
        and s0prime.get("origin_probe_auroc_after") == 0.5
        and s0prime.get("loo_ci_excludes_chance") is True
        and s0prime.get("verifier_is_oracle") is False,
        "S0' true REOPEN close-state must be recorded",
    )
    _require(
        pivot.get("task_id") == "exp4781-a1" and pivot.get("s1_authorized_by_s0prime") is True,
        "v440 pivot must record S1 authorization",
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

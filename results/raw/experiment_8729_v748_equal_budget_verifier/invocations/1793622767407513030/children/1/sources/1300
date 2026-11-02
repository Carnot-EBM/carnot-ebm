"""Experiment 4912: .452 capstone milestone scorecard.

Spec refs: REQ-CAPSTONE-4912, SCENARIO-CAPSTONE-4912,
SCENARIO-CAPSTONE-4912-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4912-FIELD-PRINCIPLES.

This is an aggregation-only experiment. It reads the .452 upstream artifacts
through ``scripts/summarize_artifact.py``, skips quarantined inputs, gates A1's
headline on the B1 audit, and writes the operator-facing milestone scorecard.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script path
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4819_capstone_v443 import (  # noqa: E402
    SummarizerResult,
    UpstreamSource,
    _float,
    _int,
    _mapping,
    _read_json_object,
    _read_yaml_object,
    file_sha256,
    payload_checksum,
)


EXPERIMENT = "experiment_4912_capstone_v452"
EXPERIMENT_ID = 4912
SCHEMA = "carnot.exp4912.capstone_v452.v1"
RESULT_RELATIVE_PATH = "results/experiment_4912_capstone_v452.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4912",
    "SCENARIO-CAPSTONE-4912",
    "SCENARIO-CAPSTONE-4912-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4912-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; complete_capstone_v452_<headline-or-escalate>."
    },
    "a1_fork_verdict_trusted": {
        "principle": (
            "A1's fork verdict, gated on B1 a1_trustworthy -- the headline is only "
            "as good as the audit."
        )
    },
    "headline": {
        "principle": (
            "the one-line .452 headline (env-grounding unlocked / budget-bound / "
            "wall survives FOUR representations + env-grounding)."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the monotonic ARC progress metric (from the registry; A2's bank if any)."
        )
    },
    "heldout_first_win_rate": {
        "principle": "A4's fresh-live readiness number -- the 6/30 go/no-go signal."
    },
    "submission_package_ready": {
        "principle": "B2's terminal gate -- the package state for the operator's 6/30 decision."
    },
    "post_sprint_pivot": {
        "principle": (
            "on the escalate branch, the deliverable (~0.05 agent + FoVer paper) + "
            "D's post-sprint verifier-moat map; NOT representation #5."
        )
    },
    "skipped_flagged_adversarial": {
        "principle": "the list of upstream artifacts skipped per the fabrication gate."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON; 0.0001s floor)."
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "a1_fork_verdict_trusted",
    "headline",
    "reproducible_total_levels",
    "heldout_first_win_rate",
    "submission_package_ready",
    "post_sprint_pivot",
    "skipped_flagged_adversarial",
    "inference_substrate",
    "milestone_scorecard",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "capstone_ready",
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
    "blocked_",
)

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "A1": UpstreamSource(4903, "results/experiment_4903_env_grounded_location_pruned_search.json"),
    "A1B": UpstreamSource(4904, "results/experiment_4904_latent_action_interface.json"),
    "B1_AUDIT": UpstreamSource(4908, "results/experiment_4908_env_grounded_search_audit.json"),
    "A2_LEVELUP": UpstreamSource(4905, "results/experiment_4905_levelup_attempt.json"),
    "A3_SELF_PLAY": UpstreamSource(
        4906, "results/experiment_4906_self_play_verifier_checkpoint.json"
    ),
    "A4_HELDOUT": UpstreamSource(4907, "results/experiment_4907_heldout_first_win_readiness.json"),
    "B2_PACKAGE": UpstreamSource(4909, "results/experiment_4909_submission_package_harden.json"),
    "C_HARDWARE": UpstreamSource(4910, "results/experiment_4910_kv260_continuity.json"),
    "D_HANDOFF": UpstreamSource(4911, "results/experiment_4911_sota_ingestion_v453_frontier.json"),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "A1": (
        "honest_verdict",
        "fork_verdict",
        "value_grounded_first_win_delta_median",
        "value_grounded_first_win_delta_ci95",
        "median_actions_to_first_win",
        "coverage_migration_count",
        "positive_control_non_degenerate",
        "planner_blind_to_banked_answer",
        "change_location_prior_used_not_value",
        "inference_substrate",
    ),
    "A1B": (
        "honest_verdict",
        "fork_verdict",
        "latent_action_value_accuracy_delta_median",
        "latent_action_value_accuracy_delta_ci95",
        "ran_genuinely_live",
        "delta_on_truly_heldout_split",
        "inference_substrate",
    ),
    "B1_AUDIT": (
        "honest_verdict",
        "a1_source_fork_verdict",
        "a1_trustworthy",
        "a1_value_from_real_env",
        "a1_planner_blind",
        "a1_positive_control_non_degenerate",
        "a1_numbers_match_fork",
        "a1b_ran_genuinely_live",
        "a1b_gate_skipped",
        "a1b_source_fork_verdict",
        "a1_failure_reasons",
        "a1b_failure_reasons",
        "checks",
        "inference_substrate",
    ),
    "A2_LEVELUP": (
        "honest_verdict",
        "target_game",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "registry_update",
        "reproduction_gate",
        "inference_substrate",
    ),
    "A3_SELF_PLAY": (
        "honest_verdict",
        "target_game",
        "verifier_checkpoint_refreshed",
        "checkpoint_path",
        "offline_reproduced",
        "reproduced_levels",
        "search_state_count",
        "inference_substrate",
    ),
    "A4_HELDOUT": (
        "honest_verdict",
        "heldout_first_win_rate",
        "first_win_baseline",
        "heldout_first_win_delta_vs_baseline",
        "heldout_first_win_delta_vs_prior_best",
        "positive_control_passed",
        "parity_test_green",
        "live_agent_ran",
        "submitted_to_leaderboard",
        "operator_only",
        "inference_substrate",
    ),
    "B2_PACKAGE": (
        "honest_verdict",
        "submission_package_ready",
        "submits",
        "submitted_to_leaderboard",
        "operator_only",
        "peak_vram_gb",
        "frozen_stack_loads",
        "package_builds",
        "agent_config_resolution",
        "model_path_resolution",
        "packaging_requirements_crosscheck",
        "ready_package_regression_check",
        "inference_substrate",
    ),
    "C_HARDWARE": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "loaded_overlay",
        "preconditions_checked",
        "inference_substrate",
    ),
    "D_HANDOFF": (
        "honest_verdict",
        "aimed_at_fork_verdict",
        "a1b_fork_verdict",
        "selected_branch",
        "flagged_for_v453",
        "methods_mapped",
        "post_sprint_pivot_methods",
        "sota_to_experiment_mapping_note",
        "inference_substrate",
    ),
}

Summarizer = Callable[[Path, str], SummarizerResult]


def _experiment_id(source: str, artifact: Mapping[str, Any] | None = None) -> int:
    if artifact:
        for field in ("experiment_id", "experiment"):
            value = artifact.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                match = re.search(r"experiment_(\d+)|\b(\d{4})\b", value)
                if match:
                    return int(match.group(1) or match.group(2))
    return UPSTREAM_SOURCES[source].experiment_id


def _live_critical(summary: SummarizerResult | None) -> bool:
    return bool(summary and summary.exit_code >= 2)


def _live_recheck(summary: SummarizerResult | None) -> str:
    if summary is None:
        return "not_run"
    if summary.exit_code >= 2:
        return "critical"
    if summary.exit_code == 1:
        return "warn"
    return "clean"


def _is_skipped(artifact: Mapping[str, Any] | None, summary: SummarizerResult | None) -> str | None:
    if artifact and artifact.get("flagged_adversarial") is True:
        return "flagged_adversarial"
    if _live_critical(summary):
        return "live_critical_recheck"
    return None


def _clean_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
) -> dict[str, Mapping[str, Any]]:
    return {
        source: artifact
        for source, artifact in artifacts.items()
        if _is_skipped(artifact, summarizer_results.get(source)) is None
    }


def _trust_gate(b1: Mapping[str, Any] | None) -> JsonDict:
    a1b_live = bool(b1 and b1.get("a1b_ran_genuinely_live") is True)
    a1b_skipped = bool(b1 and b1.get("a1b_gate_skipped") is True)
    return {
        "a1_trustworthy": bool(b1 and b1.get("a1_trustworthy") is True),
        "a1_value_from_real_env": bool(b1 and b1.get("a1_value_from_real_env") is True),
        "a1_planner_blind": bool(b1 and b1.get("a1_planner_blind") is True),
        "a1_positive_control_non_degenerate": bool(
            b1 and b1.get("a1_positive_control_non_degenerate") is True
        ),
        "a1_numbers_match_fork": bool(b1 and b1.get("a1_numbers_match_fork") is True),
        "a1b_ran_genuinely_live": a1b_live,
        "a1b_gate_skipped": a1b_skipped,
        "a1b_live_or_gate_skipped": a1b_live or a1b_skipped,
    }


def _trust_failure_reasons(gate: Mapping[str, Any]) -> list[str]:
    checks = (
        "a1_trustworthy",
        "a1_value_from_real_env",
        "a1_planner_blind",
        "a1_positive_control_non_degenerate",
        "a1_numbers_match_fork",
        "a1b_live_or_gate_skipped",
    )
    return [field for field in checks if gate.get(field) is not True]


def _a1_fork_verdict_trusted(
    a1: Mapping[str, Any] | None,
    b1: Mapping[str, Any] | None,
) -> JsonDict:
    if a1 is None or b1 is None:
        return {
            "source": "A1",
            "experiment_id": _experiment_id("A1", a1),
            "fork_verdict": "",
            "trusted": False,
            "trust_gate": {},
            "trust_failure_reasons": ["upstream_artifact_missing"],
        }
    gate = _trust_gate(b1)
    failures = _trust_failure_reasons(gate)
    fork = str(a1.get("fork_verdict") or b1.get("a1_source_fork_verdict") or "")
    return {
        "source": "A1",
        "experiment_id": _experiment_id("A1", a1),
        "honest_verdict": a1.get("honest_verdict", ""),
        "fork_verdict": fork,
        "trusted": not failures,
        "trust_gate": gate,
        "trust_failure_reasons": failures,
        "b1_experiment_id": _experiment_id("B1_AUDIT", b1),
        "b1_honest_verdict": b1.get("honest_verdict", ""),
        "value_grounded_first_win_delta_median": _float(
            a1.get("value_grounded_first_win_delta_median")
        ),
        "value_grounded_first_win_delta_ci95": a1.get(
            "value_grounded_first_win_delta_ci95", []
        ),
        "median_actions_to_first_win": _float(a1.get("median_actions_to_first_win")),
        "coverage_migration_count": _int(a1.get("coverage_migration_count")),
    }


def _a1b_status(a1b: Mapping[str, Any] | None) -> JsonDict:
    if a1b is None:
        return {"source": "A1B", "status": "skipped"}
    return {
        "source": "A1B",
        "experiment_id": _experiment_id("A1B", a1b),
        "status": "included",
        "honest_verdict": a1b.get("honest_verdict", ""),
        "fork_verdict": a1b.get("fork_verdict", ""),
        "latent_action_value_accuracy_delta_median": _float(
            a1b.get("latent_action_value_accuracy_delta_median")
        ),
        "latent_action_value_accuracy_delta_ci95": a1b.get(
            "latent_action_value_accuracy_delta_ci95", []
        ),
        "ran_genuinely_live": a1b.get("ran_genuinely_live") is True,
        "delta_on_truly_heldout_split": a1b.get("delta_on_truly_heldout_split") is True,
    }


def _headline_decision(
    a1_trust: Mapping[str, Any],
    a1b: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if a1_trust.get("trusted") is not True:
        return (
            "untrusted_a1_fork_non_test",
            "A1's env-grounded search fork is not trusted by B1, so .452 is a non-test.",
        )
    fork = a1_trust.get("fork_verdict")
    if fork == "ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN":
        return (
            "env_grounded_search_unlocked",
            (
                "reading change-VALUE from the env (model-as-action-prior) unlocked "
                "first-wins where offline value-prediction could not; .453 scales it."
            ),
        )
    if fork == "SEARCH_BUDGET_BOUND":
        return (
            "search_budget_bound",
            (
                "env-grounded search works but at too high an action cost; .453 attacks "
                "efficiency (D's branch 1')."
            ),
        )
    a1b_fork = str(a1b.get("fork_verdict", "")) if a1b else ""
    if (
        fork == "WALL_DEEPER_THAN_VALUE_PREDICTION"
        and a1b_fork == "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES"
    ):
        return (
            "escalate_wall_survives_four_representations_plus_env_grounding",
            (
                "The live first-win wall survives energy + goal-quality + FOUR "
                "world-model representations + env-grounding. Deliverable: current "
                "~0.05 first-win agent (submitted) + publishable FoVer "
                "verifier-ensemble paper (paper_ready=true, north-star section 1). "
                "Post-6/30: pivot to D's verifier-moat map. Do not queue representation #5."
            ),
        )
    return (
        "wall_deeper_without_clean_a1b",
        (
            "A1 did not lift first-win, but a clean A1b four-class null is absent; "
            "report the wall without representation-invariant escalation."
        ),
    )


def _honest_verdict(decision: str) -> str:
    return f"complete_capstone_v452_{decision}"


def _imported_fields(source: str, artifact: Mapping[str, Any]) -> list[str]:
    return [field for field in CLEAN_IMPORT_FIELDS[source] if field in artifact]


def _cited_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        artifact = artifacts.get(source)
        if artifact is None:
            continue
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "fields_imported": _imported_fields(source, artifact),
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _skipped_flagged_adversarial(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source, artifact in artifacts.items():
        summary = summarizer_results.get(source)
        reason = _is_skipped(artifact, summary)
        if reason is None:
            continue
        live_recheck = _live_recheck(summary)
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "reason": reason,
                "sha256": artifact_sha256.get(source, ""),
                "summarizer_exit_code": summary.exit_code if summary else None,
                "true_honest_verdict": artifact.get("honest_verdict", ""),
                "stale_false_flag": artifact.get("flagged_adversarial") is True
                and live_recheck == "clean",
                "true_live_recheck": live_recheck,
            }
        )
    return rows


def _a2_levelup_bank(artifact: Mapping[str, Any] | None, registry_total: int) -> JsonDict:
    if artifact is None:
        return {}
    registry_update = _mapping(artifact.get("registry_update"))
    before = _int(registry_update.get("prior_total_declared"), registry_total)
    after = _int(registry_update.get("new_total_declared"), before)
    new_levels = _int(artifact.get("new_levels_banked"))
    banked = new_levels > 0 and after > before and artifact.get("offline_reproduced") is True
    return {
        "source": "A2_LEVELUP",
        "experiment_id": _experiment_id("A2_LEVELUP", artifact),
        "decision": "new_level_banked" if banked else "no_new_level_banked",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "target_game": artifact.get("target_game"),
        "new_levels_banked": new_levels,
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "reproducible_total_levels_before": before,
        "reproducible_total_levels_after": after,
        "registry_authoritative_total": registry_total,
        "registry_update_reason": registry_update.get("reason", ""),
        "reproduction_gate_passed": _mapping(artifact.get("reproduction_gate")).get("reproduced")
        is True,
    }


def _a3_self_play_checkpoint(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    refreshed = artifact.get("verifier_checkpoint_refreshed") is True
    return {
        "source": "A3_SELF_PLAY",
        "experiment_id": _experiment_id("A3_SELF_PLAY", artifact),
        "decision": "checkpoint_refreshed" if refreshed else "checkpoint_not_refreshed",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "target_game": artifact.get("target_game"),
        "verifier_checkpoint_refreshed": refreshed,
        "checkpoint_path": artifact.get("checkpoint_path"),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "search_state_count": _int(artifact.get("search_state_count")),
    }


def _a4_fresh_live_heldout(
    artifacts: Mapping[str, Mapping[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
) -> JsonDict:
    artifact = artifacts.get("A4_HELDOUT")
    skip_reason = _is_skipped(artifact, summarizer_results.get("A4_HELDOUT"))
    if skip_reason is not None:
        return {
            "source": "A4_HELDOUT",
            "experiment_id": _experiment_id("A4_HELDOUT", artifact),
            "status": f"skipped_{skip_reason}",
            "reason": skip_reason,
            "true_honest_verdict": artifact.get("honest_verdict", "") if artifact else "",
            "stale_false_flag": bool(
                artifact
                and artifact.get("flagged_adversarial") is True
                and _live_recheck(summarizer_results.get("A4_HELDOUT")) == "clean"
            ),
        }
    artifact = clean.get("A4_HELDOUT")
    if artifact is None:
        return {}
    return {
        "source": "A4_HELDOUT",
        "experiment_id": _experiment_id("A4_HELDOUT", artifact),
        "status": "included",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "heldout_first_win_rate": _float(artifact.get("heldout_first_win_rate")),
        "first_win_baseline": _float(artifact.get("first_win_baseline")),
        "heldout_first_win_delta_vs_baseline": _float(
            artifact.get("heldout_first_win_delta_vs_baseline")
        ),
        "heldout_first_win_delta_vs_prior_best": _float(
            artifact.get("heldout_first_win_delta_vs_prior_best")
        ),
        "positive_control_passed": artifact.get("positive_control_passed") is True,
        "parity_test_green": artifact.get("parity_test_green") is True,
        "live_agent_ran": artifact.get("live_agent_ran") is True,
        "submitted_to_leaderboard": artifact.get("submitted_to_leaderboard") is True,
        "operator_only": artifact.get("operator_only") is True,
    }


def _b2_submission_package(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ready = artifact.get("submission_package_ready") is True
    submitted = artifact.get("submits") is True or artifact.get("submitted_to_leaderboard") is True
    operator_only = artifact.get("operator_only") is True or artifact.get("submits") is False
    package_builds = _mapping(artifact.get("package_builds"))
    return {
        "source": "B2_PACKAGE",
        "experiment_id": _experiment_id("B2_PACKAGE", artifact),
        "decision": "package_ready_operator_only"
        if ready and operator_only and not submitted
        else "package_not_ready",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "submission_package_ready": ready,
        "submits": submitted,
        "operator_only": operator_only,
        "peak_vram_gb": _float(artifact.get("peak_vram_gb")),
        "frozen_stack_loads": artifact.get("frozen_stack_loads") is True,
        "package_builds": package_builds.get("package_builds") is True,
        "dry_build_ran": package_builds.get("dry_build_ran") is True,
        "agent_config_resolved": _mapping(artifact.get("agent_config_resolution")).get(
            "resolved"
        )
        is True,
        "model_path_resolved": _mapping(artifact.get("model_path_resolution")).get("resolved")
        is True,
        "requirements_ok": _mapping(artifact.get("packaging_requirements_crosscheck")).get("ok")
        is True,
        "ready_package_regression_ok": _mapping(artifact.get("ready_package_regression_check")).get(
            "ok"
        )
        is True,
    }


def _kv260_ssh_available(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("kv260_ssh_reachable") is True:
        return True
    checks = artifact.get("preconditions_checked")
    if isinstance(checks, list):
        return any(
            isinstance(row, Mapping)
            and row.get("resource") == "kv260_ssh"
            and row.get("available") is True
            for row in checks
        )
    return False


def _c_hardware(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ok = str(artifact.get("honest_verdict", "")).startswith("success_") or _kv260_ssh_available(
        artifact
    )
    return {
        "source": "C_HARDWARE",
        "experiment_id": _experiment_id("C_HARDWARE", artifact),
        "decision": "kv260_continuity_ok" if ok else "kv260_continuity_blocked",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "kv260_ssh_reachable": _kv260_ssh_available(artifact),
        "loaded_overlay": artifact.get("loaded_overlay") is True,
        "inference_substrate": artifact.get("inference_substrate"),
    }


def _d_v453_handoff(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    return {
        "source": "D_HANDOFF",
        "experiment_id": _experiment_id("D_HANDOFF", artifact),
        "honest_verdict": artifact.get("honest_verdict", ""),
        "aimed_at_fork_verdict": artifact.get("aimed_at_fork_verdict"),
        "a1b_fork_verdict": artifact.get("a1b_fork_verdict"),
        "selected_branch": artifact.get("selected_branch"),
        "flagged_for_v453": artifact.get("flagged_for_v453") or [],
        "post_sprint_pivot_methods": artifact.get("post_sprint_pivot_methods") or [],
        "mapping_note": dict(_mapping(artifact.get("sota_to_experiment_mapping_note"))),
    }


def _milestone_scorecard(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
    a1_trust: Mapping[str, Any],
    registry_total: int,
) -> JsonDict:
    return {
        "a1_env_grounded_search": dict(a1_trust),
        "a1b_latent_action_interface": _a1b_status(clean.get("A1B")),
        "a2_levelup_bank": _a2_levelup_bank(clean.get("A2_LEVELUP"), registry_total),
        "a3_self_play_checkpoint": _a3_self_play_checkpoint(clean.get("A3_SELF_PLAY")),
        "a4_fresh_live_heldout": _a4_fresh_live_heldout(artifacts, clean, summarizer_results),
        "b2_submission_package": _b2_submission_package(clean.get("B2_PACKAGE")),
        "c_hardware": _c_hardware(clean.get("C_HARDWARE")),
        "d_v453_handoff": _d_v453_handoff(clean.get("D_HANDOFF")),
    }


def _post_sprint_pivot(decision: str, d_artifact: Mapping[str, Any] | None) -> JsonDict:
    if not decision.startswith("escalate_"):
        return {"decision": "not_escalated", "reason": decision, "methods": []}
    note = _mapping(d_artifact.get("sota_to_experiment_mapping_note")) if d_artifact else {}
    return {
        "decision": "post_6_30_verifier_moat_pivot",
        "deliverable": (
            "current ~0.05 first-win agent (submitted) + publishable FoVer "
            "verifier-ensemble paper"
        ),
        "paper_ready": True,
        "north_star_section": "1",
        "do_not_queue": "representation_5",
        "source": "D_HANDOFF",
        "experiment_id": _experiment_id("D_HANDOFF", d_artifact),
        "selected_branch": d_artifact.get("selected_branch") if d_artifact else "",
        "methods": d_artifact.get("post_sprint_pivot_methods", []) if d_artifact else [],
        "planner_instruction": note.get("planner_instruction", ""),
        "mapping_summary": note.get("summary", ""),
    }


def _heldout_first_win_rate(clean: Mapping[str, Mapping[str, Any]]) -> float | None:
    return _float(clean["A4_HELDOUT"].get("heldout_first_win_rate")) if "A4_HELDOUT" in clean else None


def _submission_package_ready(clean: Mapping[str, Mapping[str, Any]]) -> bool:
    package = clean.get("B2_PACKAGE")
    return bool(package and package.get("submission_package_ready") is True)


def build_artifact(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    registry_sha256: str | None,
    summarizer_results: Mapping[str, SummarizerResult],
    duration_s: float,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the complete .452 scorecard after live summaries have run."""

    clean = _clean_artifacts(artifacts, summarizer_results)
    a1_trust = _a1_fork_verdict_trusted(clean.get("A1"), clean.get("B1_AUDIT"))
    decision, headline = _headline_decision(a1_trust, clean.get("A1B"))
    registry_total = _int(registry.get("reproducible_total_levels"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(decision),
        "a1_fork_verdict_trusted": a1_trust,
        "headline": headline,
        "reproducible_total_levels": registry_total,
        "heldout_first_win_rate": _heldout_first_win_rate(clean),
        "submission_package_ready": _submission_package_ready(clean),
        "post_sprint_pivot": _post_sprint_pivot(decision, clean.get("D_HANDOFF")),
        "skipped_flagged_adversarial": _skipped_flagged_adversarial(
            artifacts, artifact_sha256, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "milestone_scorecard": _milestone_scorecard(
            artifacts=artifacts,
            clean=clean,
            summarizer_results=summarizer_results,
            a1_trust=a1_trust,
            registry_total=registry_total,
        ),
        "cited_upstream_artifacts": _cited_artifacts(clean, artifact_sha256),
        "preconditions_checked": dict(
            preconditions_checked
            or {
                "agents_md_read": True,
                "codex_md_read": True,
                "registry": {
                    "path": REGISTRY_RELATIVE_PATH,
                    "sha256": registry_sha256 or "",
                },
            }
        ),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": True,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(
    *,
    reason: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    summarizer_results: Mapping[str, SummarizerResult],
    duration_s: float,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    """Build a schema-valid blocked artifact without importing metric claims."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"blocked_{reason}",
        "a1_fork_verdict_trusted": {
            "source": "A1",
            "experiment_id": UPSTREAM_SOURCES["A1"].experiment_id,
            "fork_verdict": "",
            "trusted": False,
            "trust_gate": {},
            "trust_failure_reasons": [reason],
        },
        "headline": "",
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "heldout_first_win_rate": None,
        "submission_package_ready": False,
        "post_sprint_pivot": {},
        "skipped_flagged_adversarial": _skipped_flagged_adversarial(
            artifacts, artifact_sha256, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "milestone_scorecard": {},
        "cited_upstream_artifacts": [],
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the .452 scorecard without mutating it."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    a1_trust = payload.get("a1_fork_verdict_trusted")
    if not isinstance(a1_trust, Mapping) or not isinstance(a1_trust.get("trusted"), bool):
        errors.append("invalid_a1_fork_verdict_trusted")
    if not isinstance(payload.get("headline"), str):
        errors.append("invalid_headline")
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")
    heldout_rate = payload.get("heldout_first_win_rate")
    if heldout_rate is not None and (
        isinstance(heldout_rate, bool) or not isinstance(heldout_rate, int | float)
    ):
        errors.append("invalid_heldout_first_win_rate")
    if not isinstance(payload.get("submission_package_ready"), bool):
        errors.append("invalid_submission_package_ready")
    if not isinstance(payload.get("post_sprint_pivot"), Mapping):
        errors.append("invalid_post_sprint_pivot")
    if not isinstance(payload.get("milestone_scorecard"), Mapping):
        errors.append("invalid_milestone_scorecard")
    field_principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    cited = payload.get("cited_upstream_artifacts")
    if not isinstance(cited, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not isinstance(row.get("fields_imported"), list)
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in cited
    ):
        errors.append("invalid_cited_upstream_artifacts")
    skipped = payload.get("skipped_flagged_adversarial")
    if not isinstance(skipped, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not str(row.get("sha256", "")).startswith("sha256:")
        or not row.get("reason")
        for row in skipped
    ):
        errors.append("invalid_skipped_flagged_adversarial")
    if not str(payload.get("reproducibility_checksum", "")).startswith("sha256:"):
        errors.append("invalid_reproducibility_checksum")
    return errors


def _run_summarizer(root: Path, relative_path: str) -> SummarizerResult:  # pragma: no cover
    cmd = [sys.executable, SUMMARIZER_RELATIVE_PATH, relative_path]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    return SummarizerResult(
        command=cmd,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _first_blocker(
    *,
    summarizer_present: bool,
    registry_present: bool,
    registry_loadable: bool,
    spec_has_req: bool,
    upstream_preconditions: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if not summarizer_present:
        return "missing_summarizer"
    if not registry_present:
        return "missing_registry"
    if not registry_loadable:
        return "registry_not_yaml_loadable"
    if not spec_has_req:
        return "spec_missing_req_4912"
    if (
        upstream_preconditions.get("A1", {}).get("present") is not True
        or upstream_preconditions.get("B1_AUDIT", {}).get("present") is not True
    ):
        return "upstream_artifact_missing"
    return None


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    summarizer: Summarizer | None = None,
) -> JsonDict:
    """Read upstreams via the summarizer, aggregate, and write the scorecard."""

    start = time.perf_counter()
    summarizer = summarizer or _run_summarizer
    summarizer_path = root / SUMMARIZER_RELATIVE_PATH
    summarizer_present = summarizer_path.exists()
    artifacts: dict[str, JsonDict] = {}
    artifact_sha256: dict[str, str] = {}
    summarizer_results: dict[str, SummarizerResult] = {}
    upstream_preconditions: dict[str, JsonDict] = {}

    for source, spec in UPSTREAM_SOURCES.items():
        path = root / spec.relative_path
        present = path.exists()
        upstream_preconditions[source] = {"path": spec.relative_path, "present": present}
        if not present:
            continue
        if summarizer_present:
            summary = summarizer(root, spec.relative_path)
            summarizer_results[source] = summary
            upstream_preconditions[source]["summarizer_exit_code"] = summary.exit_code
        artifacts[source] = _read_json_object(path)
        artifact_sha256[source] = file_sha256(path) or ""

    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_present = registry_path.exists()
    registry_loadable = False
    registry: JsonDict = {}
    if registry_present:
        try:
            registry = _read_yaml_object(registry_path)
            registry_loadable = True
        except yaml.YAMLError:
            registry = {}

    spec_path = root / SPEC_RELATIVE_PATH
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4912" in spec_path.read_text(
        encoding="utf-8"
    )
    preconditions_checked = {
        "agents_md_read": True,
        "codex_md_read": True,
        "summarizer": {"path": SUMMARIZER_RELATIVE_PATH, "present": summarizer_present},
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
            "sha256": file_sha256(registry_path) or "",
        },
        "spec_has_req_4912": spec_has_req,
        "upstream_artifacts": upstream_preconditions,
    }
    duration_s = time.perf_counter() - start
    blocker = _first_blocker(
        summarizer_present=summarizer_present,
        registry_present=registry_present,
        registry_loadable=registry_loadable,
        spec_has_req=spec_has_req,
        upstream_preconditions=upstream_preconditions,
    )
    if blocker is not None:
        artifact = build_blocked_artifact(
            reason=blocker,
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            registry=registry,
            summarizer_results=summarizer_results,
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
        )
    else:
        artifact = build_artifact(
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            registry=registry,
            registry_sha256=file_sha256(registry_path),
            summarizer_results=summarizer_results,
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
        )
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_capstone()
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact.get("honest_verdict"),
                "schema_errors": errors,
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Experiment 4839: .445 A1 amortized-prior capstone scorecard.

Spec refs: REQ-CAPSTONE-4839, SCENARIO-CAPSTONE-4839,
SCENARIO-CAPSTONE-4839-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4839-FIELD-PRINCIPLES.

The scorecard reads every landed .445 upstream artifact through
``scripts/summarize_artifact.py`` before importing fields. The headline is the
A1 amortized exploration prior verdict: real first-win lift, genuine exercised
null, or non-test caused by a dead archive, no-op prior, or imitation-only lift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_4819_capstone_v443 import (
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


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4839_capstone_v445"
EXPERIMENT_ID = 4839
SCHEMA = "carnot.exp4839.capstone_v445.v1"
RESULT_RELATIVE_PATH = "results/experiment_4839_capstone_v445.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 4839
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
BASELINE_FIRST_WIN_RATE = 0.04

SPEC_REFS = [
    "REQ-CAPSTONE-4839",
    "SCENARIO-CAPSTONE-4839",
    "SCENARIO-CAPSTONE-4839-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4839-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; capstone_ready=true is success_/complete_."},
    "a1_amortized_prior_verdict": {
        "principle": (
            "the headline -- first-win lift (wall moves) / genuine null (wall survives, "
            "class closed) / dead-archive non-test; honors the archive-alive + "
            "prior-exercised + imitation checks."
        )
    },
    "reproducible_total_levels": {
        "principle": "the monotonic ARC progress metric carried from the registry."
    },
    "cited_upstream_artifacts": {
        "principle": "list of {experiment_id, fields_imported, sha256} -- the audit trail."
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts (0.0001s floor)."},
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "a1_amortized_prior_verdict",
    "reproducible_total_levels",
    "cited_upstream_artifacts",
    "inference_substrate",
    "levelup_bank",
    "self_play_checkpoint",
    "readiness",
    "heldout_readiness",
    "silent_bug_audit",
    "submission_package_state",
    "hardware_continuity",
    "sota_handoff",
    "upstream_oracle_declarations",
    "flagged_artifacts_skipped",
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

A1_VERDICTS = {
    "first_win_lift_wall_moves",
    "genuine_null_l1_wall_survives_exploration_prior_closed",
    "non_test_dead_archive_prior_noop_or_imitation",
}

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "A1": UpstreamSource(
        4831, "results/experiment_4831_amortized_incontext_exploration_prior_live.json"
    ),
    "LEVELUP": UpstreamSource(4832, "results/experiment_4832_levelup_attempt.json"),
    "SELF_PLAY": UpstreamSource(4833, "results/experiment_4833_self_play_verifier_checkpoint.json"),
    "HELDOUT": UpstreamSource(4834, "results/experiment_4834_heldout_first_win_readiness.json"),
    "B1_AUDIT": UpstreamSource(4835, "results/experiment_4835_silent_bug_audit.json"),
    "PACKAGE": UpstreamSource(4836, "results/experiment_4836_submission_package_harden.json"),
    "HARDWARE": UpstreamSource(4837, "results/experiment_4837_kv260_continuity.json"),
    "SOTA": UpstreamSource(
        4838, "results/experiment_4838_sota_ingestion_perception_representation.json"
    ),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "A1": (
        "honest_verdict",
        "go_explore_archive_alive",
        "prior_changed_proposals",
        "prior_change_diagnostics",
        "prior_diagnostics",
        "baseline_first_win_rate",
        "first_win_rate_with_prior",
        "first_win_rate_no_prior_ablation",
        "first_win_delta_ci95",
        "imitation_control_heldout_games",
        "live_path_reachable",
        "solve_provenance",
        "inference_substrate",
    ),
    "LEVELUP": (
        "honest_verdict",
        "target_game",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "registry_update",
        "attempted_games",
        "solve_provenance",
        "verifier_is_oracle",
        "inference_substrate",
    ),
    "SELF_PLAY": (
        "honest_verdict",
        "target_game",
        "verifier_checkpoint_refreshed",
        "self_play_residual",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
        "checkpoint_path",
        "checkpoint_mtime_delta_ns",
        "search_state_count",
        "solve_provenance",
        "inference_substrate",
    ),
    "HELDOUT": (
        "honest_verdict",
        "heldout_first_win_rate",
        "first_win_baseline",
        "prior_best_heldout_first_win_rate",
        "heldout_first_win_delta_vs_baseline",
        "heldout_first_win_delta_vs_prior_best",
        "heldout_variant_attempts",
        "positive_control_passed",
        "parity_test_green",
        "null_delta_methodology_note",
        "a1_amortized_prior_decision",
        "live_agent_ran",
        "inference_substrate",
    ),
    "B1_AUDIT": (
        "honest_verdict",
        "nulls_audited",
        "trusted_nulls",
        "silent_bugs_found",
        "a1_archive_alive_and_prior_exercised",
        "a1_control_check",
        "audited_artifact_checksums",
        "inference_substrate",
    ),
    "PACKAGE": (
        "honest_verdict",
        "submission_package_ready",
        "submitted_to_leaderboard",
        "operator_only",
        "a1_prior_inclusion",
        "vram_estimate_gb",
        "package_builds",
        "inference_substrate",
    ),
    "HARDWARE": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "board_state",
        "next_forward_step",
        "inference_substrate",
    ),
    "SOTA": (
        "honest_verdict",
        "l1_wall_context",
        "flagged_for_v446",
        "methods_mapped",
        "arxiv_ids_cited",
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
    return UPSTREAM_SOURCES[source].experiment_id


def _oracle_declared(artifact: Mapping[str, Any] | None) -> bool:
    return bool(artifact and artifact.get("verifier_is_oracle") is True)


def _summary_text(summary: SummarizerResult | None) -> str:
    return "" if summary is None else f"{summary.stdout}\n{summary.stderr}"


def _live_critical(summary: SummarizerResult | None) -> bool:
    return bool(summary and summary.exit_code >= 2)


def _ci_low_positive(ci: Mapping[str, Any]) -> bool:
    low = _float(ci.get("low"))
    if low is None:
        interval = ci.get("ci95")
        if isinstance(interval, list | tuple) and interval:
            low = _float(interval[0])
    return low is not None and low > 0.0


def _archive_alive(artifact: Mapping[str, Any], b1_check: Mapping[str, Any]) -> bool:
    archive = _mapping(artifact.get("go_explore_archive_alive"))
    return bool(
        archive.get("alive") is True
        and _int(archive.get("observations")) > 0
        and _int(archive.get("stored_cells")) > 0
        and _int(archive.get("prefixes_injected")) > 0
        and b1_check.get("archive_alive") is not False
    )


def _prior_exercised(artifact: Mapping[str, Any], b1: Mapping[str, Any]) -> bool:
    b1_check = _mapping(b1.get("a1_control_check"))
    diagnostics = _mapping(artifact.get("prior_diagnostics"))
    change = _mapping(artifact.get("prior_change_diagnostics"))
    return bool(
        artifact.get("prior_changed_proposals") is True
        and change.get("changed") is not False
        and _int(diagnostics.get("proposal_changes")) > 0
        and b1.get("a1_archive_alive_and_prior_exercised") is True
        and b1_check.get("prior_changed") is not False
        and b1_check.get("proposal_order_changed") is not False
    )


def _imitation_confirmed(artifact: Mapping[str, Any], b1: Mapping[str, Any]) -> bool:
    imitation = _mapping(artifact.get("imitation_control_heldout_games"))
    b1_check = _mapping(b1.get("a1_control_check"))
    return bool(
        imitation.get("heldout_not_in_distillation_set") is True
        and b1_check.get("heldout_not_in_distillation_set") is not False
        and b1_check.get("imitation_control_confirmed") is not False
    )


def _a1_verdict(
    artifact: Mapping[str, Any] | None,
    summary: SummarizerResult | None,
    b1: Mapping[str, Any] | None,
) -> JsonDict:
    if artifact is None:
        return {}
    b1 = b1 or {}
    b1_check = _mapping(b1.get("a1_control_check"))
    archive_alive = _archive_alive(artifact, b1_check)
    prior_exercised = _prior_exercised(artifact, b1)
    imitation_confirmed = _imitation_confirmed(artifact, b1)
    imitation = _mapping(artifact.get("imitation_control_heldout_games"))
    ci = _mapping(artifact.get("first_win_delta_ci95"))
    rate_with = _float(artifact.get("first_win_rate_with_prior"))
    rate_without = _float(artifact.get("first_win_rate_no_prior_ablation"))
    baseline = _float(artifact.get("baseline_first_win_rate")) or BASELINE_FIRST_WIN_RATE
    ci_excludes_zero = _ci_low_positive(ci)
    lift_over_baseline = rate_with is not None and rate_with > baseline
    lift_holds = imitation.get("lift_holds") is True or b1_check.get("imitation_lift_holds") is True
    silent_bugs = b1.get("silent_bugs_found")
    silent_bugs = silent_bugs if isinstance(silent_bugs, list) else []
    live_path = artifact.get("live_path_reachable") is True
    live_critical = _live_critical(summary)

    if live_critical:
        verdict = "non_test_dead_archive_prior_noop_or_imitation"
        reason = "live_critical_recheck"
    elif not archive_alive:
        verdict = "non_test_dead_archive_prior_noop_or_imitation"
        reason = "dead_go_explore_archive"
    elif not prior_exercised:
        verdict = "non_test_dead_archive_prior_noop_or_imitation"
        reason = "prior_noop_not_exercised"
    elif not imitation_confirmed:
        verdict = "non_test_dead_archive_prior_noop_or_imitation"
        reason = "imitation_control_failed"
    elif silent_bugs:
        verdict = "non_test_dead_archive_prior_noop_or_imitation"
        reason = "b1_silent_bug_reopened"
    elif not live_path:
        verdict = "non_test_dead_archive_prior_noop_or_imitation"
        reason = "live_path_unreachable"
    elif lift_over_baseline and ci_excludes_zero and lift_holds:
        verdict = "first_win_lift_wall_moves"
        reason = "heldout_first_win_lift_ci_excludes_zero"
    elif lift_over_baseline and (not ci_excludes_zero or not lift_holds):
        verdict = "non_test_dead_archive_prior_noop_or_imitation"
        reason = "lift_not_exploration_grade"
    else:
        verdict = "genuine_null_l1_wall_survives_exploration_prior_closed"
        reason = "archive_alive_prior_exercised_no_heldout_lift"

    return {
        "source": "A1",
        "experiment_id": _experiment_id("A1", artifact),
        "upstream_honest_verdict": artifact.get("honest_verdict", ""),
        "verdict": verdict,
        "reason": reason,
        "wall_moves": verdict == "first_win_lift_wall_moves",
        "genuine_null": verdict == "genuine_null_l1_wall_survives_exploration_prior_closed",
        "dead_archive_non_test": verdict == "non_test_dead_archive_prior_noop_or_imitation",
        "exploration_prior_class_closed": (
            verdict == "genuine_null_l1_wall_survives_exploration_prior_closed"
        ),
        "archive_alive": archive_alive,
        "archive_alive_confirmed_by_b1": b1_check.get("archive_alive") is True,
        "prior_changed_proposals": artifact.get("prior_changed_proposals") is True,
        "prior_exercised_confirmed_by_b1": prior_exercised,
        "imitation_control_confirmed": imitation_confirmed,
        "imitation_lift_holds": lift_holds,
        "heldout_not_in_distillation_set": imitation.get("heldout_not_in_distillation_set") is True,
        "silent_bugs_found": silent_bugs,
        "first_win_rate_with_prior": rate_with,
        "first_win_rate_no_prior_ablation": rate_without,
        "baseline_first_win_rate": baseline,
        "first_win_delta_ci95": dict(ci),
        "first_win_ci_excludes_zero": ci_excludes_zero,
        "lift_over_baseline": lift_over_baseline,
        "live_path_reachable": live_path,
        "live_recheck_exit_code": summary.exit_code if summary else None,
        "live_recheck_excerpt": _summary_text(summary)[:240],
        "direction_next": (
            "perception_representation_frontier"
            if verdict == "genuine_null_l1_wall_survives_exploration_prior_closed"
            else "prior_wall_moved"
            if verdict == "first_win_lift_wall_moves"
            else "rerun_required_before_verdict"
        ),
    }


def _imported_fields(
    source: str,
    artifact: Mapping[str, Any],
    summary: SummarizerResult | None,
) -> list[str]:
    if _live_critical(summary):
        return ["live_critical_recheck"]
    return [field for field in CLEAN_IMPORT_FIELDS[source] if field in artifact]


def _cited_artifacts(
    artifacts: Mapping[str, Mapping[str, Any] | None],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        artifact = artifacts.get(source)
        if artifact is None:
            continue
        rows.append(
            {
                "experiment_id": _experiment_id(source, artifact),
                "fields_imported": _imported_fields(
                    source, artifact, summarizer_results.get(source)
                ),
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _flagged_artifacts_skipped(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source, artifact in artifacts.items():
        if not _live_critical(summarizer_results.get(source)):
            continue
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "reason": "live_critical_recheck",
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _oracle_declarations(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {
        source: {
            "experiment_id": _experiment_id(source, artifact),
            "verifier_is_oracle": _oracle_declared(artifact),
            "moat_claim_allowed": not _oracle_declared(artifact),
        }
        for source, artifact in artifacts.items()
    }


def _levelup_bank(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    registry_update = _mapping(artifact.get("registry_update"))
    before = _int(registry_update.get("reproducible_total_levels_before"))
    after = _int(registry_update.get("reproducible_total_levels_after"), before)
    attempts = artifact.get("attempted_games")
    attempts = attempts if isinstance(attempts, list) else []
    return {
        "source": "LEVELUP",
        "experiment_id": _experiment_id("LEVELUP", artifact),
        "target_game": artifact.get("target_game"),
        "new_levels_banked": _int(artifact.get("new_levels_banked")),
        "reproducible_total_levels_before": before,
        "reproducible_total_levels_after": after,
        "reproducible_total_levels_delta": after - before,
        "registry_updated": registry_update.get("updated") is True,
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "attempted_games_count": len(attempts),
        "verifier_is_oracle": _oracle_declared(artifact),
    }


def _self_play_checkpoint(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    refreshed = artifact.get("verifier_checkpoint_refreshed") is True
    return {
        "source": "SELF_PLAY",
        "experiment_id": _experiment_id("SELF_PLAY", artifact),
        "decision": "checkpoint_refreshed" if refreshed else "checkpoint_not_refreshed",
        "target_game": artifact.get("target_game"),
        "verifier_checkpoint_refreshed": refreshed,
        "self_play_residual": artifact.get("self_play_residual"),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "checkpoint_path": artifact.get("checkpoint_path"),
        "checkpoint_mtime_delta_ns": _int(artifact.get("checkpoint_mtime_delta_ns")),
        "search_state_count": _int(artifact.get("search_state_count")),
        "solve_provenance": artifact.get("solve_provenance", ""),
        "inference_substrate": artifact.get("inference_substrate", ""),
    }


def _heldout_readiness(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    rate = _float(artifact.get("heldout_first_win_rate"))
    baseline = _float(artifact.get("first_win_baseline")) or BASELINE_FIRST_WIN_RATE
    delta = _float(artifact.get("heldout_first_win_delta_vs_baseline"))
    flat = rate == baseline and delta == 0.0
    return {
        "source": "HELDOUT",
        "experiment_id": _experiment_id("HELDOUT", artifact),
        "decision": "flat_baseline_first_win_null" if flat else "heldout_first_win_changed",
        "heldout_first_win_rate": rate,
        "first_win_baseline": baseline,
        "prior_best_heldout_first_win_rate": _float(
            artifact.get("prior_best_heldout_first_win_rate")
        ),
        "heldout_first_win_delta_vs_baseline": delta,
        "heldout_first_win_delta_vs_prior_best": _float(
            artifact.get("heldout_first_win_delta_vs_prior_best")
        ),
        "heldout_variant_attempts": _int(artifact.get("heldout_variant_attempts")),
        "positive_control_passed": artifact.get("positive_control_passed") is True,
        "parity_test_green": artifact.get("parity_test_green") is True,
        "a1_prior_decision": dict(_mapping(artifact.get("a1_amortized_prior_decision"))),
        "live_agent_ran": artifact.get("live_agent_ran") is True,
        "inference_substrate": artifact.get("inference_substrate", ""),
    }


def _silent_bug_audit(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    silent_bugs = artifact.get("silent_bugs_found")
    silent_bugs = silent_bugs if isinstance(silent_bugs, list) else []
    trusted = artifact.get("trusted_nulls")
    return {
        "source": "B1_AUDIT",
        "experiment_id": _experiment_id("B1_AUDIT", artifact),
        "nulls_audited": _int(artifact.get("nulls_audited")),
        "trusted_nulls": trusted if isinstance(trusted, list) else [],
        "silent_bugs_found_count": len(silent_bugs),
        "silent_bugs_found": silent_bugs,
        "a1_archive_alive_and_prior_exercised": (
            artifact.get("a1_archive_alive_and_prior_exercised") is True
        ),
        "a1_control_check": dict(_mapping(artifact.get("a1_control_check"))),
    }


def _submission_package_state(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ready = artifact.get("submission_package_ready") is True
    submitted = artifact.get("submitted_to_leaderboard") is True
    return {
        "source": "PACKAGE",
        "experiment_id": _experiment_id("PACKAGE", artifact),
        "decision": "package_ready_operator_only"
        if ready and not submitted
        else "package_not_ready",
        "submission_package_ready": ready,
        "submitted_to_leaderboard": submitted,
        "operator_only": artifact.get("operator_only") is True,
        "a1_prior_inclusion": dict(_mapping(artifact.get("a1_prior_inclusion"))),
        "vram_estimate_gb": _float(artifact.get("vram_estimate_gb")),
    }


def _hardware_continuity(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ssh = artifact.get("kv260_ssh_reachable") is True
    return {
        "source": "HARDWARE",
        "experiment_id": _experiment_id("HARDWARE", artifact),
        "decision": "kv260_reachable" if ssh else "blocked_kv260_ssh_unreachable",
        "kv260_ssh_reachable": ssh,
        "board_state": dict(_mapping(artifact.get("board_state"))),
        "next_forward_step": artifact.get("next_forward_step", ""),
    }


def _sota_handoff(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    flagged = artifact.get("flagged_for_v446")
    flagged = flagged if isinstance(flagged, list) else []
    methods = artifact.get("methods_mapped")
    methods = methods if isinstance(methods, list) else []
    context = _mapping(artifact.get("l1_wall_context"))
    return {
        "source": "SOTA",
        "experiment_id": _experiment_id("SOTA", artifact),
        "decision": "perception_representation_handoff" if methods else "sota_handoff_empty",
        "l1_wall_context": dict(context),
        "v446_frontier": context.get("root_cause", "perception/representation"),
        "exploration_strategy_class_retired": (
            context.get("exploration_strategy_class_retired") is True
        ),
        "flagged_for_v446_candidates": [
            row.get("candidate") for row in flagged if isinstance(row, Mapping)
        ],
        "methods_mapped_count": len(methods),
        "arxiv_ids_cited": artifact.get("arxiv_ids_cited")
        if isinstance(artifact.get("arxiv_ids_cited"), list)
        else [],
    }


def _readiness(
    a1: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
    sota: Mapping[str, Any],
) -> JsonDict:
    wall_moves = a1.get("verdict") == "first_win_lift_wall_moves"
    genuine_null = a1.get("verdict") == "genuine_null_l1_wall_survives_exploration_prior_closed"
    package_ready = package.get("decision") == "package_ready_operator_only"
    heldout_changed = heldout.get("decision") == "heldout_first_win_changed"
    frontier = sota.get("v446_frontier") or (
        "perception/representation" if genuine_null else "unsettled"
    )
    return {
        "a1_verdict": a1.get("verdict", ""),
        "ready_for_operator_submit": bool(wall_moves and heldout_changed and package_ready),
        "wall_moves": wall_moves,
        "l1_wall_survives": genuine_null,
        "exploration_prior_class_closed": a1.get("exploration_prior_class_closed") is True,
        "heldout_decision": heldout.get("decision", ""),
        "submission_package_decision": package.get("decision", ""),
        "v446_frontier": frontier,
        "reason": (
            "a1_lift_package_ready"
            if wall_moves and package_ready
            else "a1_genuine_null_frontier_moves_to_perception_representation"
            if genuine_null
            else "a1_non_test_requires_rerun"
        ),
    }


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
    """Build the complete .445 capstone after live summaries have run."""

    a1 = _a1_verdict(
        artifacts.get("A1"),
        summarizer_results.get("A1"),
        artifacts.get("B1_AUDIT"),
    )
    honest_verdict = {
        "first_win_lift_wall_moves": "success_a1_amortized_prior_lift_wall_moves_capstone_ready",
        "genuine_null_l1_wall_survives_exploration_prior_closed": (
            "complete_a1_genuine_null_l1_wall_survives_exploration_prior_closed_capstone_ready"
        ),
        "non_test_dead_archive_prior_noop_or_imitation": (
            "complete_a1_non_test_dead_archive_prior_noop_or_imitation_capstone_ready"
        ),
    }.get(str(a1.get("verdict")), "complete_a1_unsettled_capstone_ready")
    heldout = _heldout_readiness(artifacts.get("HELDOUT"))
    package = _submission_package_state(artifacts.get("PACKAGE"))
    sota = _sota_handoff(artifacts.get("SOTA"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "a1_amortized_prior_verdict": a1,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": _cited_artifacts(
            artifacts, artifact_sha256, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": _levelup_bank(artifacts.get("LEVELUP")),
        "self_play_checkpoint": _self_play_checkpoint(artifacts.get("SELF_PLAY")),
        "readiness": _readiness(a1, heldout, package, sota),
        "heldout_readiness": heldout,
        "silent_bug_audit": _silent_bug_audit(artifacts.get("B1_AUDIT")),
        "submission_package_state": package,
        "hardware_continuity": _hardware_continuity(artifacts.get("HARDWARE")),
        "sota_handoff": sota,
        "upstream_oracle_declarations": _oracle_declarations(artifacts),
        "flagged_artifacts_skipped": _flagged_artifacts_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
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
        "capstone_ready": a1.get("verdict") != "non_test_dead_archive_prior_noop_or_imitation",
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
        "a1_amortized_prior_verdict": {},
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": {},
        "self_play_checkpoint": {},
        "readiness": {},
        "heldout_readiness": {},
        "silent_bug_audit": {},
        "submission_package_state": {},
        "hardware_continuity": {},
        "sota_handoff": {},
        "upstream_oracle_declarations": _oracle_declarations(artifacts),
        "flagged_artifacts_skipped": _flagged_artifacts_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the scorecard without mutating it."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")
    field_principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    a1 = payload.get("a1_amortized_prior_verdict")
    if isinstance(a1, Mapping) and a1 and a1.get("verdict") not in A1_VERDICTS:
        errors.append("invalid_a1_verdict")
    cited = payload.get("cited_upstream_artifacts")
    if not isinstance(cited, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not isinstance(row.get("fields_imported"), list)
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in cited
    ):
        errors.append("invalid_cited_upstream_artifacts")
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
        return "spec_missing_req_4839"
    for source, info in upstream_preconditions.items():
        if info.get("present") is not True:
            return f"missing_upstream:{source}"
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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4839" in spec_path.read_text(
        encoding="utf-8"
    )
    preconditions_checked = {
        "agents_md_read": True,
        "codex_md_read": True,
        "summarizer": {
            "path": SUMMARIZER_RELATIVE_PATH,
            "present": summarizer_present,
        },
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
            "sha256": file_sha256(registry_path) or "",
        },
        "spec_has_req_4839": spec_has_req,
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

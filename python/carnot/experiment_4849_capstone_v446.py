"""Experiment 4849: .446 object-identity perception capstone scorecard.

Spec refs: REQ-CAPSTONE-4849, SCENARIO-CAPSTONE-4849,
SCENARIO-CAPSTONE-4849-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4849-FIELD-PRINCIPLES.

The scorecard reads every landed .446 upstream artifact through
``scripts/summarize_artifact.py`` before importing fields. The headline is the
A1 object-identity verdict, trusted only when the B1 audit confirms real frames
and a non-degenerate shape/motion tracker.
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
EXPERIMENT = "experiment_4849_capstone_v446"
EXPERIMENT_ID = 4849
SCHEMA = "carnot.exp4849.capstone_v446.v1"
RESULT_RELATIVE_PATH = "results/experiment_4849_capstone_v446.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 4849
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TARGET_GAMES = ("lp85", "r11l", "tu93")

SPEC_REFS = [
    "REQ-CAPSTONE-4849",
    "SCENARIO-CAPSTONE-4849",
    "SCENARIO-CAPSTONE-4849-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4849-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; capstone_ready=true is success_/complete_."},
    "a1_perception_probe_verdict": {
        "principle": (
            "the headline -- object-identity recovery (goal-grounding feasible) / "
            "genuine null (unrecoverable from rendered grid, deeper finding) / "
            "synthetic-only non-test; honors B1's real-frames + not-a-no-op checks."
        )
    },
    "scored_lever_state": {
        "principle": (
            "the operator 'c' deadline track -- {level_up_banked, "
            "heldout_first_win_rate, submission_package_ready}; the realistic "
            "6/30 signal."
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
    "a1_perception_probe_verdict",
    "scored_lever_state",
    "reproducible_total_levels",
    "cited_upstream_artifacts",
    "inference_substrate",
    "levelup_bank",
    "self_play_checkpoint",
    "heldout_readiness",
    "submission_package_state",
    "hardware_continuity",
    "sota_handoff",
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
    "object_identity_recovered_goal_grounding_feasible",
    "genuine_null_object_identity_unrecoverable_from_rendered_grid",
    "synthetic_only_or_degenerate_non_test",
}

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "A1": UpstreamSource(4841, "results/experiment_4841_object_identity_perception_probe.json"),
    "LEVELUP": UpstreamSource(4842, "results/experiment_4842_levelup_attempt.json"),
    "SELF_PLAY": UpstreamSource(4843, "results/experiment_4843_self_play_verifier_checkpoint.json"),
    "HELDOUT": UpstreamSource(4844, "results/experiment_4844_heldout_first_win_readiness.json"),
    "B1_AUDIT": UpstreamSource(4845, "results/experiment_4845_perception_probe_audit.json"),
    "PACKAGE": UpstreamSource(4846, "results/experiment_4846_submission_package_harden.json"),
    "HARDWARE": UpstreamSource(4847, "results/experiment_4847_kv260_continuity.json"),
    "SOTA": UpstreamSource(4848, "results/experiment_4848_sota_ingestion_object_world_model.json"),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "A1": (
        "honest_verdict",
        "measured_on_real_frames",
        "per_game_correspondence",
        "positive_control_tu93_passed",
        "games_with_recovery",
        "verifier_is_oracle",
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
        "live_agent_ran",
        "inference_substrate",
    ),
    "B1_AUDIT": (
        "honest_verdict",
        "a1_genuinely_exercised",
        "non_test_reasons",
        "recovered_games_from_rows",
        "claimed_recovery_matches_rows",
        "source_artifact_checksum",
        "checks",
        "inference_substrate",
    ),
    "PACKAGE": (
        "honest_verdict",
        "submission_package_ready",
        "submitted_to_leaderboard",
        "operator_only",
        "vram_estimate_gb",
        "package_builds",
        "packaging_requirements_crosscheck",
        "agent_config_resolution",
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
        "flagged_for_v447",
        "methods_mapped",
        "arxiv_ids_cited",
        "a1_perception_layer_input",
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


def _summary_text(summary: SummarizerResult | None) -> str:
    return "" if summary is None else f"{summary.stdout}\n{summary.stderr}"


def _live_critical(summary: SummarizerResult | None) -> bool:
    return bool(summary and summary.exit_code >= 2)


def _per_game_summary(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    rows = _mapping(artifact.get("per_game_correspondence"))
    out: dict[str, JsonDict] = {}
    for game in TARGET_GAMES:
        row = _mapping(rows.get(game))
        if not row:
            continue
        out[game] = {
            "shape_motion_score": _float(row.get("shape_motion_score")),
            "color_centroid_baseline_score": _float(row.get("color_centroid_baseline_score")),
            "delta_vs_baseline": (
                round(
                    float(row.get("shape_motion_score", 0.0))
                    - float(row.get("color_centroid_baseline_score", 0.0)),
                    6,
                )
                if _float(row.get("shape_motion_score")) is not None
                and _float(row.get("color_centroid_baseline_score")) is not None
                else None
            ),
            "n_frames": _int(row.get("n_frames")),
            "n_transition_pairs": _int(row.get("n_transition_pairs")),
            "recovered": row.get("recovered") is True,
            "source_kind": row.get("source_kind", ""),
            "source_path": row.get("source_path", ""),
            "frame_checksum": row.get("frame_checksum", ""),
        }
    return out


def _b1_checks(b1: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    checks = _mapping(b1.get("checks")) if b1 else {}
    return {
        "real_frames": _mapping(checks.get("measured_on_real_frames")),
        "tracker": _mapping(checks.get("tracker_changed_vs_baseline")),
        "positive_control": _mapping(checks.get("positive_control_and_recovery_claim")),
        "summarizer": _mapping(checks.get("summarizer_and_adversarial_verify")),
    }


def _target_rows_real(real_check: Mapping[str, Any]) -> bool:
    target_rows = _mapping(real_check.get("target_rows"))
    if real_check.get("passed") is not True:
        return False
    for game in TARGET_GAMES:
        row = _mapping(target_rows.get(game))
        if row.get("present") is not True or row.get("real_frame_backed") is not True:
            return False
        if row.get("source_kind") == "synthetic":
            return False
    return True


def _tracker_not_noop(tracker_check: Mapping[str, Any]) -> bool:
    deltas = _mapping(tracker_check.get("deltas"))
    nonzero = tracker_check.get("nonzero_delta_games")
    has_nonzero = isinstance(nonzero, list) and bool(nonzero)
    numeric_delta = any((_float(value) or 0.0) != 0.0 for value in deltas.values())
    return bool(tracker_check.get("passed") is True and (has_nonzero or numeric_delta))


def _positive_control_confirmed(
    artifact: Mapping[str, Any],
    positive_check: Mapping[str, Any],
) -> bool:
    return bool(
        artifact.get("positive_control_tu93_passed") is True
        and positive_check.get("passed") is True
        and positive_check.get("positive_control_passed") is True
    )


def _recovered_games(artifact: Mapping[str, Any]) -> list[str]:
    per_game = _per_game_summary(artifact)
    return [game for game in TARGET_GAMES if per_game.get(game, {}).get("recovered") is True]


def _a1_perception_verdict(
    artifact: Mapping[str, Any] | None,
    summary: SummarizerResult | None,
    b1: Mapping[str, Any] | None,
) -> JsonDict:
    if artifact is None:
        return {}
    checks = _b1_checks(b1)
    recovered = _recovered_games(artifact)
    real_frames = artifact.get("measured_on_real_frames") is True and _target_rows_real(
        checks["real_frames"]
    )
    tracker_not_noop = _tracker_not_noop(checks["tracker"])
    positive_control = _positive_control_confirmed(artifact, checks["positive_control"])
    non_test_reasons = b1.get("non_test_reasons") if isinstance(b1, Mapping) else []
    non_test_reasons = non_test_reasons if isinstance(non_test_reasons, list) else []
    b1_exercised = bool(
        b1
        and b1.get("a1_genuinely_exercised") is True
        and not non_test_reasons
        and checks["summarizer"].get("passed") is not False
    )
    claim_matches = bool(
        b1
        and b1.get("claimed_recovery_matches_rows") is True
        and checks["positive_control"].get("claimed_recovery_matches_rows") is not False
        and checks["positive_control"].get("verdict_matches_numbers") is not False
    )
    live_path = artifact.get("live_path_reachable") is True
    live_critical = _live_critical(summary)

    if live_critical:
        verdict = "synthetic_only_or_degenerate_non_test"
        reason = "live_critical_recheck"
    elif not b1_exercised:
        verdict = "synthetic_only_or_degenerate_non_test"
        reason = "b1_audit_non_test"
    elif not real_frames:
        verdict = "synthetic_only_or_degenerate_non_test"
        reason = "not_measured_on_real_frames"
    elif not tracker_not_noop:
        verdict = "synthetic_only_or_degenerate_non_test"
        reason = "tracker_baseline_noop"
    elif not positive_control:
        verdict = "synthetic_only_or_degenerate_non_test"
        reason = "positive_control_failed"
    elif not claim_matches:
        verdict = "synthetic_only_or_degenerate_non_test"
        reason = "recovery_claim_mismatch"
    elif not live_path:
        verdict = "synthetic_only_or_degenerate_non_test"
        reason = "live_path_unreachable"
    elif len(recovered) >= 2:
        verdict = "object_identity_recovered_goal_grounding_feasible"
        reason = "shape_motion_recovers_at_least_two_target_games"
    else:
        verdict = "genuine_null_object_identity_unrecoverable_from_rendered_grid"
        reason = "real_frame_tracker_exercised_only_one_or_zero_games_recovered"

    return {
        "source": "A1",
        "experiment_id": _experiment_id("A1", artifact),
        "upstream_honest_verdict": artifact.get("honest_verdict", ""),
        "verdict": verdict,
        "reason": reason,
        "goal_grounding_feasible": verdict == "object_identity_recovered_goal_grounding_feasible",
        "genuine_rendered_grid_null": (
            verdict == "genuine_null_object_identity_unrecoverable_from_rendered_grid"
        ),
        "synthetic_or_degenerate_non_test": verdict == "synthetic_only_or_degenerate_non_test",
        "games_with_recovery": _int(artifact.get("games_with_recovery")),
        "recovered_games": recovered,
        "measured_on_real_frames": artifact.get("measured_on_real_frames") is True,
        "measured_on_real_frames_confirmed_by_b1": real_frames,
        "tracker_not_baseline_noop": tracker_not_noop,
        "positive_control_tu93_passed": positive_control,
        "b1_audit_genuinely_exercised": b1_exercised,
        "claimed_recovery_matches_rows": claim_matches,
        "per_game_correspondence": _per_game_summary(artifact),
        "per_game_correspondence_deltas": dict(_mapping(b1.get("per_game_correspondence_deltas")))
        if b1
        else {},
        "live_path_reachable": live_path,
        "solve_provenance": artifact.get("solve_provenance", ""),
        "live_recheck_exit_code": summary.exit_code if summary else None,
        "live_recheck_excerpt": _summary_text(summary)[:240],
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


def _levelup_bank(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    registry_update = _mapping(artifact.get("registry_update"))
    before = _int(registry_update.get("reproducible_total_levels_before"))
    after = _int(registry_update.get("reproducible_total_levels_after"), before)
    attempts = artifact.get("attempted_games")
    attempts = attempts if isinstance(attempts, list) else []
    new_levels = _int(artifact.get("new_levels_banked"))
    return {
        "source": "LEVELUP",
        "experiment_id": _experiment_id("LEVELUP", artifact),
        "target_game": artifact.get("target_game"),
        "new_levels_banked": new_levels,
        "level_up_banked": bool(new_levels > 0 and after > before),
        "reproducible_total_levels_before": before,
        "reproducible_total_levels_after": after,
        "reproducible_total_levels_delta": after - before,
        "registry_updated": registry_update.get("updated") is True,
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "attempted_games_count": len(attempts),
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
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
    baseline = _float(artifact.get("first_win_baseline")) or 0.04
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
        "live_agent_ran": artifact.get("live_agent_ran") is True,
        "inference_substrate": artifact.get("inference_substrate", ""),
    }


def _submission_package_state(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ready = artifact.get("submission_package_ready") is True
    submitted = artifact.get("submitted_to_leaderboard") is True
    operator_only = artifact.get("operator_only") is True
    package_builds = _mapping(artifact.get("package_builds"))
    return {
        "source": "PACKAGE",
        "experiment_id": _experiment_id("PACKAGE", artifact),
        "decision": "package_ready_operator_only"
        if ready and operator_only and not submitted
        else "package_not_ready",
        "submission_package_ready": ready,
        "submitted_to_leaderboard": submitted,
        "operator_only": operator_only,
        "vram_estimate_gb": _float(artifact.get("vram_estimate_gb")),
        "package_builds": package_builds.get("package_builds") is True,
        "dry_build_ran": package_builds.get("dry_build_ran") is True,
        "packaging_requirements_ok": (
            _mapping(artifact.get("packaging_requirements_crosscheck")).get("ok") is True
        ),
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
    flagged = artifact.get("flagged_for_v447")
    flagged = flagged if isinstance(flagged, list) else []
    methods = artifact.get("methods_mapped")
    methods = methods if isinstance(methods, list) else []
    return {
        "source": "SOTA",
        "experiment_id": _experiment_id("SOTA", artifact),
        "decision": "object_world_model_handoff" if methods else "sota_handoff_empty",
        "flagged_for_v447_candidates": [
            row.get("candidate") for row in flagged if isinstance(row, Mapping)
        ],
        "methods_mapped_count": len(methods),
        "arxiv_ids_cited": artifact.get("arxiv_ids_cited")
        if isinstance(artifact.get("arxiv_ids_cited"), list)
        else [],
        "a1_perception_layer_input": dict(_mapping(artifact.get("a1_perception_layer_input"))),
    }


def _scored_lever_state(
    levelup: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
) -> JsonDict:
    return {
        "level_up_banked": levelup.get("level_up_banked") is True,
        "heldout_first_win_rate": heldout.get("heldout_first_win_rate"),
        "submission_package_ready": package.get("submission_package_ready") is True,
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
    """Build the complete .446 capstone after live summaries have run."""

    a1 = _a1_perception_verdict(
        artifacts.get("A1"),
        summarizer_results.get("A1"),
        artifacts.get("B1_AUDIT"),
    )
    honest_verdict = {
        "object_identity_recovered_goal_grounding_feasible": (
            "success_a1_object_identity_recovered_goal_grounding_feasible_capstone_ready"
        ),
        "genuine_null_object_identity_unrecoverable_from_rendered_grid": (
            "complete_a1_object_identity_genuine_null_rendered_grid_unrecoverable_capstone_ready"
        ),
        "synthetic_only_or_degenerate_non_test": (
            "complete_a1_object_identity_non_test_synthetic_or_degenerate_capstone_ready"
        ),
    }.get(str(a1.get("verdict")), "complete_a1_object_identity_unsettled_capstone_ready")
    levelup = _levelup_bank(artifacts.get("LEVELUP"))
    heldout = _heldout_readiness(artifacts.get("HELDOUT"))
    package = _submission_package_state(artifacts.get("PACKAGE"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "a1_perception_probe_verdict": a1,
        "scored_lever_state": _scored_lever_state(levelup, heldout, package),
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": _cited_artifacts(
            artifacts, artifact_sha256, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": levelup,
        "self_play_checkpoint": _self_play_checkpoint(artifacts.get("SELF_PLAY")),
        "heldout_readiness": heldout,
        "submission_package_state": package,
        "hardware_continuity": _hardware_continuity(artifacts.get("HARDWARE")),
        "sota_handoff": _sota_handoff(artifacts.get("SOTA")),
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
        "a1_perception_probe_verdict": {},
        "scored_lever_state": {},
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": {},
        "self_play_checkpoint": {},
        "heldout_readiness": {},
        "submission_package_state": {},
        "hardware_continuity": {},
        "sota_handoff": {},
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
    a1 = payload.get("a1_perception_probe_verdict")
    if isinstance(a1, Mapping) and a1 and a1.get("verdict") not in A1_VERDICTS:
        errors.append("invalid_a1_perception_probe_verdict")
    scored = payload.get("scored_lever_state")
    if isinstance(scored, Mapping) and scored:
        rate = scored.get("heldout_first_win_rate")
        if (
            not isinstance(scored.get("level_up_banked"), bool)
            or not isinstance(scored.get("submission_package_ready"), bool)
            or isinstance(rate, bool)
            or not isinstance(rate, int | float)
        ):
            errors.append("invalid_scored_lever_state")
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
        return "spec_missing_req_4849"
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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4849" in spec_path.read_text(
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
        "spec_has_req_4849": spec_has_req,
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

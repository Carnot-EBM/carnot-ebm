"""Experiment 4829: .444 S3 generation-lift capstone scorecard.

Spec refs: REQ-CAPSTONE-4829, SCENARIO-CAPSTONE-4829,
SCENARIO-CAPSTONE-4829-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4829-FIELD-PRINCIPLES.

The scorecard reads every landed .444 upstream artifact through
``scripts/summarize_artifact.py`` before importing fields. The headline is the
S3 generation-lift verdict: whether structural energy adds a winner to the
candidate pool that the matched bare explorer never proposed.
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
EXPERIMENT = "experiment_4829_capstone_v444"
EXPERIMENT_ID = 4829
SCHEMA = "carnot.exp4829.capstone_v444.v1"
RESULT_RELATIVE_PATH = "results/experiment_4829_capstone_v444.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 4829
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4829",
    "SCENARIO-CAPSTONE-4829",
    "SCENARIO-CAPSTONE-4829-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4829-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; capstone_ready=true is success_/complete_."},
    "s3_structural_energy_verdict": {
        "principle": (
            "the headline -- generation WIN (S4 authorized) / BOUNDED "
            "no-generation-lift / inconclusive-no-headroom; honors the "
            "matched-control + NEW-not-re-ranking checks."
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
    "s3_structural_energy_verdict",
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

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "S3": UpstreamSource(
        4821, "results/experiment_4821_structural_energy_s3_generation_lift.json"
    ),
    "LEVELUP": UpstreamSource(4822, "results/experiment_4822_levelup_attempt.json"),
    "SELF_PLAY": UpstreamSource(4823, "results/experiment_4823_self_play_verifier_checkpoint.json"),
    "HELDOUT": UpstreamSource(4824, "results/experiment_4824_heldout_first_win_readiness.json"),
    "BUG_AUDIT": UpstreamSource(4825, "results/experiment_4825_silent_bug_audit.json"),
    "PACKAGE": UpstreamSource(4826, "results/experiment_4826_submission_package_harden.json"),
    "HARDWARE": UpstreamSource(4827, "results/experiment_4827_kv260_continuity.json"),
    "SOTA": UpstreamSource(
        4828, "results/experiment_4828_sota_ingestion_cross_family_transfer.json"
    ),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "S3": (
        "honest_verdict",
        "verifier_is_oracle",
        "live_path_reachable",
        "lambda0_control",
        "lambda_guidance",
        "n_headroom_games",
        "min_headroom_games",
        "positive_control_passed",
        "new_levels_not_in_bare_pool",
        "winners_newly_entering_pool_delta",
        "winners_newly_entering_pool_delta_ci95",
        "game_results",
        "source_artifacts",
        "solve_provenance",
        "inference_substrate",
    ),
    "LEVELUP": (
        "honest_verdict",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "registry_update",
        "attempted_games",
        "dead_ends",
        "solve_provenance",
        "verifier_is_oracle",
    ),
    "SELF_PLAY": (
        "honest_verdict",
        "verifier_checkpoint_refreshed",
        "target_game",
        "self_play_residual",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
        "solve_provenance",
        "inference_substrate",
        "verifier_is_oracle",
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
        "inference_substrate",
        "verifier_is_oracle",
    ),
    "BUG_AUDIT": (
        "honest_verdict",
        "nulls_audited",
        "trusted_nulls",
        "silent_bugs_found",
        "s3_controls_verified",
        "s3_control_check",
        "s3_guidance_exercised",
        "verifier_is_oracle",
    ),
    "PACKAGE": (
        "honest_verdict",
        "submission_package_ready",
        "submitted_to_leaderboard",
        "operator_only",
        "vram_estimate_gb",
        "package_builds",
        "inference_substrate",
        "verifier_is_oracle",
    ),
    "HARDWARE": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "board_state",
        "next_forward_step",
        "inference_substrate",
        "verifier_is_oracle",
    ),
    "SOTA": (
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v445",
        "s4_context",
        "arxiv_ids_cited",
        "inference_substrate",
        "verifier_is_oracle",
    ),
}

Summarizer = Callable[[Path, str], SummarizerResult]


def _experiment_id(source: str, artifact: Mapping[str, Any] | None = None) -> int:
    if artifact:
        value = artifact.get("experiment_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return UPSTREAM_SOURCES[source].experiment_id


def _oracle_declared(artifact: Mapping[str, Any] | None) -> bool:
    return bool(artifact and artifact.get("verifier_is_oracle") is True)


def _summary_text(summary: SummarizerResult | None) -> str:
    return "" if summary is None else f"{summary.stdout}\n{summary.stderr}"


def _live_critical(summary: SummarizerResult | None) -> bool:
    return bool(summary and summary.exit_code >= 2)


def _ci_lower_positive(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and _float(value[0]) is not None
        and float(value[0]) > 0.0
    )


def _ci_includes_zero(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and _float(value[0]) is not None
        and _float(value[1]) is not None
        and float(value[0]) <= 0.0 <= float(value[1])
    )


def _banked_re_ranking_games(artifact: Mapping[str, Any]) -> list[str]:
    rows = artifact.get("game_results")
    if not isinstance(rows, list):
        return []
    games: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        banked_by_energy = row.get("banked_by_E") is True or row.get("banked-by-E") is True
        already_in_bare = (
            row.get("was_already_in_bare_pool") is True
            or row.get("was-already-in-bare-pool") is True
        )
        if banked_by_energy and already_in_bare:
            games.append(str(row.get("game", "unknown")))
    return games


def _s3_control_snapshot(audit: Mapping[str, Any] | None) -> JsonDict:
    check = _mapping(audit.get("s3_control_check")) if audit else {}
    matched = check.get("matched_lambda0_control") is True
    not_reranking = check.get("new_levels_not_in_bare_pool") is True
    reachable = check.get("positive_control_passed") is True
    guidance = audit is not None and (
        audit.get("s3_guidance_exercised") is True or check.get("s3_guidance_exercised") is True
    )
    controls = (
        audit is not None
        and audit.get("s3_controls_verified") is True
        and check.get("s3_controls_verified") is not False
        and matched
        and not_reranking
        and reachable
        and guidance
    )
    return {
        "controls_verified_by_b1": controls,
        "matched_lambda0_control_b1": matched,
        "new_levels_not_re_ranking_b1": not_reranking,
        "reachable_winner_positive_control_b1": reachable,
        "s3_guidance_exercised_b1": guidance,
    }


def _s3_verdict(
    artifact: Mapping[str, Any] | None,
    summary: SummarizerResult | None,
    audit: Mapping[str, Any] | None,
) -> JsonDict:
    if artifact is None:
        return {}

    delta = _float(artifact.get("winners_newly_entering_pool_delta"))
    ci95 = artifact.get("winners_newly_entering_pool_delta_ci95")
    ci95 = ci95 if isinstance(ci95, list) else []
    lambda0 = _mapping(artifact.get("lambda0_control"))
    lambda0_value = _float(lambda0.get("lambda"))
    matched_lambda0 = lambda0.get("matched_control") is True and lambda0_value == 0.0
    b1 = _s3_control_snapshot(audit)
    reranking_games = _banked_re_ranking_games(artifact)
    new_levels_field = artifact.get("new_levels_not_in_bare_pool")
    artifact_declares_not_reranking = new_levels_field is not False
    new_levels_not_re_ranking = (
        b1["new_levels_not_re_ranking_b1"] and artifact_declares_not_reranking and not reranking_games
    )
    positive_control = artifact.get("positive_control_passed") is True
    reachable_winner_positive_control = (
        positive_control and b1["reachable_winner_positive_control_b1"]
    )
    live_path = artifact.get("live_path_reachable") is True
    oracle = _oracle_declared(artifact)
    headroom = _int(artifact.get("n_headroom_games"))
    min_headroom = _int(artifact.get("min_headroom_games"), 1)
    headroom_floor_met = headroom >= min_headroom and min_headroom > 0
    live_critical = _live_critical(summary)
    controls_verified = (
        b1["controls_verified_by_b1"]
        and matched_lambda0
        and new_levels_not_re_ranking
        and reachable_winner_positive_control
        and b1["s3_guidance_exercised_b1"]
    )
    base: JsonDict = {
        "source": "S3",
        "experiment_id": _experiment_id("S3", artifact),
        "upstream_honest_verdict": artifact.get("honest_verdict", ""),
        "verifier_is_oracle": oracle,
        "live_path_reachable": live_path,
        "live_recheck_exit_code": summary.exit_code if summary else None,
        "live_recheck_excerpt": _summary_text(summary)[:240],
        "lambda0_control": dict(lambda0),
        "matched_lambda0_control": matched_lambda0,
        "controls_verified_by_b1": controls_verified,
        "b1_control_snapshot": b1,
        "positive_control_passed": positive_control,
        "reachable_winner_positive_control": reachable_winner_positive_control,
        "guidance_exercised": b1["s3_guidance_exercised_b1"],
        "n_headroom_games": headroom,
        "min_headroom_games": min_headroom,
        "headroom_floor_met": headroom_floor_met,
        "new_levels_not_re_ranking": new_levels_not_re_ranking,
        "new_levels_not_in_bare_pool": new_levels_field if isinstance(new_levels_field, list) else [],
        "banked_levels_already_in_bare_pool": reranking_games,
        "winners_newly_entering_pool_delta": delta,
        "winners_newly_entering_pool_delta_ci95": ci95,
        "ci_excludes_zero": _ci_lower_positive(ci95),
        "ci_includes_zero": _ci_includes_zero(ci95),
    }

    reason = ""
    if live_critical:
        reason = "live_critical_recheck"
    elif oracle:
        reason = "oracle_not_moat"
    elif not live_path:
        reason = "live_path_unreachable"
    elif reranking_games:
        reason = "banked_level_already_in_bare_pool"
    elif not controls_verified:
        reason = "s3_controls_unverified"
    elif not headroom_floor_met:
        reason = "inconclusive_no_generation_headroom"

    if reason:
        return {
            **base,
            "verdict": "inconclusive_no_headroom",
            "s4_authorized": False,
            "generation_win": False,
            "bounded_no_generation_lift": False,
            "metrics_imported": False,
            "reason": reason,
        }

    live_clean = bool(summary and summary.exit_code == 0)
    win = bool(live_clean and delta is not None and delta > 0.0 and _ci_lower_positive(ci95))
    bounded = bool(live_clean and _ci_includes_zero(ci95))
    if win:
        verdict = "generation_win_s4_authorized"
        reason = "winners_newly_entering_pool_ci_excludes_zero"
    elif bounded:
        verdict = "bounded_no_generation_lift"
        reason = "no_generation_lift_ci_includes_zero"
    else:
        verdict = "inconclusive_no_headroom"
        reason = "s3_generation_lift_gate_requirements_not_met"
    return {
        **base,
        "verdict": verdict,
        "s4_authorized": verdict == "generation_win_s4_authorized",
        "generation_win": verdict == "generation_win_s4_authorized",
        "bounded_no_generation_lift": verdict == "bounded_no_generation_lift",
        "metrics_imported": verdict
        in {"generation_win_s4_authorized", "bounded_no_generation_lift"},
        "reason": reason,
        "direction_after_s3": (
            "s4_authorized"
            if verdict == "generation_win_s4_authorized"
            else "bounded_at_real_offline_discriminator_no_live_value"
            if verdict == "bounded_no_generation_lift"
            else "inconclusive"
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
    artifacts: Mapping[str, Mapping[str, Any]],
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
    oracle = _oracle_declared(artifact)
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
        "verifier_is_oracle": oracle,
        "moat_claim": bool(after > before and not oracle),
    }


def _self_play_checkpoint(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    refreshed = artifact.get("verifier_checkpoint_refreshed") is True
    return {
        "source": "SELF_PLAY",
        "experiment_id": _experiment_id("SELF_PLAY", artifact),
        "decision": "checkpoint_refreshed" if refreshed else "checkpoint_not_refreshed",
        "verifier_checkpoint_refreshed": refreshed,
        "target_game": artifact.get("target_game"),
        "self_play_residual": artifact.get("self_play_residual"),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "solve_provenance": artifact.get("solve_provenance", ""),
        "inference_substrate": artifact.get("inference_substrate", ""),
    }


def _heldout_readiness(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    baseline_delta = _float(artifact.get("heldout_first_win_delta_vs_baseline"))
    prior_delta = _float(artifact.get("heldout_first_win_delta_vs_prior_best"))
    flat = baseline_delta == 0.0 and prior_delta == 0.0
    return {
        "source": "HELDOUT",
        "experiment_id": _experiment_id("HELDOUT", artifact),
        "decision": "flat_null_no_readiness_gain" if flat else "heldout_readiness_changed",
        "heldout_first_win_rate": _float(artifact.get("heldout_first_win_rate")),
        "first_win_baseline": _float(artifact.get("first_win_baseline")),
        "prior_best_heldout_first_win_rate": _float(
            artifact.get("prior_best_heldout_first_win_rate")
        ),
        "heldout_first_win_delta_vs_baseline": baseline_delta,
        "heldout_first_win_delta_vs_prior_best": prior_delta,
        "heldout_variant_attempts": _int(artifact.get("heldout_variant_attempts")),
        "parity_test_green": artifact.get("parity_test_green") is True,
        "positive_control_passed": artifact.get("positive_control_passed") is True,
        "null_delta_methodology_note_present": bool(artifact.get("null_delta_methodology_note")),
        "inference_substrate": artifact.get("inference_substrate", ""),
    }


def _silent_bug_audit(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    silent_bugs = artifact.get("silent_bugs_found")
    silent_bugs = silent_bugs if isinstance(silent_bugs, list) else []
    trusted = artifact.get("trusted_nulls")
    return {
        "source": "BUG_AUDIT",
        "experiment_id": _experiment_id("BUG_AUDIT", artifact),
        "nulls_audited": _int(artifact.get("nulls_audited")),
        "trusted_nulls": trusted if isinstance(trusted, list) else [],
        "silent_bugs_found_count": len(silent_bugs),
        "s3_controls_verified": artifact.get("s3_controls_verified") is True,
        "s3_guidance_exercised": artifact.get("s3_guidance_exercised") is True,
        "s3_control_check": dict(_mapping(artifact.get("s3_control_check"))),
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
    flagged = artifact.get("flagged_for_v445")
    flagged = flagged if isinstance(flagged, list) else []
    methods = artifact.get("methods_mapped")
    methods = methods if isinstance(methods, list) else []
    return {
        "source": "SOTA",
        "experiment_id": _experiment_id("SOTA", artifact),
        "decision": "sota_handoff_mapped" if methods else "sota_handoff_empty",
        "flagged_for_v445_candidates": [
            row.get("candidate") for row in flagged if isinstance(row, Mapping)
        ],
        "methods_mapped_count": len(methods),
        "s4_context": dict(_mapping(artifact.get("s4_context"))),
        "arxiv_ids_cited": artifact.get("arxiv_ids_cited")
        if isinstance(artifact.get("arxiv_ids_cited"), list)
        else [],
    }


def _readiness(
    s3_verdict: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
) -> JsonDict:
    s4_authorized = s3_verdict.get("verdict") == "generation_win_s4_authorized"
    bounded = s3_verdict.get("verdict") == "bounded_no_generation_lift"
    heldout_changed = heldout.get("decision") == "heldout_readiness_changed"
    package_ready = package.get("decision") == "package_ready_operator_only"
    return {
        "s3_verdict": s3_verdict.get("verdict", ""),
        "s4_authorized": s4_authorized,
        "heldout_decision": heldout.get("decision", ""),
        "submission_package_decision": package.get("decision", ""),
        "ready_for_operator_submit": bool(s4_authorized and heldout_changed and package_ready),
        "structural_energy_direction": (
            "s4_authorized_generation_lift"
            if s4_authorized
            else "bounded_at_real_offline_discriminator_no_live_value"
            if bounded
            else "inconclusive_no_generation_headroom"
        ),
        "reason": "requires_s3_generation_win_clean_package_and_heldout_gain",
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
    """Build the complete .444 capstone after live summaries have run."""

    s3 = _s3_verdict(
        artifacts.get("S3"),
        summarizer_results.get("S3"),
        artifacts.get("BUG_AUDIT"),
    )
    honest_verdict = {
        "generation_win_s4_authorized": "success_s3_generation_lift_s4_authorized",
        "bounded_no_generation_lift": "complete_s3_bounded_no_generation_lift_capstone_ready",
        "inconclusive_no_headroom": "complete_s3_inconclusive_no_generation_headroom_capstone_ready",
    }.get(str(s3.get("verdict")), "complete_s3_inconclusive_capstone_ready")
    heldout = _heldout_readiness(artifacts.get("HELDOUT"))
    package = _submission_package_state(artifacts.get("PACKAGE"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "s3_structural_energy_verdict": s3,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": _cited_artifacts(
            artifacts, artifact_sha256, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": _levelup_bank(artifacts.get("LEVELUP")),
        "self_play_checkpoint": _self_play_checkpoint(artifacts.get("SELF_PLAY")),
        "readiness": _readiness(s3, heldout, package),
        "heldout_readiness": heldout,
        "silent_bug_audit": _silent_bug_audit(artifacts.get("BUG_AUDIT")),
        "submission_package_state": package,
        "hardware_continuity": _hardware_continuity(artifacts.get("HARDWARE")),
        "sota_handoff": _sota_handoff(artifacts.get("SOTA")),
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
        "s3_structural_energy_verdict": {},
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
    s3 = payload.get("s3_structural_energy_verdict")
    if isinstance(s3, Mapping) and s3:
        if s3.get("verdict") not in {
            "generation_win_s4_authorized",
            "bounded_no_generation_lift",
            "inconclusive_no_headroom",
        }:
            errors.append("invalid_s3_verdict")
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
        return "spec_missing_req_4829"
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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4829" in spec_path.read_text(
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
        "spec_has_req_4829": spec_has_req,
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

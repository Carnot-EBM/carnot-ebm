"""Experiment 4880: .449 generation-wall fork capstone scorecard.

Spec refs: REQ-CAPSTONE-4880, SCENARIO-CAPSTONE-4880,
SCENARIO-CAPSTONE-4880-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4880-FIELD-PRINCIPLES.

The scorecard reads each .449 upstream artifact through
``scripts/summarize_artifact.py`` before importing fields. A1 is a headline
only when B1 confirms it ran live on GPU0, the planner was blind, the positive
control migrated, and the fork arithmetic matched. A1b is reported as an
inducer swing, with gate-skips kept distinct from zero-delta runs.
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


EXPERIMENT = "experiment_4880_capstone_v449"
EXPERIMENT_ID = 4880
SCHEMA = "carnot.exp4880.capstone_v449.v1"
RESULT_RELATIVE_PATH = "results/experiment_4880_capstone_v449.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 20260627
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
FORK_VERDICTS = ("GUIDANCE_WALL", "PLANNER_GAP", "INDUCER_CEILING")

SPEC_REFS = [
    "REQ-CAPSTONE-4880",
    "SCENARIO-CAPSTONE-4880",
    "SCENARIO-CAPSTONE-4880-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4880-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; capstone_ready=true is "
            "complete_a1_generation_wall_<fork>_capstone_ready."
        )
    },
    "a1_generation_wall_fork_verdict": {
        "principle": (
            "the headline -- {fork_verdict in "
            "GUIDANCE_WALL|PLANNER_GAP|INDUCER_CEILING, b1_trusted "
            "(ran-live-on-gpu0 + planner-blind + positive-control-migrated + "
            "numbers-match), per_game_fork}; the redirection for .450."
        )
    },
    "a1b_inducer_swing": {
        "principle": (
            "the first inducer swing -- {cegis_heldout_accuracy_delta_median, "
            "delta_ci95, a1b_delta_trustworthy, ran (false if gate-skipped)}; "
            "did CEGIS move the ~0.12 wall?"
        )
    },
    "scored_lever_state": {
        "principle": (
            "the deadline track -- {level_up_banked, heldout_first_win_rate, "
            "live_agent_ran, submission_package_ready}; the realistic 6/30 signal."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the monotonic ARC progress metric carried from the registry "
            "(66 + any A2/A3 bank)."
        )
    },
    "cited_upstream_artifacts": {
        "principle": "list of {experiment_id, fields_imported, sha256} -- the audit trail."
    },
    "flagged_artifacts_skipped": {
        "principle": "any upstream with flagged_adversarial=true that was excluded from aggregation."
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
    "a1_generation_wall_fork_verdict",
    "a1b_inducer_swing",
    "scored_lever_state",
    "reproducible_total_levels",
    "cited_upstream_artifacts",
    "flagged_artifacts_skipped",
    "inference_substrate",
    "levelup_bank",
    "self_play_checkpoint",
    "heldout_readiness",
    "submission_package_state",
    "hardware_continuity",
    "sota_handoff",
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
    "not_ready_",
)

A1_VERDICTS = {
    "generation_wall_guidance_wall",
    "generation_wall_planner_gap",
    "generation_wall_inducer_ceiling",
    "non_test_b1_untrusted",
    "non_test_invalid_fork",
}

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "A1": UpstreamSource(4871, "results/experiment_4871_generation_wall_fork_probe_gpu_fixed.json"),
    "A1B": UpstreamSource(4872, "results/experiment_4872_cegis_world_model_refinement.json"),
    "LEVELUP": UpstreamSource(4873, "results/experiment_4873_levelup_attempt.json"),
    "SELF_PLAY": UpstreamSource(4874, "results/experiment_4874_self_play_verifier_checkpoint.json"),
    "HELDOUT": UpstreamSource(4875, "results/experiment_4875_heldout_first_win_readiness.json"),
    "B1_AUDIT": UpstreamSource(4876, "results/experiment_4876_fork_probe_inducer_audit.json"),
    "PACKAGE": UpstreamSource(4877, "results/experiment_4877_submission_package_harden.json"),
    "HARDWARE": UpstreamSource(4878, "results/experiment_4878_kv260_continuity.json"),
    "SOTA": UpstreamSource(4879, "results/experiment_4879_sota_ingestion_v450_frontier.json"),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "A1": (
        "honest_verdict",
        "fork_verdict",
        "per_game_fork",
        "coverage_migration_count",
        "median_engine_heldout_accuracy",
        "positive_control_game",
        "positive_control_migrated",
        "planner_blind_to_banked_answer",
        "n_games_measured",
        "generator_backend",
        "live_path_reachable",
        "solve_provenance",
        "inference_substrate",
    ),
    "A1B": (
        "honest_verdict",
        "cegis_heldout_accuracy_delta_median",
        "cegis_heldout_accuracy_delta_ci95",
        "per_game_accuracy_delta",
        "positive_control_passed",
        "delta_on_truly_heldout_split",
        "live_path_reachable",
        "n_games_measured",
        "solve_provenance",
        "verifier_is_oracle",
        "inference_substrate",
    ),
    "LEVELUP": (
        "honest_verdict",
        "target_game",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "registry_update",
        "solve_provenance",
        "verifier_is_oracle",
        "inference_substrate",
    ),
    "SELF_PLAY": (
        "honest_verdict",
        "target_game",
        "verifier_checkpoint_refreshed",
        "checkpoint_path",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
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
        "heldout_first_win_ci",
        "positive_control_passed",
        "parity_test_green",
        "checkpoint_emitted",
        "live_agent_ran",
        "generator_backend",
        "solve_provenance",
        "inference_substrate",
    ),
    "B1_AUDIT": (
        "honest_verdict",
        "a1_source_fork_verdict",
        "a1_source_honest_verdict",
        "a1_source_n_games_measured",
        "a1_ran_live_on_gpu0",
        "a1_genuinely_diagnostic",
        "planner_blind_confirmed",
        "positive_control_confirmed",
        "numbers_match_fork",
        "a1_failure_reasons",
        "a1b_delta_median",
        "a1b_delta_ci95",
        "a1b_delta_trustworthy",
        "a1b_failure_reasons",
        "a1b_residual_reasons",
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
        "agent_config_resolution",
        "model_path_resolution",
        "packaging_requirements_crosscheck",
        "ready_package_regression_check",
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
        "aimed_at_fork_verdict",
        "flagged_for_v450",
        "methods_mapped",
        "arxiv_ids_cited",
        "sota_to_experiment_mapping_note",
        "upstream_artifacts",
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


def _trust_checks(b1: Mapping[str, Any] | None) -> dict[str, bool]:
    return {
        "a1_ran_live_on_gpu0": bool(b1 and b1.get("a1_ran_live_on_gpu0") is True),
        "planner_blind": bool(
            b1
            and (
                b1.get("planner_blind_confirmed") is True
                or b1.get("planner_blind") is True
            )
        ),
        "positive_control_migrated": bool(
            b1
            and (
                b1.get("positive_control_confirmed") is True
                or b1.get("positive_control_migrated") is True
            )
        ),
        "numbers_match_fork": bool(b1 and b1.get("numbers_match_fork") is True),
    }


def _raw_fork_claim(a1: Mapping[str, Any] | None, b1: Mapping[str, Any] | None) -> Any:
    if a1 and "fork_verdict" in a1:
        return a1.get("fork_verdict")
    if b1:
        return b1.get("a1_source_fork_verdict") or b1.get("source_fork_verdict")
    return None


def _computed_fork_from_b1(b1: Mapping[str, Any] | None) -> Any:
    checks = _mapping(b1.get("checks") if b1 else None)
    numbers = _mapping(checks.get("a1_numbers_match_fork"))
    return numbers.get("computed_fork_verdict")


def _a1_generation_wall_fork_verdict(
    a1: Mapping[str, Any] | None,
    b1: Mapping[str, Any] | None,
    a1_summary: SummarizerResult | None,
    b1_summary: SummarizerResult | None,
) -> JsonDict:
    if a1 is None and b1 is None:
        return {}
    raw_claim = _raw_fork_claim(a1, b1)
    trust_checks = _trust_checks(b1)
    diagnostic = bool(b1 and b1.get("a1_genuinely_diagnostic") is True)
    trust_failure_reasons = [
        "a1_genuinely_diagnostic",
        *[name for name, passed in trust_checks.items() if not passed],
    ]
    if diagnostic:
        trust_failure_reasons = [
            name for name in trust_failure_reasons if name != "a1_genuinely_diagnostic"
        ]
    b1_trusted = not trust_failure_reasons
    common: JsonDict = {
        "source": "A1" if a1 is not None else "B1_AUDIT",
        "a1_experiment_id": _experiment_id("A1", a1),
        "b1_experiment_id": _experiment_id("B1_AUDIT", b1),
        "upstream_honest_verdict": a1.get("honest_verdict", "")
        if a1
        else b1.get("a1_source_honest_verdict", "")
        if b1
        else "",
        "b1_honest_verdict": b1.get("honest_verdict", "") if b1 else "",
        "b1_trusted": b1_trusted,
        "trust_checks": trust_checks,
        "a1_genuinely_diagnostic": diagnostic,
        "trust_failure_reasons": trust_failure_reasons,
        "positive_control_game": a1.get("positive_control_game") if a1 else None,
        "positive_control_migrated": trust_checks["positive_control_migrated"],
        "a1_failure_reasons": b1.get("a1_failure_reasons", [])
        if b1 and isinstance(b1.get("a1_failure_reasons"), list)
        else [],
        "checks": dict(_mapping(b1.get("checks") if b1 else None)),
        "computed_fork_verdict": _computed_fork_from_b1(b1),
        "source_n_games_measured": _int(b1.get("a1_source_n_games_measured") if b1 else None),
        "live_recheck_exit_code": a1_summary.exit_code if a1_summary else None,
        "b1_live_recheck_exit_code": b1_summary.exit_code if b1_summary else None,
        "live_recheck_excerpt": _summary_text(a1_summary)[:240],
    }
    if not b1_trusted:
        return {
            **common,
            "verdict": "non_test_b1_untrusted",
            "fork_verdict": None,
            "untrusted_fork_verdict_claim": raw_claim,
            "per_game_fork": {},
            "coverage_migration_count": 0,
            "median_engine_heldout_accuracy": None,
            "next_450_pivot": "do_not_use_a1_non_test",
        }
    if raw_claim not in FORK_VERDICTS:
        return {
            **common,
            "verdict": "non_test_invalid_fork",
            "fork_verdict": None,
            "untrusted_fork_verdict_claim": raw_claim,
            "per_game_fork": {},
            "coverage_migration_count": 0,
            "median_engine_heldout_accuracy": None,
            "next_450_pivot": "do_not_use_a1_non_test",
        }
    return {
        **common,
        "verdict": f"generation_wall_{str(raw_claim).lower()}",
        "fork_verdict": raw_claim,
        "untrusted_fork_verdict_claim": None,
        "per_game_fork": dict(_mapping(a1.get("per_game_fork") if a1 else None)),
        "coverage_migration_count": _int(a1.get("coverage_migration_count") if a1 else None),
        "median_engine_heldout_accuracy": _float(
            a1.get("median_engine_heldout_accuracy") if a1 else None
        ),
        "planner_blind_to_banked_answer": bool(
            a1 and a1.get("planner_blind_to_banked_answer") is True
        ),
        "n_games_measured": _int(a1.get("n_games_measured") if a1 else None),
        "next_450_pivot": {
            "GUIDANCE_WALL": "guided_planner",
            "PLANNER_GAP": "stronger_planner",
            "INDUCER_CEILING": "stronger_world_model_inducer",
        }[str(raw_claim)],
    }


def _a1b_inducer_swing(
    a1b: Mapping[str, Any] | None,
    b1: Mapping[str, Any] | None,
) -> JsonDict:
    if a1b is None and b1 is None:
        return {}
    checks = _mapping(b1.get("checks") if b1 else None)
    delta_check = _mapping(checks.get("a1b_delta"))
    status = str(delta_check.get("status") or "")
    gate_skipped = status == "gate_skipped" or (
        a1b is not None and "gate_skipped" in str(a1b.get("honest_verdict", ""))
    )
    median = b1.get("a1b_delta_median") if b1 else None
    if median is None and a1b is not None:
        median = a1b.get("cegis_heldout_accuracy_delta_median")
    ci = b1.get("a1b_delta_ci95") if b1 else None
    if not isinstance(ci, list) and a1b is not None:
        ci = a1b.get("cegis_heldout_accuracy_delta_ci95")
    ran = not gate_skipped and (status == "ran" or median is not None)
    return {
        "source": "A1B" if a1b is not None else "B1_AUDIT",
        "experiment_id": _experiment_id("A1B", a1b),
        "b1_experiment_id": _experiment_id("B1_AUDIT", b1),
        "ran": ran,
        "gate_skipped": gate_skipped,
        "cegis_heldout_accuracy_delta_median": None if median is None else _float(median),
        "delta_ci95": ci if isinstance(ci, list) else [],
        "a1b_delta_trustworthy": bool(b1 and b1.get("a1b_delta_trustworthy") is True),
        "b1_failure_reasons": b1.get("a1b_failure_reasons", [])
        if b1 and isinstance(b1.get("a1b_failure_reasons"), list)
        else [],
        "residual_reasons": b1.get("a1b_residual_reasons", [])
        if b1 and isinstance(b1.get("a1b_residual_reasons"), list)
        else [],
        "status": status,
        "positive_control_passed": bool(a1b and a1b.get("positive_control_passed") is True),
        "delta_on_truly_heldout_split": bool(
            a1b and a1b.get("delta_on_truly_heldout_split") is True
        ),
        "n_games_measured": _int(a1b.get("n_games_measured") if a1b else None),
    }


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
                "experiment_id": _experiment_id(source, artifact),
                "fields_imported": _imported_fields(source, artifact),
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
        reason = _is_skipped(artifact, summarizer_results.get(source))
        if reason is None:
            continue
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "reason": reason,
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _levelup_bank(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    registry_update = _mapping(artifact.get("registry_update"))
    before = _int(
        registry_update.get("reproducible_total_levels_before"),
        _int(registry_update.get("prior_total_declared")),
    )
    after = _int(
        registry_update.get("reproducible_total_levels_after"),
        _int(registry_update.get("new_total_declared"), before),
    )
    new_levels = _int(artifact.get("new_levels_banked"))
    return {
        "source": "LEVELUP",
        "experiment_id": _experiment_id("LEVELUP", artifact),
        "target_game": artifact.get("target_game"),
        "new_levels_banked": new_levels,
        "level_up_banked": bool(
            new_levels > 0 and after > before and artifact.get("offline_reproduced") is True
        ),
        "reproducible_total_levels_before": before,
        "reproducible_total_levels_after": after,
        "reproducible_total_levels_delta": after - before,
        "registry_updated": registry_update.get("updated") is True,
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "solve_provenance": artifact.get("solve_provenance", ""),
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
        "inference_substrate": artifact.get("inference_substrate", ""),
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
        "checkpoint_path": artifact.get("checkpoint_path"),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "reproduction_gate": dict(_mapping(artifact.get("reproduction_gate"))),
        "search_state_count": _int(artifact.get("search_state_count")),
        "solve_provenance": artifact.get("solve_provenance", ""),
        "inference_substrate": artifact.get("inference_substrate", ""),
    }


def _heldout_readiness(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    rate = _float(artifact.get("heldout_first_win_rate"))
    baseline = _float(artifact.get("first_win_baseline"))
    if baseline is None:
        baseline = 0.04
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
        "heldout_first_win_ci": dict(_mapping(artifact.get("heldout_first_win_ci"))),
        "positive_control_passed": artifact.get("positive_control_passed") is True,
        "parity_test_green": artifact.get("parity_test_green") is True,
        "checkpoint_emitted": artifact.get("checkpoint_emitted") is True,
        "live_agent_ran": artifact.get("live_agent_ran") is True,
        "generator_backend": artifact.get("generator_backend"),
        "solve_provenance": artifact.get("solve_provenance", ""),
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
        "agent_config_resolved": _mapping(artifact.get("agent_config_resolution")).get("resolved")
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
    methods = artifact.get("methods_mapped")
    methods = methods if isinstance(methods, list) else []
    flagged = artifact.get("flagged_for_v450")
    flagged = flagged if isinstance(flagged, list) else []
    return {
        "source": "SOTA",
        "experiment_id": _experiment_id("SOTA", artifact),
        "decision": "v450_frontier_handoff" if methods else "sota_handoff_empty",
        "aimed_at_fork_verdict": artifact.get("aimed_at_fork_verdict"),
        "methods_mapped_count": len(methods),
        "flagged_for_v450_candidates": [
            row.get("candidate") for row in flagged if isinstance(row, Mapping)
        ],
        "arxiv_ids_cited": artifact.get("arxiv_ids_cited")
        if isinstance(artifact.get("arxiv_ids_cited"), list)
        else [],
        "sota_to_experiment_mapping_note": dict(
            _mapping(artifact.get("sota_to_experiment_mapping_note"))
        ),
        "upstream_artifacts": dict(_mapping(artifact.get("upstream_artifacts"))),
    }


def _scored_lever_state(
    levelup: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
) -> JsonDict:
    return {
        "level_up_banked": levelup.get("level_up_banked") is True,
        "heldout_first_win_rate": heldout.get("heldout_first_win_rate"),
        "live_agent_ran": heldout.get("live_agent_ran") is True,
        "submission_package_ready": package.get("submission_package_ready") is True,
    }


def _honest_verdict(a1: Mapping[str, Any]) -> str:
    fork = a1.get("fork_verdict")
    if fork in FORK_VERDICTS and a1.get("b1_trusted") is True:
        return f"complete_a1_generation_wall_{str(fork).lower()}_capstone_ready"
    return "complete_a1_generation_wall_non_test_capstone_ready"


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
    """Build the complete .449 scorecard after live summaries have run."""

    clean = _clean_artifacts(artifacts, summarizer_results)
    a1 = _a1_generation_wall_fork_verdict(
        clean.get("A1"),
        clean.get("B1_AUDIT"),
        summarizer_results.get("A1"),
        summarizer_results.get("B1_AUDIT"),
    )
    a1b = _a1b_inducer_swing(clean.get("A1B"), clean.get("B1_AUDIT"))
    levelup = _levelup_bank(clean.get("LEVELUP"))
    heldout = _heldout_readiness(clean.get("HELDOUT"))
    package = _submission_package_state(clean.get("PACKAGE"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(a1),
        "a1_generation_wall_fork_verdict": a1,
        "a1b_inducer_swing": a1b,
        "scored_lever_state": _scored_lever_state(levelup, heldout, package),
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": _cited_artifacts(clean, artifact_sha256),
        "flagged_artifacts_skipped": _flagged_artifacts_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": levelup,
        "self_play_checkpoint": _self_play_checkpoint(clean.get("SELF_PLAY")),
        "heldout_readiness": heldout,
        "submission_package_state": package,
        "hardware_continuity": _hardware_continuity(clean.get("HARDWARE")),
        "sota_handoff": _sota_handoff(clean.get("SOTA")),
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
        "a1_generation_wall_fork_verdict": {},
        "a1b_inducer_swing": {},
        "scored_lever_state": {},
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": [],
        "flagged_artifacts_skipped": _flagged_artifacts_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": {},
        "self_play_checkpoint": {},
        "heldout_readiness": {},
        "submission_package_state": {},
        "hardware_continuity": {},
        "sota_handoff": {},
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the .449 scorecard without mutating it."""

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
    a1 = payload.get("a1_generation_wall_fork_verdict")
    if isinstance(a1, Mapping) and a1 and a1.get("verdict") not in A1_VERDICTS:
        errors.append("invalid_a1_generation_wall_fork_verdict")
    a1b = payload.get("a1b_inducer_swing")
    if isinstance(a1b, Mapping) and a1b:
        median = a1b.get("cegis_heldout_accuracy_delta_median")
        if (
            not isinstance(a1b.get("ran"), bool)
            or not isinstance(a1b.get("gate_skipped"), bool)
            or not isinstance(a1b.get("a1b_delta_trustworthy"), bool)
            or (
                median is not None
                and (isinstance(median, bool) or not isinstance(median, int | float))
            )
            or not isinstance(a1b.get("delta_ci95"), list)
        ):
            errors.append("invalid_a1b_inducer_swing")
    scored = payload.get("scored_lever_state")
    if isinstance(scored, Mapping) and scored:
        rate = scored.get("heldout_first_win_rate")
        if (
            not isinstance(scored.get("level_up_banked"), bool)
            or not isinstance(scored.get("live_agent_ran"), bool)
            or not isinstance(scored.get("submission_package_ready"), bool)
            or (
                rate is not None
                and (isinstance(rate, bool) or not isinstance(rate, int | float))
            )
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
    flagged = payload.get("flagged_artifacts_skipped")
    if not isinstance(flagged, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not str(row.get("sha256", "")).startswith("sha256:")
        or not row.get("reason")
        for row in flagged
    ):
        errors.append("invalid_flagged_artifacts_skipped")
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
        return "spec_missing_req_4880"
    if any(info.get("present") is not True for info in upstream_preconditions.values()):
        return "upstreams_missing"
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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4880" in spec_path.read_text(
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
        "spec_has_req_4880": spec_has_req,
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

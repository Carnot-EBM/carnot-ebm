"""Experiment 4923: .453 capstone milestone scorecard.

Spec refs: REQ-CAPSTONE-4923, SCENARIO-CAPSTONE-4923,
SCENARIO-CAPSTONE-4923-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4923-FIELD-PRINCIPLES.

This experiment is an aggregation pass, not a new measurement. It reads every
.453 upstream artifact through ``scripts/summarize_artifact.py`` before using
any field, gates A1's causal-abstraction closure on the B1 audit, skips any
quarantined upstream, and writes the operator-facing scorecard for the 6/30
handoff.
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


EXPERIMENT = "experiment_4923_capstone_v453"
EXPERIMENT_ID = 4923
SCHEMA = "carnot.exp4923.capstone_v453.v1"
RESULT_RELATIVE_PATH = "results/experiment_4923_capstone_v453.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4923",
    "SCENARIO-CAPSTONE-4923",
    "SCENARIO-CAPSTONE-4923-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4923-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_capstone_v453_<closure-or-escalate> "
            "(e.g. _wall_is_hidden_state_arc_closure / "
            "_observable_variable_gap_<lever> / _diagnostic_inconclusive_escalate)."
        )
    },
    "headline": {
        "principle": (
            "the one-line .453 headline -- A1's trusted closure verdict "
            "(observable-gap lever / hidden-state closure) or the "
            "B1-untrusted/inconclusive fallback."
        )
    },
    "a1_closure_verdict_trusted": {
        "principle": (
            "A1's fork verdict gated on B1 a1_diagnostic_trustworthy -- the "
            "headline is only as good as the audit."
        )
    },
    "reproducible_total_levels": {
        "principle": "the monotonic ARC progress metric from the registry (68 + A2's bank if any)."
    },
    "heldout_first_win_rate": {
        "principle": "A4's CLEAN go/no-go number IF flag_resolved (else note it was still flagged)."
    },
    "submission_package_ready": {
        "principle": "B2's terminal gate -- the package state for the operator's 6/30 decision."
    },
    "post_sprint_pivot": {
        "principle": (
            "the post-6/30 verifier-moat handoff: D's distributional-energy-verifier "
            "scaffold + the deliverable (~0.05 agent + FoVer paper); NOT representation #5."
        )
    },
    "skipped_flagged_adversarial": {
        "principle": "the list of upstream artifacts skipped per the fabrication gate."
    },
    "cited_upstream_artifacts": {
        "principle": (
            "{experiment_id, fields_imported, sha256} per source -- the audit trail "
            "(no synthesized numbers)."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream JSON; 0.0001s floor)."
    },
    "preconditions_checked": {
        "principle": (
            "records upstream-artifact presence + registry loadability; a missing input emits blocked_."
        )
    },
    "random_seed": {
        "principle": "determinism for any tie-break in the scorecard aggregation."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (cited upstream sha256s + scorecard) so a replication catches drift."
        )
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "headline",
    "a1_closure_verdict_trusted",
    "reproducible_total_levels",
    "heldout_first_win_rate",
    "submission_package_ready",
    "post_sprint_pivot",
    "skipped_flagged_adversarial",
    "cited_upstream_artifacts",
    "inference_substrate",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "milestone_scorecard",
    "field_principles",
    "duration_s",
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
    "A1": UpstreamSource(4914, "results/experiment_4914_causal_abstraction_wall_diagnostic.json"),
    "B1_AUDIT": UpstreamSource(4918, "results/experiment_4918_causal_abstraction_audit.json"),
    "A2_LEVELUP": UpstreamSource(4915, "results/experiment_4915_levelup_attempt.json"),
    "A3_SELF_PLAY": UpstreamSource(
        4916, "results/experiment_4916_self_play_verifier_checkpoint.json"
    ),
    "A4_HELDOUT": UpstreamSource(4917, "results/experiment_4917_heldout_first_win_readiness.json"),
    "B2_PACKAGE": UpstreamSource(4919, "results/experiment_4919_submission_package_harden.json"),
    "B3_RETRO": UpstreamSource(4920, "results/experiment_4920_retro_timing_and_stamping_fix.json"),
    "C_KV260": UpstreamSource(4921, "results/experiment_4921_kv260_continuity.json"),
    "D_PIVOT": UpstreamSource(
        4922, "results/experiment_4922_distributional_energy_verifier_scaffold.json"
    ),
}

AUXILIARY_UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "A4_HELDOUT_PARTIAL": UpstreamSource(
        4917, "results/experiment_4917_heldout_first_win_readiness.partial.json"
    ),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "A1": (
        "honest_verdict",
        "fork_verdict",
        "minimal_abstraction_is_observable_subset",
        "is_decision_need_table_in_disguise",
        "positive_control_classifies_observable",
        "planner_blind_to_banked_answer",
        "verifier_is_oracle",
        "live_path_reachable",
        "n_games_measured",
        "per_game_causal_abstraction",
        "positive_control_games",
        "missing_observable_variable",
        "inference_substrate",
    ),
    "B1_AUDIT": (
        "honest_verdict",
        "a1_diagnostic_trustworthy",
        "a1_failure_reasons",
        "checks",
        "not_value_table_evidence",
        "numbers_match_fork_evidence",
        "observable_claims_spot_checked",
        "oracle_distinct_planner_blind_evidence",
        "positive_control_evidence",
        "transition_cross_checks",
        "inference_substrate",
    ),
    "A2_LEVELUP": (
        "honest_verdict",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "reproducible_total_levels_before",
        "reproducible_total_levels_after",
        "registry_update",
        "reproduction_gate",
        "target_game",
        "inference_substrate",
    ),
    "A3_SELF_PLAY": (
        "honest_verdict",
        "verifier_checkpoint_refreshed",
        "checkpoint_path",
        "checkpoint_mtime_delta_ns",
        "offline_reproduced",
        "reproduced_levels",
        "search_state_count",
        "target_game",
        "inference_substrate",
    ),
    "A4_HELDOUT": (
        "honest_verdict",
        "heldout_first_win_rate",
        "heldout_first_win_ci",
        "heldout_first_win_ci_lower",
        "first_win_baseline",
        "prior_best_heldout_first_win_rate",
        "heldout_first_win_delta_vs_baseline",
        "heldout_first_win_delta_vs_prior_best",
        "positive_control_passed",
        "parity_test_green",
        "live_agent_ran",
        "partial",
        "completed_games",
        "remaining_games",
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
        "package_build_check",
        "ready_package_regression_ok",
        "ready_package_regression_check",
        "inference_substrate",
    ),
    "B3_RETRO": (
        "honest_verdict",
        "research_conductor_modified",
        "stamping_audit",
        "stamping_audit_missing_duration",
        "mtime_fallback_window",
        "wiring_proposal_written",
        "inference_substrate",
    ),
    "C_KV260": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "loaded_overlay",
        "xmutil_requires_sudo",
        "command_probes",
        "inference_substrate",
    ),
    "D_PIVOT": (
        "honest_verdict",
        "pivot_executable_on_6_30",
        "no_headline_claim",
        "no_verifier_win_claimed",
        "comparison_stubbed",
        "self_consistency_saturated",
        "dry_run_three_columns",
        "harness_skeleton_path",
        "domain_slice_path",
        "validation_gate",
        "verifier_is_oracle",
        "inference_substrate",
    ),
}

Summarizer = Callable[[Path, str], SummarizerResult]


def _experiment_id(source: str, artifact: Mapping[str, Any] | None = None) -> int:
    """Return the upstream experiment id even when artifacts encode it in names."""

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
    checks = _mapping(b1.get("checks")) if b1 else {}
    return {
        "a1_diagnostic_trustworthy": bool(b1 and b1.get("a1_diagnostic_trustworthy") is True),
        "real_transitions": checks.get("real_transitions") is True,
        "not_value_table": checks.get("not_value_table") is True,
        "observable_claims_verified": checks.get("observable_claims_verified") is True,
        "positive_control_observable": checks.get("positive_control_observable") is True,
        "oracle_distinct_planner_blind": checks.get("oracle_distinct_planner_blind") is True,
        "numbers_match_fork": checks.get("numbers_match_fork") is True,
    }


def _trust_failure_reasons(gate: Mapping[str, Any]) -> list[str]:
    checks = (
        "a1_diagnostic_trustworthy",
        "real_transitions",
        "not_value_table",
        "observable_claims_verified",
        "positive_control_observable",
        "oracle_distinct_planner_blind",
        "numbers_match_fork",
    )
    return [field for field in checks if gate.get(field) is not True]


def _hidden_variables(a1: Mapping[str, Any] | None) -> list[str]:
    hidden: set[str] = set()
    for row in _mapping(a1.get("per_game_causal_abstraction") if a1 else {}).values():
        if not isinstance(row, Mapping):
            continue
        observable = _mapping(row.get("observable_from_interface"))
        for variable in row.get("required_variables", []):
            if isinstance(variable, str) and observable.get(variable) is False:
                hidden.add(variable)
    return sorted(hidden)


def _observable_gap_lever(a1: Mapping[str, Any] | None) -> str:
    if not a1:
        return "unknown_observable_variable"
    for field in ("missing_observable_variable", "observable_gap_lever", "fixable_observable_lever"):
        value = a1.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for row in _mapping(a1.get("per_game_causal_abstraction")).values():
        if not isinstance(row, Mapping) or row.get("classification") != "OBSERVABLE_GAP":
            continue
        observable = _mapping(row.get("observable_from_interface"))
        for variable in row.get("required_variables", []):
            if isinstance(variable, str) and observable.get(variable) is True:
                return variable
    return "unknown_observable_variable"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _a1_closure_verdict_trusted(
    a1: Mapping[str, Any] | None,
    b1: Mapping[str, Any] | None,
) -> JsonDict:
    if a1 is None or b1 is None:
        return {
            "source": "A1",
            "experiment_id": _experiment_id("A1", a1),
            "closure_verdict": "",
            "trusted": False,
            "trust_gate": {},
            "trust_failure_reasons": ["upstream_artifact_missing"],
        }
    gate = _trust_gate(b1)
    failures = _trust_failure_reasons(gate)
    fork = str(
        a1.get("fork_verdict")
        or _mapping(b1.get("numbers_match_fork_evidence")).get("declared_fork_verdict")
        or ""
    )
    return {
        "source": "A1",
        "experiment_id": _experiment_id("A1", a1),
        "honest_verdict": a1.get("honest_verdict", ""),
        "closure_verdict": fork,
        "trusted": not failures,
        "trust_gate": gate,
        "trust_failure_reasons": failures,
        "b1_experiment_id": _experiment_id("B1_AUDIT", b1),
        "b1_honest_verdict": b1.get("honest_verdict", ""),
        "b1_failure_reasons": b1.get("a1_failure_reasons", []),
        "minimal_abstraction_is_observable_subset": (
            a1.get("minimal_abstraction_is_observable_subset") is True
        ),
        "is_decision_need_table_in_disguise": a1.get("is_decision_need_table_in_disguise") is True,
        "positive_control_classifies_observable": (
            a1.get("positive_control_classifies_observable") is True
        ),
        "planner_blind_to_banked_answer": a1.get("planner_blind_to_banked_answer") is True,
        "verifier_is_oracle": a1.get("verifier_is_oracle") is True,
        "live_path_reachable": a1.get("live_path_reachable") is True,
        "n_games_measured": _int(a1.get("n_games_measured")),
        "hidden_variables_required": _hidden_variables(a1),
        "fixable_observable_lever": _observable_gap_lever(a1)
        if fork == "WALL_IS_OBSERVABLE_VARIABLE_GAP"
        else None,
    }


def _headline_decision(a1_trust: Mapping[str, Any]) -> tuple[str, str]:
    fork = str(a1_trust.get("closure_verdict", ""))
    if a1_trust.get("trusted") is not True:
        failures = ", ".join(a1_trust.get("trust_failure_reasons", [])) or "unknown_gate"
        return (
            "diagnostic_inconclusive_escalate",
            (
                "B1 did not trust A1's causal-abstraction diagnostic "
                f"({failures}); report .453 as inconclusive and escalate .454 "
                "to the operator instead of using A1's fork verdict."
            ),
        )
    if fork == "WALL_IS_OBSERVABLE_VARIABLE_GAP":
        lever = str(a1_trust.get("fixable_observable_lever") or "unknown_observable_variable")
        return (
            f"observable_variable_gap_{_slug(lever)}",
            (
                "A1/B1 close .453 as a fixable observable-variable gap: retain "
                f"{lever} in the live abstraction and carry it as the .454 "
                "post-sprint candidate."
            ),
        )
    if fork == "WALL_IS_HIDDEN_STATE":
        return (
            "wall_is_hidden_state_arc_closure",
            (
                "ARC CLOSURE: the live first-win wall is representation-invariant "
                "by construction; no representation over observable inputs recovers "
                "the discriminating variable. Deliverable locks to the current ~0.05 "
                "first-win agent (operator-only package) plus the publishable FoVer "
                "verifier-ensemble paper. Do not queue representation #5."
            ),
        )
    return (
        "diagnostic_inconclusive_escalate",
        (
            "A1's causal-abstraction diagnostic was degenerate or retired; the wall "
            "stays the trusted .452 negative and .454 escalates to the operator."
        ),
    )


def _honest_verdict(decision: str) -> str:
    return f"complete_capstone_v453_{decision}"


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
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "reason": reason,
                "sha256": artifact_sha256.get(source, ""),
                "summarizer_exit_code": summary.exit_code if summary else None,
                "true_honest_verdict": artifact.get("honest_verdict", ""),
                "true_live_recheck": _live_recheck(summary),
            }
        )
    return rows


def _a2_levelup_bank(artifact: Mapping[str, Any] | None, registry_total: int) -> JsonDict:
    if artifact is None:
        return {}
    registry_update = _mapping(artifact.get("registry_update"))
    before = _int(
        artifact.get("reproducible_total_levels_before"),
        _int(registry_update.get("prior_total_declared"), registry_total),
    )
    after = _int(
        artifact.get("reproducible_total_levels_after"),
        _int(registry_update.get("new_total_declared"), before),
    )
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
        "checkpoint_mtime_delta_ns": _int(artifact.get("checkpoint_mtime_delta_ns")),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "search_state_count": _int(artifact.get("search_state_count")),
    }


def _a4_flag_resolved(artifact: Mapping[str, Any] | None, summary: SummarizerResult | None) -> bool:
    return bool(artifact and artifact.get("flagged_adversarial") is not True and _live_recheck(summary) == "clean")


def _a4_heldout_go_no_go(
    artifacts: Mapping[str, Mapping[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
) -> JsonDict:
    artifact = artifacts.get("A4_HELDOUT")
    summary = summarizer_results.get("A4_HELDOUT")
    skip_reason = _is_skipped(artifact, summary)
    if skip_reason is not None:
        return {
            "source": "A4_HELDOUT",
            "experiment_id": _experiment_id("A4_HELDOUT", artifact),
            "status": f"skipped_{skip_reason}",
            "flag_resolved": False,
            "reason": skip_reason,
            "true_honest_verdict": artifact.get("honest_verdict", "") if artifact else "",
            "summarizer_exit_code": summary.exit_code if summary else None,
            "true_live_recheck": _live_recheck(summary),
        }
    artifact = clean.get("A4_HELDOUT")
    if artifact is None:
        return {}
    flag_resolved = _a4_flag_resolved(artifact, summary)
    return {
        "source": "A4_HELDOUT",
        "experiment_id": _experiment_id("A4_HELDOUT", artifact),
        "status": "included_clean" if flag_resolved else "included_but_flag_unresolved",
        "flag_resolved": flag_resolved,
        "honest_verdict": artifact.get("honest_verdict", ""),
        "heldout_first_win_rate": _float(artifact.get("heldout_first_win_rate"))
        if flag_resolved
        else None,
        "heldout_first_win_ci": artifact.get("heldout_first_win_ci", {}),
        "heldout_first_win_ci_lower": _float(artifact.get("heldout_first_win_ci_lower")),
        "first_win_baseline": _float(artifact.get("first_win_baseline")),
        "prior_best_heldout_first_win_rate": _float(
            artifact.get("prior_best_heldout_first_win_rate")
        ),
        "heldout_first_win_delta_vs_baseline": _float(
            artifact.get("heldout_first_win_delta_vs_baseline")
        ),
        "heldout_first_win_delta_vs_prior_best": _float(
            artifact.get("heldout_first_win_delta_vs_prior_best")
        ),
        "positive_control_passed": artifact.get("positive_control_passed") is True,
        "parity_test_green": artifact.get("parity_test_green") is True,
        "live_agent_ran": artifact.get("live_agent_ran") is True,
        "partial": artifact.get("partial") is True,
        "completed_game_count": len(artifact.get("completed_games", []))
        if isinstance(artifact.get("completed_games"), list)
        else 0,
        "remaining_game_count": len(artifact.get("remaining_games", []))
        if isinstance(artifact.get("remaining_games"), list)
        else 0,
    }


def _b2_submission_package(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    submitted = artifact.get("submits") is True or artifact.get("submitted_to_leaderboard") is True
    operator_only = artifact.get("operator_only") is True and not submitted
    package_build_check = _mapping(artifact.get("package_build_check"))
    package_builds = artifact.get("package_builds") is True or package_build_check.get(
        "package_builds"
    ) is True
    ready = artifact.get("submission_package_ready") is True
    return {
        "source": "B2_PACKAGE",
        "experiment_id": _experiment_id("B2_PACKAGE", artifact),
        "decision": "package_ready_operator_only"
        if ready and operator_only and package_builds
        else "package_not_ready",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "submission_package_ready": ready,
        "submits": submitted,
        "operator_only": operator_only,
        "peak_vram_gb": _float(artifact.get("peak_vram_gb")),
        "frozen_stack_loads": artifact.get("frozen_stack_loads") is True,
        "package_builds": package_builds,
        "dry_build_ran": package_build_check.get("dry_build_ran") is True,
        "ready_package_regression_ok": artifact.get("ready_package_regression_ok") is True
        or _mapping(artifact.get("ready_package_regression_check")).get("ok") is True,
    }


def _b3_retro_timing_stamping_fix(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    shipped = (
        str(artifact.get("honest_verdict", "")).startswith("success_")
        and artifact.get("research_conductor_modified") is False
        and artifact.get("wiring_proposal_written") is True
    )
    return {
        "source": "B3_RETRO",
        "experiment_id": _experiment_id("B3_RETRO", artifact),
        "decision": "retro_fix_shipped" if shipped else "retro_fix_incomplete",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "research_conductor_modified": artifact.get("research_conductor_modified") is True,
        "wiring_proposal_written": artifact.get("wiring_proposal_written") is True,
        "missing_duration_count": len(artifact.get("stamping_audit_missing_duration", []))
        if isinstance(artifact.get("stamping_audit_missing_duration"), list)
        else 0,
        "mtime_fallback_window": artifact.get("mtime_fallback_window", {}),
    }


def _c_kv260(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ok = str(artifact.get("honest_verdict", "")).startswith("success_") and artifact.get(
        "kv260_ssh_reachable"
    ) is True
    return {
        "source": "C_KV260",
        "experiment_id": _experiment_id("C_KV260", artifact),
        "decision": "kv260_continuity_ok" if ok else "kv260_continuity_blocked",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "kv260_ssh_reachable": artifact.get("kv260_ssh_reachable") is True,
        "loaded_overlay": artifact.get("loaded_overlay"),
        "xmutil_requires_sudo": artifact.get("xmutil_requires_sudo") is True,
        "inference_substrate": artifact.get("inference_substrate"),
    }


def _d_distributional_energy_verifier_pivot(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    columns = _mapping(artifact.get("dry_run_three_columns")).get("columns", [])
    executable = (
        artifact.get("pivot_executable_on_6_30") is True
        and artifact.get("no_headline_claim") is True
        and artifact.get("no_verifier_win_claimed") is True
        and artifact.get("verifier_is_oracle") is False
    )
    return {
        "source": "D_PIVOT",
        "experiment_id": _experiment_id("D_PIVOT", artifact),
        "decision": "pivot_scaffold_executable" if executable else "pivot_scaffold_not_ready",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "pivot_executable_on_6_30": artifact.get("pivot_executable_on_6_30") is True,
        "no_headline_claim": artifact.get("no_headline_claim") is True,
        "no_verifier_win_claimed": artifact.get("no_verifier_win_claimed") is True,
        "comparison_stubbed": artifact.get("comparison_stubbed") is True,
        "self_consistency_saturated": artifact.get("self_consistency_saturated") is True,
        "comparison_columns": columns if isinstance(columns, list) else [],
        "harness_skeleton_path": artifact.get("harness_skeleton_path"),
        "domain_slice_path": artifact.get("domain_slice_path"),
        "validation_gate": artifact.get("validation_gate", {}),
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
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
        "a1_causal_abstraction_closure": dict(a1_trust),
        "b1_causal_abstraction_audit": _b1_audit(clean.get("B1_AUDIT")),
        "a2_levelup_bank": _a2_levelup_bank(clean.get("A2_LEVELUP"), registry_total),
        "a3_self_play_checkpoint": _a3_self_play_checkpoint(clean.get("A3_SELF_PLAY")),
        "a4_heldout_go_no_go": _a4_heldout_go_no_go(artifacts, clean, summarizer_results),
        "b2_submission_package": _b2_submission_package(clean.get("B2_PACKAGE")),
        "b3_retro_timing_stamping_fix": _b3_retro_timing_stamping_fix(clean.get("B3_RETRO")),
        "c_kv260": _c_kv260(clean.get("C_KV260")),
        "d_distributional_energy_verifier_pivot": _d_distributional_energy_verifier_pivot(
            clean.get("D_PIVOT")
        ),
    }


def _b1_audit(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    checks = _mapping(artifact.get("checks"))
    return {
        "source": "B1_AUDIT",
        "experiment_id": _experiment_id("B1_AUDIT", artifact),
        "honest_verdict": artifact.get("honest_verdict", ""),
        "a1_diagnostic_trustworthy": artifact.get("a1_diagnostic_trustworthy") is True,
        "checks": dict(checks),
        "a1_failure_reasons": artifact.get("a1_failure_reasons", []),
        "observable_claims_checked_count": len(artifact.get("observable_claims_spot_checked", []))
        if isinstance(artifact.get("observable_claims_spot_checked"), list)
        else 0,
        "transition_cross_check_count": len(artifact.get("transition_cross_checks", []))
        if isinstance(artifact.get("transition_cross_checks"), list)
        else 0,
    }


def _post_sprint_pivot(
    decision: str,
    d_artifact: Mapping[str, Any] | None,
    a1_trust: Mapping[str, Any],
) -> JsonDict:
    d_lane = _d_distributional_energy_verifier_pivot(d_artifact)
    if decision.startswith("observable_variable_gap_"):
        return {
            "decision": "v454_observable_variable_candidate",
            "candidate_lever": a1_trust.get("fixable_observable_lever"),
            "deliverable": "retain the missing observable variable in the live abstraction",
            "distributional_energy_verifier_scaffold": d_lane,
            "do_not_queue": "",
        }
    if decision == "wall_is_hidden_state_arc_closure":
        return {
            "decision": "post_6_30_distributional_energy_verifier_pivot",
            "deliverable": (
                "current ~0.05 first-win agent (operator-only package) + publishable "
                "FoVer verifier-ensemble paper"
            ),
            "paper_ready": True,
            "north_star_sections": ["1", "2"],
            "do_not_queue": "representation_5",
            "distributional_energy_verifier_scaffold": d_lane,
        }
    return {
        "decision": "operator_escalation_for_v454",
        "reason": decision,
        "distributional_energy_verifier_scaffold": d_lane,
        "do_not_queue": "",
    }


def _heldout_first_win_rate(
    artifacts: Mapping[str, Mapping[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
) -> float | None:
    a4 = clean.get("A4_HELDOUT")
    if not _a4_flag_resolved(a4, summarizer_results.get("A4_HELDOUT")):
        return None
    return _float(a4.get("heldout_first_win_rate")) if a4 else None


def _submission_package_ready(clean: Mapping[str, Mapping[str, Any]]) -> bool:
    package = clean.get("B2_PACKAGE")
    return bool(package and package.get("submission_package_ready") is True)


def _checksum_payload(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": payload.get("honest_verdict"),
        "headline": payload.get("headline"),
        "cited_upstream_artifacts": payload.get("cited_upstream_artifacts"),
        "milestone_scorecard": payload.get("milestone_scorecard"),
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
    """Build the complete .453 scorecard after the upstream summaries run."""

    clean = _clean_artifacts(artifacts, summarizer_results)
    a1_trust = _a1_closure_verdict_trusted(clean.get("A1"), clean.get("B1_AUDIT"))
    decision, headline = _headline_decision(a1_trust)
    registry_total = _int(registry.get("reproducible_total_levels"))
    milestone_scorecard = _milestone_scorecard(
        artifacts=artifacts,
        clean=clean,
        summarizer_results=summarizer_results,
        a1_trust=a1_trust,
        registry_total=registry_total,
    )
    cited = _cited_artifacts(clean, artifact_sha256)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(decision),
        "headline": headline,
        "a1_closure_verdict_trusted": a1_trust,
        "reproducible_total_levels": registry_total,
        "heldout_first_win_rate": _heldout_first_win_rate(artifacts, clean, summarizer_results),
        "submission_package_ready": _submission_package_ready(clean),
        "post_sprint_pivot": _post_sprint_pivot(decision, clean.get("D_PIVOT"), a1_trust),
        "skipped_flagged_adversarial": _skipped_flagged_adversarial(
            artifacts, artifact_sha256, summarizer_results
        ),
        "cited_upstream_artifacts": cited,
        "inference_substrate": INFERENCE_SUBSTRATE,
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
        "random_seed": RANDOM_SEED,
        "milestone_scorecard": milestone_scorecard,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "capstone_ready": True,
    }
    payload["reproducibility_checksum"] = payload_checksum(_checksum_payload(payload))
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
        "headline": "",
        "a1_closure_verdict_trusted": {
            "source": "A1",
            "experiment_id": UPSTREAM_SOURCES["A1"].experiment_id,
            "closure_verdict": "",
            "trusted": False,
            "trust_gate": {},
            "trust_failure_reasons": [reason],
        },
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "heldout_first_win_rate": None,
        "submission_package_ready": False,
        "post_sprint_pivot": {},
        "skipped_flagged_adversarial": _skipped_flagged_adversarial(
            artifacts, artifact_sha256, summarizer_results
        ),
        "cited_upstream_artifacts": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "milestone_scorecard": {},
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "capstone_ready": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(_checksum_payload(payload))
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the .453 scorecard without mutating it."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    a1_trust = payload.get("a1_closure_verdict_trusted")
    if not isinstance(a1_trust, Mapping) or not isinstance(a1_trust.get("trusted"), bool):
        errors.append("invalid_a1_closure_verdict_trusted")
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
    if not isinstance(payload.get("preconditions_checked"), Mapping):
        errors.append("invalid_preconditions_checked")
    if not isinstance(payload.get("random_seed"), int) or isinstance(payload.get("random_seed"), bool):
        errors.append("invalid_random_seed")
    if not isinstance(payload.get("capstone_ready"), bool):
        errors.append("invalid_capstone_ready")
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
        return "spec_missing_req_4923"
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

    auxiliary_preconditions: dict[str, JsonDict] = {}
    for source, spec in AUXILIARY_UPSTREAM_SOURCES.items():
        path = root / spec.relative_path
        present = path.exists()
        auxiliary_preconditions[source] = {"path": spec.relative_path, "present": present}
        if present:
            auxiliary_preconditions[source]["sha256"] = file_sha256(path) or ""
            if summarizer_present:
                summary = summarizer(root, spec.relative_path)
                auxiliary_preconditions[source]["summarizer_exit_code"] = summary.exit_code

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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4923" in spec_path.read_text(
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
        "spec_has_req_4923": spec_has_req,
        "upstream_artifacts": upstream_preconditions,
        "auxiliary_upstream_artifacts": auxiliary_preconditions,
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

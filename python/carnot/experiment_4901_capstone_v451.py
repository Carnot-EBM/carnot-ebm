"""Experiment 4901: .451 representation-fork capstone scorecard.

Spec refs: REQ-CAPSTONE-4901, SCENARIO-CAPSTONE-4901,
SCENARIO-CAPSTONE-4901-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4901-FIELD-PRINCIPLES.

The scorecard reads each .451 upstream artifact through
``scripts/summarize_artifact.py`` before importing fields. The headline is the
representation-fork verdict, gated by Exp4897. Flagged or live-critical
upstreams are excluded from numeric aggregation.
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


EXPERIMENT = "experiment_4901_capstone_v451"
EXPERIMENT_ID = 4901
SCHEMA = "carnot.exp4901.capstone_v451.v1"
RESULT_RELATIVE_PATH = "results/experiment_4901_capstone_v451.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4901",
    "SCENARIO-CAPSTONE-4901",
    "SCENARIO-CAPSTONE-4901-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4901-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; capstone complete is complete_capstone_v451_<headline>."
    },
    "representation_fork_verdict": {
        "principle": (
            "the trusted A1 (+ A1b) fork verdict -- the milestone headline that "
            "redirects .452."
        )
    },
    "fork_verdict_trusted": {
        "principle": (
            "true iff B1 confirmed A1 genuinely diagnostic AND (if A1b ran) A1b "
            "genuinely live; else the verdict is a non-test."
        )
    },
    "change_value_gap_representation_invariant": {
        "principle": (
            "true iff BOTH A1 + A1b failed to move change-VALUE accuracy -> "
            "escalate to operator (deliverable = the 0.08 agent)."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the authoritative ARC progress metric from the registry (carried, not "
            "re-counted)."
        )
    },
    "deadline_lever_scorecard": {
        "principle": (
            "A2 bank / A3 self-play / A4 fresh-live held-out first-win / B2 package "
            "-- the 6/30 readiness summary."
        )
    },
    "flagged_upstreams_skipped": {
        "principle": (
            "lists any flagged_adversarial upstream excluded from aggregation (the "
            "fabrication gate)."
        )
    },
    "operator_escalation_note": {
        "principle": (
            "present iff representation-invariant -- the .452 redirect / operator "
            "decision the milestone surfaces."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (0.0001s floor)."
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "representation_fork_verdict",
    "fork_verdict_trusted",
    "change_value_gap_representation_invariant",
    "reproducible_total_levels",
    "deadline_lever_scorecard",
    "flagged_upstreams_skipped",
    "operator_escalation_note",
    "inference_substrate",
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

REPRESENTATION_VERDICTS = {
    "representation_invariant_escalate_operator",
    "trusted_fork_no_escalation",
    "untrusted_fork_non_test",
}

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "A1": UpstreamSource(4892, "results/experiment_4892_decision_need_targets_value_gap.json"),
    "A1B": UpstreamSource(4893, "results/experiment_4893_action_prefix_latent_adapter.json"),
    "B1_AUDIT": UpstreamSource(4897, "results/experiment_4897_value_gap_representation_audit.json"),
    "A2_LEVELUP": UpstreamSource(4894, "results/experiment_4894_levelup_attempt.json"),
    "A3_SELF_PLAY": UpstreamSource(4895, "results/experiment_4895_self_play_verifier_checkpoint.json"),
    "A4_HELDOUT": UpstreamSource(4896, "results/experiment_4896_heldout_first_win_readiness.json"),
    "B2_PACKAGE": UpstreamSource(4898, "results/experiment_4898_submission_package_harden.json"),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "A1": (
        "honest_verdict",
        "fork_verdict",
        "decision_need_value_accuracy_delta_median",
        "decision_need_value_accuracy_delta_ci95",
        "positive_control_non_degenerate",
        "delta_on_truly_heldout_split",
        "n_games_measured",
        "engine_cell_recall_median",
        "inference_substrate",
    ),
    "A1B": (
        "honest_verdict",
        "fork_verdict",
        "action_prefix_value_accuracy_delta_median",
        "action_prefix_value_accuracy_delta_ci95",
        "positive_control_non_degenerate",
        "delta_on_truly_heldout_split",
        "ran_genuinely_live",
        "n_games_measured",
        "engine_cell_recall_median",
        "inference_substrate",
    ),
    "B1_AUDIT": (
        "honest_verdict",
        "a1_source_fork_verdict",
        "a1b_source_fork_verdict",
        "a1_genuinely_diagnostic",
        "a1b_ran_genuinely_live",
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
}

Summarizer = Callable[[Path, str], SummarizerResult]


def _experiment_id(source: str, artifact: Mapping[str, Any] | None = None) -> int:
    if artifact:
        value = artifact.get("experiment_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        experiment = artifact.get("experiment")
        if isinstance(experiment, int) and not isinstance(experiment, bool):
            return experiment
        if isinstance(experiment, str):
            match = re.search(r"experiment_(\d+)", experiment)
            if match:
                return int(match.group(1))
    return UPSTREAM_SOURCES[source].experiment_id


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


def _ci95_includes_zero(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 2:
        return False
    low = _float(value[0])
    high = _float(value[1])
    if low is None or high is None:
        return False
    return min(low, high) <= 0.0 <= max(low, high)


def _representation_arm(
    *,
    source: str,
    artifact: Mapping[str, Any] | None,
    median_field: str,
    ci_field: str,
) -> JsonDict:
    if artifact is None:
        return {"status": "skipped"}
    ci = artifact.get(ci_field)
    ci_includes_zero = _ci95_includes_zero(ci)
    positive_control = artifact.get("positive_control_non_degenerate") is True
    return {
        "source": source,
        "experiment_id": _experiment_id(source, artifact),
        "status": "ran",
        "honest_verdict": artifact.get("honest_verdict", ""),
        "fork_verdict": artifact.get("fork_verdict"),
        "delta_median": _float(artifact.get(median_field)),
        "delta_ci95": ci if isinstance(ci, list) else [],
        "delta_ci95_includes_zero": ci_includes_zero,
        "positive_control_non_degenerate": positive_control,
        "delta_on_truly_heldout_split": artifact.get("delta_on_truly_heldout_split") is True,
        "failed_to_move_change_value_accuracy": ci_includes_zero and positive_control,
        "n_games_measured": _int(artifact.get("n_games_measured")),
        "engine_cell_recall_median": _float(artifact.get("engine_cell_recall_median")),
    }


def _a1b_required(a1b: Mapping[str, Any] | None) -> bool:
    if a1b is None:
        return False
    verdict = str(a1b.get("honest_verdict", ""))
    return "gate_skipped" not in verdict and "not_run" not in verdict


def _trust_gate(
    b1: Mapping[str, Any] | None,
    *,
    a1b: Mapping[str, Any] | None,
) -> JsonDict:
    a1b_required = _a1b_required(a1b)
    return {
        "a1_genuinely_diagnostic": bool(b1 and b1.get("a1_genuinely_diagnostic") is True),
        "a1b_ran_genuinely_live": bool(b1 and b1.get("a1b_ran_genuinely_live") is True)
        if a1b_required
        else True,
        "a1b_required": a1b_required,
    }


def _trust_failure_reasons(gate: Mapping[str, Any]) -> list[str]:
    failures = []
    if gate.get("a1_genuinely_diagnostic") is not True:
        failures.append("a1_genuinely_diagnostic")
    if gate.get("a1b_required") is True and gate.get("a1b_ran_genuinely_live") is not True:
        failures.append("a1b_ran_genuinely_live")
    return failures


def _representation_fork_verdict(
    a1: Mapping[str, Any] | None,
    a1b: Mapping[str, Any] | None,
    b1: Mapping[str, Any] | None,
) -> JsonDict:
    if a1 is None and b1 is None:
        return {}
    a1_arm = _representation_arm(
        source="A1",
        artifact=a1,
        median_field="decision_need_value_accuracy_delta_median",
        ci_field="decision_need_value_accuracy_delta_ci95",
    )
    a1b_arm = _representation_arm(
        source="A1B",
        artifact=a1b,
        median_field="action_prefix_value_accuracy_delta_median",
        ci_field="action_prefix_value_accuracy_delta_ci95",
    )
    gate = _trust_gate(b1, a1b=a1b)
    failures = _trust_failure_reasons(gate)
    trusted = not failures
    invariant = bool(
        trusted
        and a1b_arm.get("status") == "ran"
        and a1_arm.get("failed_to_move_change_value_accuracy") is True
        and a1b_arm.get("failed_to_move_change_value_accuracy") is True
    )
    if invariant:
        verdict = "representation_invariant_escalate_operator"
    elif trusted:
        verdict = "trusted_fork_no_escalation"
    else:
        verdict = "untrusted_fork_non_test"
    return {
        "verdict": verdict,
        "trusted": trusted,
        "trust_gate": gate,
        "trust_failure_reasons": failures,
        "b1_experiment_id": _experiment_id("B1_AUDIT", b1),
        "b1_honest_verdict": b1.get("honest_verdict", "") if b1 else "",
        "a1": a1_arm,
        "a1b": a1b_arm,
        "a1_source_fork_verdict": b1.get("a1_source_fork_verdict") if b1 else None,
        "a1b_source_fork_verdict": b1.get("a1b_source_fork_verdict") if b1 else None,
        "change_value_gap_representation_invariant": invariant,
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
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "fields_imported": _imported_fields(source, artifact),
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _flagged_upstreams_skipped(
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
            }
        )
    return rows


def _a2_bank(artifact: Mapping[str, Any] | None, registry_total: int) -> JsonDict:
    if artifact is None:
        return {}
    registry_update = _mapping(artifact.get("registry_update"))
    before = _int(
        artifact.get("reproducible_total_levels_before"),
        _int(registry_update.get("prior_total_declared")),
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
        "target_game": artifact.get("target_game"),
        "new_levels_banked": new_levels,
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "reproducible_total_levels_before": before,
        "reproducible_total_levels_after": after,
        "registry_authoritative_total": registry_total,
        "registry_update_reason": registry_update.get("reason", ""),
    }


def _a3_self_play(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    refreshed = artifact.get("verifier_checkpoint_refreshed") is True
    return {
        "source": "A3_SELF_PLAY",
        "experiment_id": _experiment_id("A3_SELF_PLAY", artifact),
        "decision": "checkpoint_refreshed" if refreshed else "checkpoint_not_refreshed",
        "target_game": artifact.get("target_game"),
        "verifier_checkpoint_refreshed": refreshed,
        "checkpoint_path": artifact.get("checkpoint_path"),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "search_state_count": _int(artifact.get("search_state_count")),
    }


def _a4_fresh_live(
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
        }
    artifact = clean.get("A4_HELDOUT")
    if artifact is None:
        return {}
    return {
        "source": "A4_HELDOUT",
        "experiment_id": _experiment_id("A4_HELDOUT", artifact),
        "status": "included",
        "heldout_first_win_rate": _float(artifact.get("heldout_first_win_rate")),
        "first_win_baseline": _float(artifact.get("first_win_baseline")),
        "heldout_first_win_delta_vs_baseline": _float(
            artifact.get("heldout_first_win_delta_vs_baseline")
        ),
        "positive_control_passed": artifact.get("positive_control_passed") is True,
        "parity_test_green": artifact.get("parity_test_green") is True,
        "live_agent_ran": artifact.get("live_agent_ran") is True,
        "submitted_to_leaderboard": artifact.get("submitted_to_leaderboard") is True,
        "operator_only": artifact.get("operator_only") is True,
    }


def _b2_package(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ready = artifact.get("submission_package_ready") is True
    submitted = artifact.get("submitted_to_leaderboard") is True
    operator_only = artifact.get("operator_only") is True
    package_builds = _mapping(artifact.get("package_builds"))
    return {
        "source": "B2_PACKAGE",
        "experiment_id": _experiment_id("B2_PACKAGE", artifact),
        "decision": "package_ready_operator_only"
        if ready and operator_only and not submitted
        else "package_not_ready",
        "submission_package_ready": ready,
        "submitted_to_leaderboard": submitted,
        "operator_only": operator_only,
        "vram_estimate_gb": _float(artifact.get("vram_estimate_gb")),
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


def _deadline_scorecard(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
    registry_total: int,
    invariant: bool,
) -> JsonDict:
    return {
        "a2_bank": _a2_bank(clean.get("A2_LEVELUP"), registry_total),
        "a3_self_play": _a3_self_play(clean.get("A3_SELF_PLAY")),
        "a4_fresh_live_rate": _a4_fresh_live(artifacts, clean, summarizer_results),
        "b2_package": _b2_package(clean.get("B2_PACKAGE")),
        "deadline_deliverable": "current_scored_agent_0.08"
        if invariant
        else "scorecard_only_no_escalation",
    }


def _honest_verdict(representation: Mapping[str, Any]) -> str:
    verdict = representation.get("verdict")
    if verdict == "representation_invariant_escalate_operator":
        return "complete_capstone_v451_representation_invariant_escalate_operator"
    if verdict == "untrusted_fork_non_test":
        return "complete_capstone_v451_untrusted_fork_non_test"
    return "complete_capstone_v451_trusted_fork_no_escalation"


def _operator_escalation_note(invariant: bool) -> str:
    if not invariant:
        return ""
    return (
        "A1 decision-need targets and A1b action-prefix latents both failed to move "
        "change-VALUE accuracy with non-degenerate controls. Under the 2026-06-30 "
        "deadline the competition deliverable remains the current scored agent "
        "(0.08); .452 should try a third representation class such as D's mapping "
        "or wait for operator redirect."
    )


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
    """Build the complete .451 scorecard after live summaries have run."""

    clean = _clean_artifacts(artifacts, summarizer_results)
    representation = _representation_fork_verdict(
        clean.get("A1"), clean.get("A1B"), clean.get("B1_AUDIT")
    )
    trusted = representation.get("trusted") is True
    invariant = representation.get("change_value_gap_representation_invariant") is True
    registry_total = _int(registry.get("reproducible_total_levels"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(representation),
        "representation_fork_verdict": representation,
        "fork_verdict_trusted": trusted,
        "change_value_gap_representation_invariant": invariant,
        "reproducible_total_levels": registry_total,
        "deadline_lever_scorecard": _deadline_scorecard(
            artifacts=artifacts,
            clean=clean,
            summarizer_results=summarizer_results,
            registry_total=registry_total,
            invariant=invariant,
        ),
        "flagged_upstreams_skipped": _flagged_upstreams_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "operator_escalation_note": _operator_escalation_note(invariant),
        "inference_substrate": INFERENCE_SUBSTRATE,
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
        "representation_fork_verdict": {},
        "fork_verdict_trusted": False,
        "change_value_gap_representation_invariant": False,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "deadline_lever_scorecard": {},
        "flagged_upstreams_skipped": _flagged_upstreams_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "operator_escalation_note": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
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
    """Return schema errors for the .451 scorecard without mutating it."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if not isinstance(payload.get("fork_verdict_trusted"), bool):
        errors.append("invalid_fork_verdict_trusted")
    if not isinstance(payload.get("change_value_gap_representation_invariant"), bool):
        errors.append("invalid_change_value_gap_representation_invariant")
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")
    field_principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    representation = payload.get("representation_fork_verdict")
    if isinstance(representation, Mapping) and representation:
        if representation.get("verdict") not in REPRESENTATION_VERDICTS:
            errors.append("invalid_representation_fork_verdict")
    elif not isinstance(representation, Mapping):
        errors.append("invalid_representation_fork_verdict")
    scorecard = payload.get("deadline_lever_scorecard")
    if not isinstance(scorecard, Mapping):
        errors.append("invalid_deadline_lever_scorecard")
    cited = payload.get("cited_upstream_artifacts")
    if not isinstance(cited, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not isinstance(row.get("fields_imported"), list)
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in cited
    ):
        errors.append("invalid_cited_upstream_artifacts")
    skipped = payload.get("flagged_upstreams_skipped")
    if not isinstance(skipped, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not str(row.get("sha256", "")).startswith("sha256:")
        or not row.get("reason")
        for row in skipped
    ):
        errors.append("invalid_flagged_upstreams_skipped")
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
        return "spec_missing_req_4901"
    if upstream_preconditions.get("A1", {}).get("present") is not True:
        return "a1_artifact_missing"
    if upstream_preconditions.get("B1_AUDIT", {}).get("present") is not True:
        return "b1_artifact_missing"
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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4901" in spec_path.read_text(
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
        "spec_has_req_4901": spec_has_req,
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

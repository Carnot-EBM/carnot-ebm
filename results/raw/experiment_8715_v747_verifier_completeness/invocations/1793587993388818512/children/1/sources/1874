"""Exp5577 capstone reconciliation for milestone 2026.07.504.

Spec refs: REQ-REPORT-5577, SCENARIO-REPORT-5577,
SCENARIO-REPORT-5577-MISSING-INPUT, SCENARIO-REPORT-5577-FIELD-PRINCIPLES.

This module is deliberately a synthesis ledger, not a rerun of any upstream
experiment. It reads every `.504` deliverable, records terminal blocked and
skipped evidence as terminal evidence, and keeps positive claims gated by the
strongest requirement that applies to each lane. That distinction matters for
this milestone because flagged memory evidence, blocked reset-free inference,
skipped ARC execution, and PTRM development-proxy work can explain what
happened without becoming broad CSL or ARC solve claims.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    path_sha256,
    payload_checksum,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5577_capstone_v504.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")

EXPERIMENT = "experiment_5577_capstone_v504"
EXPERIMENT_ID = "exp5577-capstone-v504"
MILESTONE = "2026.07.504"
EXPECTED_TASK_RANGE = "exp5564-exp5577"
RUN_DATE = "2026-07-11"
RANDOM_SEED = 5577
SCHEMA = "carnot.experiment_5577.capstone_v504.v1"
INFERENCE_SUBSTRATE = "aggregation_from_all_v504_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5577",
    "SCENARIO-REPORT-5577",
    "SCENARIO-REPORT-5577-MISSING-INPUT",
    "SCENARIO-REPORT-5577-FIELD-PRINCIPLES",
)

EXPECTED_ARTIFACT_PATHS = (
    Path("results/experiment_5564_transition_v504.json"),
    Path("results/experiment_5565_v504_source_delta_ingestion.json"),
    Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"),
    Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json"),
    Path("results/experiment_5568_verifier_coevolution_trigger.json"),
    Path("results/experiment_5569_causal_memory_policy_tournament.json"),
    Path("results/experiment_5570_spline_local_kan_online_energy.json"),
    Path("results/experiment_5571_reset_free_sota_continual_harness.json"),
    Path("results/experiment_5572_gated_delayed_regression_promotion.json"),
    Path("results/experiment_5573_matched_sampler_hardware_continuity.json"),
    Path("results/experiment_5573_matched_sampler_hardware_continuity_raw_rows.json"),
    Path("results/experiment_5574_ptrm_stochastic_generator_stage1.json"),
    Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json"),
    Path("results/experiment_5576_gated_sge_live_levelup.json"),
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    ROADMAP_RELATIVE_PATH,
    Path("research-roadmap-next.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("ops/exclusion_manifest.yaml"),
    REGISTRY_RELATIVE_PATH,
    Path("_bmad/traceability.md"),
    Path("research-complete.yaml"),
    CONDUCTOR_RELATIVE_PATH,
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5577_capstone_v504.py -q --no-cov",
        "outcome": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5577_capstone_v504.py "
            "-m pytest tests/python/test_experiment_5577_capstone_v504.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5577_capstone_v504.py "
            "--fail-under=100"
        ),
        "outcome": "not_run_in_default_artifact",
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "outcome": "not_run_in_default_artifact",
    },
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "milestone": "Route key for the `.504` capstone.",
    "expected_task_range": "Closed conductor boundary from transition through this capstone.",
    "upstream_artifacts_expected": "Every expected `.504` JSON artifact and sidecar before claim aggregation.",
    "upstream_artifacts_read": "Expected upstream artifacts actually parsed.",
    "missing_artifacts": "Absent expected inputs stay visible and never become success.",
    "clean_lanes": "Readable unflagged nonblocked evidence available for direct aggregation.",
    "bounded_lanes": "Useful evidence that is narrower than a headline claim.",
    "blocked_lanes": "Terminal blocked artifacts remain blockers rather than wins.",
    "flagged_lanes": "Adversarial or methodology-flagged evidence cannot support positive claims.",
    "skipped_lanes": "Conductor gate skips stay terminal but non-positive.",
    "retired_lanes": "Continuations closed by terminal null, block, or explicit retirement.",
    "promoted_lanes": "Narrow positive results promoted only when all lane-specific gates pass.",
    "solve_verify_asymmetry_supported": "True only for a clean measured positive solve-versus-verify asymmetry, not equal failure.",
    "verifier_coevolution_required": "True only when the clean co-evolution trigger artifact explicitly requires it.",
    "memory_policy_promoted": "False when the memory-policy artifact is adversarial-flagged, regardless of local policy_ready fields.",
    "kan_online_energy_promoted": "True only for clean KAN online update evidence with rollback and no unsafe false-accept increase.",
    "continuous_self_learning_claim_allowed": "Broad CSL requires unflagged memory policy, KAN promotion, reset-free harness, and delayed regression promotion.",
    "hardware_speedup_claim_allowed": "True only for matched successful hardware/device speedup above baseline, not board receipts or CPU/CUDA slowdown.",
    "ptrm_stage1_status": "Records PTRM Stage-1 development-proxy status separately from ARC solve credit.",
    "ptrm_retired": "True only when PTRM evidence explicitly retires the line.",
    "ordinary_arc_floor_satisfied": "True only when ordinary ARC live level-up produced self-discovered offline-reproduced registry delta.",
    "arc_solve_provenance": "ARC solve provenance audit; development_proxy cannot count as a solve.",
    "arc_registry_before_after": "Registry total before/after accounting; positive delta required for a solve claim.",
    "arc_registry_delta": "Counts only offline-reproduced live-agent registry increments.",
    "sge_retired": "True when the SGE live continuation is blocked or null after its precheck gate.",
    "specs_updated": "Spec reconciliation done by this workflow; ops and traceability edits may be delegated.",
    "traceability_updated": "Bare boolean for whether `_bmad/traceability.md` was edited by this workflow.",
    "ops_docs_updated": "Bare boolean for whether ops status/changelog/conductor docs were edited by this workflow.",
    "research_complete_updated": "Bare boolean for whether `research-complete.yaml` was edited by this workflow.",
    "exclusion_manifest_updated": "Bare boolean for whether exclusion or registry files were edited by this workflow.",
    "tests_run": "Commands run for the capstone and whether they passed, failed, or were not applicable.",
    "roadmap_yaml_unchanged": "Protected-file discipline; true only when `research-roadmap.yaml` is unchanged.",
    "conductor_unchanged": "Protected-file discipline; true only when `scripts/research_conductor.py` is unchanged.",
    "inference_substrate": "Must equal aggregation_from_all_v504_artifacts because Exp5577 is synthesis only.",
    "honest_verdict": "Terminal summary starting with complete: or blocked: that names the `.504` claim boundary.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "artifact_metadata",
    "source_context",
    "source_context_missing",
    "claim_boundaries",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)
BOOL_FIELDS = (
    "solve_verify_asymmetry_supported",
    "verifier_coevolution_required",
    "memory_policy_promoted",
    "kan_online_energy_promoted",
    "continuous_self_learning_claim_allowed",
    "hardware_speedup_claim_allowed",
    "ptrm_retired",
    "ordinary_arc_floor_satisfied",
    "sge_retired",
    "traceability_updated",
    "ops_docs_updated",
    "research_complete_updated",
    "exclusion_manifest_updated",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
)
LIST_FIELDS = (
    "upstream_artifacts_expected",
    "upstream_artifacts_read",
    "missing_artifacts",
    "clean_lanes",
    "bounded_lanes",
    "blocked_lanes",
    "flagged_lanes",
    "skipped_lanes",
    "retired_lanes",
    "promoted_lanes",
    "specs_updated",
    "tests_run",
)


def _read_json_any(path: Path) -> tuple[Any, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "json_type": None,
    }
    if not path.exists():
        return {}, metadata
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive metadata path
        metadata["error"] = str(exc)
        return {}, metadata
    metadata["loadable"] = True
    metadata["json_type"] = type(payload).__name__
    if isinstance(payload, list):
        metadata["length"] = len(payload)
    return payload, metadata


def _read_artifacts(root: Path) -> tuple[dict[str, Any], JsonDict, list[str], list[str]]:
    artifacts: dict[str, Any] = {}
    metadata: JsonDict = {}
    read_paths: list[str] = []
    missing_paths: list[str] = []
    for rel_path in EXPECTED_ARTIFACT_PATHS:
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[rel] = payload
        metadata[rel] = meta
        if meta["exists"] and meta["loadable"]:
            read_paths.append(rel)
        else:
            missing_paths.append(rel)
    return artifacts, metadata, read_paths, missing_paths


def _read_source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        exists = path.exists()
        records.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "read_only": True,
                "sha256": path_sha256(path),
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def _payload(artifacts: Mapping[str, Any], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


def _verdict(payload: JsonMap) -> str:
    verdict = payload.get("honest_verdict")
    return str(verdict) if verdict is not None else ""


def _status_label(payload: JsonMap) -> str:
    status = payload.get("status")
    if status is not None:
        return str(status).lower()
    verdict = _verdict(payload).lower()
    if verdict.startswith("blocked:") or verdict.startswith("blocked_"):
        return "blocked"
    if verdict.startswith("honest_null:") or "honest_null" in verdict:
        return "honest_null"
    if verdict.startswith("failed:"):
        return "failed"
    if verdict.startswith("complete:"):
        return "complete"
    return "unknown"


def _is_gate_skip(payload: JsonMap) -> bool:
    verdict = _verdict(payload)
    blocked_at_layer = str(payload.get("blocked_at_layer") or "").lower()
    return bool(
        payload.get("schema") == "blocked_gate_check_v1"
        or verdict.startswith("blocked_gate")
        or ("gate" in blocked_at_layer and _status_label(payload) == "blocked")
        or (payload.get("gate_check_summary") and _status_label(payload) == "blocked")
    )


def _is_flagged(payload: JsonMap) -> bool:
    return bool(payload.get("flagged_adversarial"))


def _is_blocked(payload: JsonMap) -> bool:
    return _status_label(payload) == "blocked" and not _is_gate_skip(payload)


def _is_honest_null(payload: JsonMap) -> bool:
    return _status_label(payload) == "honest_null"


def _is_failed(payload: JsonMap) -> bool:
    return _status_label(payload) == "failed"


def _clean_for_claim(payload: JsonMap) -> bool:
    return bool(payload) and not (
        _is_flagged(payload)
        or _is_gate_skip(payload)
        or _is_blocked(payload)
        or _is_honest_null(payload)
        or _is_failed(payload)
    )


def _int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def _float(payload: JsonMap, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _nested_floats(value: Any, key: str) -> list[float]:
    found: list[float] = []
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            if nested_key == key and isinstance(nested_value, int | float):
                found.append(float(nested_value))
            else:
                found.extend(_nested_floats(nested_value, key))
    elif isinstance(value, list):
        for nested_value in value:
            found.extend(_nested_floats(nested_value, key))
    return found


def _lane(
    lane: str,
    classification: str,
    source_artifacts: Sequence[Path],
    claim_boundary: str,
    evidence: JsonDict | None = None,
) -> JsonDict:
    return {
        "lane": lane,
        "classification": classification,
        "source_artifacts": [path.as_posix() for path in source_artifacts],
        "claim_boundary": claim_boundary,
        "evidence": evidence or {},
    }


def _registry_total(root: Path) -> int | None:
    path = root / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return None
    match = re.search(r"^reproducible_total_levels:\s*(\d+)\s*$", path.read_text(), re.M)
    return int(match.group(1)) if match else None


def _solve_verify_asymmetry_supported(payload: JsonMap) -> bool:
    deltas = _nested_floats(payload.get("solve_verify_asymmetry"), "solve_minus_verify_balanced_accuracy")
    return bool(_clean_for_claim(payload) and payload.get("panel_complete") and any(delta < 0.0 for delta in deltas))


def _kan_promoted(payload: JsonMap) -> bool:
    ci = payload.get("paired_ci_active_vs_frozen")
    ci_lower = _float(ci, "lower") if isinstance(ci, Mapping) else 0.0
    return bool(
        _clean_for_claim(payload)
        and payload.get("kan_ready")
        and _float(payload, "forward_adaptation_delta") > 0.0
        and ci_lower > 0.0
        and _float(payload, "prior_family_regression") <= 0.02
        and _float(payload, "unsafe_false_accept_delta") <= 0.0
        and payload.get("rollback_checksum_match")
        and payload.get("exact_feedback_only")
    )


def _hardware_speedup_allowed(payload: JsonMap) -> bool:
    rows = payload.get("speedup_by_pair")
    speedups = []
    if isinstance(rows, list):
        speedups = [float(row.get("speedup", 0.0)) for row in rows if isinstance(row, Mapping)]
    return bool(
        _clean_for_claim(payload)
        and _int(payload, "successful_matched_pairs") > 0
        and payload.get("hardware_speedup_claim_allowed")
        and not payload.get("board_speedup_claimed")
        and speedups
        and min(speedups) > 1.0
    )


def _ptrm_status(payload: JsonMap) -> str:
    if not payload:
        return "missing"
    if not _clean_for_claim(payload):
        return "blocked_or_flagged"
    if payload.get("stage1_training_complete") and payload.get("no_level_solve_claim"):
        return "complete_development_proxy_no_solve_claim"
    return "incomplete"


def _ordinary_arc_claim(exp5575: JsonMap, exp5576: JsonMap) -> JsonDict:
    provenance = exp5576.get("solve_provenance") or exp5575.get("solve_provenance")
    offline_reproduced = bool(exp5576.get("offline_reproduced"))
    registry_delta_raw = _int(exp5576, "registry_delta")
    claim_allowed = bool(
        _clean_for_claim(exp5576)
        and provenance == "live_agent_self_discovery"
        and offline_reproduced
        and registry_delta_raw > 0
    )
    return {
        "claim_allowed": claim_allowed,
        "solve_provenance": provenance,
        "live_path_ready": bool(exp5575.get("live_path_ready")),
        "target_unsolved": bool(exp5575.get("target_unsolved")),
        "offline_reproduced": offline_reproduced,
        "registry_delta_raw": registry_delta_raw,
        "registry_delta_counted": registry_delta_raw if claim_allowed else 0,
        "source_artifacts": [
            "results/experiment_5575_sge_anti_stagnation_live_precheck.json",
            "results/experiment_5576_gated_sge_live_levelup.json",
        ],
    }


def _build_lane_classification(artifacts: Mapping[str, Any], claims: JsonMap) -> JsonDict:
    exp5564 = _payload(artifacts, Path("results/experiment_5564_transition_v504.json"))
    exp5565 = _payload(artifacts, Path("results/experiment_5565_v504_source_delta_ingestion.json"))
    exp5566 = _payload(artifacts, Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"))
    exp5567 = _payload(artifacts, Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json"))
    exp5568 = _payload(artifacts, Path("results/experiment_5568_verifier_coevolution_trigger.json"))
    exp5569 = _payload(artifacts, Path("results/experiment_5569_causal_memory_policy_tournament.json"))
    exp5570 = _payload(artifacts, Path("results/experiment_5570_spline_local_kan_online_energy.json"))
    exp5571 = _payload(artifacts, Path("results/experiment_5571_reset_free_sota_continual_harness.json"))
    exp5572 = _payload(artifacts, Path("results/experiment_5572_gated_delayed_regression_promotion.json"))
    exp5573 = _payload(artifacts, Path("results/experiment_5573_matched_sampler_hardware_continuity.json"))
    exp5574 = _payload(artifacts, Path("results/experiment_5574_ptrm_stochastic_generator_stage1.json"))
    exp5575 = _payload(artifacts, Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json"))
    exp5576 = _payload(artifacts, Path("results/experiment_5576_gated_sge_live_levelup.json"))

    clean_lanes = [
        _lane(
            "transition_and_source_delta",
            "clean",
            (
                Path("results/experiment_5564_transition_v504.json"),
                Path("results/experiment_5565_v504_source_delta_ingestion.json"),
            ),
            "Transition and source-delta receipts are readable and do not reopen closed scopes.",
            {
                "transition_clean": _clean_for_claim(exp5564),
                "source_delta_clean": _clean_for_claim(exp5565),
                "closed_scopes_reopened": bool(exp5565.get("closed_scopes_reopened")),
            },
        )
    ]
    bounded_lanes = [
        _lane(
            "solve_verify_panel_measured_null",
            "bounded",
            (Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json"),),
            "The local SOTA panel ran, but equal/parser-failure collapse is not a positive asymmetry.",
            {
                "panel_complete": bool(exp5567.get("panel_complete")),
                "parser_failure_count": _int(exp5567, "parser_failure_count"),
                "solve_verify_asymmetry_supported": claims["solve_verify_asymmetry_supported"],
            },
        ),
        _lane(
            "matched_sampler_quality_no_speedup",
            "bounded",
            (
                Path("results/experiment_5573_matched_sampler_hardware_continuity.json"),
                Path("results/experiment_5573_matched_sampler_hardware_continuity_raw_rows.json"),
            ),
            "Matched CPU/CUDA sampler-quality rows landed, but speedup ratios do not support a hardware speedup headline.",
            {
                "successful_matched_pairs": _int(exp5573, "successful_matched_pairs"),
                "board_speedup_claimed": bool(exp5573.get("board_speedup_claimed")),
                "hardware_speedup_claim_allowed": claims["hardware_speedup_claim_allowed"],
            },
        ),
    ]
    blocked_lanes = [
        _lane(
            "reset_free_continual_harness",
            "blocked",
            (Path("results/experiment_5571_reset_free_sota_continual_harness.json"),),
            "Reset-free promotion is blocked because live local inference/offload was not authenticated.",
            {
                "honest_verdict": exp5571.get("honest_verdict"),
                "continual_harness_candidate": bool(exp5571.get("continual_harness_candidate")),
                "live_model_invoked": bool(exp5571.get("live_model_invoked")),
            },
        ),
        _lane(
            "sge_live_path_precheck",
            "blocked",
            (Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json"),),
            "SGE precheck reached the path but did not make the live path ready.",
            {
                "live_path_reachable": bool(exp5575.get("live_path_reachable")),
                "live_path_ready": bool(exp5575.get("live_path_ready")),
                "target_unsolved": bool(exp5575.get("target_unsolved")),
            },
        ),
    ]
    flagged_lanes = [
        _lane(
            "causal_memory_policy_tournament",
            "flagged",
            (Path("results/experiment_5569_causal_memory_policy_tournament.json"),),
            "The memory policy artifact is adversarial-flagged, so policy_ready cannot be promoted.",
            {
                "flagged_adversarial": bool(exp5569.get("flagged_adversarial")),
                "policy_ready": bool(exp5569.get("policy_ready")),
                "corrigendum_pending": exp5569.get("corrigendum_pending", []),
            },
        )
    ] if _is_flagged(exp5569) else []
    skipped_lanes = [
        _lane(
            "delayed_regression_promotion_gate",
            "skipped",
            (Path("results/experiment_5572_gated_delayed_regression_promotion.json"),),
            "Delayed promotion was conductor-skipped after the reset-free harness failed.",
            {
                "blocked_at_layer": exp5572.get("blocked_at_layer"),
                "gate_check_summary": exp5572.get("gate_check_summary"),
            },
        ),
        _lane(
            "sge_live_levelup_gate",
            "skipped",
            (Path("results/experiment_5576_gated_sge_live_levelup.json"),),
            "The live ARC level-up did not run because live_path_ready was false.",
            {
                "blocked_at_layer": exp5576.get("blocked_at_layer"),
                "gate_check_summary": exp5576.get("gate_check_summary"),
            },
        ),
    ]
    retired_lanes = [
        _lane(
            "broad_continuous_self_learning_claim",
            "retired",
            (
                Path("results/experiment_5569_causal_memory_policy_tournament.json"),
                Path("results/experiment_5571_reset_free_sota_continual_harness.json"),
                Path("results/experiment_5572_gated_delayed_regression_promotion.json"),
            ),
            "Broad CSL is closed for .504 because memory is flagged and the reset-free/delayed gates failed.",
            {"continuous_self_learning_claim_allowed": claims["continuous_self_learning_claim_allowed"]},
        ),
        _lane(
            "sge_anti_stagnation_continuation",
            "retired",
            (
                Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json"),
                Path("results/experiment_5576_gated_sge_live_levelup.json"),
            ),
            "The SGE continuation is terminal for .504 after precheck block and gate skip.",
            {"sge_retired": claims["sge_retired"]},
        ),
    ]
    promoted_lanes = [
        _lane(
            "exact_asp_fsm_near_miss_corpus",
            "promoted",
            (Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"),),
            "Exact ASP/FSM near-miss corpus is promoted as an authoritative verifier substrate.",
            {
                "corpus_ready": bool(exp5566.get("corpus_ready")),
                "n_rows": _int(exp5566, "n_rows"),
                "duplicate_leakage_count": _int(exp5566, "duplicate_leakage_count"),
            },
        ),
        _lane(
            "verifier_coevolution_trigger",
            "promoted",
            (Path("results/experiment_5568_verifier_coevolution_trigger.json"),),
            "A clean cached residual audit requires verifier co-evolution, not silent threshold retuning.",
            {
                "verifier_coevolution_required": claims["verifier_coevolution_required"],
                "triggered_by": exp5568.get("triggered_by", []),
            },
        ),
        _lane(
            "spline_local_kan_online_energy",
            "promoted",
            (Path("results/experiment_5570_spline_local_kan_online_energy.json"),),
            "KAN online energy update is promoted as bounded energy-parameter self-learning.",
            {
                "kan_ready": bool(exp5570.get("kan_ready")),
                "kan_online_energy_promoted": claims["kan_online_energy_promoted"],
            },
        ),
        _lane(
            "ptrm_stage1_development_proxy",
            "promoted",
            (Path("results/experiment_5574_ptrm_stochastic_generator_stage1.json"),),
            "PTRM Stage 1 is complete only as a development proxy, not as ARC solve credit.",
            {
                "ptrm_stage1_status": claims["ptrm_stage1_status"],
                "solve_provenance": exp5574.get("solve_provenance"),
                "no_level_solve_claim": bool(exp5574.get("no_level_solve_claim")),
            },
        ),
    ]
    return {
        "clean_lanes": clean_lanes,
        "bounded_lanes": bounded_lanes,
        "blocked_lanes": blocked_lanes,
        "flagged_lanes": flagged_lanes,
        "skipped_lanes": skipped_lanes,
        "retired_lanes": retired_lanes,
        "promoted_lanes": promoted_lanes,
    }


def _compute_claims(artifacts: Mapping[str, Any], root: Path) -> JsonDict:
    exp5567 = _payload(artifacts, Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json"))
    exp5568 = _payload(artifacts, Path("results/experiment_5568_verifier_coevolution_trigger.json"))
    exp5569 = _payload(artifacts, Path("results/experiment_5569_causal_memory_policy_tournament.json"))
    exp5570 = _payload(artifacts, Path("results/experiment_5570_spline_local_kan_online_energy.json"))
    exp5571 = _payload(artifacts, Path("results/experiment_5571_reset_free_sota_continual_harness.json"))
    exp5572 = _payload(artifacts, Path("results/experiment_5572_gated_delayed_regression_promotion.json"))
    exp5573 = _payload(artifacts, Path("results/experiment_5573_matched_sampler_hardware_continuity.json"))
    exp5574 = _payload(artifacts, Path("results/experiment_5574_ptrm_stochastic_generator_stage1.json"))
    exp5575 = _payload(artifacts, Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json"))
    exp5576 = _payload(artifacts, Path("results/experiment_5576_gated_sge_live_levelup.json"))

    memory_policy_promoted = bool(
        _clean_for_claim(exp5569)
        and exp5569.get("policy_ready")
        and _float(exp5569, "forward_transfer_delta") > 0.0
        and exp5569.get("rollback_success")
    )
    kan_online_energy_promoted = _kan_promoted(exp5570)
    delayed_promotion_clean = bool(_clean_for_claim(exp5572) and exp5572.get("promotion_allowed"))
    reset_free_promoted = bool(_clean_for_claim(exp5571) and exp5571.get("continual_harness_candidate"))
    ordinary_arc = _ordinary_arc_claim(exp5575, exp5576)
    arc_delta = int(ordinary_arc["registry_delta_counted"])
    registry_after = _registry_total(root)
    registry_before = registry_after - arc_delta if registry_after is not None else None
    ptrm_status = _ptrm_status(exp5574)
    sge_retired = bool(
        _is_gate_skip(exp5576)
        or _is_honest_null(exp5576)
        or (_is_blocked(exp5575) and not exp5575.get("live_path_ready"))
    )

    return {
        "solve_verify_asymmetry_supported": _solve_verify_asymmetry_supported(exp5567),
        "verifier_coevolution_required": bool(
            _clean_for_claim(exp5568) and exp5568.get("verifier_coevolution_required")
        ),
        "memory_policy_promoted": memory_policy_promoted,
        "kan_online_energy_promoted": kan_online_energy_promoted,
        "continuous_self_learning_claim_allowed": bool(
            memory_policy_promoted and kan_online_energy_promoted and reset_free_promoted and delayed_promotion_clean
        ),
        "hardware_speedup_claim_allowed": _hardware_speedup_allowed(exp5573),
        "ptrm_stage1_status": ptrm_status,
        "ptrm_retired": bool(_clean_for_claim(exp5574) and exp5574.get("retire_trm_generator_line")),
        "ordinary_arc_floor_satisfied": bool(ordinary_arc["claim_allowed"]),
        "arc_solve_provenance": {
            "ordinary_arc": ordinary_arc,
            "ptrm": {
                "source_artifact": "results/experiment_5574_ptrm_stochastic_generator_stage1.json",
                "solve_provenance": exp5574.get("solve_provenance"),
                "offline_reproduced": False,
                "registry_delta_counted": 0,
                "counts_as_ordinary_arc_slot": False,
            },
        },
        "arc_registry_before_after": {
            "source_path": REGISTRY_RELATIVE_PATH.as_posix(),
            "before": registry_before,
            "after": registry_after,
            "capstone_delta": arc_delta,
        },
        "arc_registry_delta": arc_delta,
        "sge_retired": sge_retired,
    }


def build_artifact(
    artifacts: Mapping[str, Any],
    artifact_metadata: JsonMap,
    upstream_artifacts_read: Sequence[str],
    missing_artifacts: Sequence[str],
    source_context: Sequence[JsonMap],
    source_context_missing: Sequence[str],
    *,
    root: Path,
    tests_run: Sequence[Any],
    roadmap_modified: bool,
    conductor_modified: bool,
) -> JsonDict:
    claims = _compute_claims(artifacts, root)
    lanes = _build_lane_classification(artifacts, claims)
    status_prefix = "blocked:" if missing_artifacts else "complete:"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "artifact_metadata": dict(artifact_metadata),
        "source_context": [dict(row) for row in source_context],
        "source_context_missing": list(source_context_missing),
        "claim_boundaries": [
            "Null, blocked, flagged, and skipped results are terminal evidence but not positive claims.",
            "Solve-versus-verify asymmetry is not supported because the panel measured equal collapse rather than verification lift.",
            "Verifier co-evolution is required by the clean cached residual audit.",
            "Memory policy is not promoted because Exp5569 is adversarial-flagged.",
            "KAN online energy is promoted only as bounded exact-feedback energy-parameter learning.",
            "Broad CSL is false because memory policy, reset-free harness, and delayed promotion gates did not all pass.",
            "Matched sampler-quality evidence landed, but no hardware speedup headline is allowed.",
            "PTRM Stage 1 is complete as a development proxy and cannot count as an ARC solve.",
            "Ordinary ARC floor is unsatisfied because no offline-reproduced live-agent registry delta landed.",
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "milestone": MILESTONE,
        "expected_task_range": EXPECTED_TASK_RANGE,
        "upstream_artifacts_expected": [path.as_posix() for path in EXPECTED_ARTIFACT_PATHS],
        "upstream_artifacts_read": list(upstream_artifacts_read),
        "missing_artifacts": list(missing_artifacts),
        "specs_updated": ["openspec/capabilities/research-reporting/spec.md"],
        "traceability_updated": False,
        "ops_docs_updated": False,
        "research_complete_updated": False,
        "exclusion_manifest_updated": False,
        "tests_run": list(tests_run),
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
        **lanes,
        **claims,
    }
    payload["honest_verdict"] = (
        f"{status_prefix} .504 capstone read {len(payload['upstream_artifacts_read'])}/"
        f"{len(payload['upstream_artifacts_expected'])} expected artifacts; "
        f"missing={len(payload['missing_artifacts'])}; flagged={len(payload['flagged_lanes'])}; "
        f"blocked={len(payload['blocked_lanes'])}; skipped={len(payload['skipped_lanes'])}; "
        f"promoted={len(payload['promoted_lanes'])}; "
        f"solve_verify_asymmetry_supported={payload['solve_verify_asymmetry_supported']}; "
        f"verifier_coevolution_required={payload['verifier_coevolution_required']}; "
        f"memory_policy_promoted={payload['memory_policy_promoted']}; "
        f"kan_online_energy_promoted={payload['kan_online_energy_promoted']}; "
        f"continuous_self_learning_claim_allowed={payload['continuous_self_learning_claim_allowed']}; "
        f"hardware_speedup_claim_allowed={payload['hardware_speedup_claim_allowed']}; "
        f"ordinary_arc_floor_satisfied={payload['ordinary_arc_floor_satisfied']}; "
        f"arc_registry_delta={payload['arc_registry_delta']}"
    )
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run_capstone(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, artifacts_read, missing = _read_artifacts(root)
    source_context, source_missing = _read_source_context(root)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    return build_artifact(
        artifacts,
        metadata,
        artifacts_read,
        missing,
        source_context,
        source_missing,
        root=root,
        tests_run=tests_run,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )


def validate_artifact(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(field)
    for field in BOOL_FIELDS:
        if field in payload and not isinstance(payload[field], bool):
            errors.append(field)
    for field in LIST_FIELDS:
        if field in payload and not isinstance(payload[field], list):
            errors.append(field)
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles")
    if payload.get("upstream_artifacts_expected") != [
        path.as_posix() for path in EXPECTED_ARTIFACT_PATHS
    ]:
        errors.append("upstream_artifacts_expected")
    if payload.get("memory_policy_promoted") is True:
        errors.append("memory_policy_promoted")
    if payload.get("continuous_self_learning_claim_allowed") is True and not (
        payload.get("memory_policy_promoted") and payload.get("kan_online_energy_promoted")
    ):
        errors.append("continuous_self_learning_claim_allowed")
    if payload.get("hardware_speedup_claim_allowed") is True:
        errors.append("hardware_speedup_claim_allowed")
    if payload.get("ordinary_arc_floor_satisfied") and _int(payload, "arc_registry_delta") <= 0:
        errors.append("ordinary_arc_floor_satisfied")
    if payload.get("roadmap_yaml_unchanged") is not True:
        errors.append("roadmap_yaml_unchanged")
    if payload.get("conductor_unchanged") is not True:
        errors.append("conductor_unchanged")
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone")
    if payload.get("expected_task_range") != EXPECTED_TASK_RANGE:
        errors.append("expected_task_range")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    honest_verdict = payload.get("honest_verdict")
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    return sorted(set(errors))


def write_capstone(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = run_capstone(
        root=root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - validate_artifact is tested directly
        raise ValueError(f"invalid Exp5577 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5577 artifact")
    args = parser.parse_args(argv)
    artifact = write_capstone() if args.write else run_capstone()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())

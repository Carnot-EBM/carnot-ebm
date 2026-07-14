"""Exp5647 V509 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5647, SCENARIO-CAPSTONE-5647,
SCENARIO-CAPSTONE-5647-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5647-FIELD-PRINCIPLES.

This module is an evidence ledger, not a new scientific experiment. It reads
the Exp5636-Exp5646 result artifacts, preserves gate skips and flags as
negative evidence, and writes the capstone JSON that closes milestone `.509`
without reopening timing, hardware-speedup, ARC solve, or production-enable
claims that the upstream artifacts did not earn.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
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
RESULT_RELATIVE_PATH = Path("results/experiment_5647_v509_capstone_reconciliation.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5647_v509_capstone_reconciliation"
EXPERIMENT_ID = "exp5647-v509-capstone-reconciliation"
MILESTONE = "2026.07.509"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5647
SCHEMA = "carnot.experiment_5647.v509_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-CAPSTONE-5647",
    "SCENARIO-CAPSTONE-5647",
    "SCENARIO-CAPSTONE-5647-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5647-FIELD-PRINCIPLES",
)

EXP5636_TRANSITION_PATH = Path("results/experiment_5636_transition_v509.json")
EXP5637_SOURCE_PATH = Path("results/experiment_5637_v509_source_delta_ingestion.json")
EXP5638_SCHEMA_PATH = Path("results/experiment_5638_fr11_gate_schema_corrigendum.json")
EXP5639_AUDIT_PATH = Path("results/experiment_5639_anytime_valid_csl_independent_audit.json")
EXP5640_SHADOW_PATH = Path("results/experiment_5640_fr11_shadow_pipeline_integration.json")
EXP5641_ARC_MODEL_PATH = Path("results/experiment_5641_arc_counterexample_executable_model.json")
EXP5642_ARC_LIVE_AB_PATH = Path("results/experiment_5642_arc_executable_model_live_ab.json")
EXP5643_ARC_LEVEL_PATH = Path(
    "results/experiment_5643_arc_live_self_discovery_levelup_v509.json"
)
EXP5644_TWO_AXIS_EXACT_PATH = Path(
    "results/experiment_5644_two_axis_parallel_tempering_exact_audit.json"
)
EXP5645_TWO_AXIS_QUALITY_PATH = Path(
    "results/experiment_5645_two_axis_tempering_hard_constraint_quality.json"
)
EXP5646_RUST_PARITY_PATH = Path("results/experiment_5646_two_axis_tempering_rust_parity.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5636-transition-v509": EXP5636_TRANSITION_PATH,
    "exp5637-v509-source-delta-ingestion": EXP5637_SOURCE_PATH,
    "exp5638-fr11-gate-schema-corrigendum": EXP5638_SCHEMA_PATH,
    "exp5639-anytime-valid-csl-independent-audit": EXP5639_AUDIT_PATH,
    "exp5640-fr11-shadow-pipeline-integration": EXP5640_SHADOW_PATH,
    "exp5641-arc-counterexample-executable-model": EXP5641_ARC_MODEL_PATH,
    "exp5642-arc-executable-model-live-ab": EXP5642_ARC_LIVE_AB_PATH,
    "exp5643-arc-live-self-discovery-levelup-v509": EXP5643_ARC_LEVEL_PATH,
    "exp5644-two-axis-parallel-tempering-exact-audit": EXP5644_TWO_AXIS_EXACT_PATH,
    "exp5645-two-axis-tempering-hard-constraint-quality": EXP5645_TWO_AXIS_QUALITY_PATH,
    "exp5646-two-axis-tempering-rust-parity": EXP5646_RUST_PARITY_PATH,
}
PRIMARY_ARTIFACT_PATHS = tuple(TASK_ARTIFACT_PATHS.values())
EXPECTED_TASK_IDS = tuple(TASK_ARTIFACT_PATHS)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-complete.yaml"),
    Path("research-references.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/known-issues.md"),
    Path("ops/conductor-log.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/arc_solve_registry.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("_bmad/traceability.md"),
    CONDUCTOR_RELATIVE_PATH,
)

DELEGATED_BY_STOP_RULE = (
    "ops/status.md",
    "ops/changelog.md",
    "_bmad/traceability.md",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "one-line annotations for every required capstone field.",
    "upstream_artifacts": (
        "fixed .509 upstream denominator; every claim traces to a hashed primary artifact."
    ),
    "upstream_gate_statuses": "gate-skipped, blocked, flagged, and complete work stay distinct.",
    "adversarial_verification_summary": (
        "critical flags block promotion even when an artifact otherwise completed."
    ),
    "fr11_schema_corrigendum_status": (
        "schema repair is not scientific recomputation or promotion."
    ),
    "fr11_independent_promotion_status": (
        "FR-11 promotion depends on independent anytime-valid audit gates."
    ),
    "fr11_shadow_integration_status": "shadow integration is opt-in and disabled by default.",
    "arc_executable_model_status": (
        "changed ARC mechanisms need exact replay, live reachability, utility, and zero-unsafe gates."
    ),
    "arc_registry_count_before": "authoritative live-credit baseline before Exp5643 banking.",
    "arc_registry_count_after": "authoritative live-credit total after Exp5643 banking.",
    "arc_solve_provenance": (
        "only live self-discovery plus independent reproduction and registry delta can credit a solve."
    ),
    "one_axis_replica_exchange_preserved": (
        "prior one-axis promotion remains true regardless of two-axis outcome."
    ),
    "two_axis_invariant_status": "exactness evidence is separate from quality evidence.",
    "two_axis_quality_status": (
        "quality promotion requires bounded hard-constraint evidence without material regression."
    ),
    "rust_parity_status": (
        "Rust portability is separate from exactness, quality, speed, and timing."
    ),
    "timing_claimed": "bare false keeps retired timing scopes closed.",
    "hardware_speedup_claimed": "bare false keeps retired hardware-speedup scopes closed.",
    "retirements_applied": "repeated terminal failures close or bound scopes without weakening flags.",
    "spec_reconciliation": "REQ-* alignment and backing tests are recorded.",
    "ops_reconciliation": "ops records, registries, and delegated stop-rule files are explicit.",
    "test_commands": "verification commands are reproducible.",
    "test_exit_codes": "observed command exits are recorded without inferring success.",
    "e2e_check_receipts": (
        "applicable operations checks ran and nonapplicable checks are justified."
    ),
    "inference_substrate": "must equal aggregation_from_upstream_artifacts.",
    "reproducibility_checksum": "content-addressed capstone output is stable.",
    "honest_verdict": "terminal summary starting with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "artifact_metadata",
    "source_context",
    "source_context_missing",
    "missing_artifacts",
    "malformed_artifacts",
    "terminal_status_by_task",
    "validation_results",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_VALIDATION_RESULTS = (
    {
        "command": (
            ".venv/bin/pytest "
            "tests/python/test_experiment_5647_v509_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5647_v509_capstone_reconciliation.py -m pytest "
            "tests/python/test_experiment_5647_v509_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5647_v509_capstone_reconciliation.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": "python scripts/check_spec_coverage.py",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": "python scripts/exclusion_manifest_lint.py",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            "python scripts/adversarial_verify.py "
            "results/experiment_5647_v509_capstone_reconciliation.json"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": "python scripts/root_clutter_sweep.py --check",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
)


def _read_json_any(path: Path) -> tuple[JsonDict, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "json_type": None,
        "sha256": path_sha256(path),
    }
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        metadata.update({"error": "malformed_json", "line": exc.lineno, "column": exc.colno})
        return {}, metadata
    metadata["json_type"] = type(parsed).__name__
    if not isinstance(parsed, Mapping):
        metadata["error"] = "not_json_object"
        return {}, metadata
    metadata.update({"loadable": True, "error": None})
    return dict(parsed), metadata


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


def read_artifacts(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    rows: list[JsonDict] = []
    path_to_task = {path: task_id for task_id, path in TASK_ARTIFACT_PATHS.items()}
    for rel_path in PRIMARY_ARTIFACT_PATHS:
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        metadata[rel] = meta
        payloads[rel] = payload
        rows.append(
            {
                "task_id": path_to_task[rel_path],
                "path": rel,
                "exists": bool(meta.get("exists")),
                "loadable": bool(meta.get("loadable")),
                "sha256": meta.get("sha256"),
                "schema": payload.get("schema"),
                "experiment_id": payload.get("experiment_id", payload.get("experiment")),
                "milestone": payload.get("milestone"),
                "honest_verdict": _verdict(payload) or None,
                "inference_substrate": payload.get("inference_substrate"),
                "terminal_prefix_valid": _verdict(payload).startswith(TERMINAL_PREFIXES),
            }
        )
    return payloads, metadata, rows


def _payload(artifacts: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


def _verdict(payload: JsonMap) -> str:
    verdict = payload.get("honest_verdict")
    return str(verdict) if verdict is not None else ""


def _is_gate_skip(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    blocked_at_layer = str(payload.get("blocked_at_layer") or "").lower()
    return bool(
        payload.get("schema") == "blocked_gate_check_v1"
        or verdict == "blocked_gate_check_failed"
        or ("gate" in blocked_at_layer and payload.get("status") == "blocked")
        or (payload.get("gate_check_summary") and payload.get("status") == "blocked")
    )


def _is_blocked(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(
        status == "blocked"
        or verdict.startswith("blocked:")
        or verdict.startswith("blocked_")
        or verdict.startswith("blocked ")
    )


def _is_complete(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(
        status == "complete" or verdict.startswith("complete:") or verdict.startswith("complete_")
    )


def _is_flagged(payload: JsonMap) -> bool:
    return bool(payload.get("flagged_adversarial"))


def _number(payload: JsonMap, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def _bool(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False


def _clean_payload(payload: JsonMap) -> bool:
    return bool(payload) and not (
        _is_flagged(payload) or _is_gate_skip(payload) or _is_blocked(payload)
    )


def _status_for_payload(payload: JsonMap, meta: JsonMap) -> str:
    if not meta.get("exists"):
        return "missing"
    if not meta.get("loadable"):
        return "malformed"
    if _is_flagged(payload):
        return "flagged"
    if _is_gate_skip(payload):
        return "gate_skipped"
    if _is_blocked(payload):
        return "blocked"
    if _is_complete(payload):
        return "complete"
    return "unknown"


def _severity_rank(severity: str) -> int:
    return {"none": 0, "info": 1, "warn": 2, "warning": 2, "critical": 3}.get(
        severity.lower(),
        0,
    )


def _max_severity(flags: Any) -> str:
    if not isinstance(flags, Sequence) or isinstance(flags, str | bytes | bytearray):
        return "none"
    max_seen = "none"
    for flag in flags:
        if not isinstance(flag, Mapping):
            continue
        severity = str(flag.get("severity") or "none").lower()
        if _severity_rank(severity) > _severity_rank(max_seen):
            max_seen = severity
    return max_seen


def terminal_status_by_task(
    artifacts: Mapping[str, JsonMap], metadata: JsonMap
) -> dict[str, JsonDict]:
    statuses: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        rel = rel_path.as_posix()
        payload = _payload(artifacts, rel_path)
        meta = metadata.get(rel, {})
        status = _status_for_payload(payload, meta)
        statuses[task_id] = {
            "status": status,
            "artifact_path": rel,
            "honest_verdict": _verdict(payload) or None,
            "development_proxy": payload.get("solve_provenance") == "development_proxy",
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "corrigendum_pending": payload.get("corrigendum_pending", []),
            "supports_positive_claim": status == "complete",
            "sha256": meta.get("sha256"),
            "metadata_error": meta.get("error"),
        }
    return statuses


def _benefit_gate(payload: JsonMap) -> tuple[bool, JsonDict]:
    intervals = payload.get("paired_benefit_intervals")
    if not isinstance(intervals, Mapping) or not intervals:
        return False, {"count": 0, "minimum_lower": None}
    lower_bounds = [
        _number(row, "lower") for row in intervals.values() if isinstance(row, Mapping)
    ]
    if len(lower_bounds) != len(intervals):
        return False, {"count": len(lower_bounds), "minimum_lower": None}
    minimum_lower = min(lower_bounds)
    return minimum_lower > 0.0, {"count": len(lower_bounds), "minimum_lower": minimum_lower}


def _pathwise_gate(payload: JsonMap) -> tuple[bool, JsonDict]:
    raw_rows = payload.get("pathwise_risk_upper_bound")
    risk_limit = _number(payload.get("preregistered_thresholds", {}), "risk_limit") or 0.1
    if isinstance(raw_rows, Mapping):
        risk_limit = _number(raw_rows, "risk_limit") or risk_limit
        interval_rows = raw_rows.get("risk_intervals", {})
        rows = list(interval_rows.values()) if isinstance(interval_rows, Mapping) else []
    elif isinstance(raw_rows, Sequence) and not isinstance(raw_rows, str | bytes | bytearray):
        rows = [row for row in raw_rows if isinstance(row, Mapping)]
    else:
        nested = payload.get("pathwise_risk")
        if isinstance(nested, Mapping):
            risk_limit = _number(nested, "risk_limit") or risk_limit
            interval_rows = nested.get("risk_intervals", {})
            rows = list(interval_rows.values()) if isinstance(interval_rows, Mapping) else []
        else:
            rows = []
    if not rows:
        return False, {"count": 0, "max_upper": None, "risk_limit": risk_limit}
    upper_values = [
        _number(row, "upper_bound") if "upper_bound" in row else _number(row, "upper")
        for row in rows
    ]
    max_upper = max(upper_values)
    pass_rows = all(
        _bool(row, "within_limit")
        and (
            _number(row, "upper_bound")
            if "upper_bound" in row
            else _number(row, "upper")
        )
        <= (_number(row, "risk_limit") or risk_limit)
        for row in rows
    )
    return pass_rows, {"count": len(rows), "max_upper": max_upper, "risk_limit": risk_limit}


def _coverage_gate(payload: JsonMap) -> tuple[bool, JsonDict]:
    coverage = payload.get("worst_group_coverage")
    if not isinstance(coverage, Mapping):
        return False, {"coverage": 0.0, "floor": 0.9}
    floor = _number(coverage, "floor") or _number(
        payload.get("preregistered_thresholds", {}),
        "coverage_floor",
    ) or 0.9
    observed = _number(coverage, "coverage")
    powered = _bool(coverage, "adequately_powered_groups_only")
    return bool(observed >= floor and powered), {
        "coverage": observed,
        "floor": floor,
        "adequately_powered_groups_only": powered,
        "n": _int(coverage, "n"),
    }


def _recomputation_gate(payload: JsonMap) -> tuple[bool, JsonDict]:
    receipt = payload.get("independent_metric_recomputation")
    if not isinstance(receipt, Mapping):
        receipt = payload.get("independent_recomputation")
    if not isinstance(receipt, Mapping):
        return False, {
            "row_level_replay_performed": False,
            "exp5628_aggregate_metrics_used_as_authority": None,
        }
    row_replay = _bool(receipt, "row_level_replay_performed")
    aggregate_authority = _bool(receipt, "exp5628_aggregate_metrics_used_as_authority")
    return bool(row_replay and not aggregate_authority), {
        "row_level_replay_performed": row_replay,
        "exp5628_aggregate_metrics_used_as_authority": aggregate_authority,
        "checkpoint_receipts_replayed": _int(receipt, "checkpoint_receipts_replayed"),
        "decision_ledger_rows_replayed": _int(receipt, "decision_ledger_rows_replayed"),
        "conformal_prediction_rows_replayed": _int(
            receipt, "conformal_prediction_rows_replayed"
        ),
    }


def _adversarial_controls_gate(payload: JsonMap) -> tuple[bool, list[str]]:
    controls = payload.get("adversarial_controls")
    if not isinstance(controls, Mapping) or not controls:
        return False, []
    failed = [
        str(name)
        for name, row in controls.items()
        if not isinstance(row, Mapping) or row.get("critical") is not True or row.get("pass") is not True
    ]
    return not failed, failed


def _derive_fr11_schema(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    payload = _payload(artifacts, EXP5638_SCHEMA_PATH)
    ready = bool(
        _clean_payload(payload)
        and _number(payload, "gate_contract_ready_score") == 1.0
        and _bool(payload, "source_hash_exact")
        and _int(payload, "unsafe_false_accept_count_total") == 0
        and _bool(payload, "by_arm_reconciliation_pass")
        and _bool(payload, "source_continuous_self_learning_ready")
        and not _bool(payload, "scientific_recompute_performed")
        and not _bool(payload, "source_artifact_modified")
    )
    return {
        "schema_repair_ready": ready,
        "gate_contract_ready_score": _number(payload, "gate_contract_ready_score"),
        "source_hash_exact": _bool(payload, "source_hash_exact"),
        "unsafe_false_accept_count_total": _int(payload, "unsafe_false_accept_count_total"),
        "by_arm_reconciliation_pass": _bool(payload, "by_arm_reconciliation_pass"),
        "scientific_recompute_performed": _bool(payload, "scientific_recompute_performed"),
        "source_artifact_modified": _bool(payload, "source_artifact_modified"),
        "promoted_as_science": False,
        "promoted": False,
        "boundary": "schema_contract_only_not_scientific_recompute",
    }


def _derive_fr11_independent(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    payload = _payload(artifacts, EXP5639_AUDIT_PATH)
    benefit_pass, benefit_receipt = _benefit_gate(payload)
    pathwise_pass, pathwise_receipt = _pathwise_gate(payload)
    coverage_pass, coverage_receipt = _coverage_gate(payload)
    recomputation_pass, recomputation_receipt = _recomputation_gate(payload)
    controls_pass, failed_controls = _adversarial_controls_gate(payload)
    gates_enforced = bool(
        isinstance(payload.get("upstream_gate_receipts"), Mapping)
        and payload["upstream_gate_receipts"].get("both_structured_gates_enforced") is True
    )
    checklist = {
        "clean_complete_payload": _clean_payload(payload),
        "structured_gate_receipts_enforced": gates_enforced,
        "ready_score_one": _number(payload, "fr11_independent_promotion_ready_score") == 1.0,
        "benefit_gate_pass": benefit_pass,
        "pathwise_risk_gate_pass": pathwise_pass,
        "worst_group_coverage_gate_pass": coverage_pass,
        "exact_safety_gate_pass": _int(payload, "unsafe_false_accept_count_total") == 0,
        "retention_pass": _bool(payload, "retention_pass"),
        "poison_rejection_pass": _bool(payload, "poison_rejection_pass"),
        "checkpoint_replay_pass": _bool(payload, "checkpoint_replay_pass"),
        "critical_flag_count_zero": _int(payload, "critical_flag_count") == 0,
        "independent_recomputation_pass": recomputation_pass,
        "adversarial_controls_pass": controls_pass,
    }
    failed_condition = None
    failure_order = (
        ("clean_complete_payload", "audit_not_complete_clean"),
        ("structured_gate_receipts_enforced", "structured_gate_receipts_not_enforced"),
        ("ready_score_one", "ready_score_not_one"),
        ("benefit_gate_pass", "benefit_interval_lower_bound_not_positive"),
        ("pathwise_risk_gate_pass", "pathwise_risk_gate_failed"),
        ("worst_group_coverage_gate_pass", "worst_group_coverage_gate_failed"),
        ("exact_safety_gate_pass", "unsafe_false_accept_count_nonzero"),
        ("retention_pass", "retention_gate_failed"),
        ("poison_rejection_pass", "poison_gate_failed"),
        ("checkpoint_replay_pass", "checkpoint_replay_gate_failed"),
        ("critical_flag_count_zero", "critical_flag_count_nonzero"),
        ("independent_recomputation_pass", "independent_recomputation_gate_failed"),
        ("adversarial_controls_pass", "adversarial_controls_gate_failed"),
    )
    for key, reason in failure_order:
        if not checklist[key]:
            failed_condition = reason
            break
    promoted = failed_condition is None
    return {
        "promoted": promoted,
        "ready_score": _number(payload, "fr11_independent_promotion_ready_score"),
        "checklist": checklist,
        "benefit_receipt": benefit_receipt,
        "pathwise_risk_receipt": pathwise_receipt,
        "worst_group_coverage_receipt": coverage_receipt,
        "independent_recomputation_receipt": recomputation_receipt,
        "failed_adversarial_controls": failed_controls,
        "failed_condition": failed_condition,
    }


def _derive_fr11_shadow(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    payload = _payload(artifacts, EXP5640_SHADOW_PATH)
    ready = bool(
        _clean_payload(payload)
        and _number(payload, "fr11_shadow_integration_ready_score") == 1.0
        and _bool(payload, "default_enabled") is False
        and _bool(payload, "exact_verifier_authority")
        and _bool(payload, "benefit_evidence_within_exp5639_bound")
        and _int(payload, "unsafe_update_accept_count") == 0
        and not _bool(payload, "model_weight_mutation")
        and _bool(payload, "shadow_offline_parity")
        and _bool(payload, "default_path_equivalence")
        and _bool(payload, "checkpoint_atomicity_pass")
        and _bool(payload, "restart_replay_pass")
        and _bool(payload, "rollback_pass")
        and _bool(payload, "ledger_lineage_complete")
    )
    return {
        "ready": ready,
        "default_enabled": _bool(payload, "default_enabled"),
        "feature_flag": payload.get("feature_flag"),
        "unsafe_update_accept_count": _int(payload, "unsafe_update_accept_count"),
        "model_weight_mutation": _bool(payload, "model_weight_mutation"),
        "shadow_offline_parity": _bool(payload, "shadow_offline_parity"),
        "default_path_equivalence": _bool(payload, "default_path_equivalence"),
        "automatic_production_enablement": False,
        "production_enabled": False,
        "deployment_boundary": "opt_in_shadow_only_disabled_by_default",
    }


def _derive_arc_executable(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    transition = _payload(artifacts, EXP5636_TRANSITION_PATH)
    exp5641 = _payload(artifacts, EXP5641_ARC_MODEL_PATH)
    exp5642 = _payload(artifacts, EXP5642_ARC_LIVE_AB_PATH)
    transition_text = json.dumps(
        transition.get("retired_scopes", []) + transition.get("promoted_substrates", []),
        sort_keys=True,
    )
    error_by_arm = exp5641.get("heldout_transition_error_by_arm", {})
    if not isinstance(error_by_arm, Mapping):
        error_by_arm = {}
    interval = exp5641.get("patched_vs_unpatched_error_reduction_interval")
    if isinstance(interval, Mapping):
        utility_lower = _number(interval, "lower")
    else:
        utility_lower = _number(error_by_arm, "unpatched") - _number(error_by_arm, "patched")
    exact_replay = _bool(exp5641, "all_receipt_replay_pass")
    zero_unsafe = _int(exp5641, "unsafe_patch_accept_count") == 0
    utility_pass = bool(utility_lower > 0.0 and _number(exp5641, "executable_model_ready_score") == 1.0)
    live_reachability = bool(_clean_payload(exp5642) and _bool(exp5642, "live_reachability_pass"))
    known_level_utility = bool(_clean_payload(exp5642) and _bool(exp5642, "known_level_utility_pass"))
    live_zero_unsafe = _int(exp5642, "unsafe_model_accept_count") == 0
    promoted = bool(
        _clean_payload(exp5641)
        and _number(exp5641, "executable_model_ready_score") == 1.0
        and exact_replay
        and zero_unsafe
        and live_reachability
        and known_level_utility
        and utility_pass
        and live_zero_unsafe
    )
    failed_condition = None
    if not promoted:
        if not _clean_payload(exp5641) or _number(exp5641, "executable_model_ready_score") != 1.0:
            failed_condition = "exp5641_executable_model_not_ready"
        elif not utility_pass:
            failed_condition = "known_level_utility_not_positive"
        elif not live_reachability:
            failed_condition = "exp5642_live_reachability_not_passed"
        elif not known_level_utility:
            failed_condition = "exp5642_known_level_utility_not_passed"
        elif not exact_replay or not zero_unsafe or not live_zero_unsafe:
            failed_condition = "exact_replay_or_zero_unsafe_gate_failed"
    return {
        "promoted": promoted,
        "exp5630_retired_preserved": "exp5630" in transition_text or "arc_epistemic" in transition_text,
        "exp5641_exact_replay_pass": exact_replay,
        "exp5641_zero_unsafe_pass": zero_unsafe,
        "exp5641_ready_score": _number(exp5641, "executable_model_ready_score"),
        "exp5641_known_level_utility_lower_bound": utility_lower,
        "exp5642_live_reachability_pass": live_reachability,
        "exp5642_known_level_utility_pass": known_level_utility,
        "exp5642_gate_skipped": _is_gate_skip(exp5642),
        "failed_condition": failed_condition,
        "boundary": "changed_mechanism_not_promoted_without_live_ab_and_utility",
    }


def _derive_arc_solve(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    payload = _payload(artifacts, EXP5643_ARC_LEVEL_PATH)
    flags = payload.get("corrigendum_pending", [])
    max_severity = _max_severity(flags)
    registry_before = _int(payload, "registry_count_before")
    registry_after = _int(payload, "registry_count_after")
    registry_delta = _int(payload, "registry_delta")
    reproduction_gate = payload.get("reproduction_gate", {})
    independent_reproduction = bool(
        _bool(payload, "independent_generic_reproduction")
        or (
            isinstance(reproduction_gate, Mapping)
            and reproduction_gate.get("reproduced") is True
            and _bool(payload, "offline_reproduced")
        )
    )
    critical_blocks = max_severity == "critical"
    solve_credited = bool(
        payload.get("solve_provenance") == "live_agent_self_discovery"
        and _bool(payload, "live_attempt_executed")
        and _bool(payload, "offline_reproduced")
        and independent_reproduction
        and registry_delta == 1
        and registry_after - registry_before == 1
        and _bool(payload, "registry_updated")
        and not critical_blocks
        and not _bool(payload, "source_read")
        and not _bool(payload, "game_adapter_used")
        and not _bool(payload, "outer_loop_re_used")
        and not _bool(payload, "offline_bfs_used")
    )
    return {
        "solve_credited": solve_credited,
        "solve_provenance": payload.get("solve_provenance"),
        "live_attempt_executed": _bool(payload, "live_attempt_executed"),
        "offline_reproduced": _bool(payload, "offline_reproduced"),
        "independent_generic_reproduction": independent_reproduction,
        "registry_count_before": registry_before,
        "registry_count_after": registry_after,
        "registry_delta": registry_delta,
        "registry_updated": _bool(payload, "registry_updated"),
        "new_reproducible_levels": payload.get("new_reproducible_levels", []),
        "max_flag_severity": max_severity,
        "critical_flag_blocks_credit": critical_blocks,
        "flagged_adversarial": _bool(payload, "flagged_adversarial"),
        "source_read": _bool(payload, "source_read"),
        "game_adapter_used": _bool(payload, "game_adapter_used"),
        "outer_loop_re_used": _bool(payload, "outer_loop_re_used"),
        "offline_bfs_used": _bool(payload, "offline_bfs_used"),
        "failed_condition": None
        if solve_credited
        else "no_registry_delta_independent_reproduction_or_critical_flag",
    }


def _derive_two_axis(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    transition = _payload(artifacts, EXP5636_TRANSITION_PATH)
    exact = _payload(artifacts, EXP5644_TWO_AXIS_EXACT_PATH)
    quality = _payload(artifacts, EXP5645_TWO_AXIS_QUALITY_PATH)
    rust = _payload(artifacts, EXP5646_RUST_PARITY_PATH)
    promoted_text = json.dumps(transition.get("promoted_substrates", []), sort_keys=True)
    one_axis_preserved = "one_axis" in promoted_text or "temperature_exchange" in promoted_text
    exact_metric_fields = (
        "exact_joint_target_tv",
        "exact_target_replica_tv",
        "horizontal_detailed_balance_error_max",
        "vertical_detailed_balance_error_max",
        "transition_row_error_max",
        "target_feasibility_marginal_error",
    )
    exact_metrics = {field: _number(exact, field) for field in exact_metric_fields}
    exact_promoted = bool(
        _clean_payload(exact)
        and _number(exact, "two_axis_invariant_ready_score") == 1.0
        and all(value <= 1e-9 for value in exact_metrics.values())
        and _bool(exact, "deterministic_replay_pass")
        and _bool(exact, "broken_control_rejected")
        and not _bool(exact, "timing_claimed")
        and not _bool(exact, "hardware_speedup_claimed")
    )
    target_diagnostics = quality.get("target_diagnostics", {})
    if not isinstance(target_diagnostics, Mapping):
        target_diagnostics = {}
    quality_promoted = bool(
        exact_promoted
        and _clean_payload(quality)
        and _number(quality, "two_axis_quality_ready_score") == 1.0
        and _int(quality, "invalid_execution_count") == 0
        and _int(quality, "material_quality_regression_count") == 0
        and _bool(target_diagnostics, "within_exactness_bounds")
        and not _bool(quality, "timing_claimed")
        and not _bool(quality, "hardware_speedup_claimed")
    )
    rust_promoted = bool(
        quality_promoted
        and _clean_payload(rust)
        and (
            _number(rust, "rust_parity_ready_score") == 1.0
            or _bool(rust, "parity_pass")
        )
        and not _bool(rust, "timing_claimed")
        and not _bool(rust, "hardware_speedup_claimed")
    )
    return {
        "one_axis_replica_exchange_preserved": one_axis_preserved,
        "two_axis_invariant_status": {
            "promoted": exact_promoted,
            "ready_score": _number(exact, "two_axis_invariant_ready_score"),
            "exact_metrics": exact_metrics,
            "deterministic_replay_pass": _bool(exact, "deterministic_replay_pass"),
            "broken_control_rejected": _bool(exact, "broken_control_rejected"),
            "timing_claimed": _bool(exact, "timing_claimed"),
            "hardware_speedup_claimed": _bool(exact, "hardware_speedup_claimed"),
        },
        "two_axis_quality_status": {
            "promoted": quality_promoted,
            "ready_score": _number(quality, "two_axis_quality_ready_score"),
            "successful_seed_count": _int(quality, "successful_seed_count"),
            "invalid_execution_count": _int(quality, "invalid_execution_count"),
            "material_quality_regression_count": _int(
                quality, "material_quality_regression_count"
            ),
            "within_exactness_bounds": _bool(target_diagnostics, "within_exactness_bounds"),
            "timing_claimed": _bool(quality, "timing_claimed"),
            "hardware_speedup_claimed": _bool(quality, "hardware_speedup_claimed"),
            "failed_condition": None
            if quality_promoted
            else "quality_gate_failed_or_material_regression_present",
        },
        "rust_parity_status": {
            "promoted": rust_promoted,
            "gate_skipped": _is_gate_skip(rust),
            "parity_pass": _bool(rust, "parity_pass"),
            "ready_score": _number(rust, "rust_parity_ready_score"),
            "timing_claimed": _bool(rust, "timing_claimed"),
            "hardware_speedup_claimed": _bool(rust, "hardware_speedup_claimed"),
            "failed_condition": None
            if rust_promoted
            else "quality_gate_not_promoted_or_rust_parity_gate_skipped",
        },
    }


def _derive_gate_statuses(
    artifacts: Mapping[str, JsonMap], statuses: Mapping[str, JsonMap]
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, rel_path)
        status = str(statuses[task_id]["status"])
        rows[task_id] = {
            "status": status,
            "artifact_path": rel_path.as_posix(),
            "gate_check_summary": payload.get("gate_check_summary"),
            "gates_evaluated": payload.get("gates_evaluated", []),
            "gate_skipped": status == "gate_skipped",
            "blocked": status == "blocked",
            "flagged": status == "flagged",
            "supports_promotion": status == "complete",
            "skipped_work_is_success": False,
        }
    return rows


def _derive_adversarial_summary(
    artifacts: Mapping[str, JsonMap], statuses: Mapping[str, JsonMap]
) -> JsonDict:
    flagged_rows: list[JsonDict] = []
    critical_rows: list[JsonDict] = []
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, rel_path)
        flags = payload.get("corrigendum_pending", [])
        max_severity = _max_severity(flags)
        critical_count = _int(payload, "critical_flag_count")
        if payload.get("flagged_adversarial") or max_severity != "none" or critical_count:
            row = {
                "task_id": task_id,
                "artifact_path": rel_path.as_posix(),
                "status": statuses[task_id]["status"],
                "flagged_adversarial": _bool(payload, "flagged_adversarial"),
                "max_severity": max_severity,
                "critical_flag_count": critical_count,
                "flags": flags if isinstance(flags, list) else [],
            }
            flagged_rows.append(row)
            if max_severity == "critical" or critical_count:
                critical_rows.append(row)
    exp5639_controls_pass, failed_controls = _adversarial_controls_gate(
        _payload(artifacts, EXP5639_AUDIT_PATH)
    )
    return {
        "flagged_tasks": flagged_rows,
        "critical_flags": critical_rows,
        "flagged_task_count": len(flagged_rows),
        "critical_flag_count": len(critical_rows),
        "critical_flags_block_promotion": True,
        "exp5639_adversarial_controls_pass": exp5639_controls_pass,
        "exp5639_failed_adversarial_controls": failed_controls,
        "promotion_blocked_task_ids": [str(row["task_id"]) for row in critical_rows],
    }


def _derive_retirements(
    arc_status: JsonMap,
    arc_solve: JsonMap,
    two_axis: JsonMap,
) -> list[JsonDict]:
    quality = two_axis.get("two_axis_quality_status", {})
    rust = two_axis.get("rust_parity_status", {})
    return [
        {
            "scope": "arc_epistemic_object_probe_exp5630",
            "decision": "preserved_retired_from_v508",
            "applied_to_capstone_claims": True,
            "manifest_update_required": False,
            "manifest_updated": False,
            "reason": "Exp5636 transition preserves the Exp5630 retirement.",
        },
        {
            "scope": "arc_counterexample_executable_model_exp5641",
            "decision": "retired_terminal_not_promoted",
            "applied_to_capstone_claims": True,
            "manifest_update_required": True,
            "manifest_updated": False,
            "reason": str(arc_status.get("failed_condition")),
        },
        {
            "scope": "arc_live_solve_credit_exp5643",
            "decision": "no_solve_credit_bounded_flagged_attempt",
            "applied_to_capstone_claims": True,
            "manifest_update_required": False,
            "manifest_updated": False,
            "reason": str(arc_solve.get("failed_condition")),
        },
        {
            "scope": "two_axis_quality_extension_exp5645",
            "decision": "two_axis_extension_retired_preserve_one_axis",
            "applied_to_capstone_claims": True,
            "manifest_update_required": True,
            "manifest_updated": False,
            "reason": str(quality.get("failed_condition")),
        },
        {
            "scope": "two_axis_tempering_rust_portability_exp5646",
            "decision": "not_promoted_gate_skipped",
            "applied_to_capstone_claims": True,
            "manifest_update_required": False,
            "manifest_updated": False,
            "reason": str(rust.get("failed_condition")),
        },
    ]


def _spec_reconciliation() -> JsonDict:
    return {
        "spec_path": "openspec/capabilities/capstone/spec.md",
        "spec_refs": list(SPEC_REFS),
        "req_alignment_recorded": True,
        "implementation_file": "python/carnot/experiment_5647_v509_capstone_reconciliation.py",
        "test_file": "tests/python/test_experiment_5647_v509_capstone_reconciliation.py",
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
    }


def _ops_reconciliation(
    root: Path,
    source_context: Sequence[JsonMap],
    roadmap_unchanged: bool,
    conductor_unchanged: bool,
) -> JsonDict:
    context_by_path = {str(row.get("path")): row for row in source_context}
    reviewed = [
        "research-complete.yaml",
        "research-references.md",
        "ops/exclusion_manifest.yaml",
        "ops/arc_solve_registry.yaml",
        "ops/known-issues.md",
        "ops/conductor-log.md",
        "ops/e2e-test-plan.md",
    ]
    return {
        "protected_files": {
            ROADMAP_RELATIVE_PATH.as_posix(): roadmap_unchanged,
            CONDUCTOR_RELATIVE_PATH.as_posix(): conductor_unchanged,
        },
        "research_roadmap_unchanged": roadmap_unchanged,
        "research_conductor_unchanged": conductor_unchanged,
        "research_roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "delegated_by_stop_rule": list(DELEGATED_BY_STOP_RULE),
        "read_only_ledgers_reviewed": [
            path for path in reviewed if bool(context_by_path.get(path, {}).get("exists"))
        ],
        "exclusion_manifest_update_required": True,
        "exclusion_manifest_updated": False,
        "arc_registry_update_required": False,
        "arc_registry_updated": False,
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "traceability_updated": False,
    }


def _e2e_check_receipts(validation_rows: Sequence[JsonMap]) -> list[JsonDict]:
    command_status = {str(row.get("command")): row for row in validation_rows}
    return [
        {
            "check_id": "E2E-001",
            "status": "non_applicable",
            "justification": "Capstone aggregation adds no Rust Ising training or sampling path.",
        },
        {
            "check_id": "E2E-002",
            "status": "non_applicable",
            "justification": "Capstone aggregation adds no Python/JAX Ising training path.",
        },
        {
            "check_id": "E2E-003",
            "status": "non_applicable",
            "justification": "No PyO3 binding behavior changed in Exp5647.",
        },
        {
            "check_id": "E2E-004",
            "status": "non_applicable",
            "justification": "No serialization cross-language surface changed in Exp5647.",
        },
        {
            "check_id": "E2E-005",
            "status": "non_applicable",
            "justification": "No packaged code-verification surface changed in Exp5647.",
        },
        {
            "check_id": "E2E-006",
            "status": "non_applicable",
            "justification": "No EBRM CPU/KV260 scoring path or hardware claim changed in Exp5647.",
        },
        {
            "check_id": "E2E-007",
            "status": "upstream_receipts_reconciled",
            "justification": "FR-11 evidence is reconciled from Exp5639 independent replay receipts.",
            "supporting_artifact": EXP5639_AUDIT_PATH.as_posix(),
        },
        {
            "check_id": "artifact_adversarial_verification",
            "status": str(
                command_status.get(
                    (
                        "python scripts/adversarial_verify.py "
                        "results/experiment_5647_v509_capstone_reconciliation.json"
                    ),
                    {},
                ).get("status", "not_run")
            ),
            "command": (
                "python scripts/adversarial_verify.py "
                "results/experiment_5647_v509_capstone_reconciliation.json"
            ),
        },
    ]


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, upstream_artifacts = read_artifacts(root)
    source_context, source_context_missing = _read_source_context(root)
    statuses = terminal_status_by_task(artifacts, metadata)
    missing = [rel for rel, meta in metadata.items() if not meta.get("exists")]
    malformed = [
        rel for rel, meta in metadata.items() if meta.get("exists") and not meta.get("loadable")
    ]

    fr11_schema = _derive_fr11_schema(artifacts)
    fr11_independent = _derive_fr11_independent(artifacts)
    fr11_shadow = _derive_fr11_shadow(artifacts)
    arc_status = _derive_arc_executable(artifacts)
    arc_solve = _derive_arc_solve(artifacts)
    two_axis = _derive_two_axis(artifacts)
    gate_statuses = _derive_gate_statuses(artifacts, statuses)
    adversarial_summary = _derive_adversarial_summary(artifacts, statuses)
    retirements = _derive_retirements(arc_status, arc_solve, two_axis)

    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    research_roadmap_unchanged = not roadmap_modified
    research_conductor_unchanged = not conductor_modified
    validation_rows = [dict(row) for row in (validation_results or DEFAULT_VALIDATION_RESULTS)]
    test_commands = [str(row.get("command", "")) for row in validation_rows]
    test_exit_codes = {str(row.get("command", "")): row.get("exit_code") for row in validation_rows}

    blocked = bool(
        missing
        or malformed
        or not research_roadmap_unchanged
        or not research_conductor_unchanged
    )
    honest_verdict = (
        "blocked: v509 capstone reconciliation incomplete because expected artifacts or protected-file checks failed"
        if blocked
        else (
            "complete: v509 reconciled; fr11_promoted="
            f"{bool(fr11_independent['promoted'])}; fr11_shadow_opt_in="
            f"{bool(fr11_shadow['ready'])}; arc_mechanism_promoted="
            f"{bool(arc_status['promoted'])}; arc_registry_delta="
            f"{arc_solve['registry_delta']}; two_axis_exact="
            f"{bool(two_axis['two_axis_invariant_status']['promoted'])}; "
            "two_axis_quality_promoted="
            f"{bool(two_axis['two_axis_quality_status']['promoted'])}; "
            f"rust_parity_promoted={bool(two_axis['rust_parity_status']['promoted'])}; "
            "timing_claimed=false; hardware_speedup_claimed=false"
        )
    )

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "artifact_metadata": metadata,
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "missing_artifacts": missing,
        "malformed_artifacts": malformed,
        "terminal_status_by_task": statuses,
        "upstream_artifacts": upstream_artifacts,
        "upstream_gate_statuses": gate_statuses,
        "adversarial_verification_summary": adversarial_summary,
        "fr11_schema_corrigendum_status": fr11_schema,
        "fr11_independent_promotion_status": fr11_independent,
        "fr11_shadow_integration_status": fr11_shadow,
        "arc_executable_model_status": arc_status,
        "arc_registry_count_before": arc_solve["registry_count_before"],
        "arc_registry_count_after": arc_solve["registry_count_after"],
        "arc_solve_provenance": arc_solve,
        "one_axis_replica_exchange_preserved": two_axis[
            "one_axis_replica_exchange_preserved"
        ],
        "two_axis_invariant_status": two_axis["two_axis_invariant_status"],
        "two_axis_quality_status": two_axis["two_axis_quality_status"],
        "rust_parity_status": two_axis["rust_parity_status"],
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "retirements_applied": retirements,
        "spec_reconciliation": _spec_reconciliation(),
        "ops_reconciliation": _ops_reconciliation(
            root,
            source_context,
            research_roadmap_unchanged,
            research_conductor_unchanged,
        ),
        "test_commands": test_commands,
        "test_exit_codes": test_exit_codes,
        "validation_results": validation_rows,
        "e2e_check_receipts": _e2e_check_receipts(validation_rows),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(field)
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(FIELD_PRINCIPLES):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append("field_principles")
                break
    for field in (
        "upstream_artifacts",
        "retirements_applied",
        "test_commands",
        "validation_results",
        "e2e_check_receipts",
    ):
        if not isinstance(payload.get(field), list):
            errors.append(field)
    if len(payload.get("upstream_artifacts", [])) != len(TASK_ARTIFACT_PATHS):
        errors.append("upstream_artifacts")
    for field in (
        "upstream_gate_statuses",
        "adversarial_verification_summary",
        "fr11_schema_corrigendum_status",
        "fr11_independent_promotion_status",
        "fr11_shadow_integration_status",
        "arc_executable_model_status",
        "arc_solve_provenance",
        "two_axis_invariant_status",
        "two_axis_quality_status",
        "rust_parity_status",
        "spec_reconciliation",
        "ops_reconciliation",
        "test_exit_codes",
        "terminal_status_by_task",
    ):
        if not isinstance(payload.get(field), Mapping):
            errors.append(field)
    statuses = payload.get("terminal_status_by_task")
    if isinstance(statuses, Mapping) and set(statuses) != set(EXPECTED_TASK_IDS):
        errors.append("terminal_status_by_task")
    gates = payload.get("upstream_gate_statuses")
    if isinstance(gates, Mapping):
        if set(gates) != set(EXPECTED_TASK_IDS):
            errors.append("upstream_gate_statuses")
        if gates.get("exp5642-arc-executable-model-live-ab", {}).get("status") not in {
            "gate_skipped",
            "missing",
            "malformed",
        }:
            errors.append("upstream_gate_statuses")
        if gates.get("exp5646-two-axis-tempering-rust-parity", {}).get("status") not in {
            "gate_skipped",
            "missing",
            "malformed",
        }:
            errors.append("upstream_gate_statuses")
    if not isinstance(payload.get("arc_registry_count_before"), int):
        errors.append("arc_registry_count_before")
    if not isinstance(payload.get("arc_registry_count_after"), int):
        errors.append("arc_registry_count_after")
    arc_solve = payload.get("arc_solve_provenance")
    if isinstance(arc_solve, Mapping):
        before = payload.get("arc_registry_count_before")
        after = payload.get("arc_registry_count_after")
        if isinstance(before, int) and isinstance(after, int):
            expected_delta = after - before
            if arc_solve.get("registry_delta") != expected_delta:
                errors.append("arc_solve_provenance")
        if arc_solve.get("solve_credited") is not False:
            errors.append("arc_solve_provenance")
    if payload.get("one_axis_replica_exchange_preserved") is not True:
        errors.append("one_axis_replica_exchange_preserved")
    if payload.get("timing_claimed") is not False:
        errors.append("timing_claimed")
    if payload.get("hardware_speedup_claimed") is not False:
        errors.append("hardware_speedup_claimed")
    schema_status = payload.get("fr11_schema_corrigendum_status")
    if isinstance(schema_status, Mapping) and schema_status.get("promoted_as_science") is not False:
        errors.append("fr11_schema_corrigendum_status")
    independent_status = payload.get("fr11_independent_promotion_status")
    if isinstance(independent_status, Mapping):
        no_bad_inputs = not payload.get("missing_artifacts") and not payload.get(
            "malformed_artifacts"
        )
        if no_bad_inputs and independent_status.get("promoted") is not True:
            errors.append("fr11_independent_promotion_status")
    shadow_status = payload.get("fr11_shadow_integration_status")
    if isinstance(shadow_status, Mapping):
        if shadow_status.get("default_enabled") is not False:
            errors.append("fr11_shadow_integration_status")
        if shadow_status.get("automatic_production_enablement") is not False:
            errors.append("fr11_shadow_integration_status")
    arc_status = payload.get("arc_executable_model_status")
    if isinstance(arc_status, Mapping) and arc_status.get("promoted") is not False:
        errors.append("arc_executable_model_status")
    exact_status = payload.get("two_axis_invariant_status")
    if isinstance(exact_status, Mapping):
        no_exact_input_error = EXP5644_TWO_AXIS_EXACT_PATH.as_posix() not in (
            payload.get("missing_artifacts", []) + payload.get("malformed_artifacts", [])
        )
        if no_exact_input_error and exact_status.get("promoted") is not True:
            errors.append("two_axis_invariant_status")
    quality_status = payload.get("two_axis_quality_status")
    if isinstance(quality_status, Mapping) and quality_status.get("promoted") is not False:
        errors.append("two_axis_quality_status")
    rust_status = payload.get("rust_parity_status")
    if isinstance(rust_status, Mapping) and rust_status.get("promoted") is not False:
        errors.append("rust_parity_status")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if not payload.get("reproducibility_checksum"):
        errors.append("reproducibility_checksum")
    return sorted(set(errors))


def write_capstone(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    payload = run_capstone(
        root=root,
        validation_results=validation_results,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(payload)
    if errors:
        raise ValueError(f"invalid Exp5647 capstone artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, payload)
    return payload


def _load_validation_results(path: Path | None) -> Sequence[JsonMap]:
    if path is None:
        return DEFAULT_VALIDATION_RESULTS
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("validation results must be a JSON list")
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validation-results", type=Path, default=None)
    args = parser.parse_args(argv)

    validation_results = _load_validation_results(args.validation_results)
    payload = run_capstone(root=args.root, validation_results=validation_results)
    errors = validate_artifact(payload)
    if errors:
        raise SystemExit(f"invalid Exp5647 capstone artifact fields: {', '.join(errors)}")
    output = args.output or args.root / RESULT_RELATIVE_PATH
    write_json(output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())

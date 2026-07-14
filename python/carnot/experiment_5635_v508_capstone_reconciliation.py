"""Exp5635 V508 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5635, SCENARIO-CAPSTONE-5635,
SCENARIO-CAPSTONE-5635-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5635-FIELD-PRINCIPLES.

This module is an evidence ledger, not a new scientific experiment. It reads
the Exp5625-Exp5634 result artifacts, separates gate skips and development
proxies from real promotions, and writes the exact capstone JSON that closes
milestone `.508` without reopening timing, hardware, or public-doc scopes.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5635_v508_capstone_reconciliation.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5635_v508_capstone_reconciliation"
EXPERIMENT_ID = "exp5635-v508-capstone-reconciliation"
MILESTONE = "2026.07.508"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5635
SCHEMA = "carnot.experiment_5635.v508_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-CAPSTONE-5635",
    "SCENARIO-CAPSTONE-5635",
    "SCENARIO-CAPSTONE-5635-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5635-FIELD-PRINCIPLES",
)

EXP5625_TRANSITION_PATH = Path("results/experiment_5625_transition_v508.json")
EXP5626_SOURCE_PATH = Path("results/experiment_5626_v508_source_delta_ingestion.json")
EXP5627_CONFORMAL_PATH = Path("results/experiment_5627_online_conformal_kan_qualification.json")
EXP5628_CSL_PATH = Path("results/experiment_5628_conformal_active_spline_kan_csl.json")
EXP5629_AUDIT_PATH = Path("results/experiment_5629_conformal_kan_independent_audit.json")
EXP5630_ARC_PROBE_PATH = Path("results/experiment_5630_arc_epistemic_object_probe_prototype.json")
EXP5631_ARC_AB_PATH = Path("results/experiment_5631_arc_epistemic_probe_live_ab.json")
EXP5632_ARC_LEVEL_PATH = Path("results/experiment_5632_arc_live_self_discovery_levelup_v508.json")
EXP5633_EXACT_PATH = Path("results/experiment_5633_temperature_exchange_cdls_exact_audit.json")
EXP5634_QUALITY_PATH = Path("results/experiment_5634_temperature_exchange_cdls_quality.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5625-transition-v508": EXP5625_TRANSITION_PATH,
    "exp5626-v508-source-delta-ingestion": EXP5626_SOURCE_PATH,
    "exp5627-online-conformal-kan-qualification": EXP5627_CONFORMAL_PATH,
    "exp5628-conformal-active-spline-kan-csl": EXP5628_CSL_PATH,
    "exp5629-conformal-kan-independent-audit": EXP5629_AUDIT_PATH,
    "exp5630-arc-epistemic-object-probe-prototype": EXP5630_ARC_PROBE_PATH,
    "exp5631-arc-epistemic-probe-live-ab": EXP5631_ARC_AB_PATH,
    "exp5632-arc-live-self-discovery-levelup-v508": EXP5632_ARC_LEVEL_PATH,
    "exp5633-temperature-exchange-cdls-exact-audit": EXP5633_EXACT_PATH,
    "exp5634-temperature-exchange-cdls-quality": EXP5634_QUALITY_PATH,
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
    "artifacts_expected": (
        "fixed .508 upstream denominator; must list Exp5625 through Exp5634 primary artifacts."
    ),
    "artifacts_read": "readable upstream evidence actually consumed by the capstone.",
    "gate_outcomes": "conductor gate skips are distinct from failures and successes.",
    "promotion_ledger": "only independently supported claims advance.",
    "retirement_ledger": (
        "repeated failures close mechanically while non-retired nulls stay bounded."
    ),
    "adversarial_flags": "critical issues remain visible after aggregation.",
    "continuous_self_learning_promotion": (
        "FR-11 status separates internal readiness from independent promotion."
    ),
    "arc_mechanism_promotion": (
        "development evidence stays separate from live mechanism promotion."
    ),
    "arc_registry_count_before": ("authoritative live-credit baseline before Exp5632 banking."),
    "arc_registry_count_after": ("authoritative live-credit total after Exp5632 banking."),
    "arc_registry_delta": "exactly 0 or 1 so live solve credit is auditable.",
    "replica_exchange_exact": ("invariant evidence is explicit and independent from quality."),
    "replica_exchange_quality_promoted": "quality evidence is separate from exactness.",
    "hardware_speedup_claimed": ("bare false keeps retired hardware-speedup scopes closed."),
    "timing_claimed": "bare false keeps retired timing/crossover scopes closed.",
    "documents_reconciled": (
        "internal specs and ops ledgers align or are explicitly delegated by stop rule."
    ),
    "validation_commands": "verification commands are reproducible.",
    "validation_results": "observed validation outcomes are recorded.",
    "research_roadmap_unchanged": (
        "protected-file discipline derived from git status or test override."
    ),
    "research_conductor_unchanged": (
        "protected-file discipline derived from git status or test override."
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
    "development_proxy_evidence",
    "flagged_tasks",
    "blocked_tasks",
    "gate_skipped_tasks",
    "complete_tasks",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_VALIDATION_RESULTS = (
    {
        "command": (
            ".venv/bin/pytest "
            "tests/python/test_experiment_5635_v508_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5635_v508_capstone_reconciliation.py -m pytest "
            "tests/python/test_experiment_5635_v508_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5635_v508_capstone_reconciliation.py --fail-under=100"
        ),
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
    read: list[JsonDict] = []
    path_to_task = {path: task_id for task_id, path in TASK_ARTIFACT_PATHS.items()}
    for rel_path in PRIMARY_ARTIFACT_PATHS:
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        metadata[rel] = meta
        payloads[rel] = payload
        if meta.get("exists") and meta.get("loadable"):
            read.append(
                {
                    "path": rel,
                    "task_id": path_to_task[rel_path],
                    "role": "primary_result",
                    "sha256": meta.get("sha256"),
                }
            )
    return payloads, metadata, read


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


def _zeroish(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "total" in value:
            return _int(value, "total") == 0
        return bool(value) and all(_zeroish(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return all(_zeroish(item) for item in value)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value == 0
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


def _bucket_tasks(statuses: Mapping[str, JsonMap], status: str) -> list[str]:
    return [task_id for task_id, row in statuses.items() if row.get("status") == status]


def _readiness_gate_all_true(payload: JsonMap) -> bool:
    receipt = payload.get("readiness_gate_receipt")
    return (
        isinstance(receipt, Mapping)
        and bool(receipt)
        and all(item is True for item in receipt.values())
    )


def _headline_coverage(payload: JsonMap, key: str) -> float:
    group = payload.get(key)
    if not isinstance(group, Mapping):
        return 0.0
    headline = group.get("group_conditional_online_conformal")
    if isinstance(headline, Mapping):
        heldout = headline.get("heldout")
        if isinstance(heldout, Mapping):
            return _number(heldout, "coverage")
        return _number(headline, "coverage")
    return 0.0


def _derive_csl(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    exp5625 = _payload(artifacts, EXP5625_TRANSITION_PATH)
    exp5627 = _payload(artifacts, EXP5627_CONFORMAL_PATH)
    exp5628 = _payload(artifacts, EXP5628_CSL_PATH)
    exp5629 = _payload(artifacts, EXP5629_AUDIT_PATH)

    exp5627_pass = bool(
        _clean_payload(exp5627)
        and _number(exp5627, "conformal_qualification_ready_score") == 1.0
        and _headline_coverage(exp5627, "marginal_coverage") >= 0.9
        and _headline_coverage(exp5627, "worst_group_coverage") >= 0.9
        and _zeroish(exp5627.get("exact_unsafe_accept_count"))
        and exp5627.get("leakage_control_pass") is True
    )
    exp5628_pass = bool(
        _clean_payload(exp5628)
        and exp5628.get("continuous_self_learning_ready") is True
        and _zeroish(exp5628.get("unsafe_false_accept_count"))
        and _int(exp5628, "llm_weight_updates") == 0
        and (
            _readiness_gate_all_true(exp5628)
            or bool(
                isinstance(exp5628.get("checkpoint_replay_exact"), Mapping)
                and exp5628.get("checkpoint_replay_exact", {}).get("passed") is True
            )
        )
    )
    exp5629_critical = (
        _max_severity(exp5629.get("critical_flags", exp5629.get("corrigendum_pending")))
        == "critical"
    )
    exp5629_certified = bool(
        _clean_payload(exp5629)
        and exp5629.get("independent_promotion_ready") is True
        and not exp5629_critical
    )

    if not exp5627_pass:
        failed_condition = "exp5627_conformal_qualification_not_ready"
    elif not exp5628_pass:
        failed_condition = "exp5628_internal_csl_not_ready"
    elif _is_gate_skip(exp5629):
        failed_condition = "exp5629_independent_audit_not_executed"
    elif exp5629_critical:
        failed_condition = "exp5629_independent_audit_critical_flag"
    elif not exp5629_certified:
        failed_condition = "exp5629_independent_promotion_ready_false"
    else:
        failed_condition = None

    promoted_substrate_text = json.dumps(exp5625.get("promoted_substrates", []), sort_keys=True)
    return {
        "exp5627_ready": exp5627_pass,
        "exp5628_internal_ready": exp5628_pass,
        "internal_ready": exp5627_pass and exp5628_pass,
        "independent_certified": exp5629_certified,
        "promoted": exp5627_pass and exp5628_pass and exp5629_certified,
        "failed_condition": failed_condition,
        "exact_fixture_preserved": "exact" in promoted_substrate_text
        or "constraint_stream" in promoted_substrate_text,
        "exp5629_gate_summary": exp5629.get("gate_check_summary"),
    }


def _derive_arc(artifacts: Mapping[str, JsonMap]) -> tuple[JsonDict, JsonDict]:
    exp5630 = _payload(artifacts, EXP5630_ARC_PROBE_PATH)
    exp5631 = _payload(artifacts, EXP5631_ARC_AB_PATH)
    exp5632 = _payload(artifacts, EXP5632_ARC_LEVEL_PATH)

    exp5630_nondegenerate_safe = bool(
        exp5630.get("solve_provenance") == "development_proxy"
        and _int(exp5630, "object_hypothesis_non_degenerate_count") >= 3
        and _int(exp5630, "unsafe_model_accept_count") == 0
        and exp5630.get("per_game_adapter_used") is not True
        and exp5630.get("outer_loop_recipes_used") is not True
        and exp5630.get("exhaustive_bfs_used") is not True
    )
    exp5631_ready = bool(
        _clean_payload(exp5631)
        and exp5631.get("live_epistemic_policy_ready") is True
        and _int(exp5631, "unsafe_model_accept_count") == 0
        and _int(exp5631, "known_level_regression_count") == 0
        and bool(exp5631.get("downstream_benefit"))
    )
    mechanism_promoted = exp5630_nondegenerate_safe and exp5631_ready

    registry_before = _int(exp5632, "registry_count_before")
    registry_after = _int(exp5632, "registry_count_after")
    registry_delta = _int(exp5632, "registry_delta")
    if registry_delta not in (0, 1):
        registry_delta = 0
    live_credit = bool(
        exp5632.get("solve_provenance") == "live_agent_self_discovery"
        and exp5632.get("offline_reproduced") is True
        and registry_delta == 1
        and registry_after - registry_before == 1
        and exp5632.get("source_read") is not True
        and exp5632.get("game_adapter_used") is not True
        and exp5632.get("outer_loop_re_used") is not True
    )
    return (
        {
            "exp5630_development_proxy": exp5630.get("solve_provenance") == "development_proxy",
            "exp5630_nondegenerate_safe": exp5630_nondegenerate_safe,
            "exp5630_ready_score": _number(exp5630, "epistemic_probe_ready_score"),
            "exp5631_live_ab_ready": exp5631_ready,
            "exp5631_gate_summary": exp5631.get("gate_check_summary"),
            "promoted": mechanism_promoted,
            "failed_condition": None
            if mechanism_promoted
            else (
                "exp5631_not_executed_or_not_ready"
                if exp5630_nondegenerate_safe
                else "exp5630_development_proxy_not_ready"
            ),
        },
        {
            "solve_provenance": exp5632.get("solve_provenance"),
            "live_attempt_executed": bool(exp5632.get("live_attempt_executed")),
            "offline_reproduced": bool(exp5632.get("offline_reproduced")),
            "registry_count_before": registry_before,
            "registry_count_after": registry_after,
            "registry_delta": registry_delta,
            "new_reproducible_levels": exp5632.get("new_reproducible_levels", []),
            "promoted": live_credit,
            "failed_condition": None
            if live_credit
            else "exp5632_no_independent_reproduction_or_registry_delta_zero",
        },
    )


def _derive_replica_exchange(artifacts: Mapping[str, JsonMap]) -> tuple[JsonDict, JsonDict]:
    exp5633 = _payload(artifacts, EXP5633_EXACT_PATH)
    exp5634 = _payload(artifacts, EXP5634_QUALITY_PATH)
    broken = exp5633.get("broken_controls")
    broken_controls_detected = bool(
        isinstance(broken, Sequence)
        and not isinstance(broken, str | bytes | bytearray)
        and broken
        and all(isinstance(row, Mapping) and row.get("detected") is True for row in broken)
    )
    exact_promoted = bool(
        _clean_payload(exp5633)
        and _number(exp5633, "replica_exchange_kernel_ready_score") == 1.0
        and _number(exp5633, "exact_distribution_tv_max") <= 0.02
        and _number(exp5633, "swap_detailed_balance_residual_max") <= 1e-6
        and exp5633.get("validity_regression_detected") is not True
        and exp5633.get("timing_claimed") is False
        and exp5633.get("hardware_speedup_claimed") is False
        and exp5633.get("deterministic_replay_pass") is True
        and broken_controls_detected
    )
    quality_promoted = bool(
        exact_promoted
        and _clean_payload(exp5634)
        and exp5634.get("quality_mixing_ready") is True
        and exp5634.get("target_diagnostics_within_exp5633_bounds") is True
        and exp5634.get("timing_claimed") is False
        and exp5634.get("hardware_speedup_claimed") is False
    )
    return (
        {
            "promoted": exact_promoted,
            "replica_exchange_kernel_ready_score": _number(
                exp5633, "replica_exchange_kernel_ready_score"
            ),
            "exact_distribution_tv_max": _number(exp5633, "exact_distribution_tv_max"),
            "swap_detailed_balance_residual_max": _number(
                exp5633, "swap_detailed_balance_residual_max"
            ),
            "validity_regression_detected": bool(exp5633.get("validity_regression_detected")),
            "timing_claimed": bool(exp5633.get("timing_claimed")),
            "hardware_speedup_claimed": bool(exp5633.get("hardware_speedup_claimed")),
        },
        {
            "promoted": quality_promoted,
            "quality_mixing_ready": bool(exp5634.get("quality_mixing_ready")),
            "target_diagnostics_within_exp5633_bounds": bool(
                exp5634.get("target_diagnostics_within_exp5633_bounds")
            ),
            "wall_time_provenance_only": exp5634.get("wall_time_provenance_only", {}),
            "timing_claimed": bool(exp5634.get("timing_claimed")),
            "hardware_speedup_claimed": bool(exp5634.get("hardware_speedup_claimed")),
        },
    )


def _derive_gate_outcomes(
    artifacts: Mapping[str, JsonMap], statuses: Mapping[str, JsonMap]
) -> dict[str, JsonDict]:
    outcomes: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, rel_path)
        if _is_gate_skip(payload) or payload.get("gates_evaluated") is not None:
            outcomes[task_id] = {
                "status": statuses[task_id]["status"],
                "artifact_path": rel_path.as_posix(),
                "gate_check_summary": payload.get("gate_check_summary"),
                "gates_evaluated": payload.get("gates_evaluated", []),
                "skipped_work_is_failure": False,
                "skipped_work_is_success": False,
            }
    return outcomes


def _derive_adversarial_flags(
    artifacts: Mapping[str, JsonMap], statuses: Mapping[str, JsonMap]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, rel_path)
        flags = payload.get("corrigendum_pending", [])
        max_severity = _max_severity(flags)
        if payload.get("flagged_adversarial") or max_severity != "none":
            rows.append(
                {
                    "task_id": task_id,
                    "artifact_path": rel_path.as_posix(),
                    "status": statuses[task_id]["status"],
                    "flagged_adversarial": bool(payload.get("flagged_adversarial")),
                    "max_severity": max_severity,
                    "flags": flags if isinstance(flags, list) else [],
                }
            )
    return rows


def _derive_retirement_ledger(
    csl: JsonMap,
    arc_mechanism: JsonMap,
    arc_solve: JsonMap,
    replica_exact: JsonMap,
    replica_quality: JsonMap,
) -> list[JsonDict]:
    return [
        {
            "scope": "native_runtime_certificate",
            "decision": "closed_upstream_preserved",
            "mechanically_required_manifest_update": False,
            "reason": "Exp5625 preserved the .507 native runtime closure; this capstone did not reopen it.",
        },
        {
            "scope": "cdls_timing_crossover_and_hardware_speedup",
            "decision": "retired_scope_preserved",
            "mechanically_required_manifest_update": False,
            "reason": "Exp5633 and Exp5634 claim exactness/quality only; timing and hardware remain false.",
        },
        {
            "scope": "fr11_conformal_kan_independent_promotion",
            "decision": "not_promoted_not_retired",
            "mechanically_required_manifest_update": False,
            "reason": str(csl.get("failed_condition") or "independently promoted"),
        },
        {
            "scope": "arc_epistemic_object_mechanism",
            "decision": "development_proxy_or_gate_skip_not_promoted",
            "mechanically_required_manifest_update": False,
            "reason": str(arc_mechanism.get("failed_condition") or "promoted"),
        },
        {
            "scope": "arc_live_levelup_credit",
            "decision": "bounded_null_no_broad_retirement",
            "mechanically_required_manifest_update": False,
            "reason": str(arc_solve.get("failed_condition") or "new level banked"),
        },
        {
            "scope": "replica_exchange_quality",
            "decision": "promoted" if replica_quality.get("promoted") else "not_promoted",
            "mechanically_required_manifest_update": False,
            "reason": "paired quality/mixing gate passed"
            if replica_quality.get("promoted")
            else "quality gate did not pass",
        },
        {
            "scope": "replica_exchange_exact",
            "decision": "promoted" if replica_exact.get("promoted") else "not_promoted",
            "mechanically_required_manifest_update": False,
            "reason": "invariant parity gate passed"
            if replica_exact.get("promoted")
            else "invariant parity gate did not pass",
        },
    ]


def _documents_reconciled(
    root: Path,
    source_context: Sequence[JsonMap],
    roadmap_unchanged: bool,
    conductor_unchanged: bool,
) -> JsonDict:
    read_only_paths = [
        "research-complete.yaml",
        "research-references.md",
        "ops/exclusion_manifest.yaml",
        "ops/arc_solve_registry.yaml",
        "ops/conductor-log.md",
        "ops/e2e-test-plan.md",
    ]
    context_by_path = {str(row.get("path")): row for row in source_context}
    return {
        "protected_files": {
            ROADMAP_RELATIVE_PATH.as_posix(): roadmap_unchanged,
            CONDUCTOR_RELATIVE_PATH.as_posix(): conductor_unchanged,
        },
        "delegated_by_stop_rule": list(DELEGATED_BY_STOP_RULE),
        "read_only_ledgers_reviewed": [
            path for path in read_only_paths if bool(context_by_path.get(path, {}).get("exists"))
        ],
        "research_roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "public_docs_modified": False,
        "exclusion_manifest_update_required": False,
        "arc_registry_update_required": False,
    }


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, artifacts_read = read_artifacts(root)
    source_context, source_context_missing = _read_source_context(root)
    statuses = terminal_status_by_task(artifacts, metadata)
    missing = [rel for rel, meta in metadata.items() if not meta.get("exists")]
    malformed = [
        rel for rel, meta in metadata.items() if meta.get("exists") and not meta.get("loadable")
    ]

    csl = _derive_csl(artifacts)
    arc_mechanism, arc_solve = _derive_arc(artifacts)
    replica_exact, replica_quality = _derive_replica_exchange(artifacts)
    gate_outcomes = _derive_gate_outcomes(artifacts, statuses)
    adversarial_flags = _derive_adversarial_flags(artifacts, statuses)
    retirement_ledger = _derive_retirement_ledger(
        csl, arc_mechanism, arc_solve, replica_exact, replica_quality
    )

    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    research_roadmap_unchanged = not roadmap_modified
    research_conductor_unchanged = not conductor_modified
    validation_rows = list(validation_results or DEFAULT_VALIDATION_RESULTS)
    validation_commands = [str(row.get("command", "")) for row in validation_rows]

    promotion_ledger = {
        "fr11_conformal_kan": {
            "promoted": bool(csl["promoted"]),
            "evidence": ["exp5627", "exp5628", "exp5629"],
            "failed_condition": csl.get("failed_condition"),
        },
        "arc_epistemic_mechanism": {
            "promoted": bool(arc_mechanism["promoted"]),
            "evidence": ["exp5630", "exp5631"],
            "failed_condition": arc_mechanism.get("failed_condition"),
        },
        "arc_live_solve_credit": {
            "promoted": bool(arc_solve["promoted"]),
            "evidence": ["exp5632"],
            "failed_condition": arc_solve.get("failed_condition"),
        },
        "replica_exchange_exact": {
            "promoted": bool(replica_exact["promoted"]),
            "evidence": ["exp5633"],
        },
        "replica_exchange_quality": {
            "promoted": bool(replica_quality["promoted"]),
            "evidence": ["exp5633", "exp5634"],
        },
    }

    documents_reconciled = _documents_reconciled(
        root, source_context, research_roadmap_unchanged, research_conductor_unchanged
    )
    blocked = bool(
        missing or malformed or not research_roadmap_unchanged or not research_conductor_unchanged
    )
    honest_verdict = (
        "blocked: v508 capstone reconciliation incomplete because expected artifacts or protected-file checks failed"
        if blocked
        else (
            "complete: v508 capstone reconciled; fr11_promoted="
            f"{bool(csl['promoted'])}; arc_registry_delta={arc_solve['registry_delta']}; "
            f"replica_exchange_quality_promoted={bool(replica_quality['promoted'])}; "
            "hardware_speedup_claimed=false; timing_claimed=false"
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
        "development_proxy_evidence": [
            task_id for task_id, row in statuses.items() if row.get("development_proxy")
        ],
        "flagged_tasks": _bucket_tasks(statuses, "flagged"),
        "blocked_tasks": _bucket_tasks(statuses, "blocked"),
        "gate_skipped_tasks": _bucket_tasks(statuses, "gate_skipped"),
        "complete_tasks": _bucket_tasks(statuses, "complete"),
        "artifacts_expected": [
            {"task_id": task_id, "path": path.as_posix()}
            for task_id, path in TASK_ARTIFACT_PATHS.items()
        ],
        "artifacts_read": artifacts_read,
        "gate_outcomes": gate_outcomes,
        "promotion_ledger": promotion_ledger,
        "retirement_ledger": retirement_ledger,
        "adversarial_flags": adversarial_flags,
        "continuous_self_learning_promotion": csl,
        "arc_mechanism_promotion": arc_mechanism,
        "arc_registry_count_before": arc_solve["registry_count_before"],
        "arc_registry_count_after": arc_solve["registry_count_after"],
        "arc_registry_delta": arc_solve["registry_delta"],
        "replica_exchange_exact": replica_exact,
        "replica_exchange_quality_promoted": bool(replica_quality["promoted"]),
        "replica_exchange_quality_evidence": replica_quality,
        "hardware_speedup_claimed": False,
        "timing_claimed": False,
        "documents_reconciled": documents_reconciled,
        "validation_commands": validation_commands,
        "validation_results": validation_rows,
        "research_roadmap_unchanged": research_roadmap_unchanged,
        "research_conductor_unchanged": research_conductor_unchanged,
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
        "artifacts_expected",
        "artifacts_read",
        "retirement_ledger",
        "adversarial_flags",
        "validation_commands",
        "validation_results",
    ):
        if not isinstance(payload.get(field), list):
            errors.append(field)
    if len(payload.get("artifacts_expected", [])) != len(TASK_ARTIFACT_PATHS):
        errors.append("artifacts_expected")
    gate_outcomes = payload.get("gate_outcomes")
    if (
        not isinstance(gate_outcomes, Mapping)
        or "exp5629-conformal-kan-independent-audit" not in gate_outcomes
        or "exp5631-arc-epistemic-probe-live-ab" not in gate_outcomes
    ):
        errors.append("gate_outcomes")
    for field in (
        "promotion_ledger",
        "continuous_self_learning_promotion",
        "arc_mechanism_promotion",
        "replica_exchange_exact",
        "documents_reconciled",
        "terminal_status_by_task",
    ):
        if not isinstance(payload.get(field), Mapping):
            errors.append(field)
    statuses = payload.get("terminal_status_by_task")
    if isinstance(statuses, Mapping) and set(statuses) != set(EXPECTED_TASK_IDS):
        errors.append("terminal_status_by_task")
    if payload.get("arc_registry_delta") not in (0, 1):
        errors.append("arc_registry_delta")
    if not isinstance(payload.get("arc_registry_count_before"), int):
        errors.append("arc_registry_count_before")
    if not isinstance(payload.get("arc_registry_count_after"), int):
        errors.append("arc_registry_count_after")
    for field in (
        "replica_exchange_quality_promoted",
        "hardware_speedup_claimed",
        "timing_claimed",
        "research_roadmap_unchanged",
        "research_conductor_unchanged",
    ):
        if not isinstance(payload.get(field), bool):
            errors.append(field)
    if payload.get("hardware_speedup_claimed") is not False:
        errors.append("hardware_speedup_claimed")
    if payload.get("timing_claimed") is not False:
        errors.append("timing_claimed")
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
        raise ValueError(f"invalid Exp5635 capstone artifact fields: {', '.join(errors)}")
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
        raise SystemExit(f"invalid Exp5635 capstone artifact fields: {', '.join(errors)}")
    output = args.output or args.root / RESULT_RELATIVE_PATH
    write_json(output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())

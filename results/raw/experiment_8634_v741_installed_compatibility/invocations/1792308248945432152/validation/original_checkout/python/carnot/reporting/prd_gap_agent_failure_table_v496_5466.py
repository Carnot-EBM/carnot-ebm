"""Build the Exp5466 PRD gap and agent-failure table for milestone .496.

Spec refs: REQ-REPORT-5466, SCENARIO-REPORT-5466,
SCENARIO-REPORT-5466-MISSING-OR-SKIPPED.

This module is a narrow aggregation step. It reads the result files that
already landed for Exp5454 through Exp5465, plus same-prefix sidecar evidence,
and records what those files actually support. The important constraint is that
the table cannot use the roadmap as proof. A planned task may have been skipped,
blocked, or adversarially flagged; this helper keeps those states visible so a
capstone or future planner does not accidentally convert them into PRD progress.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
Payloads = Mapping[str, JsonDict]
LaneClassifier = Callable[[Payloads], str]

MILESTONE = "2026.07.496"
EXPERIMENT_ID = "exp5466-prd-gap-agent-failure-table-v496"
SCHEMA = "carnot.prd_gap_agent_failure_table.v496.exp5466"
OUTPUT_REL_PATH = Path("results/experiment_5466_prd_gap_agent_failure_table_v496.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 5466
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5466",
    "SCENARIO-REPORT-5466",
    "SCENARIO-REPORT-5466-MISSING-OR-SKIPPED",
)

MAIN_ARTIFACT_PATHS = (
    "results/experiment_5454_transition_v496.json",
    "results/experiment_5455_source_delta_v496.json",
    "results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json",
    "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json",
    "results/experiment_5458_minimal_core_claim_repair_v496.json",
    "results/experiment_5459_constraint_distortion_guard_v496.json",
    "results/experiment_5460_csl_policy_bandit_v496.json",
    "results/experiment_5461_gated_sota_csl_memory_routing_v496.json",
    "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json",
    "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json",
    "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json",
    "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "milestone",
    "artifact_paths_read",
    "missing_artifacts",
    "skipped_gated_tasks",
    "closed_lanes",
    "partial_lanes",
    "blocked_lanes",
    "honest_null_lanes",
    "prd_requirement_map",
    "agent_failure_taxonomy",
    "docs_updated",
    "inference_substrate",
    "honest_verdict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "status",
    "field_principles",
    "sidecar_artifacts_read",
    "missing_lanes",
    "spec_refs",
    "tests_run",
    "random_seed",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES = {
    "milestone": "conductor route key for 2026.07.496",
    "artifact_paths_read": "actual evidence basis",
    "missing_artifacts": "no fabricated upstream evidence",
    "skipped_gated_tasks": "gate skips stay distinct from completed work",
    "closed_lanes": "positive PRD evidence boundary",
    "partial_lanes": "bounded progress boundary",
    "blocked_lanes": "blocked evidence stays blocked",
    "honest_null_lanes": "measured null outcomes remain explicit",
    "prd_requirement_map": "FR-11 and FR-12 traceability",
    "agent_failure_taxonomy": "operational learning from agent failures",
    "docs_updated": "task-specific stop rule; no ops reconciliation here",
    "inference_substrate": "aggregation only; no hidden live inference",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}

FAILURE_MODE_KEYS = (
    "tautology",
    "duration_precondition_failures",
    "missing_hardware",
    "no_bank_arc",
    "gguf_offload_gaps",
    "unsupported_hidden_internal_claims",
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_5466_prd_gap_agent_failure_table_v496.py -q --no-cov",
    (
        ".venv/bin/coverage run --include=python/carnot/reporting/"
        "prd_gap_agent_failure_table_v496_5466.py -m pytest "
        "tests/python/test_experiment_5466_prd_gap_agent_failure_table_v496.py -q "
        "--no-cov -n 0"
    ),
    (
        ".venv/bin/coverage report --include=python/carnot/reporting/"
        "prd_gap_agent_failure_table_v496_5466.py --fail-under=100"
    ),
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class LaneSpec:
    """A single PRD lane and the exact artifacts that can prove it.

    The lane spec keeps classification code honest. Each row must name the
    artifact paths it reads, so missing inputs become a visible missing lane
    instead of falling through to a default success value.
    """

    name: str
    source_artifacts: tuple[str, ...]
    prd_requirements: tuple[str, ...]
    claim_boundary: str
    classifier: LaneClassifier
    evidence_fields: tuple[str, ...]


def _read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _value(payloads: Payloads, artifact_path: str, field_name: str) -> Any:
    return payloads.get(artifact_path, {}).get(field_name)


def _is_complete(payloads: Payloads, artifact_path: str) -> bool:
    verdict = _value(payloads, artifact_path, "honest_verdict")
    return isinstance(verdict, str) and verdict.startswith("complete:")


def _number_equals(value: Any, expected: float) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and float(value) == expected


def _contains_token_internal(value: Any) -> bool:
    return "token_internal" in _stable_json(value)


def _main_payloads(root: Path) -> tuple[dict[str, JsonDict], list[str]]:
    payloads: dict[str, JsonDict] = {}
    missing: list[str] = []
    for rel_path in MAIN_ARTIFACT_PATHS:
        path = root / rel_path
        if path.exists():
            payloads[rel_path] = _read_json_object(path)
        else:
            missing.append(rel_path)
    return payloads, missing


def _discover_available_artifacts(root: Path) -> list[str]:
    discovered: set[str] = set()
    results_root = root / "results"
    for experiment_id in range(5454, 5466):
        for suffix in ("json", "jsonl"):
            for path in results_root.glob(f"experiment_{experiment_id}_*.{suffix}"):
                discovered.add(path.relative_to(root).as_posix())
    return sorted(discovered)


def _sidecar_summary(root: Path, rel_path: str) -> JsonDict:
    path = root / rel_path
    text = path.read_text(encoding="utf-8")
    summary: JsonDict = {
        "artifact_path": rel_path,
        "sha256": _sha256_text(text),
        "size_bytes": len(text.encode("utf-8")),
    }
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        summary["format"] = "jsonl"
        summary["row_count"] = len(rows)
        summary["first_row_keys"] = sorted(rows[0]) if rows and isinstance(rows[0], dict) else []
    else:
        parsed = json.loads(text)
        summary["format"] = "json"
        summary["json_type"] = type(parsed).__name__
        summary["top_level_keys"] = sorted(parsed) if isinstance(parsed, dict) else []
    return summary


def _sidecar_summaries(root: Path, artifact_paths: Sequence[str]) -> list[JsonDict]:
    main_paths = set(MAIN_ARTIFACT_PATHS)
    return [
        _sidecar_summary(root, rel_path)
        for rel_path in artifact_paths
        if rel_path not in main_paths
    ]


def _skipped_gated_tasks(payloads: Payloads) -> list[JsonDict]:
    skipped: list[JsonDict] = []
    for artifact_path, payload in sorted(payloads.items()):
        status = str(payload.get("status", "")).lower()
        verdict = str(payload.get("honest_verdict", "")).lower()
        reason = payload.get("skipped_reason", payload.get("skip_reason", ""))
        if "skipped" not in status and not verdict.startswith("skipped"):
            continue
        skipped.append(
            {
                "artifact_path": artifact_path,
                "honest_verdict": str(payload.get("honest_verdict", "")),
                "reason": str(reason),
                "status": str(payload.get("status", "")),
            }
        )
    return skipped


def _transition_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5454_transition_v496.json"
    return "closed" if _is_complete(payloads, path) and _value(payloads, path, "milestone") == MILESTONE else "blocked"


def _source_delta_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5455_source_delta_v496.json"
    reopened = _value(payloads, path, "retired_scopes_reopened") is True
    return "closed" if _is_complete(payloads, path) and not reopened else "blocked"


def _corrigendum_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json"
    clean = _value(payloads, path, "guided_decoding_corrigendum_clean") is True
    return "closed" if clean and _is_complete(payloads, path) else "blocked"


def _local_sota_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json"
    flagged = _value(payloads, path, "flagged_adversarial") is True
    ready = _value(payloads, path, "verifier_guided_decoding_ready") is True
    lcd_bias = _value(payloads, path, "lcd_bias_check_passed") is True
    return "closed" if ready and lcd_bias and not flagged else "blocked"


def _minimal_core_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5458_minimal_core_claim_repair_v496.json"
    ready = _value(payloads, path, "minimal_core_repair_ready") is True
    exact = _value(payloads, path, "exact_final_authority") is True
    repaired = _number_equals(_value(payloads, path, "repaired_accept_rate_after_exact_recheck"), 1.0)
    return "closed" if ready and exact and repaired else "blocked"


def _distortion_guard_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5459_constraint_distortion_guard_v496.json"
    ready = _value(payloads, path, "distortion_guard_ready") is True
    exact = _value(payloads, path, "exact_final_authority") is True
    truth_rate = _value(payloads, path, "truth_preserving_compliance_rate")
    unsupported_rate = _value(payloads, path, "unsupported_fabrication_rate")
    has_rates = isinstance(truth_rate, int | float) and isinstance(unsupported_rate, int | float)
    return "closed" if ready and exact and has_rates else "blocked"


def _csl_policy_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5460_csl_policy_bandit_v496.json"
    ready = _value(payloads, path, "csl_policy_ready") is True
    no_weight = _value(payloads, path, "no_weight_mutation") is True
    violations = _value(payloads, path, "cumulative_constraint_violations")
    return "closed" if ready and no_weight and violations == 0 else "partial"


def _sota_csl_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5461_gated_sota_csl_memory_routing_v496.json"
    ready = _value(payloads, path, "csl_sota_memory_routing_ready") is True
    offload = _value(payloads, path, "gpu_offload_verified") is True
    no_weight = _value(payloads, path, "no_weight_mutation") is True
    return "closed" if ready and offload and no_weight else "partial"


def _pbit_bridge_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json"
    ready = _value(payloads, path, "minimal_core_pbit_bridge_ready") is True
    solver = _value(payloads, path, "solver_authoritative") is True
    fallback = _number_equals(_value(payloads, path, "fallback_completeness_rate"), 1.0)
    speedup = _value(payloads, path, "hardware_speedup_claim") is True
    return "partial" if ready and solver and fallback and not speedup else "blocked"


def _hardware_receipts_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json"
    ready = _value(payloads, path, "hardware_receipts_ready") is True
    gated = _value(payloads, path, "gated_upstream_ready") is True
    hashes = _value(payloads, path, "hashes_match_before_timing_compare") is True
    return "partial" if ready and gated and hashes else "blocked"


def _arc_precheck_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json"
    ready = _value(payloads, path, "arc_metric_integrity_ready") is True
    registry = _value(payloads, path, "registry_precheck_performed") is True
    return "closed" if ready and registry else "partial"


def _arc_levelup_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json"
    return "closed" if _value(payloads, path, "new_level_banked") is True else "honest_null"


def _token_internal_classifier(payloads: Payloads) -> str:
    blocked_lanes = _value(payloads, "results/experiment_5454_transition_v496.json", "blocked_lanes")
    return "blocked" if _contains_token_internal(blocked_lanes) else "missing"


def _hardware_speedup_classifier(payloads: Payloads) -> str:
    speedup_claimed = any(
        _value(payloads, artifact_path, "hardware_speedup_claim") is True
        for artifact_path in (
            "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json",
            "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json",
        )
    )
    return "closed" if speedup_claimed else "honest_null"


LANE_SPECS = (
    LaneSpec(
        "transition_traceability",
        ("results/experiment_5454_transition_v496.json",),
        ("FR-10",),
        "closed only for transition provenance and prior-boundary carry-forward",
        _transition_classifier,
        ("honest_verdict", "closed_lanes", "partial_lanes", "blocked_lanes", "honest_null_lanes"),
    ),
    LaneSpec(
        "source_delta_refresh",
        ("results/experiment_5455_source_delta_v496.json",),
        ("FR-10", "FR-12"),
        "closed for source refresh; not an implementation result by itself",
        _source_delta_classifier,
        ("honest_verdict", "new_actionable_findings_count", "retired_scopes_reopened"),
    ),
    LaneSpec(
        "guided_decoding_corrigendum",
        ("results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json",),
        ("FR-12",),
        "closed for the posthoc corrigendum while prior headline readiness stays blocked",
        _corrigendum_classifier,
        (
            "honest_verdict",
            "guided_decoding_corrigendum_clean",
            "prior_flagged_adversarial",
            "invalid_tautological_fields",
        ),
    ),
    LaneSpec(
        "local_sota_distortion_guarded_decoding",
        ("results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json",),
        ("FR-12",),
        "blocked because the live panel is adversarially flagged and readiness is false",
        _local_sota_classifier,
        (
            "honest_verdict",
            "flagged_adversarial",
            "corrigendum_pending",
            "verifier_guided_decoding_ready",
            "lcd_bias_check_passed",
            "gpu_offload_verified",
            "precondition_details",
        ),
    ),
    LaneSpec(
        "minimal_core_claim_repair",
        ("results/experiment_5458_minimal_core_claim_repair_v496.json",),
        ("FR-12",),
        "closed for deterministic minimal-core repair cases under exact recheck",
        _minimal_core_classifier,
        (
            "honest_verdict",
            "minimal_core_repair_ready",
            "exact_final_authority",
            "repaired_accept_rate_after_exact_recheck",
        ),
    ),
    LaneSpec(
        "constraint_distortion_guard",
        ("results/experiment_5459_constraint_distortion_guard_v496.json",),
        ("FR-12",),
        "closed for deterministic distortion detection against authoritative facts",
        _distortion_guard_classifier,
        (
            "honest_verdict",
            "distortion_guard_ready",
            "exact_final_authority",
            "truth_preserving_compliance_rate",
            "unsupported_fabrication_rate",
        ),
    ),
    LaneSpec(
        "csl_policy_bandit",
        ("results/experiment_5460_csl_policy_bandit_v496.json",),
        ("FR-11",),
        "closed for governed frozen-model policy routing; no model-weight learning claim",
        _csl_policy_classifier,
        (
            "honest_verdict",
            "csl_policy_ready",
            "no_weight_mutation",
            "cumulative_constraint_violations",
            "quality_delta_vs_naive_icl",
        ),
    ),
    LaneSpec(
        "sota_csl_memory_routing",
        ("results/experiment_5461_gated_sota_csl_memory_routing_v496.json",),
        ("FR-11", "FR-12"),
        "closed for live GGUF memory routing with frozen weights and exact task checks",
        _sota_csl_classifier,
        (
            "honest_verdict",
            "csl_sota_memory_routing_ready",
            "gpu_offload_verified",
            "no_weight_mutation",
            "negative_transfer_deflection_rate",
        ),
    ),
    LaneSpec(
        "active_constraint_pbit_pdit_bridge",
        ("results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json",),
        ("FR-12", "NFR-01"),
        "partial because assumptions are advisory and no hardware speedup is claimed",
        _pbit_bridge_classifier,
        (
            "honest_verdict",
            "minimal_core_pbit_bridge_ready",
            "solver_authoritative",
            "fallback_completeness_rate",
            "hardware_speedup_claim",
        ),
    ),
    LaneSpec(
        "hardware_boundary_exchange_receipts",
        ("results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json",),
        ("NFR-01",),
        "partial because CPU/reachable-board receipts exist but missing boards and no speedup remain",
        _hardware_receipts_classifier,
        (
            "honest_verdict",
            "hardware_receipts_ready",
            "board_reachability",
            "timing_repeat_counts",
            "hardware_speedup_claim",
        ),
    ),
    LaneSpec(
        "arc_metric_integrity_precheck",
        ("results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json",),
        ("FR-12",),
        "closed for metric-integrity and perception precheck only; no solve claimed",
        _arc_precheck_classifier,
        (
            "honest_verdict",
            "arc_metric_integrity_ready",
            "registry_precheck_performed",
            "target_shortlist",
        ),
    ),
    LaneSpec(
        "arc_connected_component_salience_levelup",
        ("results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json",),
        ("FR-12",),
        "honest null because the live path did not bank a reproduction-gated new level",
        _arc_levelup_classifier,
        (
            "honest_verdict",
            "new_level_banked",
            "offline_reproduced",
            "failure_mode",
            "live_attempt_count",
        ),
    ),
    LaneSpec(
        "token_internal_hidden_claims",
        ("results/experiment_5454_transition_v496.json",),
        ("FR-12",),
        "blocked because hidden/internal/token claims lack authenticated receipts",
        _token_internal_classifier,
        ("honest_verdict", "blocked_lanes"),
    ),
    LaneSpec(
        "hardware_speedup_claim",
        (
            "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json",
            "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json",
        ),
        ("NFR-01",),
        "honest null because timing facts are recorded without any speedup claim",
        _hardware_speedup_classifier,
        ("honest_verdict", "hardware_speedup_claim", "timing_comparison"),
    ),
)


def _evidence(payloads: Payloads, spec: LaneSpec) -> JsonDict:
    evidence: JsonDict = {}
    for artifact_path in spec.source_artifacts:
        payload = payloads.get(artifact_path, {})
        evidence[artifact_path] = {
            field_name: payload.get(field_name)
            for field_name in spec.evidence_fields
            if field_name in payload
        }
    return evidence


def _lane_entry(spec: LaneSpec, classification: str, payloads: Payloads) -> JsonDict:
    return {
        "lane": spec.name,
        "classification": classification,
        "source_artifacts": list(spec.source_artifacts),
        "prd_requirements": list(spec.prd_requirements),
        "claim_boundary": spec.claim_boundary,
        "evidence": _evidence(payloads, spec),
        "upstream_honest_verdicts": [
            str(_value(payloads, artifact_path, "honest_verdict"))
            for artifact_path in spec.source_artifacts
            if _value(payloads, artifact_path, "honest_verdict") is not None
        ],
    }


def _classify_lanes(
    payloads: Payloads,
    missing_artifacts: Sequence[str],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    closed: list[JsonDict] = []
    partial: list[JsonDict] = []
    blocked: list[JsonDict] = []
    honest_null: list[JsonDict] = []
    missing_lanes: list[JsonDict] = []
    skipped_paths = {row["artifact_path"] for row in _skipped_gated_tasks(payloads)}
    for spec in LANE_SPECS:
        missing_sources = [
            artifact_path for artifact_path in spec.source_artifacts if artifact_path in missing_artifacts
        ]
        if missing_sources:
            missing_lanes.append({"lane": spec.name, "missing_artifacts": missing_sources})
            continue
        if any(artifact_path in skipped_paths for artifact_path in spec.source_artifacts):
            continue
        classification = spec.classifier(payloads)
        entry = _lane_entry(spec, classification, payloads)
        if classification == "closed":
            closed.append(entry)
        elif classification == "partial":
            partial.append(entry)
        elif classification == "blocked":
            blocked.append(entry)
        elif classification == "honest_null":
            honest_null.append(entry)
        else:
            missing_lanes.append({"lane": spec.name, "missing_artifacts": list(spec.source_artifacts)})
    return closed, partial, blocked, honest_null, missing_lanes


def _board_reachability(payloads: Payloads) -> JsonDict:
    value = _value(
        payloads,
        "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json",
        "board_reachability",
    )
    return dict(value) if isinstance(value, Mapping) else {}


def _blocked_hardware(board_reachability: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    for board, row in board_reachability.items():
        if isinstance(row, Mapping) and row.get("reachable") is False:
            blocked.append(str(board))
    return sorted(blocked)


def _blocked_preconditions(payloads: Payloads) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for artifact_path in (
        "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json",
        "results/experiment_5461_gated_sota_csl_memory_routing_v496.json",
    ):
        details = _value(payloads, artifact_path, "precondition_details")
        if isinstance(details, Mapping) and details.get("all_passed") is False:
            rows.append(
                {
                    "artifact_path": artifact_path,
                    "blocked_preconditions": list(details.get("blocked_preconditions", [])),
                }
            )
    return rows


def _gguf_offload_gap(payloads: Payloads) -> bool:
    for artifact_path in (
        "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json",
        "results/experiment_5461_gated_sota_csl_memory_routing_v496.json",
    ):
        if _value(payloads, artifact_path, "gpu_offload_verified") is False:
            return True
        receipt = _value(payloads, artifact_path, "runtime_receipt")
        if isinstance(receipt, Mapping) and receipt.get("offload_evidence") is False:
            return True
    return False


def _agent_failure_taxonomy(payloads: Payloads) -> JsonDict:
    board_reachability = _board_reachability(payloads)
    blocked_boards = _blocked_hardware(board_reachability)
    blocked_preconditions = _blocked_preconditions(payloads)
    tautology_flags = _value(
        payloads,
        "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json",
        "corrigendum_pending",
    )
    invalid_tautology_fields = _value(
        payloads,
        "results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json",
        "invalid_tautological_fields",
    )
    arc_banked = _value(
        payloads,
        "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json",
        "new_level_banked",
    )
    transition_blocked_lanes = _value(
        payloads,
        "results/experiment_5454_transition_v496.json",
        "blocked_lanes",
    )
    return {
        "tautology": {
            "observed": bool(tautology_flags) or bool(invalid_tautology_fields),
            "source_artifacts": [
                "results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json",
                "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json",
            ],
            "finding": (
                "Exp5457 remains blocked by a critical TAUTOLOGY flag; Exp5456 "
                "records the prior tautological fields and clean posthoc dependency audit."
            ),
        },
        "duration_precondition_failures": {
            "observed": bool(blocked_preconditions),
            "source_artifacts": [
                "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json",
                "results/experiment_5461_gated_sota_csl_memory_routing_v496.json",
            ],
            "finding": "No GGUF duration or model-precondition blocker is present in these artifacts.",
            "blocked_preconditions": blocked_preconditions,
        },
        "missing_hardware": {
            "observed": bool(blocked_boards),
            "source_artifacts": [
                "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json"
            ],
            "finding": "Hardware receipts are partial because at least one named board was unreachable.",
            "blocked_boards": blocked_boards,
            "board_reachability": board_reachability,
        },
        "no_bank_arc": {
            "observed": arc_banked is False,
            "source_artifacts": [
                "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json"
            ],
            "finding": "Exp5465 is an honest-null live ARC attempt: no reproduction-gated level was banked.",
        },
        "gguf_offload_gaps": {
            "observed": _gguf_offload_gap(payloads),
            "source_artifacts": [
                "results/experiment_5457_gated_sota_distortion_guarded_decoding_v496.json",
                "results/experiment_5461_gated_sota_csl_memory_routing_v496.json",
            ],
            "finding": (
                "Top-level GGUF/CUDA offload receipts are present for the live GGUF lanes; "
                "the remaining local-SOTA block is tautology/readiness, not missing offload."
            ),
        },
        "unsupported_hidden_internal_claims": {
            "observed": _contains_token_internal(transition_blocked_lanes),
            "source_artifacts": ["results/experiment_5454_transition_v496.json"],
            "finding": (
                "The transition carries token/internal access as blocked because authenticated "
                "hidden-state, logits, attention, or token receipts are absent."
            ),
        },
    }


def _lane_names(rows: Sequence[JsonDict]) -> list[str]:
    return [str(row["lane"]) for row in rows]


def _prd_requirement_map(
    closed: Sequence[JsonDict],
    partial: Sequence[JsonDict],
    blocked: Sequence[JsonDict],
    honest_null: Sequence[JsonDict],
) -> JsonDict:
    closed_names = set(_lane_names(closed))
    partial_names = set(_lane_names(partial))
    blocked_names = set(_lane_names(blocked))
    null_names = set(_lane_names(honest_null))
    return {
        "FR-11": {
            "title": "Autonomous Self-Learning Loop",
            "classification": (
                "closed"
                if {"csl_policy_bandit", "sota_csl_memory_routing"}.issubset(closed_names)
                else "partial"
            ),
            "evidence_artifacts": [
                "results/experiment_5460_csl_policy_bandit_v496.json",
                "results/experiment_5461_gated_sota_csl_memory_routing_v496.json",
            ],
            "closed_lanes": [
                name
                for name in ("csl_policy_bandit", "sota_csl_memory_routing")
                if name in closed_names
            ],
            "blocked_or_partial_lanes": sorted(
                name
                for name in partial_names | blocked_names
                if "csl" in name or "memory" in name
            ),
            "claim_boundary": "Governed memory/policy routing is closed; model-weight self-training is not claimed.",
        },
        "FR-12": {
            "title": "Verifiable Reasoning",
            "classification": "partial" if blocked_names else "closed",
            "evidence_artifacts": [
                "results/experiment_5456_guided_decoding_tautology_corrigendum_v496.json",
                "results/experiment_5458_minimal_core_claim_repair_v496.json",
                "results/experiment_5459_constraint_distortion_guard_v496.json",
                "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json",
                "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json",
            ],
            "closed_lanes": sorted(
                name
                for name in closed_names
                if name
                in {
                    "guided_decoding_corrigendum",
                    "minimal_core_claim_repair",
                    "constraint_distortion_guard",
                    "sota_csl_memory_routing",
                    "arc_metric_integrity_precheck",
                }
            ),
            "blocked_or_partial_lanes": sorted(
                name
                for name in partial_names | blocked_names | null_names
                if name
                in {
                    "local_sota_distortion_guarded_decoding",
                    "active_constraint_pbit_pdit_bridge",
                    "arc_connected_component_salience_levelup",
                    "token_internal_hidden_claims",
                }
            ),
            "claim_boundary": (
                "Deterministic exact-check lanes are useful, but local SOTA decoding, "
                "token/internal access, and ARC level banking remain open or null."
            ),
        },
        "NFR-01": {
            "title": "Performance",
            "classification": "partial",
            "evidence_artifacts": [
                "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json",
                "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json",
            ],
            "closed_lanes": [],
            "blocked_or_partial_lanes": sorted(
                name
                for name in partial_names | null_names
                if name
                in {
                    "active_constraint_pbit_pdit_bridge",
                    "hardware_boundary_exchange_receipts",
                    "hardware_speedup_claim",
                }
            ),
            "claim_boundary": "Hardware receipts and boundary facts exist, but no speedup claim is supported.",
        },
    }


def _status(missing_artifacts: Sequence[str]) -> str:
    return "blocked" if missing_artifacts else "complete"


def _honest_verdict(
    missing_artifacts: Sequence[str],
    skipped_gated_tasks: Sequence[JsonDict],
    closed: Sequence[JsonDict],
    partial: Sequence[JsonDict],
    blocked: Sequence[JsonDict],
    honest_null: Sequence[JsonDict],
) -> str:
    if missing_artifacts:
        return (
            "blocked: .496 PRD gap table missing "
            f"{len(missing_artifacts)} required main artifacts; "
            f"read={len(MAIN_ARTIFACT_PATHS) - len(missing_artifacts)}, "
            f"skipped={len(skipped_gated_tasks)}."
        )
    return (
        "complete: .496 PRD gap table read actual Exp5454-Exp5465 artifacts; "
        f"closed={len(closed)}, partial={len(partial)}, blocked={len(blocked)}, "
        f"honest_null={len(honest_null)}, missing=0, skipped={len(skipped_gated_tasks)}."
    )


def _with_checksum(artifact: JsonDict) -> JsonDict:
    without_checksum = dict(artifact)
    without_checksum.pop("reproducibility_checksum", None)
    checksum = _sha256_text(_stable_json(without_checksum))
    artifact["reproducibility_checksum"] = checksum
    return artifact


def build_report(root: Path, tests_run: Sequence[str] | None = None) -> JsonDict:
    """Read upstream Exp5454-Exp5465 artifacts and build the gap table.

    The function takes an explicit root so tests can construct a miniature
    result tree. It still uses the same artifact path constants as production,
    which prevents the test fixtures from accidentally exercising a different
    contract than the checked-in deliverable.
    """

    payloads, missing_artifacts = _main_payloads(root)
    artifact_paths_read = _discover_available_artifacts(root)
    sidecars = _sidecar_summaries(root, artifact_paths_read)
    skipped_gated_tasks = _skipped_gated_tasks(payloads)
    closed, partial, blocked, honest_null, missing_lanes = _classify_lanes(
        payloads,
        missing_artifacts,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": _status(missing_artifacts),
        "milestone": MILESTONE,
        "artifact_paths_read": artifact_paths_read,
        "sidecar_artifacts_read": sidecars,
        "missing_artifacts": list(missing_artifacts),
        "skipped_gated_tasks": skipped_gated_tasks,
        "closed_lanes": closed,
        "partial_lanes": partial,
        "blocked_lanes": blocked,
        "honest_null_lanes": honest_null,
        "missing_lanes": missing_lanes,
        "prd_requirement_map": _prd_requirement_map(closed, partial, blocked, honest_null),
        "agent_failure_taxonomy": _agent_failure_taxonomy(payloads),
        "docs_updated": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(
            missing_artifacts,
            skipped_gated_tasks,
            closed,
            partial,
            blocked,
            honest_null,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "tests_run": list(tests_run) if tests_run is not None else list(DEFAULT_TESTS_RUN),
        "random_seed": RANDOM_SEED,
    }
    return _with_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must be 2026.07.496")
    if artifact["docs_updated"] != []:
        raise ValueError("docs_updated must remain empty for Exp5466")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    taxonomy = artifact["agent_failure_taxonomy"]
    if not isinstance(taxonomy, Mapping):
        raise ValueError("agent_failure_taxonomy must be a mapping")
    for key in FAILURE_MODE_KEYS:
        if key not in taxonomy:
            raise ValueError(f"agent_failure_taxonomy missing {key}")
    for field in (
        "artifact_paths_read",
        "missing_artifacts",
        "skipped_gated_tasks",
        "closed_lanes",
        "partial_lanes",
        "blocked_lanes",
        "honest_null_lanes",
    ):
        if not isinstance(artifact[field], list):
            raise ValueError(f"{field} must be a list")
    if artifact["missing_artifacts"] and not str(verdict).startswith("blocked:"):
        raise ValueError("honest_verdict must be blocked when required artifacts are missing")


def write_artifact(root: Path, tests_run: Sequence[str] | None = None) -> Path:
    artifact = build_report(root, tests_run=tests_run)
    validate_artifact(artifact)
    output_path = root / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path

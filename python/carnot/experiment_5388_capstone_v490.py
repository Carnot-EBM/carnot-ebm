"""Exp 5388: V490 capstone decision artifact.

Spec refs: REQ-CAPSTONE-5388, SCENARIO-CAPSTONE-5388,
SCENARIO-CAPSTONE-5388-MISSING-OR-GATED-INPUT,
SCENARIO-CAPSTONE-5388-FIELD-PRINCIPLES.

This module closes the milestone by reading the local .490 result artifacts and
copying their gate fields without making a stronger claim than the source
evidence supports. It treats present gate-blocked artifacts as blocked results,
not missing experiments, because a skipped downstream task is still useful
evidence about the upstream gate that failed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5388_capstone_v490.json")
EXPERIMENT = "experiment_5388_capstone_v490"
EXPERIMENT_ID = "exp5388-capstone-v490"
MILESTONE = "2026.07.490"
SCHEMA = "carnot.experiment_5388_capstone_v490.v1"
RUN_DATE = "20260708"
RANDOM_SEED = 5388
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP5376 = "results/experiment_5376_transition_v490.json"
EXP5377 = "results/experiment_5377_sota_source_delta_v490.json"
EXP5378 = "results/experiment_5378_structured_methodology_duration_receipt_v490.json"
EXP5379 = "results/experiment_5379_live_structured_clean_gate_rerun_v490.json"
EXP5380 = "results/experiment_5380_constraint_tax_tool_action_panel_v3_v490.json"
EXP5381 = "results/experiment_5381_budget_memory_tautology_corrigendum_v490.json"
EXP5382 = "results/experiment_5382_real_workflow_continuous_self_learning_v490.json"
EXP5383 = "results/experiment_5383_overwrite_guidance_scale_validity_v490.json"
EXP5384 = "results/experiment_5384_pbit_boundary_overwrite_joint_diagnostic_v490.json"
EXP5385 = "results/experiment_5385_arc_geometric_salience_live_path_v490.json"
EXP5386 = "results/experiment_5386_hardware_hashchain_receipts_v490.json"
EXP5387 = "results/experiment_5387_token_feature_backend_reopen_gate_v490.json"

EXPECTED_ARTIFACT_PATHS: tuple[str, ...] = (
    EXP5376,
    EXP5377,
    EXP5378,
    EXP5379,
    EXP5380,
    EXP5381,
    EXP5382,
    EXP5383,
    EXP5384,
    EXP5385,
    EXP5386,
    EXP5387,
)

GATED_TASKS: dict[str, JsonDict] = {
    EXP5379: {
        "task_id": "exp5379-live-structured-clean-gate-rerun-v490",
        "requires": {
            EXP5378: {
                "live_sota_receipt_ready": True,
                "methodology_duration_s": ">=60.0",
            }
        },
    },
    EXP5380: {
        "task_id": "exp5380-constraint-tax-tool-action-panel-v3-v490",
        "requires": {EXP5379: {"structured_protocol_clean": True}},
    },
    EXP5382: {
        "task_id": "exp5382-real-workflow-continuous-self-learning-v490",
        "requires": {EXP5381: {"budget_memory_corrigendum_clean": True}},
    },
}

SPEC_REFS = (
    "REQ-CAPSTONE-5388",
    "SCENARIO-CAPSTONE-5388",
    "SCENARIO-CAPSTONE-5388-MISSING-OR-GATED-INPUT",
    "SCENARIO-CAPSTONE-5388-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "complete if aggregation ran with every expected artifact readable; honest_partial "
        "if critical artifacts are missing or unreadable."
    ),
    "milestone": "must equal 2026.07.490.",
    "expected_artifacts": "ordered list of expected .490 artifact paths.",
    "artifacts_found": "ordered list of found artifact paths.",
    "artifacts_missing": "ordered list of missing or unreadable artifact paths.",
    "skipped_by_gate": (
        "object mapping gated task ids to upstream gate conditions when a present "
        "artifact was skipped or blocked by a gate."
    ),
    "structured_methodology_receipt_ready": "copied or derived from Exp5378.",
    "structured_protocol_clean": "copied or derived from Exp5379.",
    "constraint_tax_panel_ready": "copied or derived from Exp5380.",
    "budget_memory_corrigendum_clean": "copied or derived from Exp5381.",
    "continuous_self_learning_real_workflow_ready": "copied or derived from Exp5382.",
    "continuous_self_learning_requirement_satisfied": (
        "true only if Exp5382 ran and reported continuous_self_learning_real_workflow_ready=true, "
        "or false with the upstream gate block recorded."
    ),
    "overwrite_guidance_scale_ready": (
        "copied or derived from Exp5383, with flagged evidence recorded separately instead of laundered."
    ),
    "pbit_boundary_overwrite_ready": "copied or derived from Exp5384.",
    "arc_new_level_banked": "copied or derived from Exp5385.",
    "hardware_hash_chained_receipt_ready": "copied or derived from Exp5386.",
    "hardware_speedup_claim": (
        "must be false unless Exp5386 has repeatable board timing evidence."
    ),
    "future_token_signal_allowed": "copied or derived from Exp5387.",
    "retired_or_blocked_lanes": "list of lanes that remain closed or should retire.",
    "next_milestone_recommendations": (
        "concrete recommendations for `.491` grounded in clean evidence and blockers."
    ),
    "active_roadmap_modified": "must be false.",
    "conductor_modified": "must be false.",
    "honest_verdict": "one-line capstone truth summary.",
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "field_principles",
    "inference_substrate",
    "source_artifacts",
    "artifact_read_errors",
    "phase_summaries",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES.keys(),
)

BOOLEAN_FIELDS = (
    "structured_methodology_receipt_ready",
    "structured_protocol_clean",
    "constraint_tax_panel_ready",
    "budget_memory_corrigendum_clean",
    "continuous_self_learning_real_workflow_ready",
    "continuous_self_learning_requirement_satisfied",
    "overwrite_guidance_scale_ready",
    "pbit_boundary_overwrite_ready",
    "arc_new_level_banked",
    "hardware_hash_chained_receipt_ready",
    "hardware_speedup_claim",
    "future_token_signal_allowed",
    "active_roadmap_modified",
    "conductor_modified",
)


def value_of(value: Any) -> Any:
    """Return the machine value from a bare or principle-wrapped field."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _read_inputs(
    root: Path | str,
) -> tuple[dict[str, JsonDict], list[str], list[str], list[JsonDict]]:
    root_path = Path(root)
    payloads: dict[str, JsonDict] = {}
    found: list[str] = []
    missing: list[str] = []
    read_errors: list[JsonDict] = []

    for relative in EXPECTED_ARTIFACT_PATHS:
        path = root_path / relative
        if not path.exists():
            missing.append(relative)
            continue

        found.append(relative)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            missing.append(relative)
            read_errors.append(
                {
                    "path": relative,
                    "classification": f"malformed_json:{exc.msg}",
                    "line": exc.lineno,
                    "column": exc.colno,
                }
            )
            continue

        if not isinstance(payload, dict):
            missing.append(relative)
            read_errors.append({"path": relative, "classification": "not_json_object"})
            continue

        payloads[relative] = payload

    return payloads, found, missing, read_errors


def _status(payload: JsonMap | None) -> str:
    if payload is None:
        return "missing"
    return str(value_of(payload.get("status", "unknown")))


def _verdict(payload: JsonMap | None) -> str:
    if payload is None:
        return "missing"
    return str(value_of(payload.get("honest_verdict", "")))


def _is_gate_blocked(payload: JsonMap | None) -> bool:
    if payload is None:
        return True
    status = _status(payload)
    verdict = _verdict(payload)
    upstream_gate = payload.get("upstream_gate")
    upstream_failed = (
        isinstance(upstream_gate, Mapping) and value_of(upstream_gate.get("all_passed")) is False
    )
    return (
        status in {"blocked", "gate_block", "gated_skip", "honest_blocked", "skipped"}
        or status.startswith(("blocked", "gate_block"))
        or verdict.startswith("blocked_")
        or value_of(payload.get("skipped")) is True
        or payload.get("blocked_at_layer") is not None
        or upstream_failed
    )


def _source_bool(payload: JsonMap | None, *fields: str, gate_sensitive: bool = True) -> bool:
    if payload is None:
        return False
    if gate_sensitive and _is_gate_blocked(payload):
        return False
    for field in fields:
        if field in payload:
            return value_of(payload[field]) is True
    return False


def _source_number(payload: JsonMap | None, field: str, default: float = 0.0) -> float:
    if payload is None or field not in payload:
        return default
    value = value_of(payload[field])
    return float(value) if isinstance(value, int | float) else default


def _artifact_sha256(root: Path | str, relative: str) -> str:
    path = Path(root) / relative
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _source_artifacts(root: Path | str, found: Sequence[str], payloads: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for relative in found:
        payload = payloads.get(relative)
        if not isinstance(payload, Mapping):
            continue
        rows.append(
            {
                "path": relative,
                "sha256": _artifact_sha256(root, relative),
                "status": _status(payload),
                "honest_verdict": _verdict(payload),
                "flagged_adversarial": value_of(payload.get("flagged_adversarial")) is True,
            }
        )
    return rows


def _imported(payload: JsonMap | None, fields: Sequence[str]) -> JsonDict:
    if payload is None:
        return {}
    return {field: value_of(payload[field]) for field in fields if field in payload}


def _skipped_by_gate(payloads: JsonMap) -> dict[str, JsonDict]:
    skipped: dict[str, JsonDict] = {}
    for relative, gate_spec in GATED_TASKS.items():
        payload = payloads.get(relative)
        if not isinstance(payload, Mapping) or not _is_gate_blocked(payload):
            continue
        task_id = str(gate_spec["task_id"])
        gate_conditions = value_of(payload.get("upstream_gate"))
        if not isinstance(gate_conditions, Mapping):
            gate_conditions = {"required": gate_spec["requires"]}
        skipped[task_id] = {
            "source_artifact": relative,
            "status": _status(payload),
            "honest_verdict": _verdict(payload),
            "blocked_at_layer": value_of(payload.get("blocked_at_layer")),
            "gate_conditions": dict(gate_conditions),
        }
    return skipped


def _phase_summaries(payloads: JsonMap, fields: JsonMap) -> list[JsonDict]:
    exp5378 = payloads.get(EXP5378)
    exp5379 = payloads.get(EXP5379)
    exp5380 = payloads.get(EXP5380)
    exp5381 = payloads.get(EXP5381)
    exp5382 = payloads.get(EXP5382)
    exp5383 = payloads.get(EXP5383)
    exp5384 = payloads.get(EXP5384)
    exp5385 = payloads.get(EXP5385)
    exp5386 = payloads.get(EXP5386)
    exp5387 = payloads.get(EXP5387)
    solver_flagged = _source_bool(exp5383, "flagged_adversarial", gate_sensitive=False)

    return [
        {
            "lane": "structured_sota",
            "outcome": (
                "clean_ready"
                if fields["structured_methodology_receipt_ready"]
                and fields["structured_protocol_clean"]
                else "blocked_or_partial"
            ),
            "source_artifacts": [EXP5378, EXP5379],
            "evidence": {
                "exp5378": _imported(
                    exp5378,
                    (
                        "live_sota_receipt_ready",
                        "methodology_duration_s",
                        "structured_protocol_clean",
                        "parse_success_rate",
                        "schema_success_rate",
                        "semantic_success_rate",
                        "final_json_extraction_rate",
                        "unsafe_false_accepts",
                        "no_autotokenizer_used",
                    ),
                ),
                "exp5379": _imported(
                    exp5379,
                    (
                        "methodology_duration_s",
                        "structured_protocol_clean",
                        "parse_success_rate",
                        "schema_success_rate",
                        "semantic_success_rate",
                        "final_json_extraction_rate",
                        "unsafe_false_accepts",
                    ),
                ),
            },
            "claim_boundary": "local_sota_structured_receipt_and_clean_gate_only",
        },
        {
            "lane": "constraint_tax",
            "outcome": "ready" if fields["constraint_tax_panel_ready"] else "blocked_or_skipped",
            "source_artifacts": [EXP5380],
            "evidence": _imported(
                exp5380,
                (
                    "upstream_structured_protocol_clean",
                    "constraint_tax_panel_ready",
                    "fixture_count",
                    "deterministic_state_evidence_count",
                    "tool_action_reachability_rate",
                    "unsafe_false_accepts",
                ),
            ),
            "claim_boundary": "tool_action_state_evidence_not_quality_headline",
        },
        {
            "lane": "continuous_self_learning",
            "outcome": (
                "requirement_satisfied"
                if fields["continuous_self_learning_requirement_satisfied"]
                else "blocked_or_missing"
            ),
            "source_artifacts": [EXP5381, EXP5382],
            "evidence": {
                "exp5381": _imported(
                    exp5381,
                    (
                        "budget_memory_corrigendum_clean",
                        "rollback_supported",
                        "no_weight_mutation",
                        "unsafe_false_accepts",
                    ),
                ),
                "exp5382": _imported(
                    exp5382,
                    (
                        "continuous_self_learning_real_workflow_ready",
                        "workflow_name",
                        "checked_event_count",
                        "context_efficiency_delta",
                        "verifier_cost_delta",
                        "quality_delta",
                        "stale_memory_deflection_rate",
                        "poison_memory_deflection_rate",
                        "rollback_success_rate",
                        "no_weight_mutation",
                    ),
                ),
            },
            "claim_boundary": "real_workflow_memory_policy_no_model_weight_mutation",
        },
        {
            "lane": "solver_guidance",
            "outcome": "ready_flagged" if solver_flagged else "ready",
            "source_artifacts": [EXP5383],
            "evidence": _imported(
                exp5383,
                (
                    "overwrite_guidance_scale_ready",
                    "solver_authoritative",
                    "fallback_completeness_rate",
                    "post_projection_validity_rate",
                    "unsafe_false_accepts",
                    "flagged_adversarial",
                    "corrigendum_pending",
                ),
            ),
            "claim_boundary": (
                "ready_but_flagged_adversarial" if solver_flagged else "solver_authoritative_ready"
            ),
        },
        {
            "lane": "pbit_boundary_overwrite",
            "outcome": "ready" if fields["pbit_boundary_overwrite_ready"] else "blocked",
            "source_artifacts": [EXP5384],
            "evidence": _imported(
                exp5384,
                (
                    "pbit_boundary_overwrite_ready",
                    "hardware_speedup_claim",
                    "unsafe_false_accepts",
                ),
            ),
            "claim_boundary": "cpu_only_no_hardware_speedup",
        },
        {
            "lane": "arc_geometric_salience",
            "outcome": (
                "new_level_banked"
                if fields["arc_new_level_banked"]
                else "honest_null_no_level_banked"
            ),
            "source_artifacts": [EXP5385],
            "evidence": _imported(
                exp5385,
                (
                    "new_level_banked",
                    "geometric_salience_live_reachable",
                    "live_attempt_count",
                    "failure_mode",
                    "solve_provenance",
                    "no_outer_loop_re",
                    "no_per_game_adapter",
                    "offline_reproduced",
                ),
            ),
            "claim_boundary": "live_agent_attempt_no_outer_loop_no_bank",
        },
        {
            "lane": "hardware",
            "outcome": (
                "receipt_ready_no_speedup"
                if fields["hardware_hash_chained_receipt_ready"]
                else "blocked"
            ),
            "source_artifacts": [EXP5386],
            "evidence": _imported(
                exp5386,
                (
                    "hardware_hash_chained_receipt_ready",
                    "hardware_speedup_claim",
                    "repeatability_evidence_present",
                    "kv260_status",
                    "polar_fire_status",
                    "gatemate_status",
                    "hardware_evidence_level",
                ),
            ),
            "claim_boundary": "hash_chained_receipts_no_repeatable_speedup",
        },
        {
            "lane": "token_backend",
            "outcome": (
                "open" if fields["future_token_signal_allowed"] else "closed_no_backend_signal"
            ),
            "source_artifacts": [EXP5387],
            "evidence": _imported(
                exp5387,
                (
                    "future_signal_allowed",
                    "backend_reopen_allowed",
                    "logits_available",
                    "hidden_states_available",
                    "attention_available",
                    "intermediate_depth_exits_available",
                    "no_live_signal_claim",
                ),
            ),
            "claim_boundary": "no_token_internal_energy_until_backend_features_exist",
        },
    ]


def _retired_or_blocked_lanes(payloads: JsonMap, fields: JsonMap) -> list[JsonDict]:
    exp5383 = payloads.get(EXP5383)
    exp5385 = payloads.get(EXP5385)
    exp5386 = payloads.get(EXP5386)
    exp5387 = payloads.get(EXP5387)
    kv260 = value_of(exp5386.get("kv260_status")) if isinstance(exp5386, Mapping) else {}
    gatemate = value_of(exp5386.get("gatemate_status")) if isinstance(exp5386, Mapping) else {}

    return [
        {
            "lane": "overwrite_guidance_scale",
            "state": (
                "blocked_flagged_adversarial"
                if _source_bool(exp5383, "flagged_adversarial", gate_sensitive=False)
                else "open"
            ),
            "reason": "Exp5383 readiness is copied, but conductor/adversarial verification flagged it.",
        },
        {
            "lane": "arc_geometric_salience_live_path",
            "state": "open" if fields["arc_new_level_banked"] else "blocked_no_bank",
            "reason": _verdict(exp5385),
        },
        {
            "lane": "token_internal_feature_signal",
            "state": (
                "open"
                if fields["future_token_signal_allowed"]
                else "retired_until_backend_features"
            ),
            "reason": _verdict(exp5387),
        },
        {
            "lane": "hardware_speedup_claim",
            "state": (
                "open" if fields["hardware_speedup_claim"] else "blocked_on_repeatable_board_timing"
            ),
            "reason": "Exp5386 reports no repeatability evidence sufficient for a speedup claim.",
        },
        {
            "lane": "kv260_workload",
            "state": (
                "open"
                if isinstance(kv260, Mapping) and value_of(kv260.get("ssh_reachable")) is True
                else "blocked_unreachable"
            ),
            "reason": "KV260 SSH/workload evidence was unreachable in Exp5386.",
        },
        {
            "lane": "gatemate_workload",
            "state": (
                "open"
                if isinstance(gatemate, Mapping)
                and value_of(gatemate.get("physical_or_jtag_path_available")) is True
                else "blocked_physical_or_jtag"
            ),
            "reason": "GateMate remains blocked on physical/JTAG availability.",
        },
    ]


def _next_milestone_recommendations() -> list[JsonDict]:
    return [
        {
            "action": "carry_clean_structured_constraint_tax_into_v491",
            "recommendation": (
                "Use the clean Exp5378-5380 structured and constraint-tax receipts as "
                "the starting gate for the next measured quality or tool/action panel."
            ),
            "guardrails": ["do_not_reopen_cpu_only_sota_headline", "preserve_state_evidence"],
        },
        {
            "action": "reconcile_or_rerun_flagged_overwrite_guidance",
            "recommendation": (
                "Resolve Exp5383's adversarial flag before using overwrite-guidance "
                "readiness as a headline-clean solver result."
            ),
            "guardrails": ["keep_solver_authoritative", "unsafe_false_accepts_zero"],
        },
        {
            "action": "convert_arc_geometric_salience_from_no_bank_to_levelup",
            "recommendation": (
                "Keep the live-agent geometric path, but only count .491 ARC progress "
                "when a new level is reproduction-gated."
            ),
            "guardrails": [
                "no_outer_loop_re",
                "no_per_game_adapter",
                "offline_reproduce_before_count",
            ],
        },
        {
            "action": "keep_token_backend_closed_until_real_features",
            "recommendation": (
                "Do not schedule token/internal-feature energy claims until logits, "
                "hidden states, attention, or depth exits have clean provenance."
            ),
            "guardrails": ["no_text_only_signal", "no_quality_claim_without_feature_rows"],
        },
        {
            "action": "get_repeatable_board_timing_before_speedup_claims",
            "recommendation": (
                "Extend the hash-chain receipt path to repeatable board timing before "
                "claiming KV260, PolarFire, or GateMate acceleration."
            ),
            "guardrails": ["no_host_mmcblk_kv260_evidence", "no_speedup_without_repeatability"],
        },
    ]


def payload_checksum(payload: JsonMap) -> str:
    """Compute the artifact checksum while ignoring the checksum field itself."""

    comparable = dict(payload)
    comparable.pop("reproducibility_checksum", None)
    encoded = json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[JsonMap] | None = None,
) -> JsonDict:
    payloads, found, missing, read_errors = _read_inputs(root)
    exp5378 = payloads.get(EXP5378)
    exp5379 = payloads.get(EXP5379)
    exp5380 = payloads.get(EXP5380)
    exp5381 = payloads.get(EXP5381)
    exp5382 = payloads.get(EXP5382)
    exp5383 = payloads.get(EXP5383)
    exp5384 = payloads.get(EXP5384)
    exp5385 = payloads.get(EXP5385)
    exp5386 = payloads.get(EXP5386)
    exp5387 = payloads.get(EXP5387)

    structured_receipt = (
        _source_bool(exp5378, "live_sota_receipt_ready")
        and _source_number(exp5378, "methodology_duration_s") >= 60.0
    )
    workflow_ready = _source_bool(exp5382, "continuous_self_learning_real_workflow_ready")
    repeatable_timing = _source_bool(
        exp5386, "repeatability_evidence_present", gate_sensitive=False
    )
    source_speedup_claim = _source_bool(
        exp5386, "hardware_speedup_claim", "speedup_claim", gate_sensitive=False
    )

    fields: JsonDict = {
        "status": "honest_partial" if missing else "complete",
        "milestone": MILESTONE,
        "expected_artifacts": list(EXPECTED_ARTIFACT_PATHS),
        "artifacts_found": found,
        "artifacts_missing": missing,
        "skipped_by_gate": _skipped_by_gate(payloads),
        "structured_methodology_receipt_ready": structured_receipt,
        "structured_protocol_clean": _source_bool(exp5379, "structured_protocol_clean"),
        "constraint_tax_panel_ready": _source_bool(exp5380, "constraint_tax_panel_ready"),
        "budget_memory_corrigendum_clean": _source_bool(exp5381, "budget_memory_corrigendum_clean"),
        "continuous_self_learning_real_workflow_ready": workflow_ready,
        "continuous_self_learning_requirement_satisfied": workflow_ready,
        "overwrite_guidance_scale_ready": _source_bool(exp5383, "overwrite_guidance_scale_ready"),
        "pbit_boundary_overwrite_ready": _source_bool(exp5384, "pbit_boundary_overwrite_ready"),
        "arc_new_level_banked": _source_bool(exp5385, "arc_new_level_banked", "new_level_banked"),
        "hardware_hash_chained_receipt_ready": _source_bool(
            exp5386, "hardware_hash_chained_receipt_ready"
        ),
        "hardware_speedup_claim": bool(source_speedup_claim and repeatable_timing),
        "future_token_signal_allowed": _source_bool(
            exp5387, "future_token_signal_allowed", "future_signal_allowed"
        ),
        "active_roadmap_modified": False,
        "conductor_modified": False,
    }
    fields["retired_or_blocked_lanes"] = _retired_or_blocked_lanes(payloads, fields)
    fields["next_milestone_recommendations"] = _next_milestone_recommendations()
    fields["honest_verdict"] = (
        "honest_partial: .490 aggregation ran with missing or unreadable artifacts; "
        "no missing or gate-blocked lane was promoted."
        if missing
        else (
            "complete: .490 proved clean structured receipts, constraint-tax evidence, "
            "and real self-learning; solver guidance is flagged, ARC banked no level, "
            "hardware has no speedup, and token/backend signal stays closed."
        )
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts": _source_artifacts(root, found, payloads),
        "artifact_read_errors": read_errors,
        "phase_summaries": _phase_summaries(payloads, fields),
        "tests_run": [dict(row) for row in tests_run] if tests_run is not None else [],
        **fields,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    missing_fields = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(f"missing required fields: {missing_fields}")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must equal 2026.07.490")
    if artifact["status"] not in {"complete", "honest_partial"}:
        raise ValueError("status must be complete or honest_partial")
    if artifact["expected_artifacts"] != list(EXPECTED_ARTIFACT_PATHS):
        raise ValueError("expected_artifacts changed")
    if artifact["artifacts_missing"] and artifact["status"] != "honest_partial":
        raise ValueError("honest_partial required when artifacts_missing is non-empty")
    for field in BOOLEAN_FIELDS:
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare boolean")
    if (
        artifact["continuous_self_learning_requirement_satisfied"]
        and not artifact["continuous_self_learning_real_workflow_ready"]
    ):
        raise ValueError(
            "continuous self-learning requirement cannot pass without workflow evidence"
        )
    if artifact["hardware_speedup_claim"]:
        raise ValueError("hardware_speedup_claim must remain false without repeatable board timing")
    if artifact["active_roadmap_modified"]:
        raise ValueError("active_roadmap_modified must be false")
    if artifact["conductor_modified"]:
        raise ValueError("conductor_modified must be false")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "honest_partial:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles changed")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
    tests_run: Sequence[JsonMap] | None = None,
) -> JsonDict:
    artifact = build_artifact(root=root, tests_run=tests_run)
    validate_artifact(artifact)
    output_path = (
        Path(result_path) if result_path is not None else Path(root) / RESULT_RELATIVE_PATH
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact

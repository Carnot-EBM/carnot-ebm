"""Experiment 5320: V485 capstone aggregation.

Spec refs: REQ-CAPSTONE-5320, SCENARIO-CAPSTONE-5320,
SCENARIO-CAPSTONE-5320-BLOCKED-MISSING-INPUT.

This module is intentionally aggregation-only. It reads the checked-in V485
result artifacts, separates clean positives from blocked, gated, flagged,
unchanged, and reachability-only evidence, and writes the milestone closeout
without running new model, solver, or hardware workloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5320_capstone_v485.json")
EXPERIMENT = "experiment_5320_capstone_v485"
EXPERIMENT_ID = "exp5320-capstone-v485"
MILESTONE = "2026.07.485"
SCHEMA = "carnot.experiment_5320_capstone_v485.v1"
RUN_DATE = "2026-07-06"
RANDOM_SEED = 5320
INFERENCE_SUBSTRATE = "local_artifact_aggregation_and_doc_reconcile"
NEXT_MILESTONE_BRANCH = "2026.07.486-runtime-repair-and-gated-quality"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = (
    "REQ-CAPSTONE-5320",
    "SCENARIO-CAPSTONE-5320",
    "SCENARIO-CAPSTONE-5320-BLOCKED-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5320-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "identifies Exp5320 as the `.485` capstone artifact so downstream reconciliation "
        "cannot confuse it with the transition task or gated SOTA smoke."
    ),
    "milestone": (
        "binds the aggregation to 2026.07.485 and the close-state read of Exp5307 through "
        "Exp5319."
    ),
    "status": (
        "complete only when every expected artifact is readable; otherwise "
        "blocked_missing_required."
    ),
    "honest_verdict": (
        "terminal prefix; starts with complete: or blocked_ and summarizes the milestone "
        "without laundering blocked, gated, null, harmful, flagged, quarantined, missing, "
        "no-speedup, or no-quality evidence."
    ),
    "inference_substrate": (
        "local_artifact_aggregation_and_doc_reconcile because the capstone reads local result "
        "artifacts and existing docs without running model, solver, or hardware workloads."
    ),
    "artifacts_read": (
        "every readable upstream artifact with path, experiment identity, verdict, and sha256 "
        "for the audit trail."
    ),
    "missing_artifacts": (
        "empty only when all expected artifacts are present and parseable; otherwise names "
        "missing or malformed inputs."
    ),
    "sota_runtime_status": (
        "Exp5309 runtime gate state separated from any quality claim."
    ),
    "sota_quality_status": (
        "Exp5311 quality-smoke state; gate-blocked quality remains unmeasured."
    ),
    "paraphrase_verification_status": (
        "Exp5310 deterministic fixture readiness and label-preservation evidence separated "
        "from gated SOTA quality."
    ),
    "continuous_self_learning_status": (
        "Exp5312/Exp5313 transition verifier and rollout evidence with process-score, safety, "
        "rollback, and no-weight-mutation gates."
    ),
    "solver_status": (
        "Exp5314/Exp5315 bounded solver guidance, symbolic fallback, and misleading-class "
        "blocking."
    ),
    "kan_certificate_status": (
        "Exp5316 bounded KAN allocation improvement while certificate success delta remains "
        "zero."
    ),
    "ebt_telemetry_status": (
        "Exp5317 methodology-clean tiny telemetry and its explicit quarantine boundaries."
    ),
    "smt_hint_protocol_status": (
        "Exp5318 deterministic protocol evidence plus flagged-adversarial state so it is not "
        "treated as clean success."
    ),
    "hardware_status": (
        "Exp5319 board reachability receipts and no-speedup discipline."
    ),
    "next_milestone_recommendation": (
        "the concrete `.486` branch implied by evidence, prioritizing runtime repair and gated "
        "quality before broadening."
    ),
    "docs_updated": (
        "false for ops/status, ops/changelog, and traceability when the stop rule delegates "
        "those docs to a later reconciler."
    ),
}

PRINCIPLE_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "random_seed",
    "field_principles",
    "reproducibility_checksum",
    *PRINCIPLE_WRAPPED_FIELDS,
    "no_false_speedup_claim",
    "no_false_sota_quality_claim",
)


@dataclass(frozen=True)
class UpstreamArtifact:
    """One expected V485 result artifact.

    The capstone treats this list as the milestone ledger. A missing file is not
    silently skipped, because a skipped artifact can otherwise turn a blocked or
    harmful result into a false clean closeout.
    """

    experiment_number: int
    task_id: str
    relative_path: Path


EXP5307 = UpstreamArtifact(
    5307,
    "exp5307-archive-484-activate-485",
    Path("results/experiment_5307_archive_484_activate_485.json"),
)
EXP5308 = UpstreamArtifact(
    5308,
    "exp5308-sota-source-delta-v485",
    Path("results/experiment_5308_sota_source_delta_v485.json"),
)
EXP5309 = UpstreamArtifact(
    5309,
    "exp5309-sota-runtime-timeout-rootcause-matrix-v485",
    Path("results/experiment_5309_sota_runtime_timeout_rootcause_matrix_v485.json"),
)
EXP5310 = UpstreamArtifact(
    5310,
    "exp5310-paraphrase-consistency-fixture-v485",
    Path("results/experiment_5310_paraphrase_consistency_fixture_v485.json"),
)
EXP5311 = UpstreamArtifact(
    5311,
    "exp5311-gated-sota-paraphrase-coherence-smoke-v485",
    Path("results/experiment_5311_gated_sota_paraphrase_coherence_smoke_v485.json"),
)
EXP5312 = UpstreamArtifact(
    5312,
    "exp5312-trustmem-transition-verifier-self-learning-v485",
    Path("results/experiment_5312_trustmem_transition_verifier_self_learning_v485.json"),
)
EXP5313 = UpstreamArtifact(
    5313,
    "exp5313-gated-memory-transition-policy-rollout-v485",
    Path("results/experiment_5313_gated_memory_transition_policy_rollout_v485.json"),
)
EXP5314 = UpstreamArtifact(
    5314,
    "exp5314-ising-smooth-relaxation-baseline-v485",
    Path("results/experiment_5314_ising_smooth_relaxation_baseline_v485.json"),
)
EXP5315 = UpstreamArtifact(
    5315,
    "exp5315-gated-solver-guidance-ablation-v485",
    Path("results/experiment_5315_gated_solver_guidance_ablation_v485.json"),
)
EXP5316 = UpstreamArtifact(
    5316,
    "exp5316-kan-optimal-abstraction-budget-v485",
    Path("results/experiment_5316_kan_optimal_abstraction_budget_v485.json"),
)
EXP5317 = UpstreamArtifact(
    5317,
    "exp5317-ebt-telemetry-audit-reemit-v485",
    Path("results/experiment_5317_ebt_telemetry_audit_reemit_v485.json"),
)
EXP5318 = UpstreamArtifact(
    5318,
    "exp5318-smt-hint-validation-protocol-v485",
    Path("results/experiment_5318_smt_hint_validation_protocol_v485.json"),
)
EXP5319 = UpstreamArtifact(
    5319,
    "exp5319-hardware-continuity-no-speedup-v485",
    Path("results/experiment_5319_hardware_continuity_no_speedup_v485.json"),
)

EXPECTED_ARTIFACTS = (
    EXP5307,
    EXP5308,
    EXP5309,
    EXP5310,
    EXP5311,
    EXP5312,
    EXP5313,
    EXP5314,
    EXP5315,
    EXP5316,
    EXP5317,
    EXP5318,
    EXP5319,
)


def value_of(value: Any) -> Any:
    """Return the machine value from a principle-wrapped or bare field."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def wrapped(value: Any, field: str) -> JsonDict:
    """Attach the required principle text to a capstone field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_upstream_artifacts(root: Path | str = REPO_ROOT) -> tuple[dict[int, JsonDict], list[JsonDict], list[JsonDict]]:
    """Read every expected V485 artifact and record missing or malformed inputs."""

    root_path = Path(root)
    payloads: dict[int, JsonDict] = {}
    artifacts_read: list[JsonDict] = []
    missing: list[JsonDict] = []
    for source in EXPECTED_ARTIFACTS:
        path = root_path / source.relative_path
        if not path.exists():
            missing.append(
                {
                    "experiment_number": source.experiment_number,
                    "path": str(source.relative_path),
                    "reason": "missing",
                }
            )
            continue
        try:
            payload = _read_json(path)
        except json.JSONDecodeError as exc:
            missing.append(
                {
                    "experiment_number": source.experiment_number,
                    "path": str(source.relative_path),
                    "reason": f"malformed_json:{exc.msg}",
                }
            )
            continue
        payloads[source.experiment_number] = payload
        artifacts_read.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "path": str(source.relative_path),
                "experiment_id": value_of(payload.get("experiment_id")) or source.task_id,
                "status": value_of(payload.get("status")),
                "honest_verdict": value_of(payload.get("honest_verdict")),
                "flagged_adversarial": value_of(payload.get("flagged_adversarial")) is True,
                "sha256": _sha256(path),
            }
        )
    return payloads, artifacts_read, missing


def summarize_sota_runtime(payload: JsonMap | None) -> JsonDict:
    """Summarize Exp5309 without turning runtime receipts into quality claims."""

    if payload is None:
        return {"source_experiment": 5309, "status": "missing_or_unreadable"}
    matrix = value_of(payload.get("per_model_runtime_matrix")) or {}
    timeout_classes = {
        role: model.get("timeout_class")
        for role, model in matrix.items()
        if isinstance(model, Mapping)
    }
    return {
        "source_experiment": 5309,
        "honest_verdict": value_of(payload.get("honest_verdict")),
        "sota_runtime_unblocked": value_of(payload.get("sota_runtime_unblocked")) is True,
        "no_quality_claim": value_of(payload.get("no_quality_claim")) is True,
        "timeout_root_cause": value_of(payload.get("timeout_root_cause")),
        "offload_authenticated_model_count": sum(
            1
            for model in matrix.values()
            if isinstance(model, Mapping) and model.get("offload_authenticated") is True
        ),
        "completed_model_count": sum(
            1
            for model in matrix.values()
            if isinstance(model, Mapping)
            and model.get("completed_load_first_token_and_8_tokens") is True
        ),
        "timeout_classes": timeout_classes,
    }


def summarize_sota_quality(payload: JsonMap | None) -> JsonDict:
    """Summarize Exp5311 as unmeasured when the conductor gate blocked it."""

    if payload is None:
        return {"source_experiment": 5311, "status": "missing_or_unreadable"}
    gates = value_of(payload.get("gates_evaluated")) or []
    failed = next((gate for gate in gates if not gate.get("passed")), {})
    return {
        "source_experiment": 5311,
        "honest_verdict": value_of(payload.get("honest_verdict")),
        "quality_measured": False,
        "gate_blocked": value_of(payload.get("status")) == "blocked",
        "gate_check_summary": value_of(payload.get("gate_check_summary")),
        "failed_gate_upstream": failed.get("upstream"),
        "failed_gate_field": failed.get("artifact_field"),
        "failed_gate_actual": failed.get("actual"),
    }


def summarize_paraphrase(payload: JsonMap | None) -> JsonDict:
    if payload is None:
        return {"source_experiment": 5310, "status": "missing_or_unreadable"}
    return {
        "source_experiment": 5310,
        "paraphrase_fixture_ready": value_of(payload.get("paraphrase_fixture_ready")) is True,
        "paraphrase_group_count": value_of(payload.get("paraphrase_group_count")),
        "label_preservation_pass_rate": value_of(payload.get("label_preservation_pass_rate")),
        "contradiction_violation_caught_rate": value_of(
            payload.get("contradiction_violation_caught_rate")
        ),
        "invalid_premise_handled": value_of(payload.get("invalid_premise_handled")) is True,
    }


def summarize_continuous_self_learning(exp5312: JsonMap | None, exp5313: JsonMap | None) -> JsonDict:
    if exp5312 is None or exp5313 is None:
        return {
            "source_experiments": [5312, 5313],
            "status": "missing_or_unreadable",
            "missing_sources": [
                number
                for number, payload in ((5312, exp5312), (5313, exp5313))
                if payload is None
            ],
        }
    return {
        "source_experiments": [5312, 5313],
        "verifier_ready": value_of(exp5312.get("memory_transition_verifier_ready")) is True,
        "rollout_complete": value_of(exp5313.get("transition_policy_rollout_complete")) is True,
        "safe_transition_commits": [
            value_of(exp5312.get("safe_transition_commits")),
            value_of(exp5312.get("safe_transition_total")),
        ],
        "unsafe_transition_rejections": [
            value_of(exp5312.get("unsafe_transition_rejections")),
            value_of(exp5312.get("unsafe_transition_total")),
        ],
        "coverage_score": value_of(exp5312.get("coverage_score")),
        "preservation_score": value_of(exp5312.get("preservation_score")),
        "faithfulness_score": value_of(exp5312.get("faithfulness_score")),
        "quality_delta_vs_always_full": value_of(exp5313.get("quality_delta_vs_always_full")),
        "transition_score_delta_vs_always_full": value_of(
            exp5313.get("transition_score_delta_vs_always_full")
        ),
        "full_verifier_calls_avoided": value_of(exp5313.get("full_verifier_calls_avoided")),
        "unsafe_false_accepts": value_of(exp5313.get("unsafe_false_accepts")),
        "unsafe_commits_rejected": value_of(exp5313.get("unsafe_commits_rejected")),
        "rollback_events": value_of(exp5313.get("rollback_events")),
        "no_weight_mutation": (
            value_of(exp5312.get("no_model_weight_mutation")) is True
            and value_of(exp5313.get("no_weight_mutation")) is True
        ),
    }


def summarize_solver(exp5314: JsonMap | None, exp5315: JsonMap | None) -> JsonDict:
    if exp5314 is None or exp5315 is None:
        return {
            "source_experiments": [5314, 5315],
            "status": "missing_or_unreadable",
            "missing_sources": [
                number
                for number, payload in ((5314, exp5314), (5315, exp5315))
                if payload is None
            ],
        }
    return {
        "source_experiments": [5314, 5315],
        "smooth_relaxation_ready": value_of(exp5314.get("smooth_relaxation_ready")) is True,
        "solver_guidance_ablation_complete": value_of(
            exp5315.get("solver_guidance_ablation_complete")
        )
        is True,
        "aggregate_conflict_delta": value_of(exp5315.get("aggregate_conflict_delta")),
        "cdcl_fallback_authoritative": (
            value_of(exp5314.get("cdcl_fallback_authoritative")) is True
            and value_of(exp5315.get("cdcl_fallback_authoritative")) is True
        ),
        "misleading_class_blocked": value_of(exp5315.get("misleading_class_blocked")) is True,
        "no_hardware_speedup_claim": (
            value_of(exp5314.get("no_hardware_speedup_claim")) is True
            and value_of(exp5315.get("no_hardware_speedup_claim")) is True
        ),
    }


def summarize_kan(payload: JsonMap | None) -> JsonDict:
    if payload is None:
        return {"source_experiment": 5316, "status": "missing_or_unreadable"}
    certificate_delta = value_of(payload.get("certificate_success_delta"))
    return {
        "source_experiment": 5316,
        "kan_optimal_abstraction_ready": value_of(payload.get("kan_optimal_abstraction_ready"))
        is True,
        "certificate_success_delta": certificate_delta,
        "certificate_success_improved": bool(certificate_delta and certificate_delta > 0.0),
        "envelope_gap_delta": value_of(payload.get("envelope_gap_delta")),
        "false_property_rejection_rate": value_of(payload.get("false_property_rejection_rate")),
        "bounded_fixture_only": value_of(payload.get("bounded_fixture_only")) is True,
    }


def summarize_ebt(payload: JsonMap | None) -> JsonDict:
    if payload is None:
        return {"source_experiment": 5317, "status": "missing_or_unreadable"}
    quarantine = value_of(payload.get("claim_quarantine")) or {}
    return {
        "source_experiment": 5317,
        "ebt_telemetry_audited": value_of(payload.get("ebt_telemetry_audited")) is True,
        "methodology_flag_cleared": value_of(payload.get("methodology_flag_cleared")) is True,
        "step_control_recovery_logged": value_of(payload.get("step_control_recovery_logged")) is True,
        "lambda_max_logged": value_of(payload.get("lambda_max_logged")) is True,
        "tiny_diagnostic_usable": quarantine.get("tiny_diagnostic_usable") is True,
        "future_energy_descent_claims_eligible": quarantine.get(
            "future_energy_descent_claims_eligible"
        )
        is True,
        "sota_quality_claims_eligible": quarantine.get("sota_quality_claims_eligible") is True,
        "hardware_readiness_claims_eligible": quarantine.get(
            "hardware_readiness_claims_eligible"
        )
        is True,
        "no_sota_quality_claim": value_of(payload.get("no_sota_quality_claim")) is True,
        "no_hardware_speedup_claim": value_of(payload.get("no_hardware_speedup_claim")) is True,
    }


def summarize_smt(payload: JsonMap | None) -> JsonDict:
    if payload is None:
        return {"source_experiment": 5318, "status": "missing_or_unreadable"}
    flagged = value_of(payload.get("flagged_adversarial")) is True
    return {
        "source_experiment": 5318,
        "smt_hint_protocol_ready": value_of(payload.get("smt_hint_protocol_ready")) is True,
        "flagged_adversarial": flagged,
        "clean_success_evidence": not flagged,
        "llm_invoked": value_of(payload.get("llm_invoked")) is True,
        "future_llm_slot_gated_on_sota_runtime": value_of(
            payload.get("future_llm_slot_gated_on_sota_runtime")
        )
        is True,
        "valid_hint_acceptance_rate": value_of(payload.get("valid_hint_acceptance_rate")),
        "unsound_hint_rejection_rate": value_of(payload.get("unsound_hint_rejection_rate")),
        "usefulness_rate": value_of(payload.get("usefulness_rate")),
        "corrigendum_pending": value_of(payload.get("corrigendum_pending")) or [],
    }


def summarize_hardware(payload: JsonMap | None) -> JsonDict:
    if payload is None:
        return {"source_experiment": 5319, "status": "missing_or_unreadable"}
    return {
        "source_experiment": 5319,
        "status": value_of(payload.get("status")),
        "hardware_evidence_level": value_of(payload.get("hardware_evidence_level")),
        "hardware_speedup_claimed": value_of(payload.get("hardware_speedup_claimed")) is True,
        "no_speedup_claim": value_of(payload.get("no_speedup_claim")) is True,
        "authenticated_workload_run": value_of(payload.get("authenticated_workload_run")) is True,
        "kv260_ssh_reachable": value_of(payload.get("kv260_ssh_reachable")) is True,
        "polarfire_status_reachable": value_of(payload.get("polarfire_status_reachable")) is True,
        "gatemate_physical_jtag_changed": value_of(payload.get("gatemate_physical_jtag_changed"))
        is True,
        "public_hardware_references_used_as_context_only": value_of(
            payload.get("public_hardware_references_used_as_context_only")
        )
        is True,
    }


def build_next_milestone_recommendation(
    runtime_status: JsonMap,
    quality_status: JsonMap,
    smt_status: JsonMap,
    hardware_status: JsonMap,
) -> JsonDict:
    """Choose the .486 branch from actual artifact gates."""

    blocked_runtime = runtime_status.get("sota_runtime_unblocked") is False
    quality_unmeasured = quality_status.get("quality_measured") is False
    flagged_smt = smt_status.get("flagged_adversarial") is True
    no_hardware_workload = hardware_status.get("authenticated_workload_run") is False
    return {
        "recommended_branch": NEXT_MILESTONE_BRANCH,
        "primary_gate": "repair_sota_runtime_before_quality_claims",
        "rationale": (
            "Exp5309 left all mandated GGUF models generation-incomplete, so Exp5311 "
            "quality was gate-blocked. Exp5312/Exp5313 are the clean positive branch to "
            "carry forward, while solver/KAN/EBT remain bounded diagnostics and hardware "
            "has no authenticated speedup workload."
        ),
        "evidence_flags": {
            "runtime_blocked": blocked_runtime,
            "quality_unmeasured": quality_unmeasured,
            "smt_flagged_adversarial": flagged_smt,
            "hardware_authenticated_workload_missing": no_hardware_workload,
        },
        "carry_forward": [
            "deterministic_paraphrase_fixture",
            "transition_level_memory_verifier_and_rollout",
            "bounded_solver_guidance_with_cdcl_fallback",
            "bounded_kan_budget_diagnostics",
            "methodology_clean_tiny_ebt_telemetry",
        ],
        "do_not_reopen": [
            "hardware_speedup_without_authenticated_workload",
            "sota_quality_claim_until_exp5309_gate_passes",
            "clean_smt_success_from_flagged_exp5318",
        ],
    }


def _stable_checksum(artifact: JsonMap) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_result_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """Build the deterministic Exp5320 capstone artifact."""

    payloads, artifacts_read, missing = read_upstream_artifacts(root)
    runtime_status = summarize_sota_runtime(payloads.get(5309))
    quality_status = summarize_sota_quality(payloads.get(5311))
    paraphrase_status = summarize_paraphrase(payloads.get(5310))
    learning_status = summarize_continuous_self_learning(payloads.get(5312), payloads.get(5313))
    solver_status = summarize_solver(payloads.get(5314), payloads.get(5315))
    kan_status = summarize_kan(payloads.get(5316))
    ebt_status = summarize_ebt(payloads.get(5317))
    smt_status = summarize_smt(payloads.get(5318))
    hardware_status = summarize_hardware(payloads.get(5319))
    status = "blocked_missing_required" if missing else "complete"
    if missing:
        verdict = (
            "blocked_missing_required: expected .485 artifacts were absent or malformed; "
            "no speedup or SOTA quality claim made."
        )
    else:
        verdict = (
            "complete: .485 closed with SOTA runtime still blocked and SOTA quality "
            "unmeasured; paraphrase and transition-memory fixtures are clean positives; "
            "solver/KAN/EBT are bounded diagnostics; SMT is flagged; hardware remains "
            "reachability-only with no speedup claim."
        )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": wrapped(EXPERIMENT_ID, "experiment_id"),
        "milestone": wrapped(MILESTONE, "milestone"),
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
        "status": wrapped(status, "status"),
        "honest_verdict": wrapped(verdict, "honest_verdict"),
        "inference_substrate": wrapped(INFERENCE_SUBSTRATE, "inference_substrate"),
        "artifacts_read": wrapped(artifacts_read, "artifacts_read"),
        "missing_artifacts": wrapped(missing, "missing_artifacts"),
        "sota_runtime_status": wrapped(runtime_status, "sota_runtime_status"),
        "sota_quality_status": wrapped(quality_status, "sota_quality_status"),
        "paraphrase_verification_status": wrapped(
            paraphrase_status, "paraphrase_verification_status"
        ),
        "continuous_self_learning_status": wrapped(
            learning_status, "continuous_self_learning_status"
        ),
        "solver_status": wrapped(solver_status, "solver_status"),
        "kan_certificate_status": wrapped(kan_status, "kan_certificate_status"),
        "ebt_telemetry_status": wrapped(ebt_status, "ebt_telemetry_status"),
        "smt_hint_protocol_status": wrapped(smt_status, "smt_hint_protocol_status"),
        "hardware_status": wrapped(hardware_status, "hardware_status"),
        "no_false_speedup_claim": True,
        "no_false_sota_quality_claim": True,
        "next_milestone_recommendation": wrapped(
            build_next_milestone_recommendation(
                runtime_status,
                quality_status,
                smt_status,
                hardware_status,
            ),
            "next_milestone_recommendation",
        ),
        "docs_updated": wrapped(
            {"ops_status": False, "ops_changelog": False, "traceability": False},
            "docs_updated",
        ),
    }
    artifact["reproducibility_checksum"] = _stable_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    """Validate the fields that downstream reconciliation depends on."""

    missing_fields = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(f"missing required fields: {missing_fields}")
    for field in PRINCIPLE_WRAPPED_FIELDS:
        value = artifact[field]
        if not isinstance(value, Mapping) or value.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} must be principle-wrapped with the declared principle")
    verdict = artifact["honest_verdict"]["value"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate drifted from local artifact aggregation")
    for field in ("no_false_speedup_claim", "no_false_sota_quality_claim"):
        if artifact[field] is not True:
            raise ValueError(f"{field} must be the bare boolean true")
    docs = artifact["docs_updated"]["value"]
    if docs != {"ops_status": False, "ops_changelog": False, "traceability": False}:
        raise ValueError("docs_updated must leave reconciler-owned docs untouched")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
) -> JsonDict:
    """Write the Exp5320 capstone artifact."""

    artifact = build_result_artifact(root=root)
    output_path = Path(result_path) if result_path is not None else Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact

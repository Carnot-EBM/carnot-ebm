"""Tests for Exp 5094 .467 capstone aggregation.

Spec refs: REQ-CAPSTONE-5094, SCENARIO-CAPSTONE-5094,
SCENARIO-CAPSTONE-5094-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5094_capstone_v467 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


class FlatClock:
    """SCENARIO-CAPSTONE-5094 clock keeps duration deterministic."""

    def __call__(self) -> float:
        return 5094.0


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_default_upstreams(root: Path) -> None:
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5083].relative_path,
        {
            "honest_verdict": "complete_466_archived_467_activated_endpoint_blockers_carried_forward",
            "flagged_adversarial": False,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5084].relative_path,
        {
            "honest_verdict": "success_sota_ingestion_v467_references_verified",
            "flagged_adversarial": False,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5085].relative_path,
        {
            "honest_verdict": "success_llamacpp_logprob_endpoint_ready",
            "flagged_adversarial": True,
            "completion_endpoint_ready": True,
            "logprob_endpoint_ready": True,
            "top_logprob_or_confidence_ready": True,
            "live_completion_invoked": True,
            "usable_sota_models": [{"hf_id": "model-a"}],
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5086].relative_path,
        {
            "honest_verdict": "blocked_uprm_logprob_cache_retry_endpoint_failed",
            "flagged_adversarial": False,
            "logprob_cache_ready": False,
            "step_cache_ready": False,
            "endpoint_used": "http://127.0.0.1:8001",
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5087].relative_path,
        {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5086-uprm-logprob-cache-retry-v467.logprob_cache_ready "
                "(actual=False == expected=True)"
            ),
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5088].relative_path,
        {
            "honest_verdict": "complete_temporal_consistency_prm_no_win",
            "flagged_adversarial": True,
            "beats_one_pass": False,
            "delta_vs_one_pass": 0.0,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5089].relative_path,
        {
            "honest_verdict": "complete_pbit_guided_cdcl_distribution_sensitive_no_win",
            "flagged_adversarial": True,
            "correctness_preserved": True,
            "helps_declared_family": False,
            "delta_effort_vs_pure": {"pbit_guided": 0, "random_assumption": 3},
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5090].relative_path,
        {
            "honest_verdict": "success_static_csr_masks_speedup_and_validity_win",
            "flagged_adversarial": True,
            "beats_cpu_trie": True,
            "beats_rerank_only_on_validity_or_cost": True,
            "mask_equivalence_rate": 1.0,
            "mask_speedup": 77.42,
            "validity_rate": 1.0,
            "rerank_only_validity_rate": 0.666667,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5091].relative_path,
        {
            "honest_verdict": "success_kan_pwa_milp_scale_property_verified_small",
            "flagged_adversarial": False,
            "property_holds": True,
            "property_status": "verified",
            "abstraction_built": True,
            "solver_available": True,
            "solver_status": "optimal",
            "binary_variable_count": 6,
            "pwa_piece_count": 6,
            "constraint_count": 43,
            "global_error_bound": 0.0,
            "solve_time_s": 0.008194,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5092].relative_path,
        {
            "honest_verdict": "complete_fr11_budgeted_onpolicy_memory_guarded_no_promote_delta_plus_0p000",
            "flagged_adversarial": False,
            "fr11_attempt_completed": True,
            "heldout_delta": 0.0,
            "nonforgetting_delta": 0.0,
            "contamination_guard_passed": True,
            "poison_guard_passed": True,
            "rollback_guard_passed": True,
            "promoted_count": 0,
            "promotion_decision": {
                "promoted": False,
                "no_promote_reason": "positive_utility_not_observed",
                "gate_conditions": {
                    "positive_utility_gt_zero": False,
                    "heldout_delta_gte_zero": True,
                    "nonforgetting_delta_gte_zero": True,
                    "poison_guard_passed": True,
                    "contamination_guard_passed": True,
                    "rollback_guard_passed": True,
                },
            },
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5093].relative_path,
        {
            "honest_verdict": "complete_hardware_continuity_v467_partial_board_blockers",
            "flagged_adversarial": False,
            "kv260_ssh_ready": True,
            "kv260_uio_transcript_path": None,
            "kv260_speedup_claim_allowed": False,
            "gatemate_detected": False,
            "gatemate_terminal_state": "blocked_gatemate_dirtyjtag_cable_seen_no_gatemate_idcode_terminal",
            "polarfire_detected": True,
            "polarfire_dispatch_precheck_ready": True,
            "destructive_actions_taken": [],
        },
    )


def test_req_capstone_5094_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-5094: OpenSpec anchors the ungated .467 capstone."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5094",
        "SCENARIO-CAPSTONE-5094",
        "SCENARIO-CAPSTONE-5094-FIELD-PRINCIPLES",
        "experiment_5094_capstone_v467.py",
        "results/experiment_5094_capstone_v467.json",
        "exact_verifier_pivot_positive",
        "complete_capstone_v467_exact_verifier_pivot_positive_runtime_process_blocked",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_5094_aggregates_clean_exact_pivot(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5094: clean KAN/MILP evidence drives the decision."""

    _write_default_upstreams(tmp_path)

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH, clock=FlatClock())

    assert artifact["honest_verdict"] == (
        "complete_capstone_v467_exact_verifier_pivot_positive_runtime_process_blocked"
    )
    assert artifact["duration_s"] == 0.0001
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "live_llm_inference" not in json.dumps(artifact)
    assert artifact["milestone_decision"] == "exact_verifier_pivot_positive"
    assert len(artifact["artifacts_loaded"]) == 11
    assert artifact["missing_artifacts"] == []
    assert {row["experiment_id"] for row in artifact["blocked_artifacts"]} == {5086, 5087}
    assert {row["experiment_id"] for row in artifact["flagged_upstream_artifacts"]} == {
        5085,
        5088,
        5089,
        5090,
    }

    runtime = artifact["runtime_state"]
    assert runtime["reported_completion_endpoint_ready"] is True
    assert runtime["reported_logprob_endpoint_ready"] is True
    assert runtime["headline_runtime_ready"] is False
    assert runtime["runtime_claim_excluded_reason"] == "upstream_flagged_adversarial"
    assert runtime["process_substrate_blocked"] is True

    process = artifact["process_verifier_state"]
    assert process["process_verifier_ready"] is False
    assert process["process_verifier_win"] is False
    assert process["uprm_process_retry_blocked"] is True

    exact = artifact["exact_verifier_state"]
    assert exact["path_worth_scaling"] is True
    assert exact["pbit_cdcl"]["excluded_from_headline"] is True
    assert exact["kan_milp"]["clean_positive"] is True

    constrained = artifact["constrained_generation_state"]
    assert constrained["clean_headline"] is False
    assert constrained["reported_mask_speedup"] == 77.42
    assert constrained["excluded_from_headline"] is True

    kan = artifact["kan_formal_state"]
    assert kan["property_holds"] is True
    assert kan["binary_variable_count"] == 6
    assert kan["scale_boundary"] == "small_multi_unit_property_not_architecture_scale_claim"

    fr11 = artifact["fr11_state"]
    assert fr11["safe_governed_mechanism"] is True
    assert fr11["positive_utility_observed"] is False
    assert fr11["promoted"] is False

    hardware = artifact["hardware_state"]
    assert hardware["clean_continuity_state"] is True
    assert hardware["kv260_ssh_ready"] is True
    assert hardware["gatemate_detected"] is False
    assert hardware["speedup_claim_allowed"] is False

    assert artifact["docs_updated"] == ["openspec/capabilities/capstone/spec.md"]
    assert artifact["flagged_adversarial"] is False
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_missing_exact_artifact_falls_back_to_hardware_continuity(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5094: missing artifacts are explicit, not headlined."""

    _write_default_upstreams(tmp_path)
    (tmp_path / mod.UPSTREAMS_BY_ID[5091].relative_path).unlink()

    artifact = mod.build_artifact(root=tmp_path, clock=FlatClock())

    assert artifact["milestone_decision"] == "hardware_continuity_only"
    assert artifact["honest_verdict"] == "complete_capstone_v467_hardware_continuity_only"
    assert {row["experiment_id"] for row in artifact["missing_artifacts"]} == {5091}
    assert artifact["exact_verifier_state"]["path_worth_scaling"] is False
    assert artifact["hardware_state"]["clean_continuity_state"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_decision_and_schema_error_helpers_cover_terminal_classes() -> None:
    """SCENARIO-CAPSTONE-5094-FIELD-PRINCIPLES: decisions and schema fail closed."""

    assert mod.honest_verdict_for_decision("runtime_repaired_process_verifier_ready") == (
        "complete_capstone_v467_runtime_repaired_process_verifier_ready"
    )
    assert mod.honest_verdict_for_decision("exact_verifier_pivot_positive") == (
        "complete_capstone_v467_exact_verifier_pivot_positive_runtime_process_blocked"
    )
    assert mod.honest_verdict_for_decision("fr11_governed_positive") == (
        "complete_capstone_v467_fr11_governed_positive_no_promotion"
    )
    assert mod.honest_verdict_for_decision("hardware_continuity_only") == (
        "complete_capstone_v467_hardware_continuity_only"
    )
    assert mod.honest_verdict_for_decision("execution_incomplete_endpoint_blocked") == (
        "complete_capstone_v467_execution_incomplete_endpoint_blocked"
    )
    assert mod.honest_verdict_for_decision("bounded_no_headline") == (
        "complete_capstone_v467_bounded_no_headline"
    )

    assert (
        mod.choose_milestone_decision(
            runtime_state={"runtime_ready": True},
            process_verifier_state={"process_verifier_ready": True},
            exact_verifier_state={"path_worth_scaling": False},
            fr11_state={"safe_governed_mechanism": False, "positive_utility_observed": False},
            hardware_state={"clean_continuity_state": False},
        )
        == "runtime_repaired_process_verifier_ready"
    )
    assert (
        mod.choose_milestone_decision(
            runtime_state={"runtime_ready": False, "endpoint_or_cache_blocked": True},
            process_verifier_state={"process_verifier_ready": False},
            exact_verifier_state={"path_worth_scaling": False},
            fr11_state={"safe_governed_mechanism": True, "positive_utility_observed": True},
            hardware_state={"clean_continuity_state": False},
        )
        == "fr11_governed_positive"
    )
    assert (
        mod.choose_milestone_decision(
            runtime_state={"runtime_ready": False, "endpoint_or_cache_blocked": True},
            process_verifier_state={"process_verifier_ready": False},
            exact_verifier_state={"path_worth_scaling": False},
            fr11_state={"safe_governed_mechanism": False, "positive_utility_observed": False},
            hardware_state={"clean_continuity_state": False},
        )
        == "execution_incomplete_endpoint_blocked"
    )
    assert (
        mod.choose_milestone_decision(
            runtime_state={"runtime_ready": False, "endpoint_or_cache_blocked": False},
            process_verifier_state={"process_verifier_ready": False},
            exact_verifier_state={"path_worth_scaling": False},
            fr11_state={"safe_governed_mechanism": False, "positive_utility_observed": False},
            hardware_state={"clean_continuity_state": False},
        )
        == "bounded_no_headline"
    )

    invalid = {
        "honest_verdict": "not_terminal",
        "inference_substrate": "wrong",
        "milestone_decision": "invalid",
        "docs_updated": "nope",
        "flagged_adversarial": True,
        "runtime_state": "live_llm_inference",
    }
    errors = mod.artifact_schema_errors(invalid)
    assert "missing.duration_s" in errors
    assert "honest_verdict.not_terminal" in errors
    assert "inference_substrate.not_aggregation" in errors
    assert "milestone_decision.invalid" in errors
    assert "docs_updated.not_list" in errors
    assert "flagged_adversarial.must_be_false" in errors
    assert "forbidden.live_llm_inference_claim" in errors

"""Tests for Exp 5107 .468 capstone aggregation.

Spec refs: REQ-CAPSTONE-5107, SCENARIO-CAPSTONE-5107,
SCENARIO-CAPSTONE-5107-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5107_capstone_v468 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


class FlatClock:
    """SCENARIO-CAPSTONE-5107 clock keeps duration deterministic."""

    def __call__(self) -> float:
        return 5107.0


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_default_upstreams(root: Path) -> None:
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5095].relative_path,
        {
            "honest_verdict": "complete_467_archived_468_activated_exact_verifier_pivot_carried_forward",
            "flagged_adversarial": False,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "exact_verifier_pivot": {"clean_positive": True},
            "docs_updated": [],
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5096].relative_path,
        {
            "honest_verdict": "success_sota_ingestion_v468_references_verified",
            "flagged_adversarial": False,
            "inference_substrate": "literature_review_and_repo_inspection",
            "sources_checked": [{"source_id": "beaver_prefix_bounds"}],
            "task_mapping": [{"task_id": "exp5099"}],
            "planning_hooks": [{"hook_id": "exp5101_graph_evidence_energy"}],
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5097].relative_path,
        {
            "honest_verdict": "blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs",
            "flagged_adversarial": False,
            "inference_substrate": "precondition_check_only",
            "logprob_endpoint_clean": False,
            "logprob_endpoint_ready": False,
            "completion_endpoint_ready": False,
            "live_llm_invoked": False,
            "endpoint_url": "http://127.0.0.1:58385",
            "model_specs": {"mandatory_models": [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}]},
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5098].relative_path,
        {
            "honest_verdict": "success_kan_pwa_milp_scale_v2_property_suite_clean",
            "flagged_adversarial": False,
            "inference_substrate": "exact_milp_solver_cpu",
            "properties_proved": ["exp5091_baseline_two_unit_true", "three_unit_composition_true"],
            "false_property_controls_passed": True,
            "max_scale_reached": {
                "property_id": "three_unit_composition_true",
                "binary_variable_count": 9,
                "constraint_count": 64,
                "solver_status": "optimal",
            },
            "scale_blocker": None,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5099].relative_path,
        {
            "honest_verdict": "complete_beaver_prefix_bounds_toy_only_runtime_not_clean",
            "flagged_adversarial": True,
            "inference_substrate": "deterministic_toy_finite_distribution",
            "backend_used": "toy_distribution",
            "soundness_checks_passed": True,
            "lower_bound": 0.0,
            "upper_bound": 0.222222222222,
            "bound_gap": 0.222222222222,
            "live_llm_invoked": False,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5100].relative_path,
        {
            "honest_verdict": "success_constrainprompt_code_assurance_exact_checks_passed",
            "flagged_adversarial": True,
            "inference_substrate": "deterministic_python_json_logical_tree",
            "exact_checker_backend": "python_json_logical_tree",
            "constraints_total": 9,
            "executable_constraints_total": 9,
            "positive_tests_passed": True,
            "negative_tests_passed": True,
            "adversarial_tests_passed": True,
            "llm_invoked": False,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5101].relative_path,
        {
            "honest_verdict": "success_graph_evidence_energy_separates_contradiction_from_unsupported",
            "flagged_adversarial": False,
            "inference_substrate": "synthetic_graph_exact_labels",
            "contradiction_reject_rate": 1.0,
            "unsupported_retained_rate": 1.0,
            "supported_accept_rate": 1.0,
            "stability_under_perturbation": {"passed": True},
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5102].relative_path,
        {
            "honest_verdict": "success_hubo_pspin_direct_encoding_reduces_gadget_blowup",
            "flagged_adversarial": False,
            "inference_substrate": "exact_enumeration_cpu",
            "direct_hubo_advantage": True,
            "exact_optima_verified": True,
            "auxiliary_variable_blowup": {"mean_qubo_to_hubo_variable_ratio": 1.95},
            "energy_scale_distortion": {"mean_qubo_to_hubo_coefficient_ratio": 274.0},
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5103].relative_path,
        {
            "honest_verdict": "success_taco_adaptive_heuristic_reduces_exact_solver_effort",
            "flagged_adversarial": False,
            "inference_substrate": "exact_solver_with_adaptive_cpu_heuristic",
            "correctness_preserved": True,
            "delta_effort_vs_baseline": {"adapted": -106854},
            "baseline_effort": {"total_effort_score": 108210},
            "adapted_effort": {"total_effort_score": 1356},
            "harmful_instance_count": 2,
            "instances_total": 8,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5104].relative_path,
        {
            "honest_verdict": "complete_constrained_decoding_semantic_audit_no_syntax_only_headline",
            "flagged_adversarial": True,
            "inference_substrate": "deterministic_static_csr_semantic_distribution_audit",
            "syntax_only_headline_forbidden": True,
            "syntax_validity_rate": 1.0,
            "semantic_validity_rate": 0.436364,
            "distribution_shift_metric": 0.563636,
            "live_llm_invoked": False,
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5105].relative_path,
        {
            "honest_verdict": "complete_fr11_severa_guarded_memory_no_promote_contracts_working_delta_plus_0p000",
            "flagged_adversarial": True,
            "inference_substrate": "exact_guarded_self_learning_eval",
            "heldout_delta": 0.0,
            "nonforgetting_delta": 0.0,
            "promoted_count": 0,
            "contract_pass_count": 3,
            "poison_guard_passed": True,
            "contamination_guard_passed": True,
            "rollback_guard_passed": True,
            "promotion_decision": {
                "promoted": False,
                "no_promote_reason": "positive_utility_not_observed",
            },
        },
    )
    _write_json(
        root / mod.UPSTREAMS_BY_ID[5106].relative_path,
        {
            "honest_verdict": "complete_hardware_partition_telemetry_no_speedup_claim",
            "flagged_adversarial": False,
            "inference_substrate": "hardware_smoke_and_static_mapping",
            "kv260_ssh_ready": True,
            "kv260_uio_transcript_collected": False,
            "kv260_blocker": "no_safe_kv260_uio_register_transcript_collected",
            "gatemate_detected": False,
            "gatemate_terminal_state": "blocked_gatemate_dirtyjtag_cable_seen_no_gatemate_idcode_terminal",
            "polarfire_ssh_ready": True,
            "polarfire_dispatch_precheck": {"ready": True},
            "speedup_claimed": False,
            "destructive_actions_taken": [],
            "partition_telemetry": [{"mapping_kind": "p_spin_hubo"}],
        },
    )


def test_req_capstone_5107_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-5107: OpenSpec anchors the .468 capstone."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5107",
        "SCENARIO-CAPSTONE-5107",
        "SCENARIO-CAPSTONE-5107-FIELD-PRINCIPLES",
        "experiment_5107_capstone_v468.py",
        "results/experiment_5107_capstone_v468.json",
        "complete_capstone_v468_exact_verifier_scale_decision_recorded",
        "Exp5099/Exp5100/Exp5104/Exp5105 as adversarially flagged",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_5107_aggregates_clean_scaleup_and_exclusions(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5107: clean evidence drives the .468 decision."""

    _write_default_upstreams(tmp_path)

    artifact = mod.run(
        root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH, clock=FlatClock()
    )

    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["duration_s"] == 0.0001
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "live_llm_inference" not in json.dumps(artifact)
    assert artifact["docs_updated"] == []
    assert artifact["flagged_adversarial"] is False

    assert {row["experiment_id"] for row in artifact["source_artifacts"]} == set(
        mod.UPSTREAMS_BY_ID
    )
    assert artifact["missing_artifacts"] == []
    assert {row["experiment_id"] for row in artifact["clean_positive_artifacts"]} == {
        5095,
        5096,
        5098,
        5101,
        5102,
        5103,
    }
    assert {row["experiment_id"] for row in artifact["clean_negative_artifacts"]} == {5106}
    assert {row["experiment_id"] for row in artifact["blocked_artifacts"]} == {5097}
    assert {row["experiment_id"] for row in artifact["flagged_artifacts"]} == {
        5099,
        5100,
        5104,
        5105,
    }

    milestone = artifact["milestone_decision"]
    assert milestone["decision"] == (
        "bounded_exact_verifier_scale_up_clean_runtime_blocked_fr11_no_clean_promotion_"
        "hardware_continuity_no_speedup"
    )
    assert milestone["clean_exact_verifier_scale_up"] is True
    assert milestone["clean_runtime_substrate"] is False
    assert milestone["safe_fr11_promotion"] is False
    assert milestone["constrained_generation_evidence_worth_keeping"] is False
    assert milestone["hardware_progress"] == "continuity_progress_no_speedup"
    assert milestone["flagged_artifact_count"] == 4
    assert milestone["blocked_artifact_count"] == 1
    assert milestone["missing_artifact_count"] == 0

    exact = artifact["exact_verifier_decision"]
    assert exact["decision"] == "clean_bounded_exact_verifier_scale_up_with_flagged_exclusions"
    assert exact["clean_scale_up"] is True
    assert exact["non_llm_exact_claims_not_blocked_by_exp5097"] is True
    assert exact["kan_milp"]["max_scale_reached"]["binary_variable_count"] == 9
    assert exact["kan_milp"]["false_property_controls_passed"] is True
    assert exact["graph_evidence"]["contradiction_reject_rate"] == 1.0
    assert exact["hubo_pspin"]["direct_hubo_advantage"] is True
    assert exact["adaptive_solver"]["correctness_preserved"] is True
    assert exact["beaver_prefix_bounds"]["excluded_from_headline"] is True
    assert exact["code_assurance"]["excluded_from_headline"] is True

    runtime = artifact["runtime_substrate_decision"]
    assert runtime["decision"] == "blocked_no_clean_live_logprob_substrate"
    assert runtime["logprob_endpoint_clean"] is False
    assert runtime["does_not_gate_non_llm_exact_verifiers"] is True

    fr11 = artifact["fr11_decision"]
    assert fr11["decision"] == "no_clean_fr11_promotion_flagged_artifact_requires_rerun"
    assert fr11["promotion_allowed_from_clean_evidence"] is False
    assert fr11["upstream_excluded_from_headline"] is True

    constrained = artifact["constrained_generation_decision"]
    assert constrained["decision"] == "no_clean_constrained_generation_headline_flagged_audit_only"
    assert constrained["syntax_only_headline_forbidden"] is True

    hardware = artifact["hardware_decision"]
    assert hardware["decision"] == "hardware_continuity_progress_no_speedup"
    assert hardware["kv260_ssh_ready"] is True
    assert hardware["speedup_claimed"] is False

    assert any(q["source"] == "Exp5097" for q in artifact["next_research_questions"])
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_5107_missing_and_unloadable_sources_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5107: missing inputs are listed, not inferred."""

    _write_default_upstreams(tmp_path)
    (tmp_path / mod.UPSTREAMS_BY_ID[5102].relative_path).unlink()
    (tmp_path / mod.UPSTREAMS_BY_ID[5103].relative_path).write_text("{", encoding="utf-8")

    artifact = mod.run(
        root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH, clock=FlatClock()
    )

    assert {row["experiment_id"] for row in artifact["missing_artifacts"]} == {5102, 5103}
    assert all(row["classification"] == "missing" for row in artifact["missing_artifacts"])
    assert artifact["milestone_decision"]["missing_artifact_count"] == 2
    assert artifact["exact_verifier_decision"]["clean_scale_up"] is True
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"

    bad_list = tmp_path / "bad-list.json"
    bad_list.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(bad_list)[1]["error"] == "json_not_object"

    assert mod._mapping([]) == {}
    assert mod._list("bad") == []
    assert mod._bool(None) is False
    assert mod._number(True) is None
    assert mod._number("bad") is None


def test_scenario_capstone_5107_validation_and_main_paths(
    tmp_path: Path, capsys: Any, monkeypatch: Any
) -> None:
    """SCENARIO-CAPSTONE-5107-FIELD-PRINCIPLES: schema drift fails closed."""

    _write_default_upstreams(tmp_path)
    artifact = mod.run(
        root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH, clock=FlatClock()
    )

    invalid = json.loads(json.dumps(artifact))
    invalid["honest_verdict"] = "bad"
    invalid["inference_substrate"] = "live_llm_inference"
    invalid["docs_updated"] = ["ops/status.md"]
    invalid["flagged_adversarial"] = True
    invalid["reproducibility_checksum"] = "bad"
    invalid["milestone_decision"]["clean_exact_verifier_scale_up"] = False
    invalid["exact_verifier_decision"]["clean_scale_up"] = False
    invalid["runtime_substrate_decision"]["does_not_gate_non_llm_exact_verifiers"] = False
    invalid["fr11_decision"]["promotion_allowed_from_clean_evidence"] = True
    invalid["hardware_decision"]["speedup_claimed"] = True
    invalid["source_artifacts"][0].pop("classification")
    del invalid["duration_s"]

    errors = mod.artifact_schema_errors(invalid)
    assert "missing.duration_s" in errors
    assert "honest_verdict.not_terminal" in errors
    assert "inference_substrate.not_aggregation" in errors
    assert "docs_updated.not_deferred" in errors
    assert "flagged_adversarial.must_be_false" in errors
    assert "source_artifacts.missing_classification" in errors
    assert "milestone_decision.invalid" in errors
    assert "exact_verifier_decision.invalid" in errors
    assert "runtime_substrate_decision.invalid" in errors
    assert "fr11_decision.invalid" in errors
    assert "hardware_decision.invalid" in errors
    assert "forbidden.live_llm_inference_claim" in errors
    assert "reproducibility_checksum.invalid" in errors

    assert mod.main(root=tmp_path, artifact_path=tmp_path / "out.json", clock=FlatClock()) == 0
    captured = capsys.readouterr()
    assert "experiment_5107_capstone_v468" in captured.out

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced_schema_error"])
    assert mod.main(root=tmp_path, artifact_path=tmp_path / "bad-out.json", clock=FlatClock()) == 1
    captured = capsys.readouterr()
    assert "forced_schema_error" in captured.out

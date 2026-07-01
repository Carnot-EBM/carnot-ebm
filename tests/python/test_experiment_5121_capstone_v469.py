"""Tests for Exp 5121 ungated .469 capstone aggregation.

Spec refs: REQ-CAPSTONE-5121, SCENARIO-CAPSTONE-5121,
SCENARIO-CAPSTONE-5121-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5121_capstone_v469 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_adversarial_report(path: Path) -> JsonDict:
    if "5119" in path.name:
        return {
            "loaded": True,
            "flag_count": 1,
            "max_severity": 2,
            "flags": [
                {
                    "kind": "DURATION_TOO_SHORT",
                    "severity": "critical",
                    "detail": "duration_s below live model floor",
                }
            ],
        }
    return {"loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}


def _write_default_upstreams(root: Path) -> None:
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5109].relative_path,
        {
            "experiment_id": "exp5109-archive-468-activate-469",
            "honest_verdict": "blocked_research_roadmap_next_missing",
            "flagged_adversarial": False,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "duration_s": 2.3,
            "roadmap_next_present": False,
            "active_roadmap_modified": False,
            "conductor_modified": False,
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5110].relative_path,
        {
            "experiment_id": "exp5110-source-freshness-sota-ingestion-v469",
            "honest_verdict": "complete_source_freshness_sota_ingestion_v469_references_mapped",
            "flagged_adversarial": False,
            "inference_substrate": "literature_review_and_repo_inspection",
            "duration_s": 0.01,
            "references_section_found": True,
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5111].relative_path,
        {
            "experiment_id": "exp5111-fover-in-domain-pool-v469",
            "honest_verdict": (
                "blocked_fover_indomain_pool_retracted_see_experiment_"
                "fover_stepverifier_vs_cheap_baseline"
            ),
            "flagged_adversarial": False,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "duration_s": 0.01,
            "pool_n": 0,
            "candidates_per_item": 0,
            "headroom_present": False,
            "verifier_is_oracle": False,
            "corrected_result_summary": {
                "verifier_auroc": 0.9663,
                "cheap_baseline_auroc": 0.9635,
                "delta_auroc": 0.0028,
                "delta_auroc_ci95": [-0.0244, 0.0347],
                "beats_cheap_baseline": False,
            },
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5112].relative_path,
        {
            "experiment": 5112,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "2 of 2 gate(s) failed",
            "gates_evaluated": [
                {"artifact_field": "pool_n", "actual": 0, "expected": 150, "passed": False}
            ],
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5114].relative_path,
        {
            "experiment_id": "exp5114-kan-abstraction-refinement-post-wall-v469",
            "honest_verdict": "success_kan_abstraction_refinement_post_wall_progress_n100",
            "flagged_adversarial": False,
            "inference_substrate": "kan_abstraction_refinement_cpu",
            "duration_s": 1.8,
            "technique_changed_from_exp5108": True,
            "exp5108_baseline_loaded": True,
            "solved_n": 100,
            "attempted_n": 100,
            "post_wall_progress": True,
            "certificate_soundness": True,
            "false_property_detected": True,
            "near_margin_abstained": True,
            "exp5108_baseline": {
                "largest_n_reached": 10,
                "solver_timeout_hit": True,
                "timed_out_n": 20,
            },
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5115].relative_path,
        {
            "experiment": 5115,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "1 of 1 gate(s) failed",
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5116].relative_path,
        {
            "experiment_id": "exp5116-hubo-2dpt-sampling-reference-v469",
            "honest_verdict": "complete_hubo_2dpt_reference_ready_exact_checked_cpu",
            "flagged_adversarial": False,
            "inference_substrate": "cpu_hubo_2d_parallel_tempering_reference",
            "duration_s": 0.6,
            "hubo_2dpt_reference_ready": True,
            "exact_enumeration_checked": True,
            "optimum_hit_rate": {"two_d_beta_penalty_pt": 1.0, "unguided_gibbs": 0.5},
            "hardware_speedup_claimed": False,
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5117].relative_path,
        {
            "experiment_id": "exp5117-taco-harm-gated-scale-v469",
            "honest_verdict": "success_taco_harm_gate_ready_exact_labels_preserved",
            "flagged_adversarial": False,
            "inference_substrate": "exact_solver_with_harm_gated_adaptive_cpu_heuristic",
            "duration_s": 0.05,
            "taco_harm_gate_ready": True,
            "wrong_label_count": 0,
            "average_effort_reduction_ratio_guarded": 0.963671,
            "harmful_instance_count_guarded": 4,
            "harmful_instance_count_unguarded": 7,
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5119].relative_path,
        {
            "experiment_id": "exp5119-sota-endpoint-rootcause-v469",
            "honest_verdict": "blocked_sota_endpoint_rootcause_adversarial_flag",
            "flagged_adversarial": True,
            "inference_substrate": "live_llm_inference",
            "duration_s": 30.5,
            "adversarial_verify_passed": False,
            "cache_ready": False,
            "completion_proof": {"ready": True},
            "logprob_proof": {"ready": True},
            "root_cause_tree": {"summary": "blocked_adversarial_verify"},
        },
    )
    _write_json(
        root / mod.EXPECTED_UPSTREAMS[5120].relative_path,
        {
            "experiment_id": "exp5120-hardware-residual-telemetry-v469",
            "honest_verdict": "complete_hardware_residual_telemetry_cpu_reference_no_speedup_claim",
            "flagged_adversarial": False,
            "inference_substrate": "hardware_smoke_and_residual_telemetry_or_cpu_fallback",
            "duration_s": 9.4,
            "kv260_ssh_checked": True,
            "kv260_ssh_ready": True,
            "kv260_host_block_devices_touched": False,
            "gatemate_checked": True,
            "gatemate_detected": False,
            "polarfire_checked": True,
            "polarfire_ssh_ready": True,
            "hardware_residual_telemetry_ready": True,
            "no_speedup_claim": True,
            "residual_source": "cpu_reference_residual_sweep",
            "decay_exponent": 2.0,
        },
    )


def test_req_capstone_5121_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-5121: OpenSpec anchors the .469 capstone."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5121",
        "SCENARIO-CAPSTONE-5121",
        "SCENARIO-CAPSTONE-5121-FIELD-PRINCIPLES",
        "experiment_5121_capstone_v469.py",
        "results/experiment_5121_capstone_v469.json",
        "DURATION_TOO_SHORT",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_capstone_5121_aggregates_available_artifacts_and_gaps(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5121: available artifacts drive axes while missing rows stay gaps."""

    _write_default_upstreams(tmp_path)

    artifact = mod.build_artifact(
        root=tmp_path,
        duration_s=1.23456,
        run_date="20260701",
        tests_run=["unit-test-placeholder"],
        adversarial_reporter=_fake_adversarial_report,
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == "exp5121-capstone-v469"
    assert artifact["milestone"] == "2026.07.469"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 1.23456
    assert len(artifact["artifacts_read"]) == 10
    assert [row["experiment_number"] for row in artifact["missing_artifacts"]] == [5113, 5118]
    assert [row["experiment_number"] for row in artifact["quarantined_artifacts"]] == [5119]
    assert artifact["fover_moat_state"]["state"] == "blocked"
    assert artifact["fover_moat_state"]["moat_claim_supported"] is False
    assert artifact["kan_post_wall_state"]["state"] == "clean_positive"
    assert artifact["kan_post_wall_state"]["solved_n"] == 100
    assert artifact["kan_post_wall_state"]["exp5108_largest_n_reached"] == 10
    assert artifact["solver_sampling_state"]["state"] == "clean_positive"
    assert artifact["solver_sampling_state"]["fover_transfer_gap_present"] is True
    assert artifact["fr11_state"]["state"] == "blocked"
    assert artifact["fr11_state"]["continuous_self_learning_task"] is False
    assert artifact["fr11_state"]["promotion_safe"] is False
    assert artifact["runtime_state"]["state"] == "flagged"
    assert artifact["runtime_state"]["headline_eligible"] is False
    assert artifact["hardware_state"]["state"] == "clean_positive"
    assert artifact["hardware_state"]["no_speedup_claim"] is True
    assert len(artifact["next_milestone_recommendations"]) == 5
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["flagged_adversarial"] is False


def test_scenario_capstone_5121_blocks_doomed_fover_reruns(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5121: recommendations retire same-verdict doomed FoVer reruns."""

    _write_default_upstreams(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        duration_s=0.2,
        run_date="20260701",
        tests_run=["unit-test-placeholder"],
        adversarial_reporter=_fake_adversarial_report,
    )

    retired = [
        row
        for row in artifact["next_milestone_recommendations"]
        if row["retire_same_verdict_doomed_rerun"]
    ]

    assert retired
    assert "FoVer" in retired[0]["priority"]
    assert "pool retraction" in retired[0]["rationale"]


def test_req_capstone_5121_cli_writes_and_validates(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5121: CLI writes the capstone artifact with schema validation."""

    _write_default_upstreams(tmp_path)
    output = tmp_path / "result.json"

    exit_code = mod.main(
        [
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--date",
            "20260701",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["experiment_id"] == mod.EXPERIMENT_ID
    assert payload["result_path"] == str(mod.RESULT_RELATIVE_PATH)
    mod.validate_artifact(payload)


def test_deliverable_file_validates_for_req_capstone_5121() -> None:
    """SCENARIO-CAPSTONE-5121: final deliverable JSON satisfies the .469 contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["runtime_state"]["state"] == "flagged"
    assert artifact["hardware_state"]["no_speedup_claim"] is True

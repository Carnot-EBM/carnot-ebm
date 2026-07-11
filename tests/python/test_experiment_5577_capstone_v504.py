"""Tests for Exp5577 V504 capstone reconciliation.

Spec refs: REQ-REPORT-5577, SCENARIO-REPORT-5577,
SCENARIO-REPORT-5577-MISSING-INPUT, SCENARIO-REPORT-5577-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5577_capstone_v504 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _payloads() -> dict[Path, Any]:
    return {
        Path("results/experiment_5564_transition_v504.json"): {
            "honest_verdict": "complete: transition",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5565_v504_source_delta_ingestion.json"): {
            "honest_verdict": "complete: source delta",
            "closed_scopes_reopened": False,
            "research_references_updated": True,
            "new_references_added": [{"source_id": "blind_curator_2607_07436"}],
            "inference_substrate": "web_and_repository_source_synthesis",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"): {
            "honest_verdict": "complete: exact corpus",
            "corpus_ready": True,
            "n_rows": 120,
            "positive_control_passed": True,
            "duplicate_leakage_count": 0,
            "readiness_blockers": [],
            "inference_substrate": "deterministic_exact_fixture_no_llm",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json"): {
            "honest_verdict": "complete: measured equal collapse",
            "panel_complete": True,
            "live_model_invoked": True,
            "gpu_offload_authenticated": True,
            "n_independent_instances": 36,
            "parser_failure_count": 10,
            "solve_verify_asymmetry": {
                "model": {
                    "discrete_verdict": {
                        "solve_minus_verify_balanced_accuracy": 0.0,
                        "negative_means_verification_easier": True,
                    }
                }
            },
            "inference_substrate": "live_local_sota_gguf_plus_exact_validator",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5568_verifier_coevolution_trigger.json"): {
            "honest_verdict": "complete: coevolution triggered",
            "verifier_coevolution_required": True,
            "no_retraining_performed": True,
            "cached_only": True,
            "worst_family_false_accept_rate": 1.0,
            "triggered_by": ["worst_family_false_accept_rate"],
            "inference_substrate": "cached_verifier_outputs_plus_exact_labels",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5569_causal_memory_policy_tournament.json"): {
            "honest_verdict": "complete: flagged policy",
            "flagged_adversarial": True,
            "policy_ready": True,
            "forward_transfer_delta": 0.3,
            "rollback_success": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "inference_substrate": "deterministic_exact_feedback_memory_policy_search",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5570_spline_local_kan_online_energy.json"): {
            "honest_verdict": "complete: kan ready",
            "kan_ready": True,
            "forward_adaptation_delta": 0.4,
            "paired_ci_active_vs_frozen": {"lower": 0.4},
            "prior_family_regression": 0.0,
            "unsafe_false_accept_delta": -0.1,
            "rollback_checksum_match": True,
            "exact_feedback_only": True,
            "weights_mutated": True,
            "inference_substrate": "online_kan_exact_constraint_energy",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5571_reset_free_sota_continual_harness.json"): {
            "honest_verdict": "blocked_no_cuda_offload",
            "continual_harness_candidate": False,
            "live_model_invoked": False,
            "gpu_offload_authenticated": False,
            "inference_substrate": "live_local_sota_reset_free_exact_feedback_harness",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5572_gated_delayed_regression_promotion.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "continual_harness_candidate failed",
            "gates_evaluated": [{"passed": False}],
        },
        Path("results/experiment_5573_matched_sampler_hardware_continuity.json"): {
            "honest_verdict": "complete: matched sampler no speedup",
            "successful_matched_pairs": 2,
            "hardware_speedup_claim_allowed": True,
            "board_speedup_claimed": False,
            "speedup_by_pair": [{"speedup": 0.5}, {"speedup": 0.8}],
            "kv260_mmcblk_accessed": False,
            "inference_substrate": "matched_cpu_cuda_sampling_plus_board_status_receipts",
            "milestone": mod.MILESTONE,
        },
        Path("results/experiment_5573_matched_sampler_hardware_continuity_raw_rows.json"): [
            {"backend": "cpu", "status": "success"},
            {"backend": "cuda", "status": "success"},
        ],
        Path("results/experiment_5574_ptrm_stochastic_generator_stage1.json"): {
            "honest_verdict": "complete: ptrm stage1",
            "track": "arc-trm-generator",
            "stage1_training_complete": True,
            "retire_trm_generator_line": False,
            "no_level_solve_claim": True,
            "solve_provenance": "development_proxy",
            "positive_control_passed": True,
            "loo_verdict_reached": False,
            "inference_substrate": "trained_ptrm_offline_development_proxy",
        },
        Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json"): {
            "honest_verdict": "blocked: live path not ready",
            "live_path_reachable": True,
            "live_path_ready": False,
            "target_unsolved": True,
            "target_game": "g50t",
            "target_level": 3,
            "solve_provenance": "live_agent_self_discovery",
            "prior_levels_reproduced": 2,
            "positive_control_passed": True,
            "inference_substrate": "deterministic_live_path_precheck_no_llm",
        },
        Path("results/experiment_5576_gated_sge_live_levelup.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "live_path_ready failed",
            "gates_evaluated": [{"passed": False}, {"passed": True}],
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == Path("research-roadmap-next.yaml"):
            continue
        _write_text(root, rel_path)
    _write_text(root, "ops/arc_solve_registry.yaml", "reproducible_total_levels: 75\n")
    for rel_path, payload in _payloads().items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def test_req_report_5577_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-5577: OpenSpec anchors the V504 capstone schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5577") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.EXPECTED_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5577_classifies_live_repo_without_overclaiming() -> None:
    """SCENARIO-REPORT-5577: live V504 evidence is terminal but bounded."""

    artifact = mod.run_capstone(
        root=REPO,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["expected_task_range"] == mod.EXPECTED_TASK_RANGE
    assert artifact["upstream_artifacts_expected"] == [
        path.as_posix() for path in mod.EXPECTED_ARTIFACT_PATHS
    ]
    assert artifact["upstream_artifacts_read"] == artifact["upstream_artifacts_expected"]
    assert artifact["missing_artifacts"] == []
    assert "research-roadmap-next.yaml" in artifact["source_context_missing"]
    assert {row["lane"] for row in artifact["flagged_lanes"]} == {
        "causal_memory_policy_tournament"
    }
    assert {row["lane"] for row in artifact["skipped_lanes"]} == {
        "delayed_regression_promotion_gate",
        "sge_live_levelup_gate",
    }
    assert artifact["solve_verify_asymmetry_supported"] is False
    assert artifact["verifier_coevolution_required"] is True
    assert artifact["memory_policy_promoted"] is False
    assert artifact["kan_online_energy_promoted"] is True
    assert artifact["continuous_self_learning_claim_allowed"] is False
    assert artifact["hardware_speedup_claim_allowed"] is False
    assert artifact["ptrm_stage1_status"] == "complete_development_proxy_no_solve_claim"
    assert artifact["ptrm_retired"] is False
    assert artifact["ordinary_arc_floor_satisfied"] is False
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_registry_before_after"]["before"] == artifact["arc_registry_before_after"]["after"]
    assert artifact["sge_retired"] is True
    assert artifact["specs_updated"] == ["openspec/capabilities/research-reporting/spec.md"]
    assert artifact["traceability_updated"] is False
    assert artifact["ops_docs_updated"] is False
    assert artifact["research_complete_updated"] is False
    assert artifact["exclusion_manifest_updated"] is False
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5577_missing_upstream_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5577-MISSING-INPUT: missing inputs block promotion."""

    missing = Path("results/experiment_5570_spline_local_kan_online_energy.json")
    _make_root(tmp_path, omit=missing)

    artifact = mod.run_capstone(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["kan_online_energy_promoted"] is False
    assert artifact["continuous_self_learning_claim_allowed"] is False
    assert artifact["hardware_speedup_claim_allowed"] is False
    assert artifact["ordinary_arc_floor_satisfied"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5577_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5577-FIELD-PRINCIPLES: malformed capstones fail validation."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert mod.validate_artifact(artifact) == []
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"milestone": mod.FIELD_PRINCIPLES["milestone"]}}
    )
    assert "upstream_artifacts_expected" in mod.validate_artifact(
        {**artifact, "upstream_artifacts_expected": "all"}
    )
    assert "solve_verify_asymmetry_supported" in mod.validate_artifact(
        {**artifact, "solve_verify_asymmetry_supported": "false"}
    )
    assert "memory_policy_promoted" in mod.validate_artifact(
        {**artifact, "memory_policy_promoted": True}
    )
    assert "continuous_self_learning_claim_allowed" in mod.validate_artifact(
        {**artifact, "continuous_self_learning_claim_allowed": True}
    )
    assert "hardware_speedup_claim_allowed" in mod.validate_artifact(
        {**artifact, "hardware_speedup_claim_allowed": True}
    )
    assert "ordinary_arc_floor_satisfied" in mod.validate_artifact(
        {**artifact, "ordinary_arc_floor_satisfied": True, "arc_registry_delta": 0}
    )
    assert "roadmap_yaml_unchanged" in mod.validate_artifact(
        {**artifact, "roadmap_yaml_unchanged": False}
    )
    assert "conductor_unchanged" in mod.validate_artifact(
        {**artifact, "conductor_unchanged": False}
    )
    assert "milestone" in mod.validate_artifact({**artifact, "milestone": "2026.07.503"})
    assert "expected_task_range" in mod.validate_artifact(
        {**artifact, "expected_task_range": "exp5564-exp5576"}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "promoted_lanes" in mod.validate_artifact({**artifact, "promoted_lanes": "none"})
    assert mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert mod._status_label({"honest_verdict": "honest_null: synthetic"}) == "honest_null"
    assert mod._status_label({"honest_verdict": "failed: synthetic"}) == "failed"
    assert mod._status_label({"honest_verdict": "unclear"}) == "unknown"
    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "7"}, "value") == 7
    assert mod._float({"value": "1.25"}, "value") == 1.25
    assert mod._float({"value": "bad"}, "value") == 0.0
    assert mod._float({"value": None}, "value") == 0.0
    assert mod._nested_floats([{"target": -0.1}], "target") == [-0.1]
    assert mod._registry_total(tmp_path / "missing-registry-root") is None
    assert mod._ptrm_status({}) == "missing"
    assert mod._ptrm_status({"honest_verdict": "blocked: synthetic"}) == "blocked_or_flagged"
    assert mod._ptrm_status({"honest_verdict": "complete: synthetic"}) == "incomplete"


def test_scenario_report_5577_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5577: writer persists the validated deliverable."""

    _make_root(tmp_path)

    artifact = mod.write_capstone(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert mod.validate_artifact(written) == []

"""Tests for the Exp5509 V499 capstone synthesis.

Spec refs: REQ-REPORT-5509, SCENARIO-REPORT-5509,
SCENARIO-REPORT-5509-MISSING-INPUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5509_capstone_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_context(root: Path) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"context for {rel_path.as_posix()}\n", encoding="utf-8")


def _populate_primary_artifacts(root: Path, *, omit_5508: bool = False) -> None:
    _write_json(
        root,
        "results/experiment_5496_transition_v499.json",
        {
            "status": "complete",
            "honest_verdict": "complete: .498 facts archived",
            "exp5474_tautology_still_blocks_csl_headlines": True,
            "blocked_lanes": [
                {
                    "lane": "guided_decoding_quarantine",
                    "evidence": {"quarantine_status": "quarantined"},
                }
            ],
            "clean_lanes": [{"lane": "active_constraint_subproblem_descriptors"}],
            "honest_null_lanes": [{"lane": "hardware_speedup_claim_false"}],
        },
    )
    _write_json(
        root,
        "results/experiment_5497_pretest_cascade_diagnostic_v499.json",
        {
            "status": "complete",
            "pretest_cascade_resolved": True,
            "reproduced_pretest_failure": False,
            "downstream_gate_recommendation": "open_downstream_pretest_gate",
            "commands_run": [{"command": "smart subset", "outcome": "passed"}],
            "honest_verdict": "complete: current smart subset green",
        },
    )
    _write_json(
        root,
        "results/experiment_5498_source_delta_v499.json",
        {
            "status": "complete",
            "new_actionable_findings_count": 1,
            "closed_scopes_reopened": False,
            "research_references_updated": True,
            "honest_verdict": "complete: one source delta appended",
        },
    )
    _write_json(
        root,
        "results/experiment_5499_preference_maxsat_minimal_fixture_v499.json",
        {
            "preference_maxsat_fixture_ready": True,
            "hard_constraint_pass_rate": 1.0,
            "preference_optimality_rate": 1.0,
            "false_accept_rate": 0.0,
            "independent_reference_agreement_rate": 1.0,
            "guided_decoding_used": False,
            "token_steering_used": False,
            "num_instances": 3,
            "honest_verdict": "complete: exact validators authoritative",
        },
    )
    _write_json(
        root,
        "results/experiment_5500_sota_concept_claim_panel_v499.json",
        {
            "exact_validator_accuracy": 0.333333,
            "concept_claim_telemetry_rows": 6,
            "abstention_count": 6,
            "hard_constraint_violation_rate": 0.0,
            "preference_optimality_rate": 0.0,
            "guided_decoding_used": False,
            "token_steering_used": False,
            "gpu_offload_verified": True,
            "headline_models_used": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
            ],
            "honest_verdict": "complete: live_sota_claim_panel_measured_accuracy_0.333333",
        },
    )
    _write_json(
        root,
        "results/experiment_5501_helper_contract_hierarchical_claim_fixture_v499.json",
        {
            "helper_contract_fixture_ready": True,
            "num_helper_contracts": 7,
            "local_claim_label_accuracy": 1.0,
            "rolled_up_verdict_accuracy": 1.0,
            "useful_repair_count": 2,
            "useful_repair_rate": 1.0,
            "unsupported_contract_count": 1,
            "contract_reports": [{"false_accept": False}],
            "honest_verdict": "complete: helper contracts ready",
        },
    )
    _write_json(
        root,
        "results/experiment_5502_csl_tautology_static_corrigendum_v499.json",
        {
            "tautology_flag_resolved": True,
            "metric_independence_clean": False,
            "csl_scale_headline_allowed": False,
            "retire_same_scope_if_repeated": True,
            "downstream_recommendation": "bounded_requires_rerun",
            "independence_violations": [{"kind": "policy_outcome_scalar_overlap"}],
            "honest_verdict": "complete: Exp5474 CSL scale headline is bounded",
        },
    )
    _write_json(
        root,
        "results/experiment_5503_csl_experience_graph_replay_v499.json",
        {
            "csl_experience_graph_ready": True,
            "no_memory_baseline_score": 0.0,
            "graph_memory_score": 1.0,
            "heldout_delta": 1.0,
            "negative_transfer_rate": 0.0,
            "stale_evidence_rejection_rate": 1.0,
            "model_weights_mutated": False,
            "heldout_task_ids": [
                "5503-heldout-dock-crate",
                "5503-heldout-python-loop",
                "5503-heldout-rx4-handoff",
                "5503-heldout-dock-gate",
            ],
            "control_counts": {
                "negative_transfer_candidates_accepted": 0,
                "negative_transfer_candidates_seen": 2,
                "stale_candidates_rejected": 2,
                "stale_candidates_seen": 2,
            },
            "honest_verdict": "complete: experience_graph_replay_ready_delta_+1.000000",
        },
    )
    _write_json(
        root,
        "results/experiment_5503_csl_experience_graph_memory_v499.json",
        {"schema": "memory", "graph_id": "exp5503-experience-graph", "nodes": []},
    )
    _write_json(
        root,
        "results/experiment_5503_csl_experience_graph_replay_fixture_v499.json",
        {"schema": "fixture", "heldout_task_ids": ["5503-heldout-dock-crate"]},
    )
    _write_json(
        root,
        "results/experiment_5504_sota_csl_memory_panel_v499.json",
        {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "2 of 2 gate(s) failed; first failure: "
                "metric_independence_clean actual=False"
            ),
            "gates_evaluated": [
                {"upstream": "exp5502", "artifact_field": "metric_independence_clean", "passed": False},
                {"upstream": "exp5503", "artifact_field": "csl_experience_graph_ready", "passed": False},
            ],
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        "results/experiment_5505_active_constraint_milp_descriptor_v499.json",
        {
            "descriptor_ready_for_hardware": True,
            "exact_fallback_agreement_rate": 1.0,
            "num_descriptor_rows": 7,
            "milp_style_rows": 2,
            "maxsat_style_rows": 3,
            "csp_style_rows": 2,
            "partition_update_fields_present": True,
            "hardware_speedup_claim": False,
            "honest_verdict": "complete: descriptors ready with no speedup claim",
        },
    )
    _write_json(
        root,
        "results/experiment_5506_hardware_multiboard_receipts_v499.json",
        {
            "cpu_status": "reachable",
            "cuda_status": "reachable",
            "polar_fire_status": "reachable",
            "kv260_status": "blocked_identity",
            "gatemate_status": "blocked_identity",
            "matched_timing_available": False,
            "hardware_speedup_claim": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
            "matched_hashes": [
                {"substrate": "cpu", "matched": True},
                {"substrate": "cuda", "matched": True},
                {"substrate": "polarfire", "matched": True},
            ],
            "honest_verdict": "complete: descriptor smoke receipts collected",
        },
    )
    _write_json(
        root,
        "results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json",
        {
            "status": "complete",
            "levelup_attempt_ready": True,
            "selected_game": "dc22",
            "selected_level": "L3",
            "selected_mechanism": "perception-grounded connected-component/color-blob segmentation",
            "reproducible_total_levels_before": 69,
            "solve_claimed": False,
            "null_coordinate_audit": {
                "null_coordinate_exploit_valid": False,
                "verdict": "valid_recorded_actions_with_noop_effects",
            },
            "honest_verdict": "complete: dc22 L3 precheck ready; no solve claimed",
        },
    )
    if not omit_5508:
        _write_json(
            root,
            "results/experiment_5508_arc_live_perception_generation_levelup_v499.json",
            {
                "status": "honest_null",
                "registry_before_levels": 69,
                "registry_after_levels": 69,
                "arc_registry_delta": 0,
                "selected_game": "dc22",
                "selected_level": "L3",
                "target_level": 3,
                "prior_levels_reproduced": 2,
                "post_levels_reproduced": 2,
                "offline_reproduced": False,
                "reproduced_levels": 0,
                "registry_updated": False,
                "solve_provenance": "live_agent_self_discovery",
                "live_agent_attempts": 47,
                "runtime_observation_steps": 48,
                "candidate_action_count": 49,
                "offline_bfs_used": False,
                "game_source_read": False,
                "hand_built_per_game_adapter_used": False,
                "methodology_receipt": "bounded_live_runtime budget=48",
                "trajectory_taxonomy_counts": {"factual": 23, "logical": 23, "scope_based": 1},
                "honest_verdict": (
                    "honest_null: dc22 L3 bounded_budget_no_target_level_reproduction; "
                    "registry_delta=0"
                ),
            },
        )


def test_req_report_5509_spec_declares_required_fields() -> None:
    """REQ-REPORT-5509: OpenSpec anchors the capstone artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5509") :]

    assert "SCENARIO-REPORT-5509" in section
    assert "SCENARIO-REPORT-5509-MISSING-INPUT" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5509_synthesizes_milestone_truth(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5509: actual artifacts drive the final lane verdicts."""

    _write_context(tmp_path)
    _populate_primary_artifacts(tmp_path)

    report = mod.build_report(tmp_path, tests_run=["unit 5509"])

    assert report["milestone"] == "2026.07.499"
    assert report["artifacts_expected"] == [path.as_posix() for path in mod.EXPECTED_ARTIFACTS.values()]
    assert report["artifacts_found"] == report["artifacts_expected"]
    assert report["artifacts_missing"] == []
    assert report["sidecar_artifacts_found"] == [
        "results/experiment_5503_csl_experience_graph_memory_v499.json",
        "results/experiment_5503_csl_experience_graph_replay_fixture_v499.json",
    ]

    assert report["pretest_cascade_resolved"] is True
    assert report["hard_soft_core_verdict"].startswith("bounded:")
    assert "SOTA panel abstained" in report["hard_soft_core_verdict"]
    assert report["csl_verdict"].startswith("blocked:")
    assert "Exp5474-style CSL scale headlines remain blocked" in report["csl_verdict"]
    assert report["hardware_verdict"].startswith("bounded:")
    assert "KV260=blocked_identity" in report["hardware_verdict"]
    assert report["arc_verdict"].startswith("honest_null:")
    assert report["arc_registry_delta"] == 0
    assert report["hardware_speedup_claim"] is False
    assert report["guided_decoding_quarantine_status"] == "quarantined"
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["honest_verdict"].startswith("complete:")

    gaps = {row["prd_item"]: row for row in report["prd_gap_table"]}
    assert gaps["FR-11 autonomous self-learning"]["status"] == "bounded_headline_blocked"
    assert gaps["FR-12 verifiable reasoning"]["status"] == "bounded_core_ready"
    assert gaps["NFR-01 performance and hardware"]["status"] == "receipt_only_no_speedup"
    assert gaps["ARC north-star live path"]["status"] == "honest_null_no_registry_delta"

    taxonomy = {row["failure_class"]: row for row in report["failure_taxonomy"]}
    assert taxonomy["sota_abstention_panel"]["evidence"]["abstention_count"] == 6
    assert taxonomy["csl_metric_independence_blocker"]["evidence"]["metric_independence_clean"] is False
    assert taxonomy["hardware_methodology_flag"]["evidence"]["flagged_adversarial"] is True
    assert taxonomy["arc_no_bank"]["evidence"]["arc_registry_delta"] == 0
    assert any(item["recommendation"].startswith("Retire Exp5474-style") for item in report["next_recommendations"])


def test_scenario_report_5509_missing_primary_blocks_completion(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5509-MISSING-INPUT: missing primary artifacts fail closed."""

    _write_context(tmp_path)
    _populate_primary_artifacts(tmp_path, omit_5508=True)

    report = mod.build_report(tmp_path, tests_run=["unit 5509"])

    assert "results/experiment_5508_arc_live_perception_generation_levelup_v499.json" in report[
        "artifacts_missing"
    ]
    assert report["arc_verdict"].startswith("blocked:")
    assert report["arc_registry_delta"] == 0
    assert report["honest_verdict"].startswith("blocked:")


def test_scenario_report_5509_bounded_and_missing_branches_are_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-5509: alternate lane verdicts are deterministic and auditable."""

    _write_context(tmp_path)
    _populate_primary_artifacts(tmp_path)

    transition_path = tmp_path / "results/experiment_5496_transition_v499.json"
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    transition["blocked_lanes"] = []
    _write_json(tmp_path, "results/experiment_5496_transition_v499.json", transition)

    sota_path = tmp_path / "results/experiment_5500_sota_concept_claim_panel_v499.json"
    sota = json.loads(sota_path.read_text(encoding="utf-8"))
    sota["exact_validator_accuracy"] = "not-a-float"
    _write_json(tmp_path, "results/experiment_5500_sota_concept_claim_panel_v499.json", sota)

    report = mod.build_report(tmp_path, tests_run=["unit 5509"])
    assert report["guided_decoding_quarantine_status"] == "quarantined"
    assert report["hard_soft_core_verdict"].startswith("bounded:")

    sota["exact_validator_accuracy"] = 1.0
    sota["abstention_count"] = 0
    _write_json(tmp_path, "results/experiment_5500_sota_concept_claim_panel_v499.json", sota)
    report = mod.build_report(tmp_path, tests_run=["unit 5509"])
    assert report["hard_soft_core_verdict"].startswith("headline_ready:")

    helper_path = tmp_path / "results/experiment_5501_helper_contract_hierarchical_claim_fixture_v499.json"
    helper = json.loads(helper_path.read_text(encoding="utf-8"))
    helper["helper_contract_fixture_ready"] = False
    _write_json(tmp_path, "results/experiment_5501_helper_contract_hierarchical_claim_fixture_v499.json", helper)
    report = mod.build_report(tmp_path, tests_run=["unit 5509"])
    assert report["hard_soft_core_verdict"].startswith("blocked:")

    _populate_primary_artifacts(tmp_path)
    arc = json.loads((tmp_path / "results/experiment_5508_arc_live_perception_generation_levelup_v499.json").read_text(encoding="utf-8"))
    arc["arc_registry_delta"] = 1
    arc["registry_after_levels"] = 70
    arc["offline_reproduced"] = True
    arc["reproduced_levels"] = 1
    _write_json(tmp_path, "results/experiment_5508_arc_live_perception_generation_levelup_v499.json", arc)
    report = mod.build_report(tmp_path, tests_run=["unit 5509"])
    assert report["arc_verdict"].startswith("headline_ready:")

    csl_static = json.loads((tmp_path / "results/experiment_5502_csl_tautology_static_corrigendum_v499.json").read_text(encoding="utf-8"))
    csl_static["metric_independence_clean"] = True
    _write_json(tmp_path, "results/experiment_5502_csl_tautology_static_corrigendum_v499.json", csl_static)
    csl_panel = json.loads((tmp_path / "results/experiment_5504_sota_csl_memory_panel_v499.json").read_text(encoding="utf-8"))
    csl_panel["status"] = "complete"
    _write_json(tmp_path, "results/experiment_5504_sota_csl_memory_panel_v499.json", csl_panel)
    report = mod.build_report(tmp_path, tests_run=["unit 5509"])
    assert report["csl_verdict"].startswith("bounded:")

    for rel_path, verdict_field in [
        ("results/experiment_5499_preference_maxsat_minimal_fixture_v499.json", "hard_soft_core_verdict"),
        ("results/experiment_5504_sota_csl_memory_panel_v499.json", "csl_verdict"),
        ("results/experiment_5506_hardware_multiboard_receipts_v499.json", "hardware_verdict"),
    ]:
        _populate_primary_artifacts(tmp_path)
        (tmp_path / rel_path).unlink()
        report = mod.build_report(tmp_path, tests_run=["unit 5509"])
        assert report[verdict_field].startswith("blocked:")


def test_scenario_report_5509_write_report_persists_required_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5509: written artifact remains schema-auditable."""

    _write_context(tmp_path)
    _populate_primary_artifacts(tmp_path)

    payload = mod.write_report(tmp_path, tests_run=["unit 5509"])
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == payload
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)
    assert written["source_context_missing"] == ["research-roadmap-next.yaml"]

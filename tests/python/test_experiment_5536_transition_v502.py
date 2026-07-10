"""Tests for the Exp5536 .502 transition receipt.

Spec refs: REQ-REPORT-5536, SCENARIO-REPORT-5536,
SCENARIO-REPORT-5536-BLOCKED-INPUT, SCENARIO-REPORT-5536-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5536_transition_v502 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: JsonDict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_context(root: Path, *, include_next: bool = False) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH and not include_next:
            continue
        if rel_path == mod.ROADMAP_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                yaml.safe_dump(
                    {
                        "milestone": mod.MILESTONE,
                        "milestone_doc": mod.VNEXT_RELATIVE_PATH.as_posix(),
                        "tasks": [
                            {"id": task_id, "milestone": mod.MILESTONE}
                            for task_id in mod.EXPECTED_TASK_IDS
                        ],
                    },
                    sort_keys=False,
                ),
            )
        elif rel_path == mod.VNEXT_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                "\n".join(
                    [
                        "# Research Roadmap vNEXT - Milestone 2026.07.502",
                        "",
                        "**Previous milestone:** 2026.07.501",
                        "**Task range:** Exp 5536-5549",
                        "",
                        "Structured conductor gates:",
                        "- exp5540 requires exp5538.sota_panel_duration_corrigendum_ready == true and exp5539.grammar_table_preflight_ready == true.",
                        "- exp5543 requires exp5542.csl_residue_tautology_resolved == true.",
                        "- exp5545 requires exp5541.exact_fsm_fixture_ready == true.",
                        "- exp5548 requires exp5547.arc_clean_precheck_ready == true.",
                    ]
                )
                + "\n",
            )
        elif rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                "\n".join(
                    [
                        "| 2026-07-10 07:04 UTC | Execution-time V501 source delta and experiment mapping | OK | 124 passed |",
                        "| 2026-07-10 07:20 UTC | SOTA GGUF structured-output failure taxonomy | OK | 94 passed |",
                        "| 2026-07-10 07:39 UTC | Gated SOTA GGUF structured repair loop | OK | 86 passed |",
                        "| 2026-07-10 07:56 UTC | Gated SOTA GGUF hard/soft structured panel v2 | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
                        "| 2026-07-10 08:10 UTC | CSL canonical gate artifact and sidecar-safe field receipt | OK | 85 passed |",
                        "| 2026-07-10 08:38 UTC | Gated CSL event/topic memory residue stress | FLAGGED | adversarial_verify CRITICAL: TAUTOLOGY |",
                        "| 2026-07-10 08:54 UTC | Gated SOTA GGUF CSL memory panel v2 | OK | 86 passed |",
                        "| 2026-07-10 10:08 UTC | Sparse repair scale-up with exact fallback confidence intervals | OK | 87 passed |",
                        "| 2026-07-10 10:24 UTC | Hardware receipt parser and repeatability repair | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
                        "| 2026-07-10 10:44 UTC | ARC strategy-routing and repeated-coordinate precheck | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
                        "| 2026-07-10 10:59 UTC | Gated ARC strategy-routed live level-up attempt | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
                        "| 2026-07-10 13:26 UTC | Capstone reconciliation for milestone .501 | OK | cache hit: 87 passed |",
                        "| 2026-07-10 14:14 UTC | Plan milestone 2026.07.502 | OK | 14 tasks proposed |",
                        "| 2026-07-10 14:16 UTC | Milestone 2026.07.502 activated | OK | 14 tasks queued |",
                    ]
                )
                + "\n",
            )
        else:
            _write_text(root, rel_path)
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH)


def _artifact_payloads() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5524_v501_source_delta_ingestion.json"): {
            "milestone": "2026.07.501",
            "research_references_updated": False,
            "new_references_added": [],
            "closed_scopes_reopened": False,
            "honest_verdict": "complete: no new actionable V501 execution-time source deltas after the planner refresh; references unchanged and closed scopes stayed closed.",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        Path("results/experiment_5525_sota_schema_failure_taxonomy.json"): {
            "milestone": "2026.07.501",
            "sota_schema_failure_taxonomy_ready": True,
            "schema_validity_rate": 0.0,
            "exact_validator_handoff_ready": False,
            "honest_verdict": "complete: sota_schema_failure_taxonomy_ready_prompt_contract_miss_and_semantic_candidate_absent",
            "inference_substrate": "structured_output_fixture_plus_live_llm_smoke",
        },
        Path("results/experiment_5526_sota_structured_repair_loop.json"): {
            "milestone": "2026.07.501",
            "sota_structured_repair_loop_ready": True,
            "schema_validity_before": 0.0,
            "schema_validity_after": 1.0,
            "missing_candidate_rows_after": 0,
            "exact_validator_handoff_ready": True,
            "honest_verdict": "complete: sota_structured_repair_loop_ready_schema_valid_exact_handoff_no_quality_claim",
            "inference_substrate": "structured_output_repair_fixture_plus_live_llm_smoke",
        },
        Path("results/experiment_5527_sota_hard_soft_panel_v2.json"): {
            "milestone": "2026.07.501",
            "sota_hard_soft_claim_allowed": True,
            "sota_structured_panel_ready": True,
            "schema_validity_rate": 1.0,
            "exact_validator_accuracy": 1.0,
            "preference_optimality_rate": 1.0,
            "rows_requested": 3,
            "rows_emitted": 3,
            "missing_candidate_rows": 0,
            "duration_s": 0.001162,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "honest_verdict": "complete: sota_hard_soft_panel_v2_ready_bounded_exact_validated_claim_allowed",
            "inference_substrate": "exact_validated_local_sota_gguf_panel",
        },
        Path("results/experiment_5528_csl_canonical_gate_artifact.json"): {
            "milestone": "2026.07.501",
            "continuous_self_learning_evidence": True,
            "csl_gate_fields_conductor_visible": True,
            "conductor_gate_probe_passed": True,
            "heldout_delta": 1.0,
            "stale_evidence_rejection_rate": 1.0,
            "negative_transfer_rate": 0.0,
            "honest_verdict": "complete: canonical_csl_gate_artifact_conductor_visible",
            "inference_substrate": "canonical_csl_gate_artifact_from_independent_fixture",
        },
        Path("results/experiment_5529_csl_event_topic_residue_stress.json"): {
            "milestone": "2026.07.501",
            "csl_residue_stress_ready": True,
            "event_only_score": 0.6666666667,
            "topic_only_score": 0.6666666667,
            "stale_evidence_rejection_rate": 1.0,
            "negative_transfer_rate": 0.0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "honest_verdict": "complete: csl_event_topic_residue_stress_ready",
            "inference_substrate": "deterministic_csl_memory_fixture",
        },
        Path("results/experiment_5530_sota_csl_memory_panel_v2.json"): {
            "milestone": "2026.07.501",
            "continuous_self_learning_evidence": True,
            "csl_claim_allowed": True,
            "heldout_delta": 0.6666666667,
            "stale_evidence_rejection_rate": 1.0,
            "negative_transfer_rate": 0.0,
            "upstream_gate_evidence": {
                "exp5529": {
                    "flagged_adversarial": True,
                    "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
                }
            },
            "honest_verdict": "complete: bounded_sota_csl_memory_panel_v2_claim_allowed",
            "inference_substrate": "local_sota_gguf_csl_memory_panel",
        },
        Path("results/experiment_5531_sparse_repair_scaleup_ci.json"): {
            "milestone": "2026.07.501",
            "active_constraint_sparse_repair_ready": True,
            "sparse_repair_success_rate": 1.0,
            "exact_only_success_rate": 1.0,
            "all_candidates_exact_checked": True,
            "matched_timing_available": False,
            "speedup_claim_allowed": False,
            "honest_verdict": "complete: exact_checked_sparse_repair_scaleup_ci_ready_no_speedup_claim",
            "inference_substrate": "exact_checked_sparse_repair_scaleup",
        },
        Path("results/experiment_5532_hardware_receipt_parser_repeatability.json"): {
            "milestone": "2026.07.501",
            "hardware_speedup_claim": False,
            "hardware_speedup_claim_allowed": False,
            "matched_timing_available": False,
            "duration_s": 9.160578,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
            "honest_verdict": "complete: hardware receipt parser repaired with blockers (kv260,gatemate); matched_timing_available=false; hardware_speedup_claim=false",
            "inference_substrate": "hardware_receipt_parser_repeatability",
        },
        Path("results/experiment_5533_arc_strategy_routing_precheck.json"): {
            "milestone": "2026.07.501",
            "status": "complete",
            "selected_game": "g50t",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "arc_sge_candidate_ready": True,
            "duration_s": 0.07068435614928603,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
            "honest_verdict": "complete: g50t L3 strategy-routing precheck ready; no solve claimed",
            "inference_substrate": "arc_live_path_precheck_no_solve_claim",
        },
        Path("results/experiment_5534_arc_strategy_routed_levelup.json"): {
            "milestone": "2026.07.501",
            "status": "honest_null",
            "selected_game": "g50t",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "registry_before_levels": 69,
            "registry_after_levels": 69,
            "registry_delta": 0,
            "action_entropy": 1.0,
            "repeated_coordinate_rate": 0.0,
            "duration_s": 2.702068200800568,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "IMPLAUSIBLE_PERFECT", "severity": "info"},
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
            "honest_verdict": "honest_null: g50t L3 bounded_budget_no_target_level_reproduction; entropy=1.000; repeat_rate=0.000; registry_delta=0",
            "inference_substrate": "arc_live_agent_self_discovery",
        },
        mod.PRIOR_CAPSTONE_RELATIVE_PATH: {
            "milestone": "2026.07.501",
            "structured_sota_claim_allowed": True,
            "sota_hard_soft_claim_allowed": False,
            "continuous_self_learning_evidence": True,
            "csl_claim_allowed": False,
            "sparse_repair_claim_allowed": True,
            "hardware_speedup_claim": False,
            "arc_registry_delta": 0,
            "reproduced_levels": 0,
            "missing_artifacts": ["research-roadmap-next.yaml"],
            "skipped_by_gates": [
                {
                    "artifact_path": "results/experiment_5527_sota_hard_soft_panel_v2.json",
                    "skip_reason": "flagged_adversarial",
                    "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
                },
                {
                    "artifact_path": "results/experiment_5529_csl_event_topic_residue_stress.json",
                    "skip_reason": "flagged_adversarial",
                    "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
                },
                {
                    "artifact_path": "results/experiment_5532_hardware_receipt_parser_repeatability.json",
                    "skip_reason": "flagged_adversarial",
                    "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}],
                },
                {
                    "artifact_path": "results/experiment_5533_arc_strategy_routing_precheck.json",
                    "skip_reason": "flagged_adversarial",
                    "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}],
                },
                {
                    "artifact_path": "results/experiment_5534_arc_strategy_routed_levelup.json",
                    "skip_reason": "flagged_adversarial",
                    "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
                },
            ],
            "terminal_evidence": {
                "sota_taxonomy": "complete: sota_schema_failure_taxonomy_ready_prompt_contract_miss_and_semantic_candidate_absent",
                "sota_repair_loop": "complete: sota_structured_repair_loop_ready_schema_valid_exact_handoff_no_quality_claim",
                "hard_soft_panel": "complete: sota_hard_soft_panel_v2_ready_bounded_exact_validated_claim_allowed",
                "csl_canonical_gate": "complete: canonical_csl_gate_artifact_conductor_visible",
                "csl_memory_panel": "complete: bounded_sota_csl_memory_panel_v2_claim_allowed",
                "sparse_repair": "complete: exact_checked_sparse_repair_scaleup_ci_ready_no_speedup_claim",
                "hardware": "complete: hardware receipt parser repaired with blockers (kv260,gatemate); matched_timing_available=false; hardware_speedup_claim=false",
                "arc_levelup": "honest_null: g50t L3 bounded_budget_no_target_level_reproduction; entropy=1.000; repeat_rate=0.000; registry_delta=0",
            },
            "honest_verdict": "complete: .501 capstone read 13 result artifacts; structured_sota_claim_allowed=True; sota_hard_soft_claim_allowed=False; continuous_self_learning_evidence=True; csl_claim_allowed=False; sparse_repair_claim_allowed=True; hardware_speedup_claim=False; arc_registry_delta=0",
            "inference_substrate": "capstone_aggregation_from_upstream_artifacts",
            "roadmap_yaml_unchanged": True,
            "conductor_unchanged": True,
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    _write_context(root)
    for rel_path, payload in _artifact_payloads().items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def _lane_names(rows: list[JsonDict]) -> set[str]:
    return {str(row["lane"]) for row in rows}


def test_req_report_5536_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5536: OpenSpec declares the V502 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5536") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert mod.PRIOR_CAPSTONE_RELATIVE_PATH.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5536_archives_v501_facts_and_v502_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5536: .501 facts and .502 gates are preserved."""

    _make_root(tmp_path)

    report = mod.build_report(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "complete"
    assert report["milestone"] == mod.MILESTONE
    assert report["previous_milestone"] == "2026.07.501"
    assert report["prior_capstone_path"] == "results/experiment_5535_capstone_v501.json"
    assert report["previous_task_range"] == "exp5523-exp5535"
    assert report["next_task_range"] == "exp5536-exp5549"
    assert report["roadmap_task_ids"] == mod.EXPECTED_TASK_IDS
    assert report["roadmap_doc_task_range"] == "exp5536-exp5549"
    assert report["artifacts_missing"] == []
    assert report["source_context_missing"] == [mod.ROADMAP_NEXT_RELATIVE_PATH.as_posix()]
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["sota_duration_corrigendum_required"] is True
    assert report["grammar_preflight_required"] is True
    assert report["csl_residue_corrigendum_required"] is True
    assert report["finite_state_fixture_required"] is True
    assert report["arc_clean_precheck_required"] is True
    assert report["hardware_receipt_only"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["honest_verdict"].startswith("complete:")

    assert {
        "source_delta",
        "sota_schema_taxonomy",
        "structured_repair_loop",
        "canonical_csl_gate",
        "sota_csl_memory_panel",
        "sparse_repair_scaleup",
    } <= _lane_names(report["clean_lanes"])
    assert {
        "structured_sota_repair_claim_boundary",
        "bounded_csl_memory_claim",
        "sparse_repair_no_speedup",
        "arc_live_path_provenance_no_bank",
    } <= _lane_names(report["bounded_lanes"])
    assert {"hardware_speedup_false", "arc_registry_delta_zero"} <= _lane_names(
        report["blocked_lanes"]
    )
    assert {
        "exp5527_duration_substrate",
        "exp5529_residue_tautology",
        "exp5532_hardware_receipt_methodology",
        "exp5533_arc_precheck_hygiene",
        "exp5534_arc_levelup_hygiene",
    } <= _lane_names(report["flagged_lanes"])
    assert mod.validate_artifact(report) == []


def test_scenario_report_5536_missing_capstone_and_dirty_files_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5536-BLOCKED-INPUT: missing or dirty inputs fail closed."""

    _make_root(tmp_path, omit=mod.PRIOR_CAPSTONE_RELATIVE_PATH)
    (tmp_path / mod.ROADMAP_RELATIVE_PATH).write_text(
        yaml.safe_dump({"milestone": "2026.07.501", "tasks": []}, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Missing the task-range marker\n",
        encoding="utf-8",
    )

    report = mod.build_report(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert "results/experiment_5535_capstone_v501.json_missing_or_unreadable" in report[
        "failed_preconditions"
    ]
    assert "prior_capstone_milestone_mismatch" in report["failed_preconditions"]
    assert "research-roadmap.yaml_milestone_mismatch" in report["failed_preconditions"]
    assert "roadmap_task_ids_mismatch" in report["failed_preconditions"]
    assert "vnext_task_range_mismatch" in report["failed_preconditions"]
    assert "research-roadmap.yaml_modified" in report["failed_preconditions"]
    assert "scripts/research_conductor.py_modified" in report["failed_preconditions"]
    assert report["roadmap_yaml_unchanged"] is False
    assert report["conductor_unchanged"] is False


def test_scenario_report_5536_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5536-FIELD-PRINCIPLES: malformed receipts are rejected."""

    _make_root(tmp_path)
    report = mod.build_report(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert "hardware_receipt_only" in mod.validate_artifact(
        {**report, "hardware_receipt_only": False}
    )
    assert "arc_clean_precheck_required" in mod.validate_artifact(
        {**report, "arc_clean_precheck_required": "true"}
    )
    assert "clean_lanes" in mod.validate_artifact({**report, "clean_lanes": "source"})
    assert "field_principles" in mod.validate_artifact(
        {**report, "field_principles": {"milestone": mod.FIELD_PRINCIPLES["milestone"]}}
    )
    assert "inference_substrate" in mod.validate_artifact(
        {**report, "inference_substrate": "capstone_aggregation_from_upstream_artifacts"}
    )
    assert "milestone" in mod.validate_artifact({**report, "milestone": "2026.07.501"})
    assert "previous_milestone" in mod.validate_artifact(
        {**report, "previous_milestone": "2026.07.500"}
    )
    assert "prior_capstone_path" in mod.validate_artifact(
        {**report, "prior_capstone_path": "results/experiment_5534_arc_strategy_routed_levelup.json"}
    )
    assert "previous_task_range" in mod.validate_artifact(
        {**report, "previous_task_range": "exp5510-exp5522"}
    )
    assert "next_task_range" in mod.validate_artifact(
        {**report, "next_task_range": "exp5550-exp5563"}
    )
    assert "honest_verdict" in mod.validate_artifact({**report, "honest_verdict": "maybe"})
    assert "schema" in mod.validate_artifact({k: v for k, v in report.items() if k != "schema"})


def test_write_report_persists_valid_transition_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5536: writer persists the validated deliverable."""

    _make_root(tmp_path)

    artifact = mod.write_report(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert written["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(written) == []

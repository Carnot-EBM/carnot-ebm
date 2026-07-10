"""Tests for the Exp5550 .503 transition receipt.

Spec refs: REQ-REPORT-5550, SCENARIO-REPORT-5550,
SCENARIO-REPORT-5550-BLOCKED-INPUT, SCENARIO-REPORT-5550-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5550_transition_v503 as mod


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
                        "# Research Roadmap vNEXT - 2026.07.503",
                        "",
                        "Milestone: `2026.07.503`",
                        "Previous milestone: `2026.07.502`",
                        "Task range: `exp5550` - `exp5563`",
                        "",
                        "Dependency Graph",
                        "exp5552 -> exp5553 -> exp5554",
                        "exp5555 -> exp5556",
                        "exp5557 -> exp5558 -> exp5559",
                        "exp5561 -> exp5562",
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
                        "| 2026-07-10 15:05 UTC | SOTA hard-soft panel duration and substrate corrig | OK | 88 passed |",
                        "| 2026-07-10 15:19 UTC | Gram2Token-inspired grammar table preflight | OK | 87 passed |",
                        "| 2026-07-10 15:48 UTC | Gated SOTA hard-soft live panel v3 | OK | 89 passed |",
                        "| 2026-07-10 16:02 UTC | LLM-FSM inspired exact finite-state fixture | OK | 87 passed |",
                        "| 2026-07-10 16:15 UTC | CSL residue metric independence corrigendum | OK | 85 passed |",
                        "| 2026-07-10 16:29 UTC | Gated retrieval-warmed CSL five-arm ablation | FLAGGED | adversarial_verify CRITICAL: TAUTOLOGY |",
                        "| 2026-07-10 16:46 UTC | Gated cross-model SOTA CSL transfer | OK | 85 passed |",
                        "| 2026-07-10 17:09 UTC | Gated sparse repair FSM descriptor scale | OK | 88 passed |",
                        "| 2026-07-10 17:24 UTC | Hardware receipt substrate corrigendum | OK | 85 passed |",
                        "| 2026-07-10 17:36 UTC | ARC no-LLM substrate and provenance precheck | OK | 87 passed |",
                        "| 2026-07-10 17:50 UTC | Gated ARC clean live level-up attempt | OK | 88 passed |",
                        "| 2026-07-10 19:30 UTC | V502 capstone reconciliation and claim boundaries | OK | cache hit: 87 passed |",
                        "| 2026-07-10 20:23 UTC | Plan milestone 2026.07.503 | OK | 14 tasks proposed |",
                        "| 2026-07-10 20:25 UTC | Milestone 2026.07.503 activated | OK | 14 tasks queued |",
                    ]
                )
                + "\n",
            )
        else:
            _write_text(root, rel_path)
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH)


def _artifact_payloads() -> dict[Path, JsonDict]:
    return {
        mod.PRIOR_CAPSTONE_RELATIVE_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "task_range": mod.PREVIOUS_TASK_RANGE,
            "artifacts_expected": 14,
            "artifacts_read": 14,
            "missing_artifacts": [],
            "skipped_by_gates": [
                {
                    "artifact_path": "results/experiment_5544_cross_model_sota_csl_transfer.json",
                    "skip_reason": "blocked_or_gated",
                    "honest_verdict": "blocked: cross_model_sota_csl_transfer_claim_not_allowed",
                }
            ],
            "flagged_artifacts": [
                {
                    "artifact_path": "results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json",
                    "skip_reason": "flagged_adversarial",
                    "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
                }
            ],
            "honest_nulls": [
                {
                    "artifact_path": "results/experiment_5540_sota_hard_soft_live_panel_v3.json",
                    "null_reason": "clean_honest_null",
                },
                {
                    "artifact_path": "results/experiment_5548_arc_clean_live_levelup.json",
                    "null_reason": "clean_honest_null",
                },
            ],
            "structured_sota_claim_allowed": False,
            "sota_hard_soft_claim_allowed": False,
            "continuous_self_learning_evidence": True,
            "csl_claim_allowed": False,
            "sparse_repair_claim_allowed": True,
            "hardware_speedup_claim": False,
            "arc_registry_delta": 0,
            "reproduced_levels": 0,
            "protected_files_unchanged": {
                "research-roadmap.yaml": True,
                "scripts/research_conductor.py": True,
            },
            "honest_verdict": "complete: .502 capstone read 14/14 expected artifacts; flagged=1; skipped_by_gates=1; honest_nulls=2; hardware_speedup_claim=False; arc_registry_delta=0",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        Path("results/experiment_5538_sota_panel_duration_substrate_corrigendum.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "sota_panel_duration_corrigendum_ready": True,
            "adversarial_clean": True,
            "quality_claim_allowed": False,
            "rows_requested": 3,
            "rows_emitted": 0,
            "schema_validity_rate": 0.0,
            "honest_verdict": "complete: live_sota_hard_soft_panel_claim_downgraded_no_quality_claim",
            "inference_substrate": "live_local_sota_gguf_panel_or_claim_downgrade",
        },
        Path("results/experiment_5539_gram2token_grammar_table_preflight.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "grammar_table_preflight_ready": True,
            "valid_fixture_acceptance_rate": 1.0,
            "invalid_fixture_rejection_rate": 1.0,
            "schema_transition_table_row_count": 8,
            "honest_verdict": "complete: gram2token_preflight_ready_llama_cpp_gbnf_schema_reachable_no_llm_no_speedup_or_quality_claim",
            "inference_substrate": "deterministic_grammar_table_preflight_no_llm",
        },
        Path("results/experiment_5540_sota_hard_soft_live_panel_v3.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "gates_clean": True,
            "sota_hard_soft_claim_allowed": False,
            "rows_requested": 6,
            "rows_emitted": 2,
            "schema_valid_rows": 2,
            "exact_validator_accuracy": 1.0,
            "missing_candidate_rows": 4,
            "honest_verdict": "complete: sota_hard_soft_live_panel_v3_honest_null_no_claim_missing_candidate_rows_schema_invalid_or_missing_rows",
            "inference_substrate": "live_local_sota_gguf_exact_validated_panel",
        },
        Path("results/experiment_5541_llm_fsm_exact_fixture.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "exact_fsm_fixture_ready": True,
            "satisfiable_instances": 2,
            "unsatisfiable_instances": 1,
            "ambiguous_instances": 1,
            "honest_verdict": "complete: exact_fsm_fixture_ready_sat_unsat_ambiguous_no_llm",
            "inference_substrate": "deterministic_fsm_exact_fixture_no_llm",
        },
        Path("results/experiment_5542_csl_residue_metric_independence_corrigendum.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "csl_residue_tautology_resolved": True,
            "nonidentical_metric_evidence": True,
            "csl_residue_stress_ready": True,
            "event_only_score": 0.7142857143,
            "topic_only_score": 0.4,
            "honest_verdict": "complete: csl_residue_metric_independence_corrigendum_ready",
            "inference_substrate": "deterministic_csl_residue_corrigendum_no_llm",
        },
        Path("results/experiment_5543_retrieval_warmed_csl_five_arm_ablation.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "csl_five_arm_ready": True,
            "stale_evidence_rejection_rate": 1.0,
            "negative_transfer_rate": 0.0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "honest_verdict": "complete: retrieval_warmed_csl_five_arm_ablation_ready",
            "inference_substrate": "deterministic_retrieval_warmed_csl_no_llm",
        },
        Path("results/experiment_5544_cross_model_sota_csl_transfer.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "csl_claim_allowed": False,
            "no_weight_mutation": True,
            "cross_family_delta_over_shuffled": 0.0,
            "honest_verdict": "blocked: cross_model_sota_csl_transfer_claim_not_allowed",
            "inference_substrate": "live_local_sota_gguf_cross_model_csl",
        },
        Path("results/experiment_5545_sparse_repair_fsm_descriptor_scale.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "sparse_repair_fsm_ready": True,
            "exact_validator_all_repairs_checked": True,
            "unchecked_repair_count": 0,
            "descriptor_guided_success_rate": 1.0,
            "random_block_success_rate": 0.0,
            "speedup_claim_allowed": False,
            "honest_verdict": "complete: exact_checked_sparse_repair_fsm_descriptor_scale_ready_no_speedup_claim",
            "inference_substrate": "exact_checked_sparse_repair_fsm_no_llm",
        },
        Path("results/experiment_5546_hardware_receipt_substrate_corrigendum.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "hardware_receipt_corrigendum_clean": True,
            "matched_timing_available": False,
            "hardware_speedup_claim": False,
            "no_model_specs_required": True,
            "honest_verdict": "complete: no-LLM hardware receipt corrigendum clean; matched_timing_available=false; hardware_speedup_claim=false",
            "inference_substrate": "hardware_receipt_methodology_no_llm",
        },
        Path("results/experiment_5547_arc_no_llm_substrate_precheck.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "status": "complete",
            "arc_clean_precheck_ready": True,
            "selected_game": "g50t",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "honest_verdict": "complete: g50t L3 clean no-LLM ARC precheck ready; no solve claimed",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        Path("results/experiment_5548_arc_clean_live_levelup.json"): {
            "milestone": mod.PREVIOUS_MILESTONE,
            "status": "honest_null",
            "arc_live_levelup_ready": True,
            "selected_game": "g50t",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "registry_delta": 0,
            "reproduced_levels": 0,
            "action_entropy": 1.0,
            "repeated_coordinate_rate": 0.0,
            "honest_verdict": "honest_null: g50t L3 bounded_budget_no_target_level_reproduction; entropy=1.000; repeat_rate=0.000; registry_delta=0",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    _write_context(root)
    for rel_path, payload in _artifact_payloads().items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def _lane_names(rows: list[JsonDict]) -> set[str]:
    return {str(row["lane"]) for row in rows}


def test_req_report_5550_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5550: OpenSpec anchors the V503 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5550") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert mod.PRIOR_CAPSTONE_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.ARTIFACTS.values():
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5550_archives_v502_facts_and_v503_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5550: .502 facts and .503 gates are preserved."""

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
    assert report["previous_milestone"] == mod.PREVIOUS_MILESTONE
    assert report["prior_capstone_path"] == mod.PRIOR_CAPSTONE_RELATIVE_PATH.as_posix()
    assert report["previous_task_range"] == mod.PREVIOUS_TASK_RANGE
    assert report["next_task_range"] == mod.NEXT_TASK_RANGE
    assert report["roadmap_task_ids"] == mod.EXPECTED_TASK_IDS
    assert report["roadmap_doc_task_range"] == mod.NEXT_TASK_RANGE
    assert report["artifacts_missing"] == []
    assert report["source_context_missing"] == [mod.ROADMAP_NEXT_RELATIVE_PATH.as_posix()]
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["sota_row_completion_required"] is True
    assert report["asp_fsm_fixture_required"] is True
    assert report["csl_tautology_corrigendum_required"] is True
    assert report["causal_csl_memory_required"] is True
    assert report["arc_target_rotation_required"] is True
    assert report["hardware_receipt_only"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["honest_verdict"].startswith("complete:")

    assert {
        "duration_substrate_corrigendum",
        "grammar_table_preflight",
        "exact_fsm_fixture",
        "csl_residue_independence",
        "sparse_fsm_repair_signal",
        "hardware_receipt_hygiene",
        "arc_clean_precheck",
    } <= _lane_names(report["clean_lanes"])
    assert {
        "incomplete_sota_rows",
        "causal_csl_prerequisites",
        "sparse_repair_no_speedup",
        "hardware_receipt_only",
        "arc_live_path_no_bank",
    } <= _lane_names(report["bounded_lanes"])
    assert {
        "exp5540_no_sota_hard_soft_claim",
        "hardware_speedup_false",
        "arc_registry_delta_zero",
        "exp5548_arc_honest_null",
    } <= _lane_names(report["blocked_lanes"])
    assert {"exp5543_csl_tautology"} <= _lane_names(report["flagged_lanes"])
    assert {"exp5544_cross_model_transfer_skip"} <= _lane_names(report["skipped_by_gates"])
    assert mod.validate_artifact(report) == []


def test_scenario_report_5550_live_repo_artifacts_match_terminal_boundaries() -> None:
    """SCENARIO-REPORT-5550: live repository artifacts keep the same boundaries."""

    report = mod.build_report(
        root=REPO,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "complete"
    assert report["source_context_missing"] == [mod.ROADMAP_NEXT_RELATIVE_PATH.as_posix()]
    assert report["artifacts_missing"] == []
    assert report["roadmap_task_ids"] == mod.EXPECTED_TASK_IDS
    assert report["roadmap_doc_task_range"] == mod.NEXT_TASK_RANGE
    assert "exp5543_csl_tautology" in _lane_names(report["flagged_lanes"])
    assert "exp5544_cross_model_transfer_skip" in _lane_names(report["skipped_by_gates"])
    assert "exp5540_no_sota_hard_soft_claim" in _lane_names(report["blocked_lanes"])
    assert report["hardware_receipt_only"] is True
    assert report["honest_verdict"].startswith("complete:")
    assert mod.validate_artifact(report) == []


def test_scenario_report_5550_missing_capstone_and_dirty_files_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5550-BLOCKED-INPUT: missing or dirty inputs fail closed."""

    _make_root(tmp_path, omit=mod.PRIOR_CAPSTONE_RELATIVE_PATH)
    (tmp_path / mod.ROADMAP_RELATIVE_PATH).write_text(
        yaml.safe_dump({"milestone": mod.PREVIOUS_MILESTONE, "tasks": []}, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Missing the V503 task-range marker\n",
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
    assert "results/experiment_5549_capstone_v502.json_missing_or_unreadable" in report[
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


def test_scenario_report_5550_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5550-FIELD-PRINCIPLES: malformed receipts are rejected."""

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
    assert "sota_row_completion_required" in mod.validate_artifact(
        {**report, "sota_row_completion_required": "true"}
    )
    assert "skipped_by_gates" in mod.validate_artifact(
        {**report, "skipped_by_gates": "exp5544"}
    )
    assert "field_principles" in mod.validate_artifact(
        {**report, "field_principles": {"milestone": mod.FIELD_PRINCIPLES["milestone"]}}
    )
    assert "inference_substrate" in mod.validate_artifact(
        {**report, "inference_substrate": "live_llm_inference"}
    )
    assert "milestone" in mod.validate_artifact({**report, "milestone": mod.PREVIOUS_MILESTONE})
    assert "previous_milestone" in mod.validate_artifact(
        {**report, "previous_milestone": "2026.07.501"}
    )
    assert "prior_capstone_path" in mod.validate_artifact(
        {**report, "prior_capstone_path": "results/experiment_5548_arc_clean_live_levelup.json"}
    )
    assert "previous_task_range" in mod.validate_artifact(
        {**report, "previous_task_range": "exp5523-exp5535"}
    )
    assert "next_task_range" in mod.validate_artifact(
        {**report, "next_task_range": "exp5564-exp5577"}
    )
    assert "honest_verdict" in mod.validate_artifact({**report, "honest_verdict": "maybe"})
    assert "schema" in mod.validate_artifact({k: v for k, v in report.items() if k != "schema"})
    assert mod._task_range_from_text("Task range: Exp 5550-5563") == mod.NEXT_TASK_RANGE
    assert mod._task_range_from_text("no task range here") is None
    assert mod._int({"flag": True}, "flag") == 1
    assert mod._status_label({"honest_verdict": "honest_null: no bank"}) == "honest_null"
    assert mod._status_label({"honest_verdict": "failed: bad fixture"}) == "failed"
    assert mod._status_label({"honest_verdict": "complete: clean"}) == "complete"
    assert mod._status_label({"honest_verdict": "pending"}) == "unknown"
    assert "prior_capstone_task_range_mismatch" in mod._failed_preconditions(
        artifacts={"prior_capstone": {"milestone": mod.PREVIOUS_MILESTONE, "task_range": "exp0-exp1"}},
        artifacts_missing=[],
        roadmap_milestone=mod.MILESTONE,
        roadmap_task_ids=mod.EXPECTED_TASK_IDS,
        vnext_names_milestone=True,
        vnext_task_range=mod.NEXT_TASK_RANGE,
        roadmap_modified=False,
        conductor_modified=False,
    )


def test_write_report_persists_valid_transition_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5550: writer persists the validated deliverable."""

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

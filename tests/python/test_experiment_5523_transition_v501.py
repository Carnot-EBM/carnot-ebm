"""Tests for the Exp5523 .501 transition receipt.

Spec refs: REQ-REPORT-5523, SCENARIO-REPORT-5523,
SCENARIO-REPORT-5523-BLOCKED-INPUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5523_transition_v501 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_context(root: Path) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == mod.ROADMAP_RELATIVE_PATH:
            path.write_text(
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
                encoding="utf-8",
            )
        elif rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            path.write_text("# consumed pre-staged roadmap placeholder\n", encoding="utf-8")
        elif rel_path == mod.VNEXT_RELATIVE_PATH:
            path.write_text(
                "\n".join(
                    [
                        "# Research Roadmap vNEXT - Milestone 2026.07.501",
                        "",
                        "**Previous milestone:** 2026.07.500",
                        "**Task range:** Exp 5523-5535",
                        "",
                        "Exp 5525 records the SOTA schema taxonomy before Exp 5526 repair loop.",
                        "Exp 5526 repair loop gates Exp 5527 SOTA panel v2.",
                        "Exp 5528 emits the canonical CSL gate before Exp 5529 and Exp 5530.",
                        "Exp 5533 ARC strategy precheck gates Exp 5534 live level-up.",
                        "Hardware stays receipt-only until matched timing exists.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        elif rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            path.write_text(
                "\n".join(
                    [
                        "| 2026-07-10 00:42 UTC | Structured-output positive control for SOTA hard/soft | OK | 91 passed |",
                        "| 2026-07-10 01:01 UTC | Gated SOTA GGUF hard/soft structured evidence panel | OK | 87 passed |",
                        "| 2026-07-10 01:03 UTC | Gated logits-energy sidecar on parsed SOTA rows | GATE_BLOCK | exp5513.sota_structured_panel_ready false |",
                        "| 2026-07-10 01:21 UTC | CSL independent-outcome graph replay and gate-field | OK | 87 passed |",
                        "| 2026-07-10 01:23 UTC | Gated SOTA GGUF CSL memory panel on clean independ | GATE_BLOCK | actual=None == expected=True |",
                        "| 2026-07-10 04:39 UTC | Hardware continuity and timing-methodology receipt | OK | cache hit |",
                        "| 2026-07-10 05:12 UTC | Gated ARC live action-diverse level-up attempt | OK | 90 passed |",
                        "| 2026-07-10 05:38 UTC | Capstone reconciliation for milestone .500 | OK | 86 passed |",
                        "| 2026-07-10 06:30 UTC | Plan milestone 2026.07.501 | OK | 13 tasks proposed |",
                        "| 2026-07-10 06:32 UTC | Milestone 2026.07.501 activated | OK | 13 tasks queued |",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"context for {rel_path.as_posix()}\n", encoding="utf-8")


def _write_artifacts(root: Path) -> None:
    _write_json(
        root,
        mod.ARTIFACTS[5510],
        {
            "milestone": "2026.07.500",
            "next_task_range": "exp5510-exp5522",
            "honest_verdict": "complete: archived .499 terminal evidence into .500 transition receipt",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5511],
        {
            "milestone": "2026.07.500",
            "research_references_updated": False,
            "new_references_added": [],
            "honest_verdict": "complete: V500 source delta found no new actionable deltas",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5512],
        {
            "milestone": "2026.07.500",
            "structured_output_positive_control_ready": True,
            "schema_validity_rate": 1.0,
            "exact_validator_handoff_ready": True,
            "sota_panel_gate_open": True,
            "honest_verdict": "complete: structured_output_positive_control_ready_live_llm_smoke_sota_gate_open",
            "inference_substrate": "structured_output_fixture_or_live_llm_smoke",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5513],
        {
            "milestone": "2026.07.500",
            "sota_structured_panel_ready": False,
            "sota_rows_emitted": 1,
            "missing_candidate_rows": 3,
            "schema_validity_rate": 0.0,
            "exact_validator_accuracy": 0.0,
            "parse_failure_counts": {"schema_invalid": 1},
            "readiness_blockers": [
                "exact_validator_mismatch",
                "missing_candidate_rows",
                "parse_failures",
                "schema_invalid_or_missing_rows",
            ],
            "gpu_offload_verified": True,
            "honest_verdict": "blocked: sota_hard_soft_structured_panel_not_ready",
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5514],
        {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 2 gate(s) failed; exp5513.sota_structured_panel_ready false",
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5515],
        {
            "milestone": "2026.07.500",
            "continuous_self_learning_evidence": True,
            "csl_experience_graph_ready": True,
            "metric_independence_clean": True,
            "csl_gate_fields_resolvable": True,
            "heldout_delta": 1.0,
            "graph_memory_score": 1.0,
            "no_memory_score": 0.0,
            "stale_evidence_rejection_rate": 1.0,
            "negative_transfer_rate": 0.0,
            "honest_verdict": "complete: independent_outcome_graph_memory_gate_repair_ready",
            "inference_substrate": "graph_memory_replay_with_independent_labels",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5516],
        {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "2 of 2 gate(s) failed; actual=None == expected=True",
            "gates_evaluated": [
                {"artifact_field": "metric_independence_clean", "actual": None, "passed": False},
                {"artifact_field": "csl_gate_fields_resolvable", "actual": None, "passed": False},
            ],
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5517],
        {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed; csl_experience_graph_ready actual=None",
            "gates_evaluated": [
                {"artifact_field": "csl_experience_graph_ready", "actual": None, "passed": False}
            ],
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5518],
        {
            "milestone": "2026.07.500",
            "active_constraint_sparse_repair_ready": True,
            "candidate_count": 160,
            "all_candidates_exact_checked": True,
            "exact_fallback_used": True,
            "sparse_repair_success_rate": 1.0,
            "exact_only_success_rate": 1.0,
            "mean_iterations_sparse_repair": 4.0,
            "mean_iterations_exact_only": 8.0,
            "speedup_claim_allowed": False,
            "claim_limits": ["small CPU-local hard/soft fixtures only"],
            "honest_verdict": "complete: exact_checked_sparse_repair_descriptor_interface_ready_no_speedup_claim",
            "inference_substrate": "exact_checked_sparse_repair",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5519],
        {
            "milestone": "2026.07.500",
            "hardware_speedup_claim": False,
            "hardware_speedup_claim_allowed": False,
            "matched_timing_available": False,
            "blocked_devices": [
                {"device": "cpu", "status": "blocked_toolchain"},
                {"device": "cuda", "status": "blocked_toolchain"},
                {"device": "kv260", "status": "blocked_identity"},
                {"device": "gatemate", "status": "blocked_identity"},
            ],
            "polar_fire_receipt": {"status": "reachable"},
            "timing_methodology": {
                "matched_cpu_gpu_fpga_timing_exists": False,
                "workload": "hardware-continuity metadata receipts only",
            },
            "honest_verdict": "complete: hardware continuity receipts collected with blocked devices",
            "inference_substrate": "hardware_receipts",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5520],
        {
            "milestone": "2026.07.500",
            "arc_levelup_candidate_ready": True,
            "registry_precheck_done": True,
            "selected_game": "sb26",
            "selected_level": "L3",
            "action_entropy": 2.5,
            "repeated_coordinate_rate": 0.0,
            "honest_verdict": "complete: sb26 L3 action-diversity precheck ready; no solve claimed",
            "inference_substrate": "arc_live_precheck",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5521],
        {
            "milestone": "2026.07.500",
            "status": "honest_null",
            "selected_game": "sb26",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "reproduced_levels": 0,
            "registry_before_levels": 69,
            "registry_after_levels": 69,
            "registry_delta": 0,
            "action_entropy": 2.481,
            "repeated_coordinate_rate": 0.526,
            "honest_verdict": "honest_null: sb26 L3 bounded_budget_no_target_level_reproduction; registry_delta=0",
            "inference_substrate": "arc_live_agent_self_discovery",
        },
    )
    _write_json(
        root,
        mod.PRIOR_CAPSTONE_RELATIVE_PATH,
        {
            "milestone": "2026.07.500",
            "conductor_unchanged": True,
            "structured_sota_claim_allowed": False,
            "energy_sidecar_headline_allowed": False,
            "continuous_self_learning_evidence": True,
            "csl_claim_allowed": False,
            "sparse_repair_claim_allowed": True,
            "hardware_speedup_claim": False,
            "arc_registry_delta": 0,
            "reproduced_levels": 0,
            "missing_artifacts": [],
            "claim_boundaries": [
                "No structured SOTA claim because Exp5513 is not panel-ready.",
                "No energy-sidecar headline because Exp5514 was conductor-gated and sidecar-only.",
                "No broad CSL claim because downstream SOTA memory/residue lanes did not execute.",
                "Sparse repair claim is bounded to exact-checked descriptor-interface evidence.",
                "No hardware speedup claim without matched timing.",
                "No ARC progress claim because registry_delta and reproduced_levels are zero.",
            ],
            "terminal_evidence": {
                "structured_sota": {
                    "sota_structured_panel_ready": False,
                    "sota_rows_emitted": 1,
                },
                "csl": {
                    "continuous_self_learning_evidence": True,
                    "metric_independence_clean": True,
                    "csl_experience_graph_ready": True,
                },
                "sparse_constraints": {
                    "active_constraint_sparse_repair_ready": True,
                    "speedup_claim_allowed": False,
                },
                "hardware": {
                    "matched_timing_available": False,
                    "hardware_speedup_claim_allowed": False,
                },
                "arc": {"registry_delta": 0, "reproduced_levels": 0},
            },
            "skipped_by_gates": [
                {"artifact_path": mod.ARTIFACTS[5514].as_posix(), "honest_verdict": "blocked_gate_check_failed"},
                {"artifact_path": mod.ARTIFACTS[5516].as_posix(), "honest_verdict": "blocked_gate_check_failed"},
                {"artifact_path": mod.ARTIFACTS[5517].as_posix(), "honest_verdict": "blocked_gate_check_failed"},
            ],
            "honest_verdict": "complete: .500 capstone read 14 result artifacts; arc_registry_delta 0",
            "inference_substrate": "capstone_aggregation_from_upstream_artifacts",
        },
    )


def _lane_names(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["lane"]) for row in rows}


def test_spec_contains_v501_transition_requirement() -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-5523" in spec
    assert "SCENARIO-REPORT-5523" in spec
    assert "SCENARIO-REPORT-5523-BLOCKED-INPUT" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_build_report_archives_v500_facts(tmp_path: Path) -> None:
    _write_context(tmp_path)
    _write_artifacts(tmp_path)

    report = mod.build_report(
        root=tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "complete"
    assert report["milestone"] == "2026.07.501"
    assert report["previous_milestone"] == "2026.07.500"
    assert report["prior_capstone_path"] == "results/experiment_5522_capstone_v500.json"
    assert report["previous_task_range"] == "exp5510-exp5522"
    assert report["next_task_range"] == "exp5523-exp5535"
    assert report["roadmap_doc_task_range"] == "exp5523-exp5535"
    assert report["roadmap_task_ids"] == mod.EXPECTED_TASK_IDS
    assert report["artifacts_missing"] == []
    assert report["source_context_missing"] == []
    assert report["sota_schema_repair_gate_required"] is True
    assert report["csl_canonical_gate_required"] is True
    assert report["arc_strategy_gate_required"] is True
    assert report["hardware_receipt_only"] is True
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["honest_verdict"].startswith("complete:")

    assert {
        "transition_source",
        "deterministic_structured_output_fixture",
        "csl_independent_graph_memory_positive",
        "sparse_descriptor_interface",
        "hardware_receipt_only_posture",
        "arc_target_precheck",
        "capstone_closure",
    } <= _lane_names(report["clean_lanes"])
    assert {"live_sota_schema_rows", "sparse_repair_scale_speedup"} <= _lane_names(
        report["bounded_lanes"]
    )
    assert {
        "energy_sidecar",
        "downstream_csl_gate_sidecar_selection",
        "broad_csl_claims",
        "hardware_matched_timing",
        "arc_registry_delta",
    } <= _lane_names(report["blocked_lanes"])
    assert {"arc_live_no_bank"} <= _lane_names(report["honest_null_lanes"])
    assert {"csl_sidecar_selection_risk", "arc_repeated_coordinate_risk"} <= _lane_names(
        report["flagged_lanes"]
    )
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in report
        assert field in report["field_principles"]


def test_write_report_persists_required_json(tmp_path: Path) -> None:
    _write_context(tmp_path)
    _write_artifacts(tmp_path)

    payload = mod.write_report(
        root=tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == payload
    assert written["reproducibility_checksum"].startswith("sha256:")
    assert written["tests_run"] == list(mod.DEFAULT_TESTS_RUN)


def test_build_report_fails_closed_for_missing_capstone_and_dirty_conductor(
    tmp_path: Path,
) -> None:
    _write_context(tmp_path)
    _write_artifacts(tmp_path)
    (tmp_path / mod.PRIOR_CAPSTONE_RELATIVE_PATH).unlink()
    (tmp_path / mod.ROADMAP_NEXT_RELATIVE_PATH).unlink()
    (tmp_path / mod.ROADMAP_RELATIVE_PATH).write_text(
        yaml.safe_dump({"milestone": "2026.07.500", "tasks": []}, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Missing the task-range marker for this blocked fixture\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path,
        mod.ARTIFACTS[5521],
        {
            "milestone": "2026.07.500",
            "status": "honest_null",
            "selected_game": "sb26",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "reproduced_levels": 0,
            "registry_before_levels": 69,
            "registry_after_levels": 69,
            "honest_verdict": "honest_null: sb26 L3 bounded_budget_no_target_level_reproduction",
        },
    )

    report = mod.build_report(
        root=tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert "results/experiment_5522_capstone_v500.json_missing_or_unreadable" in report[
        "failed_preconditions"
    ]
    assert "prior_capstone_milestone_mismatch" in report["failed_preconditions"]
    assert "research-roadmap.yaml_milestone_mismatch" in report["failed_preconditions"]
    assert "roadmap_task_ids_mismatch" in report["failed_preconditions"]
    assert "vnext_task_range_mismatch" in report["failed_preconditions"]
    assert "research-roadmap.yaml_modified" in report["failed_preconditions"]
    assert "scripts/research_conductor.py_modified" in report["failed_preconditions"]
    assert "research-roadmap-next.yaml" in report["source_context_missing"]
    assert report["blocked_lanes"][-1]["evidence"]["registry_delta"] == 0
    assert report["roadmap_yaml_unchanged"] is False
    assert report["conductor_unchanged"] is False

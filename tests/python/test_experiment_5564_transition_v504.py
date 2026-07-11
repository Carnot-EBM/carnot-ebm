"""Tests for the Exp5564 .504 transition receipt.

Spec refs: REQ-REPORT-5564, SCENARIO-REPORT-5564,
SCENARIO-REPORT-5564-BLOCKED-INPUT, SCENARIO-REPORT-5564-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5564_transition_v504 as mod


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


def _write_context(root: Path) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
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
                        "# Carnot Research Roadmap vNEXT",
                        "**Milestone:** 2026.07.504",
                        "**Task range:** exp5564-exp5577",
                        "Exp5566 exact corpus --[corpus_ready]--> Exp5567 SOTA panel",
                        "Exp5567 --[panel_complete]--> Exp5568 trigger",
                        "Exp5569 memory tournament --[policy_ready]--> Exp5571 reset-free",
                        "Exp5570 KAN update --[kan_ready]--> Exp5571 reset-free",
                        "Exp5571 --> Exp5572 promotion",
                        "Exp5575 SGE precheck --[live_path_ready AND target_unsolved]--> Exp5576",
                        "PTRM slot is separate from the ordinary ARC floor.",
                    ]
                )
                + "\n",
            )
        elif rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                "| 2026-07-10 23:28 UTC | Gated SOTA hard-soft panel v4 | GATE_BLOCK | "
                "Pre-emptive skip: upstream retired (exp5553-gated-gbnf-forced-sota-row-smoke) |\n"
                "| 2026-07-11 13:20 UTC | Milestone 2026.07.504 activated | OK | 14 tasks queued |\n",
            )
        else:
            _write_text(root, rel_path)


def _capstone_payload() -> JsonDict:
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "task_range": mod.PREVIOUS_TASK_RANGE,
        "artifacts_expected": 14,
        "artifacts_read": 13,
        "missing_artifacts": [],
        "structured_sota_claim_allowed": False,
        "sota_hard_soft_claim_allowed": False,
        "continuous_self_learning_evidence": True,
        "csl_claim_allowed": False,
        "cross_model_csl_claim_allowed": False,
        "asp_sparse_repair_claim_allowed": True,
        "hardware_speedup_claim": False,
        "arc_registry_delta": 0,
        "arc_live_levelup_claim_allowed": False,
        "blocked_artifacts": [
            {
                "artifact_path": mod.EXP5552_ROW_COMPLETION_PATH.as_posix(),
                "honest_verdict": "blocked: automaton_row_completion_not_ready_proposal_path_missing_required_rows",
                "block_reason": "blocked: automaton_row_completion_not_ready_proposal_path_missing_required_rows",
                "status": "blocked",
            },
            {
                "artifact_path": mod.EXP5559_CROSS_MODEL_CSL_PATH.as_posix(),
                "honest_verdict": "blocked: causal_cross_model_sota_csl_transfer_v2_claim_not_allowed",
                "block_reason": "blocked: causal_cross_model_sota_csl_transfer_v2_claim_not_allowed",
                "status": "blocked",
                "flagged_adversarial": True,
            },
        ],
        "flagged_artifacts": [
            {
                "artifact_path": mod.EXP5559_CROSS_MODEL_CSL_PATH.as_posix(),
                "honest_verdict": "blocked: causal_cross_model_sota_csl_transfer_v2_claim_not_allowed",
                "flagged_adversarial": True,
                "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            }
        ],
        "skipped_by_gates": [
            {
                "artifact_path": mod.EXP5553_GBNF_SMOKE_PATH.as_posix(),
                "honest_verdict": "blocked_gate_check_failed",
                "skip_reason": "conductor_gate_skip",
                "gate_check_summary": "exp5552 automaton_row_completion_ready failed",
            },
            {
                "artifact_path": mod.EXP5554_PANEL_PATH.as_posix(),
                "honest_verdict": "blocked_gate_check_failed",
                "skip_reason": "conductor_gate_skip_no_artifact_written",
                "source_path": mod.CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            },
        ],
        "honest_nulls": [
            {
                "artifact_path": mod.EXP5562_ARC_LEVELUP_PATH.as_posix(),
                "honest_verdict": "honest_null: r11l L3 bounded_budget_no_target_level_reproduction; registry_delta=0",
                "status": "honest_null",
            }
        ],
        "clean_artifacts": [
            {"artifact_path": mod.EXP5555_ASP_FSM_FIXTURE_PATH.as_posix()},
            {"artifact_path": mod.EXP5556_SPARSE_REPAIR_PATH.as_posix()},
            {"artifact_path": mod.EXP5557_CSL_REPAIR_PATH.as_posix()},
            {"artifact_path": mod.EXP5558_CAUSAL_MEMORY_PATH.as_posix()},
            {"artifact_path": mod.EXP5560_HARDWARE_PATH.as_posix()},
            {"artifact_path": mod.EXP5561_ARC_PRECHECK_PATH.as_posix()},
        ],
        "arc_audit": {
            "source_artifact": mod.EXP5562_ARC_LEVELUP_PATH.as_posix(),
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "registry_delta_raw": 0,
            "reproduced_levels_raw": 0,
            "capstone_counted_as_levelup_attempt": False,
        },
        "honest_verdict": mod.EXPECTED_PREVIOUS_CAPSTONE_VERDICT,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }


def _artifact_payloads() -> dict[Path, JsonDict]:
    return {
        mod.PREVIOUS_CAPSTONE_RELATIVE_PATH: _capstone_payload(),
        mod.EXP5550_TRANSITION_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "honest_verdict": "complete: transition",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        mod.EXP5551_SOURCE_DELTA_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "honest_verdict": "complete: source delta",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        mod.EXP5552_ROW_COMPLETION_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "status": "blocked",
            "honest_verdict": "blocked: automaton_row_completion_not_ready_proposal_path_missing_required_rows",
            "automaton_row_completion_ready": False,
            "row_completion_support_rate": 0.333333,
            "required_row_count": 6,
            "accepted_row_keys": ["qwen::claim_support", "gemma::claim_support"],
            "readiness_blockers": ["proposal_path_missing_required_rows"],
        },
        mod.EXP5553_GBNF_SMOKE_PATH: {
            "status": "blocked",
            "schema": "blocked_gate_check_v1",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "exp5552 automaton_row_completion_ready failed",
        },
        mod.EXP5555_ASP_FSM_FIXTURE_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "honest_verdict": "complete: exact ASP/FSM stable-model fixture ready with no LLM",
            "exact_asp_validator_ready": True,
            "exact_fsm_fixture_extended_ready": True,
            "asp_row_count": 5,
        },
        mod.EXP5556_SPARSE_REPAIR_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "honest_verdict": "complete: asp_fsm_sparse_repair_descriptor_signal_ready_no_speedup_claim",
            "asp_sparse_repair_claim_allowed": True,
            "stable_model_checked_rate": 1.0,
            "descriptor_guided_success_rate": 1.0,
            "random_block_success_rate": 0.228571,
            "exact_only_success_rate": 1.0,
            "matched_timing_available": False,
            "speedup_claim_allowed": False,
            "unchecked_repair_count": 0,
        },
        mod.EXP5557_CSL_REPAIR_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "honest_verdict": "complete: csl_five_arm_tautology_corrigendum_v2_clean",
            "csl_five_arm_clean": True,
            "tautology_resolved": True,
            "aligned_memory_score": 0.8333333333,
            "shuffled_memory_score": 0.25,
            "no_memory_score": 0.1666666667,
            "aligned_delta_over_shuffled": 0.5833333333,
            "duplicated_metric_pairs": [],
        },
        mod.EXP5558_CAUSAL_MEMORY_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "honest_verdict": "complete: causal_write_manage_read_csl_memory_ready",
            "csl_memory_ready": True,
            "csl_claim_allowed": True,
            "quality_delta_vs_shuffled_memory": 1.0,
            "action_impact_delta_vs_no_memory": 0.8333333333,
            "action_selection_changed_count": 5,
            "no_weight_mutation": True,
        },
        mod.EXP5559_CROSS_MODEL_CSL_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "status": "blocked",
            "honest_verdict": "blocked: causal_cross_model_sota_csl_transfer_v2_claim_not_allowed",
            "flagged_adversarial": True,
            "csl_claim_allowed": False,
            "cross_family_delta_over_shuffled": 0.0,
            "negative_transfer_rate": 0.8333333333,
            "aligned_memory_score": 0.1666666667,
            "shuffled_memory_score": 0.1666666667,
            "no_memory_score": 0.1666666667,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
        mod.EXP5560_HARDWARE_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "honest_verdict": "complete: hardware and timing receipt hygiene clean; matched_timing_available=false; repeated_timing_pairs=0; hardware_speedup_claim=false",
            "hardware_speedup_claim": False,
            "matched_timing_available": False,
            "repeated_timing_pairs": 0,
            "roadmap_yaml_unchanged": True,
            "conductor_modified": False,
        },
        mod.EXP5561_ARC_PRECHECK_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "status": "complete",
            "honest_verdict": "complete: r11l L3 FSM ARC precheck ready; no solve claimed",
            "selected_game": "r11l",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
        },
        mod.EXP5562_ARC_LEVELUP_PATH: {
            "milestone": mod.PREVIOUS_MILESTONE,
            "status": "honest_null",
            "honest_verdict": "honest_null: r11l L3 bounded_budget_no_target_level_reproduction; entropy=3.218; repeat_rate=0.500; registry_delta=0",
            "selected_game": "r11l",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "offline_reproduced": False,
            "registry_delta": 0,
            "reproduced_levels": 0,
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    _write_context(root)
    for rel_path, payload in _artifact_payloads().items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def _lane_names(rows: list[JsonDict]) -> set[str]:
    return {str(row["lane"]) for row in rows}


def test_req_report_5564_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5564: OpenSpec anchors the V504 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5564") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert mod.PREVIOUS_CAPSTONE_RELATIVE_PATH.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5564_locks_live_repo_v503_evidence() -> None:
    """SCENARIO-REPORT-5564: live V503 facts and V504 gates are preserved."""

    report = mod.build_report(
        root=REPO,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "complete"
    assert report["milestone"] == mod.MILESTONE
    assert report["previous_milestone"] == mod.PREVIOUS_MILESTONE
    assert report["previous_task_range"] == mod.PREVIOUS_TASK_RANGE
    assert report["next_task_range"] == mod.NEXT_TASK_RANGE
    assert report["previous_capstone_summary"] == mod.EXPECTED_PREVIOUS_CAPSTONE_VERDICT
    assert report["previous_capstone_claims"] == {
        "structured_sota_claim_allowed": False,
        "sota_hard_soft_claim_allowed": False,
        "continuous_self_learning_evidence": True,
        "csl_claim_allowed": False,
        "cross_model_csl_claim_allowed": False,
        "asp_sparse_repair_claim_allowed": True,
        "hardware_speedup_claim": False,
        "arc_registry_delta": 0,
        "arc_live_levelup_claim_allowed": False,
    }
    assert report["artifacts_read"] == 14
    assert report["json_terminal_artifacts_read"] == 13
    assert report["conductor_skip_records_read"] == 1
    assert report["source_context_missing"] == [mod.ROADMAP_NEXT_RELATIVE_PATH.as_posix()]

    assert {
        "exact_asp_fsm_fixture",
        "bounded_sparse_repair",
        "csl_tautology_repair",
        "causal_memory_action_impact",
    } <= _lane_names(report["clean_lanes"])
    assert {
        "sparse_repair_no_speedup",
        "hardware_receipt_no_speedup",
        "arc_live_path_no_bank",
    } <= _lane_names(report["bounded_lanes"])
    assert {
        "grammar_row_completion_blocked",
        "gbnf_row_smoke_gate_skip",
        "hard_soft_panel_skipped",
        "no_matched_hardware_speedup",
        "arc_registry_delta_zero",
    } <= _lane_names(report["blocked_lanes"])
    assert {"cross_family_csl_flagged_null"} <= _lane_names(report["flagged_lanes"])
    assert report["retired_continuations"] == mod.RETIRED_CONTINUATIONS
    assert [row["task_id"] for row in report["verifier_chain"]] == [
        "exp5566-exact-asp-fsm-near-miss-corpus",
        "exp5567-gated-local-sota-solve-verify-asymmetry",
        "exp5568-gated-verifier-coevolution-trigger",
    ]
    assert report["self_learning_chain"][2]["task_id"] == "exp5571-gated-reset-free-sota-continual-harness"
    assert report["arc_chain"][0]["task_id"] == "exp5575-sge-anti-stagnation-live-precheck"
    assert report["ptrm_slot_separate"] is True
    assert report["hardware_claim_allowed"] is False
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["honest_verdict"].startswith("complete:")
    assert mod.validate_artifact(report) == []


def test_scenario_report_5564_tmp_fixture_preserves_alias_mismatches(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5564: landed paths, not prompt aliases, are authoritative."""

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
    assert report["artifacts_read"] == 14
    assert report["prompt_alias_resolution"] == mod.PROMPT_ALIAS_RESOLUTION
    assert all(row["alias_exists"] is False for row in report["prompt_alias_resolution"])
    assert all(row["resolved_exists"] is True for row in report["prompt_alias_resolution"])
    assert report["failed_preconditions"] == []
    assert mod.validate_artifact(report) == []


def test_scenario_report_5564_missing_capstone_and_dirty_files_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5564-BLOCKED-INPUT: missing inputs fail closed."""

    _make_root(tmp_path, omit=mod.PREVIOUS_CAPSTONE_RELATIVE_PATH)
    (tmp_path / mod.ROADMAP_RELATIVE_PATH).write_text(
        yaml.safe_dump({"milestone": mod.PREVIOUS_MILESTONE, "tasks": []}, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Missing V504 task range\n",
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
    assert "previous_capstone_missing_or_unreadable" in report["failed_preconditions"]
    assert "previous_capstone_milestone_mismatch" in report["failed_preconditions"]
    assert "research-roadmap.yaml_milestone_mismatch" in report["failed_preconditions"]
    assert "roadmap_task_ids_mismatch" in report["failed_preconditions"]
    assert "vnext_task_range_mismatch" in report["failed_preconditions"]
    assert "research-roadmap.yaml_modified" in report["failed_preconditions"]
    assert "scripts/research_conductor.py_modified" in report["failed_preconditions"]
    assert report["hardware_claim_allowed"] is False
    assert report["ptrm_slot_separate"] is True
    assert report["roadmap_yaml_unchanged"] is False
    assert report["conductor_unchanged"] is False


def test_scenario_report_5564_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5564-FIELD-PRINCIPLES: malformed receipts are rejected."""

    _make_root(tmp_path)
    report = mod.build_report(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert "hardware_claim_allowed" in mod.validate_artifact(
        {**report, "hardware_claim_allowed": True}
    )
    assert "hardware_claim_allowed" in mod.validate_artifact(
        {**report, "hardware_claim_allowed": "false"}
    )
    assert "ptrm_slot_separate" in mod.validate_artifact(
        {**report, "ptrm_slot_separate": False}
    )
    assert "artifacts_read" in mod.validate_artifact({**report, "artifacts_read": "14"})
    assert "clean_lanes" in mod.validate_artifact({**report, "clean_lanes": "clean"})
    assert "field_principles" in mod.validate_artifact(
        {**report, "field_principles": {"milestone": mod.FIELD_PRINCIPLES["milestone"]}}
    )
    assert "milestone" in mod.validate_artifact({**report, "milestone": mod.PREVIOUS_MILESTONE})
    assert "previous_milestone" in mod.validate_artifact(
        {**report, "previous_milestone": "2026.07.502"}
    )
    assert "previous_task_range" in mod.validate_artifact(
        {**report, "previous_task_range": "exp5536-exp5549"}
    )
    assert "next_task_range" in mod.validate_artifact(
        {**report, "next_task_range": "exp5550-exp5563"}
    )
    assert "roadmap_yaml_unchanged" in mod.validate_artifact(
        {**report, "roadmap_yaml_unchanged": False}
    )
    assert "conductor_unchanged" in mod.validate_artifact(
        {**report, "conductor_unchanged": False}
    )
    assert "inference_substrate" in mod.validate_artifact(
        {**report, "inference_substrate": "live_model"}
    )
    assert "honest_verdict" in mod.validate_artifact({**report, "honest_verdict": "maybe"})
    assert "schema" in mod.validate_artifact({k: v for k, v in report.items() if k != "schema"})
    assert mod._task_range_from_text("Task range: exp5564-exp5577") == mod.NEXT_TASK_RANGE
    assert mod._task_range_from_text("Task range: Exp 5564-5577") == mod.NEXT_TASK_RANGE
    assert mod._task_range_from_text("no range") is None
    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "7"}, "value") == 7
    assert mod._int({"value": "nan"}, "value") == 0
    assert mod._float({"value": 2}, "value") == 2.0
    assert mod._float({"value": "0.25"}, "value") == 0.25
    assert mod._float({"value": "bad"}, "value") == 0.0
    assert mod._float({"value": None}, "value") == 0.0
    assert mod._status_label({"honest_verdict": "failed: bad"}) == "failed"
    assert mod._status_label({"honest_verdict": "pending"}) == "unknown"
    assert {
        "previous_capstone_task_range_mismatch",
        "previous_capstone_summary_mismatch",
    } <= set(
        mod._failed_preconditions(
            capstone={
                "milestone": mod.PREVIOUS_MILESTONE,
                "task_range": "exp0-exp1",
                "honest_verdict": "complete: stale",
            },
            capstone_meta={"loadable": True},
            terminal_records_missing=[],
            roadmap_milestone=mod.MILESTONE,
            roadmap_task_ids=mod.EXPECTED_TASK_IDS,
            vnext_names_milestone=True,
            vnext_task_range=mod.NEXT_TASK_RANGE,
            roadmap_modified=False,
            conductor_modified=False,
        )
    )


def test_write_report_persists_valid_transition_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5564: writer persists the validated deliverable."""

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

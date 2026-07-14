"""Tests for the Exp5603 V506 transition receipt.

Spec refs: REQ-REPORT-5603, SCENARIO-REPORT-5603,
SCENARIO-REPORT-5603-DEPENDENCY-MAP, SCENARIO-REPORT-5603-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5603_transition_v506 as mod


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
                        "milestone": mod.CURRENT_MILESTONE,
                        "tasks": [{"id": task_id} for task_id in mod.EXPECTED_TASK_IDS],
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
                        "# Research Roadmap vNEXT - Milestone 2026.07.506",
                        "**Task range:** Exp5603-Exp5612",
                        "envelope->SOTA panel->exact extension",
                        "KAN-only longitudinal learning",
                        "ARC filter A/B->advisory live attempt",
                        "independent cDLS benchmark->capstone",
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
                        "| 2026-07-13 23:16 UTC | Milestone 2026.07.505 activated | OK | 8 tasks queued |",
                        "| 2026-07-14 01:07 UTC | Counterexample-guided exact verifier extension from clean residuals | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5581-clean-sota-solve-verify-remeasurement) |",
                    ]
                )
                + "\n",
            )
        else:
            _write_text(root, rel_path)


def _terminal_payloads() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5578_transition_v505.json"): {
            "honest_verdict": "complete: archived .504 terminal evidence into .505 gate map",
            "status": "complete",
            "next_task_range": "exp5578-exp5591",
            "clean_lanes": [
                {"lane": "exact_asp_fsm_near_miss_corpus"},
                {
                    "lane": "spline_local_kan_online_energy",
                    "evidence": {"kan_ready": True, "rollback_checksum_match": True},
                },
            ],
            "blocked_or_flagged_lanes": [
                {"lane": "ordinary_arc_registry_delta_zero", "evidence": {"arc_registry_delta": 0}},
            ],
        },
        Path("results/experiment_5579_v505_source_delta_ingestion.json"): {
            "honest_verdict": "complete: accepted 2 non-duplicate actionable V505 source deltas",
            "new_references_added": [{"id": "arXiv:2607.09072"}, {"id": "arXiv:2607.09349"}],
            "closed_scopes_reopened": False,
        },
        Path("results/experiment_5580_parser_forensics_positive_control.json"): {
            "honest_verdict": "blocked_cached_raw_responses_unavailable_hash_only_forensics",
            "cached_rows_audited": 648,
            "raw_response_text_available": False,
            "parser_repair_ready": False,
            "failure_taxonomy": {"truncation": 468, "other": 180},
            "semantic_false_accept_count": 0,
        },
        Path("results/experiment_5581_clean_sota_solve_verify_remeasurement.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "parser_repair_ready failed",
        },
        Path("results/experiment_5583_causal_memory_metric_corrigendum.json"): {
            "honest_verdict": "complete: exp5569_policy_lane_retired_metric_corrigendum_from_cached_rows",
            "policy_ready": False,
            "forward_transfer_delta": 0.0,
            "backward_retention_delta": 0.3333333334,
            "forgetting_delta": 0.25,
            "policy_gate": {
                "policy_benefit_passed": False,
                "retirement_reasons": [
                    "forward_transfer_delta_not_positive",
                    "optimized_forgetting_loss_visible",
                ],
            },
        },
        Path("results/experiment_5584_two_timescale_exact_self_learning.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "policy_ready failed",
        },
        Path("results/experiment_5585_arc_levelup_attempt_v505.json"): {
            "status": "complete",
            "honest_verdict": "complete: no_new_arc_level_banked_lf52_reproduced_l2_prior_l6",
            "game_targeted": "lf52",
            "new_levels_banked": 0,
            "registry_total_before": 177,
            "registry_total_after": 177,
            "registry_updated": True,
        },
    }


def _outer_payloads() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5592_candidate_scoring_stack_bare_control_ab.json"): {
            "honest_verdict": "complete: candidate_stack_honest_null_headroom_present_no_delta",
            "levels_gained_full_stack_total": 1,
            "levels_gained_bare_control_total": 1,
            "efficiency_full_stack_total": 2.7778,
            "efficiency_bare_control_total": 2.7778,
        },
        Path("results/experiment_5599_reinduction_ab_lp85_levelup.json"): {
            "honest_verdict": "complete: reinduction_ab_current_plans_more_reliably",
            "per_arm_summary": {
                "current": {"n_planned": 1, "plan_rate_given_levelup": 0.3333},
                "candidate_27b": {"n_planned": 0, "plan_rate_given_levelup": 0.0},
            },
        },
        Path("results/experiment_5600_ptrm_loo_gate.json"): {
            "honest_verdict": "complete: ptrm_loo_gate_failed_no_majority_significant_and_above_baseline",
            "loo_verdict_reached": True,
            "retire_trm_generator_line": True,
            "games_ptrm_beats_non_recursive_significantly": ["ft09"],
            "heldout_games": ["ft09", "m0r0", "vc33", "sk48", "cd82"],
        },
        Path("results/experiment_5601_object_history_salience_offline_sim_prototype.json"): {
            "honest_verdict": "complete: object_history_salience_prototype_confirmed_2_hashes_with_real_change_signal_across_1_games",
            "total_hashes_with_evidence_and_nonzero_change_rate": 2,
            "total_hashes_tracked": 15,
            "adversarial_degeneracy_check": {"pairs_checked": 8, "pairs_differentiated": 0},
        },
        Path("results/experiment_5602_inert_click_pruner_matched_budget_ab.json"): {
            "honest_verdict": "complete: inert_click_pruner_ab_no_op_offline_and_live_at_this_budget_on_m0r0",
            "reduction_pct": 0.0,
            "states_expanded_reduction": 0,
            "live_wired_supplementary_check": {
                "pruner_stats": {"observed": 32, "pruned": 0, "signatures_tracked": 9}
            },
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    _write_context(root)
    for rel_path, payload in {**_terminal_payloads(), **_outer_payloads()}.items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def _by_key(rows: list[JsonDict], field: str = "key") -> dict[str, JsonDict]:
    return {str(row[field]): row for row in rows}


def test_req_report_5603_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5603: OpenSpec anchors the V506 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5603") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5603_live_repo_locks_v505_and_outer_loop_evidence() -> None:
    """SCENARIO-REPORT-5603: live repository evidence becomes a bounded V506 map."""

    artifact = mod.build_report(
        root=REPO,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["status"] == "complete"
    assert artifact["previous_milestone"] == "2026.07.505"
    assert artifact["current_milestone"] == "2026.07.506"
    assert artifact["current_task_range"] == "exp5603-exp5612"
    assert artifact["task_id_collision_avoidance"] == {
        "previous_outer_loop_last_id": "exp5602",
        "new_range_starts_at": "exp5603",
        "collision_avoided": True,
    }
    assert artifact["missing_artifacts"] == []
    assert artifact["exp5582_preemptive_skip"]["observed"] is True
    findings = _by_key(artifact["terminal_findings"])
    assert findings["hash_only_parser_forensics"]["evidence"] == {
        "cached_rows_audited": 648,
        "raw_response_text_available": False,
        "parser_repair_ready": False,
        "truncation_count": 468,
        "other_failure_count": 180,
        "semantic_false_accept_count": 0,
        "claim_imported": "unrecoverable_instrumentation_only",
    }
    assert findings["causal_memory_corrigendum"]["evidence"]["policy_ready"] is False
    assert findings["causal_memory_corrigendum"]["evidence"]["forward_transfer_delta"] == 0.0
    assert findings["arc_registry_delta"]["evidence"]["new_levels_banked"] == 0
    retired = _by_key(artifact["retired_scopes"])
    assert retired["causal_memory_pace_policy_chain"]["closed"] is True
    assert retired["ptrm_as_generator"]["closed"] is True
    substrates = _by_key(artifact["clean_substrates"])
    assert substrates["spline_local_kan"]["source_artifacts"] == [
        "results/experiment_5578_transition_v505.json"
    ]
    outer = _by_key(artifact["post_milestone_outer_loop_artifacts"], "experiment_id")
    assert outer["exp5592-candidate-scoring-stack-bare-control-ab"]["finding"] == (
        "no_level_or_efficiency_delta"
    )
    assert outer["exp5600-ptrm-loo-gate"]["finding"] == "ptrm_generator_retired"
    assert outer["exp5601-object-history-salience-offline-sim-prototype"]["finding"] == (
        "object_history_signal_found"
    )
    assert outer["exp5602-inert-click-pruner-matched-budget-ab"]["finding"] == (
        "inert_click_no_op"
    )
    assert artifact["dependency_map"]["verification_evidence_chain"]["chain"] == (
        "envelope->SOTA panel->exact extension"
    )
    assert artifact["dependency_map"]["kan_longitudinal_learning"]["chain"] == (
        "KAN-only longitudinal learning"
    )
    assert artifact["dependency_map"]["arc_filter_to_live_attempt"]["chain"] == (
        "ARC filter A/B->advisory live attempt"
    )
    assert artifact["dependency_map"]["cdls_to_capstone"]["chain"] == (
        "independent cDLS benchmark->capstone"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5603_missing_terminal_artifact_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5603: missing terminal V505 evidence blocks the receipt."""

    missing = Path("results/experiment_5580_parser_forensics_positive_control.json")
    _make_root(tmp_path, omit=missing)

    artifact = mod.build_report(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["status"] == "blocked"
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["terminal_findings"][1]["evidence"]["claim_imported"] == (
        "unrecoverable_instrumentation_only"
    )
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5603_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5603-FIELD-PRINCIPLES: malformed fields fail validation."""

    _make_root(tmp_path)
    artifact = mod.build_report(
        root=tmp_path,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert mod.validate_artifact(artifact) == []
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"artifacts_read": mod.FIELD_PRINCIPLES["artifacts_read"]}}
    )
    assert "artifacts_read" in mod.validate_artifact({**artifact, "artifacts_read": "all"})
    assert "terminal_findings" in mod.validate_artifact(
        {**artifact, "terminal_findings": "findings"}
    )
    assert "terminal_findings" in mod.validate_artifact({**artifact, "terminal_findings": []})
    assert "retired_scopes" in mod.validate_artifact({**artifact, "retired_scopes": []})
    assert "clean_substrates" in mod.validate_artifact({**artifact, "clean_substrates": []})
    assert "post_milestone_outer_loop_artifacts" in mod.validate_artifact(
        {**artifact, "post_milestone_outer_loop_artifacts": "outer"}
    )
    assert "post_milestone_outer_loop_artifacts" in mod.validate_artifact(
        {**artifact, "post_milestone_outer_loop_artifacts": []}
    )
    assert "current_task_range" in mod.validate_artifact(
        {**artifact, "current_task_range": "exp5604-exp5612"}
    )
    assert "dependency_map" in mod.validate_artifact({**artifact, "dependency_map": {}})
    assert "roadmap_yaml_unchanged" in mod.validate_artifact(
        {**artifact, "roadmap_yaml_unchanged": False}
    )
    assert "roadmap_yaml_unchanged" in mod.validate_artifact(
        {**artifact, "roadmap_yaml_unchanged": "true"}
    )
    assert "conductor_unchanged" in mod.validate_artifact(
        {**artifact, "conductor_unchanged": False}
    )
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "unknown"})
    assert mod._task_range_from_text("**Task range:** Exp5603-Exp5612") == "exp5603-exp5612"
    assert mod._task_range_from_text("no range here") is None
    assert mod._status_label({"status": "blocked"}) == "blocked"
    assert mod._status_label({"honest_verdict": "complete: x"}) == "complete"
    assert mod._status_label({"honest_verdict": "blocked_gate_check_failed"}) == "blocked"
    assert mod._status_label({"honest_verdict": "honest_null: x"}) == "honest_null"
    assert mod._status_label({"honest_verdict": "failed: x"}) == "failed"
    assert mod._status_label({"honest_verdict": "unclear"}) == "unknown"
    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "7"}, "value") == 7
    assert mod._int({"value": "bad"}, "value") == 0
    assert mod._float({"value": "1.25"}, "value") == 1.25
    assert mod._float({"value": "bad"}, "value") == 0.0
    assert mod._float({"value": None}, "value") == 0.0
    assert mod._failed_preconditions(
        [],
        exp5582_skip_observed=False,
        roadmap_modified=True,
        conductor_modified=True,
    ) == [
        "exp5582_preemptive_skip_not_observed",
        "research-roadmap.yaml_modified",
        "scripts/research_conductor.py_modified",
    ]
    assert mod._read_json_any(tmp_path / "missing.json")[1]["error"] == "missing"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._read_json_any(malformed)[1]["error"] == "malformed_json"
    list_json = tmp_path / "list.json"
    list_json.write_text("[1, 2]", encoding="utf-8")
    assert mod._read_json_any(list_json)[1]["length"] == 2


def test_scenario_report_5603_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5603: writer persists the tested transition receipt."""

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
    assert mod.validate_artifact(written) == []

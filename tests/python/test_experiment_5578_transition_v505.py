"""Tests for the Exp5578 V505 transition receipt.

Spec refs: REQ-REPORT-5578, SCENARIO-REPORT-5578,
SCENARIO-REPORT-5578-MISSING-INPUT, SCENARIO-REPORT-5578-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5578_transition_v505 as mod


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
                        "milestone": mod.MILESTONE,
                        "tasks": [{"id": task_id} for task_id in mod.EXPECTED_TASK_IDS[:8]],
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
                        "# Research Roadmap vNEXT - Milestone 2026.07.505",
                        "**Task range:** Exp5578-Exp5591",
                        "parser fixtures -> local SOTA panel -> exact verifier extension",
                        "memory corrigendum -> two-timescale -> live -> promotion",
                        "EOM-MCTS precheck -> gated ordinary ARC level-up",
                        "reserved PTRM LOO and matched hardware crossover lanes",
                    ]
                )
                + "\n",
            )
        elif rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                "| 2026-07-13 23:16 UTC | Milestone 2026.07.505 activated | OK | 8 tasks queued |\n",
            )
        else:
            _write_text(root, rel_path)


def _payloads() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5564_transition_v504.json"): {
            "honest_verdict": "complete: transition",
            "milestone": "2026.07.504",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5565_v504_source_delta_ingestion.json"): {
            "honest_verdict": "complete: accepted source delta",
            "closed_scopes_reopened": False,
            "inference_substrate": "web_and_repository_source_synthesis",
        },
        Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"): {
            "honest_verdict": "complete: exact corpus",
            "corpus_ready": True,
            "n_rows": 120,
            "duplicate_leakage_count": 0,
        },
        Path("results/experiment_5567_local_sota_solve_verify_asymmetry.json"): {
            "honest_verdict": "complete: authenticated paired panel",
            "panel_complete": True,
            "live_model_invoked": True,
            "gpu_offload_authenticated": True,
            "parser_failure_count": 648,
            "n_candidate_labels": 576,
            "error_taxonomy": {"parser_failure": 648},
            "solve_accuracy_by_model": {
                "qwen": {"accuracy": 0.0, "parser_failures": 36},
                "gemma": {"accuracy": 0.0, "parser_failures": 36},
            },
        },
        Path("results/experiment_5568_verifier_coevolution_trigger.json"): {
            "honest_verdict": "complete: coevolution triggered",
            "verifier_coevolution_required": True,
            "triggered_by": ["worst_family_false_accept_rate"],
        },
        Path("results/experiment_5569_causal_memory_policy_tournament.json"): {
            "honest_verdict": "complete: causal memory policy ready",
            "flagged_adversarial": True,
            "policy_ready": True,
            "forward_transfer_delta": 0.3333333334,
            "backward_retention_delta": 0.3333333334,
            "rollback_success": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
        Path("results/experiment_5570_spline_local_kan_online_energy.json"): {
            "honest_verdict": "complete: active_spline_online_kan_exact_energy_ready",
            "kan_ready": True,
            "forward_adaptation_delta": 0.425,
            "paired_ci_active_vs_frozen": {"lower": 0.425},
            "prior_family_regression": 0.0,
            "unsafe_false_accept_delta": -0.85,
            "rollback_checksum_match": True,
            "exact_feedback_only": True,
        },
        Path("results/experiment_5571_reset_free_sota_continual_harness.json"): {
            "honest_verdict": "blocked_no_cuda_offload",
            "continual_harness_candidate": False,
            "live_model_invoked": False,
            "gpu_offload_authenticated": False,
        },
        Path("results/experiment_5572_gated_delayed_regression_promotion.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "continual_harness_candidate failed",
        },
        Path("results/experiment_5573_matched_sampler_hardware_continuity.json"): {
            "honest_verdict": "complete: matched CPU/CUDA sampler evidence recorded",
            "successful_matched_pairs": 6,
            "hardware_speedup_claim_allowed": True,
            "board_speedup_claimed": False,
            "speedup_by_pair": [{"speedup": 0.24}, {"speedup": 0.26}],
        },
        Path("results/experiment_5574_ptrm_stochastic_generator_stage1.json"): {
            "honest_verdict": "complete: stage1_ptrm_substrate_trained_remaining_loo_gate_preserved",
            "track": "arc-trm-generator",
            "stage1_training_complete": True,
            "loo_verdict_reached": False,
            "no_level_solve_claim": True,
            "solve_provenance": "development_proxy",
            "retire_trm_generator_line": False,
        },
        Path("results/experiment_5575_sge_anti_stagnation_live_precheck.json"): {
            "honest_verdict": "blocked: live path not ready",
            "live_path_reachable": True,
            "live_path_ready": False,
            "target_unsolved": True,
            "solve_provenance": "live_agent_self_discovery",
        },
        Path("results/experiment_5576_gated_sge_live_levelup.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "live_path_ready failed",
        },
        Path("results/experiment_5577_capstone_v504.json"): {
            "honest_verdict": (
                "complete: .504 capstone read 14/14 expected artifacts; missing=0; "
                "flagged=1; blocked=2; skipped=2; promoted=4; "
                "solve_verify_asymmetry_supported=False; verifier_coevolution_required=True; "
                "memory_policy_promoted=False; kan_online_energy_promoted=True; "
                "continuous_self_learning_claim_allowed=False; "
                "hardware_speedup_claim_allowed=False; ordinary_arc_floor_satisfied=False; "
                "arc_registry_delta=0"
            ),
            "inference_substrate": "aggregation_from_all_v504_artifacts",
            "expected_task_range": "exp5564-exp5577",
            "upstream_artifacts_read": [
                "results/experiment_5564_transition_v504.json",
                "results/experiment_5565_v504_source_delta_ingestion.json",
            ],
            "missing_artifacts": [],
            "solve_verify_asymmetry_supported": False,
            "memory_policy_promoted": False,
            "kan_online_energy_promoted": True,
            "continuous_self_learning_claim_allowed": False,
            "hardware_speedup_claim_allowed": False,
            "ordinary_arc_floor_satisfied": False,
            "arc_registry_delta": 0,
            "sge_retired": True,
            "ptrm_stage1_status": "complete_development_proxy_no_solve_claim",
            "arc_solve_provenance": {
                "ptrm": {"counts_as_ordinary_arc_slot": False},
                "ordinary_arc": {"registry_delta_counted": 0},
            },
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    _write_context(root)
    for rel_path, payload in _payloads().items():
        if rel_path != omit:
            _write_json(root, rel_path, payload)


def _lane_names(rows: list[JsonDict]) -> set[str]:
    return {str(row["lane"]) for row in rows}


def test_req_report_5578_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5578: OpenSpec anchors the V505 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5578") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5578_live_repo_locks_v504_evidence() -> None:
    """SCENARIO-REPORT-5578: live V504 facts become a bounded V505 gate map."""

    artifact = mod.build_report(
        root=REPO,
        tests_run=["unit"],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["status"] == "complete"
    assert artifact["previous_task_range"] == "exp5564-exp5577"
    assert artifact["next_task_range"] == "exp5578-exp5591"
    assert artifact["missing_artifacts"] == []
    assert [row["path"] for row in artifact["artifacts_read"]] == [
        path.as_posix() for path in mod.EXPECTED_ARTIFACT_PATHS
    ]
    assert artifact["parser_collapse_preserved"] == {
        "source_artifact": "results/experiment_5567_local_sota_solve_verify_asymmetry.json",
        "parser_failure_count": 648,
        "n_candidate_labels": 576,
        "panel_complete": True,
        "live_model_invoked": True,
        "gpu_offload_authenticated": True,
        "classification": "instrumentation_failure_not_model_evidence",
        "solve_or_verify_result_imported": False,
    }
    assert {
        "exact_asp_fsm_near_miss_corpus",
        "verifier_coevolution_trigger",
        "spline_local_kan_online_energy",
    } <= _lane_names(artifact["clean_lanes"])
    assert {
        "parser_collapse_instrumentation",
        "memory_tautology_flag",
        "reset_free_cuda_block",
        "delayed_promotion_gate_skip",
        "ptrm_loo_not_reached",
        "sge_retired_or_skipped",
        "ordinary_arc_registry_delta_zero",
        "hardware_speedup_not_supported",
    } <= _lane_names(artifact["blocked_or_flagged_lanes"])
    memory = {
        row["lane"]: row for row in artifact["blocked_or_flagged_lanes"]
    }["memory_tautology_flag"]
    assert memory["evidence"]["policy_ready"] is True
    assert memory["evidence"]["flagged_adversarial"] is True
    assert memory["evidence"]["forward_transfer_delta"] == 0.3333333334
    assert memory["evidence"]["backward_retention_delta"] == 0.3333333334
    assert artifact["gate_map"]["parser_panel_exact_extension"]["steps"] == [
        "exp5580-parser-forensics-positive-control",
        "exp5581-clean-sota-solve-verify-remeasurement",
        "exp5582-exact-counterexample-verifier-extension",
    ]
    assert artifact["gate_map"]["memory_two_timescale_live_promotion"]["steps"] == [
        "exp5583-causal-memory-metric-corrigendum",
        "exp5584-two-timescale-exact-self-learning",
        "exp5585-reset-free-live-local-sota-sessions",
        "exp5586-delayed-promotion-and-poisoning-gate",
    ]
    assert artifact["gate_map"]["ordinary_arc_eom_live"]["steps"] == [
        "exp5588-epistemic-object-model-mcts-live-precheck",
        "exp5589-gated-ordinary-arc-level-up",
    ]
    assert artifact["gate_map"]["ptrm_independent_lane"]["counts_as_ordinary_arc"] is False
    assert artifact["gate_map"]["hardware_independent_lane"]["speedup_claim_allowed"] is False
    assert artifact["ptrm_slot_separate"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5578_missing_upstream_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5578-MISSING-INPUT: missing V504 evidence blocks receipt."""

    missing = Path("results/experiment_5570_spline_local_kan_online_energy.json")
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
    assert "spline_local_kan_online_energy" not in _lane_names(artifact["clean_lanes"])
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5578_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5578-FIELD-PRINCIPLES: malformed fields fail validation."""

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
    assert "clean_lanes" in mod.validate_artifact({**artifact, "clean_lanes": "clean"})
    assert "blocked_or_flagged_lanes" in mod.validate_artifact(
        {**artifact, "blocked_or_flagged_lanes": []}
    )
    assert "parser_collapse_preserved" in mod.validate_artifact(
        {**artifact, "parser_collapse_preserved": {"parser_failure_count": 0}}
    )
    assert "previous_task_range" in mod.validate_artifact(
        {**artifact, "previous_task_range": "exp5564-exp5576"}
    )
    assert "next_task_range" in mod.validate_artifact(
        {**artifact, "next_task_range": "exp5578-exp5585"}
    )
    assert "ptrm_slot_separate" in mod.validate_artifact({**artifact, "ptrm_slot_separate": False})
    assert "ptrm_slot_separate" in mod.validate_artifact(
        {**artifact, "ptrm_slot_separate": "true"}
    )
    assert "roadmap_yaml_unchanged" in mod.validate_artifact(
        {**artifact, "roadmap_yaml_unchanged": False}
    )
    assert "conductor_unchanged" in mod.validate_artifact(
        {**artifact, "conductor_unchanged": False}
    )
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_local_sota"}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "parser_collapse_preserved" in mod.validate_artifact(
        {**artifact, "parser_collapse_preserved": "parser"}
    )
    assert mod._task_range_from_text("**Task range:** exp5578-exp5591") == "exp5578-exp5591"
    assert mod._task_range_from_text("no range here") is None
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
    assert mod._minimum_speedup({}) is None
    assert mod._failed_preconditions(
        [],
        roadmap_modified=True,
        conductor_modified=True,
    ) == ["research-roadmap.yaml_modified", "scripts/research_conductor.py_modified"]
    assert mod._read_json_any(tmp_path / "missing.json")[1]["error"] == "missing"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._read_json_any(malformed)[1]["error"] == "malformed_json"
    list_json = tmp_path / "list.json"
    list_json.write_text("[1, 2]", encoding="utf-8")
    assert mod._read_json_any(list_json)[1]["length"] == 2


def test_scenario_report_5578_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5578: writer persists the tested transition receipt."""

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

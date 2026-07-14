"""Tests for Exp5612 V506 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5612, SCENARIO-CAPSTONE-5612,
SCENARIO-CAPSTONE-5612-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5612-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5612_v506_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


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
        Path("results/experiment_5603_transition_v506.json"): {
            "schema": "carnot.experiment_5603.transition_v506.v1",
            "experiment_id": "exp5603-transition-v506",
            "current_milestone": mod.MILESTONE,
            "current_task_range": "exp5603-exp5612",
            "roadmap_task_ids": list(mod.EXPECTED_TASK_IDS),
            "missing_artifacts": [],
            "honest_verdict": "complete: transition",
            "inference_substrate": "aggregation_from_repository_artifacts",
        },
        Path("results/experiment_5604_v506_source_delta_ingestion.json"): {
            "schema": "carnot.experiment_5604.v506_source_delta_ingestion.v1",
            "experiment_id": "exp5604-v506-source-delta-ingestion",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: source delta",
            "flagged_adversarial": True,
            "closed_scopes_reopened": False,
            "new_references_added": [{"source_id": "lazy_identity_deq_2607_11116"}],
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
        Path("results/experiment_5605_raw_response_evidence_envelope.json"): {
            "schema": "carnot.experiment_5605.raw_response_evidence_envelope.v506",
            "experiment_id": "exp5605-raw-response-evidence-envelope",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: envelope",
            "gpu_offload_authenticated": True,
            "response_rows_written": 8,
            "raw_payloads_preserved": True,
            "lossless_replay_rate": 1.0,
            "semantic_false_accept_count": 0,
            "parser_version_replay_passed": True,
            "payload_corruption_rejected": True,
            "truncation_controls_detected": 2,
            "envelope_ready": True,
        },
        Path("results/experiment_5606_clean_sota_solve_verify_evidence_panel.json"): {
            "schema": "carnot.experiment_5606.clean_sota_solve_verify_evidence_panel.v506",
            "experiment_id": "exp5606-clean-sota-solve-verify-evidence-panel",
            "milestone": mod.MILESTONE,
            "honest_verdict": "blocked_no_cuda_offload_authenticated_cpu_fallback_rejected",
            "panel_complete": False,
            "gpu_offload_authenticated": False,
            "raw_response_replay_passed": True,
            "maximum_parser_failure_rate": 1.0,
            "maximum_truncation_rate": 0.722222,
            "solve_verify_asymmetry_supported": False,
        },
        Path("results/experiment_5607_property_template_exact_residual_extension.json"): {
            "schema": "blocked_gate_check_v1",
            "experiment": 5607,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "2 of 2 gate(s) failed",
            "gates_evaluated": [{"passed": False}, {"passed": False}],
        },
        Path("results/experiment_5608_kan_longitudinal_self_learning.json"): {
            "schema": "carnot.experiment_5608.kan_longitudinal_self_learning.v1",
            "experiment_id": "experiment_5608_kan_longitudinal_self_learning",
            "task_id": "exp5608-kan-longitudinal-self-learning",
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: exact_gated_active_spline_kan_longitudinal_ready",
            "continuous_self_learning_task": True,
            "kan_longitudinal_ready": True,
            "promotion_gate": {
                "heldout_delta_positive": True,
                "heldout_uncertainty_excludes_zero": True,
                "backward_retention_nonnegative": True,
                "unsafe_false_accept_count_zero": True,
                "rollback_positive_control": True,
                "delayed_regression_passed": True,
                "no_model_weight_mutation": True,
            },
            "forward_transfer_delta": 0.0,
            "backward_retention_delta": 0.5,
            "forgetting_delta": -0.5,
            "unsafe_false_accept_count": 0,
            "rollback_positive_control": True,
            "delayed_regression_passed": True,
            "no_model_weight_mutation": True,
            "kan_weights_mutated": True,
            "llm_calls": 0,
            "llm_weight_training": False,
        },
        Path("results/experiment_5609_arc_filter_intermediate_invariance_ab.json"): {
            "schema": "carnot.exp5609.arc_filter_intermediate_invariance_ab.v1",
            "experiment": "experiment_5609_arc_filter_intermediate_invariance_ab",
            "honest_verdict": "complete: arc_filter_ab_reachable_repeat_noop_filters_retired",
            "solve_provenance": "development_proxy",
            "mechanism_reachability_controls": {
                "ok": True,
                "inert_click": {"reachable": True},
                "object_history": {"reachable": True},
            },
            "levels_gained_by_arm": {
                "baseline": 0,
                "inert_only": 0,
                "history_only": 0,
                "combined": 0,
            },
            "filter_promotion_decisions": {
                "inert_click": {
                    "decision": "retire_reachable_downstream_noop",
                    "reachable": True,
                    "downstream_improved": False,
                    "safety_regression": False,
                },
                "object_history": {
                    "decision": "retire_reachable_downstream_noop",
                    "reachable": True,
                    "downstream_improved": False,
                    "safety_regression": False,
                },
            },
            "offline_reproduced": {"exact_known_level_safety": True},
        },
        Path("results/experiment_5610_arc_live_self_discovery_levelup_v506.json"): {
            "schema": "arc_live_self_discovery_levelup_attempt.v1",
            "experiment": "experiment_5610_arc_live_self_discovery_levelup_v506",
            "experiment_id": 5610,
            "milestone": mod.MILESTONE,
            "honest_verdict": "complete: no_new_arc_level_banked_sk48_L8_bounded_live_attempt",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "solve_provenance": "live_agent_self_discovery",
            "live_attempt_executed": True,
            "source_files_read": False,
            "per_game_adapter_used": False,
            "levels_before": 177,
            "levels_after": 177,
            "new_reproducible_levels": [],
            "offline_reproduced": False,
            "registry_updated": False,
            "target_reached_live": False,
            "attempt_trace_path": "results/experiment_5610_arc_live_self_discovery_levelup_v506_trace.json",
        },
        Path("results/experiment_5610_arc_live_self_discovery_levelup_v506_trace.json"): {
            "schema": "arc_live_self_discovery_attempt_trace.v1",
            "experiment": "experiment_5610_arc_live_self_discovery_levelup_v506",
            "selected_game": "sk48",
            "selected_level": "L8",
            "executed_actions": [{"action": 0}],
            "observations": [{"state": 0}],
            "level_counter_changes": [],
            "reproduction_gate": {"attempted": False, "reproduced": False},
        },
        Path("results/experiment_5611_cdls_matched_sampler_crossover.json"): {
            "schema": "carnot.experiment_5611.cdls_matched_sampler_crossover.v1",
            "experiment_id": "exp5611-cdls-matched-sampler-crossover",
            "milestone": mod.MILESTONE,
            "honest_verdict": "blocked: cDLS matched CPU/CUDA comparison unavailable",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "board_speedup_claimed": False,
            "crossover_claim_allowed": False,
            "crossover_size": None,
            "successful_matched_pairs": 0,
            "quality_equivalence_gate": {"defined_before_timing": True},
            "speedup_by_pair": [],
            "preconditions": {
                "blocked_reasons": [],
                "cpu_available": True,
                "cuda_available": True,
                "descriptor_available": True,
            },
        },
    }


def _make_root(
    root: Path,
    *,
    omit: Path | None = None,
    malformed: Path | None = None,
) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == Path("research-roadmap-next.yaml"):
            continue
        _write_text(root, rel_path)
    for rel_path, payload in _payloads().items():
        if rel_path == omit:
            continue
        if rel_path == malformed:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json\n", encoding="utf-8")
            continue
        _write_json(root, rel_path, payload)


def test_req_capstone_5612_spec_declares_v506_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5612: OpenSpec declares the V506 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5612") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.PRIMARY_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5612_live_repo_preserves_terminal_statuses() -> None:
    """SCENARIO-CAPSTONE-5612: live V506 evidence stays narrow and terminal."""

    artifact = mod.run_capstone(
        root=REPO,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    statuses = artifact["terminal_status_by_task"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["expected_task_ids"] == list(mod.EXPECTED_TASK_IDS)
    assert artifact["missing_tasks"] == []
    assert artifact["malformed_tasks"] == []
    assert statuses["exp5604-v506-source-delta-ingestion"]["status"] == "flagged"
    assert statuses["exp5606-clean-sota-solve-verify-evidence-panel"]["status"] == "blocked"
    assert statuses["exp5607-property-template-exact-residual-extension"]["status"] == "gate_skipped"
    assert statuses["exp5610-arc-live-self-discovery-levelup-v506"]["status"] == "flagged"
    assert statuses["exp5611-cdls-matched-sampler-crossover"]["status"] == "flagged"

    assert artifact["headline_claims"]["response_evidence"]["claim_allowed"] is True
    assert artifact["headline_claims"]["solve_verify_asymmetry"]["claim_allowed"] is False
    assert artifact["headline_claims"]["exact_predicate_extension"]["claim_allowed"] is False
    assert artifact["headline_claims"]["kan_longitudinal_self_learning"]["claim_allowed"] is True
    assert artifact["headline_claims"]["arc_new_registry_levels"]["claim_allowed"] is False
    assert artifact["headline_claims"]["cdls_crossover"]["claim_allowed"] is False
    assert artifact["promotion_decisions"]["kan_longitudinal_self_learning"]["decision"] == "promote_bounded"
    assert artifact["retirement_decisions"]["local_sota_solve_verify_panel"]["decision"] == "retire"
    assert artifact["retirement_decisions"]["arc_inert_click_filter"]["decision"] == "retire"
    assert artifact["retirement_decisions"]["arc_object_history_filter"]["decision"] == "retire"
    assert (
        artifact["retirement_decisions"]["arc_live_levelup_standing_floor"]["decision"]
        == "do_not_retire_new_target_null"
    )
    assert artifact["arc_registry_delta"] == 0
    assert artifact["continuous_self_learning_verdict"]["claim_allowed"] is True
    assert artifact["hardware_sampling_verdict"]["crossover_claim_allowed"] is False
    assert artifact["documents_reconciled"]["protected_files"]["research-roadmap.yaml"] is True
    assert artifact["documents_reconciled"]["protected_files"]["scripts/research_conductor.py"] is True
    assert artifact["documents_reconciled"]["delegated_by_stop_rule"] == [
        "ops/status.md",
        "ops/changelog.md",
        "_bmad/traceability.md",
        "research-complete.yaml",
        "ops/exclusion_manifest.yaml",
        "ops/arc_solve_registry.yaml",
    ]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5612_missing_and_malformed_inputs_block_claims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5612-MISSING-MALFORMED: bad inputs fail closed."""

    missing = Path("results/experiment_5606_clean_sota_solve_verify_evidence_panel.json")
    malformed = Path("results/experiment_5608_kan_longitudinal_self_learning.json")
    _make_root(tmp_path, omit=missing, malformed=malformed)

    artifact = mod.run_capstone(
        root=tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["malformed_artifacts"] == [malformed.as_posix()]
    assert artifact["terminal_status_by_task"]["exp5606-clean-sota-solve-verify-evidence-panel"][
        "status"
    ] == "missing"
    assert artifact["terminal_status_by_task"]["exp5608-kan-longitudinal-self-learning"][
        "status"
    ] == "malformed"
    assert artifact["headline_claims"]["solve_verify_asymmetry"]["claim_allowed"] is False
    assert artifact["headline_claims"]["kan_longitudinal_self_learning"]["claim_allowed"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5612_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5612-FIELD-PRINCIPLES: schema drift is invalid."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert mod.validate_artifact(artifact) == []
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"honest_verdict": mod.FIELD_PRINCIPLES["honest_verdict"]}}
    )
    assert "expected_task_ids" in mod.validate_artifact({**artifact, "expected_task_ids": []})
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": 1})
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": "0"})
    assert "terminal_status_by_task" in mod.validate_artifact(
        {**artifact, "terminal_status_by_task": []}
    )
    assert "documents_reconciled" in mod.validate_artifact(
        {
            **artifact,
            "documents_reconciled": {
                **artifact["documents_reconciled"],
                "protected_files": {"research-roadmap.yaml": False},
            },
        }
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert "artifacts_found" in mod.validate_artifact({**artifact, "artifacts_found": "all"})
    bad_statuses = dict(artifact["terminal_status_by_task"])
    bad_statuses.pop("exp5603-transition-v506")
    assert "terminal_status_by_task" in mod.validate_artifact(
        {**artifact, "terminal_status_by_task": bad_statuses}
    )


def test_scenario_capstone_5612_defensive_helpers_cover_edge_cases(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5612-FIELD-PRINCIPLES: defensive paths stay explicit."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    payload, meta = mod._read_json_any(list_json)
    assert payload == {}
    assert meta["error"] == "not_json_object"

    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "-7"}, "value") == -7
    assert mod._int({"value": "bad"}, "value") == 0
    assert mod._float({"value": "1.25"}, "value") == 1.25
    assert mod._float({"value": "bad"}, "value") == 0.0
    assert mod._float({"value": None}, "value") == 0.0
    assert mod._status_for_payload({"honest_verdict": "unclear"}, {"exists": True, "loadable": True}) == "unknown"
    assert mod._all_gate_values_true({}) is False
    assert mod._all_gate_values_true("not-a-gate") is False

    assert mod._load_tests_run(None) == mod.DEFAULT_TESTS_RUN
    tests_run_path = tmp_path / "tests_run.json"
    tests_run_path.write_text(
        json.dumps([{"command": "unit", "exit_code": 0}, "ignored"]) + "\n",
        encoding="utf-8",
    )
    assert mod._load_tests_run(tests_run_path) == [{"command": "unit", "exit_code": 0}]
    bad_tests_run_path = tmp_path / "bad_tests_run.json"
    bad_tests_run_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_tests_run(bad_tests_run_path)


def test_scenario_capstone_5612_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5612: writer persists the validated deliverable."""

    _make_root(tmp_path)

    artifact = mod.write_capstone(
        root=tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert mod.validate_artifact(written) == []

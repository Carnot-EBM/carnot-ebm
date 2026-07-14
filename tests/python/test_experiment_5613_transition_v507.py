"""Tests for the Exp5613 V507 transition receipt.

Spec refs: REQ-REPORT-5613, SCENARIO-REPORT-5613,
SCENARIO-REPORT-5613-DEPENDENCY-MAP,
SCENARIO-REPORT-5613-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5613_transition_v507 as mod


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


def _write_context(root: Path, *, completed_text: str | None = None) -> None:
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
                        "# Research Roadmap vNEXT - Milestone 2026.07.507",
                        "**Task range:** Exp5613-Exp5624",
                        "native-runtime certification",
                        "exact drift fixture->duration map->predictive KAN",
                        "ARC transition-cycle prototype->live A/B->unconditional level attempt",
                        "cDLS exactness->gated crossover",
                    ]
                )
                + "\n",
            )
        elif rel_path == mod.RESEARCH_COMPLETE_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                completed_text
                or "\n".join(
                    [
                        "tasks:",
                        "  - id: exp5603-transition-v506",
                        "  - id: exp5612-v506-capstone-reconciliation",
                    ]
                )
                + "\n",
            )
        else:
            _write_text(root, rel_path)


def _payloads() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5603_transition_v506.json"): {
            "schema": "carnot.experiment_5603.transition_v506.v1",
            "experiment_id": "exp5603-transition-v506",
            "status": "complete",
            "current_milestone": mod.PREVIOUS_MILESTONE,
            "current_task_range": "exp5603-exp5612",
            "task_id_collision_avoidance": {
                "previous_outer_loop_last_id": "exp5602",
                "new_range_starts_at": "exp5603",
                "collision_avoided": True,
            },
            "honest_verdict": "complete: transition",
            "inference_substrate": "aggregation_from_repository_artifacts",
        },
        Path("results/experiment_5604_v506_source_delta_ingestion.json"): {
            "experiment_id": "exp5604-v506-source-delta-ingestion",
            "honest_verdict": "complete: source delta",
            "flagged_adversarial": True,
            "closed_scopes_reopened": False,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "new_references_added": [{"source_id": "lazy_identity_deq_2607_11116"}],
        },
        Path("results/experiment_5605_raw_response_evidence_envelope.json"): {
            "experiment_id": "exp5605-raw-response-evidence-envelope",
            "honest_verdict": "complete: envelope",
            "envelope_ready": True,
            "raw_payloads_preserved": True,
            "lossless_replay_rate": 1.0,
            "semantic_false_accept_count": 0,
            "parser_version_replay_passed": True,
            "payload_corruption_rejected": True,
            "truncation_controls_detected": 2,
            "gpu_offload_authenticated": True,
        },
        Path("results/experiment_5606_clean_sota_solve_verify_evidence_panel.json"): {
            "experiment_id": "exp5606-clean-sota-solve-verify-evidence-panel",
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
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "2 of 2 gate(s) failed",
        },
        Path("results/experiment_5608_kan_longitudinal_self_learning.json"): {
            "experiment_id": "experiment_5608_kan_longitudinal_self_learning",
            "task_id": "exp5608-kan-longitudinal-self-learning",
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
            "poison_update_disposition": {"disposition": "rolled_back", "persisted": False},
        },
        Path("results/experiment_5609_arc_filter_intermediate_invariance_ab.json"): {
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
                    "reason": "reachable control but no downstream live-path improvement",
                },
                "object_history": {
                    "decision": "retire_reachable_downstream_noop",
                    "reachable": True,
                    "downstream_improved": False,
                    "reason": "reachable control but no downstream live-path improvement",
                },
            },
        },
        Path("results/experiment_5610_arc_live_self_discovery_levelup_v506.json"): {
            "experiment_id": 5610,
            "experiment": "experiment_5610_arc_live_self_discovery_levelup_v506",
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
        },
        Path("results/experiment_5611_cdls_matched_sampler_crossover.json"): {
            "experiment_id": "exp5611-cdls-matched-sampler-crossover",
            "honest_verdict": "blocked: cDLS matched CPU/CUDA comparison unavailable",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "board_speedup_claimed": False,
            "crossover_claim_allowed": False,
            "crossover_size": None,
            "successful_matched_pairs": 0,
            "quality_equivalence_gate": {
                "defined_before_timing": True,
                "min_seeds_for_timing_interval": 3,
            },
            "seeds": [5611],
            "samples_per_pair": 10000,
        },
        Path("results/experiment_5612_v506_capstone_reconciliation.json"): {
            "experiment_id": "exp5612-v506-capstone-reconciliation",
            "honest_verdict": "complete: capstone",
            "flagged_tasks": [
                "exp5604-v506-source-delta-ingestion",
                "exp5610-arc-live-self-discovery-levelup-v506",
                "exp5611-cdls-matched-sampler-crossover",
            ],
            "blocked_tasks": ["exp5606-clean-sota-solve-verify-evidence-panel"],
            "gate_skipped_tasks": ["exp5607-property-template-exact-residual-extension"],
            "complete_tasks": [
                "exp5603-transition-v506",
                "exp5605-raw-response-evidence-envelope",
                "exp5608-kan-longitudinal-self-learning",
                "exp5609-arc-filter-intermediate-invariance-ab",
            ],
            "promotion_decisions": {
                "response_evidence_envelope": {"decision": "promote_bounded"},
                "kan_longitudinal_self_learning": {"decision": "promote_bounded"},
            },
            "retirement_decisions": {
                "local_sota_solve_verify_panel": {"decision": "retire"},
                "exact_predicate_extension_from_this_panel": {
                    "decision": "retire_until_clean_panel_exists"
                },
                "arc_inert_click_filter": {"decision": "retire"},
                "arc_object_history_filter": {"decision": "retire"},
            },
            "arc_registry_delta": 0,
            "hardware_sampling_verdict": {"crossover_claim_allowed": False},
        },
    }


def _make_root(
    root: Path,
    *,
    omit: Path | None = None,
    malformed: Path | None = None,
    completed_text: str | None = None,
) -> None:
    _write_context(root, completed_text=completed_text)
    for rel_path, payload in _payloads().items():
        if rel_path == omit:
            continue
        if rel_path == malformed:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json\n", encoding="utf-8")
            continue
        _write_json(root, rel_path, payload)


def _by_key(rows: list[JsonDict], field: str = "key") -> dict[str, JsonDict]:
    return {str(row[field]): row for row in rows}


def test_req_report_5613_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5613: OpenSpec anchors the V507 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5613") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.UPSTREAM_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5613_live_repo_locks_v506_evidence() -> None:
    """SCENARIO-REPORT-5613: live V506 facts become a bounded V507 map."""

    artifact = mod.build_report(
        root=REPO,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["status"] == "complete"
    assert artifact["previous_milestone"] == "2026.07.506"
    assert artifact["current_milestone"] == "2026.07.507"
    assert artifact["current_task_range"] == "exp5613-exp5624"
    assert artifact["current_task_collision_check"]["collision_free"] is True
    assert artifact["current_task_collision_check"]["colliding_ids"] == []
    statuses = artifact["terminal_findings"]
    assert statuses["exp5605-raw-response-evidence-envelope"]["status"] == "complete"
    assert statuses["exp5606-clean-sota-solve-verify-evidence-panel"]["status"] == "blocked"
    assert statuses["exp5607-property-template-exact-residual-extension"]["status"] == (
        "gate_skipped"
    )
    assert statuses["exp5610-arc-live-self-discovery-levelup-v506"]["status"] == "flagged"
    assert statuses["exp5611-cdls-matched-sampler-crossover"]["status"] == "flagged"

    promoted = _by_key(artifact["promoted_substrates"])
    assert promoted["lossless_response_envelope"]["source_artifacts"] == [
        "results/experiment_5605_raw_response_evidence_envelope.json"
    ]
    assert promoted["active_spline_kan"]["evidence"]["kan_weights_mutated"] is True
    retired = _by_key(artifact["retired_scopes"])
    assert retired["solve_versus_verify_panel"]["closed"] is True
    assert retired["exact_residual_extension_chain"]["closed"] is True
    assert retired["arc_inert_click_filter"]["closed"] is True
    assert retired["arc_object_history_filter"]["closed"] is True
    assert retired["unmatched_cdls_crossover"]["closed"] is True
    flags = _by_key(artifact["adversarial_flags_preserved"], "task_id")
    assert flags["exp5604-v506-source-delta-ingestion"]["upgraded_to_clean"] is False
    assert flags["exp5610-arc-live-self-discovery-levelup-v506"]["flag_kinds"] == ["TAUTOLOGY"]
    assert flags["exp5611-cdls-matched-sampler-crossover"]["flag_kinds"] == [
        "DURATION_TOO_SHORT",
        "METHODOLOGY_MISSING",
    ]

    assert artifact["dependency_map"]["native_runtime_certification"]["chain"] == (
        "native-runtime certification"
    )
    assert artifact["dependency_map"]["kan_drift_to_predictive_controller"]["chain"] == (
        "exact drift fixture->duration map->predictive KAN"
    )
    assert artifact["dependency_map"]["arc_transition_cycle_to_level_attempt"]["gates"][-1][
        "unconditional"
    ] is True
    assert artifact["dependency_map"]["cdls_exactness_to_crossover"]["chain"] == (
        "cDLS exactness->gated crossover"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5613_missing_malformed_or_colliding_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5613: missing, malformed, or colliding evidence fails closed."""

    missing = Path("results/experiment_5605_raw_response_evidence_envelope.json")
    malformed = Path("results/experiment_5608_kan_longitudinal_self_learning.json")
    _make_root(
        tmp_path,
        omit=missing,
        malformed=malformed,
        completed_text="tasks:\n  - id: exp5617-kan-critical-task-duration-map\n",
    )

    artifact = mod.build_report(
        root=tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["status"] == "blocked"
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["malformed_artifacts"] == [malformed.as_posix()]
    assert artifact["current_task_collision_check"]["collision_free"] is False
    assert artifact["current_task_collision_check"]["colliding_ids"] == [
        "exp5617-kan-critical-task-duration-map"
    ]
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5613_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5613-FIELD-PRINCIPLES: malformed fields fail validation."""

    _make_root(tmp_path)
    artifact = mod.build_report(
        root=tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert mod.validate_artifact(artifact) == []
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"honest_verdict": mod.FIELD_PRINCIPLES["honest_verdict"]}}
    )
    assert "artifacts_read" in mod.validate_artifact({**artifact, "artifacts_read": "all"})
    assert "terminal_findings" in mod.validate_artifact({**artifact, "terminal_findings": []})
    assert "promoted_substrates" in mod.validate_artifact({**artifact, "promoted_substrates": []})
    assert "retired_scopes" in mod.validate_artifact({**artifact, "retired_scopes": []})
    assert "adversarial_flags_preserved" in mod.validate_artifact(
        {**artifact, "adversarial_flags_preserved": []}
    )
    assert "current_task_range" in mod.validate_artifact(
        {**artifact, "current_task_range": "exp5614-exp5624"}
    )
    assert "current_task_collision_check" in mod.validate_artifact(
        {
            **artifact,
            "current_task_collision_check": {
                **artifact["current_task_collision_check"],
                "collision_free": False,
            },
        }
    )
    assert "dependency_map" in mod.validate_artifact({**artifact, "dependency_map": {}})
    assert "roadmap_yaml_unchanged" in mod.validate_artifact(
        {**artifact, "roadmap_yaml_unchanged": False}
    )
    assert "conductor_unchanged" in mod.validate_artifact(
        {**artifact, "conductor_unchanged": False}
    )
    assert "roadmap_yaml_unchanged" in mod.validate_artifact(
        {**artifact, "roadmap_yaml_unchanged": "true"}
    )
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "unknown"})
    bad_terminal = dict(artifact["terminal_findings"])
    bad_terminal.pop("exp5603-transition-v506")
    assert "terminal_findings" in mod.validate_artifact(
        {**artifact, "terminal_findings": bad_terminal}
    )


def test_scenario_report_5613_defensive_helpers_cover_edge_cases(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5613-FIELD-PRINCIPLES: helper boundaries stay explicit."""

    assert mod._status_for_payload({"status": "blocked"}, {"exists": True, "loadable": True}) == (
        "blocked"
    )
    assert mod._status_for_payload(
        {"schema": "blocked_gate_check_v1"}, {"exists": True, "loadable": True}
    ) == "gate_skipped"
    assert mod._status_for_payload(
        {"flagged_adversarial": True}, {"exists": True, "loadable": True}
    ) == "flagged"
    assert mod._status_for_payload(
        {"honest_verdict": "complete: x"}, {"exists": True, "loadable": True}
    ) == "complete"
    assert mod._status_for_payload({"honest_verdict": "unclear"}, {"exists": True, "loadable": True}) == (
        "unknown"
    )
    assert mod._status_for_payload({}, {"exists": False}) == "missing"
    assert mod._status_for_payload({}, {"exists": True, "loadable": False}) == "malformed"
    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "-7"}, "value") == -7
    assert mod._int({"value": "bad"}, "value") == 0
    assert mod._float({"value": "1.25"}, "value") == 1.25
    assert mod._float({"value": "bad"}, "value") == 0.0
    assert mod._float({"value": None}, "value") == 0.0
    assert mod._all_true({"a": True, "b": True}) is True
    assert mod._all_true({"a": True, "b": False}) is False
    assert mod._all_true({}) is False
    assert mod._extract_exp_number("exp5613-transition-v507") == 5613
    assert mod._extract_exp_number("no id") is None
    assert mod._task_range_from_text("exp5613-exp5624") == "exp5613-exp5624"
    assert mod._task_range_from_text("no range") is None
    assert mod._failed_preconditions(
        [],
        [],
        {"colliding_ids": [], "range_matches_roadmap": False},
        roadmap_modified=True,
        conductor_modified=True,
    ) == [
        "roadmap_task_range_mismatch",
        "research-roadmap.yaml_modified",
        "scripts/research_conductor.py_modified",
    ]

    missing_payload, missing_meta = mod._read_json_any(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._read_json_any(malformed)[1]["error"] == "malformed_json"
    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    assert mod._read_json_any(list_json)[1]["error"] == "not_json_object"


def test_scenario_report_5613_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5613: writer persists the tested transition receipt."""

    _make_root(tmp_path)

    artifact = mod.write_report(
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

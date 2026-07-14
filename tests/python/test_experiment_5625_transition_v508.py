"""Tests for the Exp5625 V508 transition receipt.

Spec refs: REQ-REPORT-5625, SCENARIO-REPORT-5625,
SCENARIO-REPORT-5625-DEPENDENCY-MAP,
SCENARIO-REPORT-5625-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5625_transition_v508 as mod


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


def _write_context(root: Path, *, completed_text: str | None = None, conductor_text: str = "") -> None:
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
                        "# Research Roadmap vNEXT - Milestone 2026.07.508",
                        "**Task range:** Exp5625-Exp5635",
                        "conformal qualification->KAN replication->independent audit",
                        "epistemic object prototype->advisory live A/B->unconditional level attempt",
                        "exact temperature exchange->gated quality trial",
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
                        "  - id: exp5613-transition-v507",
                        "  - id: exp5624-v507-capstone-reconciliation",
                    ]
                )
                + "\n",
            )
        elif rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            _write_text(root, rel_path, conductor_text or "no retired current-range entries\n")
        else:
            _write_text(root, rel_path)


def _payloads() -> dict[Path, JsonDict]:
    return {
        Path("results/experiment_5613_transition_v507.json"): {
            "experiment_id": "exp5613-transition-v507",
            "status": "complete",
            "honest_verdict": "complete: transition",
            "current_task_range": "exp5613-exp5624",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        Path("results/experiment_5614_v507_source_delta_ingestion.json"): {
            "experiment_id": "exp5614-v507-source-delta-ingestion",
            "honest_verdict": "complete: source no-op",
            "closed_scopes_reopened": False,
            "new_references_added": [],
            "planner_marker_found": True,
        },
        Path("results/experiment_5615_native_llamacpp_cuda_runtime_certificate.json"): {
            "experiment_id": "exp5615-native-llamacpp-cuda-runtime-certificate",
            "honest_verdict": "blocked_native_cuda_runtime_certificate_failed_terminal_retirement_evidence",
            "runtime_certificate_ready_score": 0.0,
            "models_certified_count": 0,
            "models_certified_denominator": 3,
            "cuda_build_capability": {"native_cuda_ready": True},
            "lossless_replay_rate": 1.0,
            "semantic_false_accept_count": 0,
            "no_task_accuracy_computed": True,
            "solve_verify_accuracy_inferred": False,
        },
        Path("results/experiment_5616_exact_nonstationary_constraint_stream.json"): {
            "experiment_id": "exp5616-exact-nonstationary-constraint-stream",
            "honest_verdict": "complete: exact stream",
            "fixture_ready_score": 1.0,
            "dataset_row_count": 17856,
            "stream_count": 1152,
            "exact_oracle_label_count": 107136,
            "oracle_label_error_count": 0,
            "llm_invoked": False,
            "policy_fit": False,
            "readiness_gates": {
                "count_validation_passed": True,
                "oracle_validation_passed": True,
                "replay_validation_passed": True,
                "schema_validation_passed": True,
                "split_validation_passed": True,
            },
        },
        Path("results/experiment_5617_kan_critical_task_duration_map.json"): {
            "experiment_id": "experiment_5617_kan_critical_task_duration_map",
            "task_id": "exp5617-kan-critical-task-duration-map",
            "honest_verdict": "complete: critical_task_duration_d16_estimated",
            "critical_task_duration": 16,
            "critical_duration_fit_r2": 0.0,
            "nondegenerate_switch_cases": [{"switch_duration": 1.375}, {"switch_duration": 32.0}],
            "lazy_identity_guard_passed": True,
            "llm_invoked": False,
            "llm_weight_training": False,
            "external_teacher_used": False,
            "unsafe_false_accept_count": {"total": 0},
        },
        Path("results/experiment_5618_predictive_window_kan_self_learning.json"): {
            "experiment_id": "experiment_5618_predictive_window_kan_self_learning",
            "task_id": "exp5618-predictive-window-kan-self-learning",
            "honest_verdict": "complete: predictive_window_active_spline_kan_self_learning_ready",
            "continuous_self_learning_ready": True,
            "delta_ale_vs_best_fixed": {"mean": 0.150694, "n": 5},
            "controller_gate_receipt": {
                "adaptive_ale_beats_best_fixed": True,
                "unsafe_false_accept_zero": True,
                "no_model_weight_mutation": True,
            },
            "unsafe_false_accept_count": {"total": 0},
            "no_model_weight_mutation": True,
            "kan_spline_state_mutated": True,
            "llm_invoked": False,
            "llm_weight_training": False,
            "rollback_positive_control": {"passed": True},
            "delayed_regression_passed": True,
            "poison_update_disposition": {"accepted": 0, "rejected": 1152},
        },
        Path("results/experiment_5619_arc_forward_inverse_transition_cycle.json"): {
            "experiment_id": 5619,
            "honest_verdict": "complete: transition_cycle_verifier_safe_over_abstaining_not_useful_terminal",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "solve_provenance": "development_proxy",
            "cycle_verifier_positive_control_rate": 0.009259,
            "valid_transition_accept_rate": 0.009259,
            "unsafe_transition_accept_count": 0,
            "corruption_reject_rate": 1.0,
            "abstention_rate": 0.75,
            "inverse_action_accuracy": 0.203704,
        },
        Path("results/experiment_5620_arc_cycle_guarded_live_update_ab.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 2 gate(s) failed",
            "gates_evaluated": [
                {
                    "artifact_field": "cycle_verifier_positive_control_rate",
                    "expected": 0.9,
                    "actual": 0.009259,
                    "passed": False,
                }
            ],
        },
        Path("results/experiment_5621_arc_live_self_discovery_levelup_v507.json"): {
            "experiment_id": 5621,
            "honest_verdict": "complete: no_new_arc_level_banked_bp35_L9_bounded_live_attempt_v507",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "solve_provenance": "live_agent_self_discovery",
            "live_attempt_executed": True,
            "levels_before": 177,
            "levels_after": 177,
            "new_reproducible_levels": [],
            "offline_reproduced": False,
            "registry_updated": False,
            "target_reached_live": False,
            "target_selection_receipt": {"selected_game": "bp35", "selected_level": "L9"},
        },
        Path("results/experiment_5622_cdls_exact_kernel_audit.json"): {
            "experiment_id": "exp5622-cdls-exact-kernel-audit",
            "honest_verdict": "complete: corrected cDLS exact kernel audit ready",
            "kernel_audit_ready_score": 1.0,
            "correction_applied": True,
            "broken_kernel_controls_rejected": True,
            "transition_row_sum_error_max": 4.440892098500626e-16,
            "detailed_balance_residual_max": 5.204170427930421e-18,
            "exact_distribution_tv_max": 5.1144852242224204e-14,
            "energy_histogram_tv_max": 4.875613801580414e-14,
            "quality_gate_specified_count": 4,
            "correction_spec": {"final_kernel": "corrected_cdls_projection_mh"},
        },
        Path("results/experiment_5623_cdls_multiseed_cpu_cuda_crossover.json"): {
            "experiment_id": "exp5623-cdls-multiseed-cpu-cuda-crossover",
            "honest_verdict": "complete: no quality-matched crossover pairs entered speedups",
            "crossover_claim_allowed": False,
            "board_speedup_claimed": False,
            "crossover_size": None,
            "successful_matched_pairs": [],
            "quality_gate_results_by_pair": [
                {"included_in_speedups": False, "exclusion_reason": "quality_gate_failed"}
            ],
            "timing_rows": [{"size": 128}],
            "timing_intervals_by_size": [],
            "speedup_by_pair": [],
            "seeds": [5623, 5624, 5625, 5626, 5627],
            "samples_per_pair": 10000,
            "upstream_gate_receipt": {"ready": True, "kernel_audit_ready_score": 1.0},
            "preconditions": {"blocked_reasons": [], "cpu_available": True, "cuda_available": True},
        },
        Path("results/experiment_5624_v507_capstone_reconciliation.json"): {
            "experiment_id": "exp5624-v507-capstone-reconciliation",
            "honest_verdict": "complete: capstone",
            "complete_tasks": [
                "exp5613-transition-v507",
                "exp5614-v507-source-delta-ingestion",
                "exp5616-exact-nonstationary-constraint-stream",
                "exp5617-kan-critical-task-duration-map",
                "exp5618-predictive-window-kan-self-learning",
                "exp5622-cdls-exact-kernel-audit",
                "exp5623-cdls-multiseed-cpu-cuda-crossover",
            ],
            "blocked_tasks": ["exp5615-native-llamacpp-cuda-runtime-certificate"],
            "gate_skipped_tasks": ["exp5620-arc-cycle-guarded-live-update-ab"],
            "flagged_tasks": [
                "exp5619-arc-forward-inverse-transition-cycle",
                "exp5621-arc-live-self-discovery-levelup-v507",
            ],
            "promoted_tasks": [
                "exp5616-exact-nonstationary-constraint-stream",
                "exp5617-kan-critical-task-duration-map",
                "exp5622-cdls-exact-kernel-audit",
            ],
            "retired_tasks": [
                "exp5615-native-llamacpp-cuda-runtime-certificate",
                "exp5623-cdls-multiseed-cpu-cuda-crossover",
            ],
            "promotion_decisions": {
                "exact_drift_fixture": {"decision": "promote_bounded"},
                "critical_duration_map": {"decision": "promote_bounded"},
                "cdls_exact_kernel": {"decision": "promote_bounded"},
                "predictive_window_kan_self_learning": {
                    "decision": "do_not_promote_preregistered_gate_failed"
                },
            },
            "retirement_decisions": {
                "native_three_model_runtime_certificate": {"decision": "retire_terminal_same_verdict"},
                "cdls_cpu_cuda_crossover": {
                    "decision": "close_current_timing_claim_until_quality_gate_fixed"
                },
            },
        },
    }


def _make_root(
    root: Path,
    *,
    omit: Path | None = None,
    malformed: Path | None = None,
    completed_text: str | None = None,
    conductor_text: str = "",
) -> None:
    _write_context(root, completed_text=completed_text, conductor_text=conductor_text)
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


def test_req_report_5625_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5625: OpenSpec anchors the V508 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5625") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.UPSTREAM_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5625_live_repo_locks_v507_evidence() -> None:
    """SCENARIO-REPORT-5625: live V507 facts become a bounded V508 map."""

    artifact = mod.build_report(
        root=REPO,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["status"] == "complete"
    assert artifact["previous_milestone"] == "2026.07.507"
    assert artifact["current_milestone"] == "2026.07.508"
    assert artifact["current_task_range"] == "exp5625-exp5635"
    assert artifact["current_task_collision_check"]["collision_free"] is True
    statuses = artifact["terminal_findings"]
    assert statuses["exp5615-native-llamacpp-cuda-runtime-certificate"]["status"] == "blocked"
    assert statuses["exp5620-arc-cycle-guarded-live-update-ab"]["status"] == "gate_skipped"
    assert statuses["exp5619-arc-forward-inverse-transition-cycle"]["status"] == "flagged"
    assert statuses["exp5621-arc-live-self-discovery-levelup-v507"]["status"] == "flagged"

    promoted = _by_key(artifact["promoted_substrates"])
    assert set(promoted) == {"exact_nonstationary_constraint_stream", "corrected_cdls_exact_kernel"}
    assert promoted["exact_nonstationary_constraint_stream"]["evidence"]["oracle_label_error_count"] == 0
    assert promoted["corrected_cdls_exact_kernel"]["evidence"]["correction_applied"] is True
    promising = _by_key(artifact["promising_unpromoted_substrates"])
    assert promising["predictive_window_active_spline_kan"]["promoted"] is False
    assert promising["predictive_window_active_spline_kan"]["blocking_gate"]["critical_duration_fit_r2"] == 0.0

    retired = _by_key(artifact["retired_scopes"])
    assert retired["native_runtime_certificate"]["closed"] is True
    assert retired["solve_versus_verify_chain"]["closed"] is True
    assert retired["arc_transition_cycle_proxy"]["closed"] is True
    assert retired["arc_cycle_guarded_live_branch"]["closed"] is True
    assert retired["cdls_timing_crossover"]["closed"] is True
    flags = _by_key(artifact["adversarial_flags_preserved"], "task_id")
    assert flags["exp5619-arc-forward-inverse-transition-cycle"]["upgraded_to_clean"] is False
    assert flags["exp5621-arc-live-self-discovery-levelup-v507"]["flag_kinds"] == ["TAUTOLOGY"]

    assert artifact["dependency_map"]["conformal_kan_qualification_to_audit"]["chain"] == (
        "conformal qualification->KAN replication->independent audit"
    )
    assert artifact["dependency_map"]["epistemic_object_arc_to_level_attempt"]["gates"][-1][
        "unconditional"
    ] is True
    assert artifact["dependency_map"]["temperature_exchange_cdls_to_quality"]["chain"] == (
        "exact temperature exchange->gated quality trial"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5625_missing_malformed_or_colliding_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5625: missing, malformed, or colliding evidence fails closed."""

    missing = Path("results/experiment_5616_exact_nonstationary_constraint_stream.json")
    malformed = Path("results/experiment_5622_cdls_exact_kernel_audit.json")
    _make_root(
        tmp_path,
        omit=missing,
        malformed=malformed,
        completed_text="tasks:\n  - id: exp5628-conformal-active-spline-kan-csl\n",
        conductor_text="| x | y | RETIRED | exp5633-temperature-exchange-cdls-exact-audit |\n",
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
    assert artifact["current_task_collision_check"]["completed_colliding_ids"] == [
        "exp5628-conformal-active-spline-kan-csl"
    ]
    assert artifact["current_task_collision_check"]["retired_colliding_ids"] == [
        "exp5633-temperature-exchange-cdls-exact-audit"
    ]
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5625_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5625-FIELD-PRINCIPLES: malformed fields fail validation."""

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
    assert "promising_unpromoted_substrates" in mod.validate_artifact(
        {**artifact, "promising_unpromoted_substrates": []}
    )
    assert "retired_scopes" in mod.validate_artifact({**artifact, "retired_scopes": []})
    assert "adversarial_flags_preserved" in mod.validate_artifact(
        {**artifact, "adversarial_flags_preserved": []}
    )
    assert "current_task_range" in mod.validate_artifact(
        {**artifact, "current_task_range": "exp5626-exp5635"}
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
    bad_terminal.pop("exp5613-transition-v507")
    assert "terminal_findings" in mod.validate_artifact(
        {**artifact, "terminal_findings": bad_terminal}
    )


def test_scenario_report_5625_defensive_helpers_cover_edge_cases(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5625-FIELD-PRINCIPLES: helper boundaries stay explicit."""

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
    assert mod._nested_total({"total": 3}) == 3
    assert mod._nested_total({"by_arm": {"a": 0, "b": 2}}) == 2
    assert mod._nested_total(True) == 1
    assert mod._nested_total(5) == 5
    assert mod._nested_total("bad") == 0
    assert mod._all_true({"a": True, "b": True}) is True
    assert mod._all_true({"a": True, "b": False}) is False
    assert mod._all_true({}) is False
    assert mod._extract_exp_number("exp5625-transition-v508") == 5625
    assert mod._extract_exp_number("no id") is None
    assert mod._task_range_from_text("Exp5625-Exp5635") == "exp5625-exp5635"
    assert mod._task_range_from_text("no range") is None
    assert mod._failed_preconditions(
        [],
        [],
        {
            "completed_colliding_ids": [],
            "retired_colliding_ids": [],
            "range_matches_roadmap": False,
        },
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


def test_scenario_report_5625_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5625: writer persists the tested transition receipt."""

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

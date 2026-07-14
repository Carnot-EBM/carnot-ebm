"""Tests for the Exp5636 V509 transition receipt.

Spec refs: REQ-REPORT-5636, SCENARIO-REPORT-5636,
SCENARIO-REPORT-5636-DEPENDENCY-MAP,
SCENARIO-REPORT-5636-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5636_transition_v509 as mod


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


def _write_context(
    root: Path,
    *,
    completed_text: str | None = None,
    conductor_text: str = "",
) -> None:
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
                        "# Research Roadmap vNEXT - Milestone 2026.07.509",
                        "**Task range:** Exp5636-Exp5647",
                        "FR-11 schema corrigendum->anytime-valid audit->shadow integration",
                        "executable ARC model->advisory known-level A/B->unconditional live attempt",
                        "two-axis invariant audit->quality->Rust parity",
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
                        "  - id: exp5625-transition-v508",
                        "  - id: exp5635-v508-capstone-reconciliation",
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
        Path("results/experiment_5625_transition_v508.json"): {
            "experiment_id": "exp5625-transition-v508",
            "status": "complete",
            "honest_verdict": "complete: transition",
            "current_task_range": "exp5625-exp5635",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "retired_scopes": [
                {"key": "native_runtime_certificate", "closed": True},
                {"key": "solve_versus_verify_chain", "closed": True},
                {"key": "cdls_timing_crossover", "closed": True},
            ],
        },
        Path("results/experiment_5626_v508_source_delta_ingestion.json"): {
            "experiment_id": "exp5626-v508-source-delta-ingestion",
            "status": "complete",
            "honest_verdict": "complete: source no-op",
            "closed_scopes_reopened": False,
            "new_references_added": [],
            "planner_marker_found": True,
        },
        Path("results/experiment_5627_online_conformal_kan_qualification.json"): {
            "experiment_id": "experiment_5627_online_conformal_kan_qualification",
            "task_id": "exp5627-online-conformal-kan-qualification",
            "honest_verdict": "complete: online_conformal_group_conditional_kan_qualification_ready",
            "conformal_qualification_ready_score": 1.0,
            "worst_group_coverage": {
                "group_conditional_online_conformal": {"coverage": 0.904762, "n": 168}
            },
            "marginal_coverage": {
                "group_conditional_online_conformal": {"heldout": {"coverage": 0.935484, "n": 4464}}
            },
            "exact_unsafe_accept_count": {"by_arm": {"group": 0}, "total": 0},
            "qualification_gate_receipt": {
                "exact_unsafe_accept_zero": True,
                "worst_group_coverage_at_least_0_90": True,
                "marginal_coverage_at_least_0_90": True,
            },
            "leakage_control_pass": True,
            "llm_invoked": False,
            "llm_weight_training": False,
        },
        Path("results/experiment_5628_conformal_active_spline_kan_csl.json"): {
            "experiment_id": "experiment_5628_conformal_active_spline_kan_csl",
            "task_id": "exp5628-conformal-active-spline-kan-csl",
            "honest_verdict": "complete: conformal_active_spline_kan_continuous_self_learning_ready",
            "continuous_self_learning_ready": True,
            "unsafe_false_accept_count": {
                "by_arm": {"predictive_window_controller": 0},
                "total": 0,
            },
            "ale_by_arm": {
                "full_conformal_kan_controller": {"mean": 0.134028},
                "best_fixed_nonoracle": {"mean": 0.289583},
            },
            "ale_paired_intervals": {"reset_adapt": {"lower": 0.087842}},
            "poison_rejection_rate": {"accepted": 0, "rejected": 1152, "injected": 1152},
            "checkpoint_replay_exact": {"passed": True, "receipt_count": 900},
            "delayed_regression_recovery": {"passed": True},
            "llm_weight_updates": 0,
            "readiness_gate_receipt": {
                "unsafe_false_accept_zero": True,
                "full_beats_every_fixed_nonoracle_with_intervals": True,
                "checkpoint_replay_exact": True,
                "poison_rejection_complete": True,
            },
        },
        Path("results/experiment_5629_conformal_kan_independent_audit.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 2 gate(s) failed; first failure: exp5628.unsafe_false_accept_count",
            "gates_evaluated": [
                {"artifact_field": "continuous_self_learning_ready", "passed": True},
                {"artifact_field": "unsafe_false_accept_count", "passed": False},
            ],
        },
        Path("results/experiment_5630_arc_epistemic_object_probe_prototype.json"): {
            "experiment_id": 5630,
            "honest_verdict": "blocked: epistemic_object_probe_degenerate_or_unreachable_terminal",
            "epistemic_probe_ready_score": 0.0,
            "object_hypothesis_non_degenerate_count": 3,
            "unsafe_model_accept_count": 0,
            "informative_control_delta": -0.24154208,
            "solve_provenance": "development_proxy",
            "exhaustive_bfs_used": False,
            "outer_loop_recipes_used": False,
            "per_game_adapter_used": False,
        },
        Path("results/experiment_5631_arc_epistemic_probe_live_ab.json"): {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 3 gate(s) failed; first failure: exp5630.epistemic_probe_ready_score",
            "gates_evaluated": [
                {"artifact_field": "epistemic_probe_ready_score", "passed": False},
                {"artifact_field": "object_hypothesis_non_degenerate_count", "passed": True},
                {"artifact_field": "unsafe_model_accept_count", "passed": True},
            ],
        },
        Path("results/experiment_5632_arc_live_self_discovery_levelup_v508.json"): {
            "experiment_id": 5632,
            "honest_verdict": "complete: no_new_arc_level_banked_lf52_L7_bounded_live_attempt_v508",
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "IMPLAUSIBLE_PERFECT", "severity": "info"},
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
            "live_attempt_executed": True,
            "registry_count_before": 177,
            "registry_count_after": 177,
            "registry_delta": 0,
            "new_reproducible_levels": [],
            "offline_reproduced": False,
            "registry_updated": False,
            "selected_game": "lf52",
            "selected_level": "L7",
            "llm_invoked": False,
            "solve_provenance": "live_agent_self_discovery",
        },
        Path("results/experiment_5633_temperature_exchange_cdls_exact_audit.json"): {
            "experiment_id": "exp5633-temperature-exchange-cdls-exact-audit",
            "honest_verdict": "complete: exact temperature-label exchange cDLS audit ready",
            "replica_exchange_kernel_ready_score": 1.0,
            "exact_distribution_tv_max": 3.1e-16,
            "swap_detailed_balance_residual_max": 4.4e-19,
            "transition_normalization_error_max": 2.3e-16,
            "cold_replica_energy_error": 1.2e-16,
            "round_trip_accounting_error": 0.0,
            "validity_regression_detected": False,
            "hardware_speedup_claimed": False,
            "timing_claimed": False,
            "broken_controls": [{"detected": True}],
        },
        Path("results/experiment_5634_temperature_exchange_cdls_quality.json"): {
            "experiment_id": "exp5634-temperature-exchange-cdls-quality",
            "honest_verdict": "complete: quality_mixing_ready true under paired exact corrected cDLS quality gate",
            "quality_mixing_ready": True,
            "target_diagnostics_within_exp5633_bounds": True,
            "hardware_speedup_claimed": False,
            "timing_claimed": False,
            "wall_time_provenance_only": {
                "speedup_claim_allowed": False,
                "speedup_computed": False,
            },
            "paired_deltas_and_intervals": {
                "temperature_exchange_cdls_vs_independent_corrected_cdls_replicas": {
                    "barrier_crossings_delta_interval_95": [61.82, 77.98],
                    "ess_delta_interval_95": [21.79, 54.2],
                    "exact_valid_rate_delta_interval_95": [-0.059, 0.076],
                }
            },
        },
        Path("results/experiment_5635_v508_capstone_reconciliation.json"): {
            "experiment_id": "exp5635-v508-capstone-reconciliation",
            "honest_verdict": "complete: v508 capstone reconciled",
            "complete_tasks": [
                "exp5625-transition-v508",
                "exp5626-v508-source-delta-ingestion",
                "exp5627-online-conformal-kan-qualification",
                "exp5628-conformal-active-spline-kan-csl",
                "exp5633-temperature-exchange-cdls-exact-audit",
                "exp5634-temperature-exchange-cdls-quality",
            ],
            "blocked_tasks": ["exp5630-arc-epistemic-object-probe-prototype"],
            "gate_skipped_tasks": [
                "exp5629-conformal-kan-independent-audit",
                "exp5631-arc-epistemic-probe-live-ab",
            ],
            "flagged_tasks": ["exp5632-arc-live-self-discovery-levelup-v508"],
            "promotion_ledger": {
                "fr11_conformal_kan": {
                    "promoted": False,
                    "failed_condition": "exp5629_independent_audit_not_executed",
                },
                "replica_exchange_exact": {"promoted": True},
                "replica_exchange_quality": {"promoted": True},
            },
            "continuous_self_learning_promotion": {
                "internal_ready": True,
                "independent_certified": False,
                "promoted": False,
            },
            "replica_exchange_exact": {
                "promoted": True,
                "replica_exchange_kernel_ready_score": 1.0,
                "hardware_speedup_claimed": False,
                "timing_claimed": False,
            },
            "replica_exchange_quality_evidence": {
                "promoted": True,
                "quality_mixing_ready": True,
                "hardware_speedup_claimed": False,
                "timing_claimed": False,
            },
            "hardware_speedup_claimed": False,
            "timing_claimed": False,
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


def test_req_report_5636_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5636: OpenSpec anchors the V509 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5636") : spec.index("REQ-REPORT-5626")]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.UPSTREAM_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5636_live_repo_locks_v508_evidence() -> None:
    """SCENARIO-REPORT-5636: live V508 facts become a bounded V509 map."""

    artifact = mod.build_report(
        root=REPO,
        tests_run=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["status"] == "complete"
    assert artifact["previous_milestone"] == "2026.07.508"
    assert artifact["current_milestone"] == "2026.07.509"
    assert artifact["current_task_range"] == "exp5636-exp5647"
    assert artifact["current_task_collision_check"]["collision_free"] is True
    statuses = artifact["terminal_findings"]
    assert statuses["exp5629-conformal-kan-independent-audit"]["status"] == "gate_skipped"
    assert statuses["exp5630-arc-epistemic-object-probe-prototype"]["status"] == "blocked"
    assert statuses["exp5631-arc-epistemic-probe-live-ab"]["status"] == "gate_skipped"
    assert statuses["exp5632-arc-live-self-discovery-levelup-v508"]["status"] == "flagged"

    promoted = _by_key(artifact["promoted_substrates"])
    assert set(promoted) == {
        "one_axis_temperature_exchange_exact",
        "one_axis_temperature_exchange_quality",
    }
    assert promoted["one_axis_temperature_exchange_exact"]["claim_boundary"].endswith(
        "no timing, CPU/CUDA crossover, board, SNN, TSU, or hardware-speedup claim"
    )
    assert (
        promoted["one_axis_temperature_exchange_quality"]["evidence"]["quality_mixing_ready"]
        is True
    )

    promising = _by_key(artifact["promising_unpromoted_substrates"])
    fr11 = promising["fr11_conformal_active_spline_kan_internal"]
    assert fr11["promoted"] is False
    assert fr11["independent_promotion"] is False
    assert fr11["blocking_gate"]["schema_gate_failure_not_scientific_negative"] is True
    assert fr11["evidence"]["continuous_self_learning_ready"] is True

    retired = _by_key(artifact["retired_scopes"])
    assert retired["arc_epistemic_object_probe"]["closed"] is True
    assert retired["fr11_independent_promotion_claim"]["scientific_negative"] is False
    assert retired["cdls_timing_crossover_and_hardware_speedup"]["closed"] is True
    assert retired["board_snn_tsu_claims"]["closed"] is True

    flags = _by_key(artifact["adversarial_flags_preserved"], "task_id")
    assert flags["exp5632-arc-live-self-discovery-levelup-v508"]["upgraded_to_clean"] is False
    assert (
        "DURATION_TOO_SHORT" in flags["exp5632-arc-live-self-discovery-levelup-v508"]["flag_kinds"]
    )

    assert (
        artifact["dependency_map"]["fr11_schema_corrigendum_to_shadow_integration"]["chain"]
        == "FR-11 schema corrigendum->anytime-valid audit->shadow integration"
    )
    assert (
        artifact["dependency_map"]["executable_arc_model_to_live_attempt"]["gates"][-1][
            "unconditional"
        ]
        is True
    )
    assert artifact["dependency_map"]["two_axis_tempering_to_rust_parity"]["tasks"] == [
        "exp5644-two-axis-parallel-tempering-exact-audit",
        "exp5645-two-axis-tempering-hard-constraint-quality",
        "exp5646-two-axis-tempering-rust-parity",
    ]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5636_missing_malformed_or_colliding_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5636: missing, malformed, or colliding evidence fails closed."""

    missing = Path("results/experiment_5628_conformal_active_spline_kan_csl.json")
    malformed = Path("results/experiment_5633_temperature_exchange_cdls_exact_audit.json")
    _make_root(
        tmp_path,
        omit=missing,
        malformed=malformed,
        completed_text="tasks:\n  - id: exp5639-anytime-valid-csl-independent-audit\n",
        conductor_text="| x | y | RETIRED | exp5644-two-axis-parallel-tempering-exact-audit |\n",
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
        "exp5639-anytime-valid-csl-independent-audit"
    ]
    assert artifact["current_task_collision_check"]["retired_colliding_ids"] == [
        "exp5644-two-axis-parallel-tempering-exact-audit"
    ]
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_5636_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5636-FIELD-PRINCIPLES: malformed fields fail validation."""

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
        {**artifact, "current_task_range": "exp5637-exp5647"}
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
    bad_terminal.pop("exp5625-transition-v508")
    assert "terminal_findings" in mod.validate_artifact(
        {**artifact, "terminal_findings": bad_terminal}
    )
    assert mod._headline_group_coverage({}) is None
    assert (
        mod._headline_group_coverage({"worst_group_coverage": {"other": {"coverage": 0.9}}}) is None
    )
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


def test_scenario_report_5636_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5636: writer persists the tested transition receipt."""

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

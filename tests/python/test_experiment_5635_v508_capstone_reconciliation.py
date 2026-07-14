"""Tests for the Exp5635 V508 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5635, SCENARIO-CAPSTONE-5635,
SCENARIO-CAPSTONE-5635-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5635-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5635_v508_capstone_reconciliation as mod


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


def _payloads() -> dict[Path, JsonDict]:
    return {
        mod.EXP5625_TRANSITION_PATH: {
            "experiment_id": "exp5625-transition-v508",
            "honest_verdict": "complete: archived .507 terminal evidence into .508 dependency map",
            "promoted_substrates": [{"key": "exact_nonstationary_constraint_stream"}],
            "retired_scopes": [{"key": "native_runtime_certificate_closed"}],
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        mod.EXP5626_SOURCE_PATH: {
            "experiment_id": "exp5626-v508-source-delta-ingestion",
            "honest_verdict": "complete: no new non-duplicate actionable V508 source deltas",
            "new_references_added": [],
            "closed_scopes_reopened": False,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        mod.EXP5627_CONFORMAL_PATH: {
            "experiment_id": "experiment_5627_online_conformal_kan_qualification",
            "honest_verdict": "complete: online_conformal_group_conditional_kan_qualification_ready",
            "conformal_qualification_ready_score": 1.0,
            "marginal_coverage": {
                "group_conditional_online_conformal": {"heldout": {"coverage": 0.935484}}
            },
            "worst_group_coverage": {"group_conditional_online_conformal": {"coverage": 0.904762}},
            "exact_unsafe_accept_count": {"total": 0},
            "leakage_control_pass": True,
            "inference_substrate": "online_conformal_calibration_over_exact_labels",
        },
        mod.EXP5628_CSL_PATH: {
            "experiment_id": "experiment_5628_conformal_active_spline_kan_csl",
            "honest_verdict": "complete: conformal_active_spline_kan_continuous_self_learning_ready",
            "continuous_self_learning_ready": True,
            "readiness_gate_receipt": {
                "full_beats_every_fixed_nonoracle_with_intervals": True,
                "unsafe_false_accept_zero": True,
                "llm_weight_updates_zero": True,
            },
            "unsafe_false_accept_count": {"total": 0},
            "llm_weight_updates": 0,
            "checkpoint_replay_exact": {"passed": True},
            "inference_substrate": (
                "active_spline_kan_with_exact_validation_and_online_conformal_control"
            ),
        },
        mod.EXP5629_AUDIT_PATH: {
            "experiment": 5629,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 2 gate(s) failed; first failure: "
                "exp5628-conformal-active-spline-kan-csl.unsafe_false_accept_count"
            ),
            "gates_evaluated": [
                {"artifact_field": "continuous_self_learning_ready", "passed": True},
                {
                    "artifact_field": "unsafe_false_accept_count",
                    "expected": 0,
                    "actual": {"total": 0},
                    "passed": False,
                },
            ],
        },
        mod.EXP5630_ARC_PROBE_PATH: {
            "experiment_id": 5630,
            "honest_verdict": "blocked: epistemic_object_probe_degenerate_or_unreachable_terminal",
            "solve_provenance": "development_proxy",
            "object_hypothesis_non_degenerate_count": 3,
            "unsafe_model_accept_count": 0,
            "epistemic_probe_ready_score": 0.0,
            "informative_control_delta": -0.24,
            "live_interface_replay_rate": 1.0,
            "per_game_adapter_used": False,
            "outer_loop_recipes_used": False,
            "exhaustive_bfs_used": False,
        },
        mod.EXP5631_ARC_AB_PATH: {
            "experiment": 5631,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 3 gate(s) failed; first failure: "
                "exp5630-arc-epistemic-object-probe-prototype.epistemic_probe_ready_score"
            ),
            "gates_evaluated": [
                {"artifact_field": "epistemic_probe_ready_score", "passed": False},
                {"artifact_field": "object_hypothesis_non_degenerate_count", "passed": True},
                {"artifact_field": "unsafe_model_accept_count", "passed": True},
            ],
        },
        mod.EXP5632_ARC_LEVEL_PATH: {
            "experiment_id": 5632,
            "honest_verdict": "complete: no_new_arc_level_banked_lf52_L7_bounded_live_attempt_v508",
            "solve_provenance": "live_agent_self_discovery",
            "live_attempt_executed": True,
            "offline_reproduced": False,
            "registry_count_before": 177,
            "registry_count_after": 177,
            "registry_delta": 0,
            "new_reproducible_levels": [],
            "reproduced_levels": 0,
            "registry_updated": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
            "llm_invoked": False,
            "source_read": False,
            "game_adapter_used": False,
            "outer_loop_re_used": False,
        },
        mod.EXP5633_EXACT_PATH: {
            "experiment_id": "exp5633-temperature-exchange-cdls-exact-audit",
            "honest_verdict": "complete: exact temperature-label exchange cDLS audit ready",
            "replica_exchange_kernel_ready_score": 1.0,
            "exact_distribution_tv_max": 3e-16,
            "swap_detailed_balance_residual_max": 4e-19,
            "transition_normalization_error_max": 2e-16,
            "round_trip_accounting_error": 0.0,
            "validity_regression_detected": False,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
            "deterministic_replay_pass": True,
            "broken_controls": [{"control_id": "wrong_energy_sign", "detected": True}],
        },
        mod.EXP5634_QUALITY_PATH: {
            "experiment_id": "exp5634-temperature-exchange-cdls-quality",
            "honest_verdict": "complete: quality_mixing_ready true under paired exact corrected cDLS quality gate",
            "quality_mixing_ready": True,
            "target_diagnostics_within_exp5633_bounds": True,
            "hardware_speedup_claimed": False,
            "timing_claimed": False,
            "wall_time_provenance_only": {"speedup_claim_allowed": False},
            "upstream_gate_receipts": {
                "exp5633": {"ready": True, "replica_exchange_kernel_ready_score": 1.0}
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
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
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


def _artifact_map(payloads: dict[Path, JsonDict] | None = None) -> dict[str, JsonDict]:
    return {path.as_posix(): payload for path, payload in (payloads or _payloads()).items()}


def test_req_capstone_5635_spec_declares_v508_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5635: OpenSpec declares the V508 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5635") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for rel_path in mod.PRIMARY_ARTIFACT_PATHS:
        assert rel_path.as_posix() in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5635_live_repo_keeps_promotion_boundaries() -> None:
    """SCENARIO-CAPSTONE-5635: live V508 evidence promotes only supported claims."""

    artifact = mod.run_capstone(
        root=REPO,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["artifacts_expected"]) == 10
    assert len(artifact["artifacts_read"]) == 10
    assert (
        artifact["gate_outcomes"]["exp5629-conformal-kan-independent-audit"]["status"]
        == "gate_skipped"
    )
    assert (
        artifact["gate_outcomes"]["exp5631-arc-epistemic-probe-live-ab"]["status"] == "gate_skipped"
    )

    csl = artifact["continuous_self_learning_promotion"]
    assert csl["internal_ready"] is True
    assert csl["independent_certified"] is False
    assert csl["promoted"] is False
    assert csl["failed_condition"] == "exp5629_independent_audit_not_executed"

    assert artifact["promotion_ledger"]["fr11_conformal_kan"]["promoted"] is False
    assert artifact["promotion_ledger"]["arc_epistemic_mechanism"]["promoted"] is False
    assert artifact["promotion_ledger"]["arc_live_solve_credit"]["promoted"] is False
    assert artifact["promotion_ledger"]["replica_exchange_exact"]["promoted"] is True
    assert artifact["promotion_ledger"]["replica_exchange_quality"]["promoted"] is True
    assert artifact["replica_exchange_exact"]["promoted"] is True
    assert artifact["replica_exchange_quality_promoted"] is True

    assert artifact["arc_mechanism_promotion"]["exp5630_development_proxy"] is True
    assert artifact["arc_mechanism_promotion"]["promoted"] is False
    assert artifact["arc_registry_count_before"] == 177
    assert artifact["arc_registry_count_after"] == 177
    assert artifact["arc_registry_delta"] == 0

    critical_flags = [
        row for row in artifact["adversarial_flags"] if row["max_severity"] == "critical"
    ]
    assert any(
        row["task_id"] == "exp5632-arc-live-self-discovery-levelup-v508" for row in critical_flags
    )
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["timing_claimed"] is False
    assert artifact["research_roadmap_unchanged"] is True
    assert artifact["research_conductor_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert "ops/status.md" in artifact["documents_reconciled"]["delegated_by_stop_rule"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5635_missing_and_malformed_inputs_block_promotions(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5635-MISSING-MALFORMED: bad inputs fail closed."""

    missing = mod.EXP5634_QUALITY_PATH
    malformed = mod.EXP5633_EXACT_PATH
    _make_root(tmp_path, omit=missing, malformed=malformed)

    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["missing_artifacts"] == [missing.as_posix()]
    assert artifact["malformed_artifacts"] == [malformed.as_posix()]
    assert artifact["replica_exchange_exact"]["promoted"] is False
    assert artifact["replica_exchange_quality_promoted"] is False
    assert artifact["promotion_ledger"]["replica_exchange_exact"]["promoted"] is False
    assert artifact["promotion_ledger"]["replica_exchange_quality"]["promoted"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5635_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5635-FIELD-PRINCIPLES: schema drift is invalid."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert mod.validate_artifact(artifact) == []
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": {"honest_verdict": mod.FIELD_PRINCIPLES["honest_verdict"]}}
    )
    assert "artifacts_expected" in mod.validate_artifact({**artifact, "artifacts_expected": []})
    assert "artifacts_read" in mod.validate_artifact({**artifact, "artifacts_read": "all"})
    assert "arc_registry_delta" in mod.validate_artifact({**artifact, "arc_registry_delta": 2})
    assert "hardware_speedup_claimed" in mod.validate_artifact(
        {**artifact, "hardware_speedup_claimed": True}
    )
    assert "timing_claimed" in mod.validate_artifact({**artifact, "timing_claimed": True})
    assert "honest_verdict" in mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    assert "inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "schema" in mod.validate_artifact({k: v for k, v in artifact.items() if k != "schema"})
    bad_gates = dict(artifact["gate_outcomes"])
    bad_gates.pop("exp5629-conformal-kan-independent-audit")
    assert "gate_outcomes" in mod.validate_artifact({**artifact, "gate_outcomes": bad_gates})


def test_scenario_capstone_5635_defensive_helpers_cover_edge_cases(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5635-FIELD-PRINCIPLES: helper edge cases stay explicit."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    payload, meta = mod._read_json_any(list_json)
    assert payload == {}
    assert meta["error"] == "not_json_object"

    assert mod._severity_rank("critical") > mod._severity_rank("warn")
    assert mod._max_severity([{"severity": "warn"}, {"severity": "critical"}]) == "critical"
    assert mod._max_severity(["bad", {"severity": "warn"}]) == "warn"
    assert mod._max_severity("bad") == "none"
    assert mod._zeroish({"total": 0}) is True
    assert mod._zeroish({"total": 1}) is False
    assert mod._zeroish({"a": 0, "b": {"total": 0}}) is True
    assert mod._zeroish([0, {"total": 0}]) is True
    assert mod._zeroish(True) is False
    assert mod._zeroish("0") is False
    assert mod._number({"value": "1.5"}, "value") == 1.5
    assert mod._number({"value": "bad"}, "value") == 0.0
    assert mod._int({"value": True}, "value") == 1
    assert mod._int({"value": "-2"}, "value") == -2
    assert mod._int({"value": "bad"}, "value") == 0
    assert mod._is_gate_skip({"schema": "blocked_gate_check_v1"}) is True
    assert (
        mod._status_for_payload(
            {"honest_verdict": "blocked: x"}, {"exists": True, "loadable": True}
        )
        == "blocked"
    )
    assert (
        mod._status_for_payload({"honest_verdict": "unclear"}, {"exists": True, "loadable": True})
        == "unknown"
    )
    assert mod._headline_coverage({}, "missing") == 0.0
    assert (
        mod._headline_coverage(
            {"marginal_coverage": {"group_conditional_online_conformal": "bad"}},
            "marginal_coverage",
        )
        == 0.0
    )
    assert mod._load_validation_results(None) == mod.DEFAULT_VALIDATION_RESULTS
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps([{"command": "unit", "exit_code": 0}, "ignored"]) + "\n",
        encoding="utf-8",
    )
    assert mod._load_validation_results(validation_path) == [{"command": "unit", "exit_code": 0}]
    bad_validation_path = tmp_path / "bad_validation.json"
    bad_validation_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_validation_results(bad_validation_path)


def test_scenario_capstone_5635_branch_decisions_cover_all_gate_reasons() -> None:
    """SCENARIO-CAPSTONE-5635: narrow failed conditions are deterministic."""

    payloads = _payloads()
    payloads[mod.EXP5627_CONFORMAL_PATH] = {
        **payloads[mod.EXP5627_CONFORMAL_PATH],
        "conformal_qualification_ready_score": 0.0,
    }
    assert (
        mod._derive_csl(_artifact_map(payloads))["failed_condition"]
        == "exp5627_conformal_qualification_not_ready"
    )

    payloads = _payloads()
    payloads[mod.EXP5628_CSL_PATH] = {
        **payloads[mod.EXP5628_CSL_PATH],
        "continuous_self_learning_ready": False,
    }
    assert (
        mod._derive_csl(_artifact_map(payloads))["failed_condition"]
        == "exp5628_internal_csl_not_ready"
    )

    payloads = _payloads()
    payloads[mod.EXP5629_AUDIT_PATH] = {
        "honest_verdict": "complete: independent audit found a critical flag",
        "independent_promotion_ready": False,
        "critical_flags": [{"severity": "critical"}],
    }
    assert (
        mod._derive_csl(_artifact_map(payloads))["failed_condition"]
        == "exp5629_independent_audit_critical_flag"
    )

    payloads = _payloads()
    payloads[mod.EXP5629_AUDIT_PATH] = {
        "honest_verdict": "complete: independent audit did not promote",
        "independent_promotion_ready": False,
        "critical_flags": [],
    }
    assert (
        mod._derive_csl(_artifact_map(payloads))["failed_condition"]
        == "exp5629_independent_promotion_ready_false"
    )

    payloads = _payloads()
    payloads[mod.EXP5629_AUDIT_PATH] = {
        "honest_verdict": "complete: independent audit promoted",
        "independent_promotion_ready": True,
        "critical_flags": [],
    }
    csl = mod._derive_csl(_artifact_map(payloads))
    assert csl["failed_condition"] is None
    assert csl["promoted"] is True

    payloads = _payloads()
    payloads[mod.EXP5632_ARC_LEVEL_PATH] = {
        **payloads[mod.EXP5632_ARC_LEVEL_PATH],
        "registry_delta": 2,
    }
    _, arc_solve = mod._derive_arc(_artifact_map(payloads))
    assert arc_solve["registry_delta"] == 0


def test_scenario_capstone_5635_writer_emits_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5635: writer persists the validated deliverable."""

    _make_root(tmp_path)

    artifact = mod.write_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert mod.validate_artifact(written) == []


def test_scenario_capstone_5635_validation_and_cli_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-5635-FIELD-PRINCIPLES: validation failures are explicit."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(
        root=tmp_path,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    bad_principles = dict(artifact["field_principles"])
    bad_principles["honest_verdict"] = "wrong"
    assert "field_principles" in mod.validate_artifact(
        {**artifact, "field_principles": bad_principles}
    )
    assert "promotion_ledger" in mod.validate_artifact({**artifact, "promotion_ledger": []})
    assert "terminal_status_by_task" in mod.validate_artifact(
        {**artifact, "terminal_status_by_task": {"only": {}}}
    )
    assert "arc_registry_count_before" in mod.validate_artifact(
        {**artifact, "arc_registry_count_before": "177"}
    )
    assert "arc_registry_count_after" in mod.validate_artifact(
        {**artifact, "arc_registry_count_after": "177"}
    )
    assert "research_roadmap_unchanged" in mod.validate_artifact(
        {**artifact, "research_roadmap_unchanged": "yes"}
    )
    assert "reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )

    monkeypatch.setattr(mod, "run_capstone", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError):
        mod.write_capstone(root=tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["schema"])
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])


def test_scenario_capstone_5635_cli_writes_requested_output(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5635: CLI emits the validated output path."""

    _make_root(tmp_path)
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps([{"command": "unit", "exit_code": 0, "status": "passed"}]) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "custom" / "capstone.json"

    assert (
        mod.main(
            [
                "--root",
                str(tmp_path),
                "--output",
                str(output_path),
                "--validation-results",
                str(validation_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID

"""Tests for the Exp5717 V511 transition receipt.

Spec refs: REQ-CAPSTONE-5717, SCENARIO-CAPSTONE-5717,
SCENARIO-CAPSTONE-5717-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5717-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5717_transition_v511 as mod


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
        mod.EXP5706_TRANSITION_PATH: {
            "schema": "carnot.experiment_5706.transition_v510.v1",
            "experiment_id": "exp5706-transition-v510",
            "status": "complete",
            "honest_verdict": "complete: v510 transition archived .509 evidence",
            "fr11_promoted": True,
            "fr11_shadow_default_enabled": False,
            "arc_registry_count": 177,
            "arc_registry_delta": 0,
            "one_axis_replica_exchange_promoted": True,
            "two_axis_quality_promoted": False,
            "retirements_applied": [
                {"scope": "arc_counterexample_patched_transition_model_exp5641"},
                {"scope": "two_axis_beta_lambda_tempering_extension_exp5645"},
            ],
        },
        mod.EXP5707_SOURCE_PATH: {
            "schema": "carnot.experiment_5707.v510_source_delta_ingestion.v1",
            "experiment_id": "exp5707-v510-source-delta-ingestion",
            "status": "complete",
            "honest_verdict": "complete: no new non-duplicate actionable V510 source deltas",
        },
        mod.EXP5708_CANARY_PATH: {
            "schema": "carnot.experiment_5708.sota_exact_constraint_canary.v1",
            "experiment_id": "experiment_5708_sota_exact_constraint_canary",
            "status": "blocked",
            "honest_verdict": "blocked: parse_failures",
            "blocked_reasons": ["parse_failures"],
            "cuda_offload_authenticated": True,
            "cuda_offload_authenticated_score": 1.0,
            "manifest_row_count": 50,
            "parse_failure_count": 47,
            "missing_row_count": 0,
            "validator_disagreement_count": 0,
            "stream_root_commitment": "sha256:stream",
            "shadow_prefix_hash": "sha256:prefix",
            "sealed_suffix_hash": "sha256:suffix",
            "row_manifest_path": mod.EXP5708_ROWS_PATH.as_posix(),
            "native_json_grammar_used": False,
            "retired_runtime_used": False,
            "external_scorer_used": False,
        },
        mod.EXP5709_SHADOW_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5709,
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 3 gate(s) failed; first failure: exp5708",
            "gates_evaluated": [
                {
                    "upstream": "exp5708-sota-exact-constraint-canary",
                    "artifact_field": "sota_canary_ready_score",
                    "actual": 0.0,
                    "expected": 1.0,
                    "passed": False,
                }
            ],
        },
        mod.EXP5711_ARC_QUAL_PATH: {
            "schema": "carnot.experiment_5711.arc_relational_goal_energy_live_qualification.v1",
            "experiment": 5711,
            "status": "complete",
            "honest_verdict": "complete: relational_goal_energy_live_route_qualified_no_solve_claim",
            "relational_goal_energy_ready_score": 1.0,
            "live_path_reachable_score": 1.0,
            "new_levels_claimed": 0,
            "solve_provenance": "development_proxy",
        },
        mod.EXP5712_ARC_AB_PATH: {
            "schema": "carnot.experiment_5712.arc_relational_goal_energy_live_ab.v1",
            "experiment": 5712,
            "status": "complete",
            "honest_verdict": "complete: relational_live_route_null_no_promotion",
            "relational_live_ab_ready_score": 0.0,
            "successful_pair_count": 6,
            "level_regression_count": 0,
            "unsafe_route_accept_count": 0,
            "new_levels_claimed": 0,
            "registry_updated": False,
            "solve_provenance": "development_proxy",
            "budget_parity_receipt": {"matched": True},
        },
        mod.EXP5713_ARC_LEVEL_PATH: {
            "schema": "arc_live_self_discovery_levelup_attempt.v510",
            "experiment_id": "exp5713-arc-live-self-discovery-levelup-v510",
            "status": "complete",
            "honest_verdict": "complete: no_new_arc_level_banked_lf52_L9_bounded_live_attempt_v510",
            "solve_provenance": "live_agent_self_discovery",
            "registry_count_before": 177,
            "registry_count_after": 177,
            "registry_delta": 0,
            "registry_updated": False,
        },
        mod.EXP5714_RUST_PARITY_PATH: {
            "schema": "carnot.experiment_5714.one_axis_tempering_rust_parity.v1",
            "experiment_id": "exp5714-one-axis-rust-python-exact-parity",
            "honest_verdict": "complete: one-axis corrected-cDLS Rust/Python parity is exact",
            "one_axis_rust_parity_ready_score": 1.0,
            "broken_control_rejected_score": 1.0,
            "checkpoint_roundtrip_pass": True,
            "cross_language_restart_pass": True,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
            "two_axis_code_added": False,
        },
        mod.EXP5715_RUST_QUALITY_PATH: {
            "schema": "carnot.experiment_5715.one_axis_tempering_rust_quality_restart.v1",
            "experiment_id": "exp5715-one-axis-tempering-rust-quality-restart",
            "honest_verdict": "complete: one-axis Rust/Python hard-instance quality pass",
            "one_axis_rust_quality_ready_score": 1.0,
            "material_regression_count": 0,
            "successful_seed_count": 5,
            "python_to_rust_restart_pass": True,
            "rust_to_python_restart_pass": True,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
            "transition_budget_parity": {"matched_corrected_transition_budget": True},
            "swap_schedule_parity": {"matched_swap_schedule": True},
        },
        mod.EXP5716_CAPSTONE_PATH: {
            "schema": "carnot.experiment_5716.v510_capstone_reconciliation.v1",
            "experiment_id": "exp5716-v510-capstone",
            "honest_verdict": "blocked: v510 reconciled; exp5709_promoted=False",
            "retirements_applied": [
                {
                    "scope": "fr11_prospective_shadow_stream_exp5709_same_verdict",
                    "manifest_update_required": True,
                    "manifest_entry_present": False,
                }
            ],
            "arc_registry_delta": 0,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
    }


def _write_rows(root: Path) -> None:
    rows: list[JsonDict] = []
    for idx in range(21):
        rows.append({"row_id": f"trunc-{idx}", "parse_ok": False, "parse_error": "truncated", "truncated": True, "validator_disagreement": False})
    for idx in range(26):
        rows.append({"row_id": f"missing-{idx}", "parse_ok": False, "parse_error": "missing_answer_line", "truncated": False, "validator_disagreement": False})
    for idx in range(3):
        rows.append({"row_id": f"parse-{idx}", "parse_ok": True, "parse_error": "", "truncated": False, "validator_disagreement": False})
    path = root / mod.EXP5708_ROWS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest_payload(include_new_retirement: bool = True) -> JsonDict:
    extras = [
        {
            "scope_key": "arc_counterexample_patched_transition_model_exp5641",
            "recorded_by_artifact": "results/experiment_5706_transition_v510.json",
        },
        {
            "scope_key": "two_axis_beta_lambda_tempering_extension_exp5645",
            "recorded_by_artifact": "results/experiment_5706_transition_v510.json",
        },
    ]
    if include_new_retirement:
        extras.append(dict(mod.REQUIRED_MANIFEST_RETIREMENT))
    return {"retired": [], "retired_experiments": [], "retired_extras": extras}


def _make_root(
    root: Path,
    *,
    include_new_retirement: bool = True,
    omit: Path | None = None,
    malformed: Path | None = None,
) -> None:
    for rel_path, payload in _payloads().items():
        if rel_path == omit:
            continue
        if rel_path == malformed:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json\n", encoding="utf-8")
            continue
        _write_json(root, rel_path, payload)
    _write_rows(root)
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-07-14 23:39 UTC | Milestone 2026.07.510 activated | OK | 11 tasks queued |",
                "| 2026-07-15 00:34 UTC | Gated on Exp5708 exact canary: prospective prequen | GATE_BLOCK | 1 of 3 gate(s) failed |",
                "| 2026-07-15 00:40 UTC | Gated on Exp5709 prospective promotion: isolated F | GATE_BLOCK | Pre-emptive skip: upstream retired |",
                "| 2026-07-19 19:01 UTC | Milestone 2026.07.511 activated | OK | 12 tasks queued |",
            ]
        )
        + "\n",
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, "tasks:\n- id: exp5717-transition-v511\n")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor\n")
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "[]\n")
    _write_json(root, mod.ARC_REGISTRY_RELATIVE_PATH, {"count": 177})
    manifest = _manifest_payload(include_new_retirement)
    manifest_path = root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def test_req_capstone_5717_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5717: OpenSpec declares the V511 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5717") :]

    assert "SCENARIO-CAPSTONE-5717" in section
    assert "results/experiment_5717_transition_v511.json" in section
    assert "sota_parse_failure_taxonomy" in section
    assert "fr11_prospective_shadow_stream_exp5709_same_verdict" in section


def test_scenario_capstone_5717_live_repo_archives_v510_and_allocates_v511() -> None:
    """SCENARIO-CAPSTONE-5717: live repo transition preserves terminal evidence."""

    artifact = mod.run_transition(
        root=REPO,
        validation_results=[
            {"command": ".venv/bin/pytest tests/python/test_experiment_5717_transition_v511.py -q", "exit_code": 0, "status": "passed"}
        ],
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["source_capstone_hash"] == mod.path_sha256(REPO / mod.EXP5716_CAPSTONE_PATH)
    assert artifact["current_task_range"] == "exp5717-exp5728"
    assert artifact["sota_parse_failure_taxonomy"] == {
        "manifest_row_count": 50,
        "truncation_count": 21,
        "missing_answer_count": 26,
        "parsed_answer_count": 3,
        "parse_failure_count": 47,
        "validator_disagreement_count": 0,
        "finish_reason_length_count": 21,
        "finish_reason_stop_count": 29,
    }
    assert artifact["cuda_offload_authenticated"] is True
    assert artifact["fr11_prospective_promoted"] is False
    assert artifact["fr11_isolated_promoted"] is False
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["arc_registry_count"] == 177
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_relational_route_promoted"] is False
    assert artifact["one_axis_rust_parity_ready_score"] == 1.0
    assert artifact["one_axis_rust_quality_ready_score"] == 1.0
    assert artifact["timing_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["inference_substrate"] == "artifact_reconciliation_only"
    assert artifact["v510_task_verdicts"]["exp5708-sota-exact-constraint-canary"]["status"] == "blocked"
    assert artifact["v510_task_verdicts"]["exp5709-fr11-prospective-shadow-stream"]["status"] == "gate_skipped"
    assert artifact["v510_task_verdicts"]["exp5710-fr11-isolated-act-on-advice-canary"]["status"] == "missing"
    assert artifact["v510_conductor_outcomes"]["exp5709-fr11-prospective-shadow-stream"]["outcome"] == "GATE_BLOCK"
    assert artifact["v510_conductor_outcomes"]["exp5710-fr11-isolated-act-on-advice-canary"]["outcome"] == "GATE_BLOCK_PREEMPTIVE_SKIP"
    assert artifact["v510_conductor_outcomes"]["milestone_2026.07.511_activation"]["outcome"] == "OK"

    applied_scopes = {row["scope"] for row in artifact["retirements_applied"]}
    assert applied_scopes == {"fr11_prospective_shadow_stream_exp5709_same_verdict"}
    assert artifact["retirements_applied"][0]["manifest_entry_present"] is True
    preserved = {row["scope"] for row in artifact["preserved_scopes"]}
    assert {
        "v509_fr11_independent_controller",
        "fr11_shadow_adapter_disabled_by_default",
        "future_clean_prospective_streams",
        "generic_lifecycle_learning",
        "generic_arc_working_memory",
        "arc_live_attempts",
        "one_axis_temperature_exchange",
    } <= preserved
    retired = {row["scope"] for row in artifact["retired_scopes"]}
    assert "fr11_prospective_shadow_stream_exp5709_same_verdict" in retired
    assert "generic_arc_working_memory" not in retired


def test_scenario_capstone_5717_fixture_missing_or_malformed_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5717-MISSING-MALFORMED: bad inputs fail closed."""

    _make_root(tmp_path, include_new_retirement=False, omit=mod.EXP5715_RUST_QUALITY_PATH)
    artifact = mod.run_transition(root=tmp_path)

    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.EXP5715_RUST_QUALITY_PATH.as_posix() in artifact["missing_artifacts"]
    assert artifact["one_axis_rust_quality_ready_score"] == 0.0
    assert artifact["retirements_applied"][0]["manifest_entry_present"] is False
    assert artifact["manifest_debt_after"] == ["fr11_prospective_shadow_stream_exp5709_same_verdict"]

    malformed_root = tmp_path / "malformed"
    _make_root(malformed_root, malformed=mod.EXP5708_CANARY_PATH)
    malformed = mod.run_transition(root=malformed_root)

    assert malformed["honest_verdict"].startswith("blocked:")
    assert mod.EXP5708_CANARY_PATH.as_posix() in malformed["malformed_artifacts"]
    assert malformed["cuda_offload_authenticated"] is False
    assert malformed["fr11_prospective_promoted"] is False


def test_scenario_capstone_5717_writer_validator_and_cli_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-5717-FIELD-PRINCIPLES: helpers stay deterministic."""

    _make_root(tmp_path)
    artifact = mod.write_transition(
        root=tmp_path,
        validation_results=[
            {"command": "focused", "exit_code": 0, "status": "passed"},
            {"command": "broad", "exit_code": 1, "status": "failed_pre_existing"},
        ],
    )
    output = tmp_path / mod.RESULT_RELATIVE_PATH

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["test_exit_codes"] == {"focused": 0, "broad": 1}
    assert mod.main(["--root", str(tmp_path), "--output", "custom/transition.json"]) == 0
    assert (tmp_path / "custom/transition.json").exists()

    bad = {**artifact, "fr11_prospective_promoted": True}
    assert "fr11_prospective_promoted" in " ".join(mod.validate_artifact(bad))
    bad_taxonomy = {**artifact, "sota_parse_failure_taxonomy": {"manifest_row_count": 50}}
    assert "sota_parse_failure_taxonomy" in " ".join(mod.validate_artifact(bad_taxonomy))
    bad_dependency = json.loads(json.dumps(artifact))
    bad_dependency["dependency_map"]["exp5728-v511-capstone-reconciliation"]["depends_on"] = [
        "exp9999-missing"
    ]
    assert "dependency_map" in " ".join(mod.validate_artifact(bad_dependency))
    bad_optional = json.loads(json.dumps(artifact))
    bad_optional["dependency_map"]["exp5727-arc-live-self-discovery-levelup-v511"][
        "optional_prerequisites"
    ] = ["exp9999-missing"]
    assert "dependency_map" in " ".join(mod.validate_artifact(bad_optional))
    bad_gate = json.loads(json.dumps(artifact))
    bad_gate["gate_map"]["exp5727-arc-live-self-discovery-levelup-v511"] = [
        {"upstream": "exp9999-missing"}
    ]
    assert "dependency_map" in " ".join(mod.validate_artifact(bad_gate))
    assert "dependency_map" in " ".join(mod.validate_artifact({**artifact, "gate_map": []}))

    rows_root = tmp_path / "rows"
    rows_path = rows_root / mod.EXP5708_ROWS_PATH
    rows_path.parent.mkdir(parents=True)
    rows_path.write_text(
        json.dumps(
            {
                "parse_ok": False,
                "parse_error": "missing_answer_line",
                "validator_disagreement": True,
                "finish_reason": "stop",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert mod._summarize_rows(rows_root)["validator_disagreement_count"] == 1

    monkeypatch.setattr(mod, "run_transition", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError, match="invalid Exp5717 transition artifact"):
        mod.write_transition(root=tmp_path)

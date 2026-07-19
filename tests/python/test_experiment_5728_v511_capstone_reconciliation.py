"""Tests for the Exp5728 V511 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5728, SCENARIO-CAPSTONE-5728,
SCENARIO-CAPSTONE-5728-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5728-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5728_v511_capstone_reconciliation as mod


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
        mod.EXP5717_TRANSITION_PATH: {
            "schema": "carnot.experiment_5717.transition_v511.v1",
            "status": "complete",
            "experiment_id": "exp5717-transition-v511",
            "honest_verdict": "complete: v511 transition archived .510 evidence",
            "current_task_range": "exp5717-exp5728",
            "cuda_offload_authenticated": True,
            "fr11_prospective_promoted": False,
            "fr11_isolated_promoted": False,
            "model_weight_mutation": False,
            "production_default_enabled": False,
            "arc_registry_count": 177,
            "arc_registry_delta": 0,
            "one_axis_rust_parity_ready_score": 1.0,
            "one_axis_rust_quality_ready_score": 1.0,
            "retirements_applied": [
                {
                    "scope": "fr11_prospective_shadow_stream_exp5709_same_verdict",
                    "manifest_entry_present": True,
                    "preserves": ["future_clean_prospective_streams"],
                }
            ],
            "preserved_scopes": [
                {"scope": "future_clean_prospective_streams"},
                {"scope": "generic_lifecycle_learning"},
                {"scope": "generic_arc_working_memory"},
                {"scope": "arc_live_attempts"},
                {"scope": "one_axis_temperature_exchange"},
                {"scope": "generic_replica_exchange"},
            ],
            "retired_scopes": [
                {"scope": "fr11_prospective_shadow_stream_exp5709_same_verdict"},
                {"scope": "two_axis_beta_lambda_tempering_extension_exp5645"},
            ],
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5718_SOURCE_PATH: {
            "schema": "carnot.experiment_5718.v511_source_delta_ingestion.v1",
            "status": "complete",
            "experiment_id": "exp5718-v511-source-delta-ingestion",
            "honest_verdict": "complete: accepted 1 non-duplicate actionable V511 source delta",
            "flagged_adversarial": True,
            "critical_flags": [{"code": "DURATION_TOO_SHORT"}],
            "references_updated": True,
            "roadmap_change_required": False,
        },
        mod.EXP5719_ANSWER_PATH: {
            "schema": "carnot.experiment_5719.sota_answer_channel_forensics.v1",
            "status": "blocked",
            "experiment_id": "exp5719-sota-answer-channel-forensics",
            "honest_verdict": "blocked: no_qualified_protocol",
            "MODEL_SPECS": [{"model_repo_id": "model-a"}, {"model_repo_id": "model-b"}],
            "qualified_model_ids": [],
            "qualified_model_count": 0,
            "qualified_protocol": {},
            "answer_channel_ready_score": 0.0,
            "positive_control_parse_rate": 0.0,
            "cuda_offload_authenticated": {"model-a": True, "model-b": True},
            "cuda_offload_authenticated_score": 0.0,
            "parse_failure_count": 82,
            "truncation_count": 41,
            "missing_answer_count": 82,
            "repetition_failure_count": 10,
            "semantic_error_count": 2,
            "validator_disagreement_count": 0,
            "native_json_grammar_used": False,
            "external_scorer_used": False,
            "retired_runtime_used": False,
            "model_hashes": {"model-a": "sha256:a"},
            "quantizations": {"model-a": "Q4"},
        },
        mod.EXP5720_STREAM_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5720,
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "4 of 4 gate(s) failed",
            "gates_evaluated": [
                {"upstream": "exp5719", "artifact_field": "answer_channel_ready_score", "passed": False}
            ],
        },
        mod.EXP5722_RECOVERY_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "experiment": 5722,
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "3 of 3 gate(s) failed",
            "gates_evaluated": [
                {"upstream": "exp5721", "artifact_field": "fr11_lifecycle_shadow_ready_score", "actual": None, "passed": False}
            ],
        },
        mod.EXP5723_RUST_BACKEND_PATH: {
            "schema": "carnot.experiment_5723.one_axis_rust_samplerbackend_integration.v1",
            "status": "complete",
            "experiment_id": "exp5723-one-axis-rust-samplerbackend-integration",
            "honest_verdict": "complete: one-axis Rust/PyO3 kernel is exposed",
            "one_axis_samplerbackend_ready_score": 1.0,
            "exact_fallback_equivalence_score": 1.0,
            "fallback_equivalence_pass": True,
            "two_axis_code_added": False,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5724_CROSSOVER_PATH: {
            "schema": "carnot.experiment_5724.one_axis_rust_python_matched_crossover.v1",
            "status": "complete",
            "experiment_id": "exp5724-one-axis-rust-python-matched-crossover",
            "honest_verdict": "complete: terminal null; no crossover proven",
            "quality_matched_pair_count": 178,
            "qualified_crossover_n": None,
            "rust_crossover_ready_score": 0.0,
            "software_speedup_claimed": False,
            "timing_claimed": True,
            "hardware_speedup_claimed": False,
            "gpu_speedup_claimed": False,
            "fpga_or_tsu_used": False,
        },
        mod.EXP5725_ARC_QUAL_PATH: {
            "schema": "carnot.experiment_5725.arc_epistemic_ledger_live_qualification.v1",
            "status": "complete",
            "experiment": 5725,
            "honest_verdict": "complete: arc_epistemic_ledger_live_reachable_safe_no_solve_claim",
            "arc_epistemic_ledger_ready_score": 1.0,
            "live_path_reachable_score": 1.0,
            "solve_provenance": "development_proxy",
            "new_levels_claimed": 0,
            "registry_updated": False,
            "game_source_read_count": 0,
            "game_adapter_count": 0,
            "outer_loop_bfs_used": False,
            "per_game_leakage_detected": False,
            "unsafe_commit_count": 0,
        },
        mod.EXP5726_ARC_AB_PATH: {
            "schema": "carnot.experiment_5726.arc_epistemic_ledger_live_ab.v1",
            "status": "complete",
            "experiment": 5726,
            "honest_verdict": "complete: epistemic_ledger_live_ab_null_no_promotion",
            "arc_epistemic_live_ab_ready_score": 0.0,
            "successful_pair_count": 6,
            "unsafe_commit_count": 0,
            "new_levels_claimed": 0,
            "registry_updated": False,
            "solve_provenance": "development_proxy",
        },
        mod.EXP5727_ARC_GAP_PATH: {
            "schema": "carnot.exp5727.arc_generalization_live_oracle_gap.v1",
            "status": "complete",
            "experiment": "experiment_5727_arc_generalization_live_oracle_gap_v511",
            "honest_verdict": "complete: arc_generalization_live_oracle_gap_4_of_183_levels_gap_179",
            "harness_used": "scripts/arc_leaderboard_eval.py",
            "policy_kind": "e3",
            "budget_per_game": 400,
            "games_measured": 25,
            "expected_registry_games": 25,
            "skipped_games": [],
            "live_levels_total": 4,
            "oracle_levels_total": 183,
            "gap_total": 179,
            "per_game_gap": [{"game": "lf52", "live_levels": 0, "oracle_levels": 10, "gap": 10}],
            "worst_gap_games": [{"game": "lf52", "gap": 10}],
            "any_new_level_found": False,
            "new_level_evidence": [],
        },
    }


def _make_root(root: Path, *, omit: Path | None = None, malformed: Path | None = None) -> None:
    for rel_path, payload in _payloads().items():
        if rel_path == omit:
            continue
        if rel_path == malformed:
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json\n", encoding="utf-8")
            continue
        _write_json(root, rel_path, payload)
    manifest = {
        "retired_extras": [
            {"scope_key": "fr11_prospective_shadow_stream_exp5709_same_verdict"},
            {"scope_key": "two_axis_beta_lambda_tempering_extension_exp5645"},
        ]
    }
    path = root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    registry = {
        "reproducible_total_levels": 183,
        "reproducible_total_games": 25,
        "games": [{"game": f"g{idx}", "full_game_clear": True} for idx in range(25)],
    }
    reg_path = root / mod.ARC_REGISTRY_RELATIVE_PATH
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-07-19 19:01 UTC | Milestone 2026.07.511 activated | OK | 12 tasks queued |",
                "| 2026-07-19 19:42 UTC | Ingest post-V511 2025-2026 source deltas with dupl | FLAGGED | adversarial_verify CRITICAL |",
                "| 2026-07-19 20:09 UTC | Gated on Exp5719 channel readiness: build a sealed | GATE_BLOCK | 4 of 4 gate(s) failed |",
                "| 2026-07-19 20:15 UTC | Gated on Exp5720 exact stream: prospective MemOps | GATE_BLOCK | Pre-emptive skip: upstream retired |",
                "| 2026-07-19 20:15 UTC | Gated on Exp5721 lifecycle readiness: isolated Com | GATE_BLOCK | 3 of 3 gate(s) failed |",
            ]
        )
        + "\n",
    )
    for rel_path in mod.FORBIDDEN_FILE_PATHS:
        _write_text(root, rel_path, "unchanged\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, SPEC_PATH.read_text(encoding="utf-8"))


def test_req_capstone_5728_spec_declares_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5728: OpenSpec declares the V511 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5728") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "live_levels_total" in section
    assert "any_new_level_found" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5728_live_repo_reconciles_terminal_v511_evidence() -> None:
    """SCENARIO-CAPSTONE-5728: live evidence is reconciled without over-credit."""

    artifact = mod.run_capstone(
        root=REPO,
        validation_results=[{"command": "unit", "exit_code": 0, "status": "passed"}],
        modification_overrides={path: False for path in mod.FORBIDDEN_FILE_PATHS},
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["milestone"] == "2026.07.511"
    assert mod.EXP5721_LIFECYCLE_PATH.as_posix() in artifact["missing_artifacts"]
    assert artifact["conductor_gate_statuses"][mod.EXP5718_TASK_ID]["outcome"] == "FLAGGED"
    assert artifact["conductor_gate_statuses"][mod.EXP5721_TASK_ID]["outcome"] == "GATE_BLOCK"

    assert artifact["transition_status"]["narrow_exp5709_retirement_applied"] is True
    assert artifact["source_ingestion_status"]["quarantined_by_adversarial_flag"] is True
    assert artifact["answer_channel_status"]["answer_channel_ready_score"] == 0.0
    assert artifact["qualified_model_ids"] == []
    assert artifact["sota_attested_stream_status"]["promoted"] is False
    assert artifact["parse_failure_count"] == 82
    assert artifact["validator_disagreement_count"] == 0
    assert artifact["attestation_coverage"]["coverage"] == 0.0
    assert artifact["stream_commitment_status"]["status"] == "gate_skipped"

    assert artifact["fr11_lifecycle_shadow_status"]["status"] == "missing"
    assert artifact["fr11_recovery_canary_status"]["status"] == "gate_skipped"
    assert artifact["continuous_self_learning_credited"] is False
    assert artifact["unsafe_false_accept_count"] is None
    assert artifact["unsafe_update_accept_count"] is None
    assert artifact["negative_transfer_count"] is None
    assert artifact["retention_regression_count"] is None
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False

    assert artifact["rust_samplerbackend_status"]["promoted"] is True
    assert artifact["rust_python_crossover_status"]["terminal_null"] is True
    assert artifact["quality_matched_pair_count"] == 178
    assert artifact["qualified_crossover_n"] is None
    assert artifact["software_speedup_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["two_axis_retirement_preserved"] is True

    assert artifact["arc_epistemic_qualification_status"]["qualified"] is True
    assert artifact["arc_epistemic_live_ab_status"]["promoted"] is False
    assert artifact["arc_live_attempt_status"]["scope"] == "arc_generalization_live_oracle_gap"
    assert artifact["arc_live_attempt_status"]["live_levels_total"] == 4
    assert artifact["arc_live_attempt_status"]["oracle_levels_total"] == 183
    assert artifact["arc_live_attempt_status"]["gap_total"] == 179
    assert artifact["arc_solve_provenance"]["exp5727"] == "measurement_not_solve_claim"
    assert artifact["arc_registry_count_before"] == 183
    assert artifact["arc_registry_count_after"] == 183
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_solve_credited"] is False
    assert artifact["arc_forbidden_path_counts"]["known_forbidden_count"] == 0

    assert {row["scope"] for row in artifact["retirements_applied"]} == {
        "fr11_prospective_shadow_stream_exp5709_same_verdict"
    }
    preserved = {row["scope"] for row in artifact["preserved_scopes"]}
    assert {
        "future_clean_prospective_streams",
        "generic_lifecycle_learning",
        "external_memory_csl",
        "samplerbackend_contract",
        "arc_epistemic_state",
        "arc_live_attempts",
    } <= preserved
    assert artifact["timing_claimed"] is True
    assert artifact["claim_boundaries"]["blocked_skipped_missing_null_cannot_promote"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_scenario_capstone_5728_missing_and_malformed_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5728-MISSING-MALFORMED: bad inputs block dependent claims."""

    _make_root(tmp_path, omit=mod.EXP5723_RUST_BACKEND_PATH, malformed=mod.EXP5724_CROSSOVER_PATH)
    artifact = mod.run_capstone(root=tmp_path)

    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.EXP5723_RUST_BACKEND_PATH.as_posix() in artifact["missing_artifacts"]
    assert mod.EXP5724_CROSSOVER_PATH.as_posix() in artifact["malformed_artifacts"]
    assert artifact["rust_samplerbackend_status"]["promoted"] is False
    assert artifact["rust_python_crossover_status"]["terminal_null"] is False
    assert artifact["software_speedup_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5728_validation_rejects_overclaims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5728-FIELD-PRINCIPLES: validation rejects laundering."""

    _make_root(tmp_path)
    artifact = mod.run_capstone(root=tmp_path)
    assert mod.validate_artifact(artifact) == []

    assert "field_principles" in " ".join(
        mod.validate_artifact({**artifact, "field_principles": {"honest_verdict": "x"}})
    )
    assert "source_ingestion_status" in " ".join(
        mod.validate_artifact(
            {**artifact, "source_ingestion_status": {"quarantined_by_adversarial_flag": False}}
        )
    )
    flagged_not_quarantined = dict(
        artifact["source_ingestion_status"],
        flagged_adversarial=True,
        quarantined_by_adversarial_flag=False,
    )
    assert "source_ingestion_status" in " ".join(
        mod.validate_artifact({**artifact, "source_ingestion_status": flagged_not_quarantined})
    )
    quarantined_counted = dict(
        artifact["source_ingestion_status"],
        quarantined_by_adversarial_flag=True,
        counts_as_success=True,
    )
    assert "source_ingestion_status" in " ".join(
        mod.validate_artifact({**artifact, "source_ingestion_status": quarantined_counted})
    )
    assert "qualified_model_ids" in " ".join(
        mod.validate_artifact({**artifact, "qualified_model_ids": ["model-a"]})
    )
    assert "sota_attested_stream_status" in " ".join(
        mod.validate_artifact({**artifact, "sota_attested_stream_status": {"promoted": True}})
    )
    assert "continuous_self_learning_credited" in " ".join(
        mod.validate_artifact({**artifact, "continuous_self_learning_credited": True})
    )
    assert "model_weight_mutation" in " ".join(
        mod.validate_artifact({**artifact, "model_weight_mutation": True})
    )
    assert "production_default_enabled" in " ".join(
        mod.validate_artifact({**artifact, "production_default_enabled": True})
    )
    assert "rust_python_crossover_status" in " ".join(
        mod.validate_artifact({**artifact, "rust_python_crossover_status": {"terminal_null": False}})
    )
    assert "software_speedup_claimed" in " ".join(
        mod.validate_artifact({**artifact, "software_speedup_claimed": True})
    )
    assert "hardware_speedup_claimed" in " ".join(
        mod.validate_artifact({**artifact, "hardware_speedup_claimed": True})
    )
    assert "two_axis_retirement_preserved" in " ".join(
        mod.validate_artifact({**artifact, "two_axis_retirement_preserved": False})
    )
    assert "arc_registry_delta" in " ".join(
        mod.validate_artifact({**artifact, "arc_registry_delta": 1})
    )
    assert "arc_solve_credited" in " ".join(
        mod.validate_artifact({**artifact, "arc_solve_credited": True})
    )
    assert "timing_claimed" in " ".join(
        mod.validate_artifact({**artifact, "timing_claimed": False})
    )
    assert "inference_substrate" in " ".join(
        mod.validate_artifact({**artifact, "inference_substrate": "aggregation_from_upstream_artifacts"})
    )
    assert "reproducibility_checksum" in " ".join(
        mod.validate_artifact({**artifact, "reproducibility_checksum": "bad"})
    )
    assert "honest_verdict" in " ".join(
        mod.validate_artifact({**artifact, "honest_verdict": "maybe"})
    )


def test_scenario_capstone_5728_writer_cli_and_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-5728-FIELD-PRINCIPLES: writer and helpers are stable."""

    _make_root(tmp_path)
    validation = [
        {"command": "focused", "exit_code": 0, "status": "passed"},
        {"command": "audit", "exit_code": 1, "status": "pre_existing_debt"},
    ]
    artifact = mod.write_capstone(root=tmp_path, validation_results=validation)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["test_exit_codes"] == {"focused": 0, "audit": 1}

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    assert mod._load_validation_results(validation_path) == validation
    bad_validation = tmp_path / "bad-validation.json"
    bad_validation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_validation_results(bad_validation)

    missing_payload, missing_meta = mod._read_json_object(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"
    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    list_payload, list_meta = mod._read_json_object(list_json)
    assert list_payload == {}
    assert list_meta["error"] == "not_json_object"

    assert mod._task_status({}, {"exists": False}) == "missing"
    assert mod._task_status({}, {"exists": True, "loadable": False}) == "malformed"
    assert mod._task_status({"flagged_adversarial": True}, {"exists": True, "loadable": True}) == "flagged"
    assert mod._task_status({"schema": "blocked_gate_check_v1"}, {"exists": True, "loadable": True}) == "gate_skipped"
    assert mod._task_status({"honest_verdict": "blocked: x"}, {"exists": True, "loadable": True}) == "blocked"
    assert mod._task_status({"honest_verdict": "complete: x"}, {"exists": True, "loadable": True}) == "complete"
    assert mod._task_status({"honest_verdict": "other"}, {"exists": True, "loadable": True}) == "unknown"
    assert mod._registry_count({"reproducible_total_levels": "183"}) == 183
    assert mod._registry_count({"reproducible_total_levels": True}) is None
    assert mod._registry_count({"reproducible_total_levels": "unknown"}) is None
    assert mod._full_clear_count({"games": [{"full_game_clear": True}, {"full_game_clear": False}]}) == 1
    assert mod._full_clear_count({"games": None}) == 0
    assert mod._extract_outcome("2026-07-19 | exp5728 | NOTE | logged") == "LOGGED"
    assert mod._applied_retirements({"retirements_applied": [None]}, {}) == []

    assert mod.main(["--root", str(tmp_path), "--output", "custom/capstone.json"]) == 0
    assert (tmp_path / "custom/capstone.json").exists()

    monkeypatch.setattr(mod, "run_capstone", lambda **_kwargs: {"schema": "bad"})
    with pytest.raises(ValueError, match="invalid Exp5728 capstone artifact"):
        mod.write_capstone(root=tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["schema"])
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])
    monkeypatch.setattr(mod, "write_capstone", lambda **_kwargs: artifact)
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tmp_path)])

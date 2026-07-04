"""Tests for Exp 5219 V477 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5219, SCENARIO-CAPSTONE-5219,
SCENARIO-CAPSTONE-5219-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5219_capstone_v477 as mod
from scripts import experiment_5219_capstone_v477 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _base(experiment: str | int, verdict: str, *, flagged: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": experiment,
        "honest_verdict": verdict,
        "duration_s": 1.0,
        "inference_substrate": "fixture",
    }
    if flagged is not None:
        payload["flagged_adversarial"] = flagged
    return payload


def _payloads() -> dict[int, dict[str, Any]]:
    return {
        5207: _base(
            "experiment_5207_archive_476_activate_477",
            "complete: .476 archived and .477 activated; handoff preserves nulls.",
        ),
        5208: _base(
            "experiment_5208_sota_ingestion_v477",
            "complete: V477 SOTA refresh found no new actionable findings.",
        ),
        5209: {
            **_base(
                "experiment_5209_gap1_set_search_holdout_hardening_v477",
                "complete: GAP-1 hardening remains positive.",
            ),
            "gap1_hardened_positive": _wrap(True),
            "heldout_pass_at_2_mean": _wrap(0.189584),
            "baseline_always_on_pass_at_2_mean": _wrap(0.088976),
            "single_refuted_directional_pass_at_2_mean": _wrap(0.147787),
            "paired_delta_ci95": _wrap("[0.023148, 0.060446]"),
            "leakage_audit_passed": _wrap(True),
            "best_subset_stable": _wrap(False),
        },
        5210: {
            "experiment": 5210,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "principle-wrapped gap1_hardened_positive compared as object",
            "gates_evaluated": [
                {
                    "upstream": "exp5209-gap1-set-search-holdout-hardening-v477",
                    "artifact_field": "gap1_hardened_positive",
                    "expected": True,
                    "actual": _wrap(True),
                    "passed": False,
                }
            ],
            "duration_s": 0.0,
        },
        5211: {
            **_base(
                "experiment_5211_gap4_sota_local_candidate_expansion_v477",
                "complete_gap4_sota_local_candidate_expansion_v477_n120_pool_ready_for_exp5212",
                flagged=True,
            ),
            "candidate_pool_n": _wrap(120),
            "gap4_expansion_usable": _wrap(True),
            "leakage_audit_passed": _wrap(True),
        },
        5212: {
            **_base(
                "experiment_5212_gap4_scale_validation_gated_v477",
                "complete_gap4_scale_validation_v477_n0_missing_protocol_pass2_fields_blocked",
                flagged=True,
            ),
            "n_scored": _wrap(0),
            "gap4_status_recommendation": _wrap("blocked"),
            "exact_test_passes_min6_rule": _wrap(False),
            "exact_test_discordant_wins": _wrap(0),
            "exact_test_discordant_losses": _wrap(0),
        },
        5213: {
            **_base(
                "experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477",
                "complete_hidden_state_v3_signal_does_not_beat_all_controls_retires_mmlu_hidden_state_path",
            ),
            "beats_all_controls": _wrap(False),
            "best_probe_accuracy": _wrap(0.075),
            "tuned_sc_accuracy": _wrap(0.075),
            "self_certainty_accuracy": _wrap(0.075),
            "clue_accuracy": _wrap(0.025),
            "radial_consensus_score_accuracy": _wrap(0.025),
            "retire_mmlu_hidden_state_path": _wrap(True),
        },
        5214: {
            **_base(
                "experiment_5214_continuous_self_learning_verifier_memory_v477",
                "complete: verifier_memory_from_upstream_artifacts_promotions_1_rollbacks_1",
            ),
            "continuous_self_learning_task": True,
            "deterministic_guardrails_enforced": True,
            "heldout_gate_required_for_promotion": True,
            "memory_artifact_path": "results/verifier_memory_v477.json",
            "memory_entries_written": 2,
            "memory_summary": {
                "promotions": 1,
                "rollbacks": 1,
                "deterministic_guardrails_enforced": True,
                "heldout_gate_required_for_promotion": True,
            },
            "promotions": 1,
            "rollbacks": 1,
        },
        5215: {
            **_base(
                "experiment_5215_arc_paw_amortization_gate_v477",
                "complete_paw_amortization_gate_not_viable_no_arc_solve_claim",
                flagged=True,
            ),
            "paw_amortization_viable": _wrap(False),
            "arc_registry_modified": _wrap(False),
            "median_remaining_actions": _wrap(29.5),
            "p75_remaining_actions": _wrap(43.75),
            "break_even_remaining_actions": _wrap(45.748641),
        },
        5216: {
            **_base(
                "experiment_5216_arc_frontier_continuity_landmark_decomposition_v477",
                "complete: frontier continuity plus landmark decomposition did not bank a level.",
            ),
            "new_levels_banked": [],
            "reproducible_total_levels_delta": 0,
            "solve_provenance": "development_proxy",
            "duplicate_registry_precheck_passed": True,
        },
        5217: {
            **_base(
                "experiment_5217_hardware_continuity",
                "complete_hardware_continuity_v477_kv260:reachable_gatemate:blocked_polarfire:reachable_no_speedup_claim",
            ),
            "boards_reachable_count": 2,
            "kv260_status": "reachable + hash-verified smoke",
            "polarfire_status": {
                "reachable": True,
                "hash_verified": True,
                "polarfire_workload_validated": False,
                "summary": "reachable + hash-verified smoke",
            },
            "gatemate_status": {
                "status": "blocked_gatemate_dirtyjtag_idcode_unresolved_v477",
                "narrowed_to": "cable_or_port",
                "leading_untested_hypothesis": "physical_board",
            },
            "gatemate_diagnostic_narrowed_to": "cable_or_port",
            "hardware_speedup_claimed": False,
        },
        5218: {
            **_base(
                "experiment_5218_verifier_authenticity_remediation_apply_v477",
                "complete: dishonest-naming risk reduced by registry flags.",
            ),
            "remediation_applied": _wrap(True),
            "headline_ineligible_until_real_verification": _wrap(True),
            "no_research_conductor_change": _wrap(True),
            "inference_substrate": _wrap("code_and_doc_remediation"),
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    payloads = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, payloads[source.experiment_number])
    _write_json(
        root / "results/experiment_5211_gap4_sota_local_candidate_expansion_v477.checkpoint.json",
        {
            "experiment": "experiment_5211_gap4_sota_local_candidate_expansion_v477",
            "accepted_rows": [],
        },
    )
    _write_json(
        root / "results/verifier_memory_v477.json",
        {
            "schema": "carnot.verifier_memory.v1",
            "summary": {
                "memory_entries_written": 2,
                "promotions": 1,
                "rollbacks": 1,
                "deterministic_guardrails_enforced": True,
                "heldout_gate_required_for_promotion": True,
            },
            "entries": [
                {"promotion_state": "rolled_back", "rollback_reason": "heldout_delta_missing"},
                {"promotion_state": "promoted", "rollback_reason": None},
            ],
        },
    )


def _reporter(path: Path) -> dict[str, Any]:
    if "5213" in path.name:
        return {
            "artifact": str(path),
            "loaded": True,
            "flag_count": 1,
            "max_severity": 1,
            "flags": [{"kind": "METHODOLOGY_NOTE", "severity": "warn"}],
        }
    return {"artifact": str(path), "loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}


def _validation() -> list[dict[str, str]]:
    return [{"command": "focused pytest", "status": "PASS", "notes": "fixture"}]


def _field(field: str, value: Any) -> dict[str, Any]:
    return _wrap(value, mod.FIELD_PRINCIPLES[field])


def test_req_capstone_5219_spec_declares_v477_contract() -> None:
    """REQ-CAPSTONE-5219: OpenSpec declares the V477 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5219") :]

    for marker in (
        "REQ-CAPSTONE-5219",
        "SCENARIO-CAPSTONE-5219",
        "SCENARIO-CAPSTONE-5219-FIELD-PRINCIPLES",
        mod.EXPERIMENT_ID,
        str(mod.RESULT_RELATIVE_PATH),
        "gap1_final_status",
        "docs_reconciled=false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5219_reconciles_v477_without_headlining_flagged_inputs(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5219: blocked and flagged upstreams stay out of headlines."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260704",
        duration_s=1.25,
        validation_commands_run=_validation(),
        adversarial_reporter=_reporter,
        conductor_untouched=True,
        docs_reconciled=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["gap1_final_status"]["value"] == "building"
    assert artifact["gap4_final_status"]["value"] == "blocked"
    assert artifact["hidden_state_path_decision"]["value"] == "retire_mmlu_path"
    assert artifact["continuous_self_learning_satisfied"]["value"] is True
    assert artifact["new_levels_banked"]["value"] == []
    assert artifact["reproducible_total_levels_delta"]["value"] == 0
    assert artifact["flagged_adversarial_artifacts_excluded"]["value"] is True
    assert artifact["docs_reconciled"]["value"] is False
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is True
    assert artifact["flagged_adversarial"] is False
    assert "GateMate=cable_or_port" in artifact["hardware_final_state"]["value"]
    assert "no speedup claim" in artifact["hardware_final_state"]["value"]
    assert "Exp5210 blocked" in artifact["status_decisions"]["gap1"]
    assert "Exp5212" in artifact["status_decisions"]["gap4"]

    summary = artifact["per_task_summary"]["value"]
    assert len(summary) == 12
    assert summary["exp5210-gap1-registry-promotion-gated-v477"]["status"] == "blocked"
    assert summary["exp5210-gap1-registry-promotion-gated-v477"]["gated_and_skipped"] is True
    assert summary["exp5211-gap4-sota-local-candidate-expansion-v477"]["headline_eligible"] is False
    assert summary["exp5212-gap4-scale-validation-gated-v477"]["headline_eligible"] is False
    assert summary["exp5215-arc-paw-amortization-gate-v477"]["headline_eligible"] is False
    assert (
        "exp5213-hidden-state-verifier-v3-layer-chunk-sweep-v477"
        in artifact["headline_eligible_task_ids"]
    )
    assert (
        "exp5211-gap4-sota-local-candidate-expansion-v477"
        not in artifact["headline_eligible_task_ids"]
    )
    assert artifact["checkpoint_artifacts"][0]["relative_path"].endswith(".checkpoint.json")


def test_req_capstone_5219_validation_and_run_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5219: schema validation rejects overclaims and stale checksums."""

    _make_repo(tmp_path, omit={5218})
    payloads = _payloads()
    payloads[5213]["flagged_adversarial"] = True
    _write_json(tmp_path / mod.source_by_number(5213).relative_path, payloads[5213])

    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260704",
        duration_s=2.0,
        validation_commands_run=_validation(),
        adversarial_reporter=_reporter,
        conductor_untouched=True,
        docs_reconciled=False,
    )

    assert artifact["missing_artifacts"] == ["exp5218-verifier-authenticity-remediation-apply-v477"]
    assert artifact["hidden_state_path_decision"]["value"] == "blocked"
    assert artifact["flagged_adversarial_artifacts_excluded"]["value"] is True
    assert (
        "exp5213-hidden-state-verifier-v3-layer-chunk-sweep-v477"
        not in artifact["headline_eligible_task_ids"]
    )
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(
            {key: value for key, value in artifact.items() if key != "duration_s"}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "done"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="field principle mismatch"):
        mod.validate_artifact(
            artifact
            | {"field_principles": artifact["field_principles"] | {"honest_verdict": "loose"}}
        )
    with pytest.raises(ValueError, match="gap1_final_status field principle mismatch"):
        mod.validate_artifact(artifact | {"gap1_final_status": _wrap("building")})
    with pytest.raises(ValueError, match="flagged_adversarial"):
        mod.validate_artifact(artifact | {"flagged_adversarial": True})
    with pytest.raises(ValueError, match="research_conductor"):
        mod.validate_artifact(
            artifact
            | {
                "research_conductor_py_untouched_confirmed": _field(
                    "research_conductor_py_untouched_confirmed", False
                )
            }
        )
    with pytest.raises(ValueError, match="gap1_final_status"):
        mod.validate_artifact(
            artifact | {"gap1_final_status": _field("gap1_final_status", "weird")}
        )
    with pytest.raises(ValueError, match="gap4_final_status"):
        mod.validate_artifact(
            artifact | {"gap4_final_status": _field("gap4_final_status", "weird")}
        )
    with pytest.raises(ValueError, match="hidden_state_path_decision"):
        mod.validate_artifact(
            artifact | {"hidden_state_path_decision": _field("hidden_state_path_decision", "weird")}
        )
    with pytest.raises(ValueError, match="headline eligible"):
        mod.validate_artifact(
            artifact
            | {
                "headline_eligible_task_ids": [
                    "exp5213-hidden-state-verifier-v3-layer-chunk-sweep-v477"
                ]
            }
        )
    with pytest.raises(ValueError, match="flagged_adversarial_artifacts_excluded"):
        mod.validate_artifact(
            artifact
            | {
                "flagged_adversarial_artifacts_excluded": _field(
                    "flagged_adversarial_artifacts_excluded", False
                )
            }
        )
    with pytest.raises(ValueError, match="per_task_summary"):
        mod.validate_artifact(artifact | {"per_task_summary": _field("per_task_summary", [])})
    with pytest.raises(ValueError, match="docs_reconciled"):
        mod.validate_artifact(artifact | {"docs_reconciled": _field("docs_reconciled", "no")})
    with pytest.raises(ValueError, match="validation_commands_run"):
        mod.validate_artifact(
            artifact | {"validation_commands_run": _field("validation_commands_run", [])}
        )
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "stale"})
    with pytest.raises(ValueError, match="reproducible_total_levels_delta"):
        mod.validate_artifact(
            artifact
            | {"reproducible_total_levels_delta": _field("reproducible_total_levels_delta", 1)}
        )

    assert mod.value_of(_wrap("x")) == "x"
    assert mod.value_of("x") == "x"
    assert mod.honest_verdict_text({"value": "complete_wrapped"}) == "complete_wrapped"
    assert mod.honest_verdict_text(None) == ""
    assert mod._number("3.5") == 3.5
    assert mod._number("bad") is None
    assert mod._number(True) is None
    assert mod._bool({"value": True}) is True
    assert mod._flag_is_critical({"severity": 2}) is True
    assert mod._flag_is_critical({"severity": "critical"}) is True
    assert (
        mod.is_gated_block({"status": "blocked", "blocked_at_layer": "conductor_pre_gate"}) is True
    )
    assert mod.is_gated_block({"status": "failed"}) is False
    with pytest.raises(KeyError):
        mod.source_by_number(9999)

    assert mod.gap1_status({}) == ("blocked", "missing Exp5209 hardening artifact")
    assert mod.gap1_status({5209: {"gap1_hardened_positive": False}})[0] == "open"
    assert mod.gap1_status({5209: {"gap1_hardened_positive": True}})[0] == "building"
    assert (
        mod.gap1_status(
            {5209: {"gap1_hardened_positive": True}, 5210: {"verifier_registered": True}}
        )[0]
        == "filled"
    )
    assert (
        mod.gap1_status({5209: {"gap1_hardened_positive": True}, 5210: {"status": "complete"}})[0]
        == "building"
    )
    assert mod.gap4_status({}) == ("blocked", "missing Exp5211 expansion artifact")
    assert (
        mod.gap4_status(
            {5211: {"gap4_expansion_usable": True}},
            ["exp5211-gap4-sota-local-candidate-expansion-v477"],
        )[0]
        == "blocked"
    )
    assert mod.gap4_status({5211: {"gap4_expansion_usable": False}})[0] == "open"
    assert mod.gap4_status({5211: {"gap4_expansion_usable": True}})[0] == "building"
    assert (
        mod.gap4_status(
            {5211: {"gap4_expansion_usable": True}, 5212: {"present": True}},
            ["exp5212-gap4-scale-validation-gated-v477"],
        )[0]
        == "blocked"
    )
    assert (
        mod.gap4_status(
            {5211: {"gap4_expansion_usable": True}, 5212: {"exact_test_passes_min6_rule": True}}
        )[0]
        == "filled"
    )
    assert (
        mod.gap4_status(
            {5211: {"gap4_expansion_usable": True}, 5212: {"gap4_status_recommendation": "blocked"}}
        )[0]
        == "blocked"
    )
    assert (
        mod.gap4_status(
            {5211: {"gap4_expansion_usable": True}, 5212: {"exact_test_passes_min6_rule": False}}
        )[0]
        == "open"
    )
    assert mod.hidden_state_decision({}) == ("blocked", "missing Exp5213 hidden-state artifact")
    assert mod.hidden_state_decision({5213: {"beats_all_controls": True}})[0] == "keep"
    assert mod.hidden_state_decision({5213: {"present": True}})[0] == "blocked"
    assert mod.self_learning_satisfied(tmp_path, {}) is False
    assert (
        mod.self_learning_satisfied(
            tmp_path,
            {5214: {"memory_artifact_path": "results/verifier_memory_v477.json"}},
            ["exp5214-continuous-self-learning-verifier-memory-v477"],
        )
        is False
    )
    assert mod.self_learning_satisfied(tmp_path, {5214: {"memory_artifact_path": 3}}) is False
    assert (
        mod.self_learning_satisfied(tmp_path, {5214: {"memory_artifact_path": "missing.json"}})
        is False
    )
    assert mod.arc_level_delta(
        {5216: {"new_levels_banked": ["L"], "reproducible_total_levels_delta": 1}},
        ["exp5216-arc-frontier-continuity-landmark-decomposition-v477"],
    ) == ([], 0)
    assert mod.hardware_summary({}) == "hardware evidence missing"
    assert (
        mod.hardware_summary(
            {5217: {"hardware_speedup_claimed": False}},
            ["exp5217-hardware-continuity-v477"],
        )
        == "hardware evidence excluded because Exp5217 is flagged"
    )

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["error"] == "malformed_json"
    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "not_json_object"

    out_path = mod.run(
        root=tmp_path,
        run_date="20260704",
        duration_s=2.5,
        validation_commands_run=_validation(),
        adversarial_reporter=_reporter,
        conductor_untouched=True,
        docs_reconciled=False,
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["duration_s"] == 2.5
    assert saved["validation_commands_run"]["value"] == _validation()
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)

    assert (
        script_mod.main(
            root=tmp_path,
            date="20260704",
            duration_s=3.0,
            validation_commands_run=_validation(),
            adversarial_reporter=_reporter,
            conductor_untouched=True,
            docs_reconciled=False,
        )
        == out_path
    )
    assert (
        script_mod.main(
            ["--root", str(tmp_path), "--date", "20260704"],
            duration_s=3.5,
            validation_commands_run=_validation(),
            adversarial_reporter=_reporter,
            conductor_untouched=True,
            docs_reconciled=False,
        )
        == out_path
    )

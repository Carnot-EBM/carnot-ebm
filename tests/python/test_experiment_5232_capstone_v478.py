"""Tests for Exp 5232 V478 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5232, SCENARIO-CAPSTONE-5232,
SCENARIO-CAPSTONE-5232-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import carnot.experiment_5232_capstone_v478 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


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
        5220: {
            **_base(
                "experiment_5220_archive_477_activate_478",
                "complete: .477 archived and .478 activated.",
            ),
            "research_roadmap_yaml_activated": _wrap(True),
        },
        5221: _base(
            "experiment_5221_sota_ingestion_v478",
            "complete: V478 SOTA refresh found no new actionable findings.",
        ),
        5222: {
            **_base(
                "experiment_5222_gap1_gate_field_registry_promotion_v478",
                "complete: GAP-1 registry promotion blocked_instability.",
            ),
            "gap1_registry_promoted": _wrap(False),
            "gap1_registry_decision": _wrap("blocked_instability"),
            "exp5209_gate_parsed_from_value": _wrap(True),
            "promoted_registry_path": _wrap(None),
            "frozen_subset": _wrap(None),
        },
        5223: {
            **_base(
                "experiment_5223_gap4_flagged_pool_authenticity_audit_v478",
                "complete: old GAP-4 pool must be regenerated.",
                flagged=True,
            ),
            "gap4_pool_repairable": False,
            "validated_pool_n": 0,
            "protocol_fields_complete": False,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
        5224: {
            **_base(
                "experiment_5224_gap4_canonical_pool_builder_v478",
                "success: canonical GAP-4 pool usable for validation.",
                flagged=True,
            ),
            "gap4_canonical_pool_usable": True,
            "canonical_pool_n": 120,
            "protocol_fields_complete": True,
            "adversarial_verify_passed": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
        5225: {
            **_base(
                "experiment_5225_gap4_clean_scale_validation_gated_v478",
                "complete: clean GAP-4 validation null decision.",
                flagged=True,
            ),
            "gap4_clean_validation_complete": True,
            "n_scored": 120,
            "wins": 0,
            "losses": 0,
            "ties": 120,
            "exact_test_passes_min6_rule": False,
            "effect_direction": "null",
            "adversarial_verify_passed": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
        5226: {
            **_base(
                "experiment_5226_veribmc_local_solver_feedback_pilot_v478",
                "complete: clean null; solver feedback did not improve over baselines.",
                flagged=True,
            ),
            "solver_feedback_pilot_complete": _wrap(True),
            "solver_only_solved": _wrap(1),
            "llm_only_solved": _wrap(1),
            "llm_solver_feedback_solved": _wrap(1),
            "solver_feedback_uplift": _wrap(0.0),
            "checker_substrate": _wrap("z3"),
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
        5227: {
            **_base(
                "experiment_5227_continuous_self_learning_multihead_memory_v478",
                "complete: typed memory consumer-ready for exp5228.",
            ),
            "continuous_self_learning_task": _wrap(True),
            "typed_memory_heads": _wrap(list(mod.TYPED_MEMORY_HEADS)),
            "memory_entries_written": _wrap(6),
            "promotions": _wrap(2),
            "rollbacks": _wrap(4),
            "retention_check_passed": _wrap(True),
            "consumer_ready_path": _wrap("results/arc_rubric_setup_from_typed_memory_v478.json"),
            "memory_artifact_path": _wrap("results/typed_multihead_verifier_memory_v478.json"),
            "broad_self_distillation_used": _wrap(False),
        },
        5228: {
            **_base(
                "experiment_5228_arc_provenance_skill_rubric_gate_v478",
                "complete: ARC skill rubric usable; no exp5229 live patch is currently gated.",
            ),
            "arc_skill_rubric_usable": True,
            "recommended_live_patch_available": False,
            "known_arc_nulls_retained": {
                "new_levels_banked": [],
                "reproducible_total_levels_delta": 0,
                "no_arc_solve_claim": True,
            },
            "registry_summary": {"present": True, "reproducible_total_levels": 69},
        },
        5229: {
            "experiment": 5229,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "recommended_live_patch_available actual=False expected=True",
            "duration_s": 0.0,
        },
        5230: {
            **_base(
                "experiment_5230_kan_milp_verifier_certificate_v478",
                "success: tiny KAEM PWA/MILP certificate produced.",
            ),
            "kan_certificate_produced": _wrap(True),
            "certificate_path": _wrap(
                "results/experiment_5230_kan_milp_verifier_certificate_v478.json"
            ),
            "solver_status": "optimal",
        },
        5231: {
            **_base(
                "experiment_5231_hardware_continuity_pbit_boundary",
                "complete_hardware_continuity_pbit_boundary_v478_kv260:reachable_polarfire:reachable_gatemate:blocked_physical_jtag_no_speedup",
            ),
            "kv260_reachable": True,
            "kv260_check_method": "ssh_only",
            "polarfire_reachable": True,
            "gatemate_status": "blocked_physical_jtag",
            "gatemate_idcode_raw": "0xffffffff",
            "pbit_boundary_plan_path": (
                "docs/research-notes/experiment_5231_pbit_boundary_exchange_timing_ratio_plan.md"
            ),
            "speedup_claimed": False,
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    payloads = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, payloads[source.experiment_number])
    _write_json(
        root / "results/typed_multihead_verifier_memory_v478.json",
        {"heads": list(mod.TYPED_MEMORY_HEADS), "entries": [{"promotion_state": "promoted"}]},
    )
    _write_json(
        root / "results/arc_rubric_setup_from_typed_memory_v478.json",
        {"consumer_ready": True, "rubric_fields": list(mod.ARC_RUBRIC_FIELDS)},
    )


def _validation() -> list[dict[str, str]]:
    return [{"command": "focused pytest", "status": "PASS", "notes": "fixture"}]


def _field(field: str, value: Any) -> dict[str, Any]:
    return _wrap(value, mod.FIELD_PRINCIPLES[field])


def test_req_capstone_5232_spec_declares_v478_contract() -> None:
    """REQ-CAPSTONE-5232: OpenSpec declares the V478 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5232") :]

    for marker in (
        "REQ-CAPSTONE-5232",
        "SCENARIO-CAPSTONE-5232",
        "SCENARIO-CAPSTONE-5232-FIELD-PRINCIPLES",
        mod.EXPERIMENT_ID,
        str(mod.RESULT_RELATIVE_PATH),
        "gap1_final_status",
        "solver_feedback_status",
        "docs_reconciled=false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5232_excludes_flagged_and_gated_artifacts(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5232: flagged and gated artifacts cannot become headlines."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260704",
        duration_s=1.25,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        docs_reconciled=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["gap1_final_status"]["value"] == "blocked"
    assert artifact["gap4_final_status"]["value"] == "blocked"
    assert artifact["solver_feedback_status"]["value"] == "blocked"
    assert artifact["continuous_self_learning_satisfied"]["value"] is True
    assert artifact["arc_new_levels_banked"]["value"] == []
    assert artifact["arc_reproducible_total_levels_delta"]["value"] == 0
    assert artifact["kan_certificate_status"]["value"] == "produced"
    assert artifact["flagged_artifacts_excluded"]["value"] is True
    assert artifact["docs_reconciled"]["value"] is False
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is True
    assert artifact["flagged_adversarial"] is False
    assert "KV260=reachable" in artifact["hardware_status"]["value"]
    assert "GateMate=blocked_physical_jtag" in artifact["hardware_status"]["value"]
    assert "no speedup claim" in artifact["hardware_status"]["value"]

    summary = artifact["per_task_summary"]["value"]
    assert len(summary) == 12
    assert summary["exp5225-gap4-clean-scale-validation-gated-v478"]["status"] == "excluded"
    assert summary["exp5226-veribmc-local-solver-feedback-pilot-v478"]["status"] == "excluded"
    assert summary["exp5229-arc-gated-live-levelup-from-rubric-v478"]["status"] == "gate_blocked"
    assert summary["exp5230-kan-milp-verifier-certificate-v478"]["headline_eligible"] is True
    assert (
        "exp5225-gap4-clean-scale-validation-gated-v478"
        in artifact["excluded_from_headline_task_ids"]
    )
    assert (
        "exp5229-arc-gated-live-levelup-from-rubric-v478"
        in artifact["excluded_from_headline_task_ids"]
    )
    assert "exp5230-kan-milp-verifier-certificate-v478" in artifact["headline_eligible_task_ids"]
    assert "flagged/gated artifacts excluded" in artifact["honest_verdict"]


def test_req_capstone_5232_validation_and_status_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5232: validation rejects overclaims and helper edges are explicit."""

    _make_repo(tmp_path, omit={5231})
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260704",
        duration_s=2.0,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        docs_reconciled=False,
    )

    assert artifact["missing_artifacts"] == ["exp5231-hardware-continuity-pbit-boundary-v478"]
    assert artifact["hardware_status"]["value"] == "hardware evidence missing"
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
            artifact | {"field_principles": artifact["field_principles"] | {"hardware_status": "x"}}
        )
    with pytest.raises(ValueError, match="gap1_final_status field principle mismatch"):
        mod.validate_artifact(artifact | {"gap1_final_status": _wrap("blocked")})
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
        mod.validate_artifact(artifact | {"gap1_final_status": _field("gap1_final_status", "open")})
    with pytest.raises(ValueError, match="gap4_final_status"):
        mod.validate_artifact(artifact | {"gap4_final_status": _field("gap4_final_status", "open")})
    with pytest.raises(ValueError, match="solver_feedback_status"):
        mod.validate_artifact(
            artifact | {"solver_feedback_status": _field("solver_feedback_status", "bad")}
        )
    with pytest.raises(ValueError, match="kan_certificate_status"):
        mod.validate_artifact(
            artifact | {"kan_certificate_status": _field("kan_certificate_status", "bad")}
        )
    with pytest.raises(ValueError, match="flagged_artifacts_excluded"):
        mod.validate_artifact(
            artifact | {"flagged_artifacts_excluded": _field("flagged_artifacts_excluded", False)}
        )
    with pytest.raises(ValueError, match="headline eligible"):
        mod.validate_artifact(
            artifact
            | {
                "headline_eligible_task_ids": [
                    *artifact["headline_eligible_task_ids"],
                    "exp5225-gap4-clean-scale-validation-gated-v478",
                ]
            }
        )
    with pytest.raises(ValueError, match="arc_reproducible_total_levels_delta"):
        mod.validate_artifact(
            artifact
            | {
                "arc_new_levels_banked": _field("arc_new_levels_banked", ["L"]),
                "arc_reproducible_total_levels_delta": _field(
                    "arc_reproducible_total_levels_delta", 0
                ),
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
    with pytest.raises(ValueError, match="continuous_self_learning_satisfied"):
        mod.validate_artifact(
            artifact
            | {
                "continuous_self_learning_satisfied": _field(
                    "continuous_self_learning_satisfied", "yes"
                )
            }
        )
    with pytest.raises(ValueError, match="hardware_status"):
        mod.validate_artifact(artifact | {"hardware_status": _field("hardware_status", [])})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "stale"})

    assert mod.source_by_number(5220).task_id == "exp5220-archive-477-activate-478"
    assert mod.value_of(_wrap("x")) == "x"
    assert mod.value_of("x") == "x"
    assert mod.honest_verdict_text({"value": "complete_wrapped"}) == "complete_wrapped"
    assert mod.honest_verdict_text(None) == ""
    assert mod.as_bool(_wrap(True)) is True
    assert mod.as_bool(False) is False
    assert mod.as_number("3.5") == 3.5
    assert mod.as_number("bad") is None
    assert mod.as_number(True) is None
    assert mod.has_critical_corrigendum({"corrigendum_pending": [{"severity": "critical"}]}) is True
    assert mod.has_critical_corrigendum({"corrigendum_pending": [3, {"severity": "warn"}]}) is (
        False
    )
    assert mod.has_critical_corrigendum({"corrigendum_pending": [{"severity": "warn"}]}) is False
    assert mod.exclusion_reasons({"adversarial_verify_passed": False}) == [
        "adversarial_verify_failed"
    ]
    assert (
        mod.is_gate_blocked({"status": "blocked", "blocked_at_layer": "conductor_pre_gate"}) is True
    )
    assert mod.is_gate_blocked({"honest_verdict": "blocked_gate_check_failed"}) is True
    assert mod.is_gate_blocked({"status": "complete"}) is False
    with pytest.raises(KeyError):
        mod.source_by_number(9999)

    assert mod.gap1_final_status({}, set())[0] == "unchanged"
    assert mod.gap1_final_status({5222: {"gap1_registry_promoted": True}}, {5222})[0] == "promoted"
    assert (
        mod.gap1_final_status({5222: {"gap1_registry_decision": "blocked_leakage"}}, {5222})[0]
        == "blocked"
    )
    assert mod.gap1_final_status({5222: {"gap1_registry_decision": "promoted"}}, set())[0] == (
        "blocked"
    )
    assert mod.gap1_final_status({5222: {"gap1_registry_decision": "other"}}, {5222})[0] == (
        "unchanged"
    )

    assert mod.gap4_final_status({}, set())[0] == "unchanged"
    assert mod.gap4_final_status({5225: {"effect_direction": "positive"}}, {5225})[0] == (
        "clean_positive"
    )
    assert mod.gap4_final_status({5225: {"effect_direction": "null"}}, {5225})[0] == "clean_null"
    assert mod.gap4_final_status({5225: {"effect_direction": "blocked"}}, {5225})[0] == "blocked"
    assert mod.gap4_final_status({5225: {"effect_direction": "null"}}, set())[0] == "blocked"

    assert mod.solver_feedback_status({}, set())[0] == "not_run"
    assert (
        mod.solver_feedback_status({5226: {"solver_feedback_uplift": 0.25}}, {5226})[0]
        == "positive"
    )
    assert mod.solver_feedback_status({5226: {"solver_feedback_uplift": 0.0}}, {5226})[0] == "null"
    assert (
        mod.solver_feedback_status({5226: {"solver_feedback_uplift": 0.0}}, set())[0] == "blocked"
    )

    assert mod.continuous_self_learning_satisfied(tmp_path, {}, set())[0] is False
    assert (
        mod.continuous_self_learning_satisfied(
            tmp_path, {5227: {"consumer_ready_path": 3, "memory_artifact_path": 4}}, {5227}
        )[0]
        is False
    )
    bad_self = _payloads()[5227] | {"broad_self_distillation_used": _wrap(True)}
    assert mod.continuous_self_learning_satisfied(tmp_path, {5227: bad_self}, {5227})[0] is False
    assert mod.arc_status({}) == ([], 0, "no ARC level-up artifact or rubric null evidence")
    assert mod.arc_status(
        {5229: {"new_levels_banked": ["L"], "reproducible_total_levels_delta": 1}}
    )[:2] == (["L"], 1)
    assert mod.kan_certificate_status({}, set())[0] == "not_run"
    assert mod.kan_certificate_status({5230: {"kan_certificate_produced": False}}, {5230})[0] == (
        "blocked"
    )
    assert mod.kan_certificate_status({5230: {"kan_certificate_produced": True}}, set())[0] == (
        "blocked"
    )
    assert mod.hardware_status({}) == "hardware evidence missing"
    assert mod.hardware_status({5231: {"kv260_reachable": False}}, excluded_numbers={5231}) == (
        "hardware evidence excluded because Exp5231 is flagged or gate-blocked"
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
        conductor_untouched=True,
        docs_reconciled=False,
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["duration_s"] == 2.5
    assert saved["validation_commands_run"]["value"] == _validation()
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)


def test_req_capstone_5232_repository_artifact_matches_schema() -> None:
    """REQ-CAPSTONE-5232: checked-in artifact is the stable deliverable."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["gap1_final_status"]["value"] == "blocked"
    assert artifact["gap4_final_status"]["value"] == "blocked"
    assert artifact["solver_feedback_status"]["value"] == "blocked"
    assert artifact["continuous_self_learning_satisfied"]["value"] is True
    assert artifact["arc_new_levels_banked"]["value"] == []
    assert artifact["arc_reproducible_total_levels_delta"]["value"] == 0
    assert artifact["kan_certificate_status"]["value"] == "produced"
    assert artifact["flagged_artifacts_excluded"]["value"] is True
    assert artifact["docs_reconciled"]["value"] is False
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")

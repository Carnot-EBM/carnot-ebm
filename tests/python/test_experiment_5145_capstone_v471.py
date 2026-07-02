"""Tests for Exp 5145 V471 capstone aggregation.

Spec refs: REQ-CAPSTONE-5145, SCENARIO-CAPSTONE-5145,
SCENARIO-CAPSTONE-5145-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5145_capstone_v471 as mod
from scripts import experiment_5145_capstone_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base(exp_id: str, verdict: str, *, flagged: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_id": exp_id,
        "milestone": mod.MILESTONE,
        "honest_verdict": verdict,
        "duration_s": 1.0,
        "inference_substrate": "fixture",
        "conductor_modified": False,
    }
    if flagged is not None:
        payload["flagged_adversarial"] = flagged
    return payload


def _payloads() -> dict[int, dict[str, Any]]:
    return {
        5134: {
            **_base("exp5134-archive-470-activate-471", "complete_archive", flagged=True),
            "v470_runtime_clean": True,
            "v470_structured_energy_quarantined": True,
        },
        5135: {
            **_base("exp5135-v471-source-scope-audit", "complete_v471_audit", flagged=True),
            "v471_reference_block_found": True,
            "sota_model_discipline_ok": True,
            "structured_gates_ok": True,
            "fover_same_scope_rerun_found": False,
            "exclusion_manifest_conflicts": [],
        },
        5136: {
            **_base("exp5136-receipt-structured-pool-v2-v471", "complete_pool"),
            "adversarial_verify_passed": True,
            "cheap_baseline_at_1": 0.291667,
            "duplicate_rate": 0.033333,
            "duration_floor_evidence": {"completed": True},
            "exact_validators_used": ["or_allocation", "graph_coloring"],
            "fover_scope_used": False,
            "oracle_at_k": 0.875,
            "parse_coverage": 0.983333,
            "pool_n": 120,
            "structured_pool_v2_clean": True,
        },
        5137: {
            **_base("exp5137-solver-verified-formulation-selector-v471", "complete_selector"),
            "delta_ci95": [0.0, 0.0],
            "feasibility_restoration_used": True,
            "formulation_selector_ready": False,
            "selector_delta_vs_best_static": 0.0,
            "solve_effort_delta": {"delta_units": 16665, "ratio": 12.132265},
            "wrong_label_count": 0,
        },
        5138: {
            **_base("exp5138-ets-ebd-guided-decoding-v471", "blocked_telemetry"),
            "delta_ci95": [None, None],
            "guided_decoding_ready": False,
            "preconditions_checked": {
                "stepwise_telemetry_available": False,
                "structured_pool_v2_clean": True,
            },
        },
        5139: {
            **_base("exp5139-abstention-and-verification-trace-v471", "complete_trace"),
            "coverage_risk_curve": [{"coverage": 0.858333, "harmful_answer_rate": 0.05}],
            "harmful_answer_reduction": 0.929412,
            "verification_trace_ready": True,
        },
        5140: {
            **_base("exp5140-symbolic-kan-certificate-distillation-v471", "success_kan"),
            "certificate_soundness": True,
            "cycle_reconstruction_rate": 1.0,
            "false_property_detected": True,
            "label_shuffle_control": {"detected": True},
            "symbolic_equivalence_rate": 1.0,
            "symbolic_kan_ready": True,
        },
        5141: {
            **_base(
                "exp5141-hubo-partition-residual-exponent-v471", "complete_partition", flagged=False
            ),
            "board_ready_workload_descriptors": [{"target_board": "kv260"} for _ in range(3)],
            "effective_sample_quality": {
                "monolithic_reference": 1.0,
                "unguided_baseline": 0.167635,
            },
            "exact_enumeration_checked": True,
            "hardware_speedup_claimed": False,
            "partition_telemetry_ready": True,
            "telemetry_stability": {"stable_enough_for_hardware_transcript_task": True},
        },
        5142: {
            **_base("exp5142-taco-harm-rootcause-scale-v471", "success_taco"),
            "average_effort_reduction_ratio_guarded": 0.463956,
            "harmful_instance_count_guarded": 41,
            "harmful_instance_root_causes": [{"root_cause_id": "dense_high_branching_symmetry"}],
            "repaired_harm_gate": {"rejected_sampler_feature_count": 28},
            "trace_suite_v2_ready": True,
            "wrong_label_count": 0,
        },
        5143: {
            **_base("exp5143-openskill-k2v-self-learning-v471", "success_fr11", flagged=True),
            "continuous_self_learning_task": True,
            "heldout_delta": 0.017003,
            "nonforgetting_delta": 0.102676,
            "no_weight_update": True,
            "promotion_blockers": [],
            "promotion_safe": True,
            "rollback_receipt": {"rollback_available": True, "rollback_applied": False},
            "virtual_task_manifest": {"exact_validated_task_count": 19, "wrong_label_count": 0},
            "wrong_label_count": 0,
        },
        5144: {
            **_base("exp5144-authenticated-board-workload-v471", "blocked_board"),
            "board_blockers": {
                "gatemate": ["blocked_gatemate_dirtyjtag_not_detected"],
                "kv260": ["no_safe_kv260_workload_manifest"],
                "polarfire": ["no_safe_polarfire_workload_manifest"],
            },
            "extropic_tsu_execution_claimed": False,
            "hardware_workload_transcripts_ready": False,
            "kv260_host_block_devices_touched": False,
            "kv260_ssh_checked": True,
            "no_speedup_claim": True,
            "safe_workload_manifest": {"present": False},
            "sample_quality_evidence": {"ready_evidence_boards": []},
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    payloads = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, payloads[source.experiment_number])


def test_req_capstone_5145_spec_declares_v471_capstone_contract() -> None:
    """REQ-CAPSTONE-5145: OpenSpec declares the V471 capstone fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5145") :]

    for marker in (
        "REQ-CAPSTONE-5145",
        "SCENARIO-CAPSTONE-5145",
        "SCENARIO-CAPSTONE-5145-FIELD-PRINCIPLES",
        mod.EXPERIMENT_ID,
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5145_classifies_v471_evidence(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5145: capstone separates clean, blocked, and quarantined axes."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=1.25,
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert len(artifact["upstream_artifacts_read"]) == len(mod.UPSTREAM_SOURCES)
    assert artifact["missing_artifacts"] == []
    assert artifact["conductor_modified"] is False

    assert artifact["source_scope_audit_state"]["classification"] == "quarantined"
    assert artifact["source_scope_audit_state"]["sota_model_discipline_ok"] is True
    assert artifact["structured_generation_state"]["classification"] == "clean"
    assert artifact["structured_generation_state"]["provenance_problem_repaired"] is True
    assert artifact["structured_generation_state"]["downstream_tasks_trustworthy"] is True
    assert artifact["solver_formulation_state"]["classification"] == "no-promote"
    assert artifact["solver_formulation_state"]["selector_delta_vs_best_static"] == 0.0
    assert artifact["guided_decoding_state"]["classification"] == "blocked"
    assert (
        artifact["guided_decoding_state"]["gate_condition"]["field"]
        == "stepwise_telemetry_available"
    )
    assert artifact["abstention_trace_state"]["classification"] == "clean"
    assert artifact["kan_symbolic_state"]["classification"] == "clean"
    assert artifact["sampling_partition_state"]["classification"] == "clean"
    assert artifact["taco_harm_state"]["classification"] == "clean"
    assert artifact["fr11_state"]["classification"] == "quarantined"
    assert artifact["fr11_state"]["promotion_assessment"] == "safe_promotion_evidence_quarantined"
    assert artifact["hardware_state"]["classification"] == "blocked"
    assert artifact["hardware_state"]["hardware_workload_transcripts_ready"] is False
    assert artifact["no_speedup_claim_preserved"] is True

    recommendations = json.dumps(artifact["retire_or_quarantine_recommendations"], sort_keys=True)
    assert "exp5137" in recommendations
    assert "exp5143" in recommendations
    assert "exp5144" in recommendations


def test_req_capstone_5145_missing_malformed_and_validation_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5145: missing artifacts and schema drift fail closed."""

    _make_repo(tmp_path, omit={5138})
    bad_path = tmp_path / mod.EXPECTED_UPSTREAMS[5144].relative_path
    bad_path.write_text("{bad json", encoding="utf-8")

    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=2.0,
        tests_run=["focused"],
    )

    assert {row["experiment_number"] for row in artifact["missing_artifacts"]} == {5138, 5144}
    assert artifact["guided_decoding_state"]["classification"] == "blocked"
    assert artifact["hardware_state"]["classification"] == "blocked"
    assert artifact["hardware_state"]["load_error"] == "malformed_json"
    assert artifact["no_speedup_claim_preserved"] is True
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(
            {key: value for key, value in artifact.items() if key != "duration_s"}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "done"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "live_generation"})
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(artifact | {"conductor_modified": True})
    with pytest.raises(ValueError, match="no_speedup_claim_preserved"):
        mod.validate_artifact(artifact | {"no_speedup_claim_preserved": False})
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(artifact | {"tests_run": []})
    with pytest.raises(ValueError, match="field principle mismatch"):
        mod.validate_artifact(
            artifact
            | {"field_principles": artifact["field_principles"] | {"tests_run": "unverified"}}
        )
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "stale"})

    out_path = mod.run(root=tmp_path, run_date="20260702", duration_s=2.5, tests_run=["run"])
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["duration_s"] == 2.5
    assert saved["tests_run"] == ["run"]
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)

    assert (
        script_mod.main(root=tmp_path, date="20260702", duration_s=3.0, tests_run=["script"])
        == out_path
    )
    assert (
        script_mod.main(
            ["--root", str(tmp_path), "--date", "20260702"],
            duration_s=3.5,
            tests_run=["script-argv"],
        )
        == out_path
    )

    assert mod._number(None) is None
    assert mod._number(True) is None
    assert mod._number("4.5") == 4.5
    assert mod._missing_for([], 999)["error"] == "artifact_missing_or_unreadable"

    missing_rows = [
        {
            "experiment_number": number,
            "classification": "blocked",
            "error": "missing",
        }
        for number in (5135, 5136, 5137, 5139, 5140, 5141, 5142, 5143)
    ]
    for state_fn in (
        mod.source_scope_audit_state,
        mod.structured_generation_state,
        mod.solver_formulation_state,
        mod.abstention_trace_state,
        mod.kan_symbolic_state,
        mod.sampling_partition_state,
        mod.taco_harm_state,
        mod.fr11_state,
    ):
        assert state_fn({}, missing_rows)["classification"] == "blocked"

    clean_fr11 = _payloads()[5143] | {"flagged_adversarial": False}
    assert mod.fr11_state({5143: clean_fr11}, [])["promotion_assessment"] == "safe_promotion"
    no_promote_fr11 = clean_fr11 | {"heldout_delta": 0.0, "promotion_safe": False}
    no_promote_state = mod.fr11_state({5143: no_promote_fr11}, [])
    assert no_promote_state["promotion_assessment"] == "rollback_or_no_promote"
    assert no_promote_state["classification"] == "no-promote"

    assert mod.no_speedup_claim_preserved({1: []}) is True
    assert mod.no_speedup_claim_preserved({1: {"hardware_speedup_claimed": True}}) is False
    assert mod.no_speedup_claim_preserved({1: {"no_speedup_claim": False}}) is False
    assert mod.no_speedup_claim_preserved({1: {"extropic_tsu_execution_claimed": True}}) is False

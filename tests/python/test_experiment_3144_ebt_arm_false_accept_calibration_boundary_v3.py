"""Tests for Exp 3144 EBT/ARM false-accept calibration boundary v3.

Spec refs: REQ-VERIFY-3144, SCENARIO-VERIFY-3144.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import ebt_arm_false_accept_calibration_boundary_v3 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "ebt_arm_false_accept_calibration_v3_ready",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "false_accept_rows_evaluated",
    "abstention_feature_candidates",
    "false_accept_separation_metrics",
    "approximation_gap_summary",
    "model_identity_confound_audit",
    "live_integration",
    "integration_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _live_row(
    row_id: str,
    exact_label: str,
    expected_action: str,
    live_decision: str,
    *,
    family: str = "arithmetic_code_assertions",
) -> dict[str, Any]:
    return {
        "fixture_id": row_id,
        "exact_label": exact_label,
        "expected_action": expected_action,
        "live_decision": live_decision,
        "extracted_answer": "VALID" if live_decision == "accept" else exact_label,
        "fixture_family": family,
        "task_family": family,
        "model_id": GEMMA26,
        "model_hash": "unit-model-hash",
        "live_correct": expected_action == live_decision,
        "exact_answer_match": expected_action == live_decision,
    }


def _sidecar_row(
    row_id: str,
    exact_outcome: str,
    energy: float,
    penalty: float,
    confidence: float,
    *,
    family: str = "arithmetic_code_assertions",
) -> dict[str, Any]:
    return {
        "fixture_id": row_id,
        "task_family": family,
        "exact_outcome": exact_outcome,
        "expected_action": "accept" if exact_outcome == "accepted" else "reject",
        "sidecar_action": "accept" if exact_outcome == "accepted" else "reject",
        "reject_or_repair_label": 0 if exact_outcome == "accepted" else 1,
        "label_blind_feature_energy": energy,
        "feature_summary": {
            "label_blind_violation": penalty,
            "surface_complexity": 0.37,
            "uses_exact_label_reference_for_score": False,
        },
        "replay_score": {
            "confidence": confidence,
            "total_energy": energy,
            "energy_terms": [
                {"name": "constraint_violation_energy", "weighted_value": penalty},
                {"name": "confidence_energy", "weighted_value": round(1.0 - confidence, 6)},
            ],
        },
    }


def _write_sources(root: Path, *, include_false_accept_ids: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("headline results need live provenance\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "Distributional EBM uncertainty\n", encoding="utf-8"
    )
    model_specs = [
        {
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "present": False,
            "selected": False,
            "cache_status": "missing",
            "role": "flagship_moe",
            "legacy_small_model": False,
        },
        {
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "present": False,
            "selected": False,
            "cache_status": "missing",
            "role": "flagship_dense",
            "legacy_small_model": False,
        },
        {
            "hf_id": GEMMA26,
            "present": True,
            "selected": True,
            "cache_status": "resolved",
            "role": "middle_moe",
            "legacy_small_model": False,
        },
    ]
    live_rows = [
        _live_row("accept-valid", "VALID", "accept", "accept"),
        _live_row("reject-invalid", "INVALID", "reject", "reject"),
        _live_row("repair-json", "REPAIRABLE", "reject", "reject", family="repairable"),
        _live_row("fa-arith", "INVALID", "reject", "accept"),
        _live_row("fa-smt", "UNSAT", "reject", "accept", family="smt_constraints"),
    ]
    _write_json(
        root,
        mod.EXP3124_REL_PATH,
        {
            "artifact": "experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6",
            "difficulty_stratified_live_sota_panel_v6_ready": True,
            "selected_model_ids": [GEMMA26],
            "model_specs": model_specs,
            "live_call_count": len(live_rows),
            "false_accept_rate": 0.5,
            "live_rows": live_rows,
            "inference_substrate": {
                "executes_models": True,
                "loads_model_weights": True,
                "new_live_model_calls": len(live_rows),
            },
        },
    )
    _write_json(
        root,
        mod.EXP3130_REL_PATH,
        {
            "artifact": "experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2",
            "arm_ebt_energy_budget_sidecar_v2_ready": True,
            "model_specs": model_specs,
            "selected_model_ids": [GEMMA26],
            "live_call_count": len(live_rows),
            "live_integration": False,
            "integration_blockers": ["no generation-path sidecar hook exercised under tests"],
            "inference_substrate": {"new_live_model_calls": 0},
        },
    )
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_autopsy_v1_ready": include_false_accept_ids,
            "false_accept_row_ids": ["fa-arith", "fa-smt"] if include_false_accept_ids else [],
            "false_accept_rows": [{"row_id": "fa-arith"}, {"row_id": "fa-smt"}],
            "source_false_accept_rate": 0.5,
        },
    )
    _write_json(
        root,
        mod.EXP3117_REL_PATH,
        {
            "artifact": "experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3",
            "sidecar_score_correlation_boundary_v3_ready": True,
            "diagnostic_rows": [
                _sidecar_row("accept-valid", "accepted", 0.31, 0.0, 0.72),
                _sidecar_row("reject-invalid", "rejected", 2.66, 1.02, 0.42),
                _sidecar_row("repair-json", "repairable", 2.60, 1.00, 0.43, family="repairable"),
                _sidecar_row("fa-arith", "rejected", 2.71, 1.04, 0.41),
                _sidecar_row("fa-smt", "rejected", 4.74, 2.00, 0.30, family="smt_constraints"),
            ],
        },
    )


def test_req_verify_3144_spec_anchor_exists() -> None:
    """REQ-VERIFY-3144: OpenSpec declares the calibration boundary first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3144" in spec
    assert "SCENARIO-VERIFY-3144" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "false_accept_separation_metrics" in spec
    assert "live_integration=false" in spec


def test_scenario_verify_3144_builds_false_accept_calibration(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3144: sidecar fields are compared against false accepts."""

    _write_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=13.25,
        tests_run=["focused-unit"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["ebt_arm_false_accept_calibration_v3_ready"] is True
    assert artifact["selected_model_ids"] == [GEMMA26]
    assert artifact["live_call_count"] == 5
    assert artifact["false_accept_rows_evaluated"] == 2
    assert artifact["live_integration"] is False
    assert artifact["false_accept_row_ids"] == ["fa-arith", "fa-smt"]
    penalty = artifact["false_accept_separation_metrics"]["deterministic_constraint_penalty"]
    assert penalty["false_accept"]["mean"] == pytest.approx(1.52)
    assert penalty["non_false_accept"]["mean"] == pytest.approx(0.673333)
    assert penalty["false_accept_recall_at_threshold"] == pytest.approx(1.0)
    assert penalty["non_false_flagged_at_threshold_count"] == 0
    assert penalty["threshold_direction"] == ">="
    quality = artifact["false_accept_separation_metrics"]["quality_proxy"]
    assert quality["threshold_direction"] == "<="
    assert quality["false_accept_recall_at_threshold"] == pytest.approx(1.0)
    assert artifact["approximation_gap_summary"]["false_accept"]["count"] == 2
    assert artifact["approximation_gap_summary"]["accepted_energy_boundary"] == pytest.approx(0.31)
    assert artifact["approximation_gap_summary"]["false_accept_below_accepted_boundary_count"] == 0
    candidates = {row["field"]: row for row in artifact["abstention_feature_candidates"]}
    assert candidates["deterministic_constraint_penalty"]["would_flag_false_accept_rows"] is True
    assert candidates["deterministic_constraint_penalty"]["live_contract_eligible"] is False
    assert (
        "missing_generation_path_integration_test"
        in candidates["deterministic_constraint_penalty"]["blocking_contract_requirements"]
    )
    assert artifact["model_identity_confound_audit"]["single_model_trace_only"] is True
    assert artifact["model_identity_confound_audit"]["model_id_used_in_sidecar_features"] is False
    assert "single selected-model trace confound" in artifact["integration_blockers"]
    assert artifact["inference_substrate"]["new_live_model_calls"] == 0
    assert artifact["inference_substrate"]["upstream_live_trace_count"] == 5
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["tests_run"] == ["focused-unit"]
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)

    relative_output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        started_s=20.0,
        now_s=21.0,
        tests_run=["relative-output"],
    )
    assert relative_output == tmp_path / mod.OUTPUT_REL_PATH


def test_req_verify_3144_blocks_without_false_accept_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-3144: missing regression rows fail closed without live integration."""

    _write_sources(tmp_path, include_false_accept_ids=False)
    blocked = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.5, tests_run=["blocked"])

    assert blocked["ebt_arm_false_accept_calibration_v3_ready"] is False
    assert blocked["false_accept_rows_evaluated"] == 0
    assert blocked["live_integration"] is False
    assert blocked["honest_verdict"].startswith("blocked_")
    assert "false_accept_ids_present" in blocked["blocked_reasons"]
    mod.validate_artifact(blocked)

    empty = mod.build_artifact(tmp_path / "empty", started_s=1.0, now_s=2.0)
    assert empty["honest_verdict"].startswith("blocked_missing_trace_source")
    assert mod.read_json_object(tmp_path / "does-not-exist.json") == {}

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_integration"):
        mod.validate_artifact(blocked | {"live_integration": True})
    with pytest.raises(ValueError, match="new_live_model_calls"):
        mod.validate_artifact(
            blocked
            | {"inference_substrate": blocked["inference_substrate"] | {"new_live_model_calls": 1}}
        )
    with pytest.raises(ValueError, match="integration_blockers"):
        mod.validate_artifact(blocked | {"integration_blockers": []})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked | {"honest_verdict": "ready"})

    assert mod.numeric_summary([]) == {"count": 0, "finite": False}
    assert mod.relative_path(tmp_path, tmp_path / "nested" / "artifact.json") == (
        "nested/artifact.json"
    )
    assert mod.relative_path(tmp_path, Path("/outside/artifact.json")) == "/outside/artifact.json"
    assert mod.joined_calibration_rows(
        [{"fixture_id": "missing-sidecar"}], {}, ["missing-sidecar"]
    ) == []
    assert mod.integration_blockers(
        [],
        {"single_model_trace_only": False, "legacy_small_model_selected": True},
        [],
        ["missing-sidecar"],
    ) == [
        "no generation-path sidecar hook exercised under tests",
        "no Exp3144 live generation or abstention integration test",
        "no trained EBT/ARM learned quality head available",
        "no per-token live energy budget or logprob trace in Exp3144",
        "exact-safe threshold not validated on unseen live rows",
        "no abstention feature candidate evaluated",
        "not all Exp3136 false-accept row IDs joined to sidecar diagnostics",
        "legacy small model selected",
    ]
    assert mod.honest_verdict(
        {
            "ebt_arm_false_accept_calibration_v3_ready": False,
            "false_accept_rows_evaluated": 1,
            "live_call_count": 2,
            "blocked_reasons": ["unit_blocker"],
        }
    ) == "blocked_incomplete_calibration: unit_blocker"
    assert mod.scale01([2.0, 2.0]) == [0.0, 0.0]
    assert mod.rate(1, 0) == 0.0
    assert mod.confound_risk({"a": 1, "b": 1}, ["a", "b"]) == "lower_multiple_model_traces"

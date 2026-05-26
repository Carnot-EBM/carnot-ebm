"""Tests for Exp 3158 EBCN energy sidecar calibration v1.

Spec refs: REQ-VERIFY-3158, SCENARIO-VERIFY-3158.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import ebcn_energy_sidecar_calibration_v1 as mod


REQUIRED_FIELDS = {
    "ebcn_energy_sidecar_calibration_v1_ready",
    "exact_labeled_row_count",
    "known_false_accept_rows_scored",
    "scalar_energy_auc",
    "violation_localization_coverage",
    "scale_compatibility_notes",
    "live_integration_claim_allowed",
    "residual_blockers",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _event_ledger(
    row_id: str,
    *,
    action: str,
    constraint_status: str | None = None,
) -> dict[str, Any]:
    constraints = []
    if constraint_status is not None:
        constraints.append(
            {
                "constraint_id": f"{row_id}:assertion",
                "status": constraint_status,
                "failing_constraint": "claimed_value == computed_value"
                if constraint_status == "fail"
                else "",
            }
        )
    return {
        "event_index": 1,
        "event_type": "constraint_ledger",
        "payload": {
            "ledger_action": action,
            "ledger_source": "fragment_checks" if constraints else "exact_label_fallback",
            "constraints": constraints,
        },
    }


def _event_final(row_id: str, *, expected: str, live: str, consistent: bool) -> dict[str, Any]:
    return {
        "event_index": 2,
        "event_type": "candidate_final_answer",
        "payload": {
            "expected_action": expected,
            "live_decision": live,
            "final_answer_consistent_with_exact": consistent,
            "final_answer_consistent_with_ledger": consistent,
            "prompt_hash": f"{row_id}-prompt",
        },
    }


def _verifier_row(
    row_id: str,
    exact_label: str,
    expected_action: str,
    live_decision: str,
    buckets: list[str],
    *,
    constraint_status: str | None = None,
    family: str = "arithmetic_code_assertions",
) -> dict[str, Any]:
    consistent = expected_action == live_decision
    return {
        "row_id": row_id,
        "exact_label": exact_label,
        "expected_action": expected_action,
        "live_decision": live_decision,
        "extracted_answer": exact_label if consistent else "VALID",
        "difficulty_buckets": buckets,
        "fixture_family": family,
        "failure_mechanism_from_exp3124": "contradiction"
        if "contradiction" in buckets and not consistent
        else "no_failure",
        "monitor_events": [
            _event_ledger(row_id, action=expected_action, constraint_status=constraint_status),
            _event_final(
                row_id,
                expected=expected_action,
                live=live_decision,
                consistent=consistent,
            ),
        ],
    }


def _calibration_row(
    row_id: str,
    exact_label: str,
    exact_outcome: str,
    expected_action: str,
    live_decision: str,
    energy: float,
    penalty: float,
    quality: float,
    *,
    false_accept: bool = False,
    family: str = "arithmetic_code_assertions",
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "exact_label": exact_label,
        "exact_outcome": exact_outcome,
        "expected_action": expected_action,
        "live_decision": live_decision,
        "false_accept": false_accept,
        "fixture_family": family,
        "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "deterministic_constraint_penalty": penalty,
        "final_energy_proxy": energy,
        "quality_proxy": quality,
        "uncertainty_proxy": round(1.0 - quality, 6),
        "approximation_gap_to_exact_binary": 0.0 if false_accept else 0.25,
        "uses_exact_label_reference_for_score": False,
    }


def _write_sources(root: Path, *, include_false_ids: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("headline results need live provenance\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "Energy-Based Constraint Networks localize violations\n", encoding="utf-8"
    )
    spec = root / "openspec/capabilities/verification/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-VERIFY-3158\nSCENARIO-VERIFY-3158\n"
        f"{mod.OUTPUT_REL_PATH.as_posix()}\n"
        "scalar_energy_auc\nviolation_localization_coverage\n"
        "live_integration_claim_allowed=false\n",
        encoding="utf-8",
    )
    calibration_rows = [
        {"row_id": "", "exact_label": ""},
        _calibration_row("clean-a", "VALID", "accepted", "accept", "accept", 0.30, 0.0, 0.72),
        _calibration_row("clean-b", "VALID", "accepted", "accept", "accept", 0.40, 0.0, 0.68),
        _calibration_row("reject-a", "INVALID", "rejected", "reject", "reject", 2.00, 1.0, 0.40),
        _calibration_row(
            "fa-a", "INVALID", "rejected", "reject", "accept", 3.00, 1.2, 0.35, false_accept=True
        ),
        _calibration_row(
            "repair-a",
            "REPAIRABLE",
            "repairable",
            "reject",
            "reject",
            2.50,
            1.0,
            0.42,
            family="repairable_invalid_candidates",
        ),
        _calibration_row(
            "fa-b",
            "UNSAT",
            "rejected",
            "reject",
            "accept",
            5.00,
            2.0,
            0.20,
            false_accept=True,
            family="smt_constraints",
        ),
    ]
    verifier_rows = [
        _verifier_row("clean-a", "VALID", "accept", "accept", ["easy", "satisfiable_drift"]),
        _verifier_row("clean-b", "VALID", "accept", "accept", ["easy", "satisfiable_drift"]),
        _verifier_row("reject-a", "INVALID", "reject", "reject", ["easy", "contradiction"]),
        _verifier_row("fa-a", "INVALID", "reject", "accept", ["easy", "contradiction"]),
        _verifier_row(
            "repair-a",
            "REPAIRABLE",
            "reject",
            "reject",
            ["hard", "fragment_code"],
            constraint_status="fail",
            family="repairable_invalid_candidates",
        ),
        _verifier_row(
            "fa-b",
            "UNSAT",
            "reject",
            "accept",
            ["medium", "contradiction"],
            family="smt_constraints",
        ),
    ]
    _write_json(
        root,
        mod.EXP3144_REL_PATH,
        {
            "artifact": "experiment_3144_ebt_arm_false_accept_calibration_boundary_v3",
            "ebt_arm_false_accept_calibration_v3_ready": True,
            "live_call_count": 6,
            "calibration_rows": calibration_rows,
            "model_identity_confound_audit": {"single_model_trace_only": True},
            "inference_substrate": {"new_live_model_calls": 0},
        },
    )
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_autopsy_v1_ready": True,
            "false_accept_row_ids": ["fa-a", "fa-b"] if include_false_ids else [],
            "verifier_rows": verifier_rows,
            "false_accept_rows": [
                row for row in verifier_rows if row["row_id"] in {"fa-a", "fa-b"}
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "artifact": "experiment_3137_exact_safe_accept_abstain_contract_v1",
            "acceptance_contract_v1_ready": True,
            "replay_rows": [
                {
                    "row_id": row["row_id"],
                    "decision": "abstain"
                    if row["row_id"].startswith("fa-")
                    else row["expected_action"],
                }
                for row in verifier_rows
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3138_REL_PATH,
        {
            "artifact": "experiment_3138_canonical_answer_vericot_grounding_pilot_v1",
            "canonical_grounding_pilot_v1_ready": True,
            "regression_row_replay": [
                {"row_id": "fa-a", "blocked_by": ["canonicalization", "ledger_replay"]},
                {"row_id": "fa-b", "blocked_by": ["canonicalization", "ledger_replay"]},
            ],
        },
    )


def test_req_verify_3158_spec_anchor_exists() -> None:
    """REQ-VERIFY-3158: OpenSpec declares the diagnostic boundary first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3158" in spec
    assert "SCENARIO-VERIFY-3158" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "scalar_energy_auc" in spec
    assert "violation_localization_coverage" in spec
    assert "live_integration_claim_allowed=false" in spec


def test_scenario_verify_3158_builds_diagnostic_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3158: exact rows get bounded energy and localization."""

    _write_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=11.5,
        tests_run=["focused-unit"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["ebcn_energy_sidecar_calibration_v1_ready"] is True
    assert artifact["exact_labeled_row_count"] == 6
    assert artifact["known_false_accept_rows_scored"] == 2
    assert artifact["scalar_energy_auc"] == pytest.approx(1.0)
    assert artifact["violation_localization_coverage"] == pytest.approx(1.0)
    assert artifact["live_integration_claim_allowed"] is False
    assert artifact["tests_run"] == ["focused-unit"]
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["inference_substrate"]["new_live_model_calls"] == 0
    assert artifact["label_leakage_audit"]["uses_exact_label_for_scalar_energy"] is False
    assert "approximation_gap_to_exact_binary" in artifact["label_leakage_audit"]["excluded_fields"]
    assert (
        "approximation_gap_to_exact_binary"
        not in artifact["scalar_energy_definition"]["score_inputs"]
    )
    assert artifact["row_category_counts"]["known_false_accept"] == 2
    assert artifact["row_category_counts"]["clean_accept"] == 2
    assert artifact["row_category_counts"]["contradiction"] == 3
    assert artifact["row_category_counts"]["satisfiable_drift"] == 2
    assert any("bounded" in note for note in artifact["scale_compatibility_notes"])
    assert "no live verifier integration implemented or exercised" in artifact["residual_blockers"]
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)

    rows = artifact["calibration_rows"]
    clean_max = max(row["scalar_energy"] for row in rows if "clean_accept" in row["categories"])
    false_rows = [row for row in rows if row["known_false_accept"] is True]
    assert all(row["scalar_energy"] > clean_max for row in false_rows)
    assert all(row["violation_localization"] for row in false_rows)
    assert all(0.0 <= row["scalar_energy"] <= 1.0 for row in rows)
    assert all(0.0 <= branch["value"] <= 1.0 for row in rows for branch in row["energy_branches"])
    assert all(
        source["exists"] is True for source in artifact["source_artifacts"] if source["required"]
    )

    relative_output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        started_s=20.0,
        now_s=21.0,
        tests_run=["relative-output"],
    )
    assert relative_output == tmp_path / mod.OUTPUT_REL_PATH


def test_req_verify_3158_blocks_and_validates_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3158: missing evidence cannot become a live integration claim."""

    empty = mod.build_artifact(tmp_path / "empty", started_s=0.0, now_s=0.5, tests_run=["empty"])
    assert empty["ebcn_energy_sidecar_calibration_v1_ready"] is False
    assert empty["exact_labeled_row_count"] == 0
    assert empty["known_false_accept_rows_scored"] == 0
    assert empty["scalar_energy_auc"] == pytest.approx(0.0)
    assert empty["violation_localization_coverage"] == pytest.approx(0.0)
    assert empty["honest_verdict"].startswith("blocked_missing_exact_evidence")
    mod.validate_artifact(empty)

    _write_sources(tmp_path, include_false_ids=False)
    blocked = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0, tests_run=["blocked"])
    assert blocked["ebcn_energy_sidecar_calibration_v1_ready"] is False
    assert "known_false_accept_rows_complete" in blocked["blocked_reasons"]
    assert blocked["honest_verdict"].startswith("blocked_incomplete_calibration")
    mod.validate_artifact(blocked)

    missing_false_root = tmp_path / "missing-false"
    _write_sources(missing_false_root)
    exp3136_path = missing_false_root / mod.EXP3136_REL_PATH
    exp3136 = json.loads(exp3136_path.read_text(encoding="utf-8"))
    exp3136["false_accept_row_ids"].append("not-present")
    exp3136_path.write_text(json.dumps(exp3136), encoding="utf-8")
    missing_false = mod.build_artifact(missing_false_root, started_s=2.0, now_s=3.0)
    assert "not all known false accepts were scored" in missing_false["residual_blockers"]

    invalid = tmp_path / mod.EXP3136_REL_PATH
    invalid.write_text("not-json\n", encoding="utf-8")
    assert mod.read_json_object(invalid) == {}

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_integration_claim_allowed"):
        mod.validate_artifact(blocked | {"live_integration_claim_allowed": True})
    with pytest.raises(ValueError, match="new_live_model_calls"):
        mod.validate_artifact(
            blocked
            | {"inference_substrate": blocked["inference_substrate"] | {"new_live_model_calls": 1}}
        )
    with pytest.raises(ValueError, match="residual_blockers"):
        mod.validate_artifact(blocked | {"residual_blockers": []})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="scalar_energy_auc"):
        mod.validate_artifact(blocked | {"scalar_energy_auc": 2.0})
    with pytest.raises(ValueError, match="violation_localization_coverage"):
        mod.validate_artifact(blocked | {"violation_localization_coverage": -0.1})

    assert mod.bounded_unit(float("nan")) == 0.0
    assert mod.rate(1, 0) == 0.0
    assert mod.auc([], [0.1]) == 0.0
    assert mod.auc([0.5], []) == 0.0
    assert mod.auc([0.5, 0.8], [0.5, 0.1]) == pytest.approx(0.875)
    assert mod.violation_localization_coverage([]) == 0.0
    assert (
        mod.relative_path(tmp_path, tmp_path / "nested" / "artifact.json") == "nested/artifact.json"
    )
    assert mod.relative_path(tmp_path, Path("/outside/artifact.json")) == "/outside/artifact.json"

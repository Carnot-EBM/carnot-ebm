"""Tests for Exp 5133 .470 ungated capstone aggregation.

Spec refs: REQ-CAPSTONE-5133, SCENARIO-CAPSTONE-5133,
SCENARIO-CAPSTONE-5133-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5133_capstone_v470 as exp
from scripts import experiment_5133_capstone_v470 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(flags: list[dict] | None = None) -> dict:
    flags = flags or []
    severity_rank = {"info": 0, "warn": 1, "critical": 2}
    return {
        "loaded": True,
        "flag_count": len(flags),
        "max_severity": max(
            (severity_rank.get(flag["severity"], -1) for flag in flags), default=-1
        ),
        "flags": flags,
    }


CRITICAL_DURATION_REPORT = _report(
    [
        {
            "kind": "DURATION_TOO_SHORT",
            "severity": "critical",
            "detail": "duration too short for claimed live GGUF path",
        }
    ]
)
INFO_ZERO_REPORT = _report(
    [
        {
            "kind": "IMPLAUSIBLE_PERFECT",
            "severity": "info",
            "detail": "heldout_delta=0.0 is an honest no-promote null",
        }
    ]
)
CLEAN_REPORT = _report()


def _reporter(path: Path) -> dict:
    name = path.name
    if "5123" in name or "5125" in name or "5126" in name:
        return CRITICAL_DURATION_REPORT
    if "5131" in name:
        return INFO_ZERO_REPORT
    return CLEAN_REPORT


def _base_payload(exp_id: str, verdict: str = "complete_placeholder") -> dict:
    return {
        "experiment_id": exp_id,
        "milestone": exp.MILESTONE,
        "honest_verdict": verdict,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 1.0,
        "flagged_adversarial": False,
    }


def _upstream_payloads() -> dict[int, dict]:
    payloads = {
        5122: {
            **_base_payload(
                "exp5122-archive-469-activate-470",
                "complete_archive_469_closed_470_active_roadmap_ready_fover_retired",
            ),
            "fover_selector_retired_for_same_verdict": True,
            "fover_retirement": {"fover_residual_fr11_should_not_rerun": True},
        },
        5123: {
            **_base_payload(
                "exp5123-v470-source-scope-audit",
                "complete_v470_source_scope_audit_clean",
            ),
            "fover_same_scope_rerun_found": False,
            "sota_model_discipline_ok": True,
        },
        5124: {
            **_base_payload(
                "exp5124-clean-sota-runtime-provenance-v470",
                "success_clean_sota_runtime_provenance_ready",
            ),
            "adversarial_verify_passed": True,
            "cache_ready": True,
            "completion_proof": {"ready": True},
            "endpoint_lifetime_s": True,
            "logprob_proof": {"ready": True},
            "sota_runtime_clean": True,
        },
        5125: {
            **_base_payload(
                "exp5125-structured-reasoning-pool-v470",
                "complete_structured_reasoning_pool_ready",
            ),
            "cheap_baseline_at_1": 0.291667,
            "duplicate_rate": 0.033854,
            "flagged_adversarial": True,
            "fover_scope_used": False,
            "oracle_at_k": 0.875,
            "parse_coverage": 0.984375,
            "pool_n": 96,
            "structured_pool_ready": True,
        },
        5126: {
            **_base_payload(
                "exp5126-distributional-energy-ranker-v470",
                "complete_distributional_energy_ranker_evaluated_not_ready_for_audit",
            ),
            "distributional_energy_delta": 0.0,
            "flagged_adversarial": True,
            "ranker_ready_for_audit": False,
            "ranker_metrics": {"accuracy_at_1": 0.5, "auroc": 1.0},
            "strongest_cheap_baseline": {"name": "constraint_count_only", "accuracy_at_1": 0.5},
        },
        5127: {
            **_base_payload("exp5127-gate", "blocked_gate_check_failed"),
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "distributional_energy_delta gate failed",
            "gates_evaluated": [{"upstream": "exp5126", "passed": False}],
            "status": "blocked",
        },
        5128: {
            **_base_payload(
                "exp5128-kan-certificate-explanation-v470",
                "success_kan_certificate_explanation_cycle_sound_breadth_ready",
            ),
            "certificate_soundness": True,
            "explanation_cycle_soundness": True,
            "false_property_detected": True,
            "kan_certificate_breadth_ready": True,
            "near_margin_abstained": True,
            "property_families": [{"family": "global_energy_upper_bound"}],
        },
        5129: {
            **_base_payload(
                "exp5129-hubo-adaptive-2dpt-v470",
                "complete_adaptive_2dpt_ready_exact_checked_cpu",
            ),
            "adaptive_2dpt_ready": True,
            "best_energy_delta_vs_baselines": {"adaptive_vs_unguided_gibbs": -2.0},
            "detailed_balance_sanity": {"passed": True},
            "exact_enumeration_checked": True,
            "hardware_speedup_claimed": False,
            "optimum_hit_rate": {"adaptive_two_d_beta_penalty_pt": 1.0, "unguided_gibbs": 0.5},
        },
        5130: {
            **_base_payload(
                "exp5130-taco-sampler-heldout-scale-v470",
                "success_heldout_csp_trace_suite_ready_exact_labels_preserved",
            ),
            "average_effort_reduction_ratio_guarded": 0.04785,
            "baseline_effort": {"total_effort_score": 6604},
            "guarded_effort": {"total_effort_score": 6288},
            "harmful_instance_count_guarded": 3,
            "harmful_instance_count_unguarded": 4,
            "heldout_csp_trace_suite_ready": True,
            "instance_count": 10,
            "sampler_feature_effort": {"total_effort_score": 6633},
            "wrong_label_count": 0,
        },
        5131: {
            **_base_payload(
                "exp5131-fr11-case-policy-self-learning-v470",
                "complete_fr11_case_policy_no_promote_validation_harm_gate_closed",
            ),
            "continuous_self_learning_task": True,
            "exact_solver_correctness_preserved": True,
            "harmful_promotion_count": 0,
            "heldout_delta": 0.0,
            "nonforgetting_delta": 0.0,
            "no_weight_update": True,
            "promotion_attempted": True,
            "promotion_safe": False,
            "rollback_receipt": {"rollback_applied": True},
        },
        5132: {
            **_base_payload(
                "exp5132-authenticated-board-timing-v470",
                "complete_authenticated_board_blockers_cpu_residual_no_speedup_claim",
            ),
            "extropic_tsu_execution_claimed": False,
            "gatemate_checked": True,
            "gatemate_detected": False,
            "kv260_host_block_devices_touched": False,
            "kv260_ssh_checked": True,
            "kv260_ssh_ready": True,
            "no_speedup_claim": True,
            "polarfire_checked": True,
            "polarfire_ssh_ready": True,
            "timing_measurements": {"full_board_speedup_evidence_present": False},
        },
    }
    return payloads


def _make_repo(tmp_path: Path, *, omit: set[int] | None = None) -> Path:
    omit = omit or set()
    for source in exp.UPSTREAM_SOURCES:
        payload = _upstream_payloads()[source.experiment_number]
        if source.experiment_number not in omit:
            _write_json(tmp_path / source.relative_path, payload)
    return tmp_path


def _ids(rows: list[dict]) -> set[int]:
    return {int(row["experiment_number"]) for row in rows}


def test_req_capstone_5133_spec_declares_v470_capstone_contract() -> None:
    """REQ-CAPSTONE-5133: OpenSpec anchors the ungated .470 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5133") :]

    for marker in (
        "REQ-CAPSTONE-5133",
        "SCENARIO-CAPSTONE-5133",
        "SCENARIO-CAPSTONE-5133-FIELD-PRINCIPLES",
        exp.EXPERIMENT_ID,
        str(exp.RESULT_RELATIVE_PATH),
        "FoVer selector",
    ):
        assert marker in section
    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5133_aggregates_clean_axes_and_quarantines_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5133: critical or stamped artifacts are excluded from headlines."""

    root = _make_repo(tmp_path)
    artifact = exp.build_artifact(
        root=root,
        duration_s=1.25,
        run_date="20260701",
        tests_run=["test_scenario_capstone_5133"],
        adversarial_reporter=_reporter,
    )

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["flagged_adversarial"] is False

    assert _ids(artifact["quarantined_artifacts"]) == {5123, 5125, 5126}
    assert _ids(artifact["gated_skips"]) == {5127}
    assert artifact["missing_artifacts"] == []
    assert artifact["fover_same_scope_retired"] is True

    assert artifact["runtime_state"]["state"] == "clean_sota_runtime_ready"
    assert artifact["runtime_state"]["endpoint_lifetime_s"] is None
    assert artifact["structured_energy_state"]["state"] == "no_surviving_positive_audit_gap"
    assert artifact["structured_energy_state"]["positive_result_survived_audit"] is False
    assert artifact["structured_energy_state"]["attempted_pool"]["pool_n"] == 96
    assert (
        artifact["structured_energy_state"]["attempted_ranker"]["distributional_energy_delta"]
        == 0.0
    )
    assert artifact["structured_energy_state"]["failure_reasons"]

    assert artifact["kan_certificate_state"]["state"] == "clean_certificate_explanation_positive"
    assert (
        artifact["solver_sampling_state"]["state"]
        == "clean_exact_checked_bounded_solver_sampling_progress"
    )
    assert artifact["solver_sampling_state"]["sampler_feature_helped_solver_effort"] is False
    assert artifact["fr11_state"]["state"] == "safe_no_promotion"
    assert artifact["fr11_state"]["continuous_self_learning_task"] is True
    assert artifact["fr11_state"]["promotion_safe"] is False
    assert artifact["fr11_state"]["rollback_applied"] is True
    assert (
        artifact["hardware_state"]["state"]
        == "continuity_with_authenticated_blockers_no_speedup_claim"
    )
    assert artifact["hardware_state"]["no_speedup_claim"] is True
    assert 3 <= len(artifact["next_milestone_recommendations"]) <= 5


def test_scenario_capstone_5133_missing_artifact_is_axis_gap_not_global_zero(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-5133: missing upstreams are per-axis gaps, not unrelated failures."""

    root = _make_repo(tmp_path, omit={5130})
    artifact = exp.build_artifact(
        root=root,
        duration_s=1.25,
        run_date="20260701",
        tests_run=["missing-axis-gap-test"],
        adversarial_reporter=_reporter,
    )

    exp.validate_artifact(artifact)
    assert _ids(artifact["missing_artifacts"]) == {5130}
    assert artifact["missing_artifacts"][0]["axis"] == "solver_sampling"
    assert artifact["solver_sampling_state"]["state"] == "gap_exp5130_missing"
    assert artifact["runtime_state"]["state"] == "clean_sota_runtime_ready"
    assert artifact["kan_certificate_state"]["state"] == "clean_certificate_explanation_positive"
    assert (
        artifact["hardware_state"]["state"]
        == "continuity_with_authenticated_blockers_no_speedup_claim"
    )


def test_req_capstone_5133_validation_rejects_malformed_artifact(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5133: terminal schema validation rejects missing required fields."""

    artifact = exp.build_artifact(
        root=_make_repo(tmp_path),
        duration_s=1.0,
        run_date="20260701",
        tests_run=["validation-test"],
        adversarial_reporter=_reporter,
    )
    malformed = dict(artifact)
    malformed.pop("structured_energy_state")

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(malformed)


def test_scenario_capstone_5133_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5133: CLI wrapper writes the JSON artifact."""

    root = _make_repo(tmp_path)
    path = script_mod.main(
        root=root,
        date="20260701",
        duration_s=1.0,
        tests_run=["script-entrypoint-test"],
        adversarial_reporter=_reporter,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["experiment_id"] == exp.EXPERIMENT_ID


def test_deliverable_file_validates_for_scenario_capstone_5133() -> None:
    """SCENARIO-CAPSTONE-5133: checked-in .470 capstone artifact validates."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["structured_energy_state"]["positive_result_survived_audit"] is False
    assert artifact["fr11_state"]["continuous_self_learning_task"] is True

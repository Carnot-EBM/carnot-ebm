"""Tests for the Exp 3121 milestone .290 capstone.

Spec refs: REQ-REPORT-3121, SCENARIO-REPORT-3121.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v290_3121 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v23",
    "verifier_gain_status",
    "formal_feedback_status",
    "repair_claim_status",
    "fr11_self_learning_status",
    "ebt_arm_status",
    "sampler_hardware_status",
    "gatemate_status",
    "ssqa_status",
    "next_recommendation",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_artifacts() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "exp3111",
            "path": "results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json",
            "role": "certified_coherence_feedback",
            "present": True,
            "readable_json_object": True,
        },
        {
            "experiment_id": "exp3115",
            "path": "results/experiment_3115_explicit_repair_gate_micro_panel_v4.json",
            "role": "explicit_repair_gate_micro_panel",
            "present": True,
            "readable_json_object": True,
        },
    ]


def _matrix_v24(
    *,
    ready: bool = True,
    blockers: int = 36,
    blocker_delta: int = 0,
    missing_artifacts: list[dict[str, Any]] | None = None,
    downgrade_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3120_cross_corpus_matrix_v24",
        "matrix_v24_ready": ready,
        "rows_total": 102,
        "status_counts": {
            "blocked": 5,
            "bounded": 13,
            "clean": 31,
            "diagnostic_only": 4,
            "flagged": 7,
            "gated_skipped": 7,
            "missing": 2,
            "model_spec_gap": 0,
            "projection_only": 2,
            "retired": 31,
        },
        "publication_blocker_count": blockers,
        "blocker_delta_from_v23": blocker_delta,
        "missing_artifacts": [] if missing_artifacts is None else missing_artifacts,
        "headline_model_spec_gaps": [
            {
                "row_id": "dot290:exp3110_sota_model_spec_cache_manifest",
                "source_artifact": (
                    "results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json"
                ),
                "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "missing_model_ids": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                ],
                "reason": "mandated headline model cache coverage incomplete",
            }
        ]
        if blockers
        else [],
        "verifier_repair_status": {
            "model_manifest_status": "bounded" if blockers else "clean",
            "certified_coherence_status": "clean",
            "diagnostic_calibration_status": "diagnostic_only" if blockers else "clean",
            "logic_pilot_status": "diagnostic_only",
            "fragment_verification_status": "clean",
            "repair_micro_panel_status": "bounded" if blockers else "clean",
            "repair_unblocked": True,
            "repair_run_executed": True,
            "repair_success_delta": 0.0 if blockers else 0.25,
            "false_repair_accept_rate": 0.0,
            "intent_preservation_rate": 0.0 if blockers else 1.0,
        },
        "fr11_status": {
            "status": "bounded_controller_only_no_weight_update_claim"
            if blockers
            else "clean_controller_only_no_weight_update_claim",
            "controller_only": True,
            "fr11_unsolvable_curriculum_ready": True,
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
            "no_weight_update_claim": True,
            "promotion_decision": "controller_only",
        },
        "architecture_boundary_status": {
            "ebt_arm_status": "projection_only_no_live_model_integration"
            if blockers
            else "clean_sidecar_promoted",
            "clut_status": "bounded_cpu_only_no_hardware_speedup"
            if blockers
            else "clean_hardware_speedup_claim_downgraded",
            "gatemate_status": "blocked_operator_evidence_incomplete"
            if blockers
            else "clean_no_publication_scope",
            "ssqa_status": "gated_skipped_host_visible_readback_missing"
            if blockers
            else "clean_no_publication_scope",
        },
        "source_artifacts": _source_artifacts(),
        "publication_blocker_downgrade_policy": downgrade_policy or {},
        "inference_substrate": {
            "kind": "aggregation_from_checked_in_artifacts",
            "no_live_llm_inference": True,
            "executes_models": False,
            "executes_hardware": False,
            "executes_conductor": False,
        },
        "honest_verdict": (
            "complete: matrix_v24_ready=true; rows_total=102; "
            f"publication_blocker_count={blockers}; blocker_delta_from_v23={blocker_delta}"
        ),
    }


def _write_matrix_and_sources(root: Path, matrix: dict[str, Any]) -> None:
    _write_json(root, mod.MATRIX_V24_REL_PATH, matrix)
    _write_json(
        root,
        mod.PRIOR_CAPSTONE_REL_PATH,
        {
            "artifact": "experiment_3108_capstone_v289",
            "capstone_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 36,
            "honest_verdict": "complete: capstone_ready=true",
        },
    )
    for row in matrix["source_artifacts"]:
        if not row["path"]:
            continue
        _write_json(
            root,
            row["path"],
            {
                "artifact": Path(row["path"]).stem,
                "ready": True,
                "honest_verdict": f"complete: {Path(row['path']).stem}",
            },
        )


def test_req_report_3121_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3121: OpenSpec declares the .290 capstone before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3121" in spec
    assert "SCENARIO-REPORT-3121" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3121_closes_from_matrix_v24_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3121: v24 evidence closes .290 without paper overclaim."""

    matrix = _matrix_v24()
    _write_matrix_and_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=6.25)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 36
    assert artifact["blocker_delta_from_v23"] == 0
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["honest_verdict"].startswith("complete:")

    assert (
        artifact["verifier_gain_status"]
        == "diagnostic_gain_recovered_but_headline_bounded_by_cache_and_prior_flags"
    )
    assert artifact["formal_feedback_status"] == "solver_certified_feedback_ready_no_live_sota_lift"
    assert artifact["repair_claim_status"] == "bounded_micro_panel_executed_zero_delta_no_promotion"
    assert (
        artifact["fr11_self_learning_status"]
        == "bounded_controller_only_soundness_zero_completeness_zero_no_weight_update"
    )
    assert artifact["ebt_arm_status"] == "projection_only_sidecar_correlation_no_live_model_integration"
    assert artifact["sampler_hardware_status"] == "bounded_clut_cpu_only_no_hardware_speedup"
    assert artifact["gatemate_status"] == "blocked_operator_evidence_incomplete_no_hardware_run"
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_readback_missing"

    assert artifact["paper_readiness_checks"] == [
        {
            "check": "capstone_ready",
            "passed": True,
            "reason": "matrix v24 authority loaded and required invariants reconciled",
        },
        {
            "check": "publication_blocker_count_zero_or_downgraded",
            "passed": False,
            "reason": "publication_blocker_count=36 and no all-blockers-downgraded policy",
        },
        {
            "check": "headline_model_spec_gaps",
            "passed": False,
            "reason": "headline_model_spec_gaps=1; missing mandated cached SOTA coverage remains bounded",
        },
    ]
    assert artifact["delta_from_v289"]["model_spec_gap"].startswith("changed:")
    assert artifact["delta_from_v289"]["gatemate_ssqa"].startswith("unchanged:")
    assert artifact["milestone_proved"][0].startswith("Matrix v24 is complete")
    assert artifact["remaining_top_gaps"] == [
        "publishable_verifier_repair_headline_evidence",
        "operator_owned_gatemate_ssqa_host_visible_evidence",
        "live_model_or_authenticated_hardware_architecture_integration",
    ]
    assert "publishable verifier/repair evidence" in artifact["next_recommendation"]
    assert artifact["ops_reconciliation_decision"]["ops_status_updated"] is False
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert sources[mod.MATRIX_V24_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V24_REL_PATH
    )


def test_req_report_3121_paper_ready_when_blockers_zero(tmp_path: Path) -> None:
    """REQ-REPORT-3121: zero blockers and no headline gaps permit paper readiness."""

    matrix = _matrix_v24(blockers=0, blocker_delta=-36)
    _write_matrix_and_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v23"] == -36
    assert artifact["paper_readiness_checks"][1]["passed"] is True
    assert artifact["paper_readiness_checks"][2]["passed"] is True


def test_req_report_3121_explicit_downgrade_policy_can_clear_paper_gate(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3121: explicitly downgraded blockers can be outside headline scope."""

    matrix = _matrix_v24(
        blockers=2,
        downgrade_policy={
            "all_remaining_blockers_downgraded": True,
            "headline_scope_after_downgrade": "none",
            "reason": "all remaining rows moved to non-headline appendix scope",
        },
    )
    matrix["headline_model_spec_gaps"] = []
    _write_matrix_and_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["paper_ready"] is True
    assert artifact["paper_readiness_checks"][1] == {
        "check": "publication_blocker_count_zero_or_downgraded",
        "passed": True,
        "reason": "all remaining blockers explicitly downgraded outside headline scope",
    }


def test_req_report_3121_blocks_on_unready_or_missing_matrix(tmp_path: Path) -> None:
    """REQ-REPORT-3121: missing or unready matrix v24 blocks capstone completion."""

    missing = mod.build_artifact(tmp_path)
    assert missing["capstone_ready"] is False
    assert missing["honest_verdict"].startswith("blocked:")
    assert "required source unreadable" in missing["invariant_violations"][0]

    matrix = _matrix_v24(ready=False)
    _write_matrix_and_sources(tmp_path, matrix)
    unready = mod.build_artifact(tmp_path)
    assert unready["capstone_ready"] is False
    assert "matrix_v24_ready is not true" in unready["invariant_violations"]


def test_req_report_3121_records_invariant_and_source_role_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3121: source classification and invariant failures are explicit."""

    matrix = _matrix_v24()
    matrix["source_artifacts"].extend(
        [
            {"path": "results/extra_matrix_context.json"},
            {"path": "results/extra_capstone_context.json"},
            {"path": "notes/source.txt"},
            {"path": ""},
            {"path": "results/extra_matrix_context.json"},
        ]
    )
    matrix["status_counts"]["clean"] = 30
    matrix["publication_blocker_count"] = -1
    matrix["inference_substrate"]["executes_models"] = True
    _write_matrix_and_sources(tmp_path, matrix)
    _write_json(tmp_path, "results/extra_matrix_context.json", {"artifact": "extra-matrix"})
    _write_json(tmp_path, "results/extra_capstone_context.json", {"artifact": "extra-capstone"})
    text_path = tmp_path / "notes/source.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("text evidence\n", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path)
    roles = {row["path"]: row["role"] for row in artifact["source_artifacts"]}
    edge_violations = mod._invariant_violations({"matrix_v24_ready": True}, [], [])

    assert mod._source_role(mod.MATRIX_V24_REL_PATH) == "matrix_v24_authority"
    assert mod._source_role(mod.PRIOR_CAPSTONE_REL_PATH) == "prior_capstone_v289"
    assert roles["results/extra_matrix_context.json"] == "matrix_context"
    assert roles["results/extra_capstone_context.json"] == "capstone_context"
    assert roles["notes/source.txt"] == "matrix_v24_source"
    assert "status_counts do not reconcile with rows_total" in artifact["invariant_violations"]
    assert "publication_blocker_count is negative" in artifact["invariant_violations"]
    assert "matrix inference_substrate is not aggregation-only" in artifact["invariant_violations"]
    assert edge_violations == ["source_artifacts list is empty"]


def test_req_report_3121_write_artifact_persists_json(tmp_path: Path) -> None:
    """REQ-REPORT-3121: write_artifact emits the requested deliverable path."""

    _write_matrix_and_sources(tmp_path, _matrix_v24())
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/custom_capstone.json"),
        started_s=4.0,
        now_s=5.0,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / "results/custom_capstone.json"
    assert payload["artifact"] == mod.ARTIFACT
    assert payload["duration_s"] == pytest.approx(1.0)

"""Tests for the Exp 3148 milestone .292 capstone.

Spec refs: REQ-REPORT-3148, SCENARIO-REPORT-3148.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v292_3148 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v25",
    "next_top_gap",
    "false_accept_recovery_status",
    "verifier_claim_status",
    "repair_gate_status",
    "repair_claim_status",
    "fr11_self_learning_status",
    "ebt_arm_status",
    "kan_status",
    "sampler_hardware_status",
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
    rows: list[dict[str, Any]] = []
    for spec in mod.SOURCE_SPECS:
        rows.append(
            {
                "experiment_id": spec["experiment_id"],
                "path": spec["path"].as_posix(),
                "role": spec["role"],
                "required": spec["required"],
                "ready_field": spec["ready_field"],
                "present": spec["experiment_id"] != "exp3141",
                "readable_json_object": spec["experiment_id"] != "exp3141",
                "source_type": "json",
            }
        )
    return rows


def _publication_blockers(count: int) -> list[dict[str, Any]]:
    scopes = [
        "live_sota_verifier_lift",
        "repair_gate_decision",
        "repair_live_rerun",
        "controller_only_environment_memory",
        "architecture_hardware_sampler_boundary",
    ]
    return [
        {
            "row_id": f"blocker:{idx}",
            "status": "blocked",
            "blocker_class": "blocked",
            "source_artifact": "results/source.json",
            "source_field": "status",
            "claim_scope": scopes[idx % len(scopes)],
        }
        for idx in range(count)
    ]


def _matrix_v26(*, ready: bool = True, blockers: int = 55) -> dict[str, Any]:
    blocked_claims = [] if blockers == 0 else [
        "comparative_sota_pair",
        "live_verifier_lift_adversarial_flag",
        "repair_headline",
        "fr11_model_weight_learning",
        "ebt_arm_live_integration",
        "kan_deployed_verifier",
        "hardware_speedup",
    ]
    return {
        "artifact": "experiment_3147_cross_corpus_matrix_v26",
        "matrix_v26_ready": ready,
        "rows_total": 1 if blockers == 0 else 124,
        "prior_publication_blocker_count": 46,
        "publication_blocker_count": blockers,
        "blocker_delta_from_v25": blockers - 46,
        "missing_artifacts": []
        if blockers == 0
        else [
            {
                "experiment_id": "exp3141",
                "path": mod.EXP3141_REL_PATH.as_posix(),
                "reason": "missing_or_malformed_dot292_artifact",
            }
        ],
        "status_counts": {
            "blocked": 0 if blockers == 0 else 10,
            "bounded": 0 if blockers == 0 else 20,
            "clean": 1 if blockers == 0 else 34,
            "diagnostic_only": 0 if blockers == 0 else 4,
            "flagged": 0 if blockers == 0 else 11,
            "gated_skipped": 0 if blockers == 0 else 8,
            "missing": 0 if blockers == 0 else 2,
            "model_spec_gap": 0,
            "projection_only": 0 if blockers == 0 else 4,
            "retired": 0 if blockers == 0 else 31,
        },
        "publication_blockers": _publication_blockers(blockers),
        "headline_claim_allowance_summary": {
            "blocked_headline_claims": blocked_claims,
            "canonical_grounding_claim_allowed": True,
            "comparative_sota_pair_allowed": blockers == 0,
            "ebt_arm_live_integration_allowed": blockers == 0,
            "exact_safe_contract_claim_allowed": True,
            "fr11_model_weight_learning_allowed": blockers == 0,
            "hardware_speedup_claim_allowed": blockers == 0,
            "kan_deployed_verifier_allowed": blockers == 0,
            "live_verifier_headline_allowed": blockers == 0,
            "missing_model_ids": [] if blockers == 0 else [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "repair_headline_claim_allowed": blockers == 0,
            "selected_headline_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "sota_cache_headline_allowed": True,
        },
        "false_accept_recovery_summary": {
            "accept_abstain_contract_status": "clean",
            "autopsy_status": "clean" if blockers == 0 else "flagged",
            "canonical_false_accept_rows_blocked": 2,
            "canonical_grounding_status": "clean",
            "corrigendum_pending_count": 0 if blockers == 0 else 6,
            "false_accept_gate_passed": True,
            "false_accept_row_ids": ["resyn-3084-arith-003", "resyn-3084-smt-000"],
            "flagged_adversarial_artifact_count": 0 if blockers == 0 else 2,
            "known_false_accept_rows_blocked": True,
            "live_verifier_status": "clean" if blockers == 0 else "flagged",
            "recomputed_false_accept_rate": 0.5,
            "recovery_claim_status": "exact_safe_recovery_ready"
            if blockers == 0
            else "blocked_by_adversarial_corrigendum",
            "rerun_abstention_rate": 0.5,
            "rerun_false_accept_rate": 0.0,
            "rerun_verifier_gain_delta": 0.5,
            "source_false_accept_rate": 0.5,
        },
        "repair_gate_summary": {
            "false_accept_gate_passed": True,
            "false_accept_rate": 0.0,
            "headline_disqualifier_count": 0 if blockers == 0 else 6,
            "headline_repair_claim_allowed": blockers == 0,
            "known_false_accepts_blocked": True,
            "regression_rows_included": True,
            "repair_blocker_count": 0 if blockers == 0 else 6,
            "repair_gate_state": "unblocked" if blockers == 0 else "blocked_other",
            "repair_gate_status": "clean" if blockers == 0 else "blocked",
            "repair_ladder_missing_path": "" if blockers == 0 else mod.EXP3141_REL_PATH.as_posix(),
            "repair_ladder_present": blockers == 0,
            "repair_ladder_status": "clean" if blockers == 0 else "gated_skipped",
            "selected_repair_rows": ["repair-row"] if blockers == 0 else [],
        },
        "fr11_summary": {
            "admitted_environment_count": 3,
            "continuous_self_learning_targeted": True,
            "experience_ledger_consistency_rate": 1.0 if blockers == 0 else 0.666667,
            "experience_memory_status": "clean" if blockers == 0 else "flagged",
            "hardened_variant_count": 3,
            "model_weight_learning_allowed": blockers == 0,
            "no_weight_update_claim": blockers != 0,
            "promotion_recommendations": ["controller memory only"] if blockers else [],
            "replay_row_count": 12,
            "residual_false_accept_risk": 0.0,
            "residual_false_reject_risk": 0.166667 if blockers else 0.0,
            "vera_evoenv_status": "clean" if blockers == 0 else "flagged",
            "vera_ledger_consistency_rate": 1.0 if blockers == 0 else 0.833333,
        },
        "architecture_boundary_summary": {
            "deployed_kan_verifier_claim": blockers == 0,
            "ebt_arm_status": "clean" if blockers == 0 else "projection_only",
            "false_accept_rows_evaluated": 2,
            "gatemate_evidence_complete": blockers == 0,
            "hardware_boundary_status": "clean" if blockers == 0 else "blocked",
            "hardware_commands_run": ["authenticated-smoke"] if blockers == 0 else [],
            "integration_blocker_count": 0 if blockers == 0 else 6,
            "kan_attached_monitor_record_count": 2,
            "kan_does_not_prove": []
            if blockers == 0
            else ["trained KAN network soundness", "hardware execution"],
            "kan_monitor_status": "clean" if blockers == 0 else "bounded",
            "kona_claim_allowed": blockers == 0,
            "live_integration": blockers == 0,
            "missing_operator_evidence_count": 0 if blockers == 0 else 8,
            "speedup_claim_allowed": blockers == 0,
            "ssqa_readback_ready": blockers == 0,
            "thrml_tsu_claim_allowed": blockers == 0,
        },
        "source_artifacts": _source_artifacts(),
        "invariant_violations": [],
        "required_source_errors": [],
        "inference_substrate": {
            "kind": "aggregation_from_checked_in_dot292_artifacts",
            "source": "matrix_v25_capstone_v291_and_dot292_artifacts",
            "executes_models": False,
            "executes_verifiers": False,
            "executes_repairs": False,
            "executes_solvers": False,
            "executes_hardware": False,
            "executes_conductor": False,
            "no_live_llm_inference": True,
            "local_repo_only": True,
        },
        "honest_verdict": "complete: matrix_v26_ready=true",
    }


def _write_matrix_and_sources(root: Path, *, blockers: int = 55) -> None:
    matrix = _matrix_v26(blockers=blockers)
    _write_json(root, mod.MATRIX_V26_REL_PATH, matrix)
    for source in matrix["source_artifacts"]:
        if source["experiment_id"] == "exp3141":
            continue
        _write_json(
            root,
            source["path"],
            {"artifact": source["experiment_id"], source["ready_field"]: True},
        )


def test_req_report_3148_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3148: OpenSpec declares the .292 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3148" in spec
    assert "SCENARIO-REPORT-3148" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3148_builds_capstone_from_matrix_v26(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3148: .292 closes with evidence boundaries preserved."""

    _write_matrix_and_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 55
    assert artifact["blocker_delta_from_v25"] == 9
    assert artifact["next_top_gap"] == "false_accept_recovery_corrigendum_repair_gate"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["false_accept_recovery_status"] == (
        "blocked_by_adversarial_corrigendum_false_accept_0.0_known_rows_blocked"
    )
    assert artifact["verifier_claim_status"] == (
        "flagged_live_verifier_false_accept_0.0_gain_0.5_no_headline"
    )
    assert artifact["repair_gate_status"] == (
        "blocked_repair_gate_state_blocked_other_blockers_6_disqualifiers_6"
    )
    assert artifact["repair_claim_status"] == (
        "blocked_repair_ladder_gated_skipped_no_selected_rows"
    )
    assert artifact["repair_ladder_status"] == "gated_skipped_missing_artifact"
    assert artifact["fr11_vera_evoenv_status"] == "flagged_ledger_0.833333_controller_only"
    assert artifact["fr11_experience_memory_status"] == (
        "flagged_ledger_0.666667_controller_only"
    )
    assert artifact["fr11_self_learning_status"] == (
        "bounded_controller_memory_only_no_weight_update_vera_0.833333_experience_0.666667"
    )
    assert artifact["ebt_arm_status"] == "projection_only_no_live_integration_blockers_6"
    assert artifact["kan_status"] == "bounded_monitor_records_2_no_deployed_verifier"
    assert artifact["sampler_hardware_status"] == (
        "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
    )
    assert artifact["local_sota_cache_status"] == (
        "bounded_single_cached_model_comparative_pair_missing_2"
    )
    assert artifact["paper_readiness_assessment"] == (
        "not_closer_publication_blockers_increased_by_9"
    )

    assert "live_verifier_lift_adversarial_flag" in artifact["what_stayed_blocked"]
    assert "repair_ladder_execution_missing" in artifact["what_stayed_blocked"]
    assert "exact_safe_accept_abstain_contract_replay" in artifact["allowed_claims"]
    assert "live_verifier_headline_lift" in artifact["forbidden_claims"]
    assert artifact["missing_artifacts"] == [
        {
            "experiment_id": "exp3141",
            "path": mod.EXP3141_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_dot292_artifact",
        }
    ]
    assert sources[mod.MATRIX_V26_REL_PATH.as_posix()]["role"] == "matrix_v26_authority"
    assert sources[mod.EXP3141_REL_PATH.as_posix()]["present"] is False
    assert sources[mod.EXP3139_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3139_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_matrix_v26_and_dot292_artifacts",
        "source": mod.MATRIX_V26_REL_PATH.as_posix(),
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
        "live_model_calls_run_by_capstone": 0,
        "hardware_commands_run_by_capstone": [],
    }
    assert artifact["next_recommendation"].startswith("Next milestone should clear")


def test_req_report_3148_clean_matrix_becomes_paper_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3148: paper readiness is explicit, not inferred from completion alone."""

    _write_matrix_and_sources(tmp_path, blockers=0)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v25"] == -46
    assert artifact["next_top_gap"] == "publication_scope_reconciliation"
    assert artifact["false_accept_recovery_status"] == "clean_exact_safe_recovery_ready"
    assert artifact["verifier_claim_status"] == (
        "clean_live_verifier_false_accept_0.0_headline_allowed"
    )
    assert artifact["repair_gate_status"] == "clean_repair_gate_unblocked"
    assert artifact["repair_claim_status"] == "clean_repair_ladder_promotable"
    assert artifact["fr11_self_learning_status"] == "clean_model_weight_learning_allowed"
    assert artifact["ebt_arm_status"] == "clean_ebt_arm_live_integration"
    assert artifact["kan_status"] == "clean_deployed_kan_verifier_claim"
    assert artifact["sampler_hardware_status"] == "clean_authenticated_sampler_hardware_speedup"
    assert artifact["local_sota_cache_status"] == "clean_comparative_sota_cache_ready"
    assert artifact["paper_readiness_assessment"] == "paper_ready_blockers_cleared"
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3148_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3148: malformed inputs fail closed and helpers remain deterministic."""

    _write_matrix_and_sources(tmp_path)
    bad = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    bad.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))
    missing = mod.build_artifact(tmp_path / "empty")
    not_ready_root = tmp_path / "not_ready"
    _write_json(not_ready_root, mod.MATRIX_V26_REL_PATH, _matrix_v26(ready=False))
    not_ready = mod.build_artifact(not_ready_root)

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.25)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._duration(5.0, 4.0) == 0.0
    assert missing["capstone_ready"] is False
    assert missing["honest_verdict"].startswith("blocked:")
    assert "matrix_v26 authority is missing or malformed" in missing["invariant_violations"]
    assert not_ready["capstone_ready"] is False
    assert "matrix_v26_ready is not true" in not_ready["invariant_violations"]

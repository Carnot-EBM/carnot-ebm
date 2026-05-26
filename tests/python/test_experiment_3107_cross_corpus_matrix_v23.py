"""Tests for Exp 3107 cross-corpus matrix v23.

Spec refs: REQ-REPORT-3107, SCENARIO-REPORT-3107.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v23_3107 as mod


REQUIRED_FIELDS = {
    "matrix_v23_ready",
    "rows_total",
    "status_counts",
    "publication_blocker_count",
    "blocker_delta_from_v22",
    "missing_artifacts",
    "headline_model_spec_gaps",
    "capstone_input_artifacts",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, status: str, claim_scope: str, evidence_class: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": evidence_class,
        "blocker_class": mod.blocker_class(status),
        "claim_scope": claim_scope,
        "summary": {"source_status": status},
        "row_origin": "matrix_v22_test",
    }


def _matrix_v22() -> dict[str, Any]:
    rows = [
        _row("clean-carry", "clean", "milestone_activation", "archive"),
        _row("flagged-carry", "flagged", "verifier_repair", "panel"),
        _row("bounded-carry", "bounded", "paper_readiness", "capstone"),
        _row("blocked-carry", "blocked", "hardware_rerun_gate", "gatemate"),
        _row("gated-carry", "gated_skipped", "repair_live_rerun", "gate"),
        _row("missing-carry", "missing", "missing_artifact", "artifact"),
        _row("projection-carry", "projection_only", "future_adapter_context", "adapter"),
        _row("retired-carry", "retired", "old_claim", "retired"),
        _row("capstone:v287_paper_readiness", "bounded", "paper_readiness", "capstone"),
        _row(
            "dot288:exp3085_icalm_task_abstention_sota_panel",
            "flagged",
            "local_sota_solution_verifier_gain",
            "live_llm",
        ),
        _row(
            "dot288:exp3086_dafny_z3_formal_feedback_pilot",
            "flagged",
            "solver_grounded_repair_feedback",
            "live_llm",
        ),
        _row(
            "dot288:exp3087_local_sota_verifier_calibration_gate",
            "gated_skipped",
            "verifier_gain_recovery_gate",
            "gate",
        ),
        _row(
            "dot288:exp3089_xgrammar_sota_repair_micro_panel",
            "missing",
            "repair_live_rerun",
            "repair",
        ),
        _row(
            "dot288:exp3091_ebt_arm_sidecar_adapter_schema_prototype",
            "projection_only",
            "future_adapter_context",
            "ebt_arm",
        ),
        _row(
            "dot288:exp3092_gatemate_operator_evidence",
            "blocked",
            "hardware_rerun_gate",
            "gatemate",
        ),
        _row(
            "dot288:exp3092_ssqa_readback_evidence",
            "gated_skipped",
            "host_visible_readback_gate",
            "ssqa",
        ),
    ]
    blockers = [
        {
            "row_id": row["row_id"],
            "status": row["status"],
            "blocker_class": row["blocker_class"],
            "source_artifact": row["source_artifact"],
            "source_field": row["source_field"],
            "claim_scope": row["claim_scope"],
        }
        for row in rows
        if row["status"] in mod.PUBLICATION_BLOCKING_STATUSES
    ]
    return {
        "artifact": "experiment_3093_cross_corpus_matrix_v22",
        "matrix_v22_ready": True,
        "rows_total": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in mod.LEGACY_STATUSES
        },
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "rows": rows,
        "honest_verdict": "complete: matrix_v22_ready=true",
    }


def _capstone_v288(blocker_count: int) -> dict[str, Any]:
    return {
        "artifact": "experiment_3094_capstone_v288",
        "capstone_ready": True,
        "paper_ready": False,
        "publication_blocker_count": blocker_count,
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _ledger(blocker_count: int) -> dict[str, Any]:
    return {
        "artifact": "experiment_3096_publication_blocker_triage_and_retirement_ledger_v2",
        "blocker_triage_ready": True,
        "publication_blocker_count_before": blocker_count,
        "blocker_categories": {"verifier_repair": [], "retired_status": []},
        "honest_verdict": "complete: blocker_triage_ready=true",
    }


def _write_sources(root: Path, *, include_3102: bool = False) -> None:
    matrix = _matrix_v22()
    blocker_count = int(matrix["publication_blocker_count"])
    _write_json(root, mod.MATRIX_V22_REL_PATH, matrix)
    _write_json(root, mod.CAPSTONE_V288_REL_PATH, _capstone_v288(blocker_count))
    _write_json(root, mod.EXP3096_REL_PATH, _ledger(blocker_count))
    _write_json(
        root,
        mod.EXP3095_REL_PATH,
        {"archive_v288_activate_v289_ready": True, "honest_verdict": "complete: archive=true"},
    )
    _write_json(
        root,
        mod.EXP3097_REL_PATH,
        {
            "eval_protocol_ready": True,
            "minimum_live_eval_count": 48,
            "usable_fixture_count": 72,
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3098_REL_PATH,
        {
            "maxsat_policy_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
            "honest_verdict": "complete: maxsat_policy_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3099_REL_PATH,
        {
            "abstention_panel_v3_ready": True,
            "exact_ground_truth_count": 48,
            "minimum_live_eval_count": 48,
            "evaluated_fixture_count": 48,
            "model_specs": [{"hf_id": "model-a"}],
            "inference_substrate": {"executes_models": True, "live_llm_inference": True},
            "honest_verdict": "complete: abstention_panel_v3_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3100_REL_PATH,
        {
            "formal_feedback_v2_ready": False,
            "headline_blocked_reason": "cached_sota_pair_unavailable",
            "model_specs": [{"hf_id": "model-a"}],
            "inference_substrate": {"live_llm_inference": False},
            "honest_verdict": "complete_blocked_headline: formal_feedback_v2_ready=false",
        },
    )
    _write_json(
        root,
        mod.EXP3101_REL_PATH,
        {"schema": "blocked_gate_check_v1", "status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
    )
    if include_3102:
        _write_json(
            root,
            mod.EXP3102_REL_PATH,
            {"gated_structured_repair_micro_panel_ready": True, "honest_verdict": "complete: panel=true"},
        )
    _write_json(
        root,
        mod.EXP3103_REL_PATH,
        {
            "fr11_stress_ready": True,
            "promotion_decision": "blocked",
            "soundness_mistakes": 0,
            "completeness_mistakes": 12,
            "honest_verdict": "complete_fr11_stress_boundary_blocks_promotion",
        },
    )
    _write_json(
        root,
        mod.EXP3104_REL_PATH,
        {
            "sidecar_boundary_v2_ready": True,
            "no_live_model_integration_claim": True,
            "remaining_integration_blockers": ["live model integration absent"],
            "honest_verdict": "complete: sidecar_boundary_v2_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3105_REL_PATH,
        {
            "status": "success",
            "clut_microbench_ready": True,
            "hardware_claim_made": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
            "honest_verdict": "complete: CPU cLUT microbench ran",
        },
    )
    _write_json(
        root,
        mod.EXP3106_REL_PATH,
        {
            "operator_evidence_ingestion_v3_ready": True,
            "gatemate_rerun_allowed": False,
            "ssqa_readback_allowed": False,
            "missing_operator_actions": [{"missing_item": "host_visible_smoke_evidence"}],
            "honest_verdict": "complete: operator_evidence_ingestion_v3_ready=true",
        },
    )


def test_req_report_3107_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3107: OpenSpec declares the v23 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3107" in spec
    assert "SCENARIO-REPORT-3107" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3107_builds_v23_without_diagnostic_overclaim(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3107: replacements preserve unresolved blocker classes."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=14.5)
    rows = {row["row_id"]: row for row in artifact["rows"]}
    blockers = {row["row_id"] for row in artifact["publication_blockers"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v23_ready"] is True
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert artifact["rows_total"] == len(artifact["rows"]) == 30
    assert artifact["status_counts"] == {
        "clean": 4,
        "flagged": 1,
        "bounded": 2,
        "blocked": 3,
        "gated_skipped": 3,
        "missing": 2,
        "retired": 9,
        "projection_only": 2,
        "diagnostic_only": 3,
        "model_spec_gap": 1,
    }
    assert artifact["publication_blocker_count"] == 14
    assert artifact["blocker_delta_from_v22"] == 0
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["dot288:exp3085_icalm_task_abstention_sota_panel"]["status"] == "retired"
    assert rows["dot289:exp3099_local_sota_confidence_abstention_panel"]["status"] == "model_spec_gap"
    assert rows["dot289:exp3098_maxsat_abstention_routing_policy"]["status"] == "diagnostic_only"
    assert rows["dot289:exp3105_clut_random_variate_sampler_microbench"]["status"] == "diagnostic_only"
    assert rows["dot289:exp3100_z3_oracle_feedback"]["status"] == "blocked"
    assert rows["dot289:exp3101_local_sota_verifier_calibration_gate"]["status"] == "gated_skipped"
    assert rows["dot289:exp3102_structured_repair_micro_panel"]["status"] == "missing"
    assert rows["dot289:exp3104_ebt_arm_sidecar_pipeline_boundary"]["status"] == "projection_only"
    assert rows["dot289:exp3106_gatemate_operator_evidence"]["status"] == "blocked"
    assert rows["dot289:exp3106_ssqa_readback_evidence"]["status"] == "gated_skipped"

    assert "dot289:exp3098_maxsat_abstention_routing_policy" not in blockers
    assert "dot289:exp3105_clut_random_variate_sampler_microbench" not in blockers
    assert artifact["missing_artifacts"] == [
        {
            "path": mod.EXP3102_REL_PATH.as_posix(),
            "reason": "expected .289 structured repair micro-panel artifact is absent",
        }
    ]
    assert artifact["headline_model_spec_gaps"] == [
        {
            "row_id": "dot289:exp3099_local_sota_confidence_abstention_panel",
            "source_artifact": mod.EXP3099_REL_PATH.as_posix(),
            "missing_model_ids": [],
            "present_model_ids": ["model-a"],
            "reason": "mandatory_headline_model_ids missing for live LLM artifact",
        }
    ]

    reconciliation = artifact["blocker_reconciliation_from_exp3096"]
    assert reconciliation["publication_blocker_count_before"] == 14
    assert reconciliation["publication_blocker_count_after"] == 14
    assert reconciliation["blocker_delta_from_v22"] == 0
    assert reconciliation["decreases"][0]["count"] == 8
    assert reconciliation["increases"][0]["count"] == 8
    assert len(reconciliation["neutral_replacements"]) == 8

    assert mod.OUTPUT_REL_PATH.as_posix() in artifact["capstone_input_artifacts"]
    assert mod.EXP3106_REL_PATH.as_posix() in artifact["capstone_input_artifacts"]
    assert sources[mod.EXP3102_REL_PATH.as_posix()]["present"] is False
    assert sources[mod.MATRIX_V22_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V22_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v22_capstone_v288_ledger_and_dot289_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def test_req_report_3107_blocks_missing_authorities(tmp_path: Path) -> None:
    """REQ-REPORT-3107: missing authority artifacts block matrix readiness."""

    artifact = mod.build_artifact(tmp_path)

    assert artifact["matrix_v23_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_matrix_v23_preconditions")
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v22"] == 0
    assert [row["path"] for row in artifact["required_source_errors"]] == [
        mod.MATRIX_V22_REL_PATH.as_posix(),
        mod.CAPSTONE_V288_REL_PATH.as_posix(),
        mod.EXP3096_REL_PATH.as_posix(),
    ]


def test_req_report_3107_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3107: helper behavior is deterministic and fail-closed."""

    _write_sources(tmp_path, include_3102=True)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v23_ready"] is True
    assert saved["missing_artifacts"] == []
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("diagnostic") == "diagnostic_only"
    assert mod.normal_status("bad") == "missing"
    assert mod.blocker_class("clean") == "none"
    assert mod.blocker_class("diagnostic_only") == "diagnostic_only"
    assert mod.blocker_class("model_spec_gap") == "model_spec_gap"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._float_or_none(True) is None
    assert mod._float_or_none("bad") is None
    assert mod._carry_forward_rows({"rows": [1, _row("carry", "clean", "scope", "evidence")]})[0]["row_id"] == "carry"
    assert mod._capstone_v288_row({"capstone_ready": False})["status"] == "blocked"
    assert mod._replacement_row_id("not-replaced") == ""

    assert mod._live_llm_status_and_gaps("empty-live", {}, Path("results/live.json"), "ready") == (
        "missing",
        [],
    )
    assert mod._model_spec_gaps(
        "non-live",
        {"mandatory_headline_model_ids": ["model-a"], "inference_substrate": {"executes_models": False}},
        Path("results/live.json"),
    ) == []
    assert mod._model_spec_gaps(
        "missing-model",
        {
            "mandatory_headline_model_ids": ["model-a", "model-b"],
            "model_specs": [{"hf_id": "model-a"}],
            "inference_substrate": {"live_llm_inference": True},
        },
        Path("results/live.json"),
    )[0]["missing_model_ids"] == ["model-b"]
    assert mod._exact_count_gaps(
        "missing-minimum",
        {"exact_ground_truth_count": 48},
        Path("results/live.json"),
    )[0]["reason"] == "minimum_live_eval_count missing for .289 protocol"
    clean_live = mod._live_llm_status_and_gaps(
        "live-clean",
        {
            "ready": True,
            "mandatory_headline_model_ids": ["model-a"],
            "model_specs": [{"hf_id": "model-a"}],
            "exact_ground_truth_count": 48,
            "minimum_live_eval_count": 48,
            "inference_substrate": {"live_llm_inference": True},
        },
        Path("results/live.json"),
        "ready",
    )
    assert clean_live == ("clean", [])
    flagged_live = mod._live_llm_status_and_gaps(
        "live-flagged",
        {
            "ready": True,
            "mandatory_headline_model_ids": ["model-a"],
            "model_specs": [{"hf_id": "model-a"}],
            "exact_ground_truth_count": 48,
            "minimum_live_eval_count": 48,
            "flagged_adversarial": True,
            "inference_substrate": {"live_llm_inference": True},
        },
        Path("results/live.json"),
        "ready",
    )
    assert flagged_live[0] == "flagged"
    blocked_live = mod._live_llm_status_and_gaps(
        "live-blocked",
        {
            "ready": False,
            "mandatory_headline_model_ids": ["model-a"],
            "model_specs": [{"hf_id": "model-a"}],
            "exact_ground_truth_count": 48,
            "minimum_live_eval_count": 48,
            "inference_substrate": {"live_llm_inference": True},
        },
        Path("results/live.json"),
        "ready",
    )
    assert blocked_live[0] == "blocked"
    too_few = mod._live_llm_status_and_gaps(
        "live-count-gap",
        {
            "ready": True,
            "mandatory_headline_model_ids": ["model-a"],
            "model_specs": [{"hf_id": "model-a"}],
            "exact_ground_truth_count": 12,
            "minimum_live_eval_count": 48,
            "inference_substrate": {"live_llm_inference": True},
        },
        Path("results/live.json"),
        "ready",
    )
    assert too_few[0] == "flagged"
    assert too_few[1][0]["reason"] == "exact fixture count below .289 protocol minimum"

    assert mod._gate_status({}) == "missing"
    assert mod._gate_status({"status": "blocked"}) == "gated_skipped"
    assert mod._gate_status({"gates_evaluated": [{"passed": True}]}) == "clean"
    assert mod._repair_panel_status({}) == "missing"
    assert mod._repair_panel_status({"gated_structured_repair_micro_panel_ready": True}) == "clean"
    assert mod._repair_panel_status({"gated_structured_repair_micro_panel_ready": False}) == "blocked"

    violations = mod._invariant_violations(
        {"matrix_v22_ready": False},
        {"capstone_ready": False},
        {"blocker_triage_ready": False},
        [_row("flagged", "flagged", "scope", "evidence")],
        {"clean": 0},
        [],
        [],
    )
    assert violations == [
        "matrix v22 authority is not ready",
        "capstone .288 authority is not ready",
        "Exp 3096 blocker triage ledger is not ready",
        "status_counts keys do not match required v23 statuses",
        "status_counts do not sum to rows_total",
        "publication_blocker_count does not match row statuses",
    ]

"""Tests for Exp 3093 cross-corpus matrix v22.

Spec refs: REQ-REPORT-3093, SCENARIO-REPORT-3093.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v22_3093 as mod


REQUIRED_FIELDS = {
    "matrix_v22_ready",
    "rows_total",
    "status_counts",
    "publication_blocker_count",
    "blocker_delta_from_v21",
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
    }


def _matrix_v21() -> dict[str, Any]:
    rows = [
        _row("clean-carry", "clean", "milestone_activation", "archive"),
        _row("verifier-old", "flagged", "local_sota_solution_verifier_gain", "panel"),
        _row("dot287:exp3070_first_token_abstention", "flagged", "verifier_gain", "panel"),
        _row(
            "dot287:exp3072_verifier_calibration_gate",
            "gated_skipped",
            "verifier_gain_recovery_gate",
            "gate",
        ),
        _row("repair-old", "flagged", "repair_live_rerun", "repair"),
        _row("dot287:exp3071_verge_mcs_feedback", "flagged", "formal_feedback", "feedback"),
        _row("dot287:exp3075_repair_micro_panel", "gated_skipped", "repair_live_rerun", "repair"),
        _row("fr11-a", "flagged", "controller_only_online_learning_budget", "fr11"),
        _row("fr11-b", "bounded", "controller_only_self_learning", "fr11"),
        _row("hardware-old", "blocked", "hardware_rerun_gate", "gatemate"),
        _row("dot287:exp3078_gatemate_operator_refresh", "blocked", "hardware_rerun_gate", "gatemate"),
        _row(
            "dot287:exp3078_ssqa_readback_refresh",
            "gated_skipped",
            "host_visible_readback_gate",
            "ssqa",
        ),
        _row("adapter-old", "projection_only", "future_adapter_context", "ebt_arm"),
        _row("dot287:exp3073_ebt_arm_adapter_feasibility", "projection_only", "future_adapter_context", "ebt_arm"),
        _row("capstone:v286_paper_readiness", "bounded", "paper_readiness", "capstone"),
        _row("missing-old", "missing", "prior_v18_carry_forward", "matrix"),
        _row("retired-old", "retired", "retired_claim", "repair"),
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
        if row["status"] not in {"clean", "retired"}
    ]
    return {
        "artifact": "experiment_3079_cross_corpus_matrix_v21",
        "matrix_v21_ready": True,
        "rows_total": len(rows),
        "status_counts": {status: sum(row["status"] == status for row in rows) for status in mod.STATUSES},
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "rows": rows,
        "honest_verdict": "complete: matrix_v21_ready=true",
    }


def _ledger() -> dict[str, Any]:
    categories = {category: [] for category in mod.LEDGER_CATEGORIES}
    categories["verifier_gain"] = [
        {"row_id": "verifier-old", "status": "flagged"},
        {"row_id": "dot287:exp3070_first_token_abstention", "status": "flagged"},
        {"row_id": "dot287:exp3072_verifier_calibration_gate", "status": "gated_skipped"},
    ]
    categories["repair_gate"] = [
        {"row_id": "repair-old", "status": "flagged"},
        {"row_id": "dot287:exp3071_verge_mcs_feedback", "status": "flagged"},
        {"row_id": "dot287:exp3075_repair_micro_panel", "status": "gated_skipped"},
    ]
    categories["fr11_budget"] = [
        {"row_id": "fr11-a", "status": "flagged"},
        {"row_id": "fr11-b", "status": "bounded"},
    ]
    categories["hardware_evidence"] = [
        {"row_id": "hardware-old", "status": "blocked"},
        {"row_id": "dot287:exp3078_gatemate_operator_refresh", "status": "blocked"},
        {"row_id": "dot287:exp3078_ssqa_readback_refresh", "status": "gated_skipped"},
    ]
    categories["adapter_projection"] = [
        {"row_id": "adapter-old", "status": "projection_only"},
        {"row_id": "dot287:exp3073_ebt_arm_adapter_feasibility", "status": "projection_only"},
    ]
    categories["bounded_status"] = [{"row_id": "capstone:v286_paper_readiness", "status": "bounded"}]
    categories["missing_artifact"] = [{"row_id": "missing-old", "status": "missing"}]
    categories["retired_status"] = [{"row_id": "retired-old", "status": "retired"}]
    return {
        "artifact": "experiment_3082_publication_blocker_reduction_ledger_v1",
        "blocker_ledger_ready": True,
        "publication_blocker_count_before": 15,
        "blocker_categories": categories,
        "honest_verdict": "complete: blocker_ledger_ready=true",
    }


def _capstone_v287() -> dict[str, Any]:
    return {
        "artifact": "experiment_3080_capstone_v287",
        "capstone_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 15,
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_sources(root: Path, *, omit_3089: bool = True) -> None:
    _write_json(root, mod.MATRIX_V21_REL_PATH, _matrix_v21())
    _write_json(root, mod.CAPSTONE_V287_REL_PATH, _capstone_v287())
    _write_json(root, mod.EXP3082_REL_PATH, _ledger())
    _write_json(
        root,
        mod.EXP3081_REL_PATH,
        {"archive_v287_activate_v288_ready": True, "honest_verdict": "complete: archive=true"},
    )
    _write_json(
        root,
        mod.EXP3083_REL_PATH,
        {"verifier_hardness_protocol_ready": True, "honest_verdict": "complete: protocol=true"},
    )
    _write_json(
        root,
        mod.EXP3084_REL_PATH,
        {"resyn_fixture_bank_ready": True, "exact_fixture_count": 72, "honest_verdict": "complete: fixtures=true"},
    )
    _write_json(
        root,
        mod.EXP3085_REL_PATH,
        {
            "abstention_panel_v2_ready": True,
            "abstention_precision": 0.0,
            "abstention_precision_reaches_0_7": False,
            "flagged_adversarial": True,
            "mandatory_headline_model_ids": ["model-a", "model-b"],
            "model_specs": [{"hf_id": "model-a"}],
            "inference_substrate": {"live_llm_inference": True},
            "honest_verdict": "complete_below_gate: abstention_panel_v2_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3086_REL_PATH,
        {
            "formal_feedback_ready": False,
            "guided_success_count": 0,
            "solver_only_success_count": 0,
            "flagged_adversarial": True,
            "mandatory_headline_model_ids": ["model-c"],
            "model_specs": [],
            "inference_substrate": {"kind": "live_llm_inference_plus_z3", "live_llm_inference": True},
            "honest_verdict": "complete: formal_feedback_ready=false",
        },
    )
    _write_json(
        root,
        mod.EXP3087_REL_PATH,
        {"schema": "blocked_gate_check_v1", "status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
    )
    _write_json(
        root,
        mod.EXP3088_REL_PATH,
        {"structured_generation_ready": True, "honest_verdict": "complete: structured=true"},
    )
    if not omit_3089:
        _write_json(
            root,
            mod.EXP3089_REL_PATH,
            {"repair_panel_ready": False, "flagged_adversarial": True, "honest_verdict": "complete_below_gate"},
        )
    _write_json(
        root,
        mod.EXP3090_REL_PATH,
        {
            "fr11_resyn_kancl_ready": True,
            "budget_gates": {
                "all_gates_passed": True,
                "soundness_mistakes": {"observed": 0, "passed": True},
                "completeness_mistakes": {"observed": 0, "passed": True},
                "controls_non_vacuous": {"observed": True, "passed": True},
            },
            "promotion_decision": "controller_only_resyn_kancl_budget_passed",
            "honest_verdict": "complete_fr11_resyn_kancl_controller_only_ready",
        },
    )
    _write_json(
        root,
        mod.EXP3091_REL_PATH,
        {
            "adapter_schema_ready": True,
            "sidecar_replay_scorer_ready": True,
            "implementation_claim_boundary": "prototype only; no live model inference integration",
            "honest_verdict": "complete: sidecar prototype only",
        },
    )
    _write_json(
        root,
        mod.EXP3092_REL_PATH,
        {
            "operator_evidence_ingestion_ready": True,
            "gatemate_rerun_allowed": False,
            "ssqa_readback_allowed": False,
            "missing_operator_actions": [{"missing_item": "host_visible_smoke_evidence"}],
            "honest_verdict": "complete: operator_evidence_ingestion_ready=true",
        },
    )


def test_req_report_3093_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3093: OpenSpec declares the v22 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3093" in spec
    assert "SCENARIO-REPORT-3093" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3093_builds_matrix_v22_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3093: completed, blocked, gated, bounded, and projection rows stay separate."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    rows = {row["row_id"]: row for row in artifact["rows"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v22_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["rows_total"] == len(artifact["rows"]) == 31
    assert artifact["status_counts"] == {
        "clean": 9,
        "flagged": 4,
        "bounded": 1,
        "blocked": 2,
        "gated_skipped": 2,
        "missing": 2,
        "retired": 9,
        "projection_only": 2,
    }
    assert artifact["publication_blocker_count"] == 13
    assert artifact["blocker_delta_from_v21"] == -2
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["fr11-a"]["status"] == "clean"
    assert rows["fr11-b"]["status"] == "clean"
    assert rows["dot287:exp3070_first_token_abstention"]["status"] == "retired"
    assert rows["dot288:exp3085_icalm_task_abstention_sota_panel"]["status"] == "flagged"
    assert rows["dot288:exp3087_local_sota_verifier_calibration_gate"]["status"] == "gated_skipped"
    assert rows["dot288:exp3089_xgrammar_sota_repair_micro_panel"]["status"] == "missing"
    assert rows["dot288:exp3091_ebt_arm_sidecar_adapter_schema_prototype"]["status"] == "projection_only"
    assert rows["dot288:exp3092_gatemate_operator_evidence"]["status"] == "blocked"
    assert rows["dot288:exp3092_ssqa_readback_evidence"]["status"] == "gated_skipped"

    reconciliation = artifact["blocker_reconciliation_from_ledger"]
    assert reconciliation["publication_blocker_count_before"] == 15
    assert reconciliation["publication_blocker_count_after"] == 13
    assert reconciliation["increases"] == []
    assert reconciliation["decreases"] == [
        {
            "count": 2,
            "ledger_category": "fr11_budget",
            "row_ids": ["fr11-a", "fr11-b"],
            "reason": "Exp 3090 passed zero-soundness, zero-completeness, non-vacuous controller-only FR-11 gates.",
        }
    ]
    assert reconciliation["neutral_replacements"][0]["old_row_id"] == "capstone:v286_paper_readiness"

    assert artifact["missing_artifacts"] == [
        {
            "path": mod.EXP3089_REL_PATH.as_posix(),
            "reason": "expected .288 repair micro-panel artifact is absent",
        }
    ]
    assert {gap["row_id"] for gap in artifact["headline_model_spec_gaps"]} == {
        "dot288:exp3085_icalm_task_abstention_sota_panel",
        "dot288:exp3086_dafny_z3_formal_feedback_pilot",
    }
    assert mod.OUTPUT_REL_PATH.as_posix() in artifact["capstone_input_artifacts"]
    assert mod.EXP3092_REL_PATH.as_posix() in artifact["capstone_input_artifacts"]
    assert sources[mod.EXP3089_REL_PATH.as_posix()]["present"] is False
    assert sources[mod.MATRIX_V21_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V21_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v21_capstone_v287_ledger_and_dot288_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def test_req_report_3093_blocks_missing_authorities(tmp_path: Path) -> None:
    """REQ-REPORT-3093: missing authority artifacts block matrix readiness."""

    artifact = mod.build_artifact(tmp_path)

    assert artifact["matrix_v22_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_matrix_v22_preconditions")
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v21"] == 0
    assert [row["path"] for row in artifact["required_source_errors"]] == [
        mod.MATRIX_V21_REL_PATH.as_posix(),
        mod.CAPSTONE_V287_REL_PATH.as_posix(),
        mod.EXP3082_REL_PATH.as_posix(),
    ]


def test_req_report_3093_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3093: helper edges are deterministic and fail closed."""

    _write_sources(tmp_path, omit_3089=False)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v22_ready"] is True
    assert saved["missing_artifacts"] == []
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("bad") == "missing"
    assert mod.blocker_class("clean") == "none"
    assert mod.blocker_class("projection_only") == "projection_only"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._float_or_none(True) is None
    assert mod._float_or_none("bad") is None
    assert mod._source_payload(tmp_path, mod.SourceSpec("bad", Path("bad.json"), "bad"))["readable_json_object"] is False
    assert mod._carry_forward_rows({"rows": [1, _row("carry", "clean", "scope", "evidence")]}, set())[0]["row_id"] == "carry"
    assert mod._replacement_row_id("not-replaced") == ""

    clean_live = mod._live_llm_row(
        "live-clean",
        {"ready": True, "mandatory_headline_model_ids": ["m"], "model_specs": [], "inference_substrate": {"live_llm_inference": True}},
        Path("results/live.json"),
        ready_field="ready",
        evidence_class="live",
        claim_scope="headline",
    )
    assert clean_live["status"] == "flagged"
    assert clean_live["summary"]["model_spec_gate_passed"] is False
    blocked_live = mod._live_llm_row(
        "live-blocked",
        {"ready": False, "mandatory_headline_model_ids": ["m"], "model_specs": [{"hf_id": "m"}], "inference_substrate": {"live_llm_inference": True}},
        Path("results/live.json"),
        ready_field="ready",
        evidence_class="live",
        claim_scope="headline",
    )
    assert blocked_live["status"] == "blocked"
    assert mod._model_spec_gaps("non-live", {"mandatory_headline_model_ids": ["m"]}, Path("x")) == []
    assert mod._model_spec_gaps(
        "live-no-mandate",
        {"model_specs": [{"name": "m"}], "inference_substrate": {"live_llm_inference": True}},
        Path("x"),
    )[0]["reason"] == "mandatory_headline_model_ids missing for live LLM artifact"

    assert mod._fr11_repair_passed({"budget_gates": {"all_gates_passed": False}}) is False
    assert mod._fr11_repair_passed({"budget_gates": {"all_gates_passed": True}}) is False
    assert mod._gate_status({}) == "missing"
    assert mod._gate_status({"status": "blocked"}) == "gated_skipped"
    assert mod._gate_status({"status": "ok", "gates_evaluated": [{"passed": True}]}) == "clean"
    assert mod._repair_panel_status({"repair_panel_ready": True}) == "clean"
    assert mod._repair_panel_status({"repair_panel_ready": True, "flagged_adversarial": True}) == "flagged"

    increased = mod._blocker_reconciliation(
        before_count=1,
        after_count=3,
        fr11_clear_ids=set(),
        rows=[],
    )
    assert increased["increases"] == [
        {
            "count": 2,
            "reason": "New v22 blocker rows exceeded retired or cleaned v21 blockers.",
        }
    ]

    violations = mod._invariant_violations(
        {"matrix_v21_ready": False},
        {"capstone_ready": False},
        {"blocker_ledger_ready": False},
        [_row("flagged", "flagged", "scope", "evidence")],
        {"clean": 0},
        [],
        [],
    )
    assert violations == [
        "matrix v21 authority is not ready",
        "capstone .287 authority is not ready",
        "Exp 3082 blocker ledger is not ready",
        "status_counts keys do not match required v22 statuses",
        "status_counts do not sum to rows_total",
        "publication_blocker_count does not match row statuses",
    ]

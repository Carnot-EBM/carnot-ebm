"""Tests for Exp 3080 milestone .287 capstone.

Spec refs: REQ-REPORT-3080, SCENARIO-REPORT-3080.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v287_3080 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "verifier_gain_status",
    "repair_claim_status",
    "fr11_self_learning_status",
    "ebt_arm_status",
    "gatemate_status",
    "ssqa_status",
    "publication_blocker_count",
    "publication_blockers",
    "next_milestone_recommendation",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
COUNT_FIELDS = {
    "clean": "clean_rows",
    "flagged": "flagged_rows",
    "bounded": "bounded_rows",
    "blocked": "blocked_rows",
    "gated_skipped": "gated_skipped_rows",
    "projection_only": "projection_only_rows",
    "missing": "missing_rows",
    "retired": "retired_rows",
}
EXPECTED_RECOMMENDATION = (
    "2026.05.288: reduce publication_blocker_count from 13 by first clearing "
    "verifier-gain recovery (fix or retire Exp3070 adversarial flags, raise "
    "abstention_precision to the gate, rerun Exp3072), then run Exp3075 repair "
    "micro-panel only after that gate passes; in parallel commit the GateMate "
    "output contract and host-visible smoke transcript so SSQA can leave "
    "gate-skipped status; keep FR-11 controller-only until the completeness "
    "budget is zero and keep EBT/ARM-EBM projection-only until an adapter "
    "implementation has tests."
)


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    row_id: str,
    status: str,
    evidence_class: str,
    claim_scope: str,
    source_artifact: str | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": source_artifact or f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": evidence_class,
        "blocker_class": mod.blocker_class(status),
        "claim_scope": claim_scope,
        "summary": {"status": status},
    }


def _matrix_v21(*, clean: bool = False) -> dict[str, Any]:
    if clean:
        rows = [
            _row("dot287:exp3070_verifier_gain_clean", "clean", "verifier_gain", "verifier_gain_recovery"),
            _row("dot287:repair_clean", "clean", "repair_rerun", "repair_live_rerun"),
            _row("dot287:fr11_controller_clean", "clean", "fr11_controller", "controller_only_online_learning"),
            _row("dot287:exp3073_ebt_arm_adapter", "clean", "ebt_arm_adapter", "adapter_implementation"),
            _row("dot287:gatemate_clean", "clean", "gatemate_output", "hardware_rerun_gate"),
            _row("dot287:ssqa_clean", "clean", "ssqa_readback", "host_visible_readback_gate"),
            _row("dot287:retired_repair", "retired", "repair_retired_claim", "retired_repair"),
        ]
    else:
        rows = [
            _row("dot287:archive_activation", "clean", "archive_activation", "milestone_activation"),
            _row("dot287:exp3069_solver_verifier_autopsy", "clean", "solver_verifier_failure_autopsy", "recovery_protocol"),
            _row("dot287:exp3074_repair_protocol", "clean", "grammar_constrained_repair_protocol", "repair_rerun_protocol"),
            _row("dot287:exp3070_first_token_abstention", "flagged", "first_token_abstention_sota_panel", "local_sota_solution_verifier_gain"),
            _row("dot287:exp3071_verge_mcs_feedback", "flagged", "verge_mcs_smt_feedback_pilot", "solver_grounded_repair_feedback"),
            _row("dot287:exp3076_fr11_budget", "flagged", "fr11_soundness_completeness_budget", "controller_only_online_learning_budget"),
            _row("dot287:exp3077_fr11_soundness_bounded_pilot", "flagged", "fr11_soundness_bounded_online_learning_pilot", "controller_only_online_learning"),
            _row("v20:repair:headline_status", "bounded", "repair_bounded_claim", "repair_headline_boundary"),
            _row("capstone:v286_paper_readiness", "bounded", "capstone_v286_authority", "paper_readiness", mod.CAPSTONE_V286_REL_PATH.as_posix()),
            _row("v20:repair:de_tautology_disqualifiers", "blocked", "repair_promotion_disqualifiers", "repair_headline_boundary"),
            _row("dot287:exp3078_gatemate_operator_refresh", "blocked", "gatemate_no_rerun_operator_refresh", "hardware_rerun_gate"),
            _row("dot287:exp3072_verifier_calibration_gate", "gated_skipped", "local_sota_verifier_calibration_gate", "verifier_gain_recovery_gate"),
            _row("dot287:exp3075_repair_micro_panel", "gated_skipped", "gated_grammar_constrained_repair_micro_panel", "repair_live_rerun", mod.CONDUCTOR_LOG_REL_PATH.as_posix()),
            _row("dot287:exp3078_ssqa_readback_refresh", "gated_skipped", "ssqa_no_rerun_operator_refresh", "host_visible_readback_gate"),
            _row("dot287:exp3073_ebt_arm_adapter_feasibility", "projection_only", "ebt_arm_adapter_feasibility_audit", "future_adapter_context"),
            _row("v20:v19:gatemate:host_visible_smoke", "missing", "host_visible_transcript", "host_visible_hardware_transcript"),
            _row("v20:repair:headline_sota_repair_clean_methodology", "retired", "repair_retired_claim", "retired_repair_headline_wording"),
        ]
        rows[3]["summary"].update(
            {
                "abstention_precision": 0.5,
                "expected_abstention_precision_min": 0.7,
                "flagged_adversarial": True,
                "verifier_gain_delta_with_abstention": 0.5,
            }
        )
        rows[4]["summary"].update(
            {
                "guided_success_count": 5,
                "solver_only_success_count": 5,
                "guided_lift_positive": False,
                "flagged_adversarial": True,
            }
        )
        rows[6]["summary"].update(
            {
                "all_gates_passed": False,
                "soundness_mistakes": 0,
                "completeness_mistakes": 1,
                "promotion_decision": "controller_only_budget_exceeded_no_stronger_promotion",
            }
        )
        rows[14]["summary"].update(
            {
                "adapter_feasible": True,
                "adapter_implementation_claimed": False,
                "status_rationale": "future_context_only_no_adapter_implementation_claimed",
            }
        )

    counts = {status: sum(1 for row in rows if row["status"] == status) for status in COUNT_FIELDS}
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
        **{COUNT_FIELDS[status]: count for status, count in counts.items()},
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "rows": rows,
        "source_artifacts": [
            {
                "experiment_id": "extra",
                "path": "results/extra_source.json",
                "role": "extra_matrix_context",
            }
        ],
        "honest_verdict": "complete: matrix_v21_ready=true",
    }


def _write_sources(tmp_path: Path, *, clean: bool = False, omit_capstone: bool = False) -> None:
    matrix = _matrix_v21(clean=clean)
    _write_json(tmp_path, mod.MATRIX_V21_REL_PATH, matrix)
    if not omit_capstone:
        _write_json(
            tmp_path,
            mod.CAPSTONE_V286_REL_PATH,
            {
                "artifact": "experiment_3066_capstone_v286",
                "capstone_ready": True,
                "paper_ready": False,
                "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
            },
        )
    for row in matrix["rows"]:
        rel_path = Path(row["source_artifact"])
        if rel_path == mod.CAPSTONE_V286_REL_PATH or rel_path.as_posix().endswith("host_visible_smoke.json"):
            continue
        if rel_path == mod.CONDUCTOR_LOG_REL_PATH:
            _write_text(tmp_path, rel_path, "| gate skip |\n")
        elif rel_path.suffix == ".json":
            _write_json(tmp_path, rel_path, {"artifact": row["row_id"], "status": row["status"]})


def test_req_report_3080_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3080: OpenSpec declares the .287 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3080" in spec
    assert "SCENARIO-REPORT-3080" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3080_builds_capstone_from_matrix_v21(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3080: dirty matrix v21 rows block paper readiness."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=4.25)
    checks = {row["check"]: row for row in artifact["paper_ready_checks"]}
    summary = artifact["matrix_v21_summary"]

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_gain_status"] == "flagged_or_gated_verifier_gain_recovery_incomplete"
    assert artifact["repair_claim_status"] == "bounded_and_gated_skipped"
    assert artifact["fr11_self_learning_status"] == "flagged_controller_only_budget_exceeded"
    assert artifact["ebt_arm_status"] == "projection_only_feasible_no_implementation"
    assert artifact["gatemate_status"] == "blocked_no_rerun_operator_actions_required"
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_smoke_missing"
    assert artifact["publication_blocker_count"] == 13
    assert len(artifact["publication_blockers"]) == 13
    assert artifact["next_milestone_recommendation"] == EXPECTED_RECOMMENDATION

    assert summary["matrix_v21_ready"] is True
    assert summary["rows_total"] == 17
    assert summary["row_count_observed"] == 17
    assert summary["counts_match_rows"] is True
    assert summary["publication_blocker_count_matches"] is True
    assert summary["status_counts"]["flagged"] == 4
    assert summary["status_by_row"]["dot287:exp3073_ebt_arm_adapter_feasibility"] == "projection_only"

    assert checks["capstone_ready"]["passed"] is True
    assert checks["matrix_has_no_publication_blockers"]["passed"] is False
    assert checks["no_required_claim_has_blocking_status"]["passed"] is False
    assert checks["no_projection_only_publication_claim"]["passed"] is False

    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    assert source_by_path[mod.MATRIX_V21_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V21_REL_PATH
    )
    assert source_by_path[mod.MATRIX_V21_REL_PATH.as_posix()]["required"] is True
    assert source_by_path[mod.CONDUCTOR_LOG_REL_PATH.as_posix()]["source_type"] == "text"
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "matrix_v21_and_checked_in_results",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def test_req_report_3080_sets_paper_ready_only_for_clean_matrix(tmp_path: Path) -> None:
    """REQ-REPORT-3080: paper readiness requires zero blockers and clean required rows."""

    _write_sources(tmp_path, clean=True)

    artifact = mod.build_artifact(tmp_path)
    checks = {row["check"]: row for row in artifact["paper_ready_checks"]}

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["publication_blockers"] == []
    assert artifact["verifier_gain_status"] == "clean_verifier_gain_recovered"
    assert artifact["repair_claim_status"] == "clean_or_retired"
    assert artifact["fr11_self_learning_status"] == "clean_controller_only"
    assert artifact["ebt_arm_status"] == "clean_adapter_implementation_evidence"
    assert artifact["gatemate_status"] == "clean_host_visible_output_ready"
    assert artifact["ssqa_status"] == "clean_host_visible_readback_ready"
    assert all(row["passed"] is True for row in checks.values())


def test_req_report_3080_blocks_missing_matrix_and_required_sources(tmp_path: Path) -> None:
    """REQ-REPORT-3080: missing authority artifacts fail closed."""

    blocked_without_matrix = mod.build_artifact(tmp_path)
    assert blocked_without_matrix["capstone_ready"] is False
    assert blocked_without_matrix["paper_ready"] is False
    assert blocked_without_matrix["honest_verdict"] == "blocked_required_matrix_v21_missing"

    _write_sources(tmp_path, omit_capstone=True)
    blocked_missing_capstone = mod.build_artifact(tmp_path)
    assert blocked_missing_capstone["capstone_ready"] is False
    assert blocked_missing_capstone["paper_ready"] is False
    assert blocked_missing_capstone["required_source_errors"] == [
        {"experiment_id": "exp3066", "reason": "missing_or_malformed_required_artifact"}
    ]
    assert blocked_missing_capstone["honest_verdict"].startswith("blocked_capstone_preconditions")


def test_req_report_3080_blocks_count_mismatches(tmp_path: Path) -> None:
    """REQ-REPORT-3080: matrix count mismatches block capstone readiness."""

    matrix = _matrix_v21()
    matrix["flagged_rows"] = 99
    _write_json(tmp_path, mod.MATRIX_V21_REL_PATH, matrix)
    _write_json(tmp_path, mod.CAPSTONE_V286_REL_PATH, {"artifact": "capstone"})

    artifact = mod.build_artifact(tmp_path)

    assert artifact["capstone_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["matrix_v21_summary"]["counts_match_rows"] is False
    assert artifact["honest_verdict"].startswith("blocked_capstone_preconditions")


def test_req_report_3080_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3080: writing and helper edge cases stay deterministic."""

    _write_sources(tmp_path)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.0)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("unknown") == "missing"
    assert mod.blocker_class("projection_only") == "projection_only"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._claim_entry({"row_id": "x", "status": "bad"})["status"] == "missing"


def test_req_report_3080_status_helper_edges() -> None:
    """REQ-REPORT-3080: status helpers keep bounded edge cases explicit."""

    assert (
        mod._verifier_gain_status(
            [_row("verifier:bounded", "bounded", "verifier_gain", "verifier_gain")]
        )
        == "bounded_verifier_gain_recovery_not_promoted"
    )
    assert mod._repair_claim_status([_row("repair:bounded", "bounded", "repair", "repair")]) == (
        "bounded"
    )
    assert mod._repair_claim_status([_row("repair:missing", "missing", "repair", "repair")]) == (
        "missing_repair_evidence"
    )
    assert mod._fr11_self_learning_status([_row("fr11:bounded", "bounded", "fr11", "fr11")]) == (
        "bounded_controller_only"
    )
    assert mod._fr11_self_learning_status([_row("fr11:blocked", "blocked", "fr11", "fr11")]) == (
        "blocked_controller_only"
    )
    assert mod._ebt_arm_status([_row("ebt:blocked", "blocked", "ebt_arm", "future_adapter")]) == (
        "bounded_or_blocked_no_implementation"
    )
    assert mod._gatemate_status([_row("gatemate:bounded", "bounded", "gatemate", "hardware")]) == (
        "bounded_gatemate_claim"
    )
    assert mod._ssqa_status([_row("ssqa:bounded", "bounded", "ssqa", "readback")]) == (
        "bounded_ssqa_claim"
    )

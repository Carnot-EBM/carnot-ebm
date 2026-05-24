"""Tests for Exp 2987 milestone .280 capstone.

Spec refs: REQ-REPORT-2987, SCENARIO-REPORT-2987.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v280_2987 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "milestone",
    "paper_ready",
    "headline_outcome",
    "clean_artifacts",
    "flagged_artifacts",
    "blocked_artifacts",
    "missing_artifacts",
    "gated_skipped_artifacts",
    "pilot_only_artifacts",
    "projection_only_artifacts",
    "gaps_closed",
    "gaps_remaining",
    "repair_ready",
    "solver_ready",
    "fr11_ready",
    "hardware_ready",
    "model_compliance_summary",
    "hardware_claim_boundary_summary",
    "retirement_recommendations",
    "next_milestone_recommendations",
    "inference_substrate",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_row(
    experiment_id: str,
    row_id: str,
    status: str,
    *,
    claim_class: str = "supporting",
    prior_failure_outcome: str = "prior_failure_addressed",
    upstream_flags: list[str] | None = None,
    model_status: str = "not_applicable",
    hardware_status: str = "not_applicable",
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "source_experiment_id": experiment_id,
        "status": status,
        "claim_class": claim_class,
        "evidence_type": "fixture",
        "prior_failure_outcome": prior_failure_outcome,
        "claim_boundary_guard_passed": True,
        "claim_boundary_violations": [],
        "source_honest_verdict": f"{status}: {experiment_id}",
        "upstream_flags": upstream_flags or [],
        "model_compliance": {"status": model_status},
        "hardware_compliance": {"status": hardware_status},
        "summary": summary or {},
    }


def _base_sources(root: Path, *, clean_core: bool = False, matrix_violations: bool = False) -> None:
    repair_clean = clean_core
    solver_flagged = not clean_core
    hardware_clean = clean_core
    exp2976_flagged = not clean_core

    _write_json(
        root,
        mod.EXP2975_REL_PATH,
        {
            "honest_verdict": "complete: archive_ready=true",
            "archive_ready": True,
            "scripts_research_conductor_modified": False,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.EXP2976_REL_PATH,
        {
            "honest_verdict": "complete: protocol ready",
            "intent_preserving_repair_protocol_ready": True,
            "trace_execution_plan_ready": True,
            "prior_failure_addressed": True,
            "mandatory_headline_model_ids": ["mandated"],
            "models_used": ["mandated"] if clean_core else [],
            "flagged_adversarial": exp2976_flagged,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}] if exp2976_flagged else [],
        },
    )
    _write_json(
        root,
        mod.EXP2977_REL_PATH,
        {
            "honest_verdict": "complete: repair clean"
            if repair_clean
            else "blocked_cached_sota_pair_unavailable_cpu_smoke_only",
            "repair_rerun_clean": repair_clean,
            "headline_result": repair_clean,
            "n_tasks": 24 if repair_clean else 2,
            "models_used": ["mandated"] if repair_clean else ["legacy"],
            "mandatory_headline_model_ids": ["mandated"],
            "legacy_model_used_only_for_smoke": not repair_clean,
            "pass_at_1_delta": 0.125 if repair_clean else 0.0,
            "pass_at_k_delta": 0.0,
            "schema_failure_rate_delta": 0.0,
            "syntax_failure_rate_delta": -0.1,
            "false_accept_delta": 0.0,
            "runtime_trace_coverage": 0.9,
        },
    )
    _write_json(
        root,
        mod.EXP2978_REL_PATH,
        {
            "honest_verdict": "complete: diagnostic telemetry only",
            "telemetry_panel_ready": True,
            "semantic_energy_signal_usable": True,
            "first_step_signal_usable": True,
            "no_headline_verifier_claim": True,
            "flagged_adversarial": not clean_core,
            "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING"}] if not clean_core else [],
        },
    )
    _write_json(
        root,
        mod.EXP2979_REL_PATH,
        {
            "honest_verdict": "complete: solver frontier ready",
            "mcs_feedback_schema_ready": True,
            "frontier_upgrade_ready": True,
            "reference_solver_verified_accuracy": 1.0,
            "reference_z3_execution_rate": 1.0,
        },
    )
    _write_json(
        root,
        mod.EXP2980_REL_PATH,
        {
            "honest_verdict": "complete: formalization clean",
            "formalization_feedback_clean": True,
            "headline_result": True,
            "n_items": 12,
            "parseability_rate": 1.0,
            "solver_verified_accuracy": 1.0,
            "answer_accuracy": 1.0,
            "z3_execution_rate": 1.0,
            "tautology_flag_rate": 0.0,
            "feedback_repair_delta": 0.5,
            "models_used": ["mandated"],
            "mandatory_headline_model_ids": ["mandated"],
            "flagged_adversarial": solver_flagged,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}] if solver_flagged else [],
        },
    )
    _write_json(
        root,
        mod.EXP2981_REL_PATH,
        {
            "honest_verdict": "complete: partial monitor promoted",
            "partial_monitor_promoted": True,
            "fixture_count": 11,
            "live_trace_count": 6,
            "prefix_failure_localization_rate": 1.0,
            "false_alarm_rate": 0.0,
            "full_streaming_verification_claim": False,
        },
    )
    _write_json(
        root,
        mod.EXP2982_REL_PATH,
        {
            "honest_verdict": "complete: fr11_independent_self_learning_ready",
            "continuous_self_learning_task": True,
            "fr11_independent_metrics_evaluated": True,
            "fr11_independent_self_learning_ready": True,
            "no_identical_metric_flag": True,
            "forgetting_guard_passed": True,
            "heldout_independent_delta_vs_random": {"pass_at_1": 0.17},
            "negative_control_delta": {"pass_at_1": 0.0},
        },
    )
    _write_json(
        root,
        mod.EXP2983_REL_PATH,
        {
            "honest_verdict": "complete: trace_to_skill_memory_ready",
            "trace_to_skill_memory_ready": True,
            "continuous_self_learning_task": True,
            "heldout_skill_reuse_delta": 0.2,
            "negative_control_delta": 0.0,
            "leakage_flag": False,
            "headline_result": False,
            "fresh_live_llm_inference_used": False,
            "flagged_adversarial": not clean_core,
            "corrigendum_pending": [{"kind": "IMPLAUSIBLE_PERFECT"}] if not clean_core else [],
        },
    )
    _write_json(
        root,
        mod.EXP2984_REL_PATH,
        {
            "honest_verdict": "complete: hardware ready"
            if hardware_clean
            else "complete: gatemate_no_readback_no_host_smoke_io",
            "board_detected": True,
            "flash_succeeded": True,
            "readback_attempted": hardware_clean,
            "readback_supported": hardware_clean,
            "readback_hash": "readback-sha" if hardware_clean else "",
            "smoke_vector_attempted": hardware_clean,
            "smoke_vector_passed": hardware_clean,
            "sampler_claim_allowed": False,
            "speedup_claim_allowed": False,
            "thermodynamic_claim_allowed": False,
            "inference_substrate": "physical_gatemate_board",
        },
    )
    _write_json(
        root,
        mod.EXP2985_REL_PATH,
        {
            "honest_verdict": "complete: projection only",
            "register_map_plan_ready": True,
            "projection_only": True,
            "sampler_claim_allowed": False,
            "speedup_claim_allowed": False,
            "thermodynamic_claim_allowed": False,
        },
    )

    rows = [
        _matrix_row("exp2975", mod.MATRIX_ROW_IDS["exp2975"], "projection-only"),
        _matrix_row(
            "exp2976",
            mod.MATRIX_ROW_IDS["exp2976"],
            "projection-only" if clean_core else "flagged",
            claim_class="repair_protocol",
            upstream_flags=[] if clean_core else ["DURATION_TOO_SHORT:critical"],
            model_status="compliant" if clean_core else "non_compliant_missing_mandated_model",
        ),
        _matrix_row(
            "exp2977",
            mod.MATRIX_ROW_IDS["exp2977"],
            "clean" if repair_clean else "blocked",
            claim_class="repair_eval",
            prior_failure_outcome="repair_rerun_evaluated"
            if repair_clean
            else "blocked_cached_sota_pair_unavailable",
            model_status="compliant" if repair_clean else "legacy_smoke_only",
        ),
        _matrix_row(
            "exp2978",
            mod.MATRIX_ROW_IDS["exp2978"],
            "pilot-only" if clean_core else "flagged",
            claim_class="repair_telemetry",
            upstream_flags=[] if clean_core else ["METHODOLOGY_MISSING:warn"],
        ),
        _matrix_row("exp2979", mod.MATRIX_ROW_IDS["exp2979"], "clean", claim_class="solver_feedback_frontier"),
        _matrix_row(
            "exp2980",
            mod.MATRIX_ROW_IDS["exp2980"],
            "clean" if clean_core else "flagged",
            claim_class="solver_eval",
            prior_failure_outcome="solver_feedback_rerun_evaluated"
            if clean_core
            else "solver_delta_clean_but_adversarially_flagged",
            upstream_flags=[] if clean_core else ["DURATION_TOO_SHORT:critical"],
            model_status="compliant" if clean_core else "flagged_mandated_model_evidence",
        ),
        _matrix_row("exp2981", mod.MATRIX_ROW_IDS["exp2981"], "clean", claim_class="partial_monitor"),
        _matrix_row("exp2982", mod.MATRIX_ROW_IDS["exp2982"], "clean", claim_class="fr11_self_learning"),
        _matrix_row(
            "exp2983",
            mod.MATRIX_ROW_IDS["exp2983"],
            "pilot-only" if clean_core else "flagged",
            claim_class="trace_to_skill_memory",
            upstream_flags=[] if clean_core else ["IMPLAUSIBLE_PERFECT:info"],
            model_status="compliant" if clean_core else "flagged_mandated_model_evidence",
        ),
        _matrix_row(
            "exp2984",
            mod.MATRIX_ROW_IDS["exp2984"],
            "clean" if hardware_clean else "blocked",
            claim_class="hardware_readback_smoke",
            prior_failure_outcome="hardware_readback_or_smoke_ready"
            if hardware_clean
            else "blocked_no_readback_or_host_visible_smoke_io",
            hardware_status="compliant_hardware_readback_or_smoke"
            if hardware_clean
            else "blocked_no_readback_or_smoke_output",
        ),
        _matrix_row(
            "exp2985",
            mod.MATRIX_ROW_IDS["exp2985"],
            "projection-only",
            claim_class="hardware_register_map_plan",
            hardware_status="projection_only",
        ),
    ]
    counts = {status: sum(row["status"] == status for row in rows) for status in mod.CLASSIFICATIONS if status != "missing"}
    violations = (
        [{"row_id": mod.MATRIX_ROW_IDS["exp2984"], "violation": "unsupported_hardware_claim_allowed"}]
        if matrix_violations
        else []
    )
    _write_json(
        root,
        mod.MATRIX_V14_REL_PATH,
        {
            "honest_verdict": "complete: matrix_v14_ready=true",
            "matrix_v14_ready": True,
            "milestone": mod.MILESTONE,
            "rows": rows,
            "row_count": len(rows),
            "clean_count": counts.get("clean", 0),
            "flagged_count": counts.get("flagged", 0),
            "blocked_count": counts.get("blocked", 0),
            "gated_skipped_count": counts.get("gated-skipped", 0),
            "pilot_only_count": counts.get("pilot-only", 0),
            "projection_only_count": counts.get("projection-only", 0),
            "repair_claim_status": "clean: repair evidence cleared v14 gates"
            if repair_clean
            else "blocked: intent-preserving repair rerun did not clear cached-SOTA gates",
            "solver_claim_status": "clean: deterministic frontier and feedback formalization both clear"
            if clean_core
            else "flagged: solver feedback row has clean Z3 metrics but unresolved artifact flags",
            "fr11_claim_status": "clean: independent FR-11 metrics evaluated with no identical-metric flag",
            "hardware_claim_status": "projection-only: hardware plan remains non-sampler evidence"
            if hardware_clean
            else "blocked: board contact exists but readback or smoke output is absent",
            "model_compliance_summary": {
                "compliant": 5 if clean_core else 0,
                "flagged_mandated_model_evidence": 0 if clean_core else 2,
                "legacy_smoke_only": 0 if clean_core else 1,
                "not_applicable": 6,
            },
            "claim_boundary_violations": violations,
            "next_milestone_recommendations": ["Matrix recommendation."],
        },
    )


def test_req_report_2987_spec_anchor_exists() -> None:
    """REQ-REPORT-2987: OpenSpec declares the capstone contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2987" in spec
    assert "SCENARIO-REPORT-2987" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2987_closes_280_without_promoting_flagged_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2987: .280 capstone keeps blocked and flagged rows non-headline."""

    _base_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.75)
    audit_by_id = {row["experiment_id"]: row for row in artifact["artifact_audit"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete: milestone_280_capstone")
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["paper_ready"] is False
    assert artifact["headline_outcome"] == (
        "not_paper_ready: repair=blocked, solver=flagged, fr11=clean, hardware=blocked"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.75)

    assert artifact["clean_artifacts"] == ["exp2979", "exp2981", "exp2982"]
    assert artifact["flagged_artifacts"] == ["exp2976", "exp2978", "exp2980", "exp2983"]
    assert artifact["blocked_artifacts"] == ["exp2977", "exp2984"]
    assert artifact["missing_artifacts"] == []
    assert artifact["gated_skipped_artifacts"] == []
    assert artifact["pilot_only_artifacts"] == []
    assert artifact["projection_only_artifacts"] == ["exp2975", "exp2985", "exp2986"]

    assert artifact["repair_ready"] is False
    assert artifact["solver_ready"] is False
    assert artifact["fr11_ready"] is True
    assert artifact["hardware_ready"] is False
    assert artifact["paper_ready_blockers"] == [
        "repair_not_ready",
        "solver_not_ready",
        "hardware_not_ready",
        "flagged_artifacts_present",
        "blocked_artifacts_present",
    ]

    assert artifact["model_compliance_summary"]["legacy_smoke_only"] == 1
    assert artifact["hardware_claim_boundary_summary"]["gatemate_readback_or_smoke_present"] is False
    assert artifact["hardware_claim_boundary_summary"]["claim_boundary_violations"] == []
    assert audit_by_id["exp2980"]["prior_failure_outcome"] == (
        "solver_delta_clean_but_adversarially_flagged"
    )
    assert audit_by_id["exp2980"]["model_compliance"]["status"] == (
        "flagged_mandated_model_evidence"
    )
    assert audit_by_id["exp2984"]["hardware_compliance"]["status"] == (
        "blocked_no_readback_or_smoke_output"
    )
    assert audit_by_id["exp2982"]["required_claim_fields"]["fr11_independent_metrics_evaluated"] is True
    assert artifact["source_checksums"][mod.EXP2986_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2986_REL_PATH
    )
    assert any("FR-11 independent metric" in gap for gap in artifact["gaps_closed"])
    assert any("Repair is not paper-ready" in gap for gap in artifact["gaps_remaining"])
    assert any("GateMate is not hardware-ready" in gap for gap in artifact["gaps_remaining"])
    assert any("Retire CPU-smoke-only" in item for item in artifact["retirement_recommendations"])
    assert "Matrix recommendation." in artifact["next_milestone_recommendations"]


def test_req_report_2987_all_local_gates_can_be_paper_ready(tmp_path: Path) -> None:
    """REQ-REPORT-2987: paper_ready uses local clean gates and claim boundaries only."""

    _base_sources(tmp_path, clean_core=True)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["paper_ready"] is True
    assert artifact["repair_ready"] is True
    assert artifact["solver_ready"] is True
    assert artifact["fr11_ready"] is True
    assert artifact["hardware_ready"] is True
    assert artifact["paper_ready_blockers"] == []
    assert artifact["flagged_artifacts"] == []
    assert artifact["blocked_artifacts"] == []
    assert artifact["missing_artifacts"] == []
    assert artifact["projection_only_artifacts"] == ["exp2975", "exp2976", "exp2985", "exp2986"]
    assert artifact["pilot_only_artifacts"] == ["exp2978", "exp2983"]
    assert artifact["headline_outcome"].startswith("paper_ready:")


def test_req_report_2987_missing_matrix_is_recorded_without_fabrication(tmp_path: Path) -> None:
    """REQ-REPORT-2987: matrix v14 absence is missing evidence, not inferred readiness."""

    _base_sources(tmp_path, clean_core=True)
    (tmp_path / mod.MATRIX_V14_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    audit_by_id = {row["experiment_id"]: row for row in artifact["artifact_audit"]}

    assert artifact["paper_ready"] is False
    assert artifact["missing_artifacts"] == ["exp2986"]
    assert "missing_artifacts_present" in artifact["paper_ready_blockers"]
    assert artifact["matrix_v14_ready"] is False
    assert audit_by_id["exp2986"]["classification"] == "missing"
    assert audit_by_id["exp2986"]["present"] is False
    assert artifact["source_checksums"][mod.EXP2986_REL_PATH.as_posix()] is None


def test_req_report_2987_claim_boundary_violation_blocks_paper_ready(tmp_path: Path) -> None:
    """REQ-REPORT-2987: hardware claim violations keep paper readiness false."""

    _base_sources(tmp_path, clean_core=True, matrix_violations=True)
    payload = json.loads((tmp_path / mod.EXP2984_REL_PATH).read_text(encoding="utf-8"))
    payload["speedup_claim_allowed"] = True
    _write_json(tmp_path, mod.EXP2984_REL_PATH, payload)

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.75)

    assert artifact["paper_ready"] is False
    assert artifact["hardware_ready"] is False
    assert "claim_boundary_violations_present" in artifact["paper_ready_blockers"]
    assert artifact["hardware_claim_boundary_summary"]["unsupported_claim_fields_by_artifact"] == {
        "exp2984": ["speedup_claim_allowed"]
    }
    assert artifact["hardware_claim_boundary_summary"]["claim_boundary_violations"] == [
        {"row_id": mod.MATRIX_ROW_IDS["exp2984"], "violation": "unsupported_hardware_claim_allowed"}
    ]


def test_req_report_2987_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-2987: write_artifact emits the required deliverable JSON."""

    _base_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=4.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["milestone"] == mod.MILESTONE
    assert saved["paper_ready"] is False
    assert saved["duration_s"] == pytest.approx(0.5)
    assert saved["artifact_classification_counts"]["flagged"] == 4


def test_req_report_2987_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-2987: helpers preserve malformed inputs and conservative gates."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._matrix_rows_by_experiment({"rows": [1, {"source_experiment_id": "expX"}]}) == {
        "expX": {"source_experiment_id": "expX"}
    }
    assert mod._blocked_verdict("gate_blocked_precondition") is True
    assert mod._blocked_verdict("complete: ok") is False
    assert mod._has_flags({"flagged_adversarial": True}) is True
    assert mod._has_flags({"corrigendum_pending": [{"kind": "X"}]}) is True
    assert mod._has_flags({}) is False
    assert mod._flag_kinds(
        {"flagged_adversarial": True, "corrigendum_pending": [{"kind": "X"}, "bad"]}
    ) == ["flagged_adversarial=true", "X"]
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
    assert mod._string_list("not-list") == []
    assert mod._unique_strings(["a", "b", "a"]) == ["a", "b"]
    assert mod._classify_artifact("expX", {"honest_verdict": "blocked"}, {}, True) == "blocked"
    assert mod._classify_artifact("expX", {"flagged_adversarial": True}, {}, True) == "flagged"
    assert mod._status_from_payload("exp2975", {"archive_ready": True}) == "projection-only"
    assert mod._status_from_payload("exp2976", {"intent_preserving_repair_protocol_ready": True}) == (
        "projection-only"
    )
    assert mod._status_from_payload("exp2978", {"telemetry_panel_ready": True}) == "pilot-only"
    assert mod._status_from_payload("exp2983", {"trace_to_skill_memory_ready": True}) == "pilot-only"
    assert mod._status_from_payload("exp2985", {"register_map_plan_ready": True}) == "projection-only"
    assert mod._status_from_payload("exp2986", {"matrix_v14_ready": True}) == "projection-only"
    assert mod._status_from_payload("exp2977", {"repair_rerun_clean": True}) == "clean"
    assert mod._status_from_payload("exp2980", {"formalization_feedback_clean": True}) == "clean"
    assert mod._status_from_payload("exp2982", {"fr11_independent_self_learning_ready": True}) == "clean"
    assert mod._status_from_payload("exp2984", {"smoke_vector_passed": True}) == "clean"
    assert mod._status_from_payload("unknown", {}) == "blocked"
    assert mod._model_compliance_from_payload(
        {"mandatory_headline_model_ids": ["m"], "models_used": ["m"], "legacy_model_used_only_for_smoke": True}
    )["status"] == "legacy_smoke_only"
    assert mod._model_compliance_from_payload(
        {"mandatory_headline_model_ids": ["m"], "models_used": []}
    )["status"] == "non_compliant_missing_mandated_model"
    assert mod._model_compliance_from_payload(
        {"mandatory_headline_model_ids": ["m"], "models_used": ["m"], "flagged_adversarial": True}
    )["status"] == "flagged_mandated_model_evidence"
    assert mod._hardware_compliance_from_payload({"speedup_claim_allowed": True})["status"] == (
        "claim_boundary_violation"
    )
    assert mod._hardware_compliance_from_payload({"inference_substrate": "physical_gatemate_board"})[
        "status"
    ] == "blocked_no_readback_or_smoke_output"
    assert mod._classification_for("missing", []) == ""
    buckets = {name: [] for name in mod.CLASSIFICATIONS}
    buckets["gated-skipped"] = ["expX"]
    assert "fr11_not_ready" in mod._paper_ready_blockers(
        True,
        True,
        False,
        True,
        buckets,
        {"claim_boundary_violations": [], "unsupported_claim_fields_by_artifact": {}},
        True,
    )
    assert "gated_skipped_artifacts_present" in mod._paper_ready_blockers(
        True,
        True,
        True,
        True,
        buckets,
        {"claim_boundary_violations": [], "unsupported_claim_fields_by_artifact": {}},
        True,
    )
    gaps = mod._gaps_remaining(False, False, False, False, buckets, {"matrix_v14_ready": True})
    assert any("FR-11 is not paper-ready" in item for item in gaps)
    assert any("Gated-skipped .280 artifacts remain" in item for item in gaps)
    assert mod._get_path({"a": 1}, "a.b") is None
    assert mod._get_path({"a": {"b": 2}}, "a.b") == 2

"""Tests for Exp 2974 milestone .279 capstone.

Spec refs: REQ-REPORT-2974, SCENARIO-REPORT-2974.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v279_2974 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "milestone",
    "paper_ready",
    "headline_outcome",
    "clean_artifacts",
    "flagged_artifacts",
    "blocked_artifacts",
    "gated_skipped_artifacts",
    "missing_artifacts",
    "pilot_only_artifacts",
    "aggregation_only_artifacts",
    "gaps_closed",
    "gaps_remaining",
    "forbidden_claims_absent",
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


def _write_roadmap(root: Path) -> None:
    (root / mod.ROADMAP_REL_PATH).write_text(
        """
milestone: "2026.05.279"
tasks:
  - id: exp2962
    deliverable: "results/experiment_2962_archive_v278_activate_v279.json"
    title: "Archive .278 + Activate .279"
    inference_substrate: aggregation_from_upstream_artifacts
  - id: exp2963
    deliverable: "results/experiment_2963_dccd_repair_protocol_manifest_v1.json"
    title: "DCCD Structured Repair Protocol Manifest v1"
    inference_substrate: aggregation_from_upstream_artifacts
  - id: exp2964
    deliverable: "results/experiment_2964_sota_dccd_repair_replication_v1.json"
    title: "Gated SOTA DCCD Code Repair Replication v1"
    inference_substrate: live_llm_inference
    gated_on:
      - upstream: exp2963
        artifact_field: dccd_repair_protocol_ready
        op: "=="
        value: true
  - id: exp2965
    deliverable: "results/experiment_2965_beaver_style_repair_certificate_v1.json"
    title: "BEAVER-Style Repair Certificate Audit v1"
    inference_substrate: deterministic_wiring
    gated_on:
      - upstream: exp2963
        artifact_field: dccd_repair_protocol_ready
        op: "=="
        value: true
  - id: exp2966
    deliverable: "results/experiment_2966_logic_frontier_materializer_v1.json"
    title: "Logic Frontier Materializer v1"
    inference_substrate: deterministic_wiring
  - id: exp2967
    deliverable: "results/experiment_2967_sota_nl_to_z3_dccd_formalization_v1.json"
    title: "Gated SOTA NL-to-Z3 DCCD Formalization v1"
    inference_substrate: live_llm_inference
    gated_on:
      - upstream: exp2966
        artifact_field: logic_frontier_materialized
        op: "=="
        value: true
  - id: exp2968
    deliverable: "results/experiment_2968_interwhen_partial_monitor_harness_v1.json"
    title: "Interwhen Partial Monitor Harness v1"
    inference_substrate: deterministic_wiring
  - id: exp2969
    deliverable: "results/experiment_2969_fr11_non_tautological_utility_gate_v3.json"
    title: "FR-11 Non-Tautological Utility Gate v3"
    inference_substrate: aggregation_from_upstream_artifacts
  - id: exp2970
    deliverable: "results/experiment_2970_kan_forgetting_guard_memory_audit_v1.json"
    title: "KAN Forgetting Guard Memory Audit v1"
    inference_substrate: deterministic_wiring
    gated_on:
      - upstream: exp2969
        artifact_field: non_tautological_self_learning_ready
        op: "=="
        value: true
  - id: exp2971
    deliverable: "results/experiment_2971_gatemate_board_detection_flash_harness_v3.json"
    title: "GateMate Board Detection Flash Harness v3"
    inference_substrate: hardware_preflight
  - id: exp2972
    deliverable: "results/experiment_2972_gatemate_post_flash_output_hash_v3.json"
    title: "GateMate Post-Flash Output Hash v3"
    inference_substrate: hardware_smoke
    gated_on:
      - upstream: exp2971
        artifact_field: gatemate_board_detected
        op: "=="
        value: true
      - upstream: exp2971
        artifact_field: bitstream_sha256_verified
        op: "=="
        value: true
  - id: exp2973
    deliverable: "results/experiment_2973_cross_corpus_matrix_v13.json"
    title: "Cross-Corpus Matrix v13"
    inference_substrate: aggregation_from_upstream_artifacts
    gated_on:
      - upstream: exp2969
        artifact_field: non_tautological_self_learning_ready
        op: "=="
        value: true
  - id: exp2974
    deliverable: "results/experiment_2974_capstone_v279.json"
    title: "Capstone .279"
    inference_substrate: aggregation_from_upstream_artifacts
""".lstrip(),
        encoding="utf-8",
    )


def _write_ready_sources(root: Path) -> None:
    _write_roadmap(root)
    _write_json(
        root,
        mod.CAPSTONE_V278_REL_PATH,
        {
            "honest_verdict": "complete: milestone_278_capstone; paper_ready=false",
            "paper_ready": False,
            "forbidden_claims_absent": True,
            "gaps_remaining": [
                "Taxonomy-guided repair delta remains flagged.",
                "FR-11 utility-gated self-learning remains flagged.",
                "NL-to-Z3 solver execution repair remains flagged.",
                "GateMate flash/timing smoke remains blocked.",
            ],
        },
    )
    _write_json(
        root,
        mod.EXP2962_REL_PATH,
        {
            "honest_verdict": "complete: archive_ready=true",
            "archive_ready": True,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.EXP2963_REL_PATH,
        {
            "honest_verdict": "complete: DCCD structured-repair protocol ready",
            "dccd_repair_protocol_ready": True,
            "n_tasks_planned_min": 20,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2964_REL_PATH,
        {
            "honest_verdict": "complete: DCCD repair replication did not clear gates",
            "n_tasks": 20,
            "baseline_pass_at_1": 0.2,
            "taxonomy_repair_pass_at_1": 0.45,
            "dccd_repair_pass_at_1": 0.0,
            "pass_at_1_delta": -0.2,
            "baseline_pass_at_k": 0.3,
            "dccd_repair_pass_at_k": 0.0,
            "pass_at_k_delta": -0.3,
            "syntax_failure_rate_delta": 0.3,
            "schema_failure_rate_delta": 0.95,
            "false_accept_delta": -0.05,
            "dccd_repair_replication_clean": False,
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2965_REL_PATH,
        {
            "honest_verdict": "complete: bounded certificate audit ready",
            "beaver_style_certificate_ready": True,
            "available_repair_candidate_count": 32,
            "available_repair_candidate_audited_count": 32,
            "full_beaver_claim": False,
            "validation_fixture_passed": True,
            "validation_fixture_count": 5,
        },
    )
    _write_json(
        root,
        mod.EXP2966_REL_PATH,
        {
            "honest_verdict": "complete: exact skill-labeled logic frontier materialized",
            "logic_frontier_materialized": True,
            "n_items": 24,
            "reference_z3_execution_rate": 1.0,
            "reference_solver_accuracy": 1.0,
            "skill_label_counts": {"symbolization": 24, "validity": 10},
            "manifest_sha256": "frontier-sha",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2967_REL_PATH,
        {
            "honest_verdict": "complete: local SOTA DCCD formalizations did not clear gate",
            "n_items": 24,
            "baseline_parseability_rate": 0.083333,
            "baseline_solver_verified_accuracy": 0.0,
            "parseability_rate": 0.25,
            "solver_verified_accuracy": 0.208333,
            "answer_accuracy": 0.25,
            "z3_execution_rate": 0.208333,
            "formalization_delta_clean": False,
            "failure_categories": {"solver_verified_correct": 5, "unparseable": 18},
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2968_REL_PATH,
        {
            "honest_verdict": "complete: deterministic partial monitor harness ready",
            "partial_monitor_harness_ready": True,
            "fixture_trace_count": 5,
            "fixture_checks_passed": True,
            "full_streaming_verification_claim": False,
            "latency_estimate_ms": 4.67,
            "coverage_by_event": {"partial_code_block": 2},
        },
    )
    _write_json(
        root,
        mod.EXP2969_REL_PATH,
        {
            "honest_verdict": "complete: non_tautological_self_learning_ready",
            "continuous_self_learning_task": True,
            "non_tautological_self_learning_ready": True,
            "leakage_check_passed": True,
            "frozen_heldout_utility": 0.033333333333,
            "random_replay_heldout_utility": 0.142857142857,
            "prior_utility_gated_heldout_utility": 0.329686371841,
            "new_heldout_utility": 0.236013686912,
            "heldout_utility_delta_vs_random": 0.093156544055,
            "negative_control_delta": 0.0,
            "forgetting_guard_passed": True,
            "rollback_triggered": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2970_REL_PATH,
        {
            "honest_verdict": "complete: kan_forgetting_guard_ready",
            "kan_forgetting_guard_ready": True,
            "selected_policy": "per_knot_importance_update",
            "forgetting_threshold": 0.05,
            "forgetting_delta_by_policy": {"eager_update": 0.75, "per_knot_importance_update": 0.0},
            "current_domain_utility": {"per_knot_importance_update": 1.0},
            "old_domain_utility": {"per_knot_importance_update": 1.0},
            "high_dimensional_claim_allowed": False,
            "no_synthesis_claim": True,
            "no_analog_claim": True,
        },
    )
    _write_json(
        root,
        mod.EXP2971_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_flash_preconditions_ready",
            "gatemate_board_detected": True,
            "bitstream_sha256_verified": True,
            "gatemate_flash_preconditions_ready": True,
            "bitstream_sha256": "bitstream-sha",
            "flash_command": "openFPGALoader -c dirtyJtag -b olimex_gatemateevb bitstream.bit",
        },
    )
    _write_json(
        root,
        mod.EXP2972_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_flash_contact_smoke_no_readback",
            "board_detected": True,
            "bitstream_sha256_verified": True,
            "flash_attempted": True,
            "flash_succeeded": True,
            "smoke_vector_passed": False,
            "observed_output_sha256": "output-sha",
            "timing_observation": {"post_flash_contact_detected": True},
        },
    )
    _write_json(
        root,
        mod.MATRIX_V13_REL_PATH,
        {
            "honest_verdict": "complete: matrix_v13_ready=true; clean=26; flagged=14",
            "matrix_v13_ready": True,
            "forbidden_claims_absent": True,
            "clean_rows": [
                "exp2965_beaver_style_certificates",
                "exp2970_kan_forgetting_guard",
                "exp2971_gatemate_flash_preflight",
                "exp2972_gatemate_flash_contact_hash",
            ],
            "flagged_rows": [
                "exp2963_dccd_repair_protocol",
                "exp2964_dccd_repair_replication",
                "exp2966_logic_frontier_materializer",
                "exp2967_solver_frontier_formalization",
                "exp2969_non_tautological_fr11",
            ],
            "blocked_rows": ["exp2957_gatemate_flash_smoke"],
            "gated_skipped_rows": [],
            "pilot_only_rows": ["exp2968_partial_monitor_harness"],
            "aggregation_only_rows": ["exp2962_archive_activation"],
            "repair_replication_summary": {
                "protocol_ready": True,
                "n_tasks": 20,
                "baseline_pass_at_1": 0.2,
                "taxonomy_repair_pass_at_1": 0.45,
                "dccd_repair_pass_at_1": 0.0,
                "pass_at_1_delta": -0.2,
                "baseline_pass_at_k": 0.3,
                "dccd_repair_pass_at_k": 0.0,
                "pass_at_k_delta": -0.3,
                "syntax_failure_rate_delta": 0.3,
                "schema_failure_rate_delta": 0.95,
                "false_accept_delta": -0.05,
                "dccd_repair_replication_clean": False,
                "artifact_flagged": True,
            },
            "solver_frontier_summary": {
                "frontier_materialized": True,
                "formalization_delta_clean": False,
                "baseline_parseability_rate": 0.083333,
                "parseability_rate": 0.25,
                "parseability_delta_vs_278": 0.166667,
                "solver_verified_accuracy": 0.208333,
                "solver_verified_accuracy_delta_vs_278": 0.208333,
                "artifact_flagged": True,
            },
            "self_learning_summary": {
                "non_tautological_self_learning_ready": True,
                "leakage_check_passed": True,
                "heldout_utility_delta_vs_random": 0.093156544055,
                "negative_control_delta": 0.0,
                "forgetting_guard_passed": True,
                "artifact_flagged": True,
            },
            "kan_memory_summary": {
                "kan_forgetting_guard_ready": True,
                "selected_policy": "per_knot_importance_update",
                "high_dimensional_claim_allowed": False,
                "no_synthesis_claim": True,
                "no_analog_claim": True,
                "claim_boundary": "bounded_fixture_no_synthesis_or_analog_claim",
            },
            "hardware_state_summary": {
                "gatemate": {
                    "prior_278_flash_state": "blocked_board_not_detected",
                    "board_detected": True,
                    "flash_preconditions_ready": True,
                    "flash_attempted": True,
                    "flash_succeeded": True,
                    "smoke_vector_passed": False,
                    "observed_output_sha256": "output-sha",
                },
                "claim_boundary": "GateMate rows record contact and transcript hashes only.",
            },
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )


def test_req_report_2974_spec_anchor_exists() -> None:
    """REQ-REPORT-2974: OpenSpec declares the capstone contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2974" in spec
    assert "SCENARIO-REPORT-2974" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2974_builds_capstone_from_available_279_artifacts(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2974: .279 closeout does not promote flagged rows."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete: milestone_279_capstone")
    assert artifact["milestone"] == "2026.05.279"
    assert artifact["paper_ready"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["forbidden_claims_absent"] is True

    assert artifact["clean_artifacts"] == ["exp2965", "exp2970", "exp2971", "exp2972"]
    assert artifact["flagged_artifacts"] == [
        "exp2963",
        "exp2964",
        "exp2966",
        "exp2967",
        "exp2969",
        "exp2973",
    ]
    assert artifact["blocked_artifacts"] == []
    assert artifact["gated_skipped_artifacts"] == []
    assert artifact["missing_artifacts"] == []
    assert artifact["pilot_only_artifacts"] == ["exp2968"]
    assert artifact["aggregation_only_artifacts"] == ["exp2962", "exp2974"]
    assert artifact["artifact_classification_counts"] == {
        "aggregation-only": 2,
        "blocked": 0,
        "clean": 4,
        "flagged": 6,
        "gated-skipped": 0,
        "missing": 0,
        "pilot-only": 1,
    }

    details = {(row["task_id"], row["classification"]) for row in artifact["classification_details"]}
    assert details >= {
        ("exp2964", "flagged"),
        ("exp2968", "pilot-only"),
        ("exp2972", "clean"),
        ("exp2973", "flagged"),
        ("exp2974", "aggregation-only"),
    }

    summaries = artifact["outcome_summaries"]
    assert summaries["dccd_code_repair"]["n_tasks"] == 20
    assert summaries["dccd_code_repair"]["pass_at_1_delta"] == pytest.approx(-0.2)
    assert summaries["dccd_code_repair"]["dccd_repair_replication_clean"] is False
    assert summaries["beaver_style_certificates"]["certificate_ready"] is True
    assert summaries["beaver_style_certificates"]["probability_bound_claimed"] is False
    assert summaries["solver_frontier_formalization"]["parseability_delta_vs_278"] == pytest.approx(
        0.166667
    )
    assert summaries["solver_frontier_formalization"]["formalization_delta_clean"] is False
    assert summaries["partial_monitors"]["status"] == "pilot_only"
    assert summaries["fr11_non_tautology"]["non_tautological_self_learning_ready"] is True
    assert summaries["fr11_non_tautology"]["artifact_flagged"] is True
    assert summaries["kan_memory"]["selected_policy"] == "per_knot_importance_update"
    assert summaries["gatemate"]["flash_succeeded"] is True
    assert summaries["gatemate"]["smoke_vector_passed"] is False
    assert summaries["matrix_v13"]["matrix_v13_ready"] is True
    assert summaries["matrix_v13"]["artifact_flagged"] is True

    assert artifact["three_biggest_gap_assessment"] == {
        "code_repair_replication": "open_flagged_regression",
        "solver_frontier_formalization": "open_flagged_partial_improvement",
        "fr11_and_hardware": "partial_gate_mate_contact_closed_fr11_flagged",
    }
    assert any("GateMate board contact and flash" in gap for gap in artifact["gaps_closed"])
    assert any("DCCD repair replication remains open" in gap for gap in artifact["gaps_remaining"])
    assert any("FR-11 non-tautology remains flagged" in gap for gap in artifact["gaps_remaining"])
    assert 2 <= len(artifact["next_milestone_recommendations"]) <= 4
    assert "DCCD" in artifact["next_milestone_recommendations"][0]

    safe_text = json.dumps(artifact["paper_v6_safe_claims"], sort_keys=True).lower()
    assert "kv260" not in safe_text
    assert "boltzmann" not in safe_text
    assert "thermalization" not in safe_text
    assert "kona" not in safe_text
    assert "native ebt" not in safe_text
    assert any("KV260" in claim for claim in artifact["forbidden_claims_reaffirmed"])
    assert artifact["source_checksums"][mod.EXP2964_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2964_REL_PATH
    )


def test_req_report_2974_gated_skip_is_not_fabricated_missing(tmp_path: Path) -> None:
    """REQ-REPORT-2974: unmet gates classify absent branches as gated-skipped."""

    _write_roadmap(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP2969_REL_PATH,
        {
            "honest_verdict": "blocked_reset_controls_failed",
            "non_tautological_self_learning_ready": False,
        },
    )
    _write_json(
        tmp_path,
        mod.EXP2971_REL_PATH,
        {
            "honest_verdict": "complete: board absent",
            "gatemate_board_detected": False,
            "bitstream_sha256_verified": True,
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert "exp2970" in artifact["gated_skipped_artifacts"]
    assert "exp2972" in artifact["gated_skipped_artifacts"]
    assert "exp2973" in artifact["gated_skipped_artifacts"]
    assert "exp2970" not in artifact["missing_artifacts"]
    assert any(
        row["task_id"] == "exp2972"
        and row["gate_blocked_by"] == ["exp2971.gatemate_board_detected"]
        for row in artifact["classification_details"]
    )


def test_req_report_2974_blocks_paper_ready_when_matrix_forbidden_claims_fail(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2974: forbidden claim failures keep the capstone non-ready."""

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.MATRIX_V13_REL_PATH,
        {
            "honest_verdict": "complete: matrix_v13_ready=true",
            "matrix_v13_ready": True,
            "forbidden_claims_absent": False,
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["paper_ready"] is False
    assert artifact["forbidden_claims_absent"] is False
    assert "Forbidden claim boundary failed in matrix v13." in artifact["gaps_remaining"]


def test_req_report_2974_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-2974: write_artifact emits the required deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=4.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["milestone"] == "2026.05.279"
    assert saved["duration_s"] == pytest.approx(1.25)
    assert saved["source_checksums"][mod.MATRIX_V13_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V13_REL_PATH
    )


def test_req_report_2974_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-2974: helper edges preserve malformed and nonnumeric inputs."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._blocked_verdict("gate_blocked_precondition") is True
    assert mod._blocked_verdict("complete: ok") is False
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
    assert mod._delta(None, 1.0) is None
    assert mod._delta(2.0, 1.25) == pytest.approx(0.75)
    assert mod._get_path({"a": 1}, "a.b") is None
    assert mod._flag_kinds({"corrigendum_pending": "not-list"}) == []
    assert mod._unique_strings(["a", "b", "a"]) == ["a", "b"]

    task = mod.TaskSpec(
        task_id="exp2970",
        title="KAN",
        deliverable=mod.EXP2970_REL_PATH,
        inference_substrate="deterministic_wiring",
        gated_on=(),
    )
    assert mod._classify_task(task, {}, False, []) == "missing"
    assert mod._classify_task(task, {}, False, ["exp2969.ready"]) == "gated-skipped"
    assert mod._classify_task(task, {"honest_verdict": "blocked"}, True, []) == "blocked"
    assert mod._classify_task(task, {"flagged_adversarial": True}, True, []) == "flagged"
    assert mod._classify_task(task, {"pilot_only": True}, True, []) == "pilot-only"
    assert mod._classify_task(task, {}, True, []) == "missing"

    aggregation_task = mod.TaskSpec(
        task_id="exp2974",
        title="Capstone",
        deliverable=mod.OUTPUT_REL_PATH,
        inference_substrate=mod.INFERENCE_SUBSTRATE,
        gated_on=(),
    )
    assert mod._classify_task(aggregation_task, {}, False, []) == "aggregation-only"

    (tmp_path / mod.ROADMAP_REL_PATH).write_text(
        """
tasks:
  - ignored
  - id: note123
    deliverable: "results/ignored.json"
  - id: exp2968
    deliverable: "results/pilot.json"
""".lstrip(),
        encoding="utf-8",
    )
    assert [loaded.task_id for loaded in mod._load_tasks(tmp_path)] == ["exp2968"]
    assert (
        mod._headline_outcome(True, [], [])
        == "paper_ready: .279 cleared the .278 flagged code, FR-11, and solver rows"
    )


def test_req_report_2974_real_timer_zero_duration_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-2974: wall-clock measurement never reports exact zero."""

    _write_ready_sources(tmp_path)
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 42.0)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["duration_s"] == pytest.approx(0.000001)

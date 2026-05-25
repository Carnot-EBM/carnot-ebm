"""Tests for Exp 3025 milestone .283 terminal capstone.

Spec refs: REQ-REPORT-3025, SCENARIO-REPORT-3025.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v283_3025 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "n_tasks_evaluated",
    "repaired_rows",
    "flagged_rows",
    "blocked_rows",
    "gated_skipped_rows",
    "missing_rows",
    "cited_upstream_artifacts",
    "inference_substrate",
    "publication_action_allowed",
    "next_milestone_recommendation",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    row_id: str,
    source_experiment_id: str,
    status: str,
    *,
    claim_class: str = "claim",
    evidence_type: str = "test_evidence",
    inference_substrate: str = mod.INFERENCE_SUBSTRATE,
    source_honest_verdict: str | None = None,
    summary: dict[str, Any] | None = None,
    upstream_flags: list[str] | None = None,
    self_learning_boundary: dict[str, Any] | None = None,
    hardware_boundary: dict[str, Any] | None = None,
    claim_boundary_violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "source_experiment_id": source_experiment_id,
        "milestone": "2026.05.283",
        "status": status,
        "claim_class": claim_class,
        "evidence_type": evidence_type,
        "inference_substrate": inference_substrate,
        "headline_eligible": status == "clean",
        "paper_claim_eligible": status == "clean",
        "claim_boundary": "test claim boundary",
        "claim_boundary_guard_passed": not claim_boundary_violations,
        "claim_boundary_violations": claim_boundary_violations or [],
        "source_honest_verdict": source_honest_verdict or f"complete: {row_id} {status}",
        "upstream_flags": upstream_flags or [],
        "hardware_boundary": hardware_boundary or {"status": "not_applicable"},
        "self_learning_boundary": self_learning_boundary or {"status": "not_applicable"},
        "summary": summary or {},
    }


def _task_matrix_rows(*, all_clean: bool = False) -> list[dict[str, Any]]:
    repair_status = "clean" if all_clean else "flagged"
    gatemate_status = "clean" if all_clean else "gated-skipped"
    ssqa_status = "clean" if all_clean else "gated-skipped"
    return [
        _row("exp3012_archive_activation", "exp3012", "projection-only", claim_class="archive_activation"),
        _row("exp3013_sota_logprob_telemetry", "exp3013", "clean" if all_clean else "flagged"),
        _row("exp3014_repair_failure_taxonomy", "exp3014", "clean" if all_clean else "flagged"),
        _row("exp3015_acceptance_controller", "exp3015", "clean" if all_clean else "flagged"),
        _row(
            "exp3016_repair_acceptance_controller",
            "exp3016",
            repair_status,
            claim_class="repair_eval",
            inference_substrate="live_sota_gguf_repair_with_acceptance_controller",
            summary={
                "repair_controller_clean": True,
                "headline_result": True,
                "pass_at_1_delta": 0.375,
                "pass_at_k_delta": 0.375,
                "false_accept_delta": 0.0,
                "syntax_failure_rate_delta": 0.0,
                "schema_failure_rate_delta": 0.0,
                "tautology_gate_clean": True,
            },
            upstream_flags=[] if all_clean else ["flagged_adversarial=true", "TAUTOLOGY:critical"],
        ),
        _row("exp3017_instruction_validator_tree", "exp3017", "clean"),
        _row("exp3018_beaver_frontier_certificate", "exp3018", "clean" if all_clean else "flagged"),
        _row("exp3019_fr11_feasibility_channel", "exp3019", "clean" if all_clean else "flagged"),
        _row(
            "exp3020_fr11_verifier_feedback_controller",
            "exp3020",
            "clean",
            claim_class="fr11_self_learning_controller",
            evidence_type="cached_exact_trace_replay_controller_only",
            inference_substrate="cached_exact_trace_replay_controller_only",
            summary={
                "verifier_feedback_controller_ready": True,
                "continuous_self_learning_task": True,
                "independent_self_learning_boundary_preserved": True,
                "n_replay_items": 68,
                "heldout_delta": 0.5,
                "negative_control_delta": 0.0,
                "forgetting_guard_passed": True,
                "drift_guard_passed": True,
                "tautology_risk_flag": False,
            },
            self_learning_boundary={
                "status": "clean",
                "boundary": "verifier-feedback controller over exact traces only",
                "continuous_self_learning_task": True,
            },
        ),
        _row("exp3021_gatemate_transport_shim", "exp3021", "clean" if all_clean else "blocked"),
        _row(
            "exp3022_gatemate_transport_flash_smoke",
            "exp3022",
            gatemate_status,
            claim_class="gatemate_host_visible_io",
            summary={
                "structured_gate_failed": not all_clean,
                "host_visible_io_ready": all_clean,
                "smoke_vector_passed": all_clean,
                "observed_output_hash_present": all_clean,
            },
        ),
        _row(
            "exp3023_ssqa_explicit_gate_artifact",
            "exp3023",
            ssqa_status,
            claim_class="ssqa_gate_artifact",
            summary={
                "ssqa_artifact_written": True,
                "ssqa_gate_status": "" if all_clean else "gate_skipped",
                "ssqa_rtl_pnr_report_ready": all_clean,
                "upstream_host_visible_io_ready": all_clean,
                "projection_only": not all_clean,
            },
            hardware_boundary={"status": "clean" if all_clean else "projection_only"},
        ),
    ]


def _matrix_v17(*, all_clean: bool = False, boundary_violation: bool = False) -> dict[str, Any]:
    rows = _task_matrix_rows(all_clean=all_clean)
    if not all_clean:
        rows.extend(
            [
                _row("carry_forward_v16:prior_flagged", "exp3010", "flagged"),
                _row("carry_forward_v16:prior_blocked", "exp3010", "blocked"),
                _row("carry_forward_v16:prior_missing", "exp3010", "missing"),
                _row("carry_forward_v16:prior_pilot", "exp3010", "pilot-only"),
            ]
        )
    violations = (
        [
            {
                "row_id": "exp3023_ssqa_explicit_gate_artifact",
                "violation": "unsupported_hardware_claim",
                "fields": ["speedup_claim_made"],
            }
        ]
        if boundary_violation
        else []
    )
    claim_rows = {
        "exp3016_repair": next(row for row in rows if row["row_id"] == "exp3016_repair_acceptance_controller"),
        "exp3020_fr11_self_learning": next(
            row for row in rows if row["row_id"] == "exp3020_fr11_verifier_feedback_controller"
        ),
        "exp3021_gatemate_transport": next(row for row in rows if row["row_id"] == "exp3021_gatemate_transport_shim"),
        "exp3022_gatemate_io": next(
            row for row in rows if row["row_id"] == "exp3022_gatemate_transport_flash_smoke"
        ),
        "exp3023_ssqa": next(row for row in rows if row["row_id"] == "exp3023_ssqa_explicit_gate_artifact"),
    }
    return {
        "schema": "carnot.cross_corpus_matrix.v17_283_claim_boundary.v1",
        "artifact": "experiment_3024_cross_corpus_matrix_v17",
        "run_date": "20260525",
        "milestone": "2026.05.283",
        "matrix_v17_ready": True,
        "honest_verdict": "complete: matrix_v17_ready=true",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "rows": rows,
        "row_count": len(rows),
        "clean_count": sum(1 for row in rows if row["status"] == "clean"),
        "flagged_count": sum(1 for row in rows if row["status"] == "flagged"),
        "blocked_count": sum(1 for row in rows if row["status"] == "blocked"),
        "gated_skipped_count": sum(1 for row in rows if row["status"] == "gated-skipped"),
        "pilot_only_count": sum(1 for row in rows if row["status"] == "pilot-only"),
        "projection_only_count": sum(1 for row in rows if row["status"] == "projection-only"),
        "missing_count": sum(1 for row in rows if row["status"] == "missing"),
        "claim_rows": claim_rows,
        "repaired_claims": [
            "exp3004_aquaforte_beaver_substrate_provenance",
            "exp3020_fr11_verifier_feedback_controller",
            "exp3023_ssqa_artifact_presence_repaired_gate_skipped_not_promotable",
        ],
        "still_blocked_claims": []
        if all_clean
        else [
            "exp3016_repair_acceptance_controller_flagged",
            "exp3022_gatemate_transport_flash_smoke_gated_skipped",
            "exp3023_ssqa_gate_skipped_until_host_visible_io_ready",
        ],
        "claim_boundary_violations": violations,
        "cited_upstream_artifacts": [
            {
                "experiment_id": "exp3016",
                "path": mod.EXP3016_REL_PATH.as_posix(),
                "present": True,
                "readable_json_object": True,
                "honest_verdict": "complete: acceptance-controlled SOTA repair rerun gates passed",
                "inference_substrate": "live_sota_gguf_repair_with_acceptance_controller",
                "model_provenance": {
                    "model_specs": {
                        "headline_models": [
                            "unsloth/Qwen3.6-35B-A3B-GGUF",
                            "unsloth/gemma-4-31B-it-GGUF",
                            "unsloth/gemma-4-26B-A4B-it-GGUF",
                        ]
                    },
                    "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                },
            }
        ],
        "recommended_next_actions": [
            "Exp3016: do not promote repair until TAUTOLOGY/METHODOLOGY flags are cleared.",
            "Exp3020: carry FR-11 forward only as verifier-feedback controller utility.",
            "Exp3021/3022: obtain host-visible GateMate output before SSQA.",
        ],
        "paper_v6_boundary_summary": {"forbidden_claims_absent": not boundary_violation},
        "hardware_boundary_summary": {"forbidden_claims_absent": not boundary_violation},
        "roadmap_acceptance_summary": {
            "aggregation_metadata_clean": True,
            "exp3016_repair_promotable": all_clean,
            "exp3020_fr11_promotable": True,
            "exp3022_gatemate_io_promotable": all_clean,
            "exp3023_ssqa_promotable": all_clean,
        },
        "missing_artifacts": [],
        "missing_documents": ["research-roadmap-next.yaml"],
    }


def _write_ready_sources(root: Path, *, all_clean: bool = False, boundary_violation: bool = False) -> None:
    _write_json(root, mod.MATRIX_V17_REL_PATH, _matrix_v17(all_clean=all_clean, boundary_violation=boundary_violation))
    _write_json(
        root,
        mod.CAPSTONE_V282_REL_PATH,
        {
            "artifact": "experiment_3011_capstone_v282",
            "capstone_ready": True,
            "paper_ready": False,
            "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
        },
    )
    for spec in mod.SOURCE_SPECS:
        if spec.experiment_id in {"exp3024", "exp3011"}:
            continue
        _write_json(root, spec.path, {"honest_verdict": f"complete: {spec.experiment_id}"})
    _write_json(
        root,
        mod.EXP3016_REL_PATH,
        {
            "honest_verdict": "complete: acceptance-controlled SOTA repair rerun gates passed",
            "inference_substrate": "live_sota_gguf_repair_with_acceptance_controller",
            "model_specs": {
                "headline_models": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                    "unsloth/gemma-4-26B-A4B-it-GGUF",
                ]
            },
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        },
    )


def test_req_report_3025_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3025: OpenSpec declares the capstone contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-3025" in spec
    assert "SCENARIO-REPORT-3025" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3025_builds_terminal_go_no_go(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3025: .283 capstone reports promotion decisions honestly."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_action_allowed"] is False
    assert artifact["n_tasks_evaluated"] == 13
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete: capstone_ready=true; paper_ready=false")
    assert artifact["next_milestone_recommendation"] == mod.NEXT_MILESTONE_RECOMMENDATION

    assert artifact["task_classification_counts"] == {
        "clean": 3,
        "flagged": 6,
        "blocked": 1,
        "gated-skipped": 2,
        "pilot-only": 0,
        "projection-only": 1,
        "missing": 0,
    }
    assert artifact["clean_task_rows"] == [
        "exp3017_instruction_validator_tree",
        "exp3020_fr11_verifier_feedback_controller",
        "exp3024_cross_corpus_matrix_v17",
    ]
    assert artifact["flagged_task_rows"] == [
        "exp3013_sota_logprob_telemetry",
        "exp3014_repair_failure_taxonomy",
        "exp3015_acceptance_controller",
        "exp3016_repair_acceptance_controller",
        "exp3018_beaver_frontier_certificate",
        "exp3019_fr11_feasibility_channel",
    ]
    assert artifact["blocked_task_rows"] == ["exp3021_gatemate_transport_shim"]
    assert artifact["gated_skipped_task_rows"] == [
        "exp3022_gatemate_transport_flash_smoke",
        "exp3023_ssqa_explicit_gate_artifact",
    ]

    assert "exp3016_repair_acceptance_controller" in artifact["flagged_rows"]
    assert "exp3021_gatemate_transport_shim" in artifact["blocked_rows"]
    assert "exp3022_gatemate_transport_flash_smoke" in artifact["gated_skipped_rows"]
    assert artifact["missing_rows"] == ["carry_forward_v16:prior_missing"]

    decisions = artifact["claim_promotion_decisions"]
    assert decisions["repair"]["promotable"] is False
    assert decisions["fr11_self_learning"]["promotable"] is True
    assert "verifier-feedback controller" in decisions["fr11_self_learning"]["claim_boundary"]
    assert decisions["gatemate_io"]["promotable"] is False
    assert decisions["ssqa"]["promotable"] is False
    assert decisions["ssqa"]["repaired_282_blocker"] is True
    assert decisions["aggregation_metadata"]["promotable"] is True

    assert artifact["repaired_282_blockers"] == [
        "exp3007_fr11_stability_repaired_by_exp3020_bounded_controller",
        "exp3009_ssqa_missing_artifact_repaired_by_exp3023_artifact_presence_only",
        "exp3011_aggregation_false_positive_risk_repaired_by_exp3024_nested_provenance",
    ]
    assert artifact["unrepaired_282_blockers"] == [
        "exp3003_repair_methodology_still_flagged_by_exp3016",
        "exp3008_hardware_io_still_gated_skipped_by_exp3022",
        "exp3009_ssqa_promotion_still_gated_skipped_by_exp3023",
    ]
    assert artifact["publication_gate_checks"] == {
        "durable_verifier_evidence_for_every_claimed_result": False,
        "no_false_sota_substitution": True,
        "no_live_substrate_ambiguity": True,
        "no_aggregation_live_inference_false_positive": True,
        "no_hardware_claim_boundary_breach": True,
        "every_promotion_gate_clean": False,
    }
    assert "repair row exp3016_repair_acceptance_controller is flagged" in artifact["paper_ready_blockers"]
    assert "matrix contains non-clean rows: flagged=7, blocked=2, gated_skipped=2, missing=1" in artifact[
        "paper_ready_blockers"
    ]

    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    serialized_citations = json.dumps(artifact["cited_upstream_artifacts"], sort_keys=True)
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in serialized_citations
    artifact_without_citations = dict(artifact)
    artifact_without_citations.pop("cited_upstream_artifacts")
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" not in json.dumps(
        artifact_without_citations,
        sort_keys=True,
    )
    assert artifact["source_checksums"][mod.MATRIX_V17_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V17_REL_PATH
    )
    assert artifact["ops_docs_reconciliation_left_to_conductor"] is True
    assert artifact["status_updates_written"] is False


def test_req_report_3025_blocks_when_required_matrix_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3025: missing matrix v17 fails closed."""

    _write_json(
        tmp_path,
        mod.CAPSTONE_V282_REL_PATH,
        {"honest_verdict": "complete: capstone_ready=true", "capstone_ready": True},
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["capstone_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["publication_action_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp3024",
            "path": mod.MATRIX_V17_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_3025_paper_ready_requires_clean_matrix_and_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-3025: clean synthetic evidence can be ready, publication still cannot."""

    _write_ready_sources(tmp_path, all_clean=True)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.125)

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_action_allowed"] is False
    assert artifact["paper_ready_blockers"] == []
    assert artifact["honest_verdict"].startswith("complete: capstone_ready=true; paper_ready=true")

    _write_ready_sources(tmp_path, all_clean=True, boundary_violation=True)
    blocked = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert blocked["paper_ready"] is False
    assert "matrix_v17 claim_boundary_violations is non-empty" in blocked["paper_ready_blockers"]
    assert blocked["publication_gate_checks"]["no_hardware_claim_boundary_breach"] is False


def test_req_report_3025_write_artifact_and_main_persist_json(tmp_path: Path) -> None:
    """REQ-REPORT-3025: write_artifact emits the deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.5)
    assert mod.main(tmp_path) == 0


def test_req_report_3025_helper_edges_keep_closeout_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3025: helpers keep malformed inputs and unknown statuses honest."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._normalized_status("unknown") == "missing"
    assert mod._status_token("gated-skipped") == "gated_skipped"
    assert mod._status_rows([{"row_id": "x", "status": "clean"}, {"status": "clean"}], "clean") == [
        "x"
    ]
    assert mod._row_by_id([{"row_id": "x", "status": "clean"}, []]) == {
        "x": {"row_id": "x", "status": "clean"}
    }
    assert mod._task_row_from_matrix("exp3024", {}, {})["status"] == "missing"
    assert mod._task_row_from_matrix("unknown", {}, {})["status"] == "missing"
    assert mod._matrix_wide_status_rows({}, "clean") == []
    assert mod._claim_row({}, "exp3016_repair", "exp3016_repair_acceptance_controller") == {}
    assert mod._paper_ready_blockers(
        {},
        {},
        [],
        {
            "durable_verifier_evidence_for_every_claimed_result": False,
            "no_false_sota_substitution": False,
            "no_live_substrate_ambiguity": False,
            "no_aggregation_live_inference_false_positive": False,
            "no_hardware_claim_boundary_breach": False,
            "every_promotion_gate_clean": False,
        },
        [],
    ) == [
        "matrix_v17_ready is not true",
        "matrix_v17 claim_boundary_violations is non-empty",
        "repair row exp3016_repair_acceptance_controller is missing",
        "FR-11 row exp3020_fr11_verifier_feedback_controller is missing",
        "GateMate IO row exp3022_gatemate_transport_flash_smoke is missing",
        "SSQA row exp3023_ssqa_explicit_gate_artifact is missing",
        "durable verifier evidence is not clean for every claimed result",
        "false SOTA substitution risk is not cleared",
        "live/substrate ambiguity is not cleared",
        "aggregation-live-inference false-positive risk is not cleared",
        "hardware claim boundary breach risk is not cleared",
        "not every promotion gate is clean",
    ]

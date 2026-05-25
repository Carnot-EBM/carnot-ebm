"""Tests for Exp 3038 cross-corpus matrix v18.

Spec refs: REQ-REPORT-3038, SCENARIO-REPORT-3038.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v18_3038 as mod


REQUIRED_FIELDS = {
    "matrix_v18_ready",
    "rows_total",
    "clean",
    "flagged",
    "blocked",
    "gated_skipped",
    "projection_only",
    "pilot_only",
    "missing",
    "retired",
    "matrix_rows",
    "cited_upstream_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ready_sources(root: Path) -> None:
    _write_json(
        root,
        mod.MATRIX_V17_REL_PATH,
        {
            "artifact": "experiment_3024_cross_corpus_matrix_v17",
            "honest_verdict": (
                "complete: matrix_v17_ready=true; clean=40; flagged=29; blocked=10; "
                "gated_skipped=3; projection_only=10; pilot_only=4; missing=1"
            ),
            "matrix_v17_ready": True,
            "clean_count": 40,
            "flagged_count": 29,
            "blocked_count": 10,
            "gated_skipped_count": 3,
            "projection_only_count": 10,
            "pilot_only_count": 4,
            "missing_count": 1,
            "rows": [
                {"row_id": "prior_clean", "status": "clean"},
                {"row_id": "prior_flagged", "status": "flagged"},
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3026_REL_PATH,
        {
            "milestone_archived": True,
            "next_milestone": "2026.05.284",
            "capstone_ready": True,
            "previous_paper_ready": False,
            "protected_files_unchanged": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "honest_verdict": "complete: milestone_archived=true; next_milestone=2026.05.284",
        },
    )
    _write_json(
        root,
        mod.EXP3027_REL_PATH,
        {
            "methodology_corrigendum_ready": True,
            "repair_rerun_required": True,
            "flagged_rows_reviewed": 29,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}],
            "honest_verdict": "complete: methodology_corrigendum_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3028_REL_PATH,
        {
            "clean_repair_rerun_ready": True,
            "clean_repair_claim_promotable_candidate": True,
            "repair_controller_clean": True,
            "n_tasks": 24,
            "n_live_transcripts": 24,
            "pass_at_1_delta": 0.375,
            "pass_at_k_delta": 0.375,
            "false_accept_delta": 0.0,
            "tautology_gate_clean": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
            "model_specs": {"headline_models": ["fixture/large-gguf"]},
            "inference_substrate": {
                "kind": "clean_repair_reconstruction",
                "gguf_cache_paths": {"fixture/large-gguf": "/models/fixture.gguf"},
            },
            "honest_verdict": "complete: clean_repair_rerun_ready=true; n_tasks=24",
        },
    )
    _write_json(
        root,
        mod.EXP3029_REL_PATH,
        {
            "repair_promotion_boundary_ready": True,
            "repair_claim_status": "bounded",
            "promotable_claims": [],
            "bounded_claims": [{"claim_id": "exp3028_clean_repair_candidate"}],
            "retired_or_blocked_claims": [
                {"claim_id": "headline_sota_repair_clean_methodology", "classification": "retired"},
                {"claim_id": "unsupported_exp3016_headline_repair_promotion", "classification": "retired"},
            ],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "honest_verdict": "complete: repair_claim_status=bounded",
        },
    )
    _write_json(
        root,
        mod.EXP3030_REL_PATH,
        {
            "validator_frontier_corrigendum_ready": True,
            "verified_region_count": 40,
            "unresolved_region_count": 2,
            "fallback_only_count": 1,
            "missing_authority_count": 0,
            "honest_verdict": "complete: validator_frontier_corrigendum_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3031_REL_PATH,
        {
            "dccd_panel_ready": True,
            "n_cases": 3,
            "false_accept_delta": 0.0,
            "intent_drift_delta": 0.0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "model_specs": {"selected_model": "fixture/large-gguf"},
            "honest_verdict": "complete: dccd structured repair panel ready; n_cases=3",
        },
    )
    _write_json(
        root,
        mod.EXP3032_REL_PATH,
        {
            "fr11_heldout_replay_ready": True,
            "continuous_self_learning_tested": True,
            "heldout_trace_count": 8,
            "feasible_infeasible_auc_delta": 0.5,
            "shuffled_feedback_delta": -0.5,
            "tautology_risk_cleared": True,
            "information_asymmetry_enforced": True,
            "invariant_violations": [],
            "honest_verdict": "complete_fr11_heldout_replay_ready",
        },
    )
    _write_json(
        root,
        mod.EXP3033_REL_PATH,
        {
            "fr11_nonforgetting_stress_ready": True,
            "fr11_self_learning_promotable": True,
            "promotion_decision": "controller_only_promotable",
            "prior_retention_delta": 0.0,
            "heldout_delta_after_update": 0.875,
            "shuffled_control_delta": -2.125,
            "drift_failures": [],
            "honest_verdict": "complete_controller_only_promotable",
        },
    )
    _write_json(
        root,
        mod.EXP3034_REL_PATH,
        {
            "gatemate_output_contract_ready": False,
            "host_visible_io_plan_ready": False,
            "selected_output_path": "explicit_no_ready_contract",
            "exact_operator_action_required": ["provide host-visible pinout"],
            "speedup_claim_made": False,
            "sampler_claim_made": False,
            "thermodynamic_claim_made": False,
            "honest_verdict": "complete: blocked_gatemate_output_contract_pinout_missing",
        },
    )
    _write_json(
        root,
        mod.EXP3035_GATE_CHECK_REL_PATH,
        {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp3034-gatemate-output-contract-pinout-decision",
                    "artifact_field": "gatemate_output_contract_ready",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                }
            ],
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3037_REL_PATH,
        {
            "ssqa_boundary_ready": True,
            "ssqa_gate_status": "gate_skipped",
            "upstream_gatemate_status": {"gatemate_flash_smoke_ready": False},
            "inference_substrate": {"host_visible_output_observed": False},
            "speedup_claim_made": False,
            "sampler_claim_made": False,
            "thermodynamic_claim_made": False,
            "honest_verdict": "complete: ssqa_gate_skipped_exp3036_missing",
        },
    )


def _rows_by_exp(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): row for row in artifact["matrix_rows"]}


def _loaded_for(experiment_id: str, *, present: bool = False) -> dict[str, Any]:
    spec = next(spec for spec in mod.SOURCE_SPECS if spec.experiment_id == experiment_id)
    return {
        "spec": spec,
        "payload": {},
        "actual_path": spec.planned_path,
        "planned_path_present": present,
        "actual_path_present": present,
    }


def test_req_report_3038_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3038: OpenSpec declares the matrix v18 contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-3038" in spec
    assert "SCENARIO-REPORT-3038" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3038_builds_complete_task_matrix(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3038: v18 represents all .284 tasks and gate states."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)
    rows = _rows_by_exp(artifact)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v18_ready"] is True
    assert artifact["honest_verdict"].startswith("complete: matrix_v18_ready=true")
    assert artifact["rows_total"] == 14
    assert len(artifact["matrix_rows"]) == 14
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["clean"] == 4
    assert artifact["flagged"] == 4
    assert artifact["blocked"] == 1
    assert artifact["gated_skipped"] == 3
    assert artifact["projection_only"] == 1
    assert artifact["pilot_only"] == 0
    assert artifact["missing"] == 1
    assert artifact["retired"] == 0

    assert rows["exp3026"]["status"] == "projection_only"
    assert rows["exp3028"]["status"] == "flagged"
    assert rows["exp3029"]["repair_claim_status"] == "bounded"
    assert rows["exp3033"]["fr11_self_learning_promotable"] is True
    assert rows["exp3034"]["status"] == "blocked"
    assert rows["exp3034"]["gatemate_output_contract_ready"] is False
    assert rows["exp3035"]["status"] == "gated_skipped"
    assert rows["exp3035"]["planned_path_present"] is False
    assert rows["exp3035"]["actual_path"] == mod.EXP3035_GATE_CHECK_REL_PATH.as_posix()
    assert rows["exp3036"]["status"] == "gated_skipped"
    assert rows["exp3036"]["host_visible_output_observed"] is False
    assert rows["exp3037"]["status"] == "gated_skipped"
    assert rows["exp3037"]["ssqa_gate_status"] == "gate_skipped"
    assert rows["exp3038"]["status"] == "clean"
    assert rows["exp3039"]["status"] == "missing"

    assert artifact["baseline_v17_summary"] == {
        "matrix_v17_ready": True,
        "clean": 40,
        "flagged": 29,
        "blocked": 10,
        "gated_skipped": 3,
        "projection_only": 10,
        "pilot_only": 4,
        "missing": 1,
    }
    assert mod.EXP3036_REL_PATH.as_posix() in artifact["missing_artifacts"]
    assert mod.EXP3039_REL_PATH.as_posix() in artifact["missing_artifacts"]
    assert artifact["source_checksums"][mod.EXP3033_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3033_REL_PATH
    )


def test_req_report_3038_keeps_live_model_metadata_inside_citations(tmp_path: Path) -> None:
    """REQ-REPORT-3038: aggregation output has no top-level live-model metadata."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    forbidden_top_level = {
        "model_specs",
        "target_model",
        "cuda",
        "CUDA",
        "gguf",
        "GGUF",
        "gpu_inventory",
        "headline_models_used",
        "live_model_metadata",
    }
    assert forbidden_top_level.isdisjoint(artifact.keys())
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "source": "checked_in_artifacts",
    }

    serialized_citations = json.dumps(artifact["cited_upstream_artifacts"], sort_keys=True)
    assert "fixture/large-gguf" in serialized_citations
    artifact_without_citations = dict(artifact)
    artifact_without_citations.pop("cited_upstream_artifacts")
    assert "fixture/large-gguf" not in json.dumps(artifact_without_citations, sort_keys=True)


def test_req_report_3038_blocks_when_matrix_v17_missing(tmp_path: Path) -> None:
    """REQ-REPORT-3038: the v17 baseline is required for a ready matrix."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.MATRIX_V17_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["matrix_v18_ready"] is False
    assert artifact["honest_verdict"] == "blocked_required_v17_baseline_missing"
    assert artifact["required_upstream_errors"] == [
        {
            "experiment_id": "exp3024",
            "path": mod.MATRIX_V17_REL_PATH.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
    ]


def test_req_report_3038_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3038: write_artifact emits the deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v18_ready"] is True
    assert saved["rows_total"] == len(saved["matrix_rows"])
    assert saved["duration_s"] == pytest.approx(0.125)


def test_req_report_3038_helper_edges_keep_statuses_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3038: helpers classify absent, malformed, and gated inputs."""

    malformed = tmp_path / "bad.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._status_from_verdict("blocked_gate_check_failed") == "gated_skipped"
    assert mod._status_from_verdict("complete: blocked_pinout_missing") == "blocked"
    assert mod._status_from_verdict("complete_flagged_methodology") == "flagged"
    assert mod._status_from_verdict("complete: ok") == "clean"
    assert mod._has_flags({"flagged_adversarial": True}) is True
    assert mod._has_flags({"corrigendum_pending": [{"kind": "X"}]}) is True
    assert mod._has_flags({}) is False
    assert mod._safe_bool(None) is None
    assert mod._safe_bool(True) is True
    assert mod._safe_bool(False) is False
    assert mod._safe_bool("yes") is None
    assert mod._source_model_details({}) == {}
    assert mod._source_model_details({"model_specs": {"x": 1}}) == {"model_specs": {"x": 1}}


def test_req_report_3038_edge_branches_remain_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3038: unusual task states stay machine-readable."""

    _write_ready_sources(tmp_path)
    monkeypatch.setattr(mod, "_matrix_rows", lambda _payloads, _loaded: [])
    coverage_blocked = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    assert coverage_blocked["matrix_v18_ready"] is False
    assert coverage_blocked["honest_verdict"] == "blocked_matrix_v18_task_coverage_incomplete"
    assert coverage_blocked["coverage_errors"] == [
        {
            "reason": "missing_task_rows",
            "experiment_ids": [f"exp{number}" for number in range(3026, 3040)],
        }
    ]
    monkeypatch.undo()

    assert mod._missing_artifacts(
        [
            {
                "planned_path_present": True,
                "present": False,
                "planned_path": "planned.json",
                "actual_path": "actual.json",
            }
        ]
    ) == ["actual.json"]
    assert mod._coverage_errors({"exp3026"}) == [
        {
            "reason": "missing_task_rows",
            "experiment_ids": [f"exp{number}" for number in range(3027, 3040)],
        }
    ]

    assert mod._exp3026_row({}, _loaded_for("exp3026"))["status"] == "missing"
    assert mod._exp3027_row({}, _loaded_for("exp3027"))["status"] == "missing"
    assert mod._exp3028_row({}, _loaded_for("exp3028"))["status"] == "missing"
    assert mod._exp3029_row({}, _loaded_for("exp3029"))["status"] == "missing"
    assert mod._exp3030_row({}, _loaded_for("exp3030"))["status"] == "missing"
    assert mod._exp3031_row({}, _loaded_for("exp3031"))["status"] == "missing"
    assert mod._exp3032_row({}, _loaded_for("exp3032"))["status"] == "missing"
    assert mod._exp3033_row({}, _loaded_for("exp3033"))["status"] == "missing"
    assert mod._exp3034_row({}, _loaded_for("exp3034"))["status"] == "missing"
    assert mod._exp3037_row({}, _loaded_for("exp3037"))["status"] == "missing"

    exp3034_ready = mod._exp3034_row(
        {"gatemate_output_contract_ready": True, "honest_verdict": "complete: ok"},
        _loaded_for("exp3034", present=True),
    )
    assert exp3034_ready["status"] == "clean"
    exp3034_flagged = mod._exp3034_row(
        {"gatemate_output_contract_ready": True, "speedup_claim_made": True},
        _loaded_for("exp3034", present=True),
    )
    assert exp3034_flagged["status"] == "flagged"

    assert (
        mod._exp3035_row({}, _loaded_for("exp3035"), exp3034_ready)["status"] == "missing"
    )
    assert (
        mod._exp3035_row(
            {"gatemate_output_shim_ready": True, "honest_verdict": "complete: shim"},
            _loaded_for("exp3035", present=True),
            exp3034_ready,
        )["status"]
        == "clean"
    )
    exp3036_blocked = mod._exp3036_row(
        {"gatemate_flash_smoke_ready": False, "honest_verdict": "complete: no output"},
        _loaded_for("exp3036", present=True),
        exp3034_ready,
    )
    assert exp3036_blocked["status"] == "blocked"

    assert (
        mod._exp3037_row(
            {"ssqa_boundary_ready": True, "speedup_claim_made": True},
            _loaded_for("exp3037", present=True),
        )["status"]
        == "flagged"
    )
    assert (
        mod._exp3037_row(
            {"ssqa_boundary_ready": True, "honest_verdict": "complete: ok"},
            _loaded_for("exp3037", present=True),
        )["status"]
        == "clean"
    )
    assert (
        mod._exp3037_row(
            {"ssqa_boundary_ready": False, "honest_verdict": "complete: no gate"},
            _loaded_for("exp3037", present=True),
        )["status"]
        == "blocked"
    )
    assert (
        mod._exp3039_row(
            {"capstone_ready": True, "paper_ready": False, "honest_verdict": "complete: capstone"},
            _loaded_for("exp3039", present=True),
        )["status"]
        == "clean"
    )

    assert mod._guarded_status({"speedup_claim_made": True}, "clean") == "flagged"
    assert mod._guarded_status({"honest_verdict": "complete: blocked_reason"}, "clean") == (
        "blocked"
    )
    assert mod._status_from_verdict("complete: retired unsupported claim") == "retired"
    assert mod._int_or(True, 7) == 7
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("not-int") is None
    assert mod._float_or(True, 1.25) == pytest.approx(1.25)
    assert mod._float_or("not-float", 2.5) == pytest.approx(2.5)
    assert mod._float_or_none(True) is None
    assert mod._float_or_none("not-float") is None

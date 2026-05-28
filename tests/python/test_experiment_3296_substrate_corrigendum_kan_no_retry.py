"""Tests for Exp 3296 substrate corrigendum and KAN no-retry ledger.

Spec refs: REQ-REPORT-3296, SCENARIO-REPORT-3296.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import substrate_corrigendum_kan_no_retry_3296 as mod


REQUIRED_FIELDS = {
    "substrate_corrigendum_ready",
    "kan_no_retry_ledger_ready",
    "kan_prompt_injection_headline_retired",
    "prior_kan_auroc",
    "prior_aligned_instruction_false_positive_rate",
    "headline_eligible_prior_metrics",
    "non_headline_prior_metrics",
    "downstream_usage_rules",
    "future_reopen_prerequisites",
    "protected_files_untouched",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3283_REL_PATH,
        {
            "experiment_id": "exp3283",
            "corrigendum_ready": True,
            "headline_eligible_metrics": [
                {
                    "source_experiment_id": "exp3272",
                    "metric": "artifact_checksums_available",
                    "value": True,
                    "boundary": "integrity claim only",
                },
                {
                    "source_experiment_id": "exp3272",
                    "metric": "split_leakage_boundary",
                    "value": {"leakage_audit_passed": True},
                    "boundary": "split leakage boundary only",
                },
            ],
            "provisional_or_sidecar_metrics": [
                {
                    "source_experiment_id": "exp3272",
                    "metric": "assembled_example_count",
                    "value": 15000,
                    "boundary": "operational inventory only because count flags remain",
                },
                {
                    "source_experiment_id": "exp3273",
                    "metric": "full_corpus_auroc",
                    "value": 0.475326,
                    "boundary": "KAN sidecar only; DeLong non-inferiority failed",
                },
            ],
            "downstream_usage_rules": {
                "garak": {"allowed": True, "rule": "rerun real Garak"},
                "repair": {"allowed": False, "headline_allowed": False},
                "kan": {"allowed": True, "headline_allowed": False},
                "paper_claims": {"headline_performance_metrics_allowed": False},
            },
            "honest_verdict": "complete: corrigendum ready",
        },
    )
    _write_json(
        root,
        mod.EXP3288_REL_PATH,
        {
            "experiment_id": "exp3288",
            "kan_boundary_decision_ready": True,
            "kan_boundary_decision": "retire_from_prompt_injection_headline",
            "prior_full_corpus_auroc": 0.475326,
            "prior_full_corpus_auprc": 0.626269,
            "prior_delong_noninferiority_passed": False,
            "aligned_instruction_false_positive_summary": {
                "aligned_instruction_false_positive_rate": 1.0,
                "aligned_instruction_case_count": 439,
            },
            "permitted_downstream_use": [
                "offline_failure_autopsy",
                "negative_control_regression_fixture",
            ],
            "prohibited_downstream_use": [
                "prompt_injection_headline_detector",
                "repair_gate_authority",
            ],
            "future_work_prerequisite": "reattempt only after clean labels and FP ceiling",
            "honest_verdict": "complete: KAN retired",
        },
    )
    _write_json(
        root,
        mod.EXP3292_REL_PATH,
        {
            "experiment_id": "exp3292",
            "matrix_v36_ready": True,
            "paper_ready": False,
            "primary_status_counts": {
                "clean": 1,
                "flagged": 7,
                "blocked": 1,
                "sidecar-only": 1,
                "paper-blocking": 1,
                "missing": 0,
            },
            "gate_summary": {
                "garak_redteam": {
                    "status": "blocked",
                    "ready": True,
                    "source_experiment_id": "exp3285",
                    "garak_gate_passed": False,
                    "dataflip_gate_passed": True,
                    "blocker_reasons": ["garak_attack_success_or_error_gate_failed"],
                },
                "repair_panel": {
                    "status": "flagged",
                    "ready": True,
                    "source_experiment_id": "exp3290",
                    "headline_claim_allowed": False,
                    "repair_panel_ran": True,
                    "blocker_reasons": [],
                },
                "kan_boundary": {
                    "status": "sidecar-only",
                    "ready": True,
                    "source_experiment_id": "exp3288",
                    "kan_boundary_decision": "retire_from_prompt_injection_headline",
                },
            },
            "rows": [
                {
                    "experiment_id": "exp3285",
                    "role": "full_garak_dataflip_redteam",
                    "status": "blocked",
                    "summary": {
                        "attack_success_rate": 0.311111,
                        "garak_gate_passed": False,
                        "garak_redteam_eval_ready": True,
                    },
                    "quality_flags": [],
                    "bounded_claims": [],
                    "blocker_reasons": ["garak_attack_success_or_error_gate_failed"],
                },
                {
                    "experiment_id": "exp3288",
                    "role": "kan_sidecar_failure_boundary",
                    "status": "sidecar-only",
                    "summary": {
                        "prior_full_corpus_auroc": 0.475326,
                        "kan_boundary_decision": "retire_from_prompt_injection_headline",
                    },
                    "quality_flags": [],
                    "bounded_claims": [
                        "kan_boundary_decision=retire_from_prompt_injection_headline"
                    ],
                    "blocker_reasons": [],
                },
                {
                    "experiment_id": "exp3290",
                    "role": "gated_sota_repair_micro_panel",
                    "status": "flagged",
                    "summary": {
                        "repair_panel_ran": True,
                        "verified_success_count": 4,
                        "false_accept_count": 0,
                        "headline_claim_allowed": False,
                    },
                    "quality_flags": [{"kind": "DURATION_TOO_SHORT", "detail": "too fast"}],
                    "bounded_claims": ["headline_claim_allowed=false"],
                    "blocker_reasons": [],
                },
            ],
            "honest_verdict": "complete: matrix v36 ready",
        },
    )
    _write_json(
        root,
        mod.EXP3293_REL_PATH,
        {
            "experiment_id": "exp3293",
            "capstone_v304_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 10,
            "garak_gate_passed": False,
            "kan_boundary_resolved": True,
            "kan_boundary_decision": "retire_from_prompt_injection_headline",
            "repair_gate_open": True,
            "repair_micro_panel_headline_eligible": False,
            "next_top_gap": "pass_garak_redteam_gate",
            "honest_verdict": "complete: capstone ready",
        },
    )
    conductor = root / mod.RESEARCH_CONDUCTOR_REL_PATH
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text("# protected conductor fixture\n", encoding="utf-8")


def test_req_report_3296_spec_anchor_and_required_fields() -> None:
    """REQ-REPORT-3296: OpenSpec declares the ledger before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3296" in spec
    assert "SCENARIO-REPORT-3296" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "aggregation_from_upstream_artifacts" in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_3296_builds_v305_boundary_ledger(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3296: prior evidence becomes bounded `.305` rules."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=100.0, now_s=101.25)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["substrate_corrigendum_ready"] is True
    assert artifact["kan_no_retry_ledger_ready"] is True
    assert artifact["kan_prompt_injection_headline_retired"] is True
    assert artifact["prior_kan_auroc"] == pytest.approx(0.475326)
    assert artifact["prior_aligned_instruction_false_positive_rate"] == pytest.approx(1.0)
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["protected_files_untouched"] is True
    assert artifact["no_new_kan_training"] is True
    assert artifact["no_new_garak_run"] is True
    assert artifact["no_new_repair_run"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["reproducibility_checksum"]) == 64

    headline_ids = {row["metric_id"] for row in artifact["headline_eligible_prior_metrics"]}
    assert {
        "exp3272.artifact_checksums_available",
        "exp3272.split_leakage_boundary",
        "exp3285.live_garak_failure_boundary",
    } <= headline_ids
    non_headline_ids = {row["metric_id"] for row in artifact["non_headline_prior_metrics"]}
    assert {
        "exp3273.full_corpus_auroc",
        "exp3290.repair_micro_panel_verified_success_count",
        "exp3272.assembled_example_count",
        "exp3292.matrix_v36_status_counts",
        "exp3293.capstone_v304_publication_blocker_count",
    } <= non_headline_ids

    classes = artifact["prior_metric_classes"]
    assert classes["blocked"][0]["source_experiment_id"] == "exp3285"
    assert classes["sidecar-only"][0]["source_experiment_id"] == "exp3288"
    assert classes["flagged"][0]["source_experiment_id"] == "exp3290"
    assert classes["aggregation-only"][0]["source_experiment_id"] == "exp3292"
    assert classes["not_headline_eligible"][0]["source_experiment_id"] == "exp3273"

    rules = artifact["downstream_usage_rules"]
    assert rules["garak"]["new_live_garak_can_be_headline_candidate"] is True
    assert rules["repair"]["new_exact_repair_can_be_headline_candidate"] is True
    assert rules["kan"]["retry_without_operator_directive_allowed"] is False
    assert rules["corpus"]["prior_dot303_corpus_headline_label_claim_allowed"] is False

    prereqs = {row["prerequisite"] for row in artifact["future_reopen_prerequisites"]}
    assert {
        "operator_directive",
        "materially_different_kan_or_ensemble",
        "leakage_provenance_clean_labels",
        "aligned_benign_false_positive_ceiling",
        "beat_regex_keyword_baselines",
        "paired_delong_noninferiority_pass",
        "garak_pressure_pass",
    } <= prereqs
    assert artifact["field_provenance"]["prior_kan_auroc"]["source"] == (
        mod.EXP3288_REL_PATH.as_posix()
    )


def test_req_report_3296_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3296: helper parsing and validation stay deterministic."""

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    array_file = tmp_path / "array.json"
    array_file.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(array_file) == {}

    assert mod.mapping({"a": 1}) == {"a": 1}
    assert mod.mapping(None) == {}
    assert mod.matrix_row({"rows": [{"experiment_id": "exp1"}]}, "exp2") == {}
    assert mod.metric_float("not-a-number") == 0.0
    assert mod.metric_float(1 / 3) == pytest.approx(0.333333)
    assert mod.duration(5.0, 4.0) == 0.0
    assert mod.file_sha256(tmp_path / "missing.txt") is None

    valid = {
        "substrate_corrigendum_ready": True,
        "kan_no_retry_ledger_ready": True,
        "kan_prompt_injection_headline_retired": True,
        "prior_kan_auroc": 0.475326,
        "prior_aligned_instruction_false_positive_rate": 1.0,
        "headline_eligible_prior_metrics": [{"metric_id": "x"}],
        "non_headline_prior_metrics": [{"metric_id": "y"}],
        "downstream_usage_rules": {"kan": {"retry_without_operator_directive_allowed": False}},
        "future_reopen_prerequisites": [{"prerequisite": "operator_directive"}],
        "protected_files_untouched": True,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "random_seed": mod.RANDOM_SEED,
        "reproducibility_checksum": "0" * 64,
        "duration_s": 0.1,
        "honest_verdict": "complete: ok",
    }
    mod.validate_artifact(valid)
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact({key: valid[key] for key in valid if key != "duration_s"})
    with pytest.raises(ValueError, match="substrate_corrigendum_ready"):
        mod.validate_artifact(valid | {"substrate_corrigendum_ready": False})
    with pytest.raises(ValueError, match="kan_no_retry_ledger_ready"):
        mod.validate_artifact(valid | {"kan_no_retry_ledger_ready": False})
    with pytest.raises(ValueError, match="kan_prompt_injection_headline_retired"):
        mod.validate_artifact(valid | {"kan_prompt_injection_headline_retired": False})
    with pytest.raises(ValueError, match="protected_files_untouched"):
        mod.validate_artifact(valid | {"protected_files_untouched": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(valid | {"honest_verdict": "blocked"})

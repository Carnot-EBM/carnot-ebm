"""Tests for the Exp 2884 milestone .272 capstone artifact.

Spec refs: REQ-REPORT-2884, SCENARIO-REPORT-2884.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v272_2884 as exp2884


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp2873() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: archive_ready=true",
        "archived_milestone": "2026.05.271",
        "activated_milestone": "2026.05.272",
        "archive_already_present": True,
        "paper_ready_from_capstone": True,
    }


def _exp2874() -> dict[str, Any]:
    return {
        "honest_verdict": "success: clean mandated SOTA GGUF runtime provenance recorded",
        "sota_runtime_clean": True,
        "sota_runtime_ready_v4": True,
        "cached_sota_pair_returned_two_loadable_specs": False,
        "llama_cpp_gpu_offload_verified": True,
        "usable_response_count": 8,
        "nonempty_response_count": 8,
        "total_tokens_generated": 8102,
        "tokens_per_second": 110.267178,
    }


def _exp2875_flagged() -> dict[str, Any]:
    return {
        "honest_verdict": "micro_panel_clean_no_benchmark_claim",
        "micro_panel_clean": True,
        "benchmark_claim_made": False,
        "n_prompts": 6,
        "n_nonempty_responses": 6,
        "logprobs_available": True,
        "substitute_telemetry_used": False,
        "auroc_if_computable": None,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
    }


def _exp2876() -> dict[str, Any]:
    return {
        "honest_verdict": "complete_corrigendum_z3_milp_bounds_distinct_no_general_kan_claim",
        "kan_corrigendum_ready": True,
        "tautology_flag_cleared": True,
        "local_error_bound": 0.0625,
        "global_error_bound": 0.09375,
        "bounds_distinct_by_construction": True,
        "milp_backend_available": True,
        "milp_backend_name": "z3",
        "solver_status": "optimal",
    }


def _exp2877() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: exact frontier touches bounded rows",
        "frontier_expansion_ready": True,
        "n_candidate_rows": 1000,
        "n_exact_supported_rows": 8,
        "n_unsupported_rows": 992,
        "unsupported_reasons": {"unsupported_no_manual_exact_constraint": 992},
    }


def _exp2878() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: HaluEval/FEVER local audit ready",
        "error_verifiability_ready": True,
        "n_rows_audited": 1000,
        "actionable_localization_rate": 0.929167,
        "label_consistency_rate": 0.447,
        "remote_llm_called": False,
    }


def _exp2879() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: MBPP/HumanEval manifest-only execution pilot ready",
        "code_manifest_pilot_ready": True,
        "n_mbpp_rows": 1,
        "n_humaneval_rows": 1,
        "deterministic_execution_used": True,
        "sandbox_status": "available: runsc",
        "headline_metric_claim_made": False,
        "pilot_rows": [
            {"corpus": "MBPP", "stable_id": "mbpp-11", "passed": True, "n_tests": 3},
            {
                "corpus": "HumanEval",
                "stable_id": "HumanEval/0",
                "passed": True,
                "n_tests": 7,
            },
        ],
    }


def _exp2880() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: cross-corpus matrix v6 built",
        "cross_corpus_matrix_built": True,
        "source_status_by_artifact": {
            "matrix_v5": "clean",
            "exact_frontier": "clean",
            "error_verifiability": "clean",
            "code_execution_pilot": "clean",
        },
        "clean_row_count": 4,
        "headline_eligible_rows": ["FoVer", "HaluEval/FEVER"],
        "pilot_only_rows": ["MBPP", "HumanEval"],
        "missing_rows": {"TruthfulQA": {"row_status": "missing"}},
        "matrix_rows": [
            {"corpus": "FoVer", "row_status": "headline_eligible"},
            {"corpus": "HaluEval/FEVER", "row_status": "headline_eligible"},
            {"corpus": "MBPP", "row_status": "pilot_only"},
            {"corpus": "HumanEval", "row_status": "pilot_only"},
        ],
        "synthetic_rows_created": False,
    }


def _exp2881() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: recurrence-triggered consolidation ready",
        "continuous_self_learning_task": True,
        "recmem_trigger_ready": True,
        "n_events_ingested": 13,
        "n_recurrence_clusters": 2,
        "n_consolidations_triggered": 2,
        "eager_consolidations_avoided": 11,
        "token_reduction_proxy_pct": 92.65250438511,
        "memory_hash_before": "before",
        "memory_hash_after": "after",
        "contradiction_rate": 0.0,
        "duplicate_rate": 0.769230769231,
        "forgetting_regression_count": 0,
        "live_llm_called": False,
    }


def _exp2882_flagged() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: RecMem-triggered replay matched eager replay",
        "continuous_self_learning_task": True,
        "recmem_trigger_ready": True,
        "recmem_replay_scaleup_ready": True,
        "n_examples": 50,
        "target_examples": 50,
        "target_examples_met": True,
        "energy_delta_mean": 0.146666666667,
        "correctness_delta": 0.0,
        "auroc_delta": 0.0,
        "token_reduction_pct": 99.030702617595,
        "memory_drift_score": 0.0,
        "forgetting_regression_count": 0,
        "model_weights_mutated": False,
        "live_llm_called": False,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
    }


def _exp2883_blocked() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: blocked_thrml_unavailable_local_fallback_ran",
        "thrml_portability_ready": False,
        "blocked_reason": "blocked_thrml_unavailable",
        "preconditions_checked": ["python_version", "thrml_import", "local_fallback_sampler"],
        "thrml_import_available": False,
        "jax_devices": ["cpu:0"],
        "local_fallback_ran": True,
        "sample_count": 32,
        "parity_metrics": {"histogram_sanity_passed": True},
        "hardware_claim_made": False,
    }


def _write_all_sources(root: Path) -> None:
    payloads = {
        exp2884.EXPECTED_ARTIFACTS["exp2873"]: _exp2873(),
        exp2884.EXPECTED_ARTIFACTS["exp2874"]: _exp2874(),
        exp2884.EXPECTED_ARTIFACTS["exp2875"]: _exp2875_flagged(),
        exp2884.EXPECTED_ARTIFACTS["exp2876"]: _exp2876(),
        exp2884.EXPECTED_ARTIFACTS["exp2877"]: _exp2877(),
        exp2884.EXPECTED_ARTIFACTS["exp2878"]: _exp2878(),
        exp2884.EXPECTED_ARTIFACTS["exp2879"]: _exp2879(),
        exp2884.EXPECTED_ARTIFACTS["exp2880"]: _exp2880(),
        exp2884.EXPECTED_ARTIFACTS["exp2881"]: _exp2881(),
        exp2884.EXPECTED_ARTIFACTS["exp2882"]: _exp2882_flagged(),
        exp2884.EXPECTED_ARTIFACTS["exp2883"]: _exp2883_blocked(),
    }
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)


def test_scenario_report_2884_preserves_flagged_pilot_and_blocked_boundaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2884: only clean evidence can create paper-v6 readiness."""

    _write_all_sources(tmp_path)

    artifact = exp2884.build_artifact(tmp_path, started_s=2.0, now_s=5.5)

    required = {
        "honest_verdict",
        "milestone",
        "paper_ready",
        "clean_artifacts",
        "flagged_artifacts",
        "blocked_artifacts",
        "missing_artifacts",
        "pilot_only_artifacts",
        "corrected_271_flags",
        "sota_runtime_clean",
        "micro_panel_clean",
        "kan_tautology_cleared",
        "cross_corpus_matrix_built",
        "headline_eligible_rows",
        "continuous_self_learning_result",
        "thrml_sampler_status",
        "paper_v6_safe_claims",
        "paper_v6_forbidden_claims",
        "top_3_next_actions",
        "field_principles",
        "run_date",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone"] == "2026.05.272"
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(3.5)

    assert artifact["paper_ready"] is True
    assert artifact["sota_runtime_clean"] is True
    assert artifact["micro_panel_clean"] is False
    assert artifact["kan_tautology_cleared"] is True
    assert artifact["cross_corpus_matrix_built"] is True
    assert artifact["headline_eligible_rows"] == ["FoVer", "HaluEval/FEVER"]

    assert artifact["clean_artifacts"] == [
        "exp2873",
        "exp2874",
        "exp2876",
        "exp2877",
        "exp2878",
        "exp2880",
        "exp2881",
    ]
    assert artifact["flagged_artifacts"] == ["exp2875", "exp2882"]
    assert artifact["blocked_artifacts"] == ["exp2883"]
    assert artifact["missing_artifacts"] == []
    assert artifact["pilot_only_artifacts"] == ["exp2879"]

    corrected = artifact["corrected_271_flags"]
    assert corrected["runtime"]["corrected"] is True
    assert corrected["micro_panel"]["corrected"] is False
    assert corrected["kan_pwa_milp"]["corrected"] is True

    matrix = artifact["matrix_v6_comparison"]
    assert matrix["v6_has_more_total_clean_or_pilot_evidence_than_v5"] is True
    assert matrix["new_headline_eligible_rows_vs_v5"] == []
    assert matrix["new_pilot_only_rows_vs_v5"] == ["MBPP", "HumanEval"]

    fr11 = artifact["continuous_self_learning_result"]
    assert fr11["recurrence_trigger_ready"] is True
    assert fr11["replay_scaleup_status"] == "flagged"
    assert fr11["scaleup_claim_clean"] is False
    assert fr11["token_reduction_proxy_pct"] == pytest.approx(92.65250438511)
    assert fr11["scaleup_token_reduction_pct"] == pytest.approx(99.030702617595)
    assert fr11["non_forgetting_status"] == "flagged_scaleup_reports_zero_forgetting"

    assert artifact["thrml_sampler_status"] == (
        "blocked_thrml_unavailable_local_fallback_ran_no_hardware_claim"
    )
    assert any("FoVer and HaluEval/FEVER" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any("Exp 2875" in claim for claim in artifact["paper_v6_forbidden_claims"])
    assert any("Exp 2882" in claim for claim in artifact["paper_v6_forbidden_claims"])
    assert any("hardware" in claim.lower() for claim in artifact["paper_v6_forbidden_claims"])
    assert len(artifact["top_3_next_actions"]) == 3
    assert "research-roadmap.yaml" in artifact["files_not_modified"]
    assert "scripts/research_conductor.py" in artifact["files_not_modified"]


def test_req_report_2884_missing_matrix_prevents_paper_ready(tmp_path: Path) -> None:
    """REQ-REPORT-2884: missing matrix evidence is listed and cannot make readiness."""

    _write_all_sources(tmp_path)
    (tmp_path / exp2884.EXPECTED_ARTIFACTS["exp2880"]).unlink()

    artifact = exp2884.build_artifact(tmp_path)

    assert artifact["paper_ready"] is False
    assert artifact["cross_corpus_matrix_built"] is False
    assert artifact["headline_eligible_rows"] == []
    assert artifact["missing_artifacts"] == ["exp2880"]
    assert artifact["matrix_v6_comparison"]["v6_has_more_total_clean_or_pilot_evidence_than_v5"] is False
    assert any("matrix v6" in action for action in artifact["top_3_next_actions"])


def test_req_report_2884_helper_edges_and_write_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-2884: helper branches classify malformed, blocked, and pilot inputs."""

    assert exp2884.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2884.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    assert exp2884.read_json(array) == {}

    assert exp2884._terminal_success(None) is False
    assert exp2884.classify_artifact("exp2874", {}, present=False) == "missing"
    assert exp2884.classify_artifact(
        "exp2874",
        {"honest_verdict": "complete: ok", "corrigendum_pending": [{"kind": "x"}]},
        present=True,
    ) == "flagged"
    assert exp2884.classify_artifact(
        "exp2874",
        {"honest_verdict": "complete: ok", "adversarial_verify_passed": False},
        present=True,
    ) == "flagged"
    assert exp2884.classify_artifact(
        "exp2874",
        {"honest_verdict": "complete: ok", "adversarial_verify_flags": [{"kind": "x"}]},
        present=True,
    ) == "flagged"
    assert exp2884.classify_artifact(
        "exp2874",
        {"honest_verdict": "complete: ok", "adversarial_verify_summary": {"flag_count": 2}},
        present=True,
    ) == "flagged"
    assert exp2884.classify_artifact(
        "exp2874",
        {"honest_verdict": "blocked_runtime_dependency"},
        present=True,
    ) == "blocked"
    assert exp2884.classify_artifact(
        "exp2883",
        {"honest_verdict": "complete: blocked", "blocked_reason": "blocked_thrml_unavailable"},
        present=True,
    ) == "blocked"
    assert exp2884.classify_artifact("exp2879", _exp2879(), present=True) == "pilot-only"
    assert exp2884.classify_artifact(
        "exp2883",
        {
            "honest_verdict": "complete: ready",
            "thrml_portability_ready": True,
            "hardware_claim_made": False,
        },
        present=True,
    ) == "clean"
    assert exp2884.classify_artifact(
        "exp2876",
        {"honest_verdict": "complete: but missing boolean", "kan_corrigendum_ready": False},
        present=True,
    ) == "blocked"
    assert exp2884.classify_artifact("unknown", {"honest_verdict": "running"}, True) == "missing"
    assert exp2884._number_or_none(True) is None
    assert exp2884._number_or_none("1.0") is None
    assert exp2884._headline_rows({"exp2880": "clean"}, {"headline_eligible_rows": "bad"}) == []
    assert exp2884._thrml_status({}, "missing") == "missing"
    assert exp2884._thrml_status({}, "flagged") == "flagged"
    assert exp2884._thrml_status({"hardware_claim_made": True}, "blocked") == (
        "invalid_hardware_claim_made"
    )
    assert exp2884._thrml_status({"thrml_portability_ready": True}, "clean") == (
        "thrml_portability_ready_no_hardware_claim"
    )
    assert exp2884._thrml_status({"blocked_reason": "blocked_thrml_unavailable"}, "blocked") == (
        "blocked_thrml_unavailable_no_hardware_claim"
    )
    assert exp2884._thrml_status({"blocked_reason": "blocked_other"}, "blocked") == "blocked"

    _write_json(
        tmp_path,
        exp2884.PRIOR_CAPSTONE_REL_PATH,
        {"headline_eligible_rows": ["FoVer"]},
    )
    assert exp2884._matrix_v6_comparison(tmp_path, _exp2880())["v5_headline_eligible_rows"] == [
        "FoVer"
    ]
    clean_fr11 = exp2884._continuous_self_learning_result(
        {
            "exp2881": _exp2881(),
            "exp2882": _exp2882_flagged()
            | {
                "flagged_adversarial": False,
                "corrigendum_pending": [],
                "forgetting_regression_count": 0,
            },
        },
        {"exp2881": "clean", "exp2882": "clean"},
    )
    assert clean_fr11["non_forgetting_status"] == "clean_scaleup_reports_zero_forgetting"
    unready_fr11 = exp2884._continuous_self_learning_result(
        {"exp2881": {}, "exp2882": {"forgetting_regression_count": 1}},
        {"exp2881": "missing", "exp2882": "blocked"},
    )
    assert unready_fr11["non_forgetting_status"] == "not_established"
    assert unready_fr11["safe_fr11_claim"] == "none"
    assert exp2884._top_3_next_actions(
        statuses={"exp2875": "clean", "exp2882": "clean"},
        paper_ready=True,
        matrix_comparison={"new_pilot_only_rows_vs_v5": [], "truthfulqa_status": "present"},
        fr11={"recurrence_trigger_ready": False, "scaleup_claim_clean": True},
    ) == ["Use matrix v6 safe claims only, and leave flagged branches out of paper-v6."]

    _write_all_sources(tmp_path)
    out = exp2884.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "results/experiment_2884_capstone_v272.json"
    assert payload["duration_s"] == pytest.approx(0.25)
    assert payload["honest_verdict"].startswith("complete:")

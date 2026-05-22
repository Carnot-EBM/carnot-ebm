"""Tests for the Exp 2872 milestone .271 capstone artifact.

Spec refs: REQ-REPORT-2872, SCENARIO-REPORT-2872.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v271_2872 as exp2872


MANDATED_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(root: Path, rel_path: str, payload: dict[str, object]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp2861() -> dict[str, object]:
    return {
        "honest_verdict": "complete: archive_ready=true",
        "archived_milestone": "2026.05.270",
        "activated_milestone": "2026.05.271",
        "paper_ready_from_capstone": False,
    }


def _exp2862_flagged() -> dict[str, object]:
    return {
        "honest_verdict": "success: mandated SOTA GGUF produced usable GPU-backed output",
        "sota_runtime_ready_v3": True,
        "selected_model_hf_id": MANDATED_MODEL,
        "selected_model_path": "/cache/gemma-4-26b.gguf",
        "cached_sota_pair_returned_two_loadable_specs": False,
        "llama_cpp_gpu_offload_verified": True,
        "usable_response_count": 1,
        "total_tokens_generated": 2,
        "tokens_per_second": 0.23,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
    }


def _exp2863() -> dict[str, object]:
    return {
        "honest_verdict": "complete: eval manifest contract ready",
        "manifest_contract_ready": True,
        "halueval_ready": True,
        "fever_ready": True,
        "mbpp_ready": True,
        "humaneval_ready": True,
        "truthfulqa_ready": True,
        "synthetic_rows_created": False,
    }


def _exp2864() -> dict[str, object]:
    return {
        "honest_verdict": "complete: HaluEval/FEVER local calibration ready",
        "halueval_fever_ready": True,
        "full_benchmark_ready": True,
        "live_model_invoked": False,
        "halueval_auroc": 0.553072,
        "fever_auroc": 0.33114331723027374,
        "halueval_n_examples": 500,
        "fever_n_examples": 500,
        "adversarial_verify_passed": True,
        "adversarial_verify_flags": [],
    }


def _exp2865() -> dict[str, object]:
    return {
        "honest_verdict": "complete: cross-corpus matrix built from 2 clean corpus rows",
        "cross_corpus_matrix_built": True,
        "paper_eligible_rows": ["FoVer", "HaluEval/FEVER"],
        "row_status_by_corpus": {
            "FoVer": "clean",
            "HaluEval/FEVER": "clean",
            "MBPP": "missing",
            "HumanEval": "missing",
            "TruthfulQA": "missing",
        },
        "verifier_corpus_dual_matrix": {
            "FoVer": {
                "corpus": "FoVer",
                "production_auroc": 0.9131336,
                "architecture_only_auroc": 0.8946624,
                "learning_contribution": 0.0184712,
                "n_examples": 1000,
                "n_seeds": 5,
                "row_status": "clean",
                "source_artifact": "results/experiment_2850_fover_dual_condition_integrity_v4.json",
            },
            "HaluEval/FEVER": {
                "corpus": "HaluEval/FEVER",
                "measured_auroc_by_dataset": {
                    "halueval": 0.553072,
                    "fever": 0.33114331723027374,
                },
                "n_examples": 1000,
                "n_examples_by_dataset": {"halueval": 500, "fever": 500},
                "row_status": "clean",
                "source_artifact": "results/experiment_2864_halueval_fever_full_calibration_v3.json",
            },
        },
        "excluded_from_headline": {
            "MBPP": "source_artifact_missing",
            "HumanEval": "source_artifact_missing",
            "TruthfulQA": "source_artifact_missing",
        },
    }


def _exp2866() -> dict[str, object]:
    return {
        "honest_verdict": "complete: tiny exact Z3 arithmetic frontier available",
        "exact_beaver_implemented": False,
        "exact_frontier_available": True,
        "n_examples": 6,
        "solver_used": "z3-solver",
    }


def _exp2867() -> dict[str, object]:
    return {
        "honest_verdict": "complete: residual-drift MUS-proxy prioritizer built",
        "drift_mus_diagnostic_ready": True,
        "n_failure_rows": 4,
        "heuristic_checks_to_conflict": 4.0,
    }


def _exp2868() -> dict[str, object]:
    return {
        "honest_verdict": "complete: offline recurrence replay backend ready",
        "offline_recurrence_backend_ready": True,
        "backend_module_path": "carnot.eval.offline_recurrence_backend_adapter_v2",
        "live_model_invoked": False,
        "no_model_weight_mutation": True,
    }


def _exp2869() -> dict[str, object]:
    return {
        "honest_verdict": "complete: offline verifier-feedback replay lowered energy",
        "continuous_self_learning_task": True,
        "fr11_self_learning_ready": True,
        "offline_recurrence_backend_used": "carnot.eval.offline_recurrence_backend_adapter_v2",
        "live_model_invoked": False,
        "no_model_weight_mutation": True,
        "n_examples": 9,
        "max_loops": 3,
        "recurrence_success_rate": 0.222222222222,
        "energy_delta_mean": 0.081481481482,
        "correctness_delta": 0.0,
        "forgetting_regression_count": 0,
        "memory_hash_before": "before-hash",
        "memory_hash_after": "after-hash",
        "source_counts": {"FoVer": 3, "HaluEval": 3, "FEVER": 3},
    }


def _exp2870_flagged() -> dict[str, object]:
    return {
        "honest_verdict": "micro_panel_complete_no_full_benchmark_claim",
        "micro_panel_ready": True,
        "live_model_invoked": True,
        "model_specs": [{"hf_id": MANDATED_MODEL, "model_path": "/cache/gemma.gguf"}],
        "models_used": [MANDATED_MODEL],
        "n_examples": 10,
        "usable_response_count": 0,
        "first_token_confidence_available": False,
        "spilled_energy_available": False,
        "first_token_confidence_auroc": None,
        "spilled_energy_auroc": None,
        "blocked_metrics": ["blocked_logprobs_unavailable"],
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
    }


def _exp2871_flagged() -> dict[str, object]:
    return {
        "honest_verdict": "complete_with_exact_enumerated_fallback_no_general_milp_or_network_claim",
        "kan_pwa_milp_verifier_ready": True,
        "pwa_abstraction_built": True,
        "milp_or_exact_property_checked": True,
        "property_verified": True,
        "local_error_bound": 0.0625,
        "global_error_bound": 0.0625,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
    }


def _write_all_inputs(root: Path) -> None:
    payloads = {
        "results/experiment_2861_archive_v270_activate_v271.json": _exp2861(),
        "results/experiment_2862_sota_runtime_cache_offload_resolver_v3.json": _exp2862_flagged(),
        "results/experiment_2863_eval_manifest_contract_v2.json": _exp2863(),
        "results/experiment_2864_halueval_fever_full_calibration_v3.json": _exp2864(),
        "results/experiment_2865_cross_corpus_matrix_v5.json": _exp2865(),
        "results/experiment_2866_beaver_exact_tiny_frontier_v1.json": _exp2866(),
        "results/experiment_2867_drift_mus_prioritizer_v2.json": _exp2867(),
        "results/experiment_2868_offline_recurrence_backend_adapter_v2.json": _exp2868(),
        "results/experiment_2869_fr11_continuous_self_learning_replay_v3.json": _exp2869(),
        "results/experiment_2870_sota_energy_baseline_micro_panel_v1.json": _exp2870_flagged(),
        "results/experiment_2871_kan_pwa_milp_tiny_verifier_v1.json": _exp2871_flagged(),
    }
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)


def test_scenario_report_2872_classifies_and_preserves_claim_boundaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2872: clean rows can coexist with flagged side artifacts."""

    _write_all_inputs(tmp_path)

    artifact = exp2872.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    serialized = json.dumps(artifact)

    required = {
        "honest_verdict",
        "milestone",
        "paper_ready",
        "sota_runtime_ready_v3",
        "manifest_contract_ready",
        "cross_corpus_matrix_built",
        "fr11_self_learning_ready",
        "continuous_self_learning_completed",
        "headline_eligible_rows",
        "clean_artifacts",
        "blocked_artifacts",
        "missing_artifacts",
        "adversarially_flagged_artifacts",
        "primary_corpus_results",
        "self_learning_summary",
        "runtime_summary",
        "claim_boundary_notes",
        "top_3_next_actions",
        "pushed",
        "scripts_research_conductor_modified",
        "field_principles",
        "run_date",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone"] == "2026.05.271"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["run_date"] == "20260522"
    assert artifact["pushed"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert MANDATED_MODEL not in serialized
    assert "GGUF" not in serialized

    assert artifact["paper_ready"] is True
    assert artifact["headline_eligible_rows"] == ["FoVer", "HaluEval/FEVER"]
    assert artifact["clean_artifacts"] == [
        "exp2861",
        "exp2863",
        "exp2864",
        "exp2865",
        "exp2866",
        "exp2867",
        "exp2868",
        "exp2869",
    ]
    assert artifact["blocked_artifacts"] == []
    assert artifact["missing_artifacts"] == []
    assert artifact["adversarially_flagged_artifacts"] == ["exp2862", "exp2870", "exp2871"]

    primary = artifact["primary_corpus_results"]
    assert primary["FoVer"]["status"] == "clean"
    assert primary["FoVer"]["production_auroc"] == pytest.approx(0.9131336)
    assert primary["HaluEval/FEVER"]["measured_auroc_by_dataset"]["halueval"] == pytest.approx(
        0.553072
    )
    assert primary["MBPP"]["status"] == "missing"
    assert primary["MBPP"]["production_auroc"] is None

    runtime = artifact["runtime_summary"]
    assert artifact["sota_runtime_ready_v3"] is True
    assert runtime["source_reported_sota_runtime_ready_v3"] is True
    assert runtime["sota_runtime_artifact_clean"] is False
    assert runtime["exp2870_invoked_mandated_sota_model"] is True
    assert runtime["exp2870_headline_clean"] is False
    assert runtime["models_used"] == ["mandated_sota_model"]
    assert runtime["mandated_sota_model_count"] == 1
    assert runtime["model_identities_redacted"] is True

    self_learning = artifact["self_learning_summary"]
    assert artifact["fr11_self_learning_ready"] is True
    assert artifact["continuous_self_learning_completed"] is True
    assert self_learning["energy_improved"] is True
    assert self_learning["correctness_improved"] is False
    assert self_learning["no_model_weight_mutation"] is True
    assert self_learning["memory_hash_changed"] is True

    assert len(artifact["top_3_next_actions"]) == 3
    assert any("adversarial" in note for note in artifact["claim_boundary_notes"])


def test_req_report_2872_missing_and_blocked_inputs_do_not_create_paper_ready(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2872: missing matrix and blocked replay keep readiness false."""

    _write_all_inputs(tmp_path)
    (tmp_path / "results" / "experiment_2865_cross_corpus_matrix_v5.json").unlink()
    _write_json(
        tmp_path,
        "results/experiment_2869_fr11_continuous_self_learning_replay_v3.json",
        {
            "honest_verdict": "blocked_missing_exp2865_artifact",
            "continuous_self_learning_task": True,
            "fr11_self_learning_ready": False,
            "live_model_invoked": False,
            "no_model_weight_mutation": True,
            "energy_delta_mean": 0.0,
            "correctness_delta": 0.0,
        },
    )

    artifact = exp2872.build_artifact(tmp_path)

    assert artifact["paper_ready"] is False
    assert artifact["cross_corpus_matrix_built"] is False
    assert artifact["continuous_self_learning_completed"] is False
    assert "exp2865" in artifact["missing_artifacts"]
    assert "exp2869" in artifact["blocked_artifacts"]
    assert artifact["headline_eligible_rows"] == []


def test_req_report_2872_helper_branches_and_write_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-2872: helper branches classify bad inputs and persist JSON."""

    assert exp2872.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert exp2872.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp2872.read_json(array) == {}

    assert exp2872.classify_artifact({}, present=False) == "missing"
    assert exp2872.classify_artifact({"honest_verdict": "blocked_cache"}, present=True) == "blocked"
    assert exp2872.classify_artifact({"flagged_adversarial": True}, present=True) == (
        "adversarially_flagged"
    )
    assert exp2872.classify_artifact({"corrigendum_pending": ["x"]}, present=True) == (
        "adversarially_flagged"
    )
    assert exp2872.classify_artifact(
        {"honest_verdict": "complete: ok", "adversarial_verify_flags": [{"kind": "x"}]},
        present=True,
    ) == "adversarially_flagged"
    assert exp2872.classify_artifact({"adversarial_verify_passed": False}, present=True) == (
        "adversarially_flagged"
    )
    assert exp2872.classify_artifact({"honest_verdict": "success: ok"}, present=True) == "clean"
    assert exp2872.classify_artifact({"honest_verdict": "running"}, present=True) == "missing"
    assert exp2872.classify_artifact({"honest_verdict": None}, present=True) == "missing"
    assert exp2872.classify_artifact(
        {"adversarial_verify_summary": {"flag_count": 1}},
        present=True,
    ) == "adversarially_flagged"
    assert exp2872.classify_artifact(
        {"honest_verdict": "complete: matrix not built", "cross_corpus_matrix_built": False},
        present=True,
        exp_id="exp2865",
    ) == "blocked"
    assert exp2872._number_or_none(True) is None
    assert exp2872._models_from_payload(
        {"model_specs": [{"hf_id": MANDATED_MODEL}, {"hf_id": 3}, "not-a-spec"]}
    ) == [MANDATED_MODEL]
    assert exp2872._models_from_payload({}) == []
    assert exp2872._rows_clean_at_source(
        tmp_path,
        {"FoVer": {"headline_eligible": True, "source_artifact": None}},
    )
    flagged_source = tmp_path / "results" / "flagged-source.json"
    flagged_source.parent.mkdir(parents=True, exist_ok=True)
    flagged_source.write_text('{"flagged_adversarial": true}', encoding="utf-8")
    assert not exp2872._rows_clean_at_source(
        tmp_path,
        {
            "FoVer": {
                "headline_eligible": True,
                "source_artifact": "results/flagged-source.json",
            }
        },
    )
    assert exp2872._top_3_next_actions(
        [],
        {"FoVer": {"status": "clean"}},
        {"correctness_improved": True},
    ) == ["Regenerate paper-v6 Section 5 only from the clean FoVer and HaluEval/FEVER matrix rows."]

    _write_all_inputs(tmp_path)
    out = exp2872.write_artifact(tmp_path, started_s=1.0, now_s=1.75)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "results/experiment_2872_capstone_v271.json"
    assert payload["duration_s"] == pytest.approx(0.75)
    assert payload["honest_verdict"].startswith("complete:")

"""Tests for the Exp 2896 milestone .273 capstone artifact.

Spec refs: REQ-REPORT-2896, SCENARIO-REPORT-2896.

The test fixtures construct each .273 source artifact with the minimum
fields the classifier actually consults, then verify the capstone:

- classifies adversarially-flagged inputs as flagged even when their
  "ready" booleans are True,
- routes deliberate pilot-only / taxonomy-only artifacts into their own
  buckets without polluting flagged/blocked,
- only marks ``paper_ready`` True when matrix v7 itself is clean with
  FoVer plus a second headline row,
- carries the .272 micro-panel and RecMem scaleup corrections honestly
  in ``corrected_272_flags``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v273_2896 as exp2896


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp2885_flagged() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: archive_ready=true; archived_milestone=2026.05.272",
        "archived_milestone": "2026.05.272",
        "activated_milestone": "2026.05.273",
        "archive_already_present": True,
        "paper_ready_from_capstone": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
            {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
        ],
    }


def _exp2886_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: micro_panel_clean_no_benchmark_claim_v3",
        "micro_panel_clean": True,
        "benchmark_claim_made": False,
        "logprobs_available": True,
        "random_seed": 2886,
        "reproducibility_checksum": "abc",
    }


def _exp2887_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: causal RecMem and fast/slow memory separated replay metrics cleanly",
        "continuous_self_learning_task": True,
        "fr11_scaleup_clean": True,
        "best_policy": "fast_slow_memory",
        "n_examples": 50,
        "policies_compared": ["eager_replay", "recmem_causal_triggered", "fast_slow_memory"],
        "energy_delta_by_policy": {
            "eager_replay": 0.139,
            "recmem_causal_triggered": 0.125,
            "fast_slow_memory": 0.132,
        },
        "correctness_delta_by_policy": {
            "eager_replay": 0.0,
            "recmem_causal_triggered": 0.0,
            "fast_slow_memory": 0.0,
        },
        "auroc_delta_by_policy": {
            "eager_replay": 0.0,
            "recmem_causal_triggered": 0.0,
            "fast_slow_memory": 0.0,
        },
        "contradiction_rate_by_policy": {
            "eager_replay": 0.0,
            "recmem_causal_triggered": 0.0,
            "fast_slow_memory": 0.0,
        },
        "duplicate_rate_by_policy": {
            "eager_replay": 0.96,
            "recmem_causal_triggered": 0.0,
            "fast_slow_memory": 0.0,
        },
        "memory_drift_by_policy": {
            "eager_replay": 0.0,
            "recmem_causal_triggered": 0.0,
            "fast_slow_memory": 0.0,
        },
        "forgetting_regression_count_by_policy": {
            "eager_replay": 0,
            "recmem_causal_triggered": 0,
            "fast_slow_memory": 0,
        },
        "policy_metrics": {
            "eager_replay": {"token_reduction_pct": 0.0},
            "recmem_causal_triggered": {"token_reduction_pct": 99.3},
            "fast_slow_memory": {"token_reduction_pct": 98.6},
        },
        "live_llm_called": False,
        "model_weights_mutated": False,
        "exp2882_flag_diagnosis": {"root_cause": "retroactive_cluster_application"},
        "adversarial_clean_checks": {
            "non_tautological_policy_energy": True,
            "fast_slow_separates_from_recmem": True,
        },
    }


def _exp2888_taxonomy() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: TruthfulQA local taxonomy manifest ready without generated-answer metrics",
        "truthfulqa_taxonomy_ready": True,
        "headline_metric_claim_made": False,
        "n_rows_available": 200,
        "n_rows_materialized": 100,
        "generated_answer_metrics_available": False,
    }


def _exp2889_flagged() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: bounded SOTA GGUF generation executed cleanly but no candidate passed tests",
        "generated_code_row_clean": True,
        "manifest_contract_ready": True,
        "headline_metric_claim_made": False,
        "pass_rate_if_computable": 0.0,
        "row_status": "pilot_only_clean_no_passes",
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
    }


def _exp2890_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: MBPP/HumanEval structural dependency verifier metadata ready",
        "structural_dependency_verifier_ready": True,
        "headline_metric_claim_made": False,
        "n_contracts_built": 10,
        "n_rows_verified": 20,
        "violation_types": {"missing_function_definition": 4, "parse_error": 5},
        "contract_schema_errors": [],
    }


def _exp2891_pilot() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: local CCTU-style executable constraint validator pilot ready",
        "cctu_validator_ready": True,
        "headline_metric_claim_made": False,
        "n_cases": 5,
        "category_coverage": {"behavior": {"passed": 0, "total": 1}},
        "executable_validation_used": True,
    }


def _exp2892_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: deterministic VeriCoT frontier rows available",
        "vericot_frontier_ready": True,
        "n_candidate_rows": 1100,
        "n_vericot_supported_rows": 25,
        "n_unsupported_rows": 1075,
        "solver_backend": "z3-solver 4.16.0",
        "autoformalization_llm_called": False,
    }


def _exp2893_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: tiny KAN PWA/MILP complexity accounting ready; no hardware execution or analog claim",
        "kan_complexity_accounting_ready": True,
        "analog_kan_claim_made": False,
        "hardware_execution_claim_made": False,
        "complexity_metrics": {"bop_count": 96},
        "status": "complete",
    }


def _exp2894_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: cross-corpus matrix v7 built from 5 clean headline/pilot/taxonomy rows",
        "cross_corpus_matrix_built": True,
        "clean_row_count": 5,
        "headline_eligible_rows": ["FoVer", "HaluEval/FEVER"],
        "pilot_only_rows": ["MBPP", "HumanEval"],
        "taxonomy_only_rows": ["TruthfulQA"],
        "source_status_by_artifact": {"matrix_v6": "clean", "generated_code": "flagged"},
    }


def _exp2895_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: paper-v6 evidence table ready; headline=2; pilot_only=2; taxonomy_only=1; blocked=2",
        "paper_evidence_table_ready": True,
        "arxiv_submission_performed": False,
        "headline_claims": [
            {"corpus": "FoVer", "statement": "..."},
            {"corpus": "HaluEval/FEVER", "statement": "..."},
        ],
        "pilot_only_statements": [{"corpus": "MBPP"}, {"corpus": "HumanEval"}],
        "taxonomy_only_statements": [{"corpus": "TruthfulQA"}],
        "blocked_claims": [{"corpus": "MBPP"}, {"corpus": "HumanEval"}],
    }


def _write_all_sources(root: Path, **overrides: dict[str, Any]) -> None:
    payloads = {
        "exp2885": _exp2885_flagged(),
        "exp2886": _exp2886_clean(),
        "exp2887": _exp2887_clean(),
        "exp2888": _exp2888_taxonomy(),
        "exp2889": _exp2889_flagged(),
        "exp2890": _exp2890_clean(),
        "exp2891": _exp2891_pilot(),
        "exp2892": _exp2892_clean(),
        "exp2893": _exp2893_clean(),
        "exp2894": _exp2894_clean(),
        "exp2895": _exp2895_clean(),
    }
    payloads.update(overrides)
    for exp_id, payload in payloads.items():
        _write_json(root, exp2896.EXPECTED_ARTIFACTS[exp_id], payload)


def _write_prior_capstone(root: Path) -> None:
    _write_json(
        root,
        exp2896.PRIOR_CAPSTONE_REL_PATH,
        {"headline_eligible_rows": ["FoVer", "HaluEval/FEVER"]},
    )


def test_scenario_report_2896_preserves_flagged_pilot_taxonomy_and_clean(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2896: clean evidence drives paper_ready and all buckets fill honestly."""

    _write_all_sources(tmp_path)
    _write_prior_capstone(tmp_path)

    artifact = exp2896.build_artifact(tmp_path, started_s=10.0, now_s=11.5)

    required = {
        "honest_verdict",
        "milestone",
        "paper_ready",
        "clean_artifacts",
        "flagged_artifacts",
        "blocked_artifacts",
        "missing_artifacts",
        "pilot_only_artifacts",
        "taxonomy_only_artifacts",
        "corrected_272_flags",
        "micro_panel_clean",
        "fr11_scaleup_clean",
        "cross_corpus_matrix_built",
        "headline_eligible_rows",
        "continuous_self_learning_result",
        "constraint_benchmark_status",
        "kan_complexity_status",
        "paper_v6_safe_claims",
        "paper_v6_forbidden_claims",
        "top_3_next_actions",
        "field_principles",
        "run_date",
        "duration_s",
        "inference_substrate",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone"] == "2026.05.273"
    assert artifact["run_date"] == "20260523"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(1.5)

    # Classification: exp2885 + exp2889 are flagged; CCTU=pilot; TruthfulQA=taxonomy.
    assert "exp2885" in artifact["flagged_artifacts"]
    assert "exp2889" in artifact["flagged_artifacts"]
    assert artifact["pilot_only_artifacts"] == ["exp2891"]
    assert artifact["taxonomy_only_artifacts"] == ["exp2888"]
    # Everything else is clean.
    for clean_id in ("exp2886", "exp2887", "exp2890", "exp2892", "exp2893", "exp2894", "exp2895"):
        assert clean_id in artifact["clean_artifacts"]
    assert artifact["blocked_artifacts"] == []
    assert artifact["missing_artifacts"] == []

    # Headline rows come only from clean matrix v7.
    assert artifact["paper_ready"] is True
    assert artifact["cross_corpus_matrix_built"] is True
    assert artifact["headline_eligible_rows"] == ["FoVer", "HaluEval/FEVER"]
    assert artifact["micro_panel_clean"] is True
    assert artifact["fr11_scaleup_clean"] is True

    # .272 flags both corrected by .273 work.
    corrected = artifact["corrected_272_flags"]
    assert corrected["micro_panel"]["corrected"] is True
    assert corrected["micro_panel"]["correcting_artifact"] == "exp2886"
    assert corrected["fr11_scaleup"]["corrected"] is True
    assert corrected["fr11_scaleup"]["correcting_artifact"] == "exp2887"
    assert corrected["fr11_scaleup"]["best_policy"] == "fast_slow_memory"
    assert corrected["fr11_scaleup"]["exp2882_root_cause_recorded"] is True

    # FR-11 result preserves the three-policy breakdown.
    fr11 = artifact["continuous_self_learning_result"]
    assert fr11["best_policy"] == "fast_slow_memory"
    assert fr11["fr11_scaleup_clean"] is True
    assert fr11["safe_fr11_claim"] == "fast_slow_vs_recmem_vs_eager_separated"
    assert fr11["non_tautological_policy_energy"] is True
    assert fr11["fast_slow_separates_from_recmem"] is True
    assert fr11["live_llm_called"] is False
    assert fr11["token_reduction_pct_by_policy"]["fast_slow_memory"] == pytest.approx(98.6)
    assert fr11["energy_delta_by_policy"]["recmem_causal_triggered"] == pytest.approx(0.125)

    # Constraint expansion summary.
    constraint = artifact["constraint_benchmark_status"]
    assert constraint["cctu_pilot"]["ready"] is True
    assert constraint["cctu_pilot"]["status"] == "pilot-only"
    assert constraint["vericot_frontier"]["ready"] is True
    assert constraint["vericot_frontier"]["n_vericot_supported_rows"] == 25
    assert constraint["structural_verifier"]["ready"] is True

    # KAN complexity status string.
    assert (
        artifact["kan_complexity_status"]
        == "complexity_accounting_ready_no_hardware_or_analog_claim"
    )

    # Matrix v7 vs v6 comparison.
    cmp = artifact["matrix_v7_comparison"]
    assert cmp["new_headline_eligible_rows_vs_v6"] == []
    assert cmp["new_pilot_only_rows_vs_v6"] == ["MBPP", "HumanEval"]
    assert cmp["new_taxonomy_only_rows_vs_v6"] == ["TruthfulQA"]
    assert cmp["matrix_v7_adds_headline_evidence_beyond_v6"] is False

    # Claims surface what is allowed and what is forbidden.
    assert any("FoVer" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any("fast/slow memory" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any("Exp 2889" in claim for claim in artifact["paper_v6_forbidden_claims"])
    assert any("matrix v7" in claim for claim in artifact["paper_v6_forbidden_claims"])
    assert any("Exp 2885" in claim for claim in artifact["paper_v6_forbidden_claims"])

    # Three actionable next steps, no more.
    assert len(artifact["top_3_next_actions"]) == 3
    assert any("Exp 2889" in action for action in artifact["top_3_next_actions"])

    # Read-only files declared as not modified.
    assert "research-roadmap.yaml" in artifact["files_not_modified"]
    assert "scripts/research_conductor.py" in artifact["files_not_modified"]


def test_req_report_2896_missing_matrix_blocks_paper_ready(tmp_path: Path) -> None:
    """REQ-REPORT-2896: when matrix v7 is missing, paper_ready cannot be True."""

    _write_all_sources(tmp_path)
    _write_prior_capstone(tmp_path)
    (tmp_path / exp2896.EXPECTED_ARTIFACTS["exp2894"]).unlink()

    artifact = exp2896.build_artifact(tmp_path)

    assert artifact["paper_ready"] is False
    assert artifact["cross_corpus_matrix_built"] is False
    assert artifact["headline_eligible_rows"] == []
    assert "exp2894" in artifact["missing_artifacts"]
    assert artifact["matrix_v7_comparison"]["matrix_v7_adds_headline_evidence_beyond_v6"] is False
    # Either the "restore clean matrix v7" action or the "add headline row to v8" action
    # surfaces the matrix gap; both qualify.
    assert any("matrix" in action for action in artifact["top_3_next_actions"])


def test_req_report_2896_classifier_branches(tmp_path: Path) -> None:
    """REQ-REPORT-2896: classifier honours all evidence buckets."""

    # Missing artifact.
    assert exp2896.classify_artifact("exp2886", {}, present=False) == "missing"

    # Flagged via flagged_adversarial wins over ready booleans.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {
                "honest_verdict": "complete: ok",
                "micro_panel_clean": True,
                "flagged_adversarial": True,
            },
            present=True,
        )
        == "flagged"
    )
    # Flagged via corrigendum_pending list.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {
                "honest_verdict": "complete: ok",
                "micro_panel_clean": True,
                "corrigendum_pending": [{"kind": "x"}],
            },
            present=True,
        )
        == "flagged"
    )
    # Flagged via adversarial_verify_flags.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {
                "honest_verdict": "complete: ok",
                "micro_panel_clean": True,
                "adversarial_verify_flags": [{"kind": "x"}],
            },
            present=True,
        )
        == "flagged"
    )
    # Flagged via adversarial_verify_summary flag_count > 0.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {
                "honest_verdict": "complete: ok",
                "micro_panel_clean": True,
                "adversarial_verify_summary": {"flag_count": 3},
            },
            present=True,
        )
        == "flagged"
    )
    # Flagged via adversarial_verify_passed=False.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {
                "honest_verdict": "complete: ok",
                "micro_panel_clean": True,
                "adversarial_verify_passed": False,
            },
            present=True,
        )
        == "flagged"
    )

    # Pilot-only path: CCTU with terminal + no headline claim.
    assert (
        exp2896.classify_artifact(
            "exp2891",
            {
                "honest_verdict": "complete: pilot",
                "cctu_validator_ready": True,
                "headline_metric_claim_made": False,
            },
            present=True,
        )
        == "pilot-only"
    )
    # Taxonomy-only path: TruthfulQA with terminal + no headline claim.
    assert (
        exp2896.classify_artifact(
            "exp2888",
            {
                "honest_verdict": "complete: taxonomy",
                "truthfulqa_taxonomy_ready": True,
                "headline_metric_claim_made": False,
            },
            present=True,
        )
        == "taxonomy-only"
    )
    # Blocked verdict.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {"honest_verdict": "blocked_runtime"},
            present=True,
        )
        == "blocked"
    )
    # Blocked via blocked_reason.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {
                "honest_verdict": "complete: but blocked downstream",
                "blocked_reason": "blocked_thrml_unavailable",
            },
            present=True,
        )
        == "blocked"
    )
    # Clean only when terminal AND required booleans present.
    assert (
        exp2896.classify_artifact("exp2886", _exp2886_clean(), present=True) == "clean"
    )
    # Terminal but required boolean missing => blocked.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {"honest_verdict": "complete: ok", "micro_panel_clean": False},
            present=True,
        )
        == "blocked"
    )
    # Non-terminal verdict => missing.
    assert (
        exp2896.classify_artifact(
            "exp2886",
            {"honest_verdict": "running"},
            present=True,
        )
        == "missing"
    )

    # _number_or_none rejects bools and strings.
    assert exp2896._number_or_none(True) is None
    assert exp2896._number_or_none("1.5") is None
    assert exp2896._number_or_none(float("nan")) is None
    assert exp2896._number_or_none(2.5) == 2.5

    # _terminal_success non-string.
    assert exp2896._terminal_success(None) is False
    assert exp2896._terminal_success("complete: ok") is True
    assert exp2896._terminal_success("passed_ok") is True
    assert exp2896._terminal_success("running") is False

    # _headline_rows returns [] when v7 not clean or rows malformed.
    assert exp2896._headline_rows({"exp2894": "flagged"}, {"headline_eligible_rows": ["x"]}) == []
    assert exp2896._headline_rows({"exp2894": "clean"}, {"headline_eligible_rows": "bad"}) == []

    # KAN status branches.
    assert exp2896._kan_complexity_status({}, "missing") == "missing"
    assert exp2896._kan_complexity_status({}, "flagged") == "flagged"
    assert (
        exp2896._kan_complexity_status({"analog_kan_claim_made": True}, "clean")
        == "invalid_hardware_or_analog_claim_made"
    )
    assert (
        exp2896._kan_complexity_status({"hardware_execution_claim_made": True}, "clean")
        == "invalid_hardware_or_analog_claim_made"
    )
    assert (
        exp2896._kan_complexity_status({}, "clean")
        == "complexity_accounting_ready_no_hardware_or_analog_claim"
    )
    assert exp2896._kan_complexity_status({}, "blocked") == "blocked"
    assert exp2896._kan_complexity_status({}, "weird") == "weird"


def test_req_report_2896_helper_io_edges(tmp_path: Path) -> None:
    """REQ-REPORT-2896: read_json tolerates missing, malformed, and non-mapping JSON."""

    assert exp2896.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert exp2896.read_json(bad) == {}
    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2]", encoding="utf-8")
    assert exp2896.read_json(array_json) == {}


def test_req_report_2896_handles_malformed_fr11_and_no_prior_capstone(tmp_path: Path) -> None:
    """REQ-REPORT-2896: tolerates malformed FR-11 sub-dicts and missing prior capstone."""

    _write_all_sources(
        tmp_path,
        exp2887={
            "honest_verdict": "complete: but malformed",
            "continuous_self_learning_task": True,
            "fr11_scaleup_clean": True,
            "policies_compared": "not-a-list",
            "energy_delta_by_policy": "bad",
            "correctness_delta_by_policy": None,
            "auroc_delta_by_policy": [],
            "contradiction_rate_by_policy": 0,
            "duplicate_rate_by_policy": "x",
            "memory_drift_by_policy": None,
            "forgetting_regression_count_by_policy": "x",
            "policy_metrics": None,
            "best_policy": None,
        },
    )
    # Intentionally NO prior capstone written so fallback path runs.

    artifact = exp2896.build_artifact(tmp_path)

    fr11 = artifact["continuous_self_learning_result"]
    # Falls back to empty containers without raising.
    assert fr11["policies_compared"] == []
    assert fr11["energy_delta_by_policy"] == {}
    assert fr11["correctness_delta_by_policy"] == {}
    assert fr11["forgetting_regression_count_by_policy"] == {}
    assert fr11["token_reduction_pct_by_policy"] == {}
    # Missing prior capstone => fallback headline list.
    assert artifact["matrix_v7_comparison"]["v6_headline_eligible_rows"] == [
        "FoVer",
        "HaluEval/FEVER",
    ]


def test_req_report_2896_write_artifact_round_trip(tmp_path: Path) -> None:
    """REQ-REPORT-2896: write_artifact emits valid JSON containing the build payload."""

    _write_all_sources(tmp_path)
    _write_prior_capstone(tmp_path)

    out_path = exp2896.write_artifact(tmp_path)
    assert out_path == tmp_path / exp2896.OUTPUT_REL_PATH

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["milestone"] == "2026.05.273"
    assert written["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert written["honest_verdict"].startswith("complete:")
    assert "paper_v6_safe_claims" in written

    # Absolute output_path is also accepted.
    abs_out = tmp_path / "results/alt.json"
    second = exp2896.write_artifact(tmp_path, output_path=abs_out)
    assert second == abs_out
    assert json.loads(abs_out.read_text(encoding="utf-8"))["milestone"] == "2026.05.273"


def test_req_report_2896_kan_invalid_claim_routes_to_forbidden(tmp_path: Path) -> None:
    """REQ-REPORT-2896: KAN hardware/analog claim produces an explicit forbidden line.

    Also exercises the policy_metrics-not-a-dict branch in the FR-11 builder so
    every non-trivial else path is covered.
    """

    overrides = {
        # KAN improperly claims analog hardware execution; status must downgrade.
        "exp2893": {
            "honest_verdict": "complete: KAN claims hardware",
            "kan_complexity_accounting_ready": True,
            "analog_kan_claim_made": True,
            "hardware_execution_claim_made": True,
        },
        # FR-11 policy_metrics has one non-dict per-policy entry so the else
        # branch in token_reduction_pct_by_policy fires.
        "exp2887": {
            "honest_verdict": "complete: fr11 ok",
            "continuous_self_learning_task": True,
            "fr11_scaleup_clean": True,
            "best_policy": "fast_slow_memory",
            "policies_compared": ["eager_replay", "fast_slow_memory"],
            "policy_metrics": {
                "eager_replay": "not-a-dict",
                "fast_slow_memory": {"token_reduction_pct": 50.0},
            },
        },
    }
    _write_all_sources(tmp_path, **overrides)
    _write_prior_capstone(tmp_path)

    artifact = exp2896.build_artifact(tmp_path)

    assert artifact["kan_complexity_status"] == "invalid_hardware_or_analog_claim_made"
    assert any(
        "KAN hardware" in claim for claim in artifact["paper_v6_forbidden_claims"]
    )
    fr11 = artifact["continuous_self_learning_result"]
    assert fr11["token_reduction_pct_by_policy"]["eager_replay"] is None
    assert fr11["token_reduction_pct_by_policy"]["fast_slow_memory"] == pytest.approx(50.0)


def test_req_report_2896_vericot_low_coverage_triggers_action(tmp_path: Path) -> None:
    """REQ-REPORT-2896: VeriCoT <10% coverage triggers an explicit expansion action."""

    overrides = {
        "exp2892": {
            "honest_verdict": "complete: deterministic VeriCoT frontier rows available",
            "vericot_frontier_ready": True,
            "n_candidate_rows": 1100,
            "n_vericot_supported_rows": 25,
            "n_unsupported_rows": 1075,
            "solver_backend": "z3-solver 4.16.0",
            "autoformalization_llm_called": False,
        },
        # Make exp2889 clean so we still have a slot left over in top_3 for the
        # VeriCoT expansion action.
        "exp2889": {
            "honest_verdict": "complete: bounded SOTA GGUF generation produced clean candidate",
            "manifest_contract_ready": True,
            "headline_metric_claim_made": False,
            "row_status": "clean",
        },
        # Make the matrix add a new headline row so the "no new headline" action
        # doesn't claim a slot.
        "exp2894": {
            "honest_verdict": "complete: cross-corpus matrix v7 built",
            "cross_corpus_matrix_built": True,
            "clean_row_count": 6,
            "headline_eligible_rows": ["FoVer", "HaluEval/FEVER", "VeriCoT"],
            "pilot_only_rows": [],
            "taxonomy_only_rows": [],
        },
        # Make exp2885 clean so its action doesn't take the slot either.
        "exp2885": {
            "honest_verdict": "complete: archive_ready=true",
            "archived_milestone": "2026.05.272",
            "activated_milestone": "2026.05.273",
            "archive_already_present": True,
            "paper_ready_from_capstone": True,
        },
    }
    _write_all_sources(tmp_path, **overrides)
    _write_prior_capstone(tmp_path)

    artifact = exp2896.build_artifact(tmp_path)

    actions = artifact["top_3_next_actions"]
    assert any("VeriCoT" in action for action in actions)

"""Tests for Exp 3651 v334 capstone and G-gate synthesis.

Spec: REQ-REPORT-3651, SCENARIO-REPORT-3651.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_and_g_gate_v334_3651 as exp3651


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metric(point: float) -> dict[str, object]:
    return {"point": point, "ci95": [round(point - 0.01, 6), round(point + 0.01, 6)]}


def _gate_data(*, paper_ready: bool = True) -> dict[str, object]:
    return {
        "paper_ready": paper_ready,
        "gates": {
            name: {"pass": paper_ready, "detail": f"{name} detail"}
            for name in ("G1", "G2", "G3", "G4")
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _seed_upstreams(
    root: Path,
    *,
    code_generalizes: bool = True,
    facts_generalize: bool = False,
    grounding_leak_free: bool = True,
    facts_auroc: float = 0.64952,
    exp3643_blocked: bool = False,
    flagged_exp3644: bool = False,
) -> None:
    results = root / "results"
    generalization_table = {
        "math": {
            "domain": "math",
            "ran_or_blocked": "ran",
            "ensemble_auroc": _metric(0.9131),
            "confidence_auroc": _metric(0.895),
            "delta": _metric(0.0181),
            "domain_verdict": "generalizes",
            "n_examples": 1000,
        },
        "code": {
            "domain": "code",
            "ran_or_blocked": "ran",
            "ensemble_auroc": _metric(0.924831 if code_generalizes else 0.52),
            "confidence_auroc": _metric(0.362753),
            "delta": _metric(0.562078 if code_generalizes else 0.01),
            "domain_verdict": "generalizes" if code_generalizes else "domain_bound",
            "n_examples": 320,
        },
        "facts": {
            "domain": "facts",
            "ran_or_blocked": "ran",
            "ensemble_auroc": _metric(facts_auroc),
            "confidence_auroc": _metric(0.744576),
            "delta": _metric(-0.095056 if not facts_generalize else 0.11),
            "domain_verdict": "generalizes" if facts_generalize else "domain_bound",
            "n_examples": 500,
        },
    }
    _write_json(
        results / "experiment_3640_build_factual_corpus_v3.json",
        {
            "honest_verdict": "complete: factual_corpus_v3_built_real_evidence_dataset_confidence_headroom_confirmed_bare_fields_emitted",
            "facts_corpus_validated": True,
            "confidence_baseline_auroc_on_corpus": 0.744576,
        },
    )
    _write_json(
        results / "experiment_3641_code_corpus_verifiers_fire_transfer_v3.json",
        {
            "honest_verdict": "complete: code_corpus_built_verifiers_fire_math_signal_transfers_to_code",
            "code_verifiers_fire": True,
            "code_confidence_baseline_auroc": 0.362753,
        },
    )
    _write_json(
        results / "experiment_3642_corrected_cross_domain_remeasurement_v4.json",
        {
            "honest_verdict": "complete: verifier_value_generalizes_to_code_not_facts_partial_scope",
            "generalization_table": generalization_table,
            "math_ensemble_auroc": 0.9131,
            "code_generalizes": code_generalizes,
            "facts_generalize": facts_generalize,
            "grounding_verifier_auroc": _metric(facts_auroc),
            "grounding_leak_free": grounding_leak_free,
            "positive_control_valid": True,
            "at_least_one_nonmath_row_ran": True,
        },
    )
    _write_json(
        results / "experiment_3643_additivity_second_pair_of_eyes_v4.json",
        {
            "honest_verdict": (
                "complete: blocked_no_nonmath_row_ran"
                if exp3643_blocked
                else "complete: ensemble_additive_to_confidence_second_pair_of_eyes_real_fusion_wins"
            ),
            "second_pair_of_eyes_real": None if exp3643_blocked else True,
            "fused_detector_auroc": None if exp3643_blocked else 0.822394,
            "confidence_alone_auroc": None if exp3643_blocked else 0.536376,
        },
    )
    _write_json(
        results / "experiment_3644_weaver_peer_comparison_v3.json",
        {
            "honest_verdict": "complete: weaver_compared_correlation_matters_carnot_differentiates_on_correlation_awareness",
            "correlation_awareness_matters": True,
            "ensemble_auroc_weaver_style": 0.87158,
            "ensemble_auroc_carnot": 0.919446,
            "flagged_adversarial": flagged_exp3644,
        },
    )
    _write_json(
        results / "experiment_3645_headroom_hybrid_verifier_vs_sc_v3.json",
        {
            "honest_verdict": "complete: verifier_beats_sc_on_headroom_corpus_hybrid_wins_under_budget",
            "oracle_minus_sc_headroom": 0.166667,
            "sc_accuracy": 0.7,
            "verifier_reranked_accuracy": 0.733333,
            "hybrid_accuracy": 0.766667,
            "verifier_beats_sc_where_headroom_exists": True,
        },
    )
    _write_json(
        results / "experiment_3646_trained_ebm_judge_ood_counterpoint_v2.json",
        {
            "honest_verdict": "complete: trained_ebm_judge_also_math_only_transfer_not_a_training_artifact",
            "trained_judge_transfers_ood": False,
            "ood_judge_auroc": 0.673554,
            "confidence_only_baseline_auroc": 0.882162,
            "trained_judge_vs_fixed_ensemble_ood_delta": 0.342554,
        },
    )


def test_req_report_3651_builds_corrected_capstone_from_clean_upstreams(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3651: fair v334 measurements correct the .329 null."""

    _seed_upstreams(tmp_path)
    artifact = exp3651.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[{"exp": exp_id, "returncode": 0} for exp_id in range(3640, 3647)],
        started_s=0.0,
        now_s=4.0,
    )

    exp3651.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: capstone_v334_329_null_was_artifact_"
        "verifier_value_code_only_facts_code_rows_ran_paper_ready_true"
    )
    assert set(exp3651.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3651.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == (
        "aggregation_from_upstream_artifacts "
        "(principle: reads the gate script + artifacts; no live inference)."
    )
    assert artifact["corrected_generalization_table"]["math"]["auroc"] == 0.9131
    assert artifact["corrected_generalization_table"]["code"]["generalizes"] is True
    assert artifact["corrected_generalization_table"]["facts"]["generalizes"] is False
    assert artifact["v329_333_record_corrected_to"].startswith(
        ".329/.330/.331/.332 asserted math-only"
    )
    assert artifact["v329_null_was_artifact_or_confirmed"] == "artifact"
    assert artifact["facts_code_rows_actually_ran"] is True
    assert artifact["grounding_leak_free"] is True
    assert artifact["code_generalizes"] is True
    assert artifact["facts_generalize"] is False
    assert artifact["second_pair_of_eyes_real"] is True
    assert artifact["weaver_differentiation"] is True
    assert artifact["verifier_beats_sc_headroom"] is True
    assert artifact["trained_judge_is_candidate_fix"] is False
    assert artifact["verifier_value_scope"] == "code_only"
    assert artifact["p01_status"] == "honest-negative"
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["duration_s"] == 4.0
    assert all("sha256" in item for item in artifact["cited_upstream_artifacts"])
    assert len(artifact["cited_upstream_artifacts"]) == 7
    assert any("code verifier value generalizes" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any("broad factual" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_report_3651_handles_gate_blocked_additivity_and_grounding_leak(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3651: blocked Exp 3643 and AUROC=1.0 leak fail closed."""

    _seed_upstreams(
        tmp_path,
        code_generalizes=False,
        facts_generalize=True,
        facts_auroc=1.0,
        grounding_leak_free=False,
        exp3643_blocked=True,
    )
    artifact = exp3651.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=1.0,
        now_s=1.5,
    )

    exp3651.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: capstone_v334_329_null_was_confirmed_"
        "verifier_value_math_only_earned_facts_code_rows_ran_paper_ready_true"
    )
    assert artifact["v329_null_was_artifact_or_confirmed"] == "confirmed"
    assert artifact["facts_generalize"] is False
    assert artifact["grounding_leak_free"] is False
    assert artifact["second_pair_of_eyes_real"] == "not_measured"
    assert artifact["verifier_value_scope"] == "math_only_earned"


def test_req_report_3651_excludes_flagged_upstreams_from_citations(tmp_path: Path) -> None:
    """REQ-REPORT-3651: flagged_adversarial artifacts are not cited forward."""

    _seed_upstreams(tmp_path, flagged_exp3644=True)
    artifact = exp3651.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=1.0,
    )

    cited_paths = {item["path"] for item in artifact["cited_upstream_artifacts"]}
    assert "results/experiment_3644_weaver_peer_comparison_v3.json" not in cited_paths
    assert artifact["weaver_differentiation"] is False
    assert any("flagged_adversarial" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_report_3651_write_artifact_and_schema_errors(tmp_path: Path) -> None:
    """REQ-REPORT-3651: the runner helper persists JSON and validation is strict."""

    _seed_upstreams(tmp_path)
    output_path = exp3651.write_artifact(
        tmp_path,
        output_path="results/custom_exp3651.json",
        gate_data=_gate_data(),
        summary_records=[],
        started_s=2.0,
        now_s=3.0,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    exp3651.validate_artifact(payload)
    assert payload["paper_ready"] is True

    missing = dict(payload)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp3651.validate_artifact(missing)

    bad_verdict = dict(payload, honest_verdict="blocked")
    with pytest.raises(ValueError, match="complete:"):
        exp3651.validate_artifact(bad_verdict)

    bad_scope = dict(payload, verifier_value_scope="broadish")
    with pytest.raises(ValueError, match="verifier_value_scope"):
        exp3651.validate_artifact(bad_scope)

    bad_paper = dict(payload, paper_ready=False)
    with pytest.raises(ValueError, match="paper_ready"):
        exp3651.validate_artifact(bad_paper)

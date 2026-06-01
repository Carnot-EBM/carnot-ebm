"""Tests for Exp 3677 v336 capstone and G-gate synthesis.

Spec: REQ-PUBLISH-3677, SCENARIO-PUBLISH-3677.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_and_g_gate_v336_3677 as exp3677


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metric(point: float) -> dict[str, object]:
    return {
        "bootstrap_seeds": [3670, 3671, 3672],
        "ci95": [round(point - 0.01, 6), round(point + 0.01, 6)],
        "point": point,
    }


def _gate_data(*, paper_ready: bool = True) -> dict[str, object]:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {
                "pass": paper_ready,
                "detail": "FoVer 0.9131, 5-seed, CI, adversarial-clean",
                "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
            },
            "G2": {"pass": paper_ready, "detail": "CI runner 26725185125"},
            "G3": {"pass": paper_ready, "detail": "narrowing-clean"},
            "G4": {"pass": paper_ready, "detail": "numbers trace to artifacts"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _seed_upstreams(
    root: Path,
    *,
    exp3667_flagged: bool = False,
    exp3667_clean: bool = True,
    exp3667_beats: bool = True,
    write_exp3668: bool = True,
    heldout_generalizes: bool = True,
    facts_outcome: str = "domain_bound_real",
    facts_auroc: float = 0.650112,
    facts_leak_free: bool = True,
    detector_shipped: bool = True,
    sc_result: str = "no_value_with_headroom",
    fr11_holds: bool = True,
) -> None:
    results = root / "results"
    _write_json(
        results / "experiment_3667_dependency_aware_weighting_clean.json",
        {
            "honest_verdict": (
                "complete: dependency_aware_weighting_beats_carnot_clean_significant_headline_candidate"
            ),
            "flagged_adversarial": exp3667_flagged,
            "adversarial_verify_clean": exp3667_clean,
            "dependency_aware_beats_carnot": exp3667_beats,
            "auroc_dependency_aware_proper": 0.933238,
            "auroc_carnot_current": 0.919446,
            "delong_p_dependency_vs_carnot": 0.000087 if exp3667_beats else 0.31,
            "dependency_aware_vs_carnot_delta_ci": (
                {"point": 0.013792, "ci95": [0.008, 0.019]}
                if exp3667_beats
                else {"point": 0.001, "ci95": [-0.006, 0.008]}
            ),
            "acceptance_gate": {"passed": exp3667_clean and not exp3667_flagged},
            "random_seed": 3667,
            "reproducibility_checksum": "a" * 64,
        },
    )
    if write_exp3668:
        _write_json(
            results / "experiment_3668_dependency_aware_weighting_heldout.json",
            {
                "honest_verdict": (
                    "complete: dependency_aware_weighting_generalizes_heldout_headline_re_freeze_candidate_for_v337"
                ),
                "dependency_aware_generalizes_heldout": heldout_generalizes,
                "heldout_auroc_dependency_aware": 0.933224,
                "heldout_auroc_carnot": 0.919964,
                "heldout_delong_p": 0.000072 if heldout_generalizes else 0.42,
                "heldout_delta_ci": (
                    {"point": 0.01326, "ci95": [0.007, 0.019]}
                    if heldout_generalizes
                    else {"point": 0.001, "ci95": [-0.006, 0.008]}
                ),
                "acceptance_gate": {"passed": True},
                "random_seed": 3668,
                "reproducibility_checksum": "b" * 64,
            },
        )
    _write_json(
        results / "experiment_3669_build_real_factual_corpus.json",
        {
            "honest_verdict": "complete: real_factual_corpus_built_ragtruth_non_degenerate",
            "real_factual_corpus_built": True,
            "corpus_non_degenerate": True,
            "n_examples": 17617,
            "confidence_baseline_auroc": 0.707994,
            "acceptance_gate": {"passed": True},
        },
    )
    _write_json(
        results / "experiment_3670_facts_row_real_benchmark.json",
        {
            "honest_verdict": "complete: facts_real_benchmark_fixture",
            "honest_outcome": facts_outcome,
            "grounding_auroc_real_corpus": _metric(facts_auroc),
            "confidence_baseline_auroc": _metric(0.707994),
            "grounding_minus_confidence_delta": _metric(facts_auroc - 0.707994),
            "facts_generalize_or_adds_value_real": facts_outcome != "domain_bound_real",
            "catch_value_at_parity": facts_outcome == "catch_value_at_parity",
            "grounding_leak_free": facts_leak_free,
            "positive_control_valid": True,
            "real_vs_synthetic_grounding_delta": {
                "synthetic_grounding_auroc": 0.743656,
                "real_grounding_auroc": facts_auroc,
                "delta": round(facts_auroc - 0.743656, 6),
            },
            "acceptance_gate": {"passed": True},
        },
    )
    _write_json(
        results / "experiment_3671_ship_second_pair_of_eyes_detector.json",
        {
            "honest_verdict": "complete: second_pair_of_eyes_detector_fixture",
            "detector_shipped": detector_shipped,
            "e2e_test_passed": detector_shipped,
            "detector_module_path": "python/carnot/pipeline/second_pair_detector.py",
            "wired_surface": "score_candidates MCP tool and carnot score-candidates CLI",
            "acceptance_gate": {"passed": detector_shipped},
        },
    )
    _write_json(
        results / "experiment_3672_ensemble_selection_where_sc_weak.json",
        {
            "honest_verdict": f"complete: sc_weak_fixture_{sc_result}",
            "honest_outcome": (
                "ensemble_adds_selection_value"
                if sc_result == "ensemble_adds_value"
                else ("no_selectable_headroom" if sc_result == "no_headroom" else "no_value_even_with_headroom")
            ),
            "ensemble_adds_selection_value_sc_weak": sc_result == "ensemble_adds_value",
            "positive_control_valid": sc_result != "no_headroom",
            "oracle_minus_sc_headroom": 0.147541 if sc_result != "no_headroom" else 0.0,
            "sc_accuracy": 0.459016,
            "oracle_bestofn_accuracy": 0.606557 if sc_result != "no_headroom" else 0.459016,
            "flip_count": 28 if sc_result != "no_headroom" else 0,
            "acceptance_gate": {
                "required_fields_present": True,
                "positive_control_valid": sc_result != "no_headroom",
            },
        },
    )
    _write_json(
        results / "experiment_3673_fr11_continuous_self_learning_v10.json",
        {
            "honest_verdict": "complete: fr11_v10_fixture",
            "collapse_detected_deploy_arm": not fr11_holds,
            "collapse_detected_control": True,
            "quality_maintained": fr11_holds,
            "pass_rate_vs_true_accuracy_distinct_assert": True,
            "online_dependency_aware_auroc_gain": 0.0018,
            "acceptance_gate": {"passed": fr11_holds},
        },
    )


def test_scenario_publish_3677_builds_clean_capstone(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3677: clean v336 upstreams yield the complete capstone."""

    _seed_upstreams(tmp_path)
    artifact = exp3677.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[{"exp": exp_id, "returncode": 0} for exp_id in range(3667, 3674)],
        started_s=10.0,
        now_s=12.0,
    )

    exp3677.validate_artifact(artifact)
    assert set(exp3677.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3677.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: capstone_v336_dependency_aware_clean_and_heldout_validated_"
        "facts_real_domain_bound_real_earned_detector_shipped_true_paper_ready_true"
    )
    assert artifact["inference_substrate"] == exp3677.INFERENCE_SUBSTRATE
    assert artifact["dependency_aware_headline_candidate_status"] == "clean_and_heldout_validated"
    assert artifact["dependency_aware_candidate"]["exp3667_adversarial_verify_clean"] is True
    assert artifact["dependency_aware_candidate"]["heldout_status"] == "validated"
    assert artifact["facts_real_benchmark_verdict"] == "domain_bound_real_earned"
    assert artifact["facts_real_benchmark"]["grounding_leak_free"] is True
    assert artifact["second_pair_of_eyes_shipped"] is True
    assert artifact["sc_weak_selection_direction_result"] == "no_value_with_headroom"
    assert artifact["fr11_v10_result"] == "held_no_collapse_quality_maintained"
    assert artifact["verifier_value_scope"] == "math_plus_code_sc_weak_no_value_with_headroom"
    assert artifact["frozen_fover_headline_auroc"] == 0.9131
    assert artifact["p01_status"] == "honest-negative"
    assert artifact["trained_judge_ood_retired"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert all("sha256" in item for item in artifact["cited_upstream_artifacts"])
    assert len(artifact["cited_upstream_artifacts"]) == 7
    assert any("headline-advancement candidate pending re-freeze" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any("Do not cite the dependency-aware win as the headline" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_publish_3677_flagged_and_leaky_inputs_fail_closed(tmp_path: Path) -> None:
    """REQ-PUBLISH-3677: flagged and leaky inputs are excluded or not measured."""

    _seed_upstreams(
        tmp_path,
        exp3667_flagged=True,
        exp3667_clean=False,
        facts_auroc=0.999,
        facts_leak_free=False,
        detector_shipped=False,
        sc_result="no_headroom",
        fr11_holds=False,
    )
    artifact = exp3677.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=1.0,
    )

    exp3677.validate_artifact(artifact)
    assert artifact["dependency_aware_headline_candidate_status"] == "flagged_still"
    assert artifact["facts_real_benchmark_verdict"] == "not_measured"
    assert artifact["facts_real_benchmark"]["grounding_leak_risk"] is True
    assert artifact["second_pair_of_eyes_shipped"] is False
    assert artifact["sc_weak_selection_direction_result"] == "no_headroom"
    assert artifact["fr11_v10_result"] == "collapse_or_quality_regression"
    assert artifact["verifier_value_scope"] == "math_only_earned_sc_weak_no_headroom"
    cited_paths = {item["path"] for item in artifact["cited_upstream_artifacts"]}
    assert "results/experiment_3667_dependency_aware_weighting_clean.json" not in cited_paths
    assert any("flagged_adversarial" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_publish_3677_skipped_heldout_is_not_measured(tmp_path: Path) -> None:
    """REQ-PUBLISH-3677: skipped Exp 3668 is recorded as not_measured."""

    _seed_upstreams(tmp_path, exp3667_beats=False, write_exp3668=False)
    artifact = exp3677.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=0.5,
    )

    exp3677.validate_artifact(artifact)
    assert artifact["dependency_aware_headline_candidate_status"] == "no_significant_gain"
    assert artifact["dependency_aware_candidate"]["heldout_status"] == "not_measured"
    assert artifact["dependency_aware_candidate"]["heldout_missing_reason"] == (
        "exp3668 skipped_or_missing_after_exp3667_no_significant_gain"
    )
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v336_dependency_aware_no_significant_gain_"
    )


def test_req_publish_3677_write_artifact_and_validation_guards(tmp_path: Path) -> None:
    """REQ-PUBLISH-3677: writing persists JSON and validation rejects regressions."""

    _seed_upstreams(tmp_path)
    output_path = exp3677.write_artifact(
        tmp_path,
        output_path="results/custom_exp3677.json",
        gate_data=_gate_data(),
        summary_records=[],
        started_s=1.0,
        now_s=3.0,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    exp3677.validate_artifact(payload)
    assert payload["paper_ready"] is True

    validation_cases: list[tuple[dict[str, object], str]] = []
    missing = dict(payload)
    missing.pop("honest_verdict")
    validation_cases.append((missing, "missing required"))
    validation_cases.append((dict(payload, honest_verdict="completeish"), "complete:"))
    validation_cases.append((dict(payload, field_principles=[]), "field_principles"))
    missing_principle = dict(payload)
    missing_principle["field_principles"] = dict(payload["field_principles"])
    missing_principle["field_principles"].pop("facts_real_benchmark_verdict")
    validation_cases.append((missing_principle, "missing field principles"))
    validation_cases.append((dict(payload, inference_substrate="live_gpu"), "inference_substrate"))
    validation_cases.append((dict(payload, dependency_aware_headline_candidate_status="win"), "dependency_aware"))
    validation_cases.append((dict(payload, facts_real_benchmark_verdict="none"), "facts_real"))
    validation_cases.append((dict(payload, sc_weak_selection_direction_result="maybe"), "sc_weak"))
    validation_cases.append((dict(payload, fr11_v10_result="unknown"), "fr11_v10_result"))
    validation_cases.append((dict(payload, verifier_value_scope="facts_only"), "verifier_value_scope"))
    validation_cases.append((dict(payload, g4=False), "g4"))
    validation_cases.append((dict(payload, paper_ready=False), "paper_ready"))
    validation_cases.append((dict(payload, unmet_gates="G2"), "unmet_gates"))
    validation_cases.append((dict(payload, p01_status="positive"), "p01_status"))
    validation_cases.append((dict(payload, trained_judge_ood_retired=False), "trained_judge_ood_retired"))
    validation_cases.append((dict(payload, second_pair_of_eyes_shipped="true"), "second_pair_of_eyes_shipped"))
    validation_cases.append((dict(payload, paper_v6_safe_claims={}), "paper_v6_safe_claims"))
    validation_cases.append((dict(payload, paper_v6_forbidden_claims={}), "paper_v6_forbidden_claims"))
    validation_cases.append((dict(payload, duration_s=-1.0), "duration_s"))
    validation_cases.append((dict(payload, cited_upstream_artifacts={}), "cited_upstream_artifacts"))
    validation_cases.append((dict(payload, cited_upstream_artifacts=[{"path": "x"}]), "sha256"))
    validation_cases.append((dict(payload, reproducibility_checksum="short"), "reproducibility_checksum"))
    for broken, pattern in validation_cases:
        with pytest.raises(ValueError, match=pattern):
            exp3677.validate_artifact(broken)


def test_req_publish_3677_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-PUBLISH-3677: helper edge cases are explicit and deterministic."""

    assert exp3677._point({"point": 0.1234567}) == 0.123457
    assert exp3677._point(0.5) == 0.5
    assert exp3677._point("0.5") is None
    assert exp3677._acceptance_pass({"acceptance_gate": {"passed": True}}) is True
    assert exp3677._acceptance_pass({"acceptance_gate": {"required_fields_present": True}}) is True
    assert exp3677._acceptance_pass({}) is False
    assert exp3677._significant_positive({"point": 0.1, "ci95": [0.01, 0.2]}, 0.04) is True
    assert exp3677._significant_positive({"point": 0.1, "ci95": [-0.01, 0.2]}, 0.04) is False
    assert exp3677._significant_positive(None, 0.04) is False
    measured_facts = {
        "grounding_auroc_real_corpus": 0.7,
        "confidence_baseline_auroc": 0.65,
        "grounding_leak_free": True,
        "positive_control_valid": True,
        "acceptance_gate": {"passed": True},
    }
    assert exp3677._facts_verdict(
        dict(measured_facts, honest_outcome="generalizes_real"), flagged=False
    )["verdict"] == "generalizes_real"
    assert exp3677._facts_verdict(
        dict(measured_facts, honest_outcome="catch_value_at_parity"), flagged=False
    )["verdict"] == "auroc_parity_with_catch_value"
    assert exp3677._sc_weak_result({"ensemble_adds_selection_value_sc_weak": True, "positive_control_valid": True}, flagged=False) == "ensemble_adds_value"
    assert exp3677._sc_weak_result({"positive_control_valid": True}, flagged=False) == "not_measured"
    assert exp3677._sc_weak_result({}, flagged=True) == "not_measured"
    assert exp3677._fr11_result({}, flagged=True) == "not_measured"
    assert exp3677._gate_pass({"gates": []}, "G1") is False
    assert exp3677._facts_verdict(
        dict(measured_facts, honest_outcome="unexpected"), flagged=False
    )["verdict"] == "not_measured"
    assert exp3677._verifier_scope(
        facts_verdict="generalizes_real",
        second_pair_shipped=True,
        sc_weak_result="ensemble_adds_value",
    ) == "math_plus_code_plus_facts_sc_weak_ensemble_adds_value"
    assert exp3677._verifier_scope(
        facts_verdict="auroc_parity_with_catch_value",
        second_pair_shipped=False,
        sc_weak_result="not_measured",
    ) == "math_plus_code_plus_facts_sc_weak_not_measured"
    assert "correct the .335" in exp3677._facts_real_vs_synthetic(
        {"verdict": "generalizes_real"}
    )
    assert "partially correct" in exp3677._facts_real_vs_synthetic(
        {"verdict": "auroc_parity_with_catch_value"}
    )
    assert "not measured" in exp3677._facts_real_vs_synthetic({"verdict": "not_measured"})
    assert any(
        "clean but not held-out" in claim
        for claim in exp3677._safe_claims(
            dependency_status="clean_but_overfit",
            facts_verdict="generalizes_real",
            second_pair_shipped=False,
            sc_weak_result="not_measured",
            fr11_result="not_measured",
        )
    )
    assert any(
        "does not provide" in claim
        for claim in exp3677._safe_claims(
            dependency_status="no_significant_gain",
            facts_verdict="auroc_parity_with_catch_value",
            second_pair_shipped=False,
            sc_weak_result="not_measured",
            fr11_result="not_measured",
        )
    )
    assert any(
        "flagged or unclean" in claim
        for claim in exp3677._safe_claims(
            dependency_status="flagged_still",
            facts_verdict="not_measured",
            second_pair_shipped=False,
            sc_weak_result="not_measured",
            fr11_result="not_measured",
        )
    )
    significant_exp3667 = {
        "adversarial_verify_clean": True,
        "dependency_aware_beats_carnot": True,
        "dependency_aware_vs_carnot_delta_ci": {"point": 0.02, "ci95": [0.01, 0.03]},
        "delong_p_dependency_vs_carnot": 0.01,
        "acceptance_gate": {"passed": True},
    }
    assert exp3677._dependency_candidate(
        significant_exp3667,
        {"acceptance_gate": {"passed": True}},
        exp3667_flagged=False,
        exp3668_flagged=True,
    )["heldout_status"] == "excluded_flagged_adversarial"
    assert exp3677._dependency_candidate(
        significant_exp3667,
        {},
        exp3667_flagged=False,
        exp3668_flagged=False,
    )["heldout_missing_reason"] == "exp3668 skipped_or_missing"
    assert exp3677._dependency_candidate(
        significant_exp3667,
        {
            "dependency_aware_generalizes_heldout": False,
            "heldout_delta_ci": {"point": 0.001, "ci95": [-0.01, 0.01]},
            "heldout_delong_p": 0.5,
            "acceptance_gate": {"passed": True},
        },
        exp3667_flagged=False,
        exp3668_flagged=False,
    )["status"] == "clean_but_overfit"
    assert exp3677._cited_upstreams(
        tmp_path,
        {"exp3667": {"honest_verdict": "complete: x", "adversarial_verify_clean": False}},
        {"exp3667": False},
    ) == []
    assert exp3677._read_optional_json_object(tmp_path / "missing.json") == {}
    list_path = tmp_path / "not_object.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp3677._read_optional_json_object(list_path)

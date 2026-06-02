"""Tests for Exp 3689 v337 capstone and G-gate synthesis.

Spec: REQ-PUBLISH-041, SCENARIO-PUBLISH-041.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_and_g_gate_v337_3689 as exp3689


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metric(point: float) -> dict[str, object]:
    return {
        "bootstrap_seeds": [3689, 3690, 3691],
        "ci95": [round(point - 0.01, 6), round(point + 0.01, 6)],
        "point": point,
    }


def _gate_data(*, paper_ready: bool = True, headline_source: str | None = None) -> dict[str, object]:
    source = headline_source or "experiment_2850_fover_dual_condition_integrity_v4.json"
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {
                "pass": paper_ready,
                "detail": "FoVer 0.9131, 5-seed, CI, adversarial-clean",
                "source": source,
            },
            "G2": {"pass": paper_ready, "detail": "CI runner 26725185125"},
            "G3": {"pass": paper_ready, "detail": "narrowing-clean"},
            "G4": {
                "pass": paper_ready,
                "detail": "numbers trace to experiment_2850_fover_dual_condition_integrity_v4",
                "source": source,
            },
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _seed_upstreams(
    root: Path,
    *,
    exp3680_flagged: bool = False,
    exp3680_clean: bool = True,
    exp3680_confirmed: bool = True,
    exp3680_auroc: float = 0.925328,
    exp3680_leak_free: bool = True,
    write_exp3681: bool = True,
    exp3681_flagged: bool = False,
    exp3681_ready: bool = True,
    write_exp3682: bool = True,
    exp3682_flagged: bool = False,
    selection_gap_closed: bool = False,
    detector_recovered: bool = False,
    product_robust: bool = True,
    fr11_recovers: bool = True,
) -> None:
    results = root / "results"
    _write_json(
        results / "experiment_3680_dependency_aware_dual_condition_integrity.json",
        {
            "honest_verdict": (
                "complete: dependency_aware_g1_rigor_confirmed_headline_candidate_exceeds_frozen_0_9131"
                if exp3680_confirmed
                else "complete: no_significant_gain_under_dual_condition_protocol"
            ),
            "flagged_adversarial": exp3680_flagged,
            "adversarial_verify_clean": exp3680_clean,
            "dependency_aware_g1_rigor_confirmed": exp3680_confirmed,
            "leak_free": exp3680_leak_free,
            "n_seeds": 5,
            "production_auroc_dependency_aware": exp3680_auroc,
            "production_auroc_carnot_current": 0.913134,
            "frozen_headline_auroc": 0.9131,
            "production_auroc_ci": _metric(exp3680_auroc),
            "learning_contribution_dependency_aware": 0.022149,
            "dependency_vs_carnot_delta_ci": _metric(exp3680_auroc - 0.913134),
            "delong_p_dependency_vs_carnot": 0.00001 if exp3680_confirmed else 0.42,
            "acceptance_gate": {"passed": exp3680_clean and not exp3680_flagged},
            "random_seed": 3680,
            "reproducibility_checksum": "0" * 64,
            "duration_s": 21.0,
        },
    )
    if write_exp3681:
        _write_json(
            results / "experiment_3681_g2_reproducer_prep_operator_refreeze_package.json",
            {
                "honest_verdict": (
                    "complete: refreeze_package_ready_for_operator_frozen_headline_unchanged"
                    if exp3681_ready
                    else "complete: blocked_g1_candidate_not_confirmed_or_reproducer_unavailable"
                ),
                "flagged_adversarial": exp3681_flagged,
                "reproducer_extended": exp3681_ready,
                "existing_0_9131_reproduction_still_green": exp3681_ready,
                "candidate_reproduction_asserts_in_ci": exp3681_ready,
                "north_star_unmodified_assert": True,
                "ci_workflow_unmodified_assert": True,
                "frozen_headline_unchanged_assert": True,
                "github_actions_run_triggered": False,
                "publication_gate_paper_ready_before": True,
                "publication_gate_paper_ready_after": True,
                "exp3680_dependency_aware_g1_rigor_confirmed": exp3680_confirmed,
                "random_seed": 3681,
                "reproducibility_checksum": "1" * 64,
                "duration_s": 25.0,
            },
        )
    if write_exp3682:
        _write_json(
            results / "experiment_3682_discrimination_vs_selection_gap.json",
            {
                "honest_verdict": (
                    "complete: selection_gap_closed_by_per_question_calibration"
                    if selection_gap_closed
                    else "complete: selection_gap_fundamental_no_fix_beats_sc_discrimination_decoupled"
                ),
                "flagged_adversarial": exp3682_flagged,
                "honest_outcome": (
                    "closed_by_per_question_calibration"
                    if selection_gap_closed
                    else "decoupling_fundamental_no_fix_helps"
                ),
                "selection_gap_closed": selection_gap_closed,
                "best_fix_method": "per_question_normalized",
                "per_candidate_auroc": 0.5555,
                "within_question_rank_corr": 0.1408,
                "positive_control_valid": True,
                "flip_count": 28,
                "acceptance_gate": {"required_fields_present": True},
                "random_seed": 3682,
                "reproducibility_checksum": "2" * 64,
                "duration_s": 0.8,
            },
        )
    _write_json(
        results / "experiment_3683_detector_code_operating_point.json",
        {
            "honest_verdict": (
                "complete: recovered_math_and_code_detector_operating_point"
                if detector_recovered
                else "complete: code_remains_math_only_detector_scoped_honestly"
            ),
            "code_operating_point_recovered": detector_recovered,
            "code_auroc_recalibrated": _metric(0.75 if detector_recovered else 0.506173),
            "code_auroc_dependency_aware": _metric(0.72 if detector_recovered else 0.463333),
            "e2e_test_passed": True,
            "acceptance_gate": {"passed": True},
            "random_seed": 3683,
            "reproducibility_checksum": "3" * 64,
            "duration_s": 3.6,
        },
    )
    _write_json(
        results / "experiment_3684_product_value_vs_self_certainty.json",
        {
            "honest_verdict": (
                "complete: ensemble_adds_value_over_self_certainty_product_value_robust"
                if product_robust
                else "complete: product_value_collapses_vs_self_certainty"
            ),
            "ensemble_adds_value_over_self_certainty": product_robust,
            "material_win_per_domain": {"math": product_robust, "code": False},
            "ensemble_minus_self_certainty_delta_ci_per_domain": {
                "math": {"point": 0.47 if product_robust else -0.01, "ci95": [0.46, 0.48] if product_robust else [-0.03, 0.01]}
            },
            "acceptance_gate": {"passed": True},
            "random_seed": 3684,
            "reproducibility_checksum": "4" * 64,
            "duration_s": 7.5,
        },
    )
    _write_json(
        results / "experiment_3685_fr11_continuous_self_learning_v11.json",
        {
            "honest_verdict": (
                "complete: fr11_v11_drift_aware_online_dependency_aware_recovers_no_collapse_quality_maintained"
                if fr11_recovers
                else "complete: fr11_v11_collapse_or_quality_regression"
            ),
            "drift_detected_deploy_arm": fr11_recovers,
            "collapse_detected_deploy_arm": not fr11_recovers,
            "collapse_detected_control": True,
            "quality_maintained": fr11_recovers,
            "pass_rate_vs_true_accuracy_distinct_assert": True,
            "post_drift_auroc_gain_over_static_carnot": 0.014668 if fr11_recovers else -0.02,
            "post_drift_auroc_gain_over_v10": 0.088142 if fr11_recovers else -0.04,
            "acceptance_gate": {"passed": fr11_recovers},
            "random_seed": 3685,
            "reproducibility_checksum": "5" * 64,
            "duration_s": 0.7,
        },
    )


def test_scenario_publish_041_builds_clean_v337_capstone(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-041: clean upstreams preserve G1-G4 and candidate scope."""

    spec = Path("openspec/capabilities/publication/spec.md").read_text(encoding="utf-8")
    assert "REQ-PUBLISH-041" in spec
    assert "SCENARIO-PUBLISH-041" in spec
    _seed_upstreams(tmp_path)
    artifact = exp3689.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[{"exp": exp_id, "returncode": 0} for exp_id in range(3680, 3686)],
        started_s=10.0,
        now_s=12.0,
    )

    exp3689.validate_artifact(artifact)
    assert set(exp3689.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3689.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: capstone_v337_dependency_aware_g1_rigor_confirmed_package_ready_"
        "selection_fundamental_decoupling_detector_code_math_only_earned_"
        "paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == exp3689.INFERENCE_SUBSTRATE
    assert artifact["dependency_aware_g1_candidate_status"] == "g1_rigor_confirmed_package_ready"
    assert artifact["refreeze_package_status"] == "ready_for_operator"
    assert artifact["selection_gap_verdict"] == "fundamental_decoupling"
    assert artifact["detector_code_operating_point"] == "math_only_earned"
    assert artifact["product_value_vs_self_certainty"] == "robust_beats_self_certainty"
    assert artifact["fr11_v11_result"] == (
        "drift_aware_online_dependency_aware_recovers_no_collapse_quality_maintained"
    )
    assert artifact["verifier_value_scope"] == (
        "math_plus_code_discrimination_facts_retired_selection_earned_negative"
    )
    assert artifact["g1"] is True
    assert artifact["g2"] is True
    assert artifact["g3"] is True
    assert artifact["g4"] is True
    assert artifact["paper_ready"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["p01_status"] == "honest-negative"
    assert artifact["facts_generalization_retired"] is True
    assert artifact["trained_judge_ood_retired"] is True
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert all("sha256" in item for item in artifact["cited_upstream_artifacts"])
    assert len(artifact["cited_upstream_artifacts"]) == 6
    assert any("headline-advancement candidate" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any("Do not cite the dependency-aware win as the headline" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_publish_041_flagged_refreeze_and_selection_fail_closed(tmp_path: Path) -> None:
    """REQ-PUBLISH-041: flagged Exp 3681/3682 are excluded and not measured."""

    _seed_upstreams(tmp_path, exp3681_flagged=True, exp3682_flagged=True)
    artifact = exp3689.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=1.0,
    )

    exp3689.validate_artifact(artifact)
    assert artifact["dependency_aware_g1_candidate_status"] == "g1_rigor_confirmed_package_blocked"
    assert artifact["refreeze_package_status"] == "not_prepared_candidate_unconfirmed"
    assert artifact["selection_gap_verdict"] == "not_measured"
    assert artifact["verifier_value_scope"] == (
        "math_plus_code_discrimination_facts_retired_selection_not_measured"
    )
    cited_paths = {item["path"] for item in artifact["cited_upstream_artifacts"]}
    assert "results/experiment_3681_g2_reproducer_prep_operator_refreeze_package.json" not in cited_paths
    assert "results/experiment_3682_discrimination_vs_selection_gap.json" not in cited_paths
    assert "results/experiment_3680_dependency_aware_dual_condition_integrity.json" in cited_paths
    assert artifact["flagged_upstream_artifacts_excluded"] == [
        "results/experiment_3681_g2_reproducer_prep_operator_refreeze_package.json",
        "results/experiment_3682_discrimination_vs_selection_gap.json",
    ]
    assert any("flagged_adversarial artifacts" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_publish_041_skipped_and_leaky_inputs_are_not_synthesized(tmp_path: Path) -> None:
    """REQ-PUBLISH-041: skipped fields and AUROC leaks fail closed."""

    _seed_upstreams(
        tmp_path,
        exp3680_confirmed=True,
        exp3680_auroc=0.999,
        exp3680_leak_free=False,
        write_exp3681=False,
        write_exp3682=False,
        detector_recovered=True,
        product_robust=False,
        fr11_recovers=False,
    )
    artifact = exp3689.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=1.0,
        now_s=1.5,
    )

    exp3689.validate_artifact(artifact)
    assert artifact["dependency_aware_g1_candidate_status"] == "flagged_still"
    assert artifact["dependency_aware_candidate"]["leak_risk"] is True
    assert artifact["refreeze_package_status"] == "not_prepared_candidate_unconfirmed"
    assert artifact["selection_gap_verdict"] == "not_measured"
    assert artifact["detector_code_operating_point"] == "recovered_math_and_code"
    assert artifact["product_value_vs_self_certainty"] == "narrowed_collapses_vs_self_certainty"
    assert artifact["fr11_v11_result"] == "collapse_or_quality_regression"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v337_dependency_aware_flagged_still_selection_not_measured_"
    )


def test_req_publish_041_no_significant_gain_keeps_refreeze_blocked(tmp_path: Path) -> None:
    """REQ-PUBLISH-041: unconfirmed Exp 3680 blocks package-ready status."""

    _seed_upstreams(tmp_path, exp3680_confirmed=False, exp3681_ready=True, selection_gap_closed=True)
    artifact = exp3689.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=0.25,
    )

    exp3689.validate_artifact(artifact)
    assert artifact["dependency_aware_g1_candidate_status"] == "no_significant_gain_under_protocol"
    assert artifact["refreeze_package_status"] == "not_prepared_candidate_unconfirmed"
    assert artifact["selection_gap_verdict"] == "closed_by_per_question_calibration"
    assert artifact["verifier_value_scope"] == (
        "math_plus_code_discrimination_facts_retired_selection_closed"
    )


def test_req_publish_041_write_artifact_and_validation_guards(tmp_path: Path) -> None:
    """REQ-PUBLISH-041: writing persists JSON and validation rejects regressions."""

    _seed_upstreams(tmp_path)
    output_path = exp3689.write_artifact(
        tmp_path,
        output_path="results/custom_exp3689.json",
        gate_data=_gate_data(),
        summary_records=[],
        started_s=2.0,
        now_s=3.0,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    exp3689.validate_artifact(payload)
    assert payload["paper_ready"] is True

    validation_cases: list[tuple[dict[str, object], str]] = []
    missing = dict(payload)
    missing.pop("honest_verdict")
    validation_cases.append((missing, "missing required"))
    validation_cases.append((dict(payload, honest_verdict="completeish"), "complete:"))
    validation_cases.append((dict(payload, field_principles=[]), "field_principles"))
    missing_principle = dict(payload)
    missing_principle["field_principles"] = dict(payload["field_principles"])
    missing_principle["field_principles"].pop("selection_gap_verdict")
    validation_cases.append((missing_principle, "missing field principles"))
    validation_cases.append((dict(payload, inference_substrate="live_gpu"), "inference_substrate"))
    validation_cases.append((dict(payload, dependency_aware_g1_candidate_status="ready"), "dependency_aware"))
    validation_cases.append((dict(payload, refreeze_package_status="ready"), "refreeze_package_status"))
    validation_cases.append((dict(payload, selection_gap_verdict="maybe"), "selection_gap_verdict"))
    validation_cases.append((dict(payload, detector_code_operating_point="code_only"), "detector_code"))
    validation_cases.append((dict(payload, product_value_vs_self_certainty="maybe"), "product_value"))
    validation_cases.append((dict(payload, fr11_v11_result="maybe"), "fr11_v11_result"))
    validation_cases.append((dict(payload, verifier_value_scope="facts_only"), "verifier_value_scope"))
    validation_cases.append((dict(payload, g1=False), "g1"))
    validation_cases.append((dict(payload, paper_ready=False), "paper_ready"))
    validation_cases.append((dict(payload, frozen_headline_unchanged=False), "frozen_headline_unchanged"))
    validation_cases.append((dict(payload, unmet_gates="G2"), "unmet_gates"))
    validation_cases.append((dict(payload, p01_status="positive"), "p01_status"))
    validation_cases.append((dict(payload, facts_generalization_retired=False), "facts_generalization_retired"))
    validation_cases.append((dict(payload, trained_judge_ood_retired=False), "trained_judge_ood_retired"))
    validation_cases.append((dict(payload, paper_v6_safe_claims={}), "paper_v6_safe_claims"))
    validation_cases.append((dict(payload, paper_v6_forbidden_claims={}), "paper_v6_forbidden_claims"))
    validation_cases.append((dict(payload, duration_s=-1.0), "duration_s"))
    validation_cases.append((dict(payload, cited_upstream_artifacts={}), "cited_upstream_artifacts"))
    validation_cases.append((dict(payload, cited_upstream_artifacts=[{"path": "x"}]), "sha256"))
    validation_cases.append((dict(payload, reproducibility_checksum="short"), "reproducibility_checksum"))
    for broken, pattern in validation_cases:
        with pytest.raises(ValueError, match=pattern):
            exp3689.validate_artifact(broken)


def test_req_publish_041_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-PUBLISH-041: helper edge cases are explicit and deterministic."""

    assert exp3689._point({"point": 0.1234567}) == 0.123457
    assert exp3689._point(0.5) == 0.5
    assert exp3689._point("0.5") is None
    assert exp3689._acceptance_pass({"acceptance_gate": {"passed": True}}) is True
    assert exp3689._acceptance_pass({"acceptance_gate": {"required_fields_present": True}}) is True
    assert exp3689._acceptance_pass({}) is False
    assert exp3689._gate_pass({"gates": {"G1": {"pass": True}}}, "G1") is True
    assert exp3689._gate_pass({"gates": {"G1": {"pass": False}}}, "G1") is False
    assert exp3689._gate_pass({}, "G1") is False
    assert exp3689._frozen_headline_unchanged({}) is False
    assert exp3689._seed_count(5) == 5
    assert exp3689._seed_count([1, 2, 3]) == 3
    assert exp3689._seed_count("123") == 0
    assert exp3689._detector_code_status({}, flagged=True)["status"] == "not_measured"
    assert exp3689._product_value_status({}, flagged=True)["status"] == "not_measured"
    assert exp3689._fr11_v11_status({}, flagged=True)["result"] == "not_measured"
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp3689._read_optional_json_object(non_object)
    assert exp3689._repo_path(Path("/tmp/root"), Path("results/x.json")) == Path("/tmp/root/results/x.json")
    assert exp3689._repo_path(Path("/tmp/root"), Path("/abs/x.json")) == Path("/abs/x.json")

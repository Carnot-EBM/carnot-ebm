"""Tests for Exp 3701 v338 re-freeze capstone and G-gate recheck.

Spec: REQ-PUBLISH-3701, SCENARIO-PUBLISH-3701.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_and_g_gate_v338_3701 as exp3701


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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
    exp3692_flagged: bool = False,
    exp3692_ready: bool = True,
    exp3693_beats: bool = False,
    exp3693_auroc: float = 0.924869,
    exp3693_external_auroc: float = 0.928737,
    exp3693_n_examples: int = 1000,
    exp3693_leak_free: bool | None = None,
    exp3694_measured: bool = True,
    exp3694_closed: bool = False,
    exp3695_recovered: bool = True,
    write_exp3696: bool = True,
    exp3696_ready: bool = True,
    exp3697_success: bool = True,
) -> None:
    results = root / "results"
    _write_json(
        results / "experiment_3692_refreeze_package_clean_reemit.json",
        {
            "honest_verdict": (
                exp3701.REFREEZE_READY_VERDICT
                if exp3692_ready
                else "complete: blocked_g1_candidate_not_confirmed_or_reproducer_unavailable"
            ),
            "flagged_adversarial": exp3692_flagged,
            "adversarial_verify_clean": not exp3692_flagged,
            "reproducer_extended": exp3692_ready,
            "existing_0_9131_reproduction_still_green": exp3692_ready,
            "candidate_reproduction_asserts_in_ci": exp3692_ready,
            "north_star_unmodified_assert": True,
            "ci_workflow_unmodified_assert": True,
            "frozen_headline_unchanged_assert": True,
            "github_actions_run_triggered": False,
            "publication_gate_paper_ready_before": True,
            "publication_gate_paper_ready_after": True,
            "random_seed": 3692,
            "reproducibility_checksum": "2" * 64,
            "duration_s": 24.0,
        },
    )
    _write_json(
        results / "experiment_3693_external_comparator_dependency_vs_deentangled.json",
        {
            "honest_verdict": (
                "complete: dependency_aware_candidate_beats_external_baseline_refreeze_candidate"
                if exp3693_beats
                else "complete: dependency_aware_candidate_ties_or_loses_external_baseline_refreeze_narrowed"
            ),
            "adversarial_verify_clean": True,
            "acceptance_gate": {"passed": True},
            "candidate_beats_external_comparator": exp3693_beats,
            "dependency_aware_auroc": exp3693_auroc,
            "external_comparator_auroc": exp3693_external_auroc,
            "dependency_vs_external_delta_ci": {
                "point": 0.012 if exp3693_beats else -0.004,
                "ci95": [0.006, 0.019] if exp3693_beats else [-0.007, -0.001],
            },
            "n_examples": exp3693_n_examples,
            "leak_free": exp3693_leak_free,
            "random_seed": 3693,
            "reproducibility_checksum": "3" * 64,
            "duration_s": 31.0,
        },
    )
    _write_json(
        results / "experiment_3694_selection_gap_proper_rediagnosis.json",
        {
            "honest_verdict": (
                "complete: selection_gap_closed_new_method"
                if exp3694_closed
                else "complete: selection_gap_fundamental_decoupling"
            ),
            "adversarial_verify_clean": True,
            "acceptance_gate": {"passed": exp3694_measured},
            "positive_control_valid": exp3694_measured,
            "non_degeneracy_assert": exp3694_measured,
            "per_candidate_auroc": 0.91 if exp3694_measured else None,
            "within_question_rank_corr": 0.33 if exp3694_measured else None,
            "selection_gap_closed": exp3694_closed,
            "random_seed": 3694,
            "reproducibility_checksum": "4" * 64,
            "duration_s": 2.0,
        },
    )
    _write_json(
        results / "experiment_3695_code_native_verifier.json",
        {
            "honest_verdict": (
                "complete: code_native_signal_recovered_beats_chance_floor"
                if exp3695_recovered
                else "complete: code_remains_math_only_detector_scoped_honestly"
            ),
            "adversarial_verify_clean": True,
            "acceptance_gate": {"passed": True},
            "code_signal_recovered": exp3695_recovered,
            "code_native_auroc": 0.82 if exp3695_recovered else 0.51,
            "n_examples_code": 60,
            "random_seed": 3695,
            "reproducibility_checksum": "5" * 64,
            "duration_s": 2.0,
        },
    )
    if write_exp3696:
        _write_json(
            results / "experiment_3696_reship_detector_math_plus_code.json",
            {
                "honest_verdict": (
                    "complete: detector_reshipped_math_plus_code_operating_point_e2e_green"
                    if exp3696_ready
                    else "complete: detector_reship_blocked"
                ),
                "adversarial_verify_clean": True,
                "acceptance_gate": {"passed": exp3696_ready},
                "module_code_path_updated": exp3696_ready,
                "math_operating_point_unchanged": exp3696_ready,
                "e2e_test_passed": exp3696_ready,
                "code_operating_point_auroc": 0.83,
                "random_seed": 3696,
                "reproducibility_checksum": "6" * 64,
                "duration_s": 5.0,
            },
        )
    _write_json(
        results / "experiment_3697_fr11_continuous_self_learning_v12.json",
        {
            "honest_verdict": (
                "complete: fr11_v12_drift_reset_and_cross_session_persistence_no_collapse_quality_maintained"
                if exp3697_success
                else "complete: fr11_v12_collapse_or_quality_regression"
            ),
            "adversarial_verify": "clean",
            "acceptance_gate": {"passed": exp3697_success},
            "drift_detected_deploy_arm": exp3697_success,
            "reset_triggered_on_transient_drift": exp3697_success,
            "structure_persisted_and_restored": exp3697_success,
            "collapse_detected_deploy_arm": not exp3697_success,
            "quality_maintained": exp3697_success,
            "pass_rate_vs_true_accuracy_distinct_assert": True,
            "random_seed": 3697,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 3.0,
        },
    )


def test_scenario_publish_3701_builds_clean_v338_capstone(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3701: clean upstreams preserve G1-G4 and scoped claims."""

    spec = Path("openspec/capabilities/publication/spec.md").read_text(encoding="utf-8")
    assert "REQ-PUBLISH-3701" in spec
    assert "SCENARIO-PUBLISH-3701" in spec
    _seed_upstreams(tmp_path)

    artifact = exp3701.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[{"exp": exp_id, "returncode": 0} for exp_id in range(3692, 3698)],
        started_s=10.0,
        now_s=12.5,
    )

    exp3701.validate_artifact(artifact)
    assert set(exp3701.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3701.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: capstone_v338_refreeze_reemitted_clean_for_operator_"
        "external_ties_or_loses_selection_fundamental_decoupling_"
        "detector_code_code_native_recovered_reshipped_"
        "paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["refreeze_package_status"] == "reemitted_clean_for_operator"
    assert artifact["candidate_beats_external_comparator"] == "ties_or_loses"
    assert artifact["selection_gap_verdict"] == "fundamental_decoupling"
    assert artifact["code_detector_status"] == "code_native_recovered_reshipped"
    assert artifact["fr11_v12_result"] == (
        "drift_reset_and_cross_session_persistence_no_collapse_quality_maintained"
    )
    assert artifact["verifier_value_scope"] == (
        "math_plus_code_discrimination_facts_retired_selection_fundamental_decoupling"
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
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert len(artifact["cited_upstream_artifacts"]) == 6
    assert all(item["adversarial_verify"] == "clean" for item in artifact["cited_upstream_artifacts"])
    assert any("headline-advancement candidate" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any("Do not cite the dependency-aware win as the headline" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_publish_3701_flagged_skipped_and_leaky_inputs_fail_closed(tmp_path: Path) -> None:
    """REQ-PUBLISH-3701: flagged, skipped, and leaky upstreams are not synthesized."""

    _seed_upstreams(
        tmp_path,
        exp3692_flagged=True,
        exp3693_auroc=0.995,
        exp3693_external_auroc=0.991,
        exp3693_n_examples=1000,
        exp3694_measured=False,
        exp3695_recovered=False,
        write_exp3696=False,
        exp3697_success=False,
    )
    artifact = exp3701.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=1.0,
        now_s=1.25,
    )

    exp3701.validate_artifact(artifact)
    assert artifact["refreeze_package_status"] == "still_flagged"
    assert artifact["candidate_beats_external_comparator"] == "not_measured"
    assert artifact["external_comparator"]["leak_risk"] is True
    assert artifact["selection_gap_verdict"] == "not_measured"
    assert artifact["code_detector_status"] == "code_remains_math_only_earned"
    assert artifact["code_detector"]["exp3696_reship_status"] == "not_measured"
    assert artifact["fr11_v12_result"] == "collapse_or_quality_regression"
    assert artifact["verifier_value_scope"] == (
        "math_only_discrimination_facts_retired_selection_not_measured"
    )
    cited_paths = {item["path"] for item in artifact["cited_upstream_artifacts"]}
    assert "results/experiment_3692_refreeze_package_clean_reemit.json" not in cited_paths
    assert "results/experiment_3693_external_comparator_dependency_vs_deentangled.json" not in cited_paths
    assert artifact["flagged_upstream_artifacts_excluded"] == [
        "results/experiment_3692_refreeze_package_clean_reemit.json",
        "results/experiment_3693_external_comparator_dependency_vs_deentangled.json",
    ]
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v338_refreeze_still_flagged_external_not_measured_"
    )


def test_req_publish_3701_external_win_and_selection_close(tmp_path: Path) -> None:
    """REQ-PUBLISH-3701: positive statuses require clean measured gates."""

    _seed_upstreams(tmp_path, exp3693_beats=True, exp3694_closed=True)
    artifact = exp3701.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=0.75,
    )

    exp3701.validate_artifact(artifact)
    assert artifact["candidate_beats_external_comparator"] == "yes"
    assert artifact["selection_gap_verdict"] == "closed_new_method"
    assert artifact["verifier_value_scope"] == (
        "math_plus_code_discrimination_facts_retired_selection_closed"
    )


def test_req_publish_3701_missing_upstreams_are_not_measured(tmp_path: Path) -> None:
    """REQ-PUBLISH-3701: absent gated tasks fail closed without fabricated None reads."""

    artifact = exp3701.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=0.5,
    )

    exp3701.validate_artifact(artifact)
    assert artifact["refreeze_package_status"] == "not_prepared"
    assert artifact["candidate_beats_external_comparator"] == "not_measured"
    assert artifact["selection_gap_verdict"] == "not_measured"
    assert artifact["code_detector_status"] == "not_measured"
    assert artifact["fr11_v12_result"] == "not_measured"
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["source_artifacts"] == []


def test_req_publish_3701_write_artifact_and_validation_guards(tmp_path: Path) -> None:
    """REQ-PUBLISH-3701: writing persists JSON and validation rejects regressions."""

    _seed_upstreams(tmp_path)
    output_path = exp3701.write_artifact(
        tmp_path,
        output_path="results/custom_exp3701.json",
        gate_data=_gate_data(),
        summary_records=[],
        started_s=2.0,
        now_s=3.0,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    exp3701.validate_artifact(payload)
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["flags"] == []

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
    validation_cases.append((dict(payload, inference_substrate="verifier_ensemble_against_cached_candidates"), "inference_substrate"))
    validation_cases.append((dict(payload, adversarial_verify_clean=False), "adversarial_verify_clean"))
    validation_cases.append((dict(payload, refreeze_package_status="ready"), "refreeze_package_status"))
    validation_cases.append((dict(payload, candidate_beats_external_comparator=True), "candidate_beats_external"))
    validation_cases.append((dict(payload, selection_gap_verdict="maybe"), "selection_gap_verdict"))
    validation_cases.append((dict(payload, code_detector_status="code_only"), "code_detector_status"))
    validation_cases.append((dict(payload, fr11_v12_result="maybe"), "fr11_v12_result"))
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
    validation_cases.append((dict(payload, model_specs={"x": "y"}), "model_specs"))
    validation_cases.append((dict(payload, target_model="x"), "target_model"))
    for broken, pattern in validation_cases:
        with pytest.raises(ValueError, match=pattern):
            exp3701.validate_artifact(broken)


def test_req_publish_3701_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-PUBLISH-3701: helper edge cases are explicit and deterministic."""

    assert exp3701._point({"point": 0.1234567}) == 0.123457
    assert exp3701._point({"ci95": [0.1, 0.2]}) is None
    assert exp3701._point(0.5) == 0.5
    assert exp3701._point("0.5") is None
    assert exp3701._acceptance_pass({"acceptance_gate": {"passed": True}}) is True
    assert exp3701._acceptance_pass({"acceptance_gate": {"required_fields_present": True}}) is True
    assert exp3701._acceptance_pass({"acceptance_gate": False}) is False
    assert exp3701._gate_pass({"gates": {"G1": {"pass": True}}}, "G1") is True
    assert exp3701._gate_pass({"gates": {"G1": {"pass": False}}}, "G1") is False
    assert exp3701._gate_pass({}, "G1") is False
    assert exp3701._frozen_headline_unchanged({}) is False
    assert exp3701._payload_declares_adversarial_clean({"adversarial_verify_clean": True}) is True
    assert exp3701._payload_declares_adversarial_clean({"adversarial_verify": "clean"}) is True
    assert exp3701._payload_declares_adversarial_clean({"adversarial_verify_report": {"flags": []}}) is True
    assert exp3701._payload_declares_adversarial_clean({}) is False
    assert exp3701.adversarial_report_is_clean({"flags": []}) is True
    assert exp3701.adversarial_report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp3701.adversarial_report_is_clean({"flags": [{"kind": "DURATION_TOO_SHORT", "severity": "warn"}]}) is False
    assert exp3701._leak_risk({"n_examples": 1000, "some_auroc": 0.991}) is True
    assert exp3701._leak_risk({"n_examples": 1000, "some_auroc_metric": {"point": 0.991}}) is True
    assert exp3701._leak_risk({"n_examples": 1000, "some_auroc": 0.991, "leak_free": True}) is False
    assert exp3701._leak_risk({"n_examples": 60, "some_auroc": 1.0}) is False
    assert exp3701._report_has_critical({"flags": [{"severity": "critical"}]}) is True
    assert exp3701._report_has_critical({"flags": [{"severity": "warn"}]}) is False
    assert exp3701._repo_path(Path("/tmp/root"), Path("results/x.json")) == Path("/tmp/root/results/x.json")
    assert exp3701._repo_path(Path("/tmp/root"), Path("/abs/x.json")) == Path("/abs/x.json")
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp3701._read_optional_json_object(non_object)

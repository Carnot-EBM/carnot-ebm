"""Tests for Exp 3723 v340 convergence capstone and hardened G-gate recheck.

Spec: REQ-PUBLISH-3723, SCENARIO-PUBLISH-3723.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_and_g_gate_v340_3723 as exp3723


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _gate_data(*, paper_ready: bool = True, source: str | None = None) -> dict[str, object]:
    headline_source = source or "experiment_2850_fover_dual_condition_integrity_v4.json"
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {
                "pass": paper_ready,
                "detail": "FoVer dual-condition AUROC 0.9131, 5-seed, CI, adversarial-clean",
                "source": headline_source,
            },
            "G2": {"pass": paper_ready, "detail": "CI runner 26725185125"},
            "G3": {"pass": paper_ready, "detail": "narrowing lint clean"},
            "G4": {
                "pass": paper_ready,
                "detail": "numbers trace to the frozen FoVer artifact",
                "source": headline_source,
            },
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _seed_status_artifacts(root: Path) -> None:
    results = root / "results"
    _write_json(
        results / "experiment_3659_trained_ebm_judge_ood_real_substrate_v3.json",
        {
            "honest_verdict": "complete: trained_judge_ood_hypothesis_retired",
            "adversarial_verify_clean": True,
            "trained_judge_ood_retired": True,
            "random_seed": 3659,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 1.0,
        },
    )
    _write_json(
        results / "experiment_3670_facts_row_real_benchmark.json",
        {
            "honest_verdict": "complete: facts_generalization_retired_real_benchmark_negative",
            "adversarial_verify_clean": True,
            "facts_generalization_retired": True,
            "random_seed": 3670,
            "reproducibility_checksum": "0" * 64,
            "duration_s": 1.0,
        },
    )
    _write_json(
        results / "experiment_3707_selection_diagnosis_formal_closure.json",
        {
            "honest_verdict": "complete: selection_diagnosis_formally_closed_retirement_recommended",
            "adversarial_verify_clean": True,
            "selection_diagnosis_closed": True,
            "retirement_recommended": True,
            "random_seed": 3707,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 1.0,
        },
    )


def _seed_upstreams(
    root: Path,
    *,
    exp3715_flagged: bool = False,
    exp3717_fully_traced: bool = True,
    energy_beats: bool = True,
    energy_auroc: float = 0.970578,
    energy_n: int = 6548,
    energy_leak_free: bool = False,
    fresh_generalizes: bool = False,
    omit_fresh_field: bool = False,
    fr11_fallback: bool = True,
    kv260_terminal: bool = True,
    operator_recorded: bool = True,
) -> None:
    results = root / "results"
    _seed_status_artifacts(root)
    _write_json(
        results / "experiment_3715_refreeze_disambiguation_clean_corrigendum.json",
        {
            "honest_verdict": "complete: refreeze_disambiguation_corrigendum_clean_no_candidate_beats_frozen_headline_stays_0_9131",
            "flagged_adversarial": exp3715_flagged,
            "adversarial_verify_clean": not exp3715_flagged,
            "no_candidate_beats_frozen": True,
            "frozen_headline_unchanged_assert": True,
            "frozen_headline_auroc": 0.9131,
            "random_seed": 3715,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 1.0,
        },
    )
    _write_json(
        results / "experiment_3716_ship_paper_v6_narrowing_lint.json",
        {
            "honest_verdict": "complete: paper_v6_narrowing_lint_shipped_g3_mechanically_enforced_current_paper_clean",
            "adversarial_verify_clean": True,
            "g3_now_mechanically_enforced": True,
            "current_paper_lint_clean": True,
            "pytest_passed": True,
            "conductor_unmodified_assert": True,
            "random_seed": 3716,
            "reproducibility_checksum": "b" * 64,
            "duration_s": 1.0,
        },
    )
    _write_json(
        results / "experiment_3717_g4_full_provenance_audit.json",
        {
            "honest_verdict": (
                "complete: g4_fully_traced_every_headline_number_to_clean_primary_artifact"
                if exp3717_fully_traced
                else "complete: g4_provenance_gap_found_operator_action_items_recorded"
            ),
            "adversarial_verify_clean": True,
            "all_numbers_trace_to_clean_artifacts": exp3717_fully_traced,
            "any_cited_source_flagged": False,
            "g4_status": "fully_traced" if exp3717_fully_traced else "gap_found",
            "random_seed": 3717,
            "reproducibility_checksum": "c" * 64,
            "duration_s": 1.0,
        },
    )
    _write_json(
        results / "experiment_3718_risk_coverage_abstention_characterization.json",
        {
            "honest_verdict": (
                "complete: energy_is_a_better_selective_prediction_signal_than_entropy_deployable_abstention_gate"
                if energy_beats
                else "complete: energy_ties_or_loses_to_entropy_abstention_baseline"
            ),
            "adversarial_verify_clean": True,
            "energy_beats_baseline_abstention": energy_beats,
            "energy_auroc": energy_auroc,
            "n_examples": energy_n,
            "leak_free": energy_leak_free,
            "random_seed": 3718,
            "reproducibility_checksum": "d" * 64,
            "duration_s": 1.0,
        },
    )
    fresh_payload: dict[str, object] = {
        "honest_verdict": (
            "complete: headline_discrimination_generalizes_to_fresh_corpus"
            if fresh_generalizes
            else "complete: headline_discrimination_is_fover_specific_generalization_narrowed_honest"
        ),
        "adversarial_verify_clean": True,
        "fresh_corpus_auroc": 0.798604,
        "n_examples": 418,
        "frozen_fover_auroc": 0.9131,
        "random_seed": 3719,
        "reproducibility_checksum": "e" * 64,
        "duration_s": 1.0,
    }
    if not omit_fresh_field:
        fresh_payload["generalizes_beyond_fover"] = fresh_generalizes
    _write_json(
        results / "experiment_3719_headline_replication_fresh_corpus.json",
        fresh_payload,
    )
    _write_json(
        results / "experiment_3720_fr11_continuous_self_learning_v14.json",
        {
            "honest_verdict": (
                "complete: fr11_v14_template_falls_back_gracefully_under_shift_no_collapse"
                if fr11_fallback
                else "complete: fr11_v14_template_hurts_under_distribution_shift"
            ),
            "adversarial_verify_clean": True,
            "template_robust_or_graceful_fallback": fr11_fallback,
            "conservative_fallback_triggered": fr11_fallback,
            "collapse_detected_deploy_arm": not fr11_fallback,
            "template_library_bounded": True,
            "random_seed": 3720,
            "reproducibility_checksum": "f" * 64,
            "duration_s": 1.0,
        },
    )
    _write_json(
        results / "experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json",
        {
            "honest_verdict": (
                "complete: kv260_terminal_confirmed_mandate_lift_recommended_polarfire_gatemate_audited"
                if kv260_terminal
                else "complete: kv260_terminal_not_confirmed"
            ),
            "adversarial_verify_clean": True,
            "kv260_terminal_condition_confirmed": kv260_terminal,
            "kv260_mandate_lift_recommendation": (
                "recommend_operator_lift_per_milestone_kv260_mandate"
                if kv260_terminal
                else "not_recommended"
            ),
            "speedup_claim_avoided_assert": True,
            "random_seed": 3721,
            "reproducibility_checksum": "1" * 64,
            "duration_s": 1.0,
        },
    )
    _write_json(
        results / "experiment_3722_convergence_synthesis_operator_next_thesis.json",
        {
            "honest_verdict": (
                "complete: convergence_synthesized_next_theses_presented_operator_decision_requested"
                if operator_recorded
                else "complete: convergence_synthesis_cannot_complete"
            ),
            "adversarial_verify_clean": True,
            "operator_decision_request": "Which thesis should drive .341+?"
            if operator_recorded
            else "",
            "all_self_generable_threads_settled": operator_recorded,
            "random_seed": 3722,
            "reproducibility_checksum": "2" * 64,
            "duration_s": 1.0,
        },
    )


def _clean_reports() -> dict[str, dict[str, object]]:
    return {name: {"flags": []} for name in exp3723.UPSTREAM_ARTIFACTS}


def test_scenario_publish_3723_builds_clean_v340_capstone(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3723: clean upstreams preserve hardened G1-G4."""

    spec = Path("openspec/capabilities/publication/spec.md").read_text(encoding="utf-8")
    assert "REQ-PUBLISH-3723" in spec
    assert "SCENARIO-PUBLISH-3723" in spec
    _seed_upstreams(tmp_path)

    artifact = exp3723.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[{"exp": exp_id, "returncode": 0} for exp_id in range(3715, 3723)],
        adversarial_reports=_clean_reports(),
        started_s=10.0,
        now_s=12.25,
    )

    exp3723.validate_artifact(artifact)
    assert set(exp3723.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3723.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: capstone_v340_convergence_gates_hardened_g3_mechanical_g4_audited_"
        "abstention_energy_better_than_entropy_fresh_corpus_fover_specific_"
        "kv260_terminal_operator_thesis_requested_paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["exp3704_corrigendum_clean"] is True
    assert artifact["g3_mechanically_enforced"] is True
    assert artifact["g4_provenance_audit_result"] == "fully_traced"
    assert artifact["energy_abstention_verdict"] == "energy_better_than_entropy"
    assert artifact["fresh_corpus_generalization"] == "fover_specific"
    assert artifact["fr11_v14_result"] == "falls_back_gracefully_under_shift_no_collapse"
    assert artifact["kv260_terminal_confirmed"] is True
    assert artifact["operator_next_thesis_recorded"] is True
    assert artifact["verifier_value_scope"] == (
        "math_discrimination_frozen_0_9131_second_corpus_datapoint_"
        "deployable_abstention_gate_if_energy_gt_entropy_code_math_only_with_abstain_"
        "facts_retired_selection_closed"
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
    assert artifact["selection_diagnosis_closed"] is True
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert len(artifact["cited_upstream_artifacts"]) == len(exp3723.UPSTREAM_ARTIFACTS)
    assert all(len(item["sha256"]) == 64 for item in artifact["cited_upstream_artifacts"])
    assert all(
        item["adversarial_verify_status"] == "no_critical_or_duration_flags"
        for item in artifact["cited_upstream_artifacts"]
    )
    assert any("FoVer headline remains 0.9131" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any(
        "Do not cite a re-freeze candidate as the headline" in claim
        for claim in artifact["paper_v6_forbidden_claims"]
    )


def test_req_publish_3723_flagged_leaky_and_missing_inputs_fail_closed(tmp_path: Path) -> None:
    """REQ-PUBLISH-3723: flagged, leaky, and missing fields are not synthesized."""

    _seed_upstreams(
        tmp_path,
        exp3715_flagged=True,
        exp3717_fully_traced=False,
        energy_auroc=0.995,
        energy_n=1200,
        omit_fresh_field=True,
        fr11_fallback=False,
        kv260_terminal=False,
        operator_recorded=False,
    )
    reports = _clean_reports()
    reports["exp3715"] = {"flags": [{"kind": "TAUTOLOGY", "severity": "critical"}]}
    artifact = exp3723.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        adversarial_reports=reports,
        started_s=1.0,
        now_s=1.5,
    )

    exp3723.validate_artifact(artifact)
    assert artifact["exp3704_corrigendum_clean"] is False
    assert artifact["g4_provenance_audit_result"] == "gap_found"
    assert artifact["energy_abstention_verdict"] == "not_measured"
    assert artifact["fresh_corpus_generalization"] == "not_measured"
    assert artifact["fr11_v14_result"] == "hurts_under_distribution_shift"
    assert artifact["kv260_terminal_confirmed"] is False
    assert artifact["operator_next_thesis_recorded"] is False
    cited_paths = {item["path"] for item in artifact["cited_upstream_artifacts"]}
    assert "results/experiment_3715_refreeze_disambiguation_clean_corrigendum.json" not in cited_paths
    assert "results/experiment_3718_risk_coverage_abstention_characterization.json" not in cited_paths
    excluded = {item["path"]: item["reason"] for item in artifact["excluded_upstream_artifacts"]}
    assert excluded["results/experiment_3715_refreeze_disambiguation_clean_corrigendum.json"] == "adversarial_blocking_flag"
    assert excluded["results/experiment_3718_risk_coverage_abstention_characterization.json"] == "leak_risk"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v340_convergence_gates_hardened_g3_mechanical_g4_audited_"
        "abstention_not_measured_fresh_corpus_not_measured_"
    )


def test_req_publish_3723_write_artifact_and_validation_guards(tmp_path: Path) -> None:
    """REQ-PUBLISH-3723: writing verifies the capstone and schema rejects regressions."""

    _seed_upstreams(tmp_path, fresh_generalizes=True, fr11_fallback=False)
    output_path = exp3723.write_artifact(
        tmp_path,
        output_path="results/custom_exp3723.json",
        gate_data=_gate_data(),
        summary_records=[],
        adversarial_reports=_clean_reports(),
        started_s=2.0,
        now_s=3.0,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    exp3723.validate_artifact(payload)
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["flags"] == []
    assert payload["fresh_corpus_generalization"] == "generalizes"
    assert payload["fr11_v14_result"] == "hurts_under_distribution_shift"

    validation_cases: list[tuple[dict[str, object], str]] = []
    missing = dict(payload)
    missing.pop("honest_verdict")
    validation_cases.append((missing, "missing required"))
    validation_cases.append((dict(payload, honest_verdict="done"), "complete:"))
    validation_cases.append((dict(payload, field_principles=[]), "field_principles"))
    missing_principle = dict(payload)
    missing_principle["field_principles"] = dict(payload["field_principles"])
    missing_principle["field_principles"].pop("g4")
    validation_cases.append((missing_principle, "missing field principles"))
    validation_cases.append((dict(payload, inference_substrate="aggregation_cuda"), "inference_substrate"))
    validation_cases.append((dict(payload, adversarial_verify_clean=False), "adversarial_verify_clean"))
    validation_cases.append((dict(payload, exp3704_corrigendum_clean="yes"), "exp3704_corrigendum_clean"))
    validation_cases.append((dict(payload, g3_mechanically_enforced="yes"), "g3_mechanically_enforced"))
    validation_cases.append((dict(payload, g4_provenance_audit_result="maybe"), "g4_provenance_audit_result"))
    validation_cases.append((dict(payload, energy_abstention_verdict="winner"), "energy_abstention_verdict"))
    validation_cases.append((dict(payload, fresh_corpus_generalization="maybe"), "fresh_corpus_generalization"))
    validation_cases.append((dict(payload, fr11_v14_result="maybe"), "fr11_v14_result"))
    validation_cases.append((dict(payload, kv260_terminal_confirmed="yes"), "kv260_terminal_confirmed"))
    validation_cases.append((dict(payload, operator_next_thesis_recorded="yes"), "operator_next_thesis_recorded"))
    validation_cases.append((dict(payload, verifier_value_scope="facts_only"), "verifier_value_scope"))
    validation_cases.append((dict(payload, g1=False), "g1"))
    validation_cases.append((dict(payload, paper_ready=False), "paper_ready"))
    validation_cases.append((dict(payload, frozen_headline_unchanged=False), "frozen_headline_unchanged"))
    validation_cases.append((dict(payload, unmet_gates="G2"), "unmet_gates"))
    validation_cases.append((dict(payload, p01_status="positive"), "p01_status"))
    validation_cases.append((dict(payload, facts_generalization_retired=False), "facts_generalization_retired"))
    validation_cases.append((dict(payload, trained_judge_ood_retired=False), "trained_judge_ood_retired"))
    validation_cases.append((dict(payload, selection_diagnosis_closed=False), "selection_diagnosis_closed"))
    validation_cases.append((dict(payload, paper_v6_safe_claims={}), "paper_v6_safe_claims"))
    validation_cases.append((dict(payload, paper_v6_forbidden_claims={}), "paper_v6_forbidden_claims"))
    validation_cases.append((dict(payload, duration_s=-1.0), "duration_s"))
    validation_cases.append((dict(payload, cited_upstream_artifacts={}), "cited_upstream_artifacts"))
    validation_cases.append((dict(payload, cited_upstream_artifacts=[{"path": "x"}]), "sha256"))
    validation_cases.append((dict(payload, reproducibility_checksum="short"), "reproducibility_checksum"))
    validation_cases.append((dict(payload, model_specs={"target_model": "gguf"}), "model_specs"))
    validation_cases.append((dict(payload, target_model="cuda"), "target_model"))
    for broken, pattern in validation_cases:
        with pytest.raises(ValueError, match=pattern):
            exp3723.validate_artifact(broken)


def test_req_publish_3723_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-PUBLISH-3723: helper edge cases remain explicit and deterministic."""

    assert exp3723._point({"point": 0.1234567}) == 0.123457
    assert exp3723._point({"ci95": [0.1, 0.2]}) is None
    assert exp3723._point(0.5) == 0.5
    assert exp3723._point("0.5") is None
    assert exp3723._gate_pass({"gates": {"G1": {"pass": True}}}, "G1") is True
    assert exp3723._gate_pass({"gates": {"G1": {"pass": False}}}, "G1") is False
    assert exp3723._gate_pass({}, "G1") is False
    assert exp3723._payload_declares_clean({"adversarial_verify_clean": True}) is True
    assert exp3723._payload_declares_clean({"adversarial_verify": "clean"}) is True
    assert exp3723._payload_declares_clean({"adversarial_verify_report": {"flags": []}}) is True
    assert exp3723._payload_declares_clean({}) is False
    assert exp3723._blocking_report({"flags": [{"severity": "critical"}]}) is True
    assert exp3723._blocking_report({"flags": [{"kind": "DURATION_TOO_SHORT"}]}) is True
    assert exp3723._blocking_report({"flags": [{"severity": "warn"}]}) is False
    assert exp3723._leak_risk({"n_examples": 1000, "energy_auroc": 0.991}) is True
    assert exp3723._leak_risk({"n_examples": 1000, "energy_auroc": 0.991, "leak_free": True}) is False
    assert exp3723._leak_risk({"n_examples": 1000, "leak_detected": True}) is True
    assert exp3723._leak_risk({"n_examples": 60, "energy_auroc": 1.0}) is False
    assert exp3723._g4_result({}, blocked=False) == "not_measured"
    assert exp3723._energy_verdict({"energy_beats_baseline_abstention": False}, blocked=False) == "energy_ties_or_loses"
    assert exp3723._energy_verdict({}, blocked=False) == "not_measured"
    assert exp3723._energy_verdict({"energy_auroc": 0.5}, blocked=False) == "not_measured"
    assert exp3723._fr11_v14_result({}, blocked=False) == "not_measured"
    assert exp3723._fr11_v14_result(
        {
            "template_robust_or_graceful_fallback": True,
            "collapse_detected_deploy_arm": False,
            "conservative_fallback_triggered": False,
        },
        blocked=False,
    ) == "robust_under_shift_no_collapse"
    assert exp3723._facts_retired({"facts_generalize_or_adds_value_real": False}, blocked=False) is True
    assert exp3723._facts_retired({"honest_outcome": "domain_bound_real"}, blocked=False) is True
    assert exp3723._judge_retired({"trained_judge_transfers_ood": False}, blocked=False) is True
    assert exp3723._judge_retired(
        {"honest_verdict": "complete: trained_judge_not_the_cross_domain_fix"},
        blocked=False,
    ) is True
    assert exp3723._selection_closed(
        {
            "question_closed": True,
            "honest_verdict": "complete: selection_diagnosis_formally_closed_retirement_recommended_to_operator",
        },
        blocked=False,
    ) is True
    assert exp3723._excluded_upstreams({"exp3715": {}}, {"exp3715": False}, {"exp3715": False}) == {}
    assert exp3723._frozen_headline_unchanged({"gates": []}) is False
    assert exp3723._repo_path(Path("/tmp/root"), Path("results/x.json")) == Path("/tmp/root/results/x.json")
    assert exp3723._repo_path(Path("/tmp/root"), Path("/abs/x.json")) == Path("/abs/x.json")
    assert exp3723._read_optional_json_object(tmp_path / "missing.json") == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp3723._read_optional_json_object(non_object)

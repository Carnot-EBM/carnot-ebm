"""Tests for Exp 3664 v335 capstone and G-gate synthesis.

Spec: REQ-PUBLISH-3664, SCENARIO-PUBLISH-3664.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_and_g_gate_v335_3664 as exp3664


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metric(point: float) -> dict[str, object]:
    return {
        "bootstrap_seeds": [3655, 3656, 3657],
        "ci95": [round(point - 0.04, 6), round(point + 0.04, 6)],
        "n": 500,
        "point": point,
    }


def _gate_data() -> dict[str, object]:
    return {
        "paper_ready": True,
        "gates": {
            "G1": {
                "pass": True,
                "detail": "FoVer 0.9131, 5-seed, CI",
                "source": "experiment_2850_fover_dual_condition_integrity_v4.json",
            },
            "G2": {"pass": True, "detail": "CI runner 26725185125"},
            "G3": {"pass": True, "detail": "narrowing-clean"},
            "G4": {"pass": True, "detail": "numbers trace to artifacts"},
        },
        "unmet_gates": [],
    }


def _seed_upstreams(
    root: Path,
    *,
    facts_generalize: bool = False,
    facts_auroc: float = 0.743656,
    confidence_auroc: float = 0.744576,
    exp3654_nli_built: bool = True,
    exp3654_leak_free: bool = True,
    exp3655_nli_built: bool | None = None,
    exp3655_leak_free: bool | None = None,
    exp3656_flagged: bool = True,
    code_replicates: bool = True,
    judge_transfers: bool = False,
) -> None:
    results = root / "results"
    _write_json(
        results / "experiment_3654_real_nli_atomic_claim_grounding_verifier.json",
        {
            "honest_verdict": (
                "complete: real_nli_grounding_verifier_built_beats_proxy_not_confidence"
            ),
            "nli_grounding_built": exp3654_nli_built,
            "grounding_leak_free": exp3654_leak_free,
            "nli_substrate": "model_based_transformers_checkpoint: fake-nli on cuda",
            "grounding_auroc": _metric(facts_auroc),
            "proxy_baseline_auroc": 0.6495,
            "grounding_auroc_vs_proxy_delta": round(facts_auroc - 0.6495, 6),
            "acceptance_gate": {"passed": exp3654_nli_built and exp3654_leak_free},
        },
    )
    exp3655: dict[str, object] = {
        "honest_verdict": (
            "complete: facts_generalize_with_real_nli"
            if facts_generalize
            else "complete: facts_domain_bound_even_with_real_nli_334_negative_confirmed_earned"
        ),
        "facts_generalize_real_nli": facts_generalize,
        "positive_control_valid": True,
        "nli_substrate": "model_based_transformers_checkpoint: fake-nli on cuda",
        "grounding_auroc_real_nli": _metric(facts_auroc),
        "confidence_baseline_auroc": _metric(confidence_auroc),
        "grounding_minus_confidence_delta": _metric(round(facts_auroc - confidence_auroc, 6)),
        "acceptance_gate": {"passed": True},
    }
    if exp3655_nli_built is not None:
        exp3655["nli_grounding_built"] = exp3655_nli_built
    if exp3655_leak_free is not None:
        exp3655["grounding_leak_free"] = exp3655_leak_free
    _write_json(results / "experiment_3655_facts_row_remeasurement_real_nli_v5.json", exp3655)
    _write_json(
        results / "experiment_3656_correlation_aware_weighting_paradox_diagnosis.json",
        {
            "honest_verdict": (
                "complete: paradox_resolved_naive_penalty_misspecified_dependency_aware_recovers"
            ),
            "flagged_adversarial": exp3656_flagged,
            "correlation_harmless_or_penalty_misspecified": (
                "H2_naive_penalty_misspecified_dependency_aware_recovers"
            ),
            "ensemble_auroc_dependency_aware_proper": 0.932562,
        },
    )
    _write_json(
        results / "experiment_3657_deployable_second_pair_of_eyes_detector.json",
        {
            "honest_verdict": (
                "complete: deployable_second_pair_of_eyes_detector_built_fusion_wins_calibrated"
            ),
            "fusion_beats_confidence_alone": True,
            "fused_detector_auroc": {"math": 0.954176, "code": 0.450241},
            "confidence_alone_auroc": {"math": 0.5, "code": 0.32504},
            "calibration_brier_ece": {"math": {"brier": 0.014645, "ece": 0.020289}},
            "acceptance_gate": {"passed": True},
        },
    )
    _write_json(
        results / "experiment_3658_code_generalization_second_corpus.json",
        {
            "honest_verdict": (
                "complete: code_generalization_replicates_on_balanced_second_corpus_claim_hardened"
                if code_replicates
                else "complete: code_generalization_does_not_replicate"
            ),
            "code_generalization_replicates": code_replicates,
            "code_verifiers_fire": code_replicates,
            "math_signal_code_auroc": _metric(0.532222),
            "confidence_baseline_auroc": _metric(0.483333),
            "class_balance": {"balanced": True, "min_class_fraction": 0.5},
            "acceptance_gate": {"passed": code_replicates},
        },
    )
    _write_json(
        results / "experiment_3659_trained_ebm_judge_ood_real_substrate_v3.json",
        {
            "honest_verdict": "complete: real_substrate_trained_judge_result",
            "trained_judge_transfers_ood": judge_transfers,
            "in_domain_judge_auroc": 0.978741,
            "ood_judge_auroc": 0.572465 if not judge_transfers else 0.812345,
            "confidence_only_baseline_auroc": 0.882162,
            "real_substrate_vs_confidence_ood_delta": -0.309697,
            "acceptance_gate": {"passed": True},
        },
    )
    _write_json(
        results / "experiment_3660_fr11_continuous_self_learning_v9.json",
        {
            "honest_verdict": "complete: fr11_v9_online_fusion_weighting_holds",
            "online_fusion_auroc_gain": 0.168457,
            "quality_maintained": True,
            "acceptance_gate": {"passed": True},
        },
    )


def test_scenario_publish_3664_builds_domain_bound_capstone_from_real_nli(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-3664: real NLI confirms facts are domain-bound."""

    _seed_upstreams(tmp_path)
    artifact = exp3664.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[{"exp": exp_id, "returncode": 0} for exp_id in range(3654, 3661)],
        started_s=10.0,
        now_s=12.5,
    )

    exp3664.validate_artifact(artifact)
    assert set(exp3664.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3664.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: capstone_v335_facts_domain_bound_with_real_nli_"
        "verifier_value_math_plus_code_paper_ready_true"
    )
    assert artifact["facts_generalize_real_nli"] is False
    assert artifact["code_generalization_replicated"] is True
    assert artifact["second_pair_of_eyes_deployable"] is True
    assert artifact["trained_judge_real_substrate_result"]["transfers_ood"] is False
    assert artifact["correlation_paradox_resolution"]["status"] == "excluded_flagged_adversarial"
    assert artifact["verifier_value_scope"] == "math_plus_code"
    assert artifact["p01_status"] == "honest-negative"
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["duration_s"] == pytest.approx(2.5)

    table = artifact["corrected_generalization_table"]
    assert table["math"]["auroc"] == 0.9131
    assert table["code"]["auroc"] == 0.532222
    assert table["code"]["delta"] == 0.048889
    assert table["facts"]["auroc"] == 0.743656
    assert table["facts"]["delta"] == -0.00092
    assert table["facts"]["ran_or_blocked"] == "ran"
    assert artifact["real_nli_vs_proxy_correction"]["status"] == (
        "real_nli_confirms_proxy_negative"
    )
    cited_paths = {row["path"] for row in artifact["cited_upstream_artifacts"]}
    assert "results/experiment_3656_correlation_aware_weighting_paradox_diagnosis.json" not in cited_paths
    assert "results/experiment_3660_fr11_continuous_self_learning_v9.json" in cited_paths
    assert any("facts remain domain-bound" in claim for claim in artifact["paper_v6_safe_claims"])
    assert any("flagged_adversarial" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_publish_3664_real_nli_generalization_expands_scope(tmp_path: Path) -> None:
    """REQ-PUBLISH-3664: facts can only expand scope when real NLI actually passes."""

    _seed_upstreams(
        tmp_path,
        facts_generalize=True,
        facts_auroc=0.84,
        confidence_auroc=0.70,
        exp3656_flagged=False,
    )
    artifact = exp3664.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=1.0,
    )

    assert artifact["honest_verdict"] == (
        "complete: capstone_v335_facts_generalize_with_real_nli_"
        "verifier_value_math_plus_code_plus_facts_paper_ready_true"
    )
    assert artifact["facts_generalize_real_nli"] is True
    assert artifact["verifier_value_scope"] == "math_plus_code_plus_facts"
    assert artifact["correlation_paradox_resolution"]["status"] == (
        "H2_naive_penalty_misspecified_dependency_aware_recovers"
    )
    assert any("real-NLI facts grounding generalizes" in claim for claim in artifact["paper_v6_safe_claims"])


@pytest.mark.parametrize(
    ("nli_built", "facts_auroc", "exp3655_leak_free"),
    [
        (False, 0.84, None),
        (True, 0.999, None),
    ],
)
def test_req_publish_3664_blocked_or_leaky_facts_are_not_measured(
    tmp_path: Path,
    nli_built: bool,
    facts_auroc: float,
    exp3655_leak_free: bool | None,
) -> None:
    """REQ-PUBLISH-3664: blocked or implausible facts rows fail closed."""

    _seed_upstreams(
        tmp_path,
        facts_generalize=True,
        facts_auroc=facts_auroc,
        confidence_auroc=0.70,
        exp3654_nli_built=nli_built,
        exp3655_nli_built=nli_built,
        exp3655_leak_free=exp3655_leak_free,
    )
    artifact = exp3664.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=0.5,
    )

    assert artifact["facts_generalize_real_nli"] == "not_measured_real_nli"
    assert artifact["corrected_generalization_table"]["facts"]["ran_or_blocked"] == (
        "not_measured_real_nli"
    )
    assert artifact["verifier_value_scope"] == "math_plus_code"
    assert artifact["real_nli_vs_proxy_correction"]["status"] == "not_measured_real_nli"


def test_req_publish_3664_write_artifact_and_validation_guards(tmp_path: Path) -> None:
    """REQ-PUBLISH-3664: writing persists JSON and validation rejects regressions."""

    _seed_upstreams(tmp_path)
    output_path = exp3664.write_artifact(
        tmp_path,
        output_path="results/custom_exp3664.json",
        gate_data=_gate_data(),
        summary_records=[],
        started_s=2.0,
        now_s=4.0,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    exp3664.validate_artifact(payload)
    assert payload["paper_ready"] is True

    missing = dict(payload)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp3664.validate_artifact(missing)

    bad_verdict = dict(payload, honest_verdict="completeish")
    with pytest.raises(ValueError, match="complete:"):
        exp3664.validate_artifact(bad_verdict)

    bad_scope = dict(payload, verifier_value_scope="facts_only")
    with pytest.raises(ValueError, match="verifier_value_scope"):
        exp3664.validate_artifact(bad_scope)

    bad_paper = dict(payload, paper_ready=False)
    with pytest.raises(ValueError, match="paper_ready"):
        exp3664.validate_artifact(bad_paper)

    validation_cases: list[tuple[dict[str, object], str]] = []
    validation_cases.append((dict(payload, field_principles=[]), "field_principles"))
    missing_principle = dict(payload)
    missing_principle["field_principles"] = dict(payload["field_principles"])
    missing_principle["field_principles"].pop("duration_s")
    validation_cases.append((missing_principle, "missing field principles"))
    validation_cases.append((dict(payload, inference_substrate="live_gpu"), "inference_substrate"))
    validation_cases.append((dict(payload, g4=False), "g4"))
    validation_cases.append((dict(payload, p01_status="positive"), "p01_status"))
    validation_cases.append((dict(payload, unmet_gates="G2"), "unmet_gates"))
    validation_cases.append((dict(payload, corrected_generalization_table={}), "corrected_generalization_table"))
    bad_row_type = dict(payload)
    bad_row_type["corrected_generalization_table"] = dict(payload["corrected_generalization_table"])
    bad_row_type["corrected_generalization_table"]["facts"] = []
    validation_cases.append((bad_row_type, "facts row"))
    bad_row_missing = dict(payload)
    bad_row_missing["corrected_generalization_table"] = dict(payload["corrected_generalization_table"])
    bad_row_missing["corrected_generalization_table"]["facts"] = dict(
        payload["corrected_generalization_table"]["facts"]
    )
    bad_row_missing["corrected_generalization_table"]["facts"].pop("delta")
    validation_cases.append((bad_row_missing, "facts row missing"))
    validation_cases.append((dict(payload, facts_generalize_real_nli=None), "facts_generalize_real_nli"))
    validation_cases.append((dict(payload, real_nli_vs_proxy_correction=[]), "real_nli_vs_proxy_correction"))
    validation_cases.append((dict(payload, correlation_paradox_resolution=[]), "correlation_paradox_resolution"))
    validation_cases.append((dict(payload, trained_judge_real_substrate_result=[]), "trained_judge"))
    validation_cases.append((dict(payload, paper_v6_safe_claims={}), "paper_v6_safe_claims"))
    validation_cases.append((dict(payload, paper_v6_forbidden_claims={}), "paper_v6_forbidden_claims"))
    validation_cases.append((dict(payload, duration_s=-1.0), "duration_s"))
    validation_cases.append((dict(payload, cited_upstream_artifacts={}), "cited_upstream_artifacts"))
    validation_cases.append((dict(payload, cited_upstream_artifacts=[{"path": "x"}]), "sha256"))
    validation_cases.append((dict(payload, reproducibility_checksum="short"), "reproducibility_checksum"))
    for broken, pattern in validation_cases:
        with pytest.raises(ValueError, match=pattern):
            exp3664.validate_artifact(broken)


def test_req_publish_3664_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-PUBLISH-3664: helper edge cases are explicit and fail closed."""

    exp3654 = {
        "nli_grounding_built": True,
        "grounding_leak_free": True,
        "nli_substrate": "model_based_transformers_checkpoint: fake-nli on cuda",
        "proxy_baseline_auroc": "not numeric",
    }
    exp3655 = {
        "facts_generalize_real_nli": False,
        "positive_control_valid": True,
        "grounding_auroc_real_nli": {"point": 0.75},
        "confidence_baseline_auroc": {"point": 0.70},
        "acceptance_gate": {"passed": True},
    }
    facts = exp3664._facts_real_nli_result(exp3654, exp3655, flagged=False)
    assert facts["delta"] == 0.05
    assert exp3664._trained_judge_result({}, flagged=True)["status"] == "excluded_flagged_adversarial"
    assert exp3664._verifier_scope(code_generalized=False, facts_generalized=True) == "broad"
    assert exp3664._verifier_scope(code_generalized=False, facts_generalized=False) == "math_only_earned"
    assert exp3664._correlation_status(
        {"honest_verdict": "complete: naive_penalty_misspecified_dependency_aware_recovers"}
    ) == "H2_naive_penalty_misspecified_dependency_aware_recovers"
    assert exp3664._correlation_status(
        {"correlation_harmless_or_penalty_misspecified": "correlation_harmless"}
    ) == "H1_correlation_harmless"
    assert exp3664._correlation_status(
        {"correlation_harmless_or_penalty_misspecified": "custom_resolution"}
    ) == "custom_resolution"
    assert exp3664._correlation_status({"honest_verdict": "complete: correlation_harmless"}) == (
        "H1_correlation_harmless"
    )
    assert exp3664._correlation_status({"honest_verdict": "complete: unresolved"}) == "unknown"
    assert exp3664._gate_pass({"gates": []}, "G1") is False
    assert exp3664._point(0.1234567) == 0.123457
    assert exp3664._difference(None, 0.1) is None
    assert exp3664._round_or_none("not numeric") is None
    list_path = tmp_path / "not_object.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp3664._read_json_object(list_path)

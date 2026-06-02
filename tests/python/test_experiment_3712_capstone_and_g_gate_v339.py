"""Tests for Exp 3712 v339 re-freeze winner capstone and G-gate recheck.

Spec: REQ-PUBLISH-3712, SCENARIO-PUBLISH-3712.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_and_g_gate_v339_3712 as exp3712


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
                "detail": "FoVer AUROC 0.9131, 5-seed, CI, adversarial-clean",
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
    exp3704_flagged: bool = False,
    strongest_candidate: str = "fusion",
    refreeze_package: bool = True,
    exp3705_flagged: bool = False,
    code_survives: bool = True,
    heldout_auroc: float = 0.812345,
    n_examples_heldout: int = 1200,
    exp3706_recalibrated: bool = True,
    exp3707_closed: bool = True,
    exp3708_success: bool = True,
    exp3709_terminal: bool = True,
    write_exp3706: bool = True,
) -> None:
    results = root / "results"
    _write_json(
        results / "experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion.json",
        {
            "honest_verdict": (
                f"complete: refreeze_disambiguated_winner_{strongest_candidate}"
                "_beats_frozen_clean_package_reemitted_frozen_headline_unchanged"
                if refreeze_package
                else "complete: refreeze_disambiguated_no_candidate_beats_frozen_headline_stays_0_9131"
            ),
            "flagged_adversarial": exp3704_flagged,
            "adversarial_verify_clean": not exp3704_flagged,
            "strongest_candidate": strongest_candidate if refreeze_package else "none",
            "strongest_candidate_beats_frozen": refreeze_package,
            "refreeze_package_reemitted_for_winner": refreeze_package,
            "dependency_aware_auroc": 0.921,
            "external_comparator_auroc": 0.925,
            "fusion_auroc": 0.932,
            "frozen_headline_unchanged_assert": True,
            "acceptance_gate": {"passed": True},
            "random_seed": 3704,
            "reproducibility_checksum": "4" * 64,
            "duration_s": 31.0,
        },
    )
    _write_json(
        results / "experiment_3705_code_native_leak_audit_heldout.json",
        {
            "honest_verdict": (
                "complete: code_native_signal_survives_heldout_real_non_leaked_signal"
                if code_survives
                else "complete: code_native_one_point_zero_was_a_leak_code_claim_narrowed_earned"
            ),
            "flagged_adversarial": exp3705_flagged,
            "adversarial_verify_clean": not exp3705_flagged,
            "leak_detected": not code_survives,
            "code_signal_survives_heldout": code_survives,
            "heldout_code_auroc": heldout_auroc,
            "n_examples_heldout": n_examples_heldout,
            "acceptance_gate": {"passed": True},
            "random_seed": 3705,
            "reproducibility_checksum": "5" * 64,
            "duration_s": 12.0,
        },
    )
    if write_exp3706:
        _write_json(
            results / "experiment_3706_reconcile_shipped_detector_heldout.json",
            {
                "honest_verdict": (
                    "complete: shipped_detector_code_recalibrated_to_heldout"
                    if exp3706_recalibrated
                    else "complete: shipped_detector_narrowed_to_math_only_abstain"
                ),
                "adversarial_verify_clean": True,
                "detector_reconciled_to_heldout": exp3706_recalibrated,
                "math_only_abstain": not exp3706_recalibrated,
                "acceptance_gate": {"passed": True},
                "random_seed": 3706,
                "reproducibility_checksum": "6" * 64,
                "duration_s": 4.0,
            },
        )
    _write_json(
        results / "experiment_3707_selection_diagnosis_formal_closure.json",
        {
            "honest_verdict": (
                "complete: selection_diagnosis_formally_closed_retirement_recommended"
                if exp3707_closed
                else "complete: selection_diagnosis_open"
            ),
            "adversarial_verify_clean": True,
            "selection_diagnosis_closed": exp3707_closed,
            "retirement_recommended": exp3707_closed,
            "acceptance_gate": {"passed": True},
            "random_seed": 3707,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 1.0,
        },
    )
    _write_json(
        results / "experiment_3708_fr11_continuous_self_learning_v13.json",
        {
            "honest_verdict": (
                "complete: fr11_v13_multi_session_consolidation_transfers_no_collapse"
                if exp3708_success
                else "complete: fr11_v13_transfer_collapse_or_regression"
            ),
            "adversarial_verify_clean": True,
            "acceptance_gate": {"passed": exp3708_success},
            "fresh_session_transfer_without_collapse": exp3708_success,
            "multi_session_consolidation_transferred": exp3708_success,
            "collapse_detected": not exp3708_success,
            "quality_maintained": exp3708_success,
            "random_seed": 3708,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 3.0,
        },
    )
    _write_json(
        results / "experiment_3709_kv260_drive_to_terminal_latency_transcript.json",
        {
            "honest_verdict": (
                "complete: kv260_board_latency_transcript_captured_poc_anchor_terminal_candidate"
                if exp3709_terminal
                else "complete: kv260_blocked_unreachable"
            ),
            "inference_substrate": "hardware_smoke",
            "adversarial_verify_clean": True,
            "acceptance_gate": {"passed": exp3709_terminal},
            "latency_transcript_captured": exp3709_terminal,
            "terminal_candidate": exp3709_terminal,
            "terminal_condition_met": exp3709_terminal,
            "kv260_ssh_reachable": exp3709_terminal,
            "speedup_claim_avoided_assert": True,
            "board_latency_samples": (
                [0.025 + index * 0.000001 for index in range(32)] if exp3709_terminal else []
            ),
            "board_latency_median_ms": 0.025465 if exp3709_terminal else None,
            "blocked_unreachable": not exp3709_terminal,
            "random_seed": 3709,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 6.0,
        },
    )


def test_scenario_publish_3712_builds_clean_v339_capstone(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3712: clean upstreams preserve G1-G4 and scoped claims."""

    spec = Path("openspec/capabilities/publication/spec.md").read_text(encoding="utf-8")
    assert "REQ-PUBLISH-3712" in spec
    assert "SCENARIO-PUBLISH-3712" in spec
    _seed_upstreams(tmp_path)

    artifact = exp3712.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[{"exp": exp_id, "returncode": 0} for exp_id in range(3704, 3710)],
        started_s=10.0,
        now_s=12.5,
    )

    exp3712.validate_artifact(artifact)
    assert set(exp3712.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3712.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: capstone_v339_refreeze_winner_fusion_code_native_"
        "survives_heldout_real_signal_selection_closed_kv260_"
        "latency_transcript_captured_terminal_candidate_paper_ready_true_"
        "frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["strongest_refreeze_candidate"] == "fusion"
    assert artifact["refreeze_package_status"] == "reemitted_clean_for_winner"
    assert artifact["code_native_heldout_verdict"] == "survives_heldout_real_signal"
    assert artifact["shipped_detector_reconciliation"] == "code_recalibrated_to_heldout"
    assert artifact["selection_diagnosis_closed"] is True
    assert artifact["fr11_v13_result"] == "multi_session_consolidation_transferred_no_collapse"
    assert artifact["kv260_terminal_status"] == "latency_transcript_captured_terminal_candidate"
    assert artifact["verifier_value_scope"] == (
        "math_discrimination_frozen_0_9131_plus_refreeze_candidate_fusion_"
        "code_heldout_real_signal_facts_retired_selection_closed"
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
    assert any("Do not cite a code AUROC=1.0" in claim for claim in artifact["paper_v6_forbidden_claims"])


def test_req_publish_3712_flagged_leaky_and_missing_inputs_fail_closed(tmp_path: Path) -> None:
    """REQ-PUBLISH-3712: flagged, leaky, and missing upstreams are not synthesized."""

    _seed_upstreams(
        tmp_path,
        exp3704_flagged=True,
        exp3705_flagged=False,
        code_survives=True,
        heldout_auroc=0.995,
        n_examples_heldout=1200,
        write_exp3706=False,
        exp3708_success=False,
        exp3709_terminal=False,
    )
    artifact = exp3712.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=1.0,
        now_s=1.25,
    )

    exp3712.validate_artifact(artifact)
    assert artifact["strongest_refreeze_candidate"] == "none"
    assert artifact["refreeze_package_status"] == "not_measured"
    assert artifact["code_native_heldout_verdict"] == "one_point_zero_was_a_leak"
    assert artifact["code_native_heldout"]["leak_risk"] is True
    assert artifact["shipped_detector_reconciliation"] == "not_measured"
    assert artifact["fr11_v13_result"] == "collapse_or_quality_regression"
    assert artifact["kv260_terminal_status"] == "blocked_unreachable"
    cited_paths = {item["path"] for item in artifact["cited_upstream_artifacts"]}
    assert "results/experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion.json" not in cited_paths
    assert "results/experiment_3705_code_native_leak_audit_heldout.json" not in cited_paths
    assert artifact["flagged_upstream_artifacts_excluded"] == [
        "results/experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion.json",
        "results/experiment_3705_code_native_leak_audit_heldout.json",
    ]
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v339_refreeze_winner_none_code_native_one_point_zero_was_a_leak_"
    )


def test_req_publish_3712_no_winner_and_math_only_abstain(tmp_path: Path) -> None:
    """REQ-PUBLISH-3712: no-win and abstain states remain explicit."""

    _seed_upstreams(
        tmp_path,
        strongest_candidate="dependency_aware",
        refreeze_package=False,
        code_survives=False,
        heldout_auroc=0.51,
        n_examples_heldout=200,
        exp3706_recalibrated=False,
    )

    artifact = exp3712.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=0.75,
    )

    exp3712.validate_artifact(artifact)
    assert artifact["strongest_refreeze_candidate"] == "none"
    assert artifact["refreeze_package_status"] == "no_candidate_beats_frozen"
    assert artifact["code_native_heldout_verdict"] == "one_point_zero_was_a_leak"
    assert artifact["shipped_detector_reconciliation"] == "narrowed_to_math_only_abstain"
    assert artifact["verifier_value_scope"] == (
        "math_discrimination_frozen_0_9131_no_refreeze_candidate_"
        "code_math_only_with_abstain_facts_retired_selection_closed"
    )


def test_req_publish_3712_missing_upstreams_are_not_measured(tmp_path: Path) -> None:
    """REQ-PUBLISH-3712: absent gated tasks fail closed without fabricated None reads."""

    _write_json(
        tmp_path / "results" / "experiment_3707_selection_diagnosis_formal_closure.json",
        {
            "honest_verdict": "complete: selection_diagnosis_formally_closed_retirement_recommended",
            "adversarial_verify_clean": True,
            "selection_diagnosis_closed": True,
            "retirement_recommended": True,
            "acceptance_gate": {"passed": True},
            "random_seed": 3707,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 1.0,
        },
    )
    artifact = exp3712.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=[],
        started_s=0.0,
        now_s=0.5,
    )

    exp3712.validate_artifact(artifact)
    assert artifact["strongest_refreeze_candidate"] == "none"
    assert artifact["refreeze_package_status"] == "not_measured"
    assert artifact["code_native_heldout_verdict"] == "not_measured"
    assert artifact["shipped_detector_reconciliation"] == "not_measured"
    assert artifact["selection_diagnosis_closed"] is True
    assert artifact["fr11_v13_result"] == "not_measured"
    assert artifact["kv260_terminal_status"] == "not_measured"
    assert [item["path"] for item in artifact["cited_upstream_artifacts"]] == [
        "results/experiment_3707_selection_diagnosis_formal_closure.json"
    ]
    assert artifact["source_artifacts"] == [
        "results/experiment_3707_selection_diagnosis_formal_closure.json"
    ]


def test_req_publish_3712_write_artifact_and_validation_guards(tmp_path: Path) -> None:
    """REQ-PUBLISH-3712: writing persists JSON and validation rejects regressions."""

    _seed_upstreams(tmp_path)
    output_path = exp3712.write_artifact(
        tmp_path,
        output_path="results/custom_exp3712.json",
        gate_data=_gate_data(),
        summary_records=[],
        started_s=2.0,
        now_s=3.0,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    exp3712.validate_artifact(payload)
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
    missing_principle["field_principles"].pop("kv260_terminal_status")
    validation_cases.append((missing_principle, "missing field principles"))
    validation_cases.append((dict(payload, inference_substrate="live_model_cuda"), "inference_substrate"))
    validation_cases.append((dict(payload, adversarial_verify_clean=False), "adversarial_verify_clean"))
    validation_cases.append((dict(payload, strongest_refreeze_candidate="current"), "strongest_refreeze_candidate"))
    validation_cases.append((dict(payload, refreeze_package_status="ready"), "refreeze_package_status"))
    validation_cases.append((dict(payload, code_native_heldout_verdict="perfect"), "code_native_heldout_verdict"))
    validation_cases.append((dict(payload, shipped_detector_reconciliation="reshipped"), "shipped_detector_reconciliation"))
    validation_cases.append((dict(payload, selection_diagnosis_closed=False), "selection_diagnosis_closed"))
    validation_cases.append((dict(payload, fr11_v13_result=True), "fr11_v13_result"))
    validation_cases.append((dict(payload, kv260_terminal_status="done"), "kv260_terminal_status"))
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
            exp3712.validate_artifact(broken)


def test_req_publish_3712_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-PUBLISH-3712: helper edge cases are explicit and deterministic."""

    assert exp3712._point({"point": 0.1234567}) == 0.123457
    assert exp3712._point({"ci95": [0.1, 0.2]}) is None
    assert exp3712._point(0.5) == 0.5
    assert exp3712._point("0.5") is None
    assert exp3712._acceptance_pass({"acceptance_gate": {"passed": True}}) is True
    assert exp3712._acceptance_pass({"acceptance_gate": {"required_fields_present": True}}) is True
    assert exp3712._acceptance_pass({"acceptance_gate": False}) is False
    assert exp3712._gate_pass({"gates": {"G1": {"pass": True}}}, "G1") is True
    assert exp3712._gate_pass({"gates": {"G1": {"pass": False}}}, "G1") is False
    assert exp3712._gate_pass({}, "G1") is False
    assert exp3712._frozen_headline_unchanged({}) is False
    assert exp3712._payload_declares_adversarial_clean({"adversarial_verify_clean": True}) is True
    assert exp3712._payload_declares_adversarial_clean({"adversarial_verify": "clean"}) is True
    assert exp3712._payload_declares_adversarial_clean({"adversarial_verify_report": {"flags": []}}) is True
    assert exp3712._payload_declares_adversarial_clean({}) is False
    assert exp3712._selection_closed({}, hard_flagged=False) is False
    assert exp3712._kv260_status({"adversarial_verify_clean": True}, hard_flagged=False)["status"] == "blocked_unreachable"
    assert exp3712.adversarial_report_is_clean({"flags": []}) is True
    assert exp3712.adversarial_report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp3712.adversarial_report_is_clean({"flags": [{"kind": "DURATION_TOO_SHORT", "severity": "warn"}]}) is False
    assert exp3712._leak_risk({"n_examples_heldout": 1000, "heldout_code_auroc": 0.991}) is True
    assert exp3712._leak_risk({"n_examples_heldout": 1000, "heldout_code_auroc": 0.991, "leak_free": True}) is False
    assert exp3712._leak_risk({"n_examples_heldout": 60, "heldout_code_auroc": 1.0}) is False
    assert exp3712._report_has_critical({"flags": [{"severity": "critical"}]}) is True
    assert exp3712._report_has_critical({"flags": [{"severity": "warn"}]}) is False
    assert exp3712._repo_path(Path("/tmp/root"), Path("results/x.json")) == Path("/tmp/root/results/x.json")
    assert exp3712._repo_path(Path("/tmp/root"), Path("/abs/x.json")) == Path("/abs/x.json")
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp3712._read_optional_json_object(non_object)

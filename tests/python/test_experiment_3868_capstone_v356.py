"""Tests for Exp 3868 capstone v356 moat adjudication.

Spec refs: REQ-CAPSTONE-3868, SCENARIO-CAPSTONE-3868.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v356_moat_3868 as exp3868


SPEC_PATH = Path("openspec/capabilities/capstone/spec.md")


def _write_json(root: Path, rel_path: str, payload: dict[str, object]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publication_gate() -> dict[str, object]:
    return {
        "paper_ready": True,
        "gates": {
            "G1": {"pass": True},
            "G2": {"pass": True},
            "G3": {"pass": True},
            "G4": {"pass": True},
        },
        "unmet_gates": [],
    }


def _clean_reports() -> dict[int, dict[str, object]]:
    return {experiment_id: {"flags": []} for experiment_id in exp3868.UPSTREAM_IDS}


def _write_complete_fixture(root: Path) -> None:
    _write_json(
        root,
        "data/step_error_balanced_v2.json",
        {
            "honest_verdict": "complete: balanced_step_error_corpus_v2_n1000",
            "n_items": 1000,
            "n_incorrect_steps": 500,
            "primary_source": "prmbench",
            "flagged_adversarial": False,
            "reproducibility_checksum": "a" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3859_moat_scissor_at_scale_v3.json",
        {
            "honest_verdict": "complete: moat_scissor_residual_catch_high",
            "n_residual_errors": 80,
            "residual_catch_rate": 0.76,
            "residual_catch_ci95": {"low": 0.71, "high": 0.82},
            "overlap": 0.11,
            "flagged_adversarial": False,
            "reproducibility_checksum": "b" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3860_verifier_reasoner_independence_audit.json",
        {
            "honest_verdict": "complete: independence_audit_high_correlation",
            "reasoner_carnot_error_correlation": 0.73,
            "independence_is_real": False,
            "flagged_adversarial": False,
            "reproducibility_checksum": "c" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3861_thinkprm_complementarity.json",
        {
            "honest_verdict": "complete: thinkprm_complementarity_union_lift_positive",
            "thinkprm_catch_rate": 0.81,
            "cheap_ensemble_catch_rate": 0.47,
            "union_catch_rate": 0.86,
            "union_lift_over_thinkprm": 0.05,
            "cheap_ensemble_adds_catch_over_thinkprm": True,
            "flagged_adversarial": False,
            "reproducibility_checksum": "d" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3862_graph_grounding_fact_verifier_prototype_v2.json",
        {
            "honest_verdict": "complete: graph_grounding_signal_flagged",
            "flagged_adversarial": True,
            "facts_catch_delta": 0.23,
            "reproducibility_checksum": "e" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3863_graph_verifier_facts_complementarity_v2.json",
        {
            "honest_verdict": "complete: graph_facts_complementarity_positive",
            "graph_facts_catch_rate": 0.72,
            "math_ensemble_facts_catch_rate": 0.51,
            "union_facts_catch_rate": 0.79,
            "union_lift_over_math_ensemble": 0.28,
            "extended_ensemble_recommended": True,
            "flagged_adversarial": False,
            "reproducibility_checksum": "f" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3864_fr11_self_learning_v23_independence_reweighting.json",
        {
            "honest_verdict": "complete: fr11_v23_state_preserved",
            "auroc_in_frozen_ci": True,
            "memory_ablation_contribution_preserved": True,
            "reweighted_ensemble_auroc": 0.90592,
            "frozen_headline_ensemble_auroc": 0.9131,
            "state_persisted_path": "results/state.json",
            "flagged_adversarial": False,
            "reproducibility_checksum": "1" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3865_ldt_lattice_margin_sharpening_v2.json",
        {
            "honest_verdict": "complete: ldt_margin_LATTICE_REAL",
            "ensemble_vs_score_matched_margin": 0.0056,
            "margin_ci95": [0.0021, 0.0080],
            "frozen_fover_auroc_unchanged": True,
            "flagged_adversarial": False,
            "reproducibility_checksum": "2" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3866_gatemate_ising_tile_flash_v2.json",
        {
            "honest_verdict": "success: gatemate_flashed_terminal_flagged",
            "gatemate_bitstream_flashed": True,
            "flagged_adversarial": True,
            "reproducibility_checksum": "3" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3867_polarfire_soc_smoke_v4.json",
        {
            "honest_verdict": "success: polarfire_hash_verified_terminal",
            "polarfire_workload_validated": True,
            "result_hash_match": True,
            "no_fpga_fabric_claim": True,
            "flagged_adversarial": False,
            "reproducibility_checksum": "4" * 64,
        },
    )


def test_req_capstone_3868_spec_declares_conditioned_moat_contract() -> None:
    """REQ-CAPSTONE-3868: OpenSpec names the conditioned moat contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-3868" in spec
    assert "SCENARIO-CAPSTONE-3868" in spec
    assert "condition the moat-durability verdict" in spec
    assert "exp3860 independence audit" in spec


def test_req_capstone_3868_conditioned_verdict_never_uses_residual_alone() -> None:
    """REQ-CAPSTONE-3868: residual catch is gated by exp3860 independence."""

    scissor_high = {
        "residual_catch_ci95": {"low": 0.72, "high": 0.86},
        "residual_catch_rate": 0.79,
        "overlap": 0.12,
    }

    fake = exp3868.conditioned_moat_verdict(
        scissor_high,
        {"reasoner_carnot_error_correlation": 0.74, "independence_is_real": False},
    )
    durable = exp3868.conditioned_moat_verdict(
        scissor_high,
        {"reasoner_carnot_error_correlation": 0.08, "independence_is_real": True},
    )
    subsumed = exp3868.conditioned_moat_verdict(
        {"residual_catch_ci95": [0.05, 0.18], "residual_catch_rate": 0.11},
        {"reasoner_carnot_error_correlation": 0.04, "independence_is_real": True},
    )
    blocked = exp3868.conditioned_moat_verdict(
        {"honest_verdict": "blocked_gate_check_failed", "residual_catch_ci95": None},
        {"reasoner_carnot_error_correlation": 0.04, "independence_is_real": True},
    )
    absent_metric = exp3868.conditioned_moat_verdict(
        {"residual_catch_ci95": None},
        {"reasoner_carnot_error_correlation": 0.04, "independence_is_real": True},
    )
    middle = exp3868.conditioned_moat_verdict(
        {"residual_catch_ci95": [0.42, 0.51]},
        {"reasoner_carnot_error_correlation": 0.04, "independence_is_real": True},
    )
    missing = exp3868.conditioned_moat_verdict(scissor_high, None)

    assert fake["verdict"] == "MOAT FAKE-INDEPENDENCE"
    assert fake["moat_is_real_independence"] is False
    assert "fake moat" in fake["rationale"]
    assert durable["verdict"] == "MOAT DURABLE"
    assert durable["moat_is_real_independence"] is True
    assert subsumed["verdict"] == "MOAT SUBSUMED"
    assert blocked["verdict"] == "INCONCLUSIVE"
    assert absent_metric["verdict"] == "INCONCLUSIVE"
    assert middle["verdict"] == "INCONCLUSIVE"
    assert missing["verdict"] == "INCONCLUSIVE"
    assert missing["moat_is_real_independence"] is False
    assert exp3868.numeric("not-a-number") is None
    assert exp3868.ci_low("not-a-ci") is None
    assert exp3868.has_live_critical(None) is False
    assert exp3868.summarize_self_learning(None)["outcome"] == "missing_exp3864"
    assert exp3868.summarize_ldt_margin(None)["outcome"] == "missing_exp3865"
    assert exp3868.summarize_ldt_margin(
        {"ensemble_vs_score_matched_margin": 0.012, "margin_ci95": [0.011, 0.014]}
    )["edge"] == "real_at_or_above_0_010"
    assert exp3868.summarize_ldt_margin(
        {"ensemble_vs_score_matched_margin": 0.004, "margin_ci95": [-0.001, 0.006]}
    )["edge"] == "marginal_or_unproven"
    assert exp3868.summarize_hardware(
        {3866: {"gatemate_bitstream_flashed": True, "honest_verdict": "success: ok"}},
        set(),
    ) == {
        "gatemate": {"state": "terminal_flashed", "source_verdict": "success: ok"},
        "polarfire": {"state": "missing_exp3867"},
    }
    assert exp3868.operator_recommendation("MOAT DURABLE").startswith("Treat the verifier moat")
    assert exp3868.operator_recommendation("MOAT SUBSUMED").startswith("Treat the moat as subsumed")


def test_scenario_capstone_3868_complete_fixture_excludes_flagged_and_writes_schema(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-3868: complete aggregation excludes Fabrication Gate inputs."""

    _write_complete_fixture(tmp_path)
    artifact = exp3868.build_artifact(
        tmp_path,
        adversarial_reports=_clean_reports(),
        publication_gate_data=_publication_gate(),
        summary_statuses={experiment_id: {"returncode": 0} for experiment_id in exp3868.UPSTREAM_IDS},
        started_s=1.0,
        now_s=1.00005,
    )

    exp3868.validate_artifact(artifact)
    skipped_ids = {item["experiment_id"] for item in artifact["artifacts_skipped_flagged"]}
    cited_ids = {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}

    assert artifact["honest_verdict"] == (
        "complete: capstone_v356_moat_fake_independence_independence_fake_"
        "facts_new_architecture_opened_paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["moat_durability_verdict"].startswith("MOAT FAKE-INDEPENDENCE")
    assert artifact["moat_is_real_independence"] is False
    assert artifact["thinkprm_complementarity_summary"]["cheap_ensemble_adds_catch"] is True
    assert artifact["facts_new_architecture_outcome"]["outcome"] == "new_architecture_opened"
    assert artifact["ldt_margin_outcome"]["threshold_0_010_held"] is False
    assert artifact["ldt_margin_outcome"]["edge"] == "real_but_below_0_010"
    assert artifact["hardware_board_states"]["gatemate"]["state"] == "excluded_flagged"
    assert artifact["hardware_board_states"]["polarfire"]["state"] == "terminal_hash_verified"
    assert artifact["paper_ready"] is True
    assert artifact["frozen_fover_auroc_unchanged"] is True
    assert skipped_ids == {3862, 3866}
    assert {3862, 3866}.isdisjoint(cited_ids)
    assert artifact["duration_s"] == pytest.approx(0.0001)
    assert exp3868.is_sha256(artifact["reproducibility_checksum"])
    assert set(exp3868.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    output = exp3868.write_artifact(
        tmp_path,
        output_path="results/out.json",
        adversarial_reports=_clean_reports(),
        publication_gate_data=_publication_gate(),
        summary_statuses={experiment_id: {"returncode": 0} for experiment_id in exp3868.UPSTREAM_IDS},
        started_s=1.0,
        now_s=1.2,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3868.validate_artifact(saved)
    assert saved["honest_verdict"] == artifact["honest_verdict"]


def test_req_capstone_3868_missing_audit_and_live_critical_are_inconclusive(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-3868: missing exp3860 blocks a real-moat claim."""

    _write_json(
        tmp_path,
        "data/step_error_balanced_v2.json",
        {"honest_verdict": "complete: corpus", "n_incorrect_steps": 500},
    )
    _write_json(
        tmp_path,
        "results/experiment_3859_moat_scissor_at_scale_v3.json",
        {"honest_verdict": "blocked_gate_check_failed", "status": "blocked"},
    )
    _write_json(
        tmp_path,
        "results/experiment_3863_graph_verifier_facts_complementarity_v2.json",
        {
            "honest_verdict": "blocked_graph_prototype_unavailable",
            "flagged_adversarial": False,
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_3864_fr11_self_learning_v23_independence_reweighting.json",
        {
            "honest_verdict": "complete: fr11_v23",
            "auroc_in_frozen_ci": True,
            "memory_ablation_contribution_preserved": True,
            "reweighted_ensemble_auroc": 0.90592,
            "frozen_headline_ensemble_auroc": 0.9131,
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_3865_ldt_lattice_margin_sharpening_v2.json",
        {
            "honest_verdict": "complete: ldt_margin_LATTICE_REAL",
            "ensemble_vs_score_matched_margin": 0.0056,
            "margin_ci95": [0.0021, 0.0080],
            "frozen_fover_auroc_unchanged": True,
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_3867_polarfire_soc_smoke_v4.json",
        {
            "honest_verdict": "success: polarfire_hash_verified_terminal",
            "polarfire_workload_validated": True,
            "result_hash_match": True,
        },
    )
    reports = _clean_reports()
    reports[3863] = {
        "flags": [
            {
                "severity": "critical",
                "kind": "DURATION_TOO_SHORT",
                "detail": "fixture critical",
            }
        ]
    }

    artifact = exp3868.build_artifact(
        tmp_path,
        adversarial_reports=reports,
        publication_gate_data=_publication_gate(),
        summary_statuses={3858: {"returncode": 0}, 3859: {"returncode": 0}},
        started_s=2.0,
        now_s=2.25,
    )

    exp3868.validate_artifact(artifact)
    assert artifact["moat_durability_verdict"].startswith("INCONCLUSIVE")
    assert artifact["moat_is_real_independence"] is False
    assert artifact["facts_new_architecture_outcome"]["outcome"] == "excluded_flagged_or_live_critical"
    assert artifact["artifacts_skipped_live_critical"][0]["experiment_id"] == 3863
    assert artifact["preconditions_checked"]["upstream_artifacts"][3860]["exists"] is False
    assert artifact["preconditions_checked"]["upstream_artifacts"][3861]["exists"] is False
    assert artifact["honest_verdict"] == (
        "complete: capstone_v356_moat_inconclusive_independence_mixed_"
        "facts_excluded_paper_ready_true_frozen_headline_unchanged"
    )

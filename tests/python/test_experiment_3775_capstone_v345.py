"""Tests for Exp 3775 capstone v345.

Spec refs: REQ-REPORT-3775, SCENARIO-REPORT-3775,
SCENARIO-REPORT-3775-MISSING, SCENARIO-REPORT-3775-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v345_recovery_3775 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_reports() -> dict[int, dict[str, object]]:
    return {experiment_id: {"flags": []} for experiment_id in mod.UPSTREAM_IDS}


def _summary_records() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": experiment_id,
            "returncode": 0,
            "stdout_sha256": f"{experiment_id:064x}"[-64:],
            "stderr_sha256": "0" * 64,
        }
        for experiment_id in mod.UPSTREAM_IDS
    ]


def _payloads(*, g2_auroc_in_ci95: bool = True) -> dict[int, dict[str, object]]:
    return {
        3765: {
            "honest_verdict": (
                "complete: archived_v344_zero_experiments_skip_cascade_recorded_"
                "v345_recovery_active_paper_ready_true_frozen_headline_unchanged"
            ),
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "unlanded_v344_agenda_carried_to_v345": ["exp3754", "exp3755"],
            "paper_ready_preserved": True,
            "paper_ready_evidence": {
                "g1": True,
                "g2": True,
                "g3": True,
                "g4": True,
                "paper_ready": True,
                "frozen_headline_unchanged": True,
                "frozen_headline_auroc": 0.9131,
            },
            "random_seed": 3765,
            "reproducibility_checksum": "5" * 64,
            "duration_s": 0.1,
        },
        3766: {
            "honest_verdict": (
                "complete: thesis_a_definitive_reconciled_part_a_PASS_discriminative_"
                "part_b_BOUNDED_not_generative_in_loop_chain_superseded_menu_updated_not_retired"
            ),
            "thesis_a_part_a_outcome": "PASS_discriminative: held-out margin learned",
            "thesis_a_part_b_outcome": "BOUNDED_at_scale_not_generative: matched AR wins",
            "ebt_discriminative_not_generative": True,
            "in_loop_chain_superseded": True,
            "thesis_menu_updated": True,
            "not_added_to_exclusion_manifest": True,
            "random_seed": 3766,
            "reproducibility_checksum": "6" * 64,
            "duration_s": 0.1,
        },
        3767: {
            "honest_verdict": (
                "complete: g2_local_reproducer_committed_auroc_0.913134_"
                f"in_ci95_{str(g2_auroc_in_ci95).lower()}_frozen_headline_confirmed_unchanged"
            ),
            "auroc_in_ci95": g2_auroc_in_ci95,
            "frozen_headline_unchanged": True,
            "reproduced_auroc_mean": 0.913134,
            "source_headline": {
                "headline_matches_frozen_0_9131": True,
                "condition_a_production_auroc_mean": 0.9131336,
            },
            "random_seed": 3767,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 0.1,
        },
        3768: {
            "honest_verdict": (
                "complete: g3_narrowing_lint_shipped_19_phrasings_"
                "12th_added_violations_0_precommit_wired"
            ),
            "lint_extended_and_wired": True,
            "paper_v6_json_scan_extended": True,
            "precommit_hook_wired": True,
            "twelfth_retraction_added": True,
            "violations_found": 0,
            "random_seed": 3768,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 0.1,
        },
        3769: {
            "honest_verdict": (
                "complete: phase1_e2e_smoke_package_import_pipeline_"
                "mcp_protocol_cli_passed_wiring_smoke_not_accuracy_claim"
            ),
            "package_importable": True,
            "pipeline_e2e_passed": True,
            "cli_passed": True,
            "mcp_protocol_exchange_passed": True,
            "surfaces_passed": ["package_import", "pipeline", "mcp_protocol", "cli"],
            "is_wiring_smoke_not_accuracy_claim": True,
            "random_seed": 3769,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 0.1,
        },
        3770: {
            "honest_verdict": (
                "complete: distribution_mirror_readiness_audited_pypi_true_hf_true_"
                "ipfs_true_operator_checklist_emitted_agent_published_nothing"
            ),
            "pypi_workflow_ready": True,
            "hf_mirror_documented": True,
            "ipfs_plan_documented": True,
            "operator_publish_checklist": [{"operator_only": "OPERATOR ACTION -- agent must not execute"}],
            "agent_published_nothing": True,
            "random_seed": 3770,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 0.1,
        },
        3771: {
            "honest_verdict": (
                "complete: certified_abstention_point_threshold_0.733216_"
                "coverage_0.998218_at_risk_0.05_certified_split_conformal_delta_0.05_n2619"
            ),
            "usable_operating_point_exists": True,
            "selected_threshold": 0.733216,
            "coverage_at_operating_point": 0.998218,
            "certified_risk_bound": 0.037646,
            "risk_target": 0.05,
            "random_seed": 3771,
            "reproducibility_checksum": "b" * 64,
            "duration_s": 0.1,
        },
        3772: {
            "honest_verdict": (
                "complete: fr11_v17_tier1_verifier_precision_tracker_pivoted_to_"
                "live_verifier_memory_contribution_preserved_state_persisted"
            ),
            "continuous_self_learning_task": True,
            "memory_contribution_preserved": True,
            "tracker_state_persisted": True,
            "pivoted_off_dead_ebt_lineage": True,
            "acceptance_gate": {"passed": True},
            "random_seed": 3772,
            "reproducibility_checksum": "c" * 64,
            "duration_s": 0.1,
        },
        3773: {
            "honest_verdict": (
                "complete: verifier_product_positioned_vs_prm_sota_leads_cost_"
                "objectivity_certifiability_does_not_lead_f1_or_ood_no_generalization_retest"
            ),
            "peer_numbers_are_as_reported_not_re_derived": True,
            "no_generalization_retest_run": True,
            "where_carnot_leads": "leads on cost, objectivity, and certifiability",
            "where_carnot_does_not_lead": "does NOT lead on raw F1 or OOD generalization",
            "product_value_proposition": "certified complement to generative PRMs",
            "random_seed": 3773,
            "reproducibility_checksum": "d" * 64,
            "duration_s": 0.1,
        },
        3774: {
            "honest_verdict": (
                "complete: kv260_terminal_state_holds_ssh_reachable_"
                "accelerator_loadable_opportunistic_audit"
            ),
            "terminal_state_holds": True,
            "kv260_ssh_reachable": True,
            "kv260_overlay_loadable": True,
            "speedup_claim_made": False,
            "random_seed": 3774,
            "reproducibility_checksum": "e" * 64,
            "duration_s": 0.1,
        },
    }


def _operator_surface() -> dict[str, object]:
    return {
        "honest_verdict": (
            "complete: next_phase3_thesis_menu_ranked_top_edlm_residual_corrector_"
            "supersedes_340_menu_all_routes_sidestep_both_negatives_for_operator_seeding"
        ),
        "loop_will_not_self_seed": True,
        "supersedes_340_menu": True,
        "ranked_thesis_menu": [{"route": "EDLM"}],
    }


def _seed_clean_upstreams(
    root: Path,
    *,
    g2_auroc_in_ci95: bool = True,
    flagged: set[int] | None = None,
    missing: set[int] | None = None,
) -> None:
    flagged = flagged or set()
    missing = missing or set()
    for experiment_id, payload in _payloads(g2_auroc_in_ci95=g2_auroc_in_ci95).items():
        if experiment_id in missing:
            continue
        if experiment_id in flagged:
            payload["flagged_adversarial"] = True
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)
    _write_json(root / mod.OPERATOR_SURFACE_REL_PATH, _operator_surface())


def test_req_report_3775_spec_anchor_exists() -> None:
    """REQ-REPORT-3775: OpenSpec declares the v345 capstone contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3775" in spec
    assert "SCENARIO-REPORT-3775" in spec
    assert "SCENARIO-REPORT-3775-MISSING" in spec
    assert "SCENARIO-REPORT-3775-FLAGGED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3775_clean_capstone_rebanks_product(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3775: clean upstreams produce recovery/convergence capstone."""
    _seed_clean_upstreams(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        capstone_adversarial_verify_clean=True,
        started_s=2.0,
        now_s=2.5,
    )

    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: capstone_v345_skip_cascade_recovered_thesis_a_closed_"
        "both_energy_routes_bounded_gates_mechanized_verifier_banked_"
        "abstention_point_shipped_fr11_v17_prm_positioned_paper_ready_true_"
        "frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["v344_skip_cascade_recovered"] is True
    assert artifact["thesis_a_definitively_closed"] is True
    assert artifact["both_energy_routes_bounded"] is True
    assert artifact["gates_mechanized"] is True
    assert artifact["verifier_banked_for_ship"] is True
    assert artifact["certified_abstention_point_status"] == "shipped"
    assert artifact["verifier_positioned_vs_prm_sota"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["next_thesis_remains_operator_surface"] is True
    assert artifact["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["energy_as_selector_status"] == "honest-negative-bounded"
    assert artifact["energy_as_generator_status"] == "honest-negative-bounded"
    assert artifact["no_new_existential_claim"] is True
    assert artifact["fr11_v17_memory_contribution_preserved"] is True
    assert artifact["kv260_terminal_confirmed"] is True
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["not_landed_artifacts_recorded_honestly"] == []
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    assert artifact["supporting_operator_surface_artifact"]["experiment_id"] == 3763
    assert {item["experiment_id"] for item in artifact["summarized_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    blob = json.dumps(artifact, sort_keys=True)
    assert "live_llm_inference" not in blob
    assert "model_specs" not in blob
    assert "target_model" not in blob
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_3775_abstention_skips_without_g2_ci_pass(tmp_path: Path) -> None:
    """REQ-REPORT-3775: abstention ships only if Exp 3767 reproduces the headline."""
    _seed_clean_upstreams(tmp_path, g2_auroc_in_ci95=False)

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        capstone_adversarial_verify_clean=True,
        started_s=3.0,
        now_s=3.25,
    )

    assert artifact["certified_abstention_point_status"] == "skipped"
    assert artifact["gates_mechanized"] is False
    assert artifact["honest_verdict"] == (
        "complete: capstone_v345_skip_cascade_recovered_thesis_a_closed_"
        "both_energy_routes_bounded_gates_not_mechanized_verifier_banked_"
        "abstention_point_skipped_fr11_v17_prm_positioned_paper_ready_true_"
        "frozen_headline_unchanged"
    )


def test_scenario_report_3775_flagged_upstream_is_excluded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3775-FLAGGED: flagged artifacts are quarantined."""
    _seed_clean_upstreams(tmp_path, flagged={3770})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        capstone_adversarial_verify_clean=True,
        started_s=4.0,
        now_s=4.25,
    )

    assert artifact["verifier_banked_for_ship"] is False
    assert artifact["flagged_artifacts_excluded"] == [
        {
            "experiment_id": 3770,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3770]),
            "reason": "flagged_adversarial=true",
        }
    ]
    assert 3770 not in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    assert 3770 not in artifact["headline_aggregation_experiment_ids"]


def test_scenario_report_3775_missing_upstream_is_not_research_negative(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3775-MISSING: missing source is recorded as not-landed."""
    _seed_clean_upstreams(tmp_path, missing={3766})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        capstone_adversarial_verify_clean=True,
        started_s=5.0,
        now_s=5.25,
    )

    assert artifact["thesis_a_definitively_closed"] is False
    assert artifact["both_energy_routes_bounded"] is False
    assert artifact["not_landed_artifacts_recorded_honestly"] == [
        {
            "experiment_id": 3766,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3766]),
            "status": "not-landed",
            "reason": "artifact_missing",
        }
    ]
    assert "not-landed" in artifact["thesis_a_status_note"]
    assert "not a research negative" in artifact["thesis_a_status_note"]


def test_validate_artifact_reports_schema_hygiene_and_checksum_errors() -> None:
    """REQ-REPORT-3775: malformed capstones fail closed before reporting."""
    errors = mod.validate_artifact({})

    assert any(error.startswith("missing required artifact fields:") for error in errors)
    assert "honest_verdict must be a terminal Exp 3775 verdict" in errors
    assert "inference_substrate must declare the v345 aggregation-only substrate" in errors
    assert "certified_abstention_point_status must be shipped or skipped" in errors
    assert "paper_ready_preserved must be true" in errors
    assert "frozen_headline_unchanged must be true" in errors
    assert "cited_upstream_artifacts must be a list" in errors

    valid = {
        "honest_verdict": mod.terminal_verdict(
            skip_recovered=True,
            thesis_closed=True,
            both_energy_bounded=True,
            gates_mechanized=True,
            verifier_banked=True,
            abstention_status="shipped",
            fr11_v17=True,
            prm_positioned=True,
            paper_ready=True,
            frozen=True,
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "v344_skip_cascade_recovered": True,
        "thesis_a_definitively_closed": True,
        "both_energy_routes_bounded": True,
        "gates_mechanized": True,
        "verifier_banked_for_ship": True,
        "certified_abstention_point_status": "shipped",
        "verifier_positioned_vs_prm_sota": True,
        "paper_ready_preserved": True,
        "frozen_headline_unchanged": True,
        "next_thesis_remains_operator_surface": True,
        "flagged_artifacts_excluded": [],
        "not_landed_artifacts_recorded_honestly": [],
        "cited_upstream_artifacts": [
            {
                "experiment_id": 3765,
                "path": "results/experiment_3765.json",
                "fields_imported": ["honest_verdict"],
                "sha256": "1" * 64,
            }
        ],
        "field_principles": dict(mod.FIELD_PRINCIPLES),
        "random_seed": mod.RANDOM_SEED,
        "duration_s": 0.1,
        "adversarial_verify_report": {"flags": []},
        "reproducibility_checksum": "",
    }
    valid["reproducibility_checksum"] = mod.payload_checksum(valid)
    assert mod.validate_artifact(valid) == []

    live_marker = dict(valid)
    live_marker["copied_substrate"] = "live_llm_inference"
    live_marker["reproducibility_checksum"] = mod.payload_checksum(live_marker)
    assert "artifact must not copy live-model substrate markers" in mod.validate_artifact(
        live_marker
    )

    bad_checksum = dict(valid)
    bad_checksum["reproducibility_checksum"] = "2" * 64
    assert "reproducibility_checksum does not match artifact content" in mod.validate_artifact(
        bad_checksum
    )

    bad_citation = dict(valid)
    bad_citation["cited_upstream_artifacts"] = [123, {"experiment_id": 3765}]
    bad_citation["reproducibility_checksum"] = mod.payload_checksum(bad_citation)
    citation_errors = mod.validate_artifact(bad_citation)
    assert "each citation must be an object" in citation_errors
    assert "each citation must include fields_imported" in citation_errors

    critical_report = dict(valid)
    critical_report["adversarial_verify_report"] = {
        "flags": [{"severity": "critical", "kind": "TEST", "detail": "blocked"}]
    }
    critical_report["reproducibility_checksum"] = mod.payload_checksum(critical_report)
    assert "adversarial verifier must report no critical flag" in mod.validate_artifact(
        critical_report
    )


def test_run_writes_artifact_and_rejects_array_sources(tmp_path: Path) -> None:
    """REQ-REPORT-3775: CLI path writes stable JSON and source JSON objects only."""
    _seed_clean_upstreams(tmp_path)

    out_path = mod.run(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        started_s=6.0,
        now_s=6.25,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["flags"] == []
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)

    array_path = tmp_path / "array.json"
    array_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod.read_json_object(array_path)


def test_fallback_and_error_branches_are_honest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3775: fallback branches report missing or blocked state honestly."""
    _seed_clean_upstreams(tmp_path)

    bad_root = tmp_path / "bad_paper_ready"
    _seed_clean_upstreams(bad_root)
    bad_3765 = _payloads()[3765]
    bad_3765["paper_ready_preserved"] = False
    _write_json(bad_root / mod.DEFAULT_UPSTREAM_PATHS[3765], bad_3765)
    with pytest.raises(ValueError, match="paper_ready_preserved must be true"):
        mod.build_artifact(
            bad_root,
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            started_s=1.0,
            now_s=1.25,
        )

    alternate = tmp_path / "alternate" / "results" / "experiment_3765_alternate_name.json"
    _write_json(alternate, {"honest_verdict": "complete: alternate"})
    assert mod.resolve_upstream_path(tmp_path / "alternate", 3765) == alternate
    assert mod.resolve_upstream_path(tmp_path / "missing", 3765) == (
        tmp_path / "missing" / mod.DEFAULT_UPSTREAM_PATHS[3765]
    )

    assert mod.certified_abstention_status({"auroc_in_ci95": False}, {"usable_operating_point_exists": True}) == "skipped"
    assert mod.certified_abstention_status(_payloads()[3767], {}) == "skipped"
    assert mod.next_thesis_operator_surface(None) is False
    assert mod.publication_gate_state({}) == {
        "paper_ready": False,
        "g1": False,
        "g2": False,
        "g3": False,
        "g4": False,
    }
    assert mod.supporting_operator_surface(tmp_path / "no_operator_surface", None)["status"] == "not-landed"
    assert mod.critical_adversarial_flags(
        {3765: {"flags": [123, {"severity": "critical", "kind": "X", "detail": "bad"}]}}
    ) == [{"experiment_id": 3765, "kind": "X", "detail": "bad"}]
    assert mod.report_is_clean(None) is True
    assert mod.report_is_clean({"flags": [123, {"severity": "critical"}]}) is False
    assert mod.numeric(True) is None
    assert mod.numeric("0.9131") is None

    def _critical_report(path: Path) -> dict[str, object]:
        assert path.name == mod.OUTPUT_REL_PATH.name
        return {"flags": [{"severity": "critical", "kind": "TEST", "detail": "forced"}]}

    monkeypatch.setattr(mod.adversarial_verify, "verify_artifact", _critical_report)
    with pytest.raises(ValueError, match="adversarial verifier must report no critical flag"):
        mod.run(
            tmp_path,
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            started_s=7.0,
            now_s=7.25,
        )

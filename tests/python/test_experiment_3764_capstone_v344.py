"""Tests for Exp 3764 capstone v344.

Spec refs: REQ-REPORT-3764, SCENARIO-REPORT-3764,
SCENARIO-REPORT-3764-GATED, SCENARIO-REPORT-3764-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v344_thesis_a_closed_3764 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_data(*, paper_ready: bool = True) -> dict[str, object]:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": paper_ready, "detail": "headline measured"},
            "G2": {"pass": paper_ready, "detail": "independent reproducer"},
            "G3": {"pass": paper_ready, "detail": "narrowing clean"},
            "G4": {"pass": paper_ready, "detail": "numbers traced"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


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


def _seed_clean_upstreams(
    root: Path,
    *,
    g2_auroc_in_ci95: bool = True,
    flagged: set[int] | None = None,
) -> None:
    flagged = flagged or set()
    payloads: dict[int, dict[str, object]] = {
        3754: {
            "honest_verdict": "complete: archived_v343_activated_v344_paper_ready_true_frozen_headline_unchanged",
            "paper_ready_preserved": True,
            "g1": True,
            "g2": True,
            "g3": True,
            "g4": True,
            "frozen_fover_auroc": 0.9131,
            "random_seed": 3754,
            "reproducibility_checksum": "4" * 64,
            "duration_s": 0.1,
        },
        3755: {
            "honest_verdict": "complete: thesis_a_definitively_closed_part_a_pass_part_b_bounded_at_scale_in_loop_chain_superseded",
            "thesis_a_definitively_closed": True,
            "part_a_pass_discriminative": True,
            "part_b_bounded_at_scale_not_generative": True,
            "in_loop_chain_superseded": True,
            "both_energy_routes_bounded": True,
            "energy_as_selector_status": "honest-negative-bounded",
            "energy_as_generator_status": "bounded-at-scale-discriminative-not-generative",
            "random_seed": 3755,
            "reproducibility_checksum": "5" * 64,
            "duration_s": 0.1,
        },
        3756: {
            "honest_verdict": "complete: g2_local_reproducer_fover_0_9131_auroc_in_ci95",
            "g2_local_reproducer_shipped": True,
            "auroc_in_ci95": g2_auroc_in_ci95,
            "reproduced_auroc": 0.9131,
            "random_seed": 3756,
            "reproducibility_checksum": "6" * 64,
            "duration_s": 0.1,
        },
        3757: {
            "honest_verdict": "complete: g3_narrowing_lint_shipped_current_paper_clean_energy_generator_phrase_guarded",
            "g3_narrowing_lint_shipped": True,
            "current_paper_lint_clean": True,
            "energy_as_generator_forbidden_phrase_guard": True,
            "random_seed": 3757,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 0.1,
        },
        3758: {
            "honest_verdict": "complete: package_cli_mcp_e2e_smoke_passed",
            "package_e2e_smoke_passed": True,
            "cli_e2e_smoke_passed": True,
            "mcp_e2e_smoke_passed": True,
            "random_seed": 3758,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 0.1,
        },
        3759: {
            "honest_verdict": "complete: distribution_mirror_ready_operator_publish_checklist_ready_agent_published_nothing",
            "distribution_mirror_ready": True,
            "operator_publish_checklist_ready": True,
            "agent_published_nothing": True,
            "random_seed": 3759,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 0.1,
        },
        3760: {
            "honest_verdict": "complete: certified_abstention_operating_point_shipped",
            "certified_abstention_point_status": "shipped",
            "deployable_operating_point_selected": True,
            "exp3756_auroc_in_ci95": True,
            "random_seed": 3760,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 0.1,
        },
        3761: {
            "honest_verdict": "complete: fr11_v17_pivoted_to_live_verifier_memory_contribution_preserved",
            "fr11_v17_pivoted_to_live_verifier": True,
            "memory_contribution_preserved": True,
            "memory_contribution_delta": 0.0185,
            "random_seed": 3761,
            "reproducibility_checksum": "b" * 64,
            "duration_s": 0.1,
        },
        3762: {
            "honest_verdict": "complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit",
            "terminal_state_holds": True,
            "kv260_ssh_reachable": True,
            "kv260_overlay_loadable": True,
            "speedup_claim_made": False,
            "random_seed": 3762,
            "reproducibility_checksum": "c" * 64,
            "duration_s": 0.1,
        },
        3763: {
            "honest_verdict": "complete: next_phase3_thesis_menu_ranked_top_edlm_residual_corrector_supersedes_340_menu_all_routes_sidestep_both_negatives_for_operator_seeding",
            "loop_will_not_self_seed": True,
            "supersedes_340_menu": True,
            "ranked_thesis_menu": [{"route": "EDLM"}],
            "random_seed": 3763,
            "reproducibility_checksum": "d" * 64,
            "duration_s": 0.1,
        },
    }
    for experiment_id, payload in payloads.items():
        if experiment_id in flagged:
            payload["flagged_adversarial"] = True
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_report_3764_spec_anchor_exists() -> None:
    """REQ-REPORT-3764: OpenSpec declares the v344 capstone contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3764" in spec
    assert "SCENARIO-REPORT-3764" in spec
    assert "SCENARIO-REPORT-3764-GATED" in spec
    assert "SCENARIO-REPORT-3764-FLAGGED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3764_clean_capstone_banks_product_without_existential_claim(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3764: clean upstreams produce the intended capstone."""
    _seed_clean_upstreams(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
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
        "complete: capstone_v344_thesis_a_closed_both_energy_routes_bounded_"
        "gates_mechanized_verifier_banked_abstention_point_shipped_"
        "fr11_pivoted_next_thesis_to_operator_paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["thesis_a_definitively_closed"] is True
    assert artifact["both_energy_routes_bounded"] is True
    assert artifact["gates_mechanized"] is True
    assert artifact["verifier_banked_for_ship"] is True
    assert artifact["certified_abstention_point_status"] == "shipped"
    assert artifact["paper_ready_preserved"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["next_thesis_handed_to_operator"] is True
    assert artifact["no_new_existential_claim"] is True
    assert artifact["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["energy_as_selector_status"] == "honest-negative-bounded"
    assert artifact["energy_as_generator_status"] == "bounded-at-scale-discriminative-not-generative"
    assert artifact["fr11_v17_pivoted_to_live_verifier"] is True
    assert artifact["kv260_terminal_confirmed"] is True
    assert artifact["flagged_artifacts_excluded"] == []
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    assert {item["experiment_id"] for item in artifact["summarized_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    assert "no live model" in artifact["inference_substrate"]
    assert "live_llm_inference" not in json.dumps(artifact, sort_keys=True)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_3764_gated_abstention_skips_without_g2_ci_pass(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3764-GATED: Exp 3760 is skipped unless Exp 3756 passes."""
    _seed_clean_upstreams(tmp_path, g2_auroc_in_ci95=False)

    artifact = mod.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        capstone_adversarial_verify_clean=True,
        started_s=3.0,
        now_s=3.25,
    )

    assert artifact["certified_abstention_point_status"] == "skipped"
    assert artifact["gates_mechanized"] is False
    assert artifact["honest_verdict"] == (
        "complete: capstone_v344_thesis_a_closed_both_energy_routes_bounded_"
        "gates_not_mechanized_verifier_banked_abstention_point_skipped_"
        "fr11_pivoted_next_thesis_to_operator_paper_ready_true_frozen_headline_unchanged"
    )


def test_scenario_report_3764_flagged_upstream_is_excluded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3764-FLAGGED: flagged artifacts are quarantined."""
    _seed_clean_upstreams(tmp_path, flagged={3759})

    artifact = mod.build_artifact(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        capstone_adversarial_verify_clean=True,
        started_s=4.0,
        now_s=4.25,
    )

    assert artifact["verifier_banked_for_ship"] is False
    assert artifact["flagged_artifacts_excluded"] == [
        {
            "experiment_id": 3759,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3759]),
            "reason": "flagged_adversarial=true",
        }
    ]
    assert 3759 not in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    assert 3759 not in artifact["headline_aggregation_experiment_ids"]


def test_validate_artifact_reports_schema_hygiene_and_checksum_errors() -> None:
    """REQ-REPORT-3764: malformed capstones fail closed before reporting."""
    errors = mod.validate_artifact({})

    assert any(error.startswith("missing required artifact fields:") for error in errors)
    assert "honest_verdict must be a terminal Exp 3764 verdict" in errors
    assert "inference_substrate must declare the v344 aggregation-only substrate" in errors
    assert "certified_abstention_point_status must be shipped or skipped" in errors
    assert "paper_ready_preserved must be true" in errors
    assert "frozen_headline_unchanged must be true" in errors
    assert "cited_upstream_artifacts must be a list" in errors

    valid = {
        "honest_verdict": mod.terminal_verdict(
            gates_mechanized=True,
            verifier_banked=True,
            abstention_status="shipped",
            fr11_pivoted=True,
            next_thesis=True,
            paper_ready=True,
            frozen=True,
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "thesis_a_definitively_closed": True,
        "both_energy_routes_bounded": True,
        "gates_mechanized": True,
        "verifier_banked_for_ship": True,
        "certified_abstention_point_status": "shipped",
        "paper_ready_preserved": True,
        "frozen_headline_unchanged": True,
        "next_thesis_handed_to_operator": True,
        "flagged_artifacts_excluded": [],
        "cited_upstream_artifacts": [
            {
                "experiment_id": 3754,
                "path": "results/experiment_3754.json",
                "fields_imported": ["honest_verdict"],
                "sha256": "1" * 64,
            }
        ],
        "field_principles": dict(mod.FIELD_PRINCIPLES),
        "random_seed": mod.RANDOM_SEED,
        "duration_s": 0.1,
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
    bad_citation["cited_upstream_artifacts"] = [123, {"experiment_id": 3754}]
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
    """REQ-REPORT-3764: CLI path writes stable JSON and source JSON objects only."""
    _seed_clean_upstreams(tmp_path)

    out_path = mod.run(
        tmp_path,
        gate_data=_gate_data(),
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        started_s=5.0,
        now_s=5.25,
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


def test_fallback_and_error_branches_are_honest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-3764: fallback branches report missing/blocked state honestly."""
    _seed_clean_upstreams(tmp_path)

    with pytest.raises(ValueError, match="paper_ready_preserved must be true"):
        mod.build_artifact(
            tmp_path,
            gate_data=_gate_data(paper_ready=False),
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            capstone_adversarial_verify_clean=True,
            started_s=1.0,
            now_s=1.25,
        )

    alternate = tmp_path / "alternate" / "results" / "experiment_3754_alternate_name.json"
    _write_json(alternate, {"honest_verdict": "complete: alternate"})
    assert mod.resolve_upstream_path(tmp_path / "alternate", 3754) == alternate
    assert mod.resolve_upstream_path(tmp_path / "missing", 3754) == (
        tmp_path / "missing" / mod.DEFAULT_UPSTREAM_PATHS[3754]
    )

    assert mod.certified_abstention_status({"g2_local_reproducer_shipped": True, "auroc_in_ci95": True}, {}) == "skipped"
    assert mod.certified_abstention_status(
        {"g2_local_reproducer_shipped": True, "auroc_in_ci95": True},
        {"certified_abstention_point_status": "gate_skipped"},
    ) == "skipped"
    assert mod.frozen_headline_unchanged(_gate_data(), {}) is True
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
            gate_data=_gate_data(),
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            started_s=5.0,
            now_s=5.25,
        )

"""Tests for the Exp 1439 `.111` carry-forward activation manifest.

Spec: REQ-REPORT-036, SCENARIO-REPORT-036.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_111_carryforward_activation_manifest import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _read_json,
    _read_text,
    _relative_path,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete(verdict: str, **fields: object) -> dict[str, object]:
    payload: dict[str, object] = {"status": "complete", "honest_verdict": verdict}
    payload.update(fields)
    return payload


def _retro_payload() -> dict[str, object]:
    def prior(experiment_id: str, verdict: str, retire: bool = True) -> dict[str, object]:
        return {
            "experiment_id": experiment_id,
            "evidence_path": f"results/{experiment_id}.json",
            "verdict": verdict,
            "retire_if_same_verdict": retire,
        }

    return {
        "status": "complete",
        "milestone": "2026.04.110",
        "criteria_met": 12,
        "criteria_total": 14,
        "honest_verdict": (
            "milestone_110_12_of_14_criteria_met_threshold_met_repair_dvi_prm_dpo_"
            "latent_positive_fr11_growth_and_rtl_source_carry_forward"
        ),
        "carry_forward_tasks": [
            {
                "id": "repair_v2_live_sota_headline_scaleup",
                "title": "Convert repair v2 prototype wins into live-SOTA evidence",
                "prior_failures": [
                    prior(
                        "exp1428",
                        "retro_shorthand_repair_v2_prototype",
                    ),
                    prior(
                        "exp1431",
                        "retro_shorthand_pipeline_micro_prototype",
                    ),
                ],
                "retire_if_same_verdict": True,
            },
            {
                "id": "fr11_positive_growth_followup",
                "title": "Diagnose deployed-DVI FR-11 zero growth",
                "prior_failures": [
                    prior("exp1433", "fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline")
                ],
                "retire_if_same_verdict": True,
            },
            {
                "id": "test_debt_spec_coverage_cluster",
                "title": "Fix prioritized spec-coverage traceability metadata",
                "prior_failures": [
                    prior(
                        "exp1426",
                        "diagnostic_cluster_map_complete_collection_clean_spec_coverage_red",
                        retire=False,
                    )
                ],
                "retire_if_same_verdict": False,
            },
            {
                "id": "dpo_adapter_or_reranker_only",
                "title": "Keep DPO reranker-only without local adapter tooling",
                "prior_failures": [
                    prior(
                        "exp1435",
                        "dpo_headline_not_ready_reranker_only_until_adapter_or_conversion_tooling",
                    )
                ],
                "retire_if_same_verdict": True,
            },
            {
                "id": "hardware_rtl_source_before_lint_sim",
                "title": "Implement missing Discrete SB RTL source",
                "prior_failures": [prior("exp1437", "blocked_missing_discrete_sb_rtl_source")],
                "retire_if_same_verdict": True,
            },
        ],
        "prm_verdict": {
            "summary": "prm_labels_completed_and_selector_non_degrading_but_selector_no_improvement",
            "evidence_paths": ["results/experiment_1430_prm_guided_repair_selector.json"],
        },
    }


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1426": _complete(
            "diagnostic_cluster_map_complete_collection_clean_spec_coverage_red_71_full_suite_"
            "not_rerun_exp1421_runtime_debt_partitioned",
            failure_cluster_map_complete=True,
            spec_coverage_debt_count=71,
            next_cluster_recommended="spec_coverage_traceability_metadata",
        ),
        "exp1428": _complete(
            "complete_dccd_schema_constrained_repair_v2_nonzero_repairs_prototype_no_headline_"
            "sota_claim",
            repaired_cases_successful=20,
            repaired_case_success_rate=1.0,
            local_sota_model_inference_used=False,
        ),
        "exp1430": _complete(
            "complete_prm_guided_selector_no_improvement_prototype_candidate_pool_no_headline_"
            "claim",
            prm_guided_selection_ready=True,
            selected_repair_success_rate=1.0,
            raw_best_of_n_repair_success_rate=1.0,
            selection_improvement_pp=0.0,
        ),
        "exp1431": _complete(
            "complete_micro_validation_beats_exp1419_baseline_prototype_no_headline_scaleup",
            cases_evaluated=50,
            full_pipeline_pass_rate=0.62,
            beats_exp1419_baseline=True,
            runtime_evidence_allows_headline_scaleup=False,
        ),
        "exp1433": _complete(
            "fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline",
            v6_new_promoted_count=0,
            self_learning_delta_overall=0,
            headline_result_allowed=False,
            dvi_v3_checkpoint_active=True,
        ),
        "exp1435": _complete(
            "dpo_headline_not_ready_reranker_only_until_adapter_or_conversion_tooling",
            headline_provenance_ready=False,
            reranker_track_relabelled=True,
            direct_gguf_finetune_supported=False,
        ),
        "exp1437": _complete(
            "blocked_missing_discrete_sb_rtl_source",
            status="blocked",
            rtl_lint_complete=False,
            simulation_complete=False,
            hardware_claim_allowed=False,
            rtl_sources_checked=[{"exists": False, "path": "hardware/kv260/discrete_sb_256.v"}],
        ),
    }


def test_scenario_report_036_maps_all_110_unresolved_tracks() -> None:
    """SCENARIO-REPORT-036: every unresolved .110 track receives .111 activation."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        sources=_source_payloads(),
        manifest_path="ops/milestone_111_carryforward_manifest.md",
        roadmap_text="exp1440 exp1441 exp1442 exp1443 exp1444 exp1445 exp1446 exp1447 exp1448 exp1451",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["prior_milestone"] == "2026.04.110"
    assert artifact["carryforward_manifest_complete"] is True
    assert artifact["carryforward_task_count"] == 6
    assert {row["track_id"] for row in artifact["manifest_rows"]} == {
        "repair_v2_live_sota_headline_scaleup",
        "fr11_positive_growth_followup",
        "test_debt_spec_coverage_cluster",
        "dpo_adapter_or_reranker_only",
        "hardware_rtl_source_before_lint_sim",
        "prm_selector_no_improvement",
    }
    assert {item["forbidden_scope_id"] for item in artifact["forbidden_exact_reruns"]} == {
        "prototype_repair_scaleup",
        "fr11_zero_growth",
        "prm_v1_no_improvement",
        "missing_source_rtl_lint_sim",
    }
    assert "NON-HEADLINE RETIREMENT" in manifest
    assert "exp1448" in manifest


def test_req_report_036_preserves_exact_source_verdicts() -> None:
    """REQ-REPORT-036: same-verdict rules preserve source artifact verdicts."""

    artifact, _manifest = build_artifact(
        retro=_retro_payload(),
        sources=_source_payloads(),
        manifest_path="ops/milestone_111_carryforward_manifest.md",
        roadmap_text="",
    )

    by_experiment = {
        rule["experiment_id"]: rule["prior_verdict"]
        for rule in artifact["same_verdict_retirement_rules"]
    }

    assert by_experiment["exp1428"] == (
        "complete_dccd_schema_constrained_repair_v2_nonzero_repairs_prototype_no_headline_"
        "sota_claim"
    )
    assert by_experiment["exp1431"] == (
        "complete_micro_validation_beats_exp1419_baseline_prototype_no_headline_scaleup"
    )
    assert by_experiment["exp1430"] == (
        "complete_prm_guided_selector_no_improvement_prototype_candidate_pool_no_headline_claim"
    )


def test_req_report_036_unknown_track_prevents_completion() -> None:
    """REQ-REPORT-036: unmapped tracks cannot claim activation success."""

    retro = _retro_payload()
    retro["carry_forward_tasks"] = list(retro["carry_forward_tasks"]) + [
        {"id": "unexpected-track", "title": "Unexpected track", "prior_failures": []}
    ]

    artifact, manifest = build_artifact(
        retro=retro,
        sources=_source_payloads(),
        manifest_path="ops/milestone_111_carryforward_manifest.md",
        roadmap_text="",
    )

    assert artifact["status"] == "blocked"
    assert artifact["carryforward_manifest_complete"] is False
    assert artifact["unmapped_tracks"] == ["unexpected-track"]
    assert "unexpected-track" in manifest


def test_req_report_036_run_writes_manifest_and_final_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-036: run writes bootstrap, manifest, and terminal JSON."""

    out_path = tmp_path / "results" / "experiment_1439_110_carryforward_activation_manifest.json"
    manifest_path = tmp_path / "ops" / "milestone_111_carryforward_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1438_milestone_110_retro.json", _retro_payload())
    for source_id, payload in _source_payloads().items():
        _write_json(tmp_path / "results" / SOURCE_FILES[source_id], payload)
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "exp1440 exp1441 exp1442 exp1443 exp1444 exp1445 exp1446 exp1447 exp1448 exp1451",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text("exp1439 active roadmap", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "research_conductor.py").write_text("# unchanged\n", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, manifest_path=manifest_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["carryforward_manifest_path"] == "ops/milestone_111_carryforward_manifest.md"
    assert written["no_change_confirmations"] == {
        "scripts/research_conductor.py": "no activation-manifest changes needed",
        "research-roadmap.yaml": "no activation-manifest changes needed",
    }
    assert (
        "| track | prior evidence | .111 task | gate rule | retire-if-same-verdict rule |"
        in manifest
    )
    assert "exp1451" in manifest
    assert "missing-source RTL lint/sim" in manifest


def test_req_report_036_defensive_missing_and_retro_only_branches(tmp_path: Path) -> None:
    """REQ-REPORT-036: missing files and retro-only priors stay explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"

    artifact, manifest = build_artifact(
        retro={
            "milestone": "2026.04.110",
            "carry_forward_tasks": [
                {
                    "id": "retro-only-track",
                    "title": "Retro-only Track",
                    "prior_failures": [
                        "malformed-prior-entry",
                        {
                            "experiment_id": "exp999-retro-only",
                            "verdict": "retro_only_verdict",
                            "retire_if_same_verdict": True,
                        },
                    ],
                }
            ],
        },
        sources={},
        manifest_path="ops/milestone_111_carryforward_manifest.md",
        roadmap_text="",
    )

    assert artifact["forbidden_exact_reruns"] == []
    assert artifact["same_verdict_retirement_rules"][0]["source_artifact"] is None
    assert artifact["same_verdict_retirement_rules"][0]["prior_verdict"] == "retro_only_verdict"
    assert "retro-only evidence" in manifest
    assert "- None recorded." in manifest

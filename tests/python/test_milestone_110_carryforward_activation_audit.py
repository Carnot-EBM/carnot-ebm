"""Tests for the Exp 1425 `.110` carry-forward activation audit.

Spec: REQ-REPORT-033, SCENARIO-REPORT-033.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_110_carryforward_activation_audit import (
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
    def prior(
        experiment_id: str,
        verdict: str,
        addressed_by: str = "next task addresses the root cause",
        retire: bool = False,
    ) -> dict[str, object]:
        return {
            "experiment_id": experiment_id,
            "verdict": verdict,
            "addressed_by": addressed_by,
            "retire_if_same_verdict": retire,
        }

    return {
        "status": "complete",
        "milestone": "2026.04.109",
        "honest_verdict": (
            "milestone_109_10_of_13_criteria_met_threshold_met_but_repair_dvi_fr11_"
            "and_pipeline_carry_forward"
        ),
        "carry_forward_tasks": [
            {
                "id": "repair-executor-v2-root-cause",
                "title": "Repair executor v2 must prove nonzero validated repairs before scale-up",
                "prior_failures": [
                    prior(
                        "exp1414-certificate-llm-repair-executor-v1",
                        "complete_repair_executor_no_successful_repairs",
                    ),
                    prior(
                        "exp1419-fullscale-pipeline-v3-repair-executor",
                        "not_headline_full_pipeline_below_0_40",
                        "no 200-case rerun until nonzero accepted repairs exist",
                        True,
                    ),
                ],
            },
            {
                "id": "dvi-v3-nonforgetting-gate-fix",
                "title": "DVI v3 needs a nonforgetting-preserving training pass",
                "prior_failures": [
                    prior(
                        "exp1415-dvi-v3-1508-fresh-cases",
                        "dvi_v3_blocked_nonforgetting_below_gate",
                        retire=True,
                    )
                ],
            },
            {
                "id": "fr11-v6-after-dvi-v3",
                "title": "FR-11 v6 remains gated on deployable DVI v3",
                "prior_failures": [
                    prior(
                        "exp1418-fr11-self-learning-v6-dvi-v3",
                        "gate_blocked_upstream_dvi_v3_not_deployed",
                    )
                ],
            },
            {
                "id": "dpo-headline-validation-or-finetune-support",
                "title": "DPO path needs headline-valid local-model provenance",
                "prior_failures": [
                    prior(
                        "exp1420-dpo-verified-pairs-1508",
                        "gguf_dpo_unsupported_reranker_fallback_measured",
                    )
                ],
            },
            {
                "id": "test-suite-remaining-debt",
                "title": "Full Python suite and spec-coverage debt remain after focused fix",
                "prior_failures": [
                    prior(
                        "exp1421-test-suite-execution-debt-v1",
                        "focused_runtime_failures_fixed_remaining_debt",
                    )
                ],
            },
            {
                "id": "prm-label-completion",
                "title": "PRM v1 should fill missing local step labels before headline use",
                "prior_failures": [
                    prior(
                        "exp1423-process-reward-model-v1-fover-1508",
                        "prmv1_trained_with_missing_local_labels",
                    )
                ],
            },
        ],
        "retired_experiments": [
            {
                "experiment_id": "exp1419-fullscale-pipeline-v3-repair-executor",
                "result_artifact": (
                    "results/experiment_1419_fullscale_pipeline_v3_repair_executor.json"
                ),
                "retirement_scope": "exact rerun without a new repair-success root-cause fix",
            }
        ],
    }


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "exp1414": _complete(
            "complete_repair_executor_no_successful_repairs",
            repaired_cases_successful=0,
            repaired_case_success_rate=0.0,
        ),
        "exp1415": _complete(
            "dvi_v3_blocked_nonforgetting_below_gate",
            status="blocked",
            nonforgetting_rate=0.968604,
            dvi_v3_deployed=False,
        ),
        "exp1419": _complete(
            "not_headline_full_pipeline_below_0_40",
            cases_evaluated=200,
            repaired_cases_successful=0,
            repair_success_rate=0.0,
            full_pipeline_pass_rate=0.305,
        ),
        "exp1420": _complete(
            "gguf_dpo_unsupported_reranker_fallback_measured",
            dpo_full_finetune_performed=False,
            dpo_reranker_fallback_used=True,
            headline_result_allowed=False,
        ),
        "exp1421": _complete(
            "focused_embedding_store_runtime_failures_fixed_collection_clean_targeted_tests_green_"
            "100pct_store_coverage_full_suite_and_preexisting_spec_coverage_debt_remain",
            spec_coverage_checked=True,
            remaining_debt=["full suite remains red"],
        ),
        "exp1423": _complete(
            "prmv1_trained_on_available_step_labels_with_478_promoted_traces_missing_local_labels",
            training_traces_used=1030,
            missing_trace_labels=478,
        ),
    }


def test_scenario_report_033_maps_all_carryforward_tracks() -> None:
    """SCENARIO-REPORT-033: every .109 unresolved track receives .110 activation."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        sources=_source_payloads(),
        manifest_path="ops/milestone_110_carryforward_manifest.md",
        roadmap_text="exp1425 exp1426 exp1427 exp1428 exp1431 exp1432 exp1433 exp1434 exp1435",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["prior_milestone"] == "2026.04.109"
    assert artifact["carryforward_manifest_complete"] is True
    assert artifact["carryforward_task_count"] == 6
    assert {row["track_id"] for row in artifact["manifest_rows"]} == {
        task["id"] for task in _retro_payload()["carry_forward_tasks"]
    }
    assert "exp1419-fullscale-pipeline-v3-repair-executor" in {
        rule["experiment_id"] for rule in artifact["same_verdict_retirement_rules"]
    }
    assert artifact["forbidden_exact_reruns"][0]["experiment_id"] == (
        "exp1419-fullscale-pipeline-v3-repair-executor"
    )
    assert "without nonzero accepted repair evidence" in manifest


def test_req_report_033_uses_source_verdicts_over_retro_shorthand() -> None:
    """REQ-REPORT-033: exact prior verdicts come from source artifacts when present."""

    artifact, _manifest = build_artifact(
        retro=_retro_payload(),
        sources=_source_payloads(),
        manifest_path="ops/milestone_110_carryforward_manifest.md",
        roadmap_text="exp1434",
    )

    by_experiment = {
        rule["experiment_id"]: rule["prior_verdict"]
        for rule in artifact["same_verdict_retirement_rules"]
    }

    assert by_experiment["exp1421-test-suite-execution-debt-v1"] == (
        "focused_embedding_store_runtime_failures_fixed_collection_clean_targeted_tests_green_"
        "100pct_store_coverage_full_suite_and_preexisting_spec_coverage_debt_remain"
    )
    assert by_experiment["exp1423-process-reward-model-v1-fover-1508"] == (
        "prmv1_trained_on_available_step_labels_with_478_promoted_traces_missing_local_labels"
    )


def test_req_report_033_unknown_track_prevents_completion() -> None:
    """REQ-REPORT-033: unmapped carry-forward rows do not claim activation success."""

    retro = _retro_payload()
    retro["carry_forward_tasks"] = list(retro["carry_forward_tasks"]) + [
        {"id": "unexpected-track", "title": "Unexpected track", "prior_failures": []}
    ]

    artifact, manifest = build_artifact(
        retro=retro,
        sources=_source_payloads(),
        manifest_path="ops/milestone_110_carryforward_manifest.md",
        roadmap_text="",
    )

    assert artifact["status"] == "blocked"
    assert artifact["carryforward_manifest_complete"] is False
    assert artifact["unmapped_tracks"] == ["unexpected-track"]
    assert "unexpected-track" in manifest


def test_req_report_033_run_writes_manifest_and_final_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-033: run writes the manifest, no-change confirmation, and JSON."""

    out_path = tmp_path / "results" / "experiment_1425_109_carryforward_activation_audit.json"
    manifest_path = tmp_path / "ops" / "milestone_110_carryforward_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1424_milestone_109_retro.json", _retro_payload())
    for source_id, payload in _source_payloads().items():
        _write_json(tmp_path / "results" / SOURCE_FILES[source_id], payload)
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "exp1425 exp1426 exp1427 exp1428 exp1429 exp1430 exp1431 exp1432 exp1433 exp1434 exp1435",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text("exp1425 existing roadmap", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "research_conductor.py").write_text("# unchanged\n", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, manifest_path=manifest_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["carryforward_manifest_path"] == ("ops/milestone_110_carryforward_manifest.md")
    assert written["no_change_confirmations"] == {
        "scripts/research_conductor.py": "no activation-audit changes needed",
        "research-roadmap.yaml": "no activation-audit changes needed",
    }
    assert (
        "| track | prior evidence | .110 task | gate rule | retire-if-same-verdict rule |"
        in manifest
    )
    assert "exp1432" in manifest
    assert "exp1419 200-case" in manifest


def test_req_report_033_defensive_missing_and_retro_only_branches(tmp_path: Path) -> None:
    """REQ-REPORT-033: missing files and retro-only priors stay explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"

    artifact, manifest = build_artifact(
        retro={
            "milestone": "2026.04.109",
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
        manifest_path="ops/milestone_110_carryforward_manifest.md",
        roadmap_text="",
    )

    assert artifact["forbidden_exact_reruns"] == []
    assert artifact["same_verdict_retirement_rules"][0]["source_artifact"] is None
    assert artifact["same_verdict_retirement_rules"][0]["prior_verdict"] == "retro_only_verdict"
    assert "retro-only evidence" in manifest
    assert "- None recorded." in manifest

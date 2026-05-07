"""Tests for the Exp 1478 milestone .113 retrospective.

Spec: REQ-REPORT-049, SCENARIO-REPORT-049.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_113 as retro113
from carnot.reporting.milestone_retro_113 import (
    MET,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    UNMET,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1467": {
            "status": "complete",
            "milestone": "2026.04.113",
            "predecessor_milestone": "2026.04.112",
            "criteria_met": 14,
            "criteria_total": 14,
            "activation_manifest_complete": True,
            "retired_lineages_preserved": True,
            "predecessor_honest_verdict": (
                "milestone_112_14_of_14_criteria_met_scope_reduction_satisfied"
            ),
            "honest_verdict": "milestone_113_activation_complete_112_archived_retirements_preserved",
        },
        "exp1468": {
            "status": "complete",
            "live_sota_model_inference_used": True,
            "topk_logprobs_available": True,
            "logits_available": True,
            "telemetry_cases_completed": 12,
            "telemetry_manifest_path": "results/live_sota_telemetry_manifest_1468.jsonl",
            "blockers": [],
            "honest_verdict": "live_sota_topk_telemetry_ready",
        },
        "exp1469": {
            "status": "complete",
            "telemetry_rows_loaded": 12,
            "halt_features_computed": True,
            "spilled_energy_features_computed": True,
            "telemetry_diagnostic_complete": True,
            "best_signal_name": "marginal_energy_proxy_trend",
            "length_or_format_confound_checked": True,
            "diagnostic_lineage_preserved": False,
            "diagnostic_lineage_retired": True,
            "honest_verdict": "retired_non_headline_telemetry_flat_or_confounded",
        },
        "exp1470": {
            "status": "complete",
            "bound_is_sound": True,
            "mock_or_live_logprobs": "live_exp1468",
            "empirical_violation_rates": [0.0, 0.0, 0.0],
            "honest_verdict": "sound_bound_live_exp1468",
        },
        "exp1471": {
            "status": "complete",
            "self_learning_artifact_ready": True,
            "self_learning_delta_overall": 12,
            "new_promoted_count": 12,
            "nonforgetting_rate": 1.0,
            "pivot_preserved": True,
            "pivot_retired": False,
            "headline_result_allowed": True,
            "soundness_mistakes": 0,
            "completeness_mistakes": 140,
            "honest_verdict": "fr11_v8_positive_verified_memory_growth_persisted_without_forgetting",
        },
        "exp1472": {
            "status": "complete",
            "soundness_mistakes": 0,
            "completeness_mistakes": 140,
            "pareto_decision": (
                "preserve_narrow_claim_on_soundness_frontier_with_completeness_caveat"
            ),
            "self_learning_claim_preserved": True,
            "self_learning_claim_retired": False,
            "honest_verdict": "self_learning_claim_preserved_zero_soundness_mistakes",
        },
        "exp1473": {
            "status": "complete",
            "telemetry_validity_verdict": (
                "invalid_for_headline_claim_superficial_or_mechanical_gate"
            ),
            "length_confound_checked": True,
            "format_confound_checked": True,
            "mock_logprob_leakage_checked": True,
            "prompt_family_confound_checked": True,
            "claim_allowed": False,
            "superficial_baseline_results": {
                "claim_blockers": ["source_diagnostic_lineage_retired"],
                "telemetry": {
                    "best_superficial_baseline": {"name": "response_char_length"}
                },
            },
            "honest_verdict": "telemetry_claim_blocked_adversarial_audit",
        },
        "exp1474": {
            "status": "complete",
            "zero_violation_projection": True,
            "baseline_verifier_agreement": True,
            "toy_cases_evaluated": 3,
            "max_constraint_violation": 0.0,
            "honest_verdict": "complete_cpu_only_zero_violation_baseline_agreement",
        },
        "exp1475": {
            "status": "complete",
            "exact_acceptance_equivalent": True,
            "csr_latency_ms_p50": 0.004,
            "existing_path_latency_ms_p50": 0.013,
            "llm_inference_run": False,
            "repair_loop_run": False,
            "honest_verdict": "complete_bounded_case_equivalence_csr_faster_no_generation_or_repair",
        },
        "exp1476": {
            "status": "complete",
            "rtl_regression_complete": True,
            "board_execution_performed": False,
            "bitfile_produced": False,
            "latency_claimed": False,
            "prior_boundary_preserved": True,
            "honest_verdict": (
                "rtl_regression_manifest_complete_source_level_only_no_board_bitfile_or_latency_claim"
            ),
        },
        "exp1477": {
            "status": "complete",
            "hardware_claim_allowed": False,
            "simulator_only": True,
            "thrml_available": False,
            "parity_metric": {
                "status": "blocked",
                "metric": "max_abs_energy_error",
                "reason": "missing Python module while importing THRML: thrml",
                "value": None,
            },
            "carnot_sampler_cases": [{"case": "n3_biased_chain", "sample_count": 64}],
            "npim_cases": [{"case": "n3_biased_chain", "energy_delta_vs_fixed_baseline": 0.0}],
            "npim_energy_delta": {"value": 0.0, "unit": "ising_energy"},
            "blockers": [
                {
                    "blocker": "thrml_not_importable",
                    "detail": "missing Python module while importing THRML: thrml",
                }
            ],
            "honest_verdict": "complete_thrml_unavailable_npim_simulator_probe_recorded",
        },
    }


def _conductor_log_text() -> str:
    return "\n".join(
        [
            "| 2026-05-07 07:06 UTC | Live SOTA GGUF Logprob Telemetry Preflight | FAIL | Codex CLI error: _fn=lambda _spec, _case: pytest.fail(\"generation must not run\") |",
            "| 2026-05-07 07:08 UTC | Live SOTA GGUF Logprob Telemetry Preflight | OK | Deliverable already exists in repo |",
            "| 2026-05-07 07:23 UTC | HALT + Spilled Energy Diagnostic Micro-Benchmark - | FAIL | artifact_not_updated_past_bootstrap (deliverable=results/experiment_1469_halt_spilled_energy_telemetry_diagnostic.json) |",
            "| 2026-05-07 07:48 UTC | BEAVER-Lite Deterministic Bound Smoke - Adopted Ex | OK | 81 passed |",
            "| 2026-05-07 08:55 UTC | Live Telemetry Adversarial Validity Audit - Length | FAIL | artifact_not_updated_past_bootstrap (deliverable=results/experiment_1473_live_telemetry_adversarial_validity_audit.json) |",
            "| 2026-05-07 10:48 UTC | KV260 Discrete SB RTL Regression Pack - Source-Lev | FAIL | artifact_not_updated_past_bootstrap (deliverable=results/experiment_1476_kv260_discrete_sb_rtl_regression_pack.json) |",
        ]
    )


def test_req_report_049_scores_all_113_criteria_and_lineage_decisions() -> None:
    """REQ-REPORT-049: all 12 criteria are scored from terminal source fields."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=[],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=False,
        conductor_log_text=_conductor_log_text(),
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert artifact["criteria_total"] == 12
    assert artifact["criteria_met"] == 12
    assert artifact["ops_docs_updated"] is False
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["success_criteria_results"]["halt_energy_diagnostic"]["status"] == MET
    assert artifact["success_criteria_results"]["thrml_npim_parity"]["status"] == MET
    assert artifact["missing_artifacts"] == [
        {"path": "research-roadmap-next.yaml", "reason": "requested_input_missing"}
    ]
    assert {item["lineage"] for item in artifact["retired_lineages"]} == {
        "HALT/Spilled Energy telemetry diagnostic"
    }
    assert {item["lineage"] for item in artifact["preserved_lineages"]} == {
        "FR-11 v8 verified-memory-growth pivot",
        "T-SKM toy linear projection",
        "STATIC CSR certificate automaton",
        "KV260 source-level RTL regression",
        "THRML/NPIM simulator-only parity probe",
    }
    blocked = {(item["experiment_id"], item["kind"]) for item in artifact["blocked_tasks"]}
    assert ("exp1477", "terminal_environment_blocker") in blocked
    assert ("exp1469", "failed_conductor_attempt") in blocked
    assert artifact["honest_verdict"].startswith("milestone_113_12_of_12_criteria_met")


def test_scenario_report_049_gate_off_halt_needs_terminal_logprob_skip() -> None:
    """SCENARIO-REPORT-049: gated-off HALT only counts with explicit logprob evidence."""

    sources = _scenario_sources()
    sources["exp1468"]["topk_logprobs_available"] = False
    sources["exp1469"] = {
        "status": "skipped",
        "telemetry_diagnostic_complete": False,
        "gated_off_reason": "missing top-k logprobs from exp1468",
        "honest_verdict": "terminal_skip_missing_logprobs",
    }

    artifact = build_artifact(
        sources,
        missing_source_ids=[],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=True,
        conductor_log_text="",
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert artifact["success_criteria_results"]["halt_energy_diagnostic"]["status"] == MET
    assert artifact["criteria_met"] == 12

    sources["exp1469"] = {
        "status": "skipped",
        "telemetry_diagnostic_complete": False,
        "honest_verdict": "terminal_skip_without_reason",
    }
    artifact = build_artifact(
        sources,
        missing_source_ids=[],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=True,
        conductor_log_text="",
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert artifact["success_criteria_results"]["halt_energy_diagnostic"]["status"] == UNMET
    assert artifact["criteria_met"] == 10


def test_req_report_049_missing_or_failed_terminal_artifacts_are_unmet() -> None:
    """REQ-REPORT-049: missing and failed source artifacts are reported, not inferred."""

    sources = _scenario_sources()
    sources["exp1470"]["bound_is_sound"] = False
    del sources["exp1469"]
    del sources["exp1475"]

    artifact = build_artifact(
        sources,
        missing_source_ids=["exp1469", "exp1475"],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=True,
        conductor_log_text="",
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert artifact["success_criteria_results"]["beaver_smoke"]["status"] == UNMET
    assert artifact["success_criteria_results"]["halt_energy_diagnostic"]["status"] == UNMET
    assert artifact["success_criteria_results"]["static_automaton_smoke"]["status"] == UNMET
    assert artifact["criteria_met"] == 8
    assert {
        "path": "results/experiment_1469_halt_spilled_energy_telemetry_diagnostic.json",
        "reason": "source_artifact_missing",
    } in artifact["missing_artifacts"]
    assert {
        "path": "results/experiment_1475_static_csr_certificate_automaton_smoke.json",
        "reason": "source_artifact_missing",
    } in artifact["missing_artifacts"]


def test_req_report_049_step0_skeleton_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-049: STEP 0 writes an auditable in-progress skeleton first."""

    out_path = tmp_path / "results" / "experiment_1478_milestone_113_retro.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "in_progress"
    assert set(written) == set(REQUIRED_ARTIFACT_FIELDS)


def test_req_report_049_run_writes_step0_before_loading_sources(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-049: run writes status=in_progress before source-artifact reads."""

    observations: dict[str, object] = {}
    out_path = tmp_path / "results" / "experiment_1478_milestone_113_retro.json"

    def fake_load_sources(_results_dir: Path):
        observations["step0"] = json.loads(out_path.read_text(encoding="utf-8"))
        return _scenario_sources(), []

    monkeypatch.setattr(retro113, "_load_sources", fake_load_sources)
    monkeypatch.setattr(retro113, "_read_text", lambda _path: _conductor_log_text())

    artifact = run(root=tmp_path, out_path=out_path)

    step0 = observations["step0"]
    assert isinstance(step0, dict)
    assert step0["status"] == "in_progress"
    assert set(step0) == set(REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"


def test_scenario_report_049_run_reads_real_source_files(tmp_path: Path) -> None:
    """SCENARIO-REPORT-049: the runner loads .113 source files and writes JSON."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1478_milestone_113_retro.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# roadmap\n",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.113\n", encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["criteria_met"] == 12
    assert written["source_artifacts_checked"][0] == {
        "experiment_id": "exp1467",
        "path": "results/experiment_1467_112_completion_archive_113_activation.json",
        "exists": True,
    }
    assert written["roadmap_inputs"]["requested_research_roadmap_next_present"] is False

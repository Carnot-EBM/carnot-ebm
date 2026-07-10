"""Tests for the Exp5510 .500 transition receipt.

Spec refs: REQ-REPORT-5510, SCENARIO-REPORT-5510,
SCENARIO-REPORT-5510-BLOCKED-INPUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5510_transition_v500 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_context(root: Path) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            continue
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == mod.ROADMAP_RELATIVE_PATH:
            path.write_text(
                yaml.safe_dump(
                    {
                        "milestone": mod.MILESTONE,
                        "milestone_doc": mod.VNEXT_RELATIVE_PATH.as_posix(),
                        "tasks": [
                            {"id": task_id, "milestone": mod.MILESTONE}
                            for task_id in mod.EXPECTED_TASK_IDS
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        elif rel_path == mod.VNEXT_RELATIVE_PATH:
            path.write_text(
                "\n".join(
                    [
                        "# Research Roadmap vNEXT - Milestone 2026.07.500",
                        "",
                        "**Previous milestone:** 2026.07.499",
                        "**Task range:** Exp 5510-5522",
                        "",
                        "Exp 5513 requires Exp 5512 `structured_output_positive_control_ready == true`.",
                        "Exp 5516 requires Exp 5515 `metric_independence_clean == true` and `csl_gate_fields_resolvable == true`.",
                        "Exp 5521 requires Exp 5520 `arc_levelup_candidate_ready == true`.",
                        "Hardware remains receipt-only until matched timing exists.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        elif rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            path.write_text(
                "\n".join(
                    [
                        "| 2026-07-09 18:05 UTC | Gated SOTA GGUF concept/claim evidence panel | OK | 90 passed |",
                        "| 2026-07-09 18:53 UTC | Gated SOTA GGUF CSL memory panel with independent metrics | GATE_BLOCK | 2 of 2 gate(s) failed; first failure: exp5502-csl-tautology-static-corrigendum-v499.metric_independence_clean |",
                        "| 2026-07-09 20:00 UTC | Multi-board hardware receipt continuity without speedup claim | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
                        "| 2026-07-09 20:41 UTC | Gated ARC live perception-generation level-up attempt | OK | 89 passed |",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"context for {rel_path.as_posix()}\n", encoding="utf-8")


def _write_artifacts(root: Path) -> None:
    _write_json(
        root,
        mod.ARTIFACTS[5496],
        {
            "milestone": "2026.07.499",
            "previous_milestone": "2026.07.498",
            "next_task_range": "exp5496-exp5509",
            "honest_verdict": "complete: archived .498 terminal evidence into .499 transition receipt",
            "roadmap_yaml_unchanged": True,
            "conductor_unchanged": True,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5497],
        {
            "milestone": "2026.07.499",
            "pretest_cascade_resolved": True,
            "reproduced_pretest_failure": False,
            "downstream_gate_recommendation": "open_downstream_pretest_gate",
            "honest_verdict": "complete: .498 pretest cascade audited",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5498],
        {
            "milestone": "2026.07.499",
            "new_references_added": [{"title": "Constrained Decoding for Diffusion LMs"}],
            "closed_scopes_reopened": False,
            "research_references_updated": True,
            "honest_verdict": "complete: 1 new actionable V499 execution-time source delta appended",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5499],
        {
            "milestone": "2026.07.499",
            "preference_maxsat_fixture_ready": True,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: preference_maxsat_minimal_fixture_ready_exact_validators_authoritative",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5500],
        {
            "milestone": "2026.07.499",
            "concept_claim_telemetry_rows": 6,
            "abstention_count": 6,
            "exact_validator_accuracy": 0.333333,
            "headline_models_used": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-26B-A4B-it-GGUF",
            ],
            "gpu_offload_verified": True,
            "honest_verdict": "complete: live_sota_claim_panel_measured_accuracy_0.333333",
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5501],
        {
            "milestone": "2026.07.499",
            "helper_contract_fixture_ready": True,
            "rolled_up_verdict_accuracy": 1.0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: helper_contract_hierarchical_claim_fixture_ready_exact_predicates_authoritative",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5502],
        {
            "milestone": "2026.07.499",
            "status": "complete",
            "tautology_flag_resolved": True,
            "metric_independence_clean": False,
            "csl_scale_headline_allowed": False,
            "downstream_recommendation": "bounded_requires_rerun",
            "retire_same_scope_if_repeated": True,
            "honest_verdict": "complete: Exp5474 CSL scale headline is bounded, not clean",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5503],
        {
            "milestone": "2026.07.499",
            "task_id": "exp5503-csl-experience-graph-replay-v499",
            "csl_experience_graph_ready": True,
            "graph_memory_score": 1.0,
            "no_memory_baseline_score": 0.0,
            "heldout_delta": 1.0,
            "negative_transfer_rate": 0.0,
            "stale_evidence_rejection_rate": 1.0,
            "model_weights_mutated": False,
            "honest_verdict": "complete: experience_graph_replay_ready_delta_+1.000000",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5504],
        {
            "experiment": 5504,
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "2 of 2 gate(s) failed; first failure: exp5502 metric_independence_clean",
            "gates_evaluated": [
                {"artifact_field": "metric_independence_clean", "actual": False, "passed": False},
                {"artifact_field": "csl_experience_graph_ready", "actual": None, "passed": False},
            ],
            "blocked_at_layer": "conductor_pre_gate",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5505],
        {
            "milestone": "2026.07.499",
            "descriptor_ready_for_hardware": True,
            "num_descriptor_rows": 7,
            "exact_fallback_agreement_rate": 1.0,
            "hardware_speedup_claim": False,
            "honest_verdict": "complete: active-constraint descriptors are exact-fallback checked",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5506],
        {
            "milestone": "2026.07.499",
            "cpu_status": "reachable",
            "cuda_status": "reachable",
            "polar_fire_status": "reachable",
            "kv260_status": "blocked_identity",
            "gatemate_status": "blocked_identity",
            "matched_timing_available": False,
            "hardware_speedup_claim": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
            "honest_verdict": "complete: descriptor smoke receipts collected with honest blocked statuses",
            "inference_substrate": "hardware_smoke",
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5507],
        {
            "milestone": "2026.07.499",
            "selected_game": "dc22",
            "selected_level": "L3",
            "honest_verdict": "complete: dc22 L3 perception-generation precheck ready; no solve claimed",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )
    _write_json(
        root,
        mod.ARTIFACTS[5508],
        {
            "milestone": "2026.07.499",
            "status": "honest_null",
            "selected_game": "dc22",
            "selected_level": "L3",
            "solve_provenance": "live_agent_self_discovery",
            "live_agent_attempts": 47,
            "registry_before_levels": 69,
            "registry_after_levels": 69,
            "arc_registry_delta": 0,
            "reproduced_levels": 0,
            "offline_reproduced": False,
            "registry_updated": False,
            "trajectory_taxonomy_counts": {"factual": 23, "logical": 23, "scope_based": 1},
            "honest_verdict": "honest_null: dc22 L3 bounded_budget_no_target_level_reproduction; registry_delta=0",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
    )
    _write_json(
        root,
        mod.PRIOR_CAPSTONE_RELATIVE_PATH,
        {
            "milestone": mod.PREVIOUS_MILESTONE,
            "status": "complete",
            "artifacts_missing": [],
            "pretest_cascade_resolved": True,
            "hard_soft_core_verdict": "bounded: exact hard/soft core landed; SOTA panel abstained",
            "csl_verdict": "blocked: metric_independence_clean=false and Exp5504 gate-blocked",
            "hardware_verdict": "bounded: matched_timing_available=False; hardware_speedup_claim=False",
            "arc_verdict": "honest_null: dc22 L3 no bank",
            "arc_registry_delta": 0,
            "hardware_speedup_claim": False,
            "roadmap_yaml_unchanged": True,
            "conductor_unchanged": True,
            "honest_verdict": "complete: .499 capstone read actual Exp5496-Exp5508 artifacts",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
    )


def test_req_report_5510_spec_declares_required_fields() -> None:
    """REQ-REPORT-5510: OpenSpec anchors the transition artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5510") :]

    assert "SCENARIO-REPORT-5510" in section
    assert "SCENARIO-REPORT-5510-BLOCKED-INPUT" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5510_summarizes_v499_facts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5510: observed .499 artifacts drive the lane summary."""

    _write_context(tmp_path)
    _write_artifacts(tmp_path)

    report = mod.build_report(tmp_path, tests_run=["unit 5510"])

    assert report["milestone"] == "2026.07.500"
    assert report["previous_milestone"] == "2026.07.499"
    assert report["prior_capstone_path"] == "results/experiment_5509_capstone_v499.json"
    assert report["previous_task_range"] == "exp5496-exp5509"
    assert report["next_task_range"] == "exp5510-exp5522"
    assert {row["lane"] for row in report["clean_lanes"]} == {
        "transition",
        "pretest_cascade_repair",
        "source_delta",
        "preference_maxsat_fixture",
        "helper_contract_fixture",
        "experience_graph_replay_readiness",
    }
    assert {row["lane"] for row in report["bounded_lanes"]} == {
        "sota_missing_candidate_panel",
        "active_constraint_exact_fallback",
        "hardware_receipt_only_timing",
    }
    assert {row["lane"] for row in report["blocked_lanes"]} == {
        "csl_metric_independence_block",
        "csl_gate_field_mismatch",
        "hardware_identity_blocks",
    }
    assert {row["lane"] for row in report["honest_null_lanes"]} == {
        "arc_no_bank",
        "arc_registry_delta_zero",
    }
    assert {row["lane"] for row in report["flagged_lanes"]} == {
        "csl_headline_leakage_unresolved",
        "hardware_timing_methodology_flags",
        "arc_repeated_pattern_no_bank",
    }
    assert report["structured_sota_gate_required"] is True
    assert report["csl_independent_metric_gate_required"] is True
    assert report["arc_live_levelup_gate_required"] is True
    assert report["hardware_receipt_only"] is True
    assert report["artifact_aliases"][0]["actual_path"] == mod.ARTIFACTS[5502].as_posix()
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["source_context_missing"] == ["research-roadmap-next.yaml"]
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")


def test_scenario_report_5510_dirty_protected_file_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5510-BLOCKED-INPUT: protected-file dirt fails closed."""

    _write_context(tmp_path)
    _write_artifacts(tmp_path)

    report = mod.build_report(
        tmp_path,
        tests_run=["unit 5510"],
        modification_overrides={mod.ROADMAP_RELATIVE_PATH: True},
    )

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["roadmap_yaml_unchanged"] is False
    assert report["conductor_unchanged"] is True
    assert "research-roadmap.yaml_modified" in report["failed_preconditions"]
    assert {row["lane"] for row in report["clean_lanes"]} >= {
        "transition",
        "experience_graph_replay_readiness",
    }


def test_scenario_report_5510_missing_required_artifact_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5510-BLOCKED-INPUT: missing evidence is not inferred."""

    _write_context(tmp_path)
    _write_artifacts(tmp_path)
    (tmp_path / mod.ARTIFACTS[5500]).unlink()

    report = mod.build_report(tmp_path, tests_run=["unit 5510"])

    assert report["status"] == "blocked"
    assert mod.ARTIFACTS[5500].as_posix() in report["artifacts_missing"]
    assert f"{mod.ARTIFACTS[5500].as_posix()}_missing_or_unreadable" in report[
        "failed_preconditions"
    ]
    assert {row["lane"] for row in report["bounded_lanes"]} >= {
        "sota_missing_candidate_panel"
    }


def test_scenario_report_5510_precondition_mismatches_are_explicit() -> None:
    """SCENARIO-REPORT-5510-BLOCKED-INPUT: mismatches name exact gates."""

    failed = mod._failed_preconditions(
        ["results/missing.json"],
        {"milestone": "2026.07.498"},
        {"milestone": "2026.07.499"},
        ["exp5510-transition-v500"],
        "exp5500-exp5501",
        False,
        True,
    )

    assert failed == [
        "results/missing.json_missing_or_unreadable",
        "prior_capstone_milestone_mismatch",
        "research-roadmap.yaml_milestone_mismatch",
        "roadmap_task_ids_mismatch",
        "vnext_task_range_mismatch",
        "scripts/research_conductor.py_modified",
    ]


def test_scenario_report_5510_write_report_persists_required_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5510: written artifact contains the required fields."""

    _write_context(tmp_path)
    _write_artifacts(tmp_path)

    payload = mod.write_report(tmp_path, tests_run=["unit 5510"])
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == payload
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

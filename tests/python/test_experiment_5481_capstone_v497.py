"""Tests for the Exp5481 V497 capstone aggregator.

Spec refs: REQ-REPORT-5481, SCENARIO-REPORT-5481,
SCENARIO-REPORT-5481-MISSING-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5481_capstone_v497 as exp5481


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_context(root: Path) -> None:
    for rel_path in exp5481.CONTEXT_PATHS:
        path = root / rel_path
        if rel_path == "results":
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"context for {rel_path}\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text("milestone: 2026.07.497\n", encoding="utf-8")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# conductor fixture\n", encoding="utf-8")


def _artifact_payloads() -> dict[str, dict[str, Any]]:
    return {
        "results/experiment_5468_transition_v497.json": {
            "milestone": "2026.07.497",
            "status": "complete",
            "honest_verdict": "complete: transition preserved guided decoding quarantine",
            "blocked_lanes": [{"lane": "guided_decoding", "headline_blockers": ["flagged_adversarial"]}],
            "bounded_lanes": [{"lane": "hardware_receipts"}],
            "honest_null_lanes": [{"lane": "arc"}],
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "results/experiment_5469_source_delta_v497.json": {
            "milestone": "2026.07.497",
            "status": "complete",
            "honest_verdict": "complete: source delta appended",
            "new_actionable_findings_count": 2,
            "closed_scopes_reopened": False,
            "research_references_updated": True,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "results/experiment_5470_rewrite_state_semantic_fixture_v497.json": {
            "milestone": "2026.07.497",
            "honest_verdict": "complete: guided decoding remains quarantined",
            "rewrite_state_fixture_ready": True,
            "guided_decoding_quarantine_lifted": False,
            "exact_validator_agreement": 1.0,
            "hidden_premise_catch_rate": 1.0,
            "unlicensed_mutation_catch_rate": 1.0,
            "factual_distortion_rate": 0.0,
            "semantic_false_accept_rate": 0.0,
        },
        "results/experiment_5471_guard_composition_scale_v497.json": {
            "milestone": "2026.07.497",
            "honest_verdict": "complete: guard composition ready; quarantine remains",
            "guard_composition_ready": True,
            "guided_decoding_quarantine_lifted": False,
            "exact_final_agreement": 1.0,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
        },
        "results/experiment_5472_sota_evidence_telemetry_v497.json": {
            "milestone": "2026.07.497",
            "honest_verdict": "complete: local SOTA telemetry collected",
            "sota_evidence_telemetry_ready": True,
            "guided_decoding_used": False,
            "headline_models_run": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
            "exact_validator_accuracy": 0.5,
            "semantic_false_accept_rate": 0.0,
            "factual_distortion_rate": 0.0,
            "runtime_precondition_receipt": {"runtime_ready": True, "llama_cpp_gpu_offload": True},
            "gpu_offload_receipts": [{"offload_verified": True}],
        },
        "results/experiment_5473_csl_kan_surrogate_assurance_v497.json": {
            "milestone": "2026.07.497",
            "status": "complete",
            "honest_verdict": "complete: CSL KAN surrogate ready",
            "csl_kan_surrogate_ready": True,
            "model_weight_mutation": False,
            "constraint_violation_count": 0,
            "negative_transfer_deflection_rate": 1.0,
        },
        "results/experiment_5474_sota_csl_scale_v497.json": {
            "milestone": "2026.07.497",
            "honest_verdict": "complete: CSL scale ready",
            "csl_scale_ready": True,
            "model_weight_mutation": False,
            "negative_transfer_deflection_rate": 1.0,
            "exact_validator_pass_rate": 1.0,
            "delta_vs_no_memory": 0.75,
            "delta_vs_naive_icl": 0.5,
            "context_token_cost_delta": 0.08596,
            "runtime_precondition_receipt": {"runtime_ready": True, "llama_cpp_gpu_offload": True},
            "gpu_offload_receipts": [{"offload_verified": True}],
        },
        "results/experiment_5475_csl_behavioral_memory_ladder_v497.json": {
            "milestone": "2026.07.497",
            "honest_verdict": "complete: behavioral memory ready",
            "csl_behavioral_memory_ready": True,
            "model_weight_mutation": False,
            "downstream_action_use_rate": 1.0,
            "stale_memory_rejection_rate": 1.0,
        },
        "results/experiment_5476_helper_lemma_core_witness_repair_v497.json": {
            "milestone": "2026.07.497",
            "honest_verdict": "complete: helper witness repair ready",
            "helper_lemma_repair_ready": True,
            "exact_recheck_pass_rate": 0.833333,
            "false_accept_count": 0,
            "repeated_failure_reduction_rate": 1.0,
        },
        "results/experiment_5477_pdit_lns_boundary_exchange_v497.json": {
            "milestone": "2026.07.497",
            "status": "complete",
            "honest_verdict": "complete: boundary exchange ready; no speedup",
            "boundary_exchange_ready": True,
            "exact_fallback_completeness_rate": 1.0,
            "unsafe_false_accept_count": 0,
            "advisory_improvement_delta": 2.518519,
            "hardware_speedup_claim": False,
        },
        "results/experiment_5478_hardware_receipts_v497.json": {
            "milestone": "2026.07.497",
            "honest_verdict": "complete: hardware receipts ready; no speedup",
            "hardware_receipts_ready": True,
            "hardware_speedup_claim": False,
            "result_hash_match_rate": 1.0,
            "reachable_boards": ["polarfire"],
            "unreachable_boards": [{"board_identity": "kv260", "blocked_reason": "blocked_kv260_ssh"}],
        },
        "results/experiment_5479_arc_target_rotation_precheck_v497.json": {
            "milestone": "2026.07.497",
            "status": "complete",
            "honest_verdict": "complete: sb26 L3 precheck ready; no bank",
            "arc_target_rotation_ready": True,
            "solve_claimed": False,
            "live_path_reachable": True,
            "selected_game": "sb26",
            "selected_target_level": 3,
        },
        "results/experiment_5480_arc_live_salience_levelup_v497.json": {
            "milestone": "2026.07.497",
            "status": "honest_null",
            "honest_verdict": "honest_null: sb26 L3 bounded_budget_no_target_level_reproduction",
            "new_level_banked": False,
            "offline_reproduced": False,
            "reproduced_levels_before": 2,
            "reproduced_levels_after": 2,
            "reproduced_levels": 0,
            "registry_updated": False,
            "failure_mode": "bounded_budget_no_target_level_reproduction",
            "action_count": 47,
            "explored_state_count": 9,
        },
    }


def _populate_artifacts(root: Path, payloads: dict[str, dict[str, Any]]) -> None:
    _write_context(root)
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)


def test_req_report_5481_spec_declares_required_fields() -> None:
    """REQ-REPORT-5481: OpenSpec anchors the Exp5481 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5481") :]

    assert "SCENARIO-REPORT-5481" in section
    assert "SCENARIO-REPORT-5481-MISSING-FLAGGED" in section
    assert str(exp5481.OUTPUT_REL_PATH) in section
    for field in exp5481.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5481_builds_capstone_from_existing_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5481: complete inputs produce the V497 truth table."""

    _populate_artifacts(tmp_path, _artifact_payloads())

    report = exp5481.build_report(tmp_path, tests_run=["unit 5481"])

    assert report["milestone"] == "2026.07.497"
    assert report["artifact_paths"] == sorted(exp5481.EXPECTED_ARTIFACT_PATHS)
    assert report["missing_artifacts"] == []
    assert report["flagged_artifacts"] == []
    assert report["guided_decoding_quarantine_status"] == "quarantined"
    assert report["csl_status"].startswith("headline_ready:")
    assert report["arc_registry_delta"] == 0
    assert report["hardware_speedup_claim"] is False
    assert report["ops_status_updated"] is False
    assert report["ops_changelog_updated"] is False
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["honest_verdict"].startswith("complete:")

    assert report["truth_table"]["csl"]["classification"] == "headline_ready"
    assert report["truth_table"]["guided_decoding"]["classification"] == "blocked"
    assert report["truth_table"]["hardware_receipts"]["classification"] == "bounded"
    assert report["truth_table"]["hardware_speedup_claim"]["classification"] == "honest_null"
    assert report["truth_table"]["arc_live_path"]["classification"] == "honest_null"
    assert {row["lane"] for row in report["headline_ready_lanes"]} == {
        "verifiable_reasoning_guards",
        "csl",
    }
    assert {row["lane"] for row in report["bounded_lanes"]} == {
        "transition_source_refresh",
        "local_sota_runtime",
        "pdit_lns_boundary_exchange",
        "hardware_receipts",
    }
    assert {row["lane"] for row in report["honest_null_lanes"]} == {
        "arc_live_path",
        "hardware_speedup_claim",
    }
    assert report["prd_gap_table"]["FR-11 continuous self-learning"]["status"].startswith(
        "headline_ready"
    )
    assert report["prd_gap_table"]["hardware acceleration"]["status"] == "bounded_receipts_only"
    assert report["failure_taxonomy"]["arc_no_bank"]["failure_mode"] == (
        "bounded_budget_no_target_level_reproduction"
    )


def test_scenario_report_5481_missing_and_flagged_inputs_stay_non_headline(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5481-MISSING-FLAGGED: bad inputs stay visible."""

    payloads = _artifact_payloads()
    payloads["results/experiment_5472_sota_evidence_telemetry_v497.json"][
        "flagged_adversarial"
    ] = True
    payloads["results/experiment_5473_csl_kan_surrogate_assurance_v497.json"][
        "adversarial_verdict"
    ] = "flagged"
    del payloads["results/experiment_5478_hardware_receipts_v497.json"]
    _populate_artifacts(tmp_path, payloads)
    (tmp_path / "ops/status.md").unlink()

    report = exp5481.build_report(tmp_path, tests_run=["unit missing flagged"])

    assert report["honest_verdict"].startswith("blocked:")
    assert report["missing_artifacts"] == [
        "results/experiment_5478_hardware_receipts_v497.json"
    ]
    assert report["flagged_artifacts"] == [
        "results/experiment_5472_sota_evidence_telemetry_v497.json",
        "results/experiment_5473_csl_kan_surrogate_assurance_v497.json",
    ]
    assert report["truth_table"]["local_sota_runtime"]["classification"] == "flagged"
    assert report["truth_table"]["csl"]["classification"] == "flagged"
    assert report["truth_table"]["hardware_receipts"]["classification"] == "missing"
    assert report["truth_table"]["hardware_speedup_claim"]["classification"] == "missing"
    assert {row["lane"] for row in report["flagged_lanes"]} == {
        "csl",
        "guided_decoding",
        "local_sota_runtime",
    }
    assert {row["lane"] for row in report["missing_lanes"]} == {
        "hardware_receipts",
        "hardware_speedup_claim",
    }
    assert report["source_context_missing"] == ["ops/status.md"]
    assert report["csl_status"].startswith("flagged:")
    assert exp5481._arc_registry_delta({}) == 0


def test_scenario_report_5481_main_writes_deliverable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5481: CLI writes the deliverable JSON."""

    _populate_artifacts(tmp_path, _artifact_payloads())

    exit_code = exp5481.main(
        [
            "--root",
            str(tmp_path),
            "--output",
            "results/experiment_5481_capstone_v497.json",
            "--test-command",
            "unit 5481",
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (tmp_path / "results/experiment_5481_capstone_v497.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["tests_run"] == ["unit 5481"]
    assert payload["honest_verdict"].startswith("complete:")


def test_scenario_report_5481_rejects_non_object_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5481-MISSING-FLAGGED: malformed artifacts fail closed."""

    _write_context(tmp_path)
    bad_path = tmp_path / exp5481.EXPECTED_ARTIFACT_PATHS[0]
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="did not contain a JSON object"):
        exp5481.build_report(tmp_path)


def test_scenario_report_5481_protected_git_status_is_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-5481: protected-file booleans come from git status."""

    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""

    def fake_run(command: list[str], **_: Any) -> Result:
        calls.append(command)
        return Result()

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(exp5481.subprocess, "run", fake_run)

    assert exp5481._protected_file_clean(tmp_path, "research-roadmap.yaml") is True
    assert calls == [["git", "status", "--short", "--", "research-roadmap.yaml"]]

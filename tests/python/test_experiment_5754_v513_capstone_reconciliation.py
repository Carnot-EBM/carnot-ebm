"""Tests for Exp5754 V513 capstone reconciliation.

Spec refs: REQ-REPORT-5754, SCENARIO-REPORT-5754,
SCENARIO-REPORT-5754-MISSING-AND-GATE-SKIPPED,
SCENARIO-REPORT-5754-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5754_v513_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id in mod.EXPECTED_TASK_IDS:
        row: JsonDict = {"id": task_id, "deliverable": f"results/{task_id}.json"}
        if task_id == "exp5747-sota-exact-proposal-utility-panel":
            row["gated_on"] = [
                {
                    "upstream": "exp5746-exact-proposal-utility-benchmark",
                    "artifact_field": "benchmark_ready_score",
                    "op": ">=",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5748-selective-exact-feedback-search":
            row["gated_on"] = [
                {
                    "upstream": "exp5747-sota-exact-proposal-utility-panel",
                    "artifact_field": "overall_proposal_utility_positive",
                    "op": "==",
                    "value": True,
                }
            ]
        if task_id == "exp5750-dependent-task-continuous-self-learning":
            row["gated_on"] = [
                {
                    "upstream": "exp5749-csl-render-matched-mechanism-audit",
                    "artifact_field": "kan_mechanism_residual",
                    "op": ">",
                    "value": 0.0,
                }
            ]
        if task_id == "exp5752-one-axis-allocation-free-10x-crossover":
            row["prior_failures"] = [
                {
                    "experiment_id": "exp5739-one-axis-batched-10x-crossover",
                    "verdict": "complete: terminal null; matched batched Rust/Python CPU evidence did not prove the strict consecutive larger-size 10x lower-bound rule",
                    "retire_if_same_verdict": True,
                }
            ]
            row["gated_on"] = [
                {
                    "upstream": "exp5751-rust-restart-parity-repair",
                    "artifact_field": "restart_parity_ready_score",
                    "op": ">=",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5753-arc-generic-primitive-live-registry-ab":
            row["gated_on"] = [
                {
                    "upstream": "exp5745-arc-causal-gate-schema-corrigendum",
                    "artifact_field": "counterfactual_receipt_coverage_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        tasks.append(row)
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _artifact_payloads() -> dict[Path, JsonDict]:
    return {
        mod.EXP5743_PATH: {
            "status": "complete",
            "honest_verdict": "complete: archived terminal .512 evidence into .513",
            "proposal_channel_ready": True,
            "sota_proposal_stream_ready": True,
            "continuous_self_learning_credited": True,
            "batch_backend_ready": True,
            "rust_batched_10x_ready": False,
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5744_PATH: {
            "status": "complete",
            "honest_verdict": "complete: no new non-duplicate actionable V513 source deltas",
            "accepted_findings": [],
            "watch_only_findings": [{"source_id": "logical_intelligence_kona"}],
            "benchmark_compute_claimed": False,
            "inference_substrate": "web_and_bibliographic_search_only",
        },
        mod.EXP5745_PATH: {
            "honest_verdict": "complete: exp5740_lossless_scalar_gate_corrigendum_positive_count_7_admitted_leaks_0_registry_delta_0",
            "counterfactual_receipt_coverage_score": 1.0,
            "admitted_source_leak_count": 0,
            "admitted_game_identity_leak_count": 0,
            "positive_causal_primitive_count": 7,
            "solve_provenance": "development_proxy",
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
        },
        mod.EXP5746_PATH: {
            "status": "complete",
            "honest_verdict": "complete: exact_proposal_utility_benchmark_ready",
            "benchmark_ready_score": 1.0,
            "instance_count": 180,
            "structure_receipts": {f"s{i}": {} for i in range(180)},
            "solution_receipts": {f"q{i}": {} for i in range(180)},
            "exact_optimum_receipts": {f"o{i}": {} for i in range(180)},
            "structure_receipt_failure_count": 0,
            "solution_receipt_failure_count": 0,
            "candidate_domain_incomplete_count": 0,
            "validator_disagreement_count": 0,
            "llm_inference_used": False,
        },
        mod.EXP5747_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "3 of 3 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5746-exact-proposal-utility-benchmark",
                    "artifact_field": "benchmark_ready_score",
                    "expected": 1.0,
                    "actual": None,
                    "passed": False,
                }
            ],
            "blocked_at_layer": "conductor_pre_gate",
        },
        mod.EXP5749_PATH: {
            "honest_verdict": "complete: kan_mechanism_residual_negative_fr11_safety_retained",
            "continuous_self_learning_credited": True,
            "kan_mechanism_residual": -0.084269,
            "prefix_retention_pass_score": 1.0,
            "unsafe_update_count": 0,
            "model_weight_mutation": False,
            "production_default_enabled": False,
        },
        mod.EXP5750_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "1 of 3 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5749-csl-render-matched-mechanism-audit",
                    "artifact_field": "kan_mechanism_residual",
                    "expected": 0.0,
                    "actual": -0.084269,
                    "passed": False,
                }
            ],
            "blocked_at_layer": "conductor_pre_gate",
        },
        mod.EXP5751_PATH: {
            "honest_verdict": "complete: restart parity repaired; no timing or hardware claim",
            "restart_parity_ready_score": 1.0,
            "restart_parity": {"all_repaired_suffix_hashes_match": True},
            "distributional_parity": {"passed": True},
            "fallback_equivalence": {"exact_fallback_equivalence": True},
            "production_backend_reachable": {"passed": True},
            "timing_claimed": False,
            "hardware_speedup_claimed": False,
        },
        mod.EXP5752_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "3 of 4 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5751-rust-restart-parity-repair",
                    "artifact_field": "distributional_parity",
                    "expected": True,
                    "actual": {"passed": True},
                    "passed": False,
                }
            ],
            "blocked_at_layer": "conductor_pre_gate",
        },
        mod.EXP5753_PATH: {
            "honest_verdict": "complete: generic_primitive_live_registry_ab_delta_0_registry_credit_0",
            "solve_provenance": "development_proxy",
            "public_game_count": 25,
            "registry_level_count": 183,
            "baseline_live_levels_reproduced": 1,
            "primitive_live_levels_reproduced": 1,
            "live_level_reproduction_delta": 0,
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
            "primitive_live_reachable": True,
            "source_leak_count": 0,
            "game_identity_leak_count": 0,
        },
    }


def _make_root(root: Path, *, omit: Path | None = None) -> None:
    for rel_path, payload in _artifact_payloads().items():
        if rel_path == omit:
            continue
        _write_json(root, rel_path, payload)
    _write_json(root, mod.EXP5746_PREFLIGHT_PATH, {"schema": "preflight", "preflight_ready": True})
    _write_json(root, Path("results/experiment_5749_adversarial_probe.json"), {"status": "clean"})

    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            continue
        if rel_path == mod.ROADMAP_RELATIVE_PATH:
            _write_text(root, rel_path, yaml.safe_dump(_roadmap_payload(), sort_keys=False))
        elif rel_path == mod.CONDUCTOR_LOG_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                "\n".join(
                    [
                        "| t | exp5743-transition-v513 | OK | done |",
                        "| t | exp5744-v513-source-delta-ingestion | OK | done |",
                        "| t | exp5745-arc-causal-gate-schema-corrigendum | OK | done |",
                        "| t | exp5746-exact-proposal-utility-benchmark | OK | done |",
                        "| t | exp5747-sota-exact-proposal-utility-panel | GATE_BLOCK | 3 failed |",
                        "| t | exp5748-selective-exact-feedback-search | GATE_BLOCK | upstream retired |",
                        "| t | exp5749-csl-render-matched-mechanism-audit | OK | negative residual |",
                        "| t | exp5750-dependent-task-continuous-self-learning | GATE_BLOCK | 1 failed |",
                        "| t | exp5751-rust-restart-parity-repair | OK | parity only |",
                        "| t | exp5752-one-axis-allocation-free-10x-crossover | GATE_BLOCK | schema gate |",
                        "| t | exp5753-arc-generic-primitive-live-registry-ab | OK | delta zero |",
                    ]
                )
                + "\n",
            )
        elif rel_path == mod.ARC_REGISTRY_RELATIVE_PATH:
            _write_text(
                root,
                rel_path,
                yaml.safe_dump({"reproducible_total_games": 25, "reproducible_total_levels": 183}),
            )
        else:
            _write_text(root, rel_path)
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor fixture\n")


def test_spec_contains_req_report_5754_contract() -> None:
    """REQ-REPORT-5754: the OpenSpec contract names the artifact and scenarios."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5754") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "SCENARIO-REPORT-5754-MISSING-AND-GATE-SKIPPED" in section
    assert "development_proxy" in section
    assert "experiment_5754_v513_capstone_reconciliation.py" in section


def test_scenario_report_5754_preserves_independent_branch_truth(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5754: branch verdicts stay separate and uninflated."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["honest_verdict"].startswith("complete:")
    assert report["proposal_transport_ready"] is True
    assert report["proposal_benchmark_ready"] is True
    assert report["proposal_exact_authority_receipts"] == {
        "structure_receipt_count": 180,
        "solution_receipt_count": 180,
        "exact_optimum_receipt_count": 180,
        "structure_receipt_failure_count": 0,
        "solution_receipt_failure_count": 0,
        "candidate_domain_incomplete_count": 0,
        "validator_disagreement_count": 0,
        "llm_inference_used": False,
    }
    assert report["proposal_utility_ready"] is False
    assert report["selective_feedback_ready"] is False
    assert report["missing_artifact_manifest"] == [
        {
            "task_id": "exp5748-selective-exact-feedback-search",
            "path": mod.EXP5748_PATH.as_posix(),
            "reason": "expected_artifact_missing",
        }
    ]

    assert set(report["gate_skip_manifest"]) == {
        "exp5747-sota-exact-proposal-utility-panel",
        "exp5748-selective-exact-feedback-search",
        "exp5750-dependent-task-continuous-self-learning",
        "exp5752-one-axis-allocation-free-10x-crossover",
    }
    assert report["conductor_outcomes"]["tasks"]["exp5748-selective-exact-feedback-search"][
        "outcome"
    ] == "GATE_BLOCK"
    assert report["conductor_outcomes"]["preflight_artifacts"][0]["path"] == (
        mod.EXP5746_PREFLIGHT_PATH.as_posix()
    )
    assert report["conductor_outcomes"]["adversarial_artifacts"][0]["path"].endswith(
        "experiment_5749_adversarial_probe.json"
    )

    assert report["continuous_self_learning_credited"] is True
    assert report["kan_mechanism_residual"] == -0.084269
    assert report["dependent_task_csl_ready"] is False
    assert report["rust_restart_parity_ready"] is True
    assert report["rust_batched_10x_ready"] is False
    assert report["rust_10x_retired"] is False
    assert report["arc_gate_schema_corrected"] is True
    assert report["arc_live_ab_completed"] is True
    assert report["arc_live_level_reproduction_delta"] == 0
    assert report["solve_provenance"] == "development_proxy"
    assert report["arc_registry_delta"] == 0
    assert report["arc_solve_credited"] is False
    assert report["timing_claimed"] is False
    assert report["software_speedup_claimed"] is False
    assert report["hardware_speedup_claimed"] is False
    assert report["hardware_status"]["fpga_lanes"]["kv260"]["status"] == "terminal"
    assert report["hardware_status"]["tsu"]["status"] == "watch_only"
    assert report["ops_reconciliation"]["mode"] == "deferred_to_reconciler"

    assert set(report).issubset(report["field_principles"])
    assert all(report["field_principles"][field] for field in report)
    assert report["reproducibility_checksum"]


def test_scenario_report_5754_missing_inputs_are_not_reconstructed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5754-MISSING-AND-GATE-SKIPPED: missing artifacts stay missing."""

    _make_root(tmp_path, omit=mod.EXP5746_PATH)
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    missing = {row["task_id"] for row in report["missing_artifact_manifest"]}
    assert "exp5746-exact-proposal-utility-benchmark" in missing
    assert "exp5748-selective-exact-feedback-search" in missing
    assert report["task_artifact_hashes"]["exp5746-exact-proposal-utility-benchmark"][
        "sha256"
    ] is None
    assert report["proposal_benchmark_ready"] is False
    assert report["proposal_exact_authority_receipts"]["structure_receipt_count"] == 0
    assert report["proposal_utility_ready"] is False
    assert report["selective_feedback_ready"] is False


def test_scenario_report_5754_emit_report_and_helpers_cover_edges(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5754-FIELD-PRINCIPLES: artifact writing is stable."""

    _make_root(tmp_path)
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    report = mod.emit_report(
        tmp_path,
        output_path=output,
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["reproducibility_checksum"] == report["reproducibility_checksum"]
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert mod._status_for_payload({}, {"exists": False}) == "missing"
    assert mod._status_for_payload({}, {"exists": True, "loadable": False}) == "malformed"
    assert mod._status_for_payload({"schema": "blocked_gate_check_v1"}, {}) == "gate_skipped"
    assert mod._status_for_payload({"status": "blocked"}, {}) == "gate_skipped"
    assert mod._status_for_payload({"flagged_adversarial": True}, {}) == "flagged"
    assert mod._status_for_payload({"honest_verdict": "blocked: no"}, {}) == "blocked"
    assert mod._status_for_payload({"honest_verdict": "complete: yes"}, {}) == "complete"
    assert mod._status_for_payload({"status": "weird"}, {}) == "weird"
    assert mod._status_for_payload({}, {}) == "unknown"
    assert mod._number_value({"x": True}, "x") == 1.0
    assert mod._number_value({"x": "bad"}, "x", default=2.0) == 2.0
    assert mod._bool_value({"x": "true"}, "x") is True
    assert mod._bool_value({"x": 0}, "x") is False
    assert mod._read_yaml_mapping(tmp_path / "missing.yaml") == {}
    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("scalar\n", encoding="utf-8")
    assert mod._read_yaml_mapping(scalar_yaml) == {}
    assert mod._latest_log_line("a\nneedle first\nneedle second\n", ("needle",)) == (
        "needle second"
    )
    assert mod._outcome_from_line("| t | task | OK | done |") == "OK"
    assert mod._outcome_from_line("| t | task | WEIRD | done |") == "LOGGED"
    assert mod._outcome_from_line(None) == "MISSING_LOG_LINE"
    assert mod._fallback_outcome("missing") == "MISSING"
    assert mod._fallback_outcome("surprise") == "UNKNOWN"
    assert mod._scan_auxiliary_artifacts(tmp_path / "empty", "preflight") == []
    assert mod._load_tests_run(None)[0]["status"] == "not_run"

    original = mod.FIELD_PRINCIPLES.pop("schema")
    try:
        with pytest.raises(KeyError, match="missing field principles"):
            mod.build_report(
                tmp_path,
                modification_overrides={
                    mod.ROADMAP_RELATIVE_PATH: False,
                    mod.CONDUCTOR_RELATIVE_PATH: False,
                },
            )
    finally:
        mod.FIELD_PRINCIPLES["schema"] = original

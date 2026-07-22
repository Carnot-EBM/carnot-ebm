"""Tests for the Exp5795 V516 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5795, SCENARIO-CAPSTONE-5795,
SCENARIO-CAPSTONE-5795-GATE-REPLAY,
SCENARIO-CAPSTONE-5795-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5795_v516_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5782-transition-v516": {
            "status": "complete",
            "honest_verdict": "complete: transition ready",
            "inference_substrate": "local_exact_artifact_and_conductor_reconciliation_no_llm",
            "next_range_collision_count": 0,
            "research_complete_append_count": 0,
            "test_exit_codes": {"focused": 0},
        },
        "exp5783-v516-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V516 source deltas",
            "inference_substrate": "primary_source_metadata_and_local_ledger_synthesis_no_experiment_llm",
            "accepted_finding_count": 0,
            "test_exit_codes": {"focused": 0},
        },
        "exp5784-evidence-index-terminal-qualification": {
            "status": "complete",
            "honest_verdict": "complete: exact index qualified",
            "inference_substrate": "local_filesystem_metadata_hashes_and_explicit_test_receipts_no_llm",
            "evidence_index_ready_score": 1.0,
            "next_range_collision_count": 0,
            "unresolved_canonical_count": 0,
            "history_mutation_count": 0,
            "producer_gate_fields": {
                "evidence_index_ready_score": 1.0,
                "next_range_collision_count": 0,
                "unresolved_canonical_count": 0,
                "history_mutation_count": 0,
            },
            "test_exit_codes": {"focused": 0},
        },
        "exp5785-hardness-surface-prospective-fixture": {
            "status": "complete",
            "honest_verdict": "complete: sealed_hardness_surface_exact_fixture_ready",
            "inference_substrate": "deterministic_local_fixture_generation_z3_and_exact_validators_no_llm",
            "fixture_ready_score": 1.0,
            "exact_label_coverage": 1.0,
            "parser_control_pass_rate": 1.0,
            "producer_gate_fields": [
                "fixture_ready_score",
                "exact_label_coverage",
                "parser_control_pass_rate",
            ],
            "test_exit_codes": {"focused": 0},
        },
        "exp5786-sota-hardness-controlled-constraint-stream": {
            "status": "complete",
            "honest_verdict": "complete: sota_constraint_response_stream_collected_not_ready:parser_failure_threshold",
            "inference_substrate": "real_local_llama_cpp_cuda_gguf_generation_plus_exact_z3_validation",
            "stream_ready_score": 0.0,
            "real_sota_model_count": 3,
            "exact_label_coverage": 1.0,
            "satisfiable_drift_count": 25,
            "protected_fact_distortion_count": 10,
            "failure_taxonomy_counts": {"parser_failure": 360, "protected_fact_distortion": 10},
            "gpu_offload_receipts": {"model": {"cuda_offload_authenticated": True}},
            "test_exit_codes": {"focused": 0},
        },
        "exp5787-validation-gated-constraint-skill-ab": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": (
                "2 of 5 gate(s) failed; first failure: "
                "exp5786-sota-hardness-controlled-constraint-stream.stream_ready_score"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp5786-sota-hardness-controlled-constraint-stream",
                    "artifact_field": "stream_ready_score",
                    "op": "==",
                    "expected": 1.0,
                    "actual": 0.0,
                    "passed": False,
                    "reason": "actual=0.0 == expected=1.0",
                },
                {
                    "upstream": "exp5786-sota-hardness-controlled-constraint-stream",
                    "artifact_field": "real_sota_model_count",
                    "op": ">=",
                    "expected": 3,
                    "actual": 3,
                    "passed": True,
                    "reason": "actual=3 >= expected=3",
                },
                {
                    "upstream": "exp5786-sota-hardness-controlled-constraint-stream",
                    "artifact_field": "exact_label_coverage",
                    "op": "==",
                    "expected": 1.0,
                    "actual": 1.0,
                    "passed": True,
                    "reason": "actual=1.0 == expected=1.0",
                },
                {
                    "upstream": "exp5786-sota-hardness-controlled-constraint-stream",
                    "artifact_field": "satisfiable_drift_count",
                    "op": ">=",
                    "expected": 30,
                    "actual": 25,
                    "passed": False,
                    "reason": "actual=25 >= expected=30",
                },
            ],
        },
        "exp5789-constraint-skill-shadow-adapter": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "5 of 5 gate(s) failed; upstream artifact not found",
            "gates_evaluated": [
                {
                    "upstream": "exp5788-constraint-skill-transfer-audit",
                    "artifact_field": "transfer_ready_score",
                    "op": "==",
                    "expected": 1.0,
                    "actual": None,
                    "passed": False,
                    "reason": "upstream artifact not found for task id 'exp5788-constraint-skill-transfer-audit'",
                }
            ],
        },
        "exp5790-arc-world-model-admission-contract": {
            "status": "complete",
            "honest_verdict": "complete: immutable_world_model_admission_contract_ready_no_solve_credit",
            "inference_substrate": "immutable_executable_world_model_replay_over_agent_owned_arc_transitions_no_llm",
            "admission_contract_ready_score": 1.0,
            "pivotal_fixture_coverage_score": 1.0,
            "source_leak_count": 0,
            "solve_claimed": False,
            "registry_credit": False,
            "test_exit_codes": {"focused": 0},
        },
        "exp5791-arc-sota-independent-hypothesis-panel": {
            "status": "blocked",
            "honest_verdict": "blocked: headline_gpu_offload_receipts_present",
            "inference_substrate": "real_local_llama_cpp_cuda_single_shot_model_synthesis_plus_immutable_agent_owned_arc_replay",
            "panel_ready_score": 0.0,
            "admissible_hypothesis_count": 0,
            "real_sota_model_count": 3,
            "solve_claimed": False,
            "registry_credit": False,
            "gpu_offload_receipts": [{"offload_ok": True, "headline_load_verified": False}],
            "test_exit_codes": {"focused": 0},
        },
        "exp5793-arc-live-world-model-ab": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "5 of 5 gate(s) failed; upstream artifact not found",
            "gates_evaluated": [
                {
                    "upstream": "exp5792-arc-calibration-only-selector",
                    "artifact_field": "selector_ready_score",
                    "op": "==",
                    "expected": 1.0,
                    "actual": None,
                    "passed": False,
                    "reason": "upstream artifact not found for task id 'exp5792-arc-calibration-only-selector'",
                }
            ],
        },
        "exp5794-hardware-terminal-action-receipt": {
            "status": "complete_cached_hardware_reconciliation_no_board_commands",
            "honest_verdict": "complete: cached hardware reconciliation no_speedup_claim no_energy_claim no_production_ready_claim",
            "inference_substrate": "exact_cached_hardware_artifact_reconciliation_with_changed_precondition_only_bounded_checks_no_llm",
            "changed_preconditions": {"kv260": False, "polarfire": False, "gatemate": False},
            "commands_run": [],
            "commands_skipped": [{"board": "kv260"}],
            "speedup_claimed": False,
            "energy_claimed": False,
            "production_ready_claimed": False,
            "test_exit_codes": {"focused": 0},
        },
    }
    return payloads[task_id]


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        row: JsonDict = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": mod.TASK_TITLES[task_id],
            "deliverable": rel_path.as_posix(),
        }
        if task_id in mod.GATE_DEFINITIONS:
            row["gated_on"] = mod.GATE_DEFINITIONS[task_id]
        if task_id == "exp5789-constraint-skill-shadow-adapter":
            row["prior_failures"] = [
                {
                    "experiment_id": "exp5775-constraint-sidecar-shadow-integration",
                    "verdict": "blocked_gate_check_failed",
                    "retire_if_same_verdict": True,
                }
            ]
        if task_id == "exp5793-arc-live-world-model-ab":
            row["prior_failures"] = [
                {
                    "experiment_id": "exp5767-arc-game-blind-composition-hardening",
                    "verdict": "blocked_gate_check_failed",
                    "retire_if_same_verdict": True,
                }
            ]
        tasks.append(row)
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _research_complete_payload() -> JsonDict:
    return {
        "milestones": [
            {
                "id": mod.MILESTONE,
                "tasks": [
                    {
                        "id": task_id,
                        "title": mod.TASK_TITLES[task_id],
                        "deliverable": rel_path.as_posix(),
                    }
                    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items()
                ],
            }
        ]
    }


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-22 04:19 UTC | Milestone 2026.07.516 activated | OK | 14 tasks queued |",
            "| 2026-07-22 07:35 UTC | Transition terminal .515 evidence and allocate col | OK | 89 passed, 1 warning in 10.89s |",
            "| 2026-07-22 08:33 UTC | Time-windowed literature freshness receipt | OK | 88 passed, 2 warnings in 11.17s |",
            "| 2026-07-22 09:30 UTC | Qualify the existing exact-deliverable index with  | OK | 93 passed, 1 warning in 35.64s |",
            "| 2026-07-22 10:15 UTC | Gated on Exp5784 readiness: build a sealed hardnes | OK | 87 passed, 1 warning in 20.73s |",
            "| 2026-07-22 11:06 UTC | Gated on Exp5785 fixture readiness: run the three- | OK | 85 passed, 1 warning in 18.78s |",
            "| 2026-07-22 11:08 UTC | Gated on Exp5786 clean drift headroom: run continu | GATE_BLOCK | 2 of 5 gate(s) failed |",
            "| 2026-07-22 11:10 UTC | Gated on Exp5786 clean drift headroom: run continu | GATE_BLOCK | 2 of 5 gate(s) failed |",
            "| 2026-07-22 11:12 UTC | Gated on Exp5786 clean drift headroom: run continu | GATE_BLOCK | 2 of 5 gate(s) failed |",
            "| 2026-07-22 11:14 UTC | Causal future-family holdout of versioned rule sta | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-07-22 11:18 UTC | Gated on Exp5788 transfer: wire a disabled typed-c | GATE_BLOCK | 5 of 5 gate(s) failed |",
            "| 2026-07-22 11:36 UTC | Pivotal-dynamics accreditation contract for immuta | OK | 87 passed, 1 warning in 11.18s |",
            "| 2026-07-22 12:22 UTC | Gated on Exp5790 admission readiness: run a matche | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-22 12:48 UTC | Gated on Exp5790 admission readiness: run a matche | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-22 13:19 UTC | Gated on Exp5790 admission readiness: run a matche | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-22 13:21 UTC | Frozen calibration chooser over immutable simulato | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-07-22 13:25 UTC | Gated on Exp5792 selector benefit: measure selecte | GATE_BLOCK | 5 of 5 gate(s) failed |",
            "| 2026-07-22 13:51 UTC | Board-state hash ledger and operator handoff packe | OK | 91 passed, 1 warning in 10.48s |",
        ]
    )


def _make_root(root: Path) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        if task_id in {"exp5788-constraint-skill-transfer-audit", "exp5792-arc-calibration-only-selector"}:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(_research_complete_payload()))
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments:\n")
    _write_text(root, mod.ARC_REGISTRY_RELATIVE_PATH, "reproducible_total_levels: 69\n")
    _write_text(root, mod.RESEARCH_REFERENCES_RELATIVE_PATH, "# refs\n")
    _write_text(root, mod.STATUS_RELATIVE_PATH, "# status\n")
    _write_text(root, mod.CHANGELOG_RELATIVE_PATH, "# changelog\n")
    _write_text(root, mod.KNOWN_ISSUES_RELATIVE_PATH, "# known\n")
    _write_text(root, mod.TRACEABILITY_RELATIVE_PATH, "# trace\n")
    _write_text(root, mod.PRD_RELATIVE_PATH, "# prd\n")
    _write_text(root, mod.ARCHITECTURE_RELATIVE_PATH, "# architecture\n")
    _write_text(root, mod.RESEARCH_PROGRAM_RELATIVE_PATH, "# program\n")
    _write_text(root, mod.CODEX_RELATIVE_PATH, "# codex\n")
    _write_text(root, mod.AGENTS_RELATIVE_PATH, "# agents\n")
    _write_text(root, mod.CLAUDE_RELATIVE_PATH, "# claude\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "REQ-CAPSTONE-5795\nSCENARIO-CAPSTONE-5795\n")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor\n")


def test_req_capstone_5795_spec_declares_exact_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5795: OpenSpec declares exact identity, gates, and no-promotion rules."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-5795") :]

    assert "results/experiment_5795_v516_capstone_reconciliation.json" in section
    assert "complete-positive, complete-null, complete-negative" in section
    assert "SCENARIO-CAPSTONE-5795-GATE-REPLAY" in section
    assert "cached hardware, development-proxy, tiny-model" in section


def test_scenario_capstone_5795_reconciles_fixture_denominator(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5795: every .516 task keeps its honest denominator class."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        tests_run=[{"command": "focused", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["positive_task_ids"] == [
        "exp5782-transition-v516",
        "exp5784-evidence-index-terminal-qualification",
        "exp5785-hardness-surface-prospective-fixture",
        "exp5790-arc-world-model-admission-contract",
        "exp5794-hardware-terminal-action-receipt",
    ]
    assert report["scientific_null_task_ids"] == ["exp5783-v516-source-delta-ingestion"]
    assert report["negative_task_ids"] == [
        "exp5786-sota-hardness-controlled-constraint-stream"
    ]
    assert report["blocked_gate_task_ids"] == [
        "exp5787-validation-gated-constraint-skill-ab",
        "exp5789-constraint-skill-shadow-adapter",
        "exp5793-arc-live-world-model-ab",
    ]
    assert report["missing_task_ids"] == [
        "exp5788-constraint-skill-transfer-audit",
        "exp5792-arc-calibration-only-selector",
    ]
    assert report["failed_delivery_task_ids"] == [
        "exp5791-arc-sota-independent-hypothesis-panel"
    ]
    assert report["blocked_precondition_task_ids"] == [
        "exp5791-arc-sota-independent-hypothesis-panel"
    ]
    assert report["canonical_artifact_hashes"]["exp5788-constraint-skill-transfer-audit"][
        "status"
    ] == "missing"
    assert report["canonical_task_matrix"]["exp5786-sota-hardness-controlled-constraint-stream"][
        "outcome_class"
    ] == "complete-negative"
    assert report["canonical_task_matrix"]["exp5791-arc-sota-independent-hypothesis-panel"][
        "delivery_failure_count"
    ] == 3
    assert report["research_complete_append_count"] == 0
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True


def test_scenario_capstone_5795_gate_replay_and_branch_decisions(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5795-GATE-REPLAY: replay explains blocked promotions."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    exp5787_gates = report["gate_replay_receipts"][
        "exp5787-validation-gated-constraint-skill-ab"
    ]["gates"]
    assert exp5787_gates[0]["artifact_field"] == "stream_ready_score"
    assert exp5787_gates[0]["passed"] is False
    assert exp5787_gates[3]["artifact_field"] == "satisfiable_drift_count"
    assert exp5787_gates[3]["passed"] is False
    assert report["gate_replay_receipts"]["exp5789-constraint-skill-shadow-adapter"][
        "discrepancies"
    ] == []
    assert report["gate_replay_receipts"]["exp5793-arc-live-world-model-ab"]["gates"][0][
        "actual"
    ] is None
    assert report["constraint_branch_decision"]["promoted"] is False
    assert report["constraint_branch_decision"]["default_enabled"] is False
    assert "exp5787-validation-gated-constraint-skill-ab" in report["constraint_branch_decision"][
        "blocking_task_ids"
    ]
    assert report["arc_branch_decision"]["promoted"] is False
    assert report["arc_branch_decision"]["default_enabled"] is False
    assert report["arc_registry_unchanged"] is True
    assert report["solve_claim_count"] == 0
    assert report["hardware_branch_decision"]["speedup_claimed"] is False


def test_scenario_capstone_5795_retirements_and_telemetry(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5795: repeat gate blocks retire narrowly and telemetry stays operational."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    receipts = report["prior_failure_retirement_receipts"]
    assert receipts["exp5789-constraint-skill-shadow-adapter"]["retirement_required"] is True
    assert receipts["exp5793-arc-live-world-model-ab"]["retirement_required"] is True
    assert report["canonical_task_matrix"]["exp5789-constraint-skill-shadow-adapter"][
        "prior_failure_retirement_result"
    ] == receipts["exp5789-constraint-skill-shadow-adapter"]
    assert report["retired_task_ids"] == [
        "exp5775-constraint-sidecar-shadow-integration",
        "exp5767-arc-game-blind-composition-hardening",
    ]
    assert report["retry_counts"]["exp5791-arc-sota-independent-hypothesis-panel"] == 2
    assert report["gate_skipped_agent_calls"] == 7
    assert report["phase_telemetry"]["integration"]["gate_block_count"] == 1
    assert report["phase_telemetry"]["arc"]["failed_delivery_count"] == 3
    assert report["avoidable_orchestration_time_min"] >= 0
    assert report["slowest_tasks"][0]["task_id"] == "exp5791-arc-sota-independent-hypothesis-panel"
    assert report["gpu_cpu_receipts"]["gpu_receipt_task_ids"] == [
        "exp5786-sota-hardness-controlled-constraint-stream",
        "exp5791-arc-sota-independent-hypothesis-panel",
    ]


def test_scenario_capstone_5795_emit_report_and_fail_closed_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-5795-FIELD-PRINCIPLES: artifact is stable and blocks overclaims."""

    _make_root(tmp_path)
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    report = mod.emit_report(
        tmp_path,
        output_path=output,
        tests_run=[{"command": "focused", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["reproducibility_checksum"] == report["reproducibility_checksum"]
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert set(report).issubset(report["field_principles"])
    assert report["docs_reconciled"]["ops_status_updated"] is False
    assert report["traceability_reconciled"] is False
    assert report["public_claims_changed"] is False
    assert mod._read_yaml_mapping(tmp_path / "missing.yaml")[1]["present"] is False
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("bad: [yaml\n", encoding="utf-8")
    assert mod._read_yaml_mapping(bad_yaml)[1]["parsed"] is False
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    assert mod._read_yaml_mapping(list_yaml)[1]["error"] == "expected mapping, got list"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json_mapping(bad_json)[1]["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json_mapping(list_json)[1]["error"] == "expected mapping, got list"
    assert mod._compare_gate("bad", 1, 1) is False
    assert mod._compare_gate(">", 2, 1) is True
    assert mod._coerce_scalar("true") is True
    assert mod._coerce_scalar("false") is False
    assert mod._coerce_scalar("1.5") == 1.5
    assert mod._coerce_scalar("not-a-number") == "not-a-number"
    assert mod._coerce_scalar({"wrapped": 1}) is None
    assert mod._parse_conductor_log(tmp_path / "missing-log-root") == []
    odd_log_root = tmp_path / "odd-log-root"
    _write_text(odd_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "not a row\n| short |\n")
    assert mod._parse_conductor_log(odd_log_root) == []
    assert mod._parse_timestamp("not a timestamp") is None
    monkeypatch.setattr(mod, "_git_modified", lambda _root, _rel_path: True)
    assert mod._protected_modified(tmp_path, Path("untracked"), {}) is True
    assert mod._artifact_status({}, {"present": True, "loadable": False}) == "malformed"
    assert (
        mod._artifact_status({"status": "blocked"}, {"present": True, "loadable": True})
        == "blocked-precondition"
    )
    assert mod._artifact_status({}, {"present": True, "loadable": True}) == "unknown"
    assert (
        mod._classify_task("exp5791-arc-sota-independent-hypothesis-panel", {}, {"status": "malformed"}, {})[0]
        == "blocked-precondition"
    )

    payloads, hashes = mod._load_artifacts(tmp_path)
    fallback_receipt = mod._replay_task_gates(
        "exp5790-arc-world-model-admission-contract",
        payloads,
        hashes,
        {},
    )
    assert fallback_receipt["gates"]
    payloads_with_mismatch = {task_id: dict(payload) for task_id, payload in payloads.items()}
    payloads_with_mismatch["exp5787-validation-gated-constraint-skill-ab"] = {
        **payloads_with_mismatch["exp5787-validation-gated-constraint-skill-ab"],
        "gates_evaluated": [
            {
                "upstream": "exp5786-sota-hardness-controlled-constraint-stream",
                "artifact_field": "stream_ready_score",
                "actual": 99,
                "passed": True,
            }
        ],
    }
    mismatch_receipt = mod._replay_task_gates(
        "exp5787-validation-gated-constraint-skill-ab",
        payloads_with_mismatch,
        hashes,
        {"gated_on": mod.GATE_DEFINITIONS["exp5787-validation-gated-constraint-skill-ab"]},
    )
    assert mismatch_receipt["discrepancies"] == [
        "passed_mismatch:exp5786-sota-hardness-controlled-constraint-stream.stream_ready_score",
        "actual_mismatch:exp5786-sota-hardness-controlled-constraint-stream.stream_ready_score",
    ]
    prior_receipts = mod._prior_failure_receipts(
        payloads,
        {"exp5789-constraint-skill-shadow-adapter": {"prior_failures": ["not-a-mapping"]}},
        "",
    )
    assert prior_receipts["exp5789-constraint-skill-shadow-adapter"]["prior_failures"] == []

    modified = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )
    assert modified["status"] == "blocked"
    assert "research_roadmap_modified" in modified["preconditions_checked"][
        "failed_preconditions"
    ]
    assert "research_conductor_modified" in modified["preconditions_checked"][
        "failed_preconditions"
    ]

    bad_active_root = tmp_path / "bad-active"
    _make_root(bad_active_root)
    _write_text(bad_active_root, mod.ROADMAP_RELATIVE_PATH, "bad: [yaml\n")
    bad_active = mod.build_report(
        bad_active_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "active_roadmap_unparseable" in bad_active["preconditions_checked"]["failed_preconditions"]

    wrong_milestone_root = tmp_path / "wrong-milestone"
    _make_root(wrong_milestone_root)
    wrong_roadmap = _roadmap_payload()
    wrong_roadmap["milestone"] = "2026.07.515"
    _write_text(wrong_milestone_root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(wrong_roadmap))
    wrong_milestone = mod.build_report(
        wrong_milestone_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "active_roadmap_milestone='2026.07.515'" in wrong_milestone["preconditions_checked"][
        "failed_preconditions"
    ]

    mismatch_root = tmp_path / "mismatch"
    _make_root(mismatch_root)
    mismatch_roadmap = _roadmap_payload()
    mismatch_roadmap["tasks"][0]["deliverable"] = "results/wrong.json"
    _write_text(mismatch_root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(mismatch_roadmap))
    mismatch = mod.build_report(
        mismatch_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "declared_deliverable_mismatch:exp5782-transition-v516" in mismatch[
        "preconditions_checked"
    ]["failed_preconditions"]

    original = mod.FIELD_PRINCIPLES.pop("status")
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
        mod.FIELD_PRINCIPLES["status"] = original

    monkeypatch.setattr(mod, "EXPECTED_TASK_IDS", ("exp9999-missing",))
    with pytest.raises(KeyError):
        mod.build_report(
            tmp_path,
            modification_overrides={
                mod.ROADMAP_RELATIVE_PATH: False,
                mod.CONDUCTOR_RELATIVE_PATH: False,
            },
        )

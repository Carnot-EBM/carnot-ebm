"""Tests for the Exp5931 V526 capstone reconciliation.

Spec refs: REQ-REPORT-5931, REQ-CAPSTONE-5931,
SCENARIO-REPORT-5931-EXACT-MATRIX, SCENARIO-REPORT-5931-TERMINAL-CLASSES,
SCENARIO-REPORT-5931-BRANCH-SEMANTICS,
SCENARIO-REPORT-5931-APPEND-RETIRE-AND-RECOMMEND.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import runpy
import sys
from typing import Any

import pytest
import yaml

from carnot import experiment_5931_v526_capstone_reconciliation as mod


JsonDict = dict[str, Any]

TASKS: tuple[tuple[str, str, str], ...] = (
    (
        "exp5918-transition-v526",
        "Exact terminal-boundary handoff from .525 into .526",
        "results/experiment_5918_transition_v526.json",
    ),
    (
        "exp5919-v526-source-delta-ingestion",
        "Dated evidence refresh after the V526 planner marker",
        "results/experiment_5919_v526_source_delta_ingestion.json",
    ),
    (
        "exp5920-prospective-event-stream-admission",
        "Prospective stream admission and task-owned execution boundary",
        "results/experiment_5920_prospective_event_stream_admission.json",
    ),
    (
        "exp5921-schema-derived-constraintir-support",
        "Schema compiler and adversarial fixture",
        "results/experiment_5921_schema_derived_constraintir_support.json",
    ),
    (
        "exp5922-gguf-schema-decoder-bridge",
        "Gated on Exp5921 ready: embedded-GGUF schema decoder bridge",
        "results/experiment_5922_gguf_schema_decoder_bridge.json",
    ),
    (
        "exp5923-sota-schema-supported-constraintir-ab",
        "Gated on Exp5922 ready: all-three-model schema-supported ConstraintIR A/B",
        "results/experiment_5923_sota_schema_supported_constraintir_ab.json",
    ),
    (
        "exp5924-transactional-constraint-memory-v2",
        "Gated on Exp5920 ready: transactional constraint-memory clean-room fixture",
        "results/experiment_5924_transactional_constraint_memory_v2.json",
    ),
    (
        "exp5925-sota-transactional-csl-prospective",
        "Gated on Exp5923 stream and Exp5924 fixture: prospective all-three-model continuous self-learning",
        "results/experiment_5925_sota_transactional_csl_prospective.json",
    ),
    (
        "exp5926-adaptive-state-abi-v2-parity",
        "Gated on Exp5924 ready: adaptive-state ABI v2 Python/Rust/PyO3 parity",
        "results/experiment_5926_adaptive_state_abi_v2_parity.json",
    ),
    (
        "exp5927-coordinate-router-progress-qualification",
        "Powered coordinate-router qualification",
        "results/experiment_5927_coordinate_router_progress_qualification.json",
    ),
    (
        "exp5928-arc-live-runner-execution-binding",
        "Actual live-runner execution binding",
        "results/experiment_5928_arc_live_runner_execution_binding.json",
    ),
    (
        "exp5929-arc-structured-memory-bound-live-ab",
        "Gated on Exp5928 ready: adapter-disabled held structured-memory live A/B",
        "results/experiment_5929_arc_structured_memory_bound_live_ab.json",
    ),
    (
        "exp5930-adaptive-state-board-mapping",
        "Gated on Exp5926 ready: adaptive-state ABI v2 attached-board capability mapping",
        "results/experiment_5930_adaptive_state_board_mapping.json",
    ),
    (
        "exp5931-v526-capstone-reconciliation",
        "Branch-independent .526 capstone and exact reconciliation",
        "results/experiment_5931_v526_capstone_reconciliation.json",
    ),
)


def _write_text(root: Path, rel_path: str, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: str, payload: JsonDict) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml(root: Path, rel_path: str, payload: JsonDict) -> None:
    _write_text(root, rel_path, yaml.safe_dump(payload, sort_keys=False))


def _artifact_for(task_id: str) -> JsonDict:
    base: JsonDict = {
        "status": "complete",
        "honest_verdict": "complete: placeholder",
        "inference_substrate": "deterministic_fixture",
    }
    if task_id == "exp5918-transition-v526":
        return {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked: Exp5918 transition preconditions failed",
            "failed_preconditions": ["required_tests_failed"],
        }
    if task_id == "exp5919-v526-source-delta-ingestion":
        return {
            **base,
            "honest_verdict": "complete_null: no accepted post-V526 source deltas",
            "accepted_finding_count": 0,
        }
    if task_id == "exp5920-prospective-event-stream-admission":
        return {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: prospective_event_stream_admission_boundary_ready",
            "prospective_stream_admission_ready_score": 1.0,
            "chronology_split_and_visibility_receipts": {
                "event_order_is_chronological": True,
                "future_label_visibility_count": 0,
                "split_counts": {"train": 3, "held": 2},
            },
            "exact_label_authority": {
                "authority": "exact_verifier",
                "visible_after_proposal_only": True,
            },
        }
    if task_id == "exp5921-schema-derived-constraintir-support":
        return {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: schema support is structural only",
            "schema_decode_contract_ready_score": 1.0,
            "schema_to_grammar_type_scope_compiler_receipt": {
                "mechanically_derived_from_signature_schema": True
            },
            "semantic_authority_boundary": {
                "grammar_type_scope_counted_as_semantic_correct": False,
                "unsafe_semantic_acceptance_count": 0,
            },
        }
    if task_id == "exp5922-gguf-schema-decoder-bridge":
        return {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: tokenizer bridge ready",
            "gguf_schema_decoder_bridge_ready_score": 1.0,
            "gate_replay_receipt": {"ok": True, "exp5921_artifact_present": True},
            "per_model_terminal_token_mapping": {"qwen": {}, "gemma31": {}, "gemma26": {}},
        }
    if task_id == "exp5923-sota-schema-supported-constraintir-ab":
        return {
            **base,
            "status": "retired",
            "honest_verdict": "retired: schema-supported ConstraintIR decoding failed exact-semantic retirement gates",
            "schema_decode_live_ready_score": 0.0,
            "exact_semantic_primary_comparison_and_intervals": {
                "held_exact_semantic_delta_vs_best_control": 0.0,
                "by_arm_held": {"direct": {"exact_semantic_success_rate": 0.0}},
            },
            "per_model_arm_structural_and_exact_metrics": {
                "by_arm": {"schema_first_ir_token": {"parse_rate": 1.0}}
            },
            "chronological_event_stream_ready_score": 1.0,
            "retirement_decision": {
                "retire": True,
                "next_action": "retire_mechanism_no_reprompt",
                "reasons": ["all_three_models_zero_exact_success_in_schema_supported_arms"],
            },
        }
    if task_id == "exp5924-transactional-constraint-memory-v2":
        return {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: transactional_constraint_memory_v2_ready",
            "transactional_memory_fixture_ready_score": 1.0,
            "gate_replay_receipt": {"exp5920_ready_score": 1.0, "stream_replay_ok": True},
            "poison_burst_quarantine_recovery_and_retention": {
                "protected_prefix_retention_score": 1.0,
                "poison_quarantined_count": 2,
                "rollback_recovery_hash_matches": True,
            },
            "hardware_mapping_contract": {"finite_operation_set": True},
        }
    if task_id == "exp5926-adaptive-state-abi-v2-parity":
        return {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: adaptive_state_abi_v2_python_rust_pyo3_parity",
            "adaptive_state_abi_v2_ready_score": 1.0,
            "gate_replay_receipt": {"exp5924_ready_score": 1.0, "operations_present": True},
            "adaptive_state_abi_v2_schema_and_operations": {
                "supported_operations": ["snapshot", "commit", "rollback", "recover"]
            },
            "byte_state_status_and_error_parity": {
                "byte_parity": True,
                "status_error_parity": True,
            },
            "crash_prefix_recovery_and_rollback": {
                "crash_prefix_exact": True,
                "rollback_exact": True,
            },
        }
    if task_id == "exp5927-coordinate-router-progress-qualification":
        return {
            **base,
            "status": "complete_underpowered",
            "honest_verdict": "complete_underpowered: hard_progress_positive_count_27_below_30_no_promotion",
            "coordinate_router_progress_ready_score": 0.0,
            "hard_progress_positive_count_and_power_gate": {
                "hard_progress_positive_count": 27,
                "min_positive_rows": 30,
                "powered": False,
            },
            "no_level_solve_or_registry_update": {
                "offline_qualification_receives_level_credit": False,
                "no_level_solve_claimed": True,
            },
            "registry_precheck_receipt": {"registry_sha256": "sha256:registry"},
        }
    if task_id == "exp5928-arc-live-runner-execution-binding":
        return {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: actual_child_live_runner_execution_binding_ready",
            "live_runner_execution_binding_ready_score": 1.0,
            "actual_live_entrypoint_consumption_receipt": {
                "actual_live_entrypoint": "runner",
                "capability_consumed_before_environment_action": True,
                "fixture_only_validation": False,
                "model_load_count": 0,
                "level_attempt_count": 0,
            },
            "registry_unchanged": {
                "registry_hash_before": "sha256:registry",
                "registry_hash_after": "sha256:registry",
                "unchanged": True,
            },
        }
    if task_id == "exp5929-arc-structured-memory-bound-live-ab":
        return {
            **base,
            "status": "blocked_precondition",
            "honest_verdict": "blocked_precondition: bound_live_runner",
            "structured_memory_live_ready_score": 0.0,
            "actual_bound_e3_entrypoint_receipt": {
                "actual_capability_bound_adapter_disabled_e3_path": True,
                "live_inference_started": False,
                "model_load_count": 0,
                "level_attempt_count": 0,
            },
            "primary_live_utility_comparison_and_intervals": {
                "complete_bound_live_rows": 0,
                "groups": [],
            },
            "registry_unchanged": True,
        }
    if task_id == "exp5930-adaptive-state-board-mapping":
        return {
            **base,
            "status": "complete_static_mapping_no_physical_probe",
            "honest_verdict": "complete_static_mapping: ABI v2 trace parity complete; physical_probe_executed=false; no speed power energy claim",
            "board_abi_mapping_ready_score": 1.0,
            "physical_probe_executed": False,
            "abi_v2_schema_hash_and_operation_mapping": {
                "schema_hash": "sha256:abi",
                "operation_mapping": {"commit": {"present_in_exp5926": True}},
            },
            "simulator_reference_trace_parity": {"state_hash_parity": True},
            "static_synthesis_timing_estimate_and_resource_reports": {
                "board_targets": ["gatemate"]
            },
            "kv260_polarfire_and_gatemate_state_receipts": {
                "kv260": "programmed_image_poc",
                "polarfire": "prior_physical_workload_only",
                "gatemate": "blocked_idcode",
            },
        }
    raise AssertionError(task_id)


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in (
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "scripts/adversarial_verify.py",
        "scripts/in_process_doc_reconcile.py",
        "scripts/research_conductor.py",
        "ops/status.md",
        "ops/changelog.md",
        "ops/conductor-log.md",
        "_bmad/traceability.md",
        "ops/arc_solve_registry.yaml",
    ):
        _write_text(root, rel, f"{rel}\nREQ-REPORT-5931\n")
    _write_text(root, "openspec/capabilities/research-reporting/spec.md", "REQ-REPORT-5931\n")
    _write_text(root, "openspec/capabilities/capstone/spec.md", "REQ-CAPSTONE-5931\n")
    _write_text(root, "openspec/change-proposals/research-roadmap-vNEXT.md", "2026.07.526\n")
    _write_yaml(root, "ops/exclusion_manifest.yaml", {"retired": [], "retired_experiments": []})
    _write_yaml(
        root,
        "research-roadmap.yaml",
        {
            "milestone": "2026.07.526",
            "milestone_title": "V526",
            "tasks": [
                {
                    "id": task_id,
                    "milestone": "2026.07.526",
                    "title": title,
                    "deliverable": deliverable,
                    "prior_failures": (
                        [
                            {
                                "experiment_id": "exp5909-sota-constraint-synthesis-ab",
                                "verdict": "complete_null: structured prompt arms did not improve exact synthesis",
                                "retire_if_same_verdict": True,
                            }
                        ]
                        if task_id == "exp5923-sota-schema-supported-constraintir-ab"
                        else [
                            {
                                "experiment_id": "exp5917-v525-capstone-reconciliation",
                                "verdict": "complete_with_nulls: .525 reconciled by exact declared deliverables",
                                "retire_if_same_verdict": True,
                            }
                        ]
                        if task_id == "exp5931-v526-capstone-reconciliation"
                        else []
                    ),
                }
                for task_id, title, deliverable in TASKS
            ],
        },
    )
    _write_yaml(
        root,
        "research-complete.yaml",
        {
            "milestones": [
                {"id": "2026.07.525", "tasks": []},
                {"id": "2026.07.525", "tasks": []},
            ]
        },
    )
    log_lines = ["| ts | Milestone 2026.07.526 activated | OK | 14 tasks queued |"]
    for task_id, title, _deliverable in TASKS:
        status = "OK"
        if task_id == "exp5925-sota-transactional-csl-prospective":
            status = "GATE_BLOCK"
        log_lines.append(f"| ts | {title[:55]} | {status} | receipt for {task_id} |")
    _write_text(root, "ops/conductor-log.md", "\n".join(log_lines) + "\n")
    for task_id, _title, deliverable in TASKS:
        if task_id in {
            "exp5925-sota-transactional-csl-prospective",
            "exp5931-v526-capstone-reconciliation",
        }:
            continue
        _write_json(root, deliverable, _artifact_for(task_id))
    return root


def _adversarial_receipt(root: Path) -> JsonDict:
    reports = []
    for task_id, _title, deliverable in TASKS:
        if task_id in {
            "exp5925-sota-transactional-csl-prospective",
            "exp5931-v526-capstone-reconciliation",
        }:
            continue
        reports.append(
            {
                "artifact": deliverable,
                "loaded": (root / deliverable).exists(),
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
            }
        )
    return {
        "command": ".venv/bin/python scripts/adversarial_verify.py --json ...",
        "exit_code": 0,
        "stdout_sha256": "sha256:verifier",
        "flagged_count": 0,
        "warnings": [],
        "reports": reports,
    }


def _test_results() -> list[JsonDict]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5931_v526_capstone_reconciliation.py -q --no-cov -n 0",
            "exit_code": 0,
            "summary": "focused Exp5931 tests passed",
        },
        {
            "command": ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5931_v526_capstone_reconciliation.py --fail-under=100",
            "exit_code": 0,
            "summary": "new module coverage 100%",
        },
    ]


def test_req_5931_builds_exact_matrix_and_terminal_classes(tmp_path: Path) -> None:
    root = _make_root(tmp_path)

    payload = mod.reconcile_v526(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=False,
        update_ledgers=False,
    )

    matrix = payload["activated_task_and_declared_deliverable_matrix"]
    assert matrix["activated_task_count"] == 14
    assert [row["task_id"] for row in matrix["tasks"]] == [
        task_id for task_id, _title, _path in TASKS
    ]
    assert matrix["selection_policy"] == "exact_declared_deliverable"
    assert all(row["identity"][2] == row["declared_deliverable"] for row in matrix["tasks"])

    classes = payload["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"]["exp5918-transition-v526"] == "blocked-precondition"
    assert classes["terminal_class_by_task_id"]["exp5919-v526-source-delta-ingestion"] == "null"
    assert (
        classes["terminal_class_by_task_id"]["exp5923-sota-schema-supported-constraintir-ab"]
        == "retired"
    )
    assert (
        classes["terminal_class_by_task_id"]["exp5925-sota-transactional-csl-prospective"]
        == "gate-blocked"
    )
    assert (
        classes["terminal_class_by_task_id"]["exp5927-coordinate-router-progress-qualification"]
        == "underpowered"
    )
    assert (
        classes["terminal_class_by_task_id"]["exp5929-arc-structured-memory-bound-live-ab"]
        == "blocked-precondition"
    )
    assert (
        classes["terminal_class_by_task_id"]["exp5930-adaptive-state-board-mapping"] == "no-change"
    )
    assert classes["all_activated_classified_once"] is True


def test_req_5931_preserves_branch_independence_and_semantic_boundaries(
    tmp_path: Path,
) -> None:
    root = _make_root(tmp_path)

    payload = mod.reconcile_v526(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=False,
        update_ledgers=False,
    )

    branch = payload["branch_independence_receipt"]
    assert branch["branch_independence_preserved"] is True
    assert branch["branch_classes"]["schema_constraint_ir"] == [
        "positive",
        "positive",
        "retired",
    ]
    assert branch["branch_classes"]["continuous_self_learning"] == [
        "positive",
        "gate-blocked",
        "positive",
    ]
    assert branch["branch_classes"]["arc"] == ["underpowered", "positive", "blocked-precondition"]
    assert branch["branch_classes"]["hardware"] == ["no-change"]

    schema = payload["schema_constraintir_structural_and_exact_semantic_receipt"]
    assert schema["structural_support_ready"] is True
    assert schema["structural_validity_is_semantic_success"] is False
    assert schema["exact_semantic_success_promoted"] is False
    assert schema["exp5923_terminal_class"] == "retired"

    arc = payload["arc_coordinate_execution_binding_and_live_receipt"]
    assert arc["offline_qualification_receives_live_credit"] is False
    assert arc["actual_execution_binding_ready"] is True
    assert arc["live_rows_completed"] == 0

    hardware = payload["adaptive_state_abi_and_hardware_receipt"]
    assert hardware["abi_v2_ready"] is True
    assert hardware["static_mapping_ready"] is True
    assert hardware["static_mapping_is_physical_acceleration"] is False
    assert hardware["physical_probe_executed"] is False


def test_req_5931_append_retirement_recommendations_and_schema_are_exact(
    tmp_path: Path,
) -> None:
    root = _make_root(tmp_path)
    before_registry = (root / "ops/arc_solve_registry.yaml").read_bytes()
    before_status = (root / "ops/status.md").read_bytes()

    payload = mod.reconcile_v526(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=True,
        update_ledgers=True,
    )
    second = mod.reconcile_v526(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=True,
        update_ledgers=True,
    )

    assert payload["research_complete_append_receipt"]["append_count"] == 1
    assert second["research_complete_append_receipt"]["append_count"] == 0
    complete = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    assert [row["id"] for row in complete["milestones"]].count("2026.07.526") == 1
    exp526 = [row for row in complete["milestones"] if row["id"] == "2026.07.526"][0]
    assert len(exp526["tasks"]) == 14
    assert exp526["tasks"][7]["id"] == "exp5925-sota-transactional-csl-prospective"
    assert "gate-blocked" in exp526["tasks"][7]["result"]

    retirement = payload["prior_failure_and_retirement_decisions"]
    assert retirement["manifest_append_count"] == 1
    assert second["prior_failure_and_retirement_decisions"]["manifest_append_count"] == 0
    assert retirement["retire_if_same_verdict_decisions"][0]["task_id"] == (
        "exp5923-sota-schema-supported-constraintir-ab"
    )
    assert retirement["retire_if_same_verdict_decisions"][0]["same_verdict_recurred"] is True
    assert all(
        row["task_id"] != "exp5931-v526-capstone-reconciliation"
        or row["same_verdict_recurred"] is False
        for row in retirement["prior_failure_audit"]
    )

    recommendations = payload["next_three_falsifiable_recommendations"]
    assert len(recommendations) == 3
    for row in recommendations:
        assert row["prerequisites"]
        assert row["stop_rules"]
        assert row["authority_boundaries"]
        assert row["excluded_scopes"]
        assert row["falsifiable_success_condition"]
        text = json.dumps(row)
        assert "exp5923" not in text.lower()
        assert "exp5925" not in text.lower()

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_provenance"]
        assert field in payload["field_principles"]
    assert payload["inference_substrate"] == "aggregation_from_exact_declared_artifacts"
    assert payload["honest_verdict"].startswith("complete_with_nulls:")
    assert payload["duplicate_history_amplification_count"] == 0
    assert payload["registry_unchanged"]["unchanged"] is True
    assert payload["protected_files_unchanged"]["all_unchanged"] is True
    assert (root / "ops/arc_solve_registry.yaml").read_bytes() == before_registry
    assert (root / "ops/status.md").read_bytes() == before_status
    assert (root / "results/experiment_5931_v526_capstone_reconciliation.json").exists()


def test_req_5931_terminal_defensive_paths_are_explicit(tmp_path: Path) -> None:
    assert mod._terminal_class(
        "exp5999-synthetic", {}, {"present": True}, {}, {"flag_count": 1, "max_severity": 2}
    ) == ("missing", "adversarial-verifier-critical")
    assert mod._terminal_class(
        "exp5999-synthetic", {}, {"present": False}, {"latest_status": "OK"}, {}
    ) == ("missing", "declared-deliverable-missing")
    assert mod._terminal_class(
        "exp5999-synthetic",
        {"status": "complete_underpowered", "honest_verdict": "complete_underpowered: n"},
        {"present": True},
        {},
        {},
    ) == ("underpowered", "underpowered")
    assert mod._terminal_class(
        "exp5999-synthetic",
        {"status": "complete_static_mapping_no_physical_probe"},
        {"present": True},
        {},
        {},
    ) == ("no-change", "static-mapping-no-physical-probe")
    assert mod._terminal_class(
        "exp5999-synthetic", {"status": "unknown"}, {"present": True}, {}, {}
    ) == ("missing", "unrecognized-terminal-treated-as-missing")

    root = _make_root(tmp_path)
    _write_yaml(root, "ops/exclusion_manifest.yaml", {"retired": []})
    payload = mod.reconcile_v526(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=False,
        update_ledgers=True,
    )
    assert payload["prior_failure_and_retirement_decisions"]["manifest_append_count"] == 1

    assert not any(
        re.search(r"numeric-prefix", json.dumps(row).lower())
        for row in payload["activated_task_and_declared_deliverable_matrix"]["tasks"]
    )


def test_req_5931_defensive_loaders_blocked_status_and_cli_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _make_root(tmp_path)

    bad_json = root / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    payload, meta = mod._read_json(bad_json)
    assert payload == {}
    assert meta["error"].startswith("json_error:")
    non_mapping_json = root / "array.json"
    non_mapping_json.write_text("[]", encoding="utf-8")
    assert mod._read_json(non_mapping_json)[1]["error"] == "json_not_mapping"

    bad_yaml = root / "bad.yaml"
    bad_yaml.write_text("key: [", encoding="utf-8")
    assert mod._read_yaml(bad_yaml)[1]["error"].startswith("yaml_error:")
    scalar_yaml = root / "scalar.yaml"
    scalar_yaml.write_text("- item\n", encoding="utf-8")
    assert mod._read_yaml(scalar_yaml)[1]["error"] == "yaml_not_mapping"

    assert mod._terminal_class(
        "exp5999-synthetic",
        {"schema": "blocked_gate_check_v1"},
        {"present": True},
        {},
        {},
    ) == ("gate-blocked", "conductor-gate-check")
    assert mod._terminal_class(
        "exp5999-synthetic",
        {"status": "complete", "honest_verdict": "complete: done"},
        {"present": True},
        {},
        {},
    ) == ("positive", "complete")
    assert mod._current_task_retirement_recurred(
        "exp5998-synthetic",
        {
            "repeated_verdict_retirement_decision": {
                "retire_if_same_verdict": True,
                "same_verdict_recurred": True,
            }
        },
    ) == (True, "artifact_repeated_verdict_retirement_decision")

    tasks = [
        {"id": "exp5998-bad-priors", "prior_failures": "not-a-list"},
        {"id": "exp5997-bad-prior-row", "prior_failures": ["not-a-mapping"]},
    ]
    retirement = mod._retirement_decisions(root, tasks, {}, update_ledgers=False)
    assert retirement["prior_failure_audit"] == []

    def _raise_oserror(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(mod.os, "replace", _raise_oserror)
    assert mod._atomic_output_receipt(root)["ok"] is False

    receipt = _adversarial_receipt(root)
    receipt["reports"][0]["flag_count"] = 1
    receipt["reports"][0]["max_severity"] = 2
    blocked = mod.reconcile_v526(
        root=root,
        adversarial_receipt=receipt,
        test_results=_test_results(),
        write=False,
        update_ledgers=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert (
        "exp5918-transition-v526"
        in blocked["exact_terminal_classification"]["invalid_artifact_task_ids"]
    )

    roadmap = yaml.safe_load((root / "research-roadmap.yaml").read_text(encoding="utf-8"))
    roadmap["milestone"] = "2026.07.999"
    _write_yaml(root, "research-roadmap.yaml", roadmap)
    wrong_milestone = mod.reconcile_v526(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=False,
        update_ledgers=False,
    )
    assert wrong_milestone["status"] == "blocked"

    receipt_path = root / "receipt.json"
    _write_json(root, "receipt.json", _adversarial_receipt(root))
    test_results_path = root / "test-results.json"
    _write_json(root, "test-results.json", {"results": _test_results()})
    assert mod._load_receipt(None) == {}
    assert mod._load_test_results(None) == []
    assert mod._load_receipt(receipt_path)["exit_code"] == 0
    assert len(mod._load_test_results(test_results_path)) == len(_test_results())
    list_results_path = root / "test-results-list.json"
    _write_text(root, "test-results-list.json", json.dumps(_test_results()))
    assert len(mod._load_test_results(list_results_path)) == len(_test_results())
    invalid_results_path = root / "test-results-invalid.json"
    _write_text(root, "test-results-invalid.json", json.dumps("bad"))
    with pytest.raises(SystemExit):
        mod._load_test_results(invalid_results_path)
    with pytest.raises(SystemExit):
        mod._load_receipt(bad_json)

    roadmap["milestone"] = "2026.07.526"
    _write_yaml(root, "research-roadmap.yaml", roadmap)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m carnot.experiment_5931_v526_capstone_reconciliation",
            "--root",
            str(root),
            "--adversarial-receipt",
            str(receipt_path),
            "--test-results",
            str(test_results_path),
            "--no-write",
            "--no-ledgers",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("carnot.experiment_5931_v526_capstone_reconciliation", run_name="__main__")
    assert excinfo.value.code == 0
    assert '"experiment_id": "exp5931-v526-capstone-reconciliation"' in capsys.readouterr().out

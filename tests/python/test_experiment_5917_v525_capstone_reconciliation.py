"""Tests for the Exp5917 V525 capstone reconciliation.

Spec refs: REQ-REPORT-5917, SCENARIO-REPORT-5917-EXACT-MATRIX,
SCENARIO-REPORT-5917-DISJOINT-TERMINALS,
SCENARIO-REPORT-5917-CSL-ARC-DISCIPLINE,
SCENARIO-REPORT-5917-APPEND-RETIRE-AND-SCHEMA.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot import experiment_5917_v525_capstone_reconciliation as mod


JsonDict = dict[str, Any]

TASKS: tuple[tuple[str, str, str], ...] = (
    (
        "exp5905-transition-v525",
        "Exact terminal-boundary handoff from .524 into .525",
        "results/experiment_5905_transition_v525.json",
    ),
    (
        "exp5906-v525-source-delta-ingestion",
        "Dated evidence refresh after the V525 planner marker",
        "results/experiment_5906_v525_source_delta_ingestion.json",
    ),
    (
        "exp5907-constraint-ir-replay-contract",
        "Canonical producer-consumer replay contract for typed ConstraintIR",
        "results/experiment_5907_constraint_ir_replay_contract.json",
    ),
    (
        "exp5908-verisynth-constraint-fixture",
        "Gated on Exp5907 replay: VeriSynth-style decomposition and retrieval fixture",
        "results/experiment_5908_verisynth_constraint_fixture.json",
    ),
    (
        "exp5909-sota-constraint-synthesis-ab",
        "Gated on Exp5908 fixture: three-family direct, decomposed, and retrieval synthesis",
        "results/experiment_5909_sota_constraint_synthesis_ab.json",
    ),
    (
        "exp5910-verification-guided-constraint-repair",
        "Gated on Exp5909 residual headroom: exact-diagnostic constraint repair",
        "results/experiment_5910_verification_guided_constraint_repair.json",
    ),
    (
        "exp5911-constraint-repair-portability-audit",
        "Gated on Exp5910 repair: model, family, camouflage, and portability audit",
        "results/experiment_5911_constraint_repair_portability_audit.json",
    ),
    (
        "exp5912-csl-exact-slot-requalification",
        "Frozen-science requalification of Exp5895 continuous self-learning",
        "results/experiment_5912_csl_exact_slot_requalification.json",
    ),
    (
        "exp5913-transactional-constraint-memory-fixture",
        "Gated on Exp5912 and Exp5909: transactional read-before-write fixture",
        "results/experiment_5913_transactional_constraint_memory_fixture.json",
    ),
    (
        "exp5914-sota-transactional-continuous-self-learning",
        "Gated on Exp5913 mechanism: prospective SOTA transactional continuous self-learning",
        "results/experiment_5914_sota_transactional_continuous_self_learning.json",
    ),
    (
        "exp5915-arc-live-runner-capability-lease",
        "Scoped capability lease for the adapter-disabled held ARC live runner",
        "results/experiment_5915_arc_live_runner_capability_lease.json",
    ),
    (
        "exp5916-arc-structured-memory-live-held-ab",
        "Gated on Exp5915 capability: adapter-disabled held structured-memory live A/B",
        "results/experiment_5916_arc_structured_memory_live_held_ab.json",
    ),
    (
        "exp5917-v525-capstone-reconciliation",
        "Branch-independent terminal reconciliation for milestone .525",
        "results/experiment_5917_v525_capstone_reconciliation.json",
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
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }
    if task_id == "exp5905-transition-v525":
        return {
            **base,
            "status": "blocked",
            "honest_verdict": "blocked: Exp5905 transition preconditions failed",
        }
    if task_id == "exp5906-v525-source-delta-ingestion":
        return {
            **base,
            "honest_verdict": "complete_null: no accepted post-V525 source deltas",
            "accepted_finding_count": 0,
        }
    if task_id == "exp5907-constraint-ir-replay-contract":
        return {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: replay contract ready",
            "constraint_ir_replay_contract_ready_score": 1.0,
            "canonical_projection_schema_and_version": {"projection_schema_version": "v1"},
            "fresh_twin_producer_consumer_replay": {"shared_helper_parity": True},
            "legacy_exp5896_adjudication": {
                "historical_checksum_mismatch_preserved": True,
                "new_contract_replay_ready": True,
            },
            "tamper_detection_matrix": {"all_rejected": True},
        }
    if task_id == "exp5908-verisynth-constraint-fixture":
        return {
            **base,
            "status": "ready",
            "honest_verdict": "ready: fixture replays",
            "verisynth_constraint_fixture_ready_score": 1.0,
        }
    if task_id == "exp5909-sota-constraint-synthesis-ab":
        return {
            **base,
            "honest_verdict": "complete_null: structured prompt arms did not improve exact synthesis",
            "constraint_stream_ready_score": 1.0,
            "verification_repair_admission_ready_score": 1.0,
            "parse_type_compile_and_semantic_metrics": {
                "overall": {"exact_semantic_equivalence_rate": 0.0, "n": 198}
            },
            "chronological_raw_stream_receipt": {
                "path": "results/experiment_5909_sota_constraint_synthesis_ab.raw.jsonl",
                "expected_row_count": 198,
                "event_order_is_chronological": True,
            },
            "embedded_tokenizer_and_loader_cuda_receipts": {
                "model_resolution": {"resolved_hf_ids": ["qwen", "gemma-dense", "gemma-moe"]},
                "runtime_loader_receipts": [{"gpu_offload_verified": True}],
            },
            "gpu_utilization_vram_latency_and_energy_receipts": {"latency_s_total": 12.5},
            "model_file_hashes": {"all_hashed": True},
            "model_specs": [{"hf_id": "qwen"}, {"hf_id": "gemma-dense"}, {"hf_id": "gemma-moe"}],
        }
    if task_id == "exp5910-verification-guided-constraint-repair":
        return {
            **base,
            "honest_verdict": "complete_null: exact diagnostics did not beat controls",
            "verification_guided_repair_ready_score": 0.0,
            "exact_semantic_repair_and_regression_metrics": {
                "exact_repair_success_rate": 0.0,
                "compile_delta_exact_vs_no_repair": -0.1,
            },
            "matched_no_diagnostic_no_information_and_shuffled_controls": {
                "shuffled_same_error_class_rate": 0.0
            },
            "embedded_tokenizer_loader_cuda_and_gpu_receipts": {
                "model_resolution": {"resolved_hf_ids": ["qwen", "gemma-dense", "gemma-moe"]},
                "runtime_loader_receipts": [{"gpu_offload_verified": True}],
            },
            "gpu_utilization_vram_latency_and_energy_receipts": {"latency_s_total": 4.0},
            "model_file_hashes": {"all_hashed": True},
            "model_specs": [{"hf_id": "qwen"}, {"hf_id": "gemma-dense"}, {"hf_id": "gemma-moe"}],
        }
    if task_id == "exp5911-constraint-repair-portability-audit":
        return {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5910-verification-guided-constraint-repair",
                    "artifact_field": "verification_guided_repair_ready_score",
                    "expected": 1.0,
                    "actual": 0.0,
                    "passed": False,
                }
            ],
        }
    if task_id == "exp5912-csl-exact-slot-requalification":
        return {
            **base,
            "status": "retired",
            "honest_verdict": "retired: repeated_global_suite_exit_2_after_frozen_science_parity",
            "continuous_self_learning_task": True,
            "csl_exact_slot_ready_score": 0.0,
            "deterministic_science_parity": {"matches_historical": True},
            "no_model_weight_mutation": {"all_unchanged": True},
            "prospective_lift_retention_safety_rollback_and_state_receipts": {
                "primary_zero_false_accepts": True,
                "retention": 1.0,
            },
            "repeated_verdict_retirement_decision": {
                "retire_if_same_verdict": True,
                "same_verdict_recurred": True,
                "decision": "retire_exact_scope",
            },
        }
    if task_id == "exp5913-transactional-constraint-memory-fixture":
        return {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "1 of 2 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5912-csl-exact-slot-requalification",
                    "artifact_field": "csl_exact_slot_ready_score",
                    "expected": 1.0,
                    "actual": 0.0,
                    "passed": False,
                }
            ],
        }
    if task_id == "exp5915-arc-live-runner-capability-lease":
        return {
            **base,
            "status": "complete_ready",
            "honest_verdict": "complete_ready: live runner capability lease bound",
            "live_runner_capability_ready_score": 1.0,
            "registry_precheck": {
                "ok": True,
                "registry_hash_before": "sha256:registry",
                "registry_hash_after": "sha256:registry",
                "registry_update_allowed": False,
            },
            "registry_unchanged": True,
            "state_isolation_and_teardown_receipts": {"ok": True, "teardown_called_count": 2},
            "scored_public_execution_count": 0,
            "model_load_count": 0,
        }
    if task_id == "exp5916-arc-structured-memory-live-held-ab":
        return {
            **base,
            "status": "blocked_precondition",
            "honest_verdict": "blocked_precondition: live_runner_execution_binding",
            "structured_memory_live_ready_score": 0.0,
            "public_level_solve_claimed": False,
            "registry_precheck": {
                "ok": True,
                "registry_hash_before": "sha256:registry",
                "registry_update_allowed": False,
            },
            "registry_unchanged": True,
            "incidental_completion_receipts": {
                "registry_updated": False,
                "public_level_solve_claimed": False,
            },
            "source_bfs_adapter_prior_game_and_hidden_state_access_count": 0,
            "submitted_e3_and_adapter_disabled_receipts": {"ok": True},
            "state_isolation_and_teardown_receipts": {"ok": True},
            "embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts": {
                "model_resolution": {"resolved_hf_ids": ["qwen", "gemma-moe"]},
                "dual_rtx3090_health": {"ok": True, "healthy_rtx3090_count": 2},
            },
            "model_file_hashes": {"all_hashed": True},
            "model_specs": [{"hf_id": "qwen"}, {"hf_id": "gemma-moe"}],
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
        "_bmad/traceability.md",
        "openspec/capabilities/research-reporting/spec.md",
        "ops/arc_solve_registry.yaml",
    ):
        _write_text(root, rel, f"{rel}\nREQ-REPORT-5917\n")
    _write_json(
        root, "results/experiment_5904_click_target_discrimination.json", {"status": "external"}
    )
    _write_text(
        root, "python/carnot/experiment_5904_click_target_discrimination.py", "# external\n"
    )
    _write_yaml(root, "ops/exclusion_manifest.yaml", {"retired": [], "retired_experiments": []})
    _write_yaml(
        root,
        "research-roadmap.yaml",
        {
            "milestone": "2026.07.525",
            "milestone_title": "Verified Constraint Synthesis",
            "tasks": [
                {
                    "id": task_id,
                    "milestone": "2026.07.525",
                    "title": title,
                    "deliverable": deliverable,
                    "prior_failures": (
                        [
                            {
                                "experiment_id": "exp5895",
                                "verdict": "complete_null: shortcut_safe_csl_not_promotion_eligible",
                                "retire_if_same_verdict": True,
                            }
                        ]
                        if task_id == "exp5912-csl-exact-slot-requalification"
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
                {"id": "2026.07.524", "tasks": []},
                {"id": "2026.07.524", "tasks": []},
            ]
        },
    )
    log_lines = ["| ts | Milestone 2026.07.525 activated | OK | 13 tasks queued |"]
    for task_id, title, _deliverable in TASKS:
        status = (
            "GATE_BLOCK"
            if task_id == "exp5914-sota-transactional-continuous-self-learning"
            else "OK"
        )
        if task_id in {
            "exp5911-constraint-repair-portability-audit",
            "exp5913-transactional-constraint-memory-fixture",
        }:
            status = "GATE_BLOCK"
        log_lines.append(f"| ts | {title[:55]} | {status} | receipt for {task_id} |")
    _write_text(root, "ops/conductor-log.md", "\n".join(log_lines) + "\n")
    for task_id, _title, deliverable in TASKS:
        if task_id in {
            "exp5914-sota-transactional-continuous-self-learning",
            "exp5917-v525-capstone-reconciliation",
        }:
            continue
        _write_json(root, deliverable, _artifact_for(task_id))
    return root


def _adversarial_receipt(root: Path) -> JsonDict:
    reports = []
    for task_id, _title, deliverable in TASKS:
        if task_id in {
            "exp5914-sota-transactional-continuous-self-learning",
            "exp5917-v525-capstone-reconciliation",
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
            "command": ".venv/bin/pytest tests/python/test_experiment_5917_v525_capstone_reconciliation.py -q --no-cov -n 0",
            "exit_code": 0,
            "summary": "focused Exp5917 tests passed",
        },
        {
            "command": ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5917_v525_capstone_reconciliation.py --fail-under=100",
            "exit_code": 0,
            "summary": "new module coverage 100%",
        },
    ]


def test_req_5917_builds_exact_matrix_and_disjoint_terminal_classes(tmp_path: Path) -> None:
    root = _make_root(tmp_path)

    payload = mod.reconcile_v525(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=False,
        update_ledgers=False,
    )

    matrix = payload["activated_task_and_declared_deliverable_matrix"]
    assert matrix["activated_task_count"] == 13
    assert [row["task_id"] for row in matrix["tasks"]] == [
        task_id for task_id, _title, _path in TASKS
    ]
    assert "exp5904" not in json.dumps(matrix)
    assert matrix["selection_policy"] == "exact_declared_deliverable"
    assert payload["milestone_and_task_range"]["exp5904_reserved_outside_matrix"] is True

    classes = payload["exact_terminal_classification"]
    assert (
        classes["terminal_class_by_task_id"]["exp5907-constraint-ir-replay-contract"] == "positive"
    )
    assert classes["terminal_class_by_task_id"]["exp5909-sota-constraint-synthesis-ab"] == "null"
    assert (
        classes["terminal_class_by_task_id"]["exp5912-csl-exact-slot-requalification"] == "retired"
    )
    assert (
        classes["terminal_class_by_task_id"]["exp5913-transactional-constraint-memory-fixture"]
        == "gate-blocked"
    )
    assert (
        classes["terminal_class_by_task_id"]["exp5914-sota-transactional-continuous-self-learning"]
        == "gate-blocked"
    )
    assert (
        classes["terminal_subclass_by_task_id"]["exp5916-arc-structured-memory-live-held-ab"]
        == "blocked-precondition"
    )
    assert classes["all_activated_classified_once"] is True


def test_req_5917_preserves_branch_independence_csl_and_arc_discipline(tmp_path: Path) -> None:
    root = _make_root(tmp_path)

    payload = mod.reconcile_v525(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=False,
        update_ledgers=False,
    )

    summary = payload["branch_independent_science_summary"]
    assert summary["branch_independence_preserved"] is True
    assert summary["constraint_synthesis"]["terminal_classes"] == [
        "positive",
        "positive",
        "null",
        "null",
        "gate-blocked",
    ]
    assert summary["continuous_self_learning"]["terminal_classes"] == [
        "retired",
        "gate-blocked",
        "gate-blocked",
    ]
    assert summary["arc_live"]["terminal_classes"] == ["positive", "blocked"]

    csl = payload["continuous_self_learning_slot_receipt"]
    assert csl["exp5912"]["continuous_self_learning_task"] is True
    assert csl["exp5914"]["declared_deliverable_present"] is False
    assert csl["exp5914"]["terminal_class"] == "gate-blocked"

    arc = payload["arc_generalization_and_live_capability_receipt"]
    assert arc["capability_ready_is_live_success"] is False
    assert arc["public_solve_credit_available"] is False
    assert arc["registry_updated"] is False
    assert arc["registry_unchanged"] is True


def test_req_5917_append_retirement_recommendations_and_schema_are_exact(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    before_registry = (root / "ops/arc_solve_registry.yaml").read_bytes()
    before_status = (root / "ops/status.md").read_bytes()

    payload = mod.reconcile_v525(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=True,
        update_ledgers=True,
    )
    second = mod.reconcile_v525(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=True,
        update_ledgers=True,
    )

    assert payload["research_complete_append_count"] == 1
    assert second["research_complete_append_count"] == 0
    complete = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    assert [row["id"] for row in complete["milestones"]].count("2026.07.525") == 1
    exp525 = [row for row in complete["milestones"] if row["id"] == "2026.07.525"][0]
    assert len(exp525["tasks"]) == 13
    assert exp525["tasks"][9]["id"] == "exp5914-sota-transactional-continuous-self-learning"
    assert "gate-blocked" in exp525["tasks"][9]["result"]

    retirement = payload["exclusion_and_retirement_decisions"]
    assert retirement["retire_if_same_verdict_decisions"][0]["same_verdict_recurred"] is True
    assert retirement["manifest_append_count"] == 1
    assert second["exclusion_and_retirement_decisions"]["manifest_append_count"] == 0

    recommendations = payload["next_three_falsifiable_recommendations"]
    assert len(recommendations) == 3
    assert all(row["success_condition"] and row["terminal_evidence"] for row in recommendations)
    assert not any(re.search(r"\bexp\d{4}\b", json.dumps(row).lower()) for row in recommendations)

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_provenance"]
        assert field in payload["field_principles"]
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert payload["honest_verdict"].startswith("complete_with_nulls:")
    assert payload["duplicate_history_amplification_count"] == 0
    assert payload["adversarial_verifier_receipts"]["exit_code"] == 0
    assert payload["protected_files_unchanged"]["all_unchanged"] is True
    assert (root / "ops/arc_solve_registry.yaml").read_bytes() == before_registry
    assert (root / "ops/status.md").read_bytes() == before_status
    assert (root / "results/experiment_5917_v525_capstone_reconciliation.json").exists()


def test_req_5917_terminal_defensive_paths_are_explicit(tmp_path: Path) -> None:
    assert mod._terminal_class(
        "exp5999-synthetic", {}, {"present": True}, {}, {"flag_count": 1, "max_severity": 2}
    ) == ("unsafe", "adversarial-verifier-critical")
    assert mod._terminal_class(
        "exp5999-synthetic", {}, {"present": False}, {"latest_status": "OK"}, {}
    ) == ("missing", "declared-deliverable-missing")
    assert mod._terminal_class(
        "exp5999-synthetic",
        {"status": "complete", "honest_verdict": "complete: done"},
        {"present": True},
        {},
        {},
    ) == ("positive", "complete")
    assert mod._terminal_class(
        "exp5999-synthetic", {"status": "unknown"}, {"present": True}, {}, {}
    ) == ("blocked", "unrecognized-terminal-treated-as-blocked")

    root = _make_root(tmp_path)
    _write_yaml(root, "ops/exclusion_manifest.yaml", {"retired": []})
    payload = mod.reconcile_v525(
        root=root,
        adversarial_receipt=_adversarial_receipt(root),
        test_results=_test_results(),
        write=False,
        update_ledgers=True,
    )
    assert payload["exclusion_and_retirement_decisions"]["manifest_append_count"] == 1

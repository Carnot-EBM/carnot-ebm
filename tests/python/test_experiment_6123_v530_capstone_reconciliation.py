"""Tests for the Exp6123 V530 capstone reconciliation.

Spec refs: REQ-REPORT-6123,
SCENARIO-REPORT-6123-EXACT-MATRIX,
SCENARIO-REPORT-6123-GATES,
SCENARIO-REPORT-6123-BRANCH-INDEPENDENCE,
SCENARIO-REPORT-6123-ADVERSARIAL-EXCLUSION,
SCENARIO-REPORT-6123-SCHEMA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6123_v530_capstone_reconciliation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id, title, rel_path in mod.UPSTREAM_TASKS:
        row: JsonDict = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": title,
            "deliverable": rel_path.as_posix(),
        }
        if task_id == "exp6115-phase-d-calibration-pool":
            row["gated_on"] = [
                {
                    "upstream": "exp6114-phase-d-gpu-ladder-canary",
                    "artifact_field": "phase_d_compute_and_ladder_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp6116-phase-d-held-candidate-pool":
            row["gated_on"] = [
                {
                    "upstream": "exp6115-phase-d-calibration-pool",
                    "artifact_field": "phase_d_calibration_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp6117-phase-d-headroom-audit":
            row["gated_on"] = [
                {
                    "upstream": "exp6116-phase-d-held-candidate-pool",
                    "artifact_field": "candidate_pool_integrity_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp6118-phase-d-per-layer-surface":
            row["gated_on"] = [
                {
                    "upstream": "exp6117-phase-d-headroom-audit",
                    "artifact_field": "phase_d_headroom_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp6119-phase-d-hidden-state-selector":
            row["gated_on"] = [
                {
                    "upstream": "exp6118-phase-d-per-layer-surface",
                    "artifact_field": "per_layer_surface_ready_score",
                    "op": "==",
                    "value": 1.0,
                },
                {
                    "upstream": "exp6117-phase-d-headroom-audit",
                    "artifact_field": "phase_d_headroom_ready_score",
                    "op": "==",
                    "value": 1.0,
                },
            ]
        tasks.append(row)
    tasks.append(
        {
            "id": mod.EXPERIMENT_ID,
            "milestone": mod.MILESTONE,
            "title": "Branch-independent .530 capstone and architecture reconciliation",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
            "prior_failures": [
                {
                    "experiment_id": "exp5973-v528-capstone-reconciliation",
                    "verdict": "complete_with_blocks: prior blocked branches preserved",
                    "retire_if_same_verdict": True,
                }
            ],
        }
    )
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": mod.MILESTONE_TITLE,
        "tasks": tasks,
    }


def _blocked_gate(upstream: str, field: str, actual: Any) -> JsonDict:
    return {
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": f"1 of 1 gate(s) failed; first failure: {upstream}.{field}",
        "gates_evaluated": [
            {
                "upstream": upstream,
                "artifact_field": field,
                "op": "==",
                "expected": 1.0,
                "actual": actual,
                "passed": False,
                "reason": f"actual={actual!r} == expected=1.0",
            }
        ],
    }


def _artifact(task_id: str) -> JsonDict:
    fixtures: dict[str, JsonDict] = {
        "exp6112-transition-v530": {
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: archived .529 into .530",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp6113-v530-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V530 source deltas",
            "inference_substrate": "aggregation_from_external_primary_sources",
        },
        "exp6114-phase-d-gpu-ladder-canary": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: canary ready",
            "inference_substrate": "live_local_sota_gguf_cuda_generation",
            "phase_d_compute_and_ladder_ready_score": 1.0,
        },
        "exp6115-phase-d-calibration-pool": {
            "status": "complete_null",
            "honest_verdict": "complete_null: no_calibration_stratum_decode_policy_met_gate",
            "inference_substrate": "live_local_sota_gguf_cuda_generation_plus_exact_validation",
            "phase_d_calibration_ready_score": 0.0,
            "selected_stratum_and_fixed_decode_policy": {"selected": None},
            "all_wrong_oracle_tuned_sc_and_solver_strata": {
                "overall": {"all_wrong_rate": 0.88, "oracle_at_k": 0.12}
            },
        },
        "exp6116-phase-d-held-candidate-pool": _blocked_gate(
            "exp6115-phase-d-calibration-pool",
            "phase_d_calibration_ready_score",
            0.0,
        ),
        "exp6118-phase-d-per-layer-surface": _blocked_gate(
            "exp6117-phase-d-headroom-audit",
            "phase_d_headroom_ready_score",
            None,
        ),
        "exp6120-outcome-committed-reduced-order-csl": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: equal utility lower state pareto",
            "inference_substrate": "deterministic_exact_verifier_and_versioned_external_state_no_llm",
            "outcome_committed_csl_ready_score": 1.0,
            "qualification_gate_matrix": {"all_gates_passed": True},
            "model_weight_immutability_receipt": {
                "all_unchanged": True,
                "weight_update_count": 0,
            },
            "unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics": {
                "unsafe_accept_count": 0,
                "nonforgetting": {"nonforgetting_ready": True},
            },
            "python_rust_pyo3_fixed_width_abi_parity": {
                "all_operation_version_reason_hash_and_energy_parity": True
            },
        },
        "exp6121-gatemate-changed-state-gate-v530": {
            "status": "blocked_physical_action",
            "honest_verdict": "blocked_physical_action: unchanged physical state",
            "inference_substrate": "hardware_state_gate_with_optional_non_destructive_detect",
            "physical_state_changed": False,
            "detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code": {
                "allowed": False,
                "attempt_count": 0,
            },
            "hardware_execution_authenticated": {"authenticated": False},
            "flash_synthesis_place_route_pack_and_firmware_mutation_counts": {"flash": 0},
            "speed_power_and_terminal_claim_counts": {"speedup": 0},
            "retirement_triggered": True,
        },
        "exp6122-arc-primitive-reachability-loo": {
            "status": "complete_null",
            "honest_verdict": "complete_null: no supported primitive no solve claim",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "registry_precheck_and_postcheck": {"ok": True, "registry_delta": 0},
            "target_level_solve_claim_count": 0,
            "solve_provenance": "live_agent_self_discovery",
            "duplicate_level_and_unreachable_solver_credit_counts": {
                "duplicate_level_credit_count": 0,
                "unreachable_solver_credit_count": 0,
            },
            "submitted_defaults_unchanged": {"unchanged": True},
            "offline_reproduced_new_level": False,
        },
    }
    return fixtures[task_id]


def _make_repo(root: Path) -> None:
    _write_text(
        root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload(), sort_keys=False)
    )
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-08-04 18:35 UTC | Milestone 2026.08.530 activated | OK | 12 tasks queued |",
                "| 2026-08-04 21:16 UTC | Gated calibration-only authentic Phase-D candidate | OK | 86 passed |",
                "| 2026-08-04 21:18 UTC | Gated held authentic same-model Phase-D candidate | GATE_BLOCK | 1 of 1 gate(s) failed |",
                "| 2026-08-04 21:24 UTC | Gated question-clustered Phase-D authenticity and | GATE_BLOCK | Pre-emptive skip: upstream retired |",
                "| 2026-08-04 21:24 UTC | Gated matching-base per-layer hidden-state surface | GATE_BLOCK | 1 of 1 gate(s) failed |",
                "| 2026-08-04 21:30 UTC | Gated internal-state Phase-D selector against tune | GATE_BLOCK | Pre-emptive skip: upstream retired |",
                "| 2026-08-04 21:47 UTC | Outcome-committed reduced-order continuous self-le | OK | 89 passed |",
                "| 2026-08-04 23:23 UTC | GateMate changed-physical-state continuity gate | OK | 94 passed |",
                "| 2026-08-04 23:39 UTC | ARC live-path generic primitive reachability and h | OK | 86 passed |",
            ]
        )
        + "\n",
    )
    for rel_path in mod.PRECONDITION_HASH_PATHS:
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH:
            continue
        if rel_path in {
            mod.ROADMAP_RELATIVE_PATH,
            mod.RESEARCH_COMPLETE_RELATIVE_PATH,
            mod.CONDUCTOR_LOG_RELATIVE_PATH,
        }:
            continue
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-6123\n")
    _write_text(
        root, "ops/arc_solve_registry.yaml", "schema_version: 1\nreproducible_total_levels: 183\n"
    )
    _write_text(
        root, "research-hardware-wishlist.md", "GateMate unchanged physical state fixture\n"
    )
    for task_id, _title, rel_path in mod.UPSTREAM_TASKS:
        if task_id in {"exp6117-phase-d-headroom-audit", "exp6119-phase-d-hidden-state-selector"}:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6117_numeric_prefix_alias_success.json",
        {"status": "complete_positive", "honest_verdict": "complete_positive: alias ignored"},
    )


def _receipts() -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for task_id, _title, rel_path in mod.UPSTREAM_TASKS:
        if task_id in {"exp6117-phase-d-headroom-audit", "exp6119-phase-d-hidden-state-selector"}:
            continue
        flags: list[JsonDict] = []
        if task_id == "exp6115-phase-d-calibration-pool":
            flags = [
                {
                    "kind": "METHODOLOGY_MISSING",
                    "severity": "warn",
                    "detail": "Compute-bound artifact missing model_specs/target_model.",
                }
            ]
        stdout_json = {
            "reports": [
                {
                    "artifact": rel_path.as_posix(),
                    "loaded": True,
                    "flag_count": len(flags),
                    "max_severity": 1 if flags else -1,
                    "flags": flags,
                }
            ],
            "flagged_count": len(flags),
        }
        receipts[task_id] = {
            "task_id": task_id,
            "artifact_path": rel_path.as_posix(),
            "command": f".venv/bin/python scripts/adversarial_verify.py --json {rel_path.as_posix()}",
            "exit_code": 1 if flags else 0,
            "stdout_json": stdout_json,
            "stderr": "",
            "receipt_hash": mod.sha256_json(stdout_json),
        }
    return receipts


def _test_receipts() -> list[JsonDict]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_6123_v530_capstone_reconciliation.py -q --no-cov -n 0",
            "exit_code": 0,
            "suite_kind": "unit",
            "ownership_class": "task_owned",
        },
        {
            "command": ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6123_v530_capstone_reconciliation.py --fail-under=100",
            "exit_code": 0,
            "suite_kind": "coverage",
            "ownership_class": "task_owned",
        },
        {
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 1,
            "suite_kind": "global",
            "ownership_class": "inherited",
            "phase": "before",
            "failure_node_ids": ["tests/python/inherited/test_old.py::test_known"],
        },
        {
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 1,
            "suite_kind": "global",
            "ownership_class": "inherited",
            "phase": "after",
            "failure_node_ids": ["tests/python/inherited/test_old.py::test_known"],
        },
    ]


def _build(root: Path) -> JsonDict:
    return mod.build_artifact(
        root=root,
        adversarial_receipts=_receipts(),
        test_receipts=_test_receipts(),
    )


def test_req_report_6123_spec_exists() -> None:
    """REQ-REPORT-6123: the capstone contract is spec-anchored."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-REPORT-6123" in text
    assert "SCENARIO-REPORT-6123-EXACT-MATRIX" in text
    assert "SCENARIO-REPORT-6123-ADVERSARIAL-EXCLUSION" in text


def test_exact_identity_matrix_and_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6123-EXACT-MATRIX: aliases cannot satisfy declared deliverables."""
    _make_repo(tmp_path)
    artifact = _build(tmp_path)

    matrix = artifact["milestone_task_and_declared_deliverable_matrix"]["tasks"]
    assert list(matrix) == [task_id for task_id, _title, _path in mod.UPSTREAM_TASKS]
    assert matrix["exp6117-phase-d-headroom-audit"]["present"] is False
    assert (
        matrix["exp6117-phase-d-headroom-audit"]["terminal_class"]
        == "conductor_gate_skipped_missing"
    )
    assert matrix["exp6117-phase-d-headroom-audit"]["same_number_aliases_ignored"] == [
        "results/experiment_6117_numeric_prefix_alias_success.json"
    ]
    assert (
        matrix["exp6116-phase-d-held-candidate-pool"]["terminal_class"] == "conductor_gate_blocked"
    )
    assert matrix["exp6116-phase-d-held-candidate-pool"]["executed_by_task"] is False
    assert (
        artifact["per_task_terminal_class_and_reason"]["all_tasks_have_one_terminal_class"] is True
    )


def test_gate_recomputation_preserves_null_to_skip_cascade(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6123-GATES: structured gates are recomputed from exact artifacts."""
    _make_repo(tmp_path)
    artifact = _build(tmp_path)

    gates = artifact["gate_recomputation_and_title_yaml_alignment"]["by_task"]
    assert gates["exp6115-phase-d-calibration-pool"]["all_gates_passed"] is True
    assert gates["exp6116-phase-d-held-candidate-pool"]["all_gates_passed"] is False
    assert gates["exp6116-phase-d-held-candidate-pool"]["gates"][0]["actual"] == 0.0
    assert gates["exp6117-phase-d-headroom-audit"]["gates"][0]["actual"] is None
    assert (
        gates["exp6118-phase-d-per-layer-surface"]["gates"][0]["upstream_artifact_present"] is False
    )
    assert gates["exp6119-phase-d-hidden-state-selector"]["all_gates_passed"] is False
    assert (
        artifact["candidate_pool_headroom_surface_selector_csl_hardware_and_arc_gate_matrix"][
            "phase_d"
        ]["selector"]["state"]
        == "blocked"
    )


def test_branch_independence_and_adversarial_positive_exclusion(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6123-BRANCH-INDEPENDENCE and ADVERSARIAL-EXCLUSION."""
    _make_repo(tmp_path)
    artifact = _build(tmp_path)

    synthesis = artifact["branch_independent_scientific_synthesis"]
    assert synthesis["phase_d"]["terminal_class"] == "complete_null_with_gate_blocked_cascade"
    assert synthesis["continuous_self_learning"]["terminal_class"] == "complete_positive"
    assert synthesis["hardware"]["terminal_class"] == "blocked_physical_action_retired"
    assert synthesis["arc"]["terminal_class"] == "complete_null_no_solve_credit"
    assert synthesis["borrowed_evidence_count"] == 0

    exclusions = artifact["adversarial_verifier_receipts_and_positive_claim_exclusions"]
    assert exclusions["flagged_task_ids"] == ["exp6115-phase-d-calibration-pool"]
    assert exclusions["positive_synthesis_task_ids"] == [
        "exp6120-outcome-committed-reduced-order-csl"
    ]
    assert exclusions["positive_claim_excluded_task_ids"] == ["exp6115-phase-d-calibration-pool"]


def test_artifact_schema_checksum_and_protected_files(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6123-SCHEMA: required fields and checksum are stable."""
    _make_repo(tmp_path)
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        output_path=output,
        adversarial_receipts=_receipts(),
        test_receipts=_test_receipts(),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_provenance"])
    assert artifact["status"] == "complete_with_blocks"
    assert artifact["honest_verdict"].startswith("complete_with_blocks:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["research_complete_append_readiness"]["append_ready"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    counts = artifact[
        "executed_skipped_missing_blocked_retired_underpowered_null_ready_positive_and_flagged_counts"
    ]
    assert counts["terminal_class_counts"]["conductor_gate_skipped_missing"] == 2
    assert counts["outcome_signal_counts"]["positive"] == 1
    assert counts["outcome_signal_counts"]["flagged"] == 1


def test_defensive_paths_and_live_receipt_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6123: malformed inputs fail closed and integration hooks stay bounded."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    payload, meta = mod._read_json_mapping(bad_json)
    assert payload == {}
    assert meta["error"].startswith("json_error:")

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    payload, meta = mod._read_json_mapping(list_json)
    assert payload == {}
    assert meta["error"] == "json_not_mapping"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("x: [", encoding="utf-8")
    payload, meta = mod._read_yaml_mapping(bad_yaml)
    assert payload == {}
    assert meta["error"].startswith("yaml_error:")

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    payload, meta = mod._read_yaml_mapping(list_yaml)
    assert payload == {}
    assert meta["error"] == "yaml_not_mapping"

    _write_text(tmp_path, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump({"tasks": "bad"}))
    assert mod._roadmap_tasks(tmp_path) == {}
    assert mod._same_number_aliases(tmp_path, "not-an-exp-id", Path("results/x.json")) == []
    assert (
        mod._classify_task({}, {"present": False}, {"latest_status": ""})[0]
        == mod.TerminalClass.MISSING
    )
    assert (
        mod._classify_task({"status": "underpowered"}, {"present": True}, {})[0]
        == mod.TerminalClass.UNDERPOWERED
    )
    assert (
        mod._classify_task({"status": "complete_partial"}, {"present": True}, {})[0]
        == mod.TerminalClass.PARTIAL
    )
    assert (
        mod._classify_task({"status": "blocked"}, {"present": True}, {})[0]
        == mod.TerminalClass.BLOCKED
    )
    assert mod._receipt_reports("not-json") == []
    assert mod._receipt_flag_count({"flag_count": 3}) == 3
    assert mod._normalize_adversarial_receipts(None, {}) == {}
    debt = mod._debt_delta(
        [
            {
                "command": "task-owned failing command",
                "exit_code": 1,
                "ownership_class": "task_owned",
            },
            {
                "command": "inherited failing command",
                "exit_code": 1,
                "ownership_class": "inherited",
                "failure_node_ids": ["old::test"],
            },
        ]
    )
    assert debt["task_owned_failure_commands"] == ["task-owned failing command"]
    assert debt["inherited_failure_node_ids_observed_after"] == ["old::test"]

    gate_rows = mod._recompute_gates(
        {"exp6115-phase-d-calibration-pool": {"title": "Gated x", "gated_on": ["bad"]}},
        {},
        {},
        {"exp6115-phase-d-calibration-pool": {"latest_status": "GATE_BLOCK"}},
    )
    assert gate_rows["by_task"]["exp6115-phase-d-calibration-pool"]["gate_declared_count"] == 0
    assert (
        mod._prior_failure_receipts({mod.EXPERIMENT_ID: {"prior_failures": ["bad"]}})["receipts"]
        == []
    )

    class _Result:
        stdout = " M file.py\n"

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: _Result())
    assert mod._git_status_short(tmp_path) == [" M file.py"]

    _make_repo(tmp_path)
    monkeypatch.setattr(mod, "run_live_adversarial_receipts", lambda root, metadata: _receipts())
    artifact = mod.build_artifact(
        root=tmp_path, adversarial_receipts=None, test_receipts=_test_receipts()
    )
    assert artifact["adversarial_verifier_receipts_and_positive_claim_exclusions"][
        "flagged_task_ids"
    ] == ["exp6115-phase-d-calibration-pool"]

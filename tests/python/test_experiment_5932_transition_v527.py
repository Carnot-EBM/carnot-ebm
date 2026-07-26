"""Tests for the Exp5932 V527 transition receipt.

Spec refs: REQ-REPORT-5932,
SCENARIO-REPORT-5932-EXACT-MATRIX,
SCENARIO-REPORT-5932-TERMINAL-CLASSES,
SCENARIO-REPORT-5932-TASK-OWNED-GATES,
SCENARIO-REPORT-5932-HISTORY-AND-RANGE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5932_transition_v527 as mod


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


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5918-transition-v526": {
            "status": "blocked",
            "honest_verdict": "blocked: Exp5918 transition preconditions failed",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp5919-v526-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V526 source deltas",
            "accepted_finding_count": 0,
            "inference_substrate": "aggregation_from_external_primary_sources_no_experimental_llm",
        },
        "exp5920-prospective-event-stream-admission": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: prospective_event_stream_admission_boundary_ready",
            "prospective_event_stream_ready_score": 1.0,
        },
        "exp5921-schema-derived-constraintir-support": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: schema support ready",
            "schema_support_ready_score": 1.0,
        },
        "exp5922-gguf-schema-decoder-bridge": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: embedded tokenizer bridge ready",
            "gguf_schema_decoder_bridge_ready_score": 1.0,
        },
        "exp5923-sota-schema-supported-constraintir-ab": {
            "status": "retired",
            "honest_verdict": "retired: schema-supported ConstraintIR decoding failed exact-semantic retirement gates",
            "retire_if_same_verdict": True,
        },
        "exp5924-transactional-constraint-memory-v2": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: transactional_constraint_memory_v2_ready",
            "transactional_constraint_memory_ready_score": 1.0,
        },
        "exp5926-adaptive-state-abi-v2-parity": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: adaptive_state_abi_v2_python_rust_pyo3_parity",
            "adaptive_state_abi_ready_score": 1.0,
        },
        "exp5927-coordinate-router-progress-qualification": {
            "status": "complete_underpowered",
            "honest_verdict": "complete_underpowered: hard_progress_positive_count_27_below_30_no_promotion",
            "hard_progress_positive_count": 27,
        },
        "exp5928-arc-live-runner-execution-binding": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: actual_child_live_runner_execution_binding_ready",
            "live_runner_execution_binding_ready_score": 1.0,
        },
        "exp5929-arc-structured-memory-bound-live-ab": {
            "status": "blocked_precondition",
            "honest_verdict": "blocked_precondition: bound_live_runner",
            "structured_memory_bound_live_ready_score": 0.0,
        },
        "exp5930-adaptive-state-board-mapping": {
            "status": "complete_static_mapping_no_physical_probe",
            "honest_verdict": "complete_static_mapping: static board mapping only",
            "physical_probe_executed": False,
        },
        "exp5931-v526-capstone-reconciliation": {
            "status": "complete_with_nulls",
            "honest_verdict": "complete_with_nulls: .526 reconciled with post-emission QA flag preserved",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"code": "DURATION_TOO_SHORT", "severity": "CRITICAL"}],
            "inference_substrate": "aggregation_from_exact_declared_artifacts",
        },
    }
    return payloads[task_id]


def _conductor(task_id: str) -> JsonDict:
    status = "OK"
    if task_id == "exp5918-transition-v526":
        status = "FAIL"
    if task_id == "exp5925-sota-transactional-csl-prospective":
        status = "GATE_BLOCK"
    if task_id == "exp5931-v526-capstone-reconciliation":
        status = "FLAGGED"
    return {
        "attempt_count": 1,
        "latest_status": status,
        "latest_line": f"| 2026-07-26 00:00 UTC | {task_id} | {status} | fixture |",
    }


def _capstone_payload() -> JsonDict:
    rows: list[JsonDict] = []
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        artifact = (
            _artifact(task_id) if task_id != "exp5925-sota-transactional-csl-prospective" else {}
        )
        capstone_self = task_id == "exp5931-v526-capstone-reconciliation"
        rows.append(
            {
                "identity": [mod.MILESTONE_FROM, task_id, rel_path.as_posix()],
                "milestone": mod.MILESTONE_FROM,
                "task_id": task_id,
                "title": mod.ACTIVATED_TASK_TITLES[task_id],
                "declared_deliverable": rel_path.as_posix(),
                "declared_deliverable_present": task_id
                != "exp5925-sota-transactional-csl-prospective"
                and not capstone_self,
                "declared_deliverable_loadable": task_id
                != "exp5925-sota-transactional-csl-prospective"
                and not capstone_self,
                "status": "" if capstone_self else artifact.get("status", ""),
                "honest_verdict": "" if capstone_self else artifact.get("honest_verdict", ""),
                "conductor": _conductor(task_id),
                "terminal_class": mod.EXPECTED_TERMINAL_CLASSES[task_id],
                "terminal_subclass": mod.EXPECTED_TERMINAL_SUBCLASSES[task_id],
            }
        )
    return {
        "status": "complete_with_nulls",
        "honest_verdict": "complete_with_nulls: .526 reconciled by exact declared deliverables",
        "inference_substrate": "aggregation_from_exact_declared_artifacts",
        "activated_task_and_declared_deliverable_matrix": {
            "activated_task_count": 14,
            "selection_policy": "exact_declared_deliverable",
            "tasks": rows,
        },
        "exact_terminal_classification": {
            "terminal_class_by_task_id": dict(mod.EXPECTED_TERMINAL_CLASSES),
            "terminal_subclass_by_task_id": dict(mod.EXPECTED_TERMINAL_SUBCLASSES),
            "task_ids_by_terminal_class": mod.group_expected_terminal_classes(),
            "all_activated_classified_once": True,
        },
    }


def _completion_payload(include_526_blocks: int = 1) -> JsonDict:
    block = {
        "id": mod.MILESTONE_FROM,
        "title": mod.MILESTONE_FROM_TITLE,
        "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-07-26",
        "finding": "Terminal outcomes preserved by Exp5931 capstone; see artifact.",
        "tasks": [
            {
                "id": task_id,
                "title": mod.ACTIVATED_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
                "result": mod.EXPECTED_TERMINAL_CLASSES[task_id],
            }
            for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
        ],
    }
    old_duplicate = {
        "id": "2026.07.510",
        "title": "Historical duplicate",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    return {
        "milestones": [
            deepcopy(old_duplicate),
            deepcopy(old_duplicate),
            *[deepcopy(block) for _ in range(include_526_blocks)],
        ]
    }


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "milestone_title": mod.MILESTONE_TO_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "title": f"title for {task_id}",
                "deliverable": rel_path.as_posix(),
            }
            for task_id, rel_path in mod.ALLOCATION_TASK_ARTIFACT_PATHS.items()
            if int(task_id[3:7]) <= 5937
        ],
    }


def _make_root(root: Path, *, include_526_blocks: int = 1) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id == "exp5925-sota-transactional-csl-prospective":
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_json(root, mod.EXP5931_CAPSTONE_RELATIVE_PATH, _capstone_payload())
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_526_blocks=include_526_blocks)),
    )
    allocation_table = "\n".join(
        f"| Exp{number} | `results/experiment_{number}_allocated.json` | allocation |"
        for number in range(5932, 5944)
    )
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, f"# vNEXT\n\n{allocation_table}\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "conductor fixture\n")
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.KNOWN_ISSUES_RELATIVE_PATH, "known issue fixture\n")
    _write_text(root, mod.ADVERSARIAL_VERIFY_RELATIVE_PATH, "# verifier fixture\n")
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.NORTH_STAR_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
    ):
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "### REQ-REPORT-5932\nfixture\n")


def _receipt(task_id: str) -> JsonDict:
    artifact_path = mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix()
    flags = (
        [{"code": "DURATION_TOO_SHORT", "severity": "CRITICAL"}]
        if task_id == "exp5931-v526-capstone-reconciliation"
        else []
    )
    stdout_json = {
        "reports": [
            {
                "artifact": artifact_path,
                "loaded": True,
                "flag_count": len(flags),
                "max_severity": 3 if flags else -1,
                "flags": flags,
            }
        ],
        "flagged_count": len(flags),
    }
    return {
        "task_id": task_id,
        "artifact_path": artifact_path,
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {artifact_path}",
        "exit_code": 1 if task_id == "exp5931-v526-capstone-reconciliation" else 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id)
        for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS
        if task_id != "exp5925-sota-transactional-csl-prospective"
    }


def _test_receipts() -> list[JsonDict]:
    rows = [
        {
            "command": f"task-owned {kind}",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": kind,
        }
        for kind in mod.REQUIRED_TASK_OWNED_GATE_KINDS
    ]
    rows.extend(
        [
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 1,
                "ownership_class": "global_suite",
                "phase": "before",
                "failure_node_ids": ["tests/python/inherited/test_global.py::test_old"],
            },
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 1,
                "ownership_class": "global_suite",
                "phase": "after",
                "failure_node_ids": ["tests/python/inherited/test_global.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/check_spec_coverage.py",
                "exit_code": 1,
                "ownership_class": "spec_coverage",
                "phase": "before",
                "missing_node_ids": ["tests/python/inherited/test_spec.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/check_spec_coverage.py",
                "exit_code": 1,
                "ownership_class": "spec_coverage",
                "phase": "after",
                "missing_node_ids": ["tests/python/inherited/test_spec.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
                "exit_code": 0,
                "ownership_class": "root_clutter",
                "phase": "before",
                "root_clutter_paths": ["old_probe.py"],
            },
            {
                "command": ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
                "exit_code": 0,
                "ownership_class": "root_clutter",
                "phase": "after",
                "root_clutter_paths": ["old_probe.py"],
            },
        ]
    )
    return rows


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts(),
        duration_s=1.25,
    )


def test_req_report_5932_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5932: OpenSpec names exact transition and debt gates."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5932") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5932-EXACT-MATRIX" in section
    assert "SCENARIO-REPORT-5932-TERMINAL-CLASSES" in section
    assert "SCENARIO-REPORT-5932-TASK-OWNED-GATES" in section
    assert "SCENARIO-REPORT-5932-HISTORY-AND-RANGE" in section
    assert "global_suite_failure_delta <= 0" in section
    assert "Exp5932 through Exp5943" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5932_exact_matrix_and_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5932-EXACT-MATRIX: .526 identities classify once."""

    _make_root(tmp_path, include_526_blocks=2)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.526",
        "destination_milestone": "2026.07.527",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 14
    assert matrix["exp5925-sota-transactional-csl-prospective"]["present"] is False
    assert matrix["exp5931-v526-capstone-reconciliation"]["present"] is True
    assert matrix["exp5931-v526-capstone-reconciliation"]["capstone_declared_present"] is False
    assert matrix["exp5931-v526-capstone-reconciliation"]["post_emission_qa_flag"] is True

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["terminal_subclass_by_task_id"] == mod.EXPECTED_TERMINAL_SUBCLASSES
    assert classes["all_activated_terminal"] is True
    assert classes["task_ids_by_terminal_class"]["retired"] == [
        "exp5923-sota-schema-supported-constraintir-ab"
    ]
    assert classes["task_ids_by_terminal_class"]["gate-blocked"] == [
        "exp5925-sota-transactional-csl-prospective"
    ]
    assert classes["task_ids_by_terminal_class"]["underpowered"] == [
        "exp5927-coordinate-router-progress-qualification"
    ]
    assert classes["task_ids_by_terminal_class"]["no-change"] == [
        "exp5930-adaptive-state-board-mapping"
    ]
    assert (
        report["adversarial_verifier_receipts"]["verified_present_declared_deliverable_count"] == 13
    )
    assert report["adversarial_verifier_receipts"]["post_emission_qa_flag_task_ids"] == [
        "exp5931-v526-capstone-reconciliation"
    ]
    mod.validate_artifact(report)


def test_scenario_report_5932_append_once_optional_staging_and_range(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5932-HISTORY-AND-RANGE: history and allocation stay sealed."""

    _make_root(tmp_path, include_526_blocks=2)
    report = _build(tmp_path)

    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_amplification_count"] == 0
    assert report["research_complete_append_receipt"]["reason"] == "exact_milestone_block_present"
    assert report["research_complete_append_receipt"]["before_milestone_block_count"] == 2
    assert report["optional_staging_roadmap_receipt"] == {
        "path": "research-roadmap-next.yaml",
        "present": False,
        "loadable": False,
        "milestone": None,
        "absence_is_failure": False,
        "reason": "optional_after_activation",
        "principle": mod.FIELD_PRINCIPLES["optional_staging_roadmap_receipt"],
    }
    assert report["next_task_range"] == {
        "start": "exp5932",
        "end": "exp5943",
        "count": 12,
        "declared_allocation_task_ids": list(mod.ALLOCATION_TASK_ARTIFACT_PATHS),
    }
    assert report["next_range_collision_count"] == 0
    mod.validate_artifact(report)

    absent = tmp_path / "absent"
    _make_root(absent, include_526_blocks=0)
    first = _build(absent)
    assert first["research_complete_append_count"] == 1
    assert first["duplicate_history_amplification_count"] == 0
    second = _build(absent)
    assert second["research_complete_append_count"] == 0
    assert second["duplicate_history_amplification_count"] == 0


def test_scenario_report_5932_task_owned_gates_disclose_inherited_debt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5932-TASK-OWNED-GATES: global debt is delta-gated."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    gate = report["task_owned_gate_receipts"]
    assert gate["all_required_gate_kinds_present"] is True
    assert gate["task_owned_failures"] == []
    debt = report["global_suite_spec_and_root_debt_baselines_and_deltas"]
    assert debt["global_suite_failure_delta"] == 0
    assert debt["global_spec_gap_delta"] == 0
    assert debt["root_clutter_delta"] == 0
    assert debt["non_amplification_gate_passed"] is True
    assert report["preconditions_checked"]["failed_preconditions"] == []
    mod.validate_artifact(report)

    amplified_receipts = deepcopy(_test_receipts())
    for row in amplified_receipts:
        if row.get("ownership_class") == "global_suite" and row.get("phase") == "after":
            row["failure_node_ids"].append("tests/python/new/test_owned.py::test_new")
    amplified = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=amplified_receipts,
        duration_s=1.25,
    )
    assert amplified["status"] == "blocked"
    assert (
        "global_suite_debt_amplified" in amplified["preconditions_checked"]["failed_preconditions"]
    )
    with pytest.raises(ValueError, match="debt non-amplification"):
        mod.validate_artifact(amplified)

    task_failure_receipts = deepcopy(_test_receipts())
    task_failure_receipts[0]["exit_code"] = 1
    task_failure = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=task_failure_receipts,
        duration_s=1.25,
    )
    assert task_failure["status"] == "blocked"
    assert "task_owned_gate_failed" in task_failure["preconditions_checked"]["failed_preconditions"]


def test_scenario_report_5932_range_collision_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5932-HISTORY-AND-RANGE: bare zero collisions are required."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5940_stale_collision.json", {"status": "stale"})
    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 1
    assert report["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {
            "path": "results/experiment_5940_stale_collision.json",
            "kind": "unexpected_next_range_reference",
        }
    ]
    with pytest.raises(ValueError, match="next_range_collision_count must be zero"):
        mod.validate_artifact(report)


def test_scenario_report_5932_schema_checksum_and_protection(tmp_path: Path) -> None:
    """REQ-REPORT-5932: schema fields, protection, and checksum are enforced."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_5932_present": True,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
        "principle": mod.FIELD_PRINCIPLES["docs_reconciled"],
    }
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in report["field_provenance"]
        assert report["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]
    mod.validate_artifact(report)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (
            lambda artifact: artifact.update(inference_substrate="live_llm_inference"),
            "inference_substrate",
        ),
        (
            lambda artifact: artifact.update(honest_verdict="complete_with_nulls: bad"),
            "honest_verdict",
        ),
        (
            lambda artifact: artifact.update(next_range_collision_count="0"),
            "next_range_collision_count",
        ),
        (
            lambda artifact: artifact.update(research_complete_append_count=2),
            "research_complete_append_count",
        ),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp5931-v526-capstone-reconciliation"
            ),
            "exactly fourteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp5931-v526-capstone-reconciliation": []}
            ),
            "exactly fourteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5918-transition-v526"
            ].update(identity=["2026.07.526", "exp5918-transition-v526", "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact.update(exact_terminal_classification=[]),
            "terminal classes missing",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({"exp5927-coordinate-router-progress-qualification": "positive"}),
            "terminal classes",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_subclass_by_task_id"
            ].update({"exp5927-coordinate-router-progress-qualification": "ready-or-positive"}),
            "terminal classes",
        ),
        (
            lambda artifact: artifact.update(adversarial_verifier_receipts=[]),
            "adversarial verifier receipts missing",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"].update(
                verified_present_declared_deliverable_count=1
            ),
            "missing adversarial verifier receipt",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].pop(
                "receipt_hash"
            ),
            "missing adversarial verifier receipt fields",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].update(
                command="python other.py"
            ),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact["task_owned_gate_receipts"].update(
                all_required_gate_kinds_present=False
            ),
            "task-owned gate",
        ),
        (
            lambda artifact: artifact["optional_staging_roadmap_receipt"].update(
                absence_is_failure=True
            ),
            "optional staging roadmap",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"].update(all_unchanged=False),
            "protected file",
        ),
        (
            lambda artifact: (
                artifact["protected_files_unchanged"].update(all_unchanged=True),
                next(iter(artifact["protected_files_unchanged"]["files"].values())).update(
                    unchanged=False
                ),
            ),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(principle="wrong"),
            "field provenance missing",
        ),
    ]
    for mutate, needle in mutations:
        artifact = deepcopy(report)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        with pytest.raises(ValueError, match=needle):
            mod.validate_artifact(artifact)

    checksum_drift = deepcopy(report)
    checksum_drift["duration_s"] = 9.5
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum_drift)


def test_req_report_5932_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5932: helper failures produce explicit blocked preconditions."""

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    _, bad_json_meta = mod._read_json_mapping(bad_json)
    assert bad_json_meta["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    _, list_json_meta = mod._read_json_mapping(list_json)
    assert list_json_meta["error"] == "json_not_mapping"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [\n", encoding="utf-8")
    _, bad_yaml_meta = mod._read_yaml_mapping(bad_yaml)
    assert bad_yaml_meta["error"].startswith("yaml_error:")
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- a\n", encoding="utf-8")
    _, list_yaml_meta = mod._read_yaml_mapping(list_yaml)
    assert list_yaml_meta["error"] == "yaml_not_mapping"

    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    assert mod._task_signature({"tasks": "bad"}) == ()
    assert mod._capstone_task_rows({}) == []
    assert mod._capstone_terminal_classes({}) == {}
    assert mod._receipt_flags({}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": []}}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flag_count({"flag_count": 4}) == 4
    assert mod._receipt_max_severity({"max_severity": 3}) == 3
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert mod._status_and_verdict([], {"terminal_class_by_task_id": {"x": "positive"}}) == (
        "complete",
        "complete: archived terminal .526 identities into .527 with collision-free allocation and no inherited debt amplification",
    )
    assert mod._status_and_verdict([], {"terminal_class_by_task_id": {"x": "retired"}}) == (
        "complete_with_terminal_receipts",
        "complete: archived terminal .526 identities into .527 without outcome laundering; inherited global debt not amplified; next_range_collision_count=0",
    )
    assert (
        mod._fallback_terminal_class({"schema": "blocked_gate_check_v1"}, {"present": True})
        == "gate-blocked"
    )
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {5932})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.CONDUCTOR_LOG_RELATIVE_PATH, {5932})
        == "transition_owned_conductor_receipt"
    )
    assert mod._root_clutter_inventory(tmp_path / "does-not-exist") == []

    after_only_debt = mod._debt_baselines_and_deltas(
        tmp_path,
        [
            {
                "ownership_class": "global_suite",
                "phase": "after",
                "failure_node_ids": ["global_after"],
            },
            {
                "ownership_class": "spec_coverage",
                "phase": "after",
                "missing_node_ids": ["spec_after"],
            },
            {
                "ownership_class": "root_clutter",
                "phase": "after",
                "root_clutter_paths": ["root_after.py"],
            },
        ],
    )
    assert after_only_debt["non_amplification_gate_passed"] is True
    before_only_debt = mod._debt_baselines_and_deltas(
        tmp_path,
        [
            {
                "ownership_class": "global_suite",
                "phase": "before",
                "failure_node_ids": ["global_before"],
            },
            {
                "ownership_class": "spec_coverage",
                "phase": "before",
                "missing_node_ids": ["spec_before"],
            },
            {
                "ownership_class": "root_clutter",
                "phase": "before",
                "root_clutter_paths": ["root_before.py"],
            },
        ],
    )
    assert before_only_debt["non_amplification_gate_passed"] is True

    assert mod._append_completion_if_absent(tmp_path / "nonterminal", terminal=False) == {
        "append_count": 0,
        "appended": False,
        "reason": "nonterminal_identity_present",
        "before_sha256": None,
        "after_sha256": None,
        "before_duplicate_history_count": 0,
        "after_duplicate_history_count": 0,
        "before_milestone_block_count": 0,
        "after_milestone_block_count": 0,
        "duplicate_history_amplification_count": 0,
    }
    absent_root = tmp_path / "absent-history"
    _make_root(absent_root, include_526_blocks=0)
    append = mod._append_completion_if_absent(absent_root, terminal=True)
    assert append["append_count"] == 1
    no_history_root = tmp_path / "no-history"
    append_without_existing_file = mod._append_completion_if_absent(no_history_root, terminal=True)
    assert append_without_existing_file["append_count"] == 1
    assert (no_history_root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).exists()

    bad_root = tmp_path / "bad-preconditions"
    _make_root(bad_root)
    _write_text(bad_root, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.EXP5931_CAPSTONE_RELATIVE_PATH, "{")
    (bad_root / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()
    bad_report = mod.build_report(
        bad_root,
        adversarial_receipts={},
        tests_run=[{"command": "focused", "exit_code": 1, "ownership_class": "task_owned"}],
        duration_s=1.0,
    )
    assert {
        "active_roadmap_unloadable",
        "roadmap_next_unloadable",
        "research_complete_unparseable",
        "exclusion_manifest_unparseable",
        "exp5931_capstone_unreadable",
        "live_verifier_missing",
        "missing_adversarial_receipts",
        "task_owned_gate_failed",
    } <= set(bad_report["preconditions_checked"]["failed_preconditions"])

    mismatch_root = tmp_path / "mismatch-preconditions"
    _make_root(mismatch_root)
    active = _active_roadmap_payload()
    active["milestone"] = "2026.07.999"
    _write_text(mismatch_root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(active))
    failed_receipts = _receipts()
    failed_receipts["exp5918-transition-v526"]["exit_code"] = 1
    debt_receipts = deepcopy(_test_receipts())
    for row in debt_receipts:
        if row.get("ownership_class") == "spec_coverage" and row.get("phase") == "after":
            row["missing_node_ids"].append("new_spec_gap")
        if row.get("ownership_class") == "root_clutter" and row.get("phase") == "after":
            row["root_clutter_paths"].append("new_root_probe.py")
    _write_text(mismatch_root, mod.SPEC_RELATIVE_PATH, "fixture without req\n")
    mismatch_report = mod.build_report(
        mismatch_root,
        adversarial_receipts=failed_receipts,
        tests_run=debt_receipts,
        duration_s=1.0,
    )
    assert {
        "active_roadmap_milestone_mismatch",
        "adversarial_verifier_failed",
        "global_spec_debt_amplified",
        "root_clutter_debt_amplified",
        "openspec_req_5932_missing",
    } <= set(mismatch_report["preconditions_checked"]["failed_preconditions"])

    branch_root = tmp_path / "branch-preconditions"
    _make_root(branch_root)
    monkeypatch.setattr(
        mod,
        "_append_completion_if_absent",
        lambda root, terminal: {
            "append_count": 0,
            "appended": False,
            "reason": "fixture",
            "before_sha256": None,
            "after_sha256": None,
            "before_duplicate_history_count": 0,
            "after_duplicate_history_count": 1,
            "before_milestone_block_count": 0,
            "after_milestone_block_count": 0,
            "duplicate_history_amplification_count": 1,
        },
    )
    monkeypatch.setattr(
        mod,
        "_exact_terminal_classification",
        lambda payloads, metadata, capstone_classes: {
            "terminal_class_by_task_id": {
                **mod.EXPECTED_TERMINAL_CLASSES,
                "exp5927-coordinate-router-progress-qualification": "positive",
            },
            "terminal_subclass_by_task_id": dict(mod.EXPECTED_TERMINAL_SUBCLASSES),
            "task_ids_by_terminal_class": {},
            "all_activated_terminal": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "_protected_files_unchanged",
        lambda root, before: {"all_unchanged": False, "files": {}},
    )
    monkeypatch.setattr(
        mod,
        "_resource_receipts",
        lambda root: {"disk": {"ok": False}, "memory": {"ok": True}},
    )
    monkeypatch.setattr(mod, "_atomic_output_receipt", lambda path: {"ok": False})
    branch_report = mod.build_report(
        branch_root,
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts(),
        duration_s=1.0,
    )
    assert {
        "duplicate_history_amplified",
        "terminal_outcomes_not_preserved",
        "protected_file_modified",
        "insufficient_resources",
        "atomic_output_unavailable",
    } <= set(branch_report["preconditions_checked"]["failed_preconditions"])

    emit_root = tmp_path / "emit"
    _make_root(emit_root)
    emitted = mod.emit_report(
        emit_root,
        output_path=emit_root / "results/out.json",
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts(),
    )
    assert json.loads((emit_root / "results/out.json").read_text(encoding="utf-8")) == emitted

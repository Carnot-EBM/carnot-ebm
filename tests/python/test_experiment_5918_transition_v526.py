"""Tests for the Exp5918 V526 transition receipt.

Spec refs: REQ-REPORT-5918,
SCENARIO-REPORT-5918-EXACT-MATRIX,
SCENARIO-REPORT-5918-TERMINAL-CLASSES,
SCENARIO-REPORT-5918-APPEND-ONCE-AND-EXP5904,
SCENARIO-REPORT-5918-RANGE-COLLISION-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5918_transition_v526 as mod


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
        "exp5905-transition-v525": {
            "status": "blocked",
            "honest_verdict": "blocked: Exp5905 transition preconditions failed",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp5906-v525-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V525 source deltas",
            "accepted_finding_count": 0,
            "inference_substrate": "aggregation_from_external_primary_sources",
        },
        "exp5907-constraint-ir-replay-contract": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: ConstraintIR replay contract is shared",
            "constraint_stream_ready_score": 1.0,
            "inference_substrate": "deterministic_artifact_replay_no_llm",
        },
        "exp5908-verisynth-constraint-fixture": {
            "status": "ready",
            "honest_verdict": "ready: deterministic ConstraintIR decomposition replays",
            "verisynth_fixture_ready_score": 1.0,
            "inference_substrate": "deterministic_exact_solver_labeled_dataset_no_llm",
        },
        "exp5909-sota-constraint-synthesis-ab": {
            "status": "complete",
            "honest_verdict": "complete_null: structured prompt arms did not improve exact synthesis",
            "constraint_stream_ready_score": 1.0,
            "synthesis_success_rate": 0.0,
            "inference_substrate": "live_llm_inference",
        },
        "exp5910-verification-guided-constraint-repair": {
            "status": "complete",
            "honest_verdict": "complete_null: exact diagnostics did not beat matched repair controls",
            "verification_guided_repair_ready_score": 0.0,
            "inference_substrate": "live_llm_inference",
        },
        "exp5911-constraint-repair-portability-audit": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5910-verification-guided-constraint-repair",
                    "artifact_field": "verification_guided_repair_ready_score",
                    "actual": 0.0,
                    "expected": 1.0,
                    "passed": False,
                }
            ],
        },
        "exp5912-csl-exact-slot-requalification": {
            "status": "retired",
            "honest_verdict": "retired: repeated_global_suite_exit_2_after_frozen_science_parity",
            "continuous_self_learning_task": True,
            "csl_exact_slot_ready_score": 0.0,
            "inference_substrate": "deterministic_artifact_replay_no_llm",
        },
        "exp5913-transactional-constraint-memory-fixture": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5912-csl-exact-slot-requalification",
                    "artifact_field": "csl_exact_slot_ready_score",
                    "actual": 0.0,
                    "expected": 1.0,
                    "passed": False,
                }
            ],
        },
        "exp5915-arc-live-runner-capability-lease": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: exp5916_live_runner_capability_lease_bound",
            "live_runner_capability_ready_score": 1.0,
            "public_level_solve_claimed": False,
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        "exp5916-arc-structured-memory-live-held-ab": {
            "status": "blocked_precondition",
            "honest_verdict": "blocked_precondition: live_runner_execution_binding",
            "structured_memory_live_ready_score": 0.0,
            "public_level_solve_claimed": False,
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
    }
    return payloads[task_id]


def _completion_payload(*, include_525: bool = True, duplicate_old_blocks: int = 1) -> JsonDict:
    duplicate_block = {
        "id": "2026.07.510",
        "title": "Historical duplicate",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    milestones = [deepcopy(duplicate_block) for _ in range(duplicate_old_blocks)]
    if include_525:
        milestones.append(
            {
                "id": mod.MILESTONE_FROM,
                "title": mod.MILESTONE_FROM_TITLE,
                "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                "completed": "2026-07-25",
                "finding": "Terminal outcomes preserved by Exp5917 capstone; see artifact.",
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
        )
    return {"milestones": milestones}


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
            for task_id, rel_path in mod.NEXT_TASK_ARTIFACT_PATHS.items()
        ],
    }


def _conductor(task_id: str) -> JsonDict:
    status = "OK"
    if task_id in {
        "exp5911-constraint-repair-portability-audit",
        "exp5913-transactional-constraint-memory-fixture",
        "exp5914-sota-transactional-continuous-self-learning",
    }:
        status = "GATE_BLOCK"
    if task_id == "exp5905-transition-v525":
        status = "FAIL"
    return {
        "attempt_count": 1,
        "latest_status": status,
        "latest_line": f"| 2026-07-25 00:00 UTC | {task_id} | {status} | fixture |",
    }


def _capstone_payload() -> JsonDict:
    rows = []
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        artifact = _artifact(task_id) if task_id not in {
            "exp5914-sota-transactional-continuous-self-learning",
            "exp5917-v525-capstone-reconciliation",
        } else {}
        rows.append(
            {
                "identity": [mod.MILESTONE_FROM, task_id, rel_path.as_posix()],
                "milestone": mod.MILESTONE_FROM,
                "task_id": task_id,
                "title": mod.ACTIVATED_TASK_TITLES[task_id],
                "declared_deliverable": rel_path.as_posix(),
                "declared_deliverable_present": task_id
                != "exp5914-sota-transactional-continuous-self-learning",
                "declared_deliverable_loadable": task_id
                != "exp5914-sota-transactional-continuous-self-learning",
                "status": artifact.get("status", ""),
                "honest_verdict": artifact.get("honest_verdict", ""),
                "conductor": _conductor(task_id),
            }
        )
    terminal_classes = dict(mod.EXPECTED_TERMINAL_CLASSES)
    capstone_classes = dict(terminal_classes)
    capstone_classes["exp5916-arc-structured-memory-live-held-ab"] = "blocked"
    return {
        "status": "complete_with_nulls",
        "honest_verdict": "complete: all .525 identities terminal with nulls and blocks preserved",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "activated_task_and_declared_deliverable_matrix": {
            "activated_task_count": 13,
            "selection_policy": "exact_declared_deliverable",
            "tasks": rows,
        },
        "exact_terminal_classification": {
            "terminal_class_by_task_id": capstone_classes,
            "terminal_subclass_by_task_id": {
                task_id: (
                    "blocked-precondition"
                    if task_id == "exp5916-arc-structured-memory-live-held-ab"
                    else terminal
                )
                for task_id, terminal in terminal_classes.items()
            },
        },
    }


def _make_root(
    root: Path,
    *,
    include_525_complete: bool = True,
    duplicate_old_blocks: int = 1,
) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id == "exp5914-sota-transactional-continuous-self-learning":
            continue
        payload = (
            _capstone_payload()
            if task_id == "exp5917-v525-capstone-reconciliation"
            else _artifact(task_id)
        )
        _write_json(root, rel_path, payload)
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(
            _completion_payload(
                include_525=include_525_complete,
                duplicate_old_blocks=duplicate_old_blocks,
            )
        ),
    )
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, "# vNEXT\n\n**Milestone:** 2026.07.526\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "conductor fixture\n")
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.ADVERSARIAL_VERIFY_RELATIVE_PATH, "# verifier fixture\n")
    _write_text(root, mod.EVIDENCE_INDEX_RELATIVE_PATH, "# evidence fixture\n")
    _write_text(root, mod.DOC_RECONCILE_RELATIVE_PATH, "# reconcile fixture\n")
    _write_text(root, "python/carnot/experiment_5904_click_target_discrimination.py", "# exp5904\n")
    _write_json(root, "results/experiment_5904_click_target_discrimination.json", {"status": "draft"})
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
    _write_text(root, mod.SPEC_RELATIVE_PATH, "### REQ-REPORT-5918\nfixture\n")


def _receipt(task_id: str) -> JsonDict:
    artifact_path = mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix()
    stdout_json = {
        "reports": [
            {
                "artifact": artifact_path,
                "loaded": True,
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
            }
        ],
        "flagged_count": 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": artifact_path,
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {artifact_path}",
        "exit_code": 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id)
        for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS
        if task_id != "exp5914-sota-transactional-continuous-self-learning"
    }


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_5918_transition_v526.py -q --no-cov -n 0",
                "exit_code": 0,
            },
            {
                "command": ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5918_transition_v526.py --fail-under=100",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        duration_s=1.25,
    )


def test_req_report_5918_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5918: OpenSpec names exact V526 transition gates."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5918") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5918-EXACT-MATRIX" in section
    assert "SCENARIO-REPORT-5918-TERMINAL-CLASSES" in section
    assert "SCENARIO-REPORT-5918-APPEND-ONCE-AND-EXP5904" in section
    assert "SCENARIO-REPORT-5918-RANGE-COLLISION-SCHEMA" in section
    assert "Exp5918 through Exp5931" in section
    assert "`next_range_collision_count`" in section
    assert "aggregation_from_upstream_artifacts" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5918_exact_matrix_and_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5918-EXACT-MATRIX: .525 identities classify once."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.525",
        "destination_milestone": "2026.07.526",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 13
    assert "exp5904-click-target-discrimination" not in matrix
    assert matrix["exp5914-sota-transactional-continuous-self-learning"]["present"] is False
    assert matrix["exp5914-sota-transactional-continuous-self-learning"][
        "declared_deliverable"
    ] == "results/experiment_5914_sota_transactional_continuous_self_learning.json"

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["disjoint_terminal_class_count"] == 13
    assert classes["all_activated_terminal"] is True
    assert classes["task_ids_by_terminal_class"]["null"] == [
        "exp5906-v525-source-delta-ingestion",
        "exp5909-sota-constraint-synthesis-ab",
        "exp5910-verification-guided-constraint-repair",
    ]
    assert classes["task_ids_by_terminal_class"]["retired"] == [
        "exp5912-csl-exact-slot-requalification"
    ]

    blocked = report["blocked_retired_gate_blocked_and_missing_receipts"]
    assert blocked["blocked_task_ids"] == ["exp5905-transition-v525"]
    assert blocked["blocked_precondition_task_ids"] == [
        "exp5916-arc-structured-memory-live-held-ab"
    ]
    assert blocked["retired_task_ids"] == ["exp5912-csl-exact-slot-requalification"]
    assert blocked["gate_blocked_task_ids"] == [
        "exp5911-constraint-repair-portability-audit",
        "exp5913-transactional-constraint-memory-fixture",
        "exp5914-sota-transactional-continuous-self-learning",
    ]
    assert blocked["missing_declared_deliverable_task_ids"] == [
        "exp5914-sota-transactional-continuous-self-learning"
    ]
    assert all(row["treated_as_success"] is False for row in blocked["receipts"])
    assert len(report["adversarial_verifier_receipts"]["reports"]) == 12
    mod.validate_artifact(report)


def test_scenario_report_5918_append_once_exp5904_and_range(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5918-APPEND-ONCE-AND-EXP5904: history and Exp5904 stay sealed."""

    _make_root(tmp_path, include_525_complete=False, duplicate_old_blocks=2)
    report = _build(tmp_path)

    assert report["research_complete_append_count"] == 1
    assert report["duplicate_history_amplification_count"] == 0
    complete = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    blocks = [block for block in complete["milestones"] if block["id"] == mod.MILESTONE_FROM]
    assert len(blocks) == 1
    assert [row["id"] for row in blocks[0]["tasks"]] == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)

    second = _build(tmp_path)
    assert second["research_complete_append_count"] == 0
    assert second["duplicate_history_amplification_count"] == 0

    exp5904 = report["exp5904_separate_evidence_receipt"]
    assert exp5904["exp5904_separate"] is True
    assert exp5904["edited_by_exp5918"] is False
    assert exp5904["classified_by_exp5918"] is False
    assert exp5904["appended_as_v525_task"] is False
    assert exp5904["included_in_activated_matrix"] is False
    assert "results/experiment_5904_click_target_discrimination.json" in exp5904["path_hashes"]

    assert report["next_task_range"] == {
        "start": "exp5918",
        "end": "exp5931",
        "count": 14,
        "task_ids": list(mod.NEXT_TASK_ARTIFACT_PATHS),
    }
    assert report["next_range_collision_count"] == 0
    assert report["preconditions_checked"]["range_collision_scan"]["collision_count"] == 0
    mod.validate_artifact(report)


def test_scenario_report_5918_range_collision_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5918-RANGE-COLLISION-SCHEMA: bare zero is required."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5924_stale_collision.json", {"status": "stale"})
    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 1
    assert "next_range_collision" in report["preconditions_checked"]["failed_preconditions"]
    assert report["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {"path": "results/experiment_5924_stale_collision.json", "kind": "unexpected_next_range_reference"}
    ]
    with pytest.raises(ValueError, match="next_range_collision_count must be zero"):
        mod.validate_artifact(report)


def test_scenario_report_5918_schema_checksum_and_protection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5918-RANGE-COLLISION-SCHEMA: schema fields are enforced."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_5918_present": True,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
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
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(honest_verdict="complete_with_nulls: bad"), "honest_verdict"),
        (lambda artifact: artifact.update(next_range_collision_count="0"), "next_range_collision_count"),
        (lambda artifact: artifact.update(next_range_collision_count=1), "next_range_collision_count must be zero"),
        (lambda artifact: artifact.update(research_complete_append_count=2), "research_complete_append_count"),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact["exp5904_separate_evidence_receipt"].update(
                exp5904_separate=False
            ),
            "Exp5904 separate evidence",
        ),
        (
            lambda artifact: artifact["exp5904_separate_evidence_receipt"].update(
                classified_by_exp5918=True
            ),
            "Exp5904 separate evidence",
        ),
        (
            lambda artifact: artifact["blocked_retired_gate_blocked_and_missing_receipts"][
                "receipts"
            ][0].update(treated_as_success=True),
            "terminal receipt",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp5917-v525-capstone-reconciliation"
            ),
            "exactly thirteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp5904-click-target-discrimination": {"present": True}}
            ),
            "Exp5904 must not be in activated matrix",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp5905-transition-v525": []}
            ),
            "exactly thirteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5905-transition-v525"
            ].update(identity=["2026.07.525", "exp5905-transition-v525", "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact.update(exact_terminal_classification=[]),
            "terminal classes missing",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][
                0
            ].pop("receipt_hash"),
            "missing adversarial verifier receipt fields",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][
                0
            ].update(command="python other.py"),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact.update(
                blocked_retired_gate_blocked_and_missing_receipts=[]
            ),
            "terminal receipt missing",
        ),
        (
            lambda artifact: artifact.update(adversarial_verifier_receipts=[]),
            "adversarial verifier receipts missing",
        ),
        (
            lambda artifact: artifact.update(exp5904_separate_evidence_receipt=[]),
            "Exp5904 separate evidence missing",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"]["files"][
                mod.CONDUCTOR_RELATIVE_PATH.as_posix()
            ].update(unchanged=False),
            "protected file",
        ),
        (
            lambda artifact: artifact.update(protected_files_unchanged=[]),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(
                principle="wrong"
            ),
            "field provenance missing",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({"exp5910-verification-guided-constraint-repair": "positive"}),
            "terminal classes",
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


def test_req_report_5918_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5918: helper failures produce blocked preconditions."""

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
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {5918})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.ROADMAP_RELATIVE_PATH, {5922, 5931})
        == "allowed_allocation_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.CONDUCTOR_LOG_RELATIVE_PATH, {5918})
        == "transition_owned_conductor_attempt_reference"
    )
    assert mod._fallback_terminal_class("x", {}, {"present": False}) == "missing"
    assert (
        mod._fallback_terminal_class("x", {"status": "blocked_precondition"}, {"present": True})
        == "blocked-precondition"
    )
    assert (
        mod._fallback_terminal_class("x", {"schema": "blocked_gate_check_v1"}, {"present": True})
        == "gate-blocked"
    )
    assert mod._fallback_terminal_class("x", {"status": "retired"}, {"present": True}) == "retired"
    assert (
        mod._fallback_terminal_class("x", {"honest_verdict": "complete_null: none"}, {"present": True})
        == "null"
    )
    assert (
        mod._fallback_terminal_class("x", {"honest_verdict": "blocked: none"}, {"present": True})
        == "blocked"
    )
    assert mod._fallback_terminal_class("x", {"status": "complete"}, {"present": True}) == "positive"
    assert mod._fallback_terminal_class("x", {}, {"present": True}) == "missing"
    assert mod._status_and_verdict([], {"terminal_class_by_task_id": {"x": "positive"}}) == (
        "complete",
        "complete: archived terminal .525 identities into .526 with collision-free allocation",
    )
    assert mod._status_and_verdict(
        [], {"terminal_class_by_task_id": {"x": "unsafe"}}
    ) == ("blocked", "blocked: unsafe .525 identity present")

    assert mod._append_completion_if_absent(tmp_path / "nonterminal", terminal=False) == {
        "append_count": 0,
        "appended": False,
        "reason": "nonterminal_identity_present",
        "before_sha256": None,
        "after_sha256": None,
        "before_duplicate_block_count": 0,
        "after_duplicate_block_count": 0,
        "duplicate_history_amplification_count": 0,
    }
    absent_root = tmp_path / "absent-history"
    append = mod._append_completion_if_absent(absent_root, terminal=True)
    assert append["append_count"] == 1
    assert (absent_root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).exists()

    missing_receipts = tmp_path / "missing-receipts"
    _make_root(missing_receipts)
    missing_report = mod.build_report(
        missing_receipts,
        adversarial_receipts={},
        tests_run=[{"command": "focused", "exit_code": 0}],
        duration_s=1.0,
    )
    assert "missing_adversarial_receipts" in missing_report["preconditions_checked"][
        "failed_preconditions"
    ]
    with pytest.raises(ValueError, match="missing adversarial verifier receipt"):
        mod.validate_artifact(missing_report)

    bad_root = tmp_path / "bad-preconditions"
    _make_root(bad_root)
    _write_text(bad_root, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.EXP5917_CAPSTONE_RELATIVE_PATH, "{")
    (bad_root / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()
    bad_report = mod.build_report(
        bad_root,
        adversarial_receipts={},
        tests_run=[{"command": "focused", "exit_code": 1}],
        duration_s=1.0,
    )
    assert {
        "active_roadmap_unloadable",
        "roadmap_next_unloadable",
        "research_complete_unparseable",
        "exclusion_manifest_unparseable",
        "exp5917_capstone_unreadable",
        "live_verifier_missing",
        "required_tests_failed",
        "missing_adversarial_receipts",
    } <= set(bad_report["preconditions_checked"]["failed_preconditions"])

    mismatch_root = tmp_path / "mismatch-preconditions"
    _make_root(mismatch_root)
    active = _active_roadmap_payload()
    active["milestone"] = "2026.07.999"
    _write_text(mismatch_root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(active))
    failed_receipts = _receipts()
    failed_receipts["exp5905-transition-v525"]["exit_code"] = 1
    mismatch_report = mod.build_report(
        mismatch_root,
        adversarial_receipts=failed_receipts,
        tests_run=[],
    )
    assert {
        "active_roadmap_milestone_mismatch",
        "adversarial_verifier_failed",
    } <= set(mismatch_report["preconditions_checked"]["failed_preconditions"])
    assert isinstance(mismatch_report["duration_s"], float)

    emit_root = tmp_path / "emit"
    _make_root(emit_root)
    emitted = mod.emit_report(
        emit_root,
        output_path=emit_root / "results/out.json",
        adversarial_receipts=_receipts(),
        tests_run=[],
    )
    assert json.loads((emit_root / "results/out.json").read_text(encoding="utf-8")) == emitted

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
            "before_duplicate_block_count": 0,
            "after_duplicate_block_count": 1,
            "duplicate_history_amplification_count": 1,
        },
    )
    monkeypatch.setattr(
        mod,
        "_exp5904_separate_evidence_receipt",
        lambda root: {
            "exp5904_separate": False,
            "edited_by_exp5918": False,
            "classified_by_exp5918": False,
            "appended_as_v525_task": False,
        },
    )
    monkeypatch.setattr(
        mod,
        "_exact_terminal_classification",
        lambda payloads, metadata, capstone_classes: {
            "terminal_class_by_task_id": {
                **mod.EXPECTED_TERMINAL_CLASSES,
                "exp5906-v525-source-delta-ingestion": "positive",
            },
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
        tests_run=[{"command": "focused", "exit_code": 0}],
        duration_s=1.0,
    )
    assert {
        "duplicate_history_amplified",
        "exp5904_not_separate",
        "terminal_outcomes_not_preserved",
        "protected_file_modified",
        "insufficient_resources",
        "atomic_output_unavailable",
    } <= set(branch_report["preconditions_checked"]["failed_preconditions"])

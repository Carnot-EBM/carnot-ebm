"""Tests for the Exp5961 V528 transition receipt.

Spec refs: REQ-REPORT-5961,
SCENARIO-REPORT-5961-ACTIVATED-MATRIX,
SCENARIO-REPORT-5961-TERMINAL-CLASSES,
SCENARIO-REPORT-5961-RESERVATIONS-AND-HISTORY,
SCENARIO-REPORT-5961-DEBT-AND-VERIFIER,
SCENARIO-REPORT-5961-RANGE-COLLISION,
SCENARIO-REPORT-5961-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5961_transition_v528 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact(task_id: str) -> JsonDict:
    return {
        "exp5932-transition-v527": {
            "schema": "carnot.experiment_5932.transition_v527.v1",
            "experiment_id": "exp5932-transition-v527",
            "milestone": "2026.07.527",
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: archived terminal .526 identities into .527",
            "inference_substrate": "aggregation_from_upstream_artifacts_no_llm",
        },
        "exp5933-aggregation-substrate-qa-repair": {
            "experiment": "5933",
            "status": "complete_partial",
            "honest_verdict": "complete_partial: substrate classifier repair is ready",
            "inference_substrate": "deterministic_qa_regression_no_llm",
        },
        "exp5934-v527-source-delta-ingestion": {
            "schema": "carnot.experiment_5934.v527_source_delta_ingestion.v1",
            "experiment_id": "exp5934-v527-source-delta-ingestion",
            "milestone": "2026.07.527",
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V527 source deltas",
            "inference_substrate": "aggregation_from_external_primary_sources_no_experimental_llm",
        },
        "exp5935-non-pruning-atomic-constraint-support": {
            "schema": "carnot.experiment_5935.non_pruning_atomic_support.v1",
            "experiment_id": "experiment_5935_non_pruning_atomic_constraint_support",
            "status": "complete_ready",
            "honest_verdict": "complete_ready: non-pruning atomic support is ready",
            "inference_substrate": "deterministic_exact_executor_fixture_no_llm",
        },
        "exp5936-sota-atomic-support-union-ab": {
            "schema": "carnot.experiment_5936.sota_atomic_support_union_ab.v1",
            "experiment_id": "experiment_5936_sota_atomic_support_union_ab",
            "status": "retired",
            "honest_verdict": "retired: transformed-view atomic union failed exact-semantic gates",
            "inference_substrate": "local_mandated_gguf_public_llama_cpp_cuda_atomic_support_union",
        },
    }[task_id]


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


def _completion_payload(include_527_blocks: int = 1) -> JsonDict:
    block = {
        "id": mod.MILESTONE_FROM,
        "title": mod.MILESTONE_FROM_TITLE,
        "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-07-26",
        "finding": "Terminal outcomes preserved by fixture.",
        "tasks": [
            {
                "id": task_id,
                "title": mod.ACTIVATED_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
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
            *[deepcopy(block) for _ in range(include_527_blocks)],
        ]
    }


def _make_root(root: Path, *, include_527_blocks: int = 1) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id == "exp5937-excluded-pool-coverage-audit":
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_527_blocks=include_527_blocks)),
    )
    allocation_table = "\n".join(
        f"### Exp{number}\n**Deliverable:** `results/experiment_{number}_allocated.json`"
        for number in range(5961, 5974)
    )
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        f"# Research Roadmap vNEXT\n\n**Experiment range:** Exp5961-Exp5973\n{allocation_table}\n",
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-07-26 04:45 UTC | Plan milestone 2026.07.527 | OK | 6 tasks proposed |",
                "| 2026-07-26 04:48 UTC | Milestone 2026.07.527 activated | OK | 6 tasks queued |",
                "| 2026-07-26 12:57 UTC | Gated on Exp5936 stream: exact included/excluded-p | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            ]
        )
        + "\n",
    )
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.KNOWN_ISSUES_RELATIVE_PATH, "known issue fixture\n")
    _write_text(root, mod.ADVERSARIAL_VERIFY_RELATIVE_PATH, "# verifier fixture\n")
    _write_text(root, "results/experiment_5950_click_pixel_sampling_smoke.json", "{}\n")
    _write_text(root, "results/experiment_5960_hud_mask_repair_smoke.json", "{}\n")
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
        mod.EVIDENCE_INDEX_RELATIVE_PATH,
        mod.DOC_RECONCILE_RELATIVE_PATH,
    ):
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "### REQ-REPORT-5961\nfixture\n")


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
        if task_id != "exp5937-excluded-pool-coverage-audit"
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


def test_req_report_5961_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5961: OpenSpec names the activated-identity and collision gates."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5961") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5961-ACTIVATED-MATRIX" in section
    assert "SCENARIO-REPORT-5961-TERMINAL-CLASSES" in section
    assert "SCENARIO-REPORT-5961-RESERVATIONS-AND-HISTORY" in section
    assert "SCENARIO-REPORT-5961-DEBT-AND-VERIFIER" in section
    assert "SCENARIO-REPORT-5961-RANGE-COLLISION" in section
    assert "global_suite_failure_delta <= 0" in section
    assert "Exp5961 through Exp5973" in section
    assert "ops/.test_suite_mutation_runs/" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5961_exact_matrix_terminal_classes_and_reservations(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5961-ACTIVATED-MATRIX: six .527 identities classify once."""

    _make_root(tmp_path, include_527_blocks=2)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.527",
        "destination_milestone": "2026.07.528",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 6
    assert matrix["exp5937-excluded-pool-coverage-audit"]["present"] is False
    assert (
        matrix["exp5937-excluded-pool-coverage-audit"]["conductor"]["latest_status"] == "GATE_BLOCK"
    )

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["all_activated_terminal"] is True
    assert classes["task_ids_by_terminal_class"]["retired"] == [
        "exp5936-sota-atomic-support-union-ab"
    ]
    assert classes["task_ids_by_terminal_class"]["gate-blocked"] == [
        "exp5937-excluded-pool-coverage-audit"
    ]

    reservations = report["unactivated_reservation_receipt"]
    assert list(reservations["reservation_task_ids"]) == list(mod.UNACTIVATED_RESERVATIONS)
    assert reservations["activated_as_results_count"] == 0
    assert reservations["fabricated_outcome_count"] == 0
    assert (
        report["adversarial_verifier_receipts"]["verified_present_declared_deliverable_count"] == 5
    )
    assert report["adversarial_verifier_receipts"]["aggregation_artifact_task_ids_clean"] == [
        "exp5932-transition-v527"
    ]
    mod.validate_artifact(report)


def test_scenario_report_5961_append_once_and_duplicate_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5961-RESERVATIONS-AND-HISTORY: history appends at most once."""

    _make_root(tmp_path, include_527_blocks=2)
    report = _build(tmp_path)

    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_amplification_count"] == 0
    assert report["research_complete_append_receipt"]["reason"] == "exact_milestone_block_present"
    assert report["research_complete_append_receipt"]["before_milestone_block_count"] == 2
    mod.validate_artifact(report)

    absent = tmp_path / "absent"
    _make_root(absent, include_527_blocks=0)
    first = _build(absent)
    assert first["research_complete_append_count"] == 1
    assert first["duplicate_history_amplification_count"] == 0
    second = _build(absent)
    assert second["research_complete_append_count"] == 0
    assert second["duplicate_history_amplification_count"] == 0


def test_scenario_report_5961_debt_delta_and_collision_blocking(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5961-DEBT-AND-VERIFIER: inherited debt is delta-gated."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    debt = report["inherited_debt_baselines_and_deltas"]
    assert debt["global_suite_failure_delta"] == 0
    assert debt["global_spec_gap_delta"] == 0
    assert debt["root_clutter_delta"] == 0
    assert debt["non_amplification_gate_passed"] is True
    assert report["preconditions_checked"]["failed_preconditions"] == []
    mod.validate_artifact(report)

    amplified_rows = deepcopy(_test_receipts())
    for row in amplified_rows:
        if row.get("ownership_class") == "global_suite" and row.get("phase") == "after":
            row["failure_node_ids"].append("tests/python/new/test_owned.py::test_new")
    amplified = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=amplified_rows,
        duration_s=1.25,
    )
    assert amplified["status"] == "blocked"
    assert (
        "global_suite_debt_amplified" in amplified["preconditions_checked"]["failed_preconditions"]
    )
    with pytest.raises(ValueError, match="debt non-amplification"):
        mod.validate_artifact(amplified)

    task_failure_rows = deepcopy(_test_receipts())
    task_failure_rows[0]["exit_code"] = 1
    task_failure = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=task_failure_rows,
        duration_s=1.25,
    )
    assert task_failure["status"] == "blocked"
    assert "task_owned_gate_failed" in task_failure["preconditions_checked"]["failed_preconditions"]

    _write_json(tmp_path, "results/experiment_5966_stale_collision.json", {"status": "stale"})
    collision = _build(tmp_path)
    assert collision["status"] == "blocked"
    assert collision["honest_verdict"].startswith("blocked:")
    assert collision["next_range_collision_count"] == 1
    assert collision["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {
            "path": "results/experiment_5966_stale_collision.json",
            "kind": "unexpected_next_range_reference",
            "numbers": [5966],
        }
    ]
    mod.validate_artifact(collision)

    _write_json(
        tmp_path / "transient",
        "ops/.test_suite_mutation_runs/ppid-1-20260803.writes.log",
        {"path": "results/experiment_5966_stale_collision.json"},
    )
    _make_root(tmp_path / "transient")
    transient = _build(tmp_path / "transient")
    assert transient["next_range_collision_count"] == 0
    assert transient["status"] == "complete_with_terminal_receipts"
    mod.validate_artifact(transient)


def test_scenario_report_5961_preconditions_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5961: blocked reports name exact failed preconditions."""

    unloadable = tmp_path / "unloadable"
    _make_root(unloadable)
    _write_text(unloadable, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    unloadable_report = mod.build_report(
        unloadable,
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts(),
        duration_s=1.25,
    )
    assert (
        "active_roadmap_unloadable"
        in unloadable_report["preconditions_checked"]["failed_preconditions"]
    )

    root = tmp_path / "bad"
    _make_root(root)
    bad_roadmap = _active_roadmap_payload()
    bad_roadmap["milestone"] = "2026.07.999"
    bad_roadmap["tasks"] = bad_roadmap["tasks"][:-1]
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(bad_roadmap))
    _write_text(root, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    _write_text(root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "a: [\n")
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "missing conductor plan\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "fixture without the required marker\n")
    (root / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()
    _write_json(
        root,
        mod.ACTIVATED_TASK_ARTIFACT_PATHS["exp5936-sota-atomic-support-union-ab"],
        {"status": "complete", "honest_verdict": "complete: laundered"},
    )

    receipts = dict(_receipts())
    receipts.pop("exp5934-v527-source-delta-ingestion")
    receipts["exp5932-transition-v527"] = {
        **receipts["exp5932-transition-v527"],
        "exit_code": 1,
        "stdout_json": {
            "reports": [
                {
                    "artifact": "x",
                    "loaded": True,
                    "flag_count": 1,
                    "max_severity": 3,
                    "flags": [{"kind": "CRITICAL", "severity": "critical"}],
                }
            ]
        },
    }
    rows = _test_receipts()
    rows = [row for row in rows if row.get("suite_kind") != "coverage"]
    for row in rows:
        if row.get("ownership_class") == "spec_coverage" and row.get("phase") == "after":
            row["missing_node_ids"].append("tests/python/new/test_spec.py::test_new")
        if row.get("ownership_class") == "root_clutter" and row.get("phase") == "after":
            row["root_clutter_paths"].append("new_probe.py")

    monkeypatch.setattr(
        mod,
        "_append_completion_if_absent",
        lambda _root, _terminal: {
            "append_count": 0,
            "duplicate_history_amplification_count": 1,
        },
    )
    monkeypatch.setattr(
        mod,
        "_protected_files_unchanged",
        lambda _root, _before: {
            "files": {"research-roadmap.yaml": {"unchanged": False}},
            "all_unchanged": False,
            "principle": mod.FIELD_PRINCIPLES["protected_files_unchanged"],
        },
    )
    monkeypatch.setattr(
        mod,
        "_outer_loop_id_separation_receipt",
        lambda _root, _before: {
            "protected_outer_loop_ids": ["Exp5950", "Exp5960"],
            "all_unchanged": False,
            "principle": mod.FIELD_PRINCIPLES["outer_loop_id_separation_receipt"],
        },
    )
    monkeypatch.setattr(
        mod,
        "_resource_receipts",
        lambda _root: {"disk": {"ok": False}, "memory": {"ok": True}},
    )
    monkeypatch.setattr(
        mod,
        "_atomic_output_receipt",
        lambda _path: {"ok": False},
    )

    report = mod.build_report(root, adversarial_receipts=receipts, tests_run=rows, duration_s=1.25)
    failed = set(report["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_milestone_mismatch",
        "active_roadmap_task_ids_mismatch",
        "roadmap_next_unloadable",
        "research_complete_unparseable",
        "exclusion_manifest_unparseable",
        "v527_plan_line_missing_or_not_ok",
        "v527_activation_line_missing_or_not_six",
        "live_verifier_missing",
        "terminal_outcomes_not_preserved",
        "missing_adversarial_receipts",
        "adversarial_verifier_failed",
        "aggregation_artifact_not_clean",
        "task_owned_gate_missing",
        "global_spec_debt_amplified",
        "root_clutter_debt_amplified",
        "duplicate_history_amplified",
        "openspec_req_5961_missing",
        "protected_file_modified",
        "outer_loop_id_modified",
        "insufficient_resources",
        "atomic_output_unavailable",
    } <= failed


def test_scenario_report_5961_schema_checksum_and_protection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5961-SCHEMA: required fields, protection, and checksum are enforced."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_5961_present": True,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
        "principle": mod.FIELD_PRINCIPLES["docs_reconciled"],
    }
    assert report["outer_loop_id_separation_receipt"]["protected_outer_loop_ids"] == [
        "Exp5950",
        "Exp5960",
    ]
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
            lambda artifact: artifact.update(honest_verdict="complete_partial: bad"),
            "honest_verdict",
        ),
        (
            lambda artifact: artifact.update(next_range_collision_count="0"),
            "next_range_collision_count",
        ),
        (
            lambda artifact: artifact.update(next_range_collision_count=1),
            "next_range_collision_count must be zero",
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
                "exp5937-excluded-pool-coverage-audit"
            ),
            "exactly six",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp5937-excluded-pool-coverage-audit": []}
            ),
            "exactly six",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5932-transition-v527"
            ].update(identity=["2026.07.527", "exp5932-transition-v527", "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["unactivated_reservation_receipt"].update(
                fabricated_outcome_count=1
            ),
            "unactivated reservations",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({"exp5936-sota-atomic-support-union-ab": "complete"}),
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
            lambda artifact: artifact["adversarial_verifier_receipts"].update(
                aggregation_artifact_task_ids_clean=[]
            ),
            "aggregation verifier receipt",
        ),
        (
            lambda artifact: artifact["task_owned_gate_receipts"].update(
                all_required_gate_kinds_present=False
            ),
            "task-owned gate",
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


def test_req_report_5961_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-5961: helper failures produce explicit blocked preconditions."""

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
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    assert mod._history_blocks(tmp_path) == []
    nonterminal_append = mod._append_completion_if_absent(tmp_path, terminal=False)
    assert nonterminal_append["reason"] == "nonterminal_identity_present"
    absent_history = tmp_path / "absent_history"
    absent_append = mod._append_completion_if_absent(absent_history, terminal=True)
    assert absent_append["append_count"] == 1

    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    assert mod._task_signature({"tasks": "bad"}) == ()
    assert mod._receipt_flags({}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": []}}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flag_count({"flag_count": 4}) == 4
    assert mod._receipt_max_severity({"max_severity": 3}) == 3
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert mod._status_and_verdict([], {"terminal_class_by_task_id": {"x": "complete"}}) == (
        "complete",
        "complete: archived terminal .527 identities into .528 with collision-free allocation and no inherited debt amplification",
    )
    assert mod._status_and_verdict([], {"terminal_class_by_task_id": {"x": "retired"}}) == (
        "complete_with_terminal_receipts",
        "complete: archived terminal .527 identities into .528 without outcome laundering; inherited global debt not amplified; next_range_collision_count=0",
    )
    assert (
        mod._classify_task(
            "exp5937-excluded-pool-coverage-audit",
            {},
            {"present": False},
            {"latest_status": "GATE_BLOCK"},
        )
        == "gate-blocked"
    )
    assert (
        mod._classify_task(
            "exp5937-excluded-pool-coverage-audit",
            {},
            {"present": False},
            {"latest_status": ""},
        )
        == "missing"
    )
    assert mod._classify_task("x", {"status": "unknown"}, {"present": True}, {}) == "missing"
    assert mod._outer_loop_hashes(tmp_path / "missing_results") == {}
    staging = mod._optional_staging_roadmap_receipt(
        {"milestone": "2026.07.528"},
        {"present": True, "loadable": True},
    )
    assert staging["reason"] == "present_optional_staging"
    assert mod._range_number_mentions("Exp5961 and experiment_5973") == {5961, 5973}
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {5961})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.ROADMAP_RELATIVE_PATH, {5961, 5973})
        == "allowed_allocation_reference"
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
    assert after_only_debt["global_suite_failure_delta"] == 0
    assert after_only_debt["global_spec_gap_delta"] == 0
    assert after_only_debt["root_clutter_delta"] == 0

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
    assert before_only_debt["global_suite_failure_after_node_ids"] == ["global_before"]
    assert before_only_debt["global_spec_gap_after_node_ids"] == ["spec_before"]
    assert before_only_debt["root_clutter_after_paths"] == ["root_before.py"]

    no_debt_rows = mod._debt_baselines_and_deltas(tmp_path, [])
    assert no_debt_rows["root_clutter_delta"] == 0

    _make_root(tmp_path / "receipt_root")
    receipt_report = _build(tmp_path / "receipt_root")
    receipt_matrix = receipt_report["activated_task_and_deliverable_matrix"]
    receipt_payloads = {
        task_id: _artifact(task_id)
        for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS
        if task_id != "exp5937-excluded-pool-coverage-audit"
    }
    receipt_payloads["exp5937-excluded-pool-coverage-audit"] = {}
    sparse_receipts = dict(_receipts())
    sparse_receipts.pop("exp5934-v527-source-delta-ingestion")
    sparse_receipts["exp5935-non-pruning-atomic-constraint-support"] = {
        **sparse_receipts["exp5935-non-pruning-atomic-constraint-support"],
        "stdout_json": {
            "reports": [
                {
                    "artifact": "x",
                    "loaded": True,
                    "flag_count": 1,
                    "max_severity": 1,
                    "flags": [{"kind": "WARN", "severity": "warn"}],
                }
            ]
        },
    }
    sparse_receipts["exp5936-sota-atomic-support-union-ab"] = {
        **sparse_receipts["exp5936-sota-atomic-support-union-ab"],
        "exit_code": 1,
        "stdout_json": {
            "reports": [
                {
                    "artifact": "x",
                    "loaded": True,
                    "flag_count": 1,
                    "max_severity": 3,
                    "flags": [{"kind": "CRITICAL", "severity": "critical"}],
                }
            ]
        },
    }
    grouped = mod._adversarial_receipts_group(sparse_receipts, receipt_matrix, receipt_payloads)
    assert grouped["failed_receipt_task_ids"] == ["exp5936-sota-atomic-support-union-ab"]
    assert grouped["warning_receipt_task_ids"] == ["exp5935-non-pruning-atomic-constraint-support"]

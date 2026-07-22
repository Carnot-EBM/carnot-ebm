"""Tests for the Exp5771 exact-deliverable evidence index.

Spec refs: REQ-REPORT-5771,
SCENARIO-REPORT-5771-EXACT-LOOKUP,
SCENARIO-REPORT-5771-FAIL-CLOSED,
SCENARIO-REPORT-5771-HISTORY-READONLY,
SCENARIO-REPORT-5771-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evidence_index_collision_preflight as mod  # noqa: E402


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: str | Path, payload: JsonDict) -> Path:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(root: Path, rel_path: str | Path, text: str) -> Path:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _task(task_id: Any, deliverable: Any, **extra: Any) -> JsonDict:
    row: JsonDict = {
        "id": task_id,
        "title": f"title for {mod.unwrap_value(task_id)}",
        "deliverable": deliverable,
        "result": "OK (conductor)",
    }
    row.update(extra)
    return row


def _history(
    tasks: list[JsonDict], *, copies: int = 1, milestone: str = mod.MILESTONE_FROM
) -> JsonDict:
    block = {
        "id": milestone,
        "title": "fixture milestone",
        "completed": "2026-07-21",
        "tasks": tasks,
    }
    return {"milestones": [deepcopy(block) for _ in range(copies)]}


def _minimal_root(root: Path, tasks: list[JsonDict], *, copies: int = 1) -> None:
    _write_text(
        root,
        "research-roadmap.yaml",
        yaml.safe_dump({"milestone": mod.MILESTONE_TO, "tasks": []}),
    )
    _write_text(root, "ops/conductor-log.md", "| t | task | OK | details |\n")
    _write_text(root, "scripts/research_conductor.py", "# conductor fixture\n")
    _write_text(root, "research-complete.yaml", yaml.safe_dump(_history(tasks, copies=copies)))
    for row in tasks:
        deliverable = mod.unwrap_value(row.get("deliverable"))
        if deliverable:
            _write_json(
                root,
                deliverable,
                {
                    "status": "complete",
                    "honest_verdict": f"complete: {mod.unwrap_value(row.get('id'))}",
                },
            )


def _qualification_root(root: Path) -> None:
    _minimal_root(root, [_task("exp5760-qualification", "results/experiment_5760_q.json")])
    _write_text(
        root,
        "openspec/capabilities/research-reporting/spec.md",
        "### REQ-REPORT-5784\n",
    )
    _write_text(root, "tests/python/test_evidence_index_collision_preflight.py", "# tests\n")
    _write_text(root, "scripts/evidence_index_collision_preflight.py", "# script\n")
    _write_text(root, "scripts/conductor_gates.py", "# gate script\n")
    _write_text(root, "scripts/exclusion_manifest_lint.py", "# lint script\n")
    _write_text(
        root,
        "research-roadmap.yaml",
        yaml.safe_dump(
            {
                "milestone": "2026.07.516",
                "tasks": [
                    {
                        "id": "exp5785-hardness-surface-prospective-fixture",
                        "deliverable": "results/experiment_5785_hardness_surface_fixture.json",
                        "gated_on": [
                            {
                                "upstream": mod.QUALIFICATION_EXPERIMENT_ID,
                                "artifact_field": "evidence_index_ready_score",
                                "op": "==",
                                "value": 1.0,
                            },
                            {
                                "upstream": mod.QUALIFICATION_EXPERIMENT_ID,
                                "artifact_field": "next_range_collision_count",
                                "op": "==",
                                "value": 0,
                            },
                            {
                                "upstream": mod.QUALIFICATION_EXPERIMENT_ID,
                                "artifact_field": "unresolved_canonical_count",
                                "op": "==",
                                "value": 0,
                            },
                            {
                                "upstream": mod.QUALIFICATION_EXPERIMENT_ID,
                                "artifact_field": "history_mutation_count",
                                "op": "==",
                                "value": 0,
                            },
                        ],
                    }
                ],
            }
        ),
    )
    mod.emit_report(
        root,
        output_path=root / mod.RESULT_RELATIVE_PATH,
        tests_run=[{"command": "prior-focused", "exit_code": 0}],
    )


def _passing_qualification_receipts() -> list[JsonDict]:
    return [
        {
            "run_id": "fixture",
            "command": "focused unit and coverage",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": "focused",
            "failure_signature": "",
        },
        {
            "run_id": "fixture",
            "command": "integration gate replay and hygiene",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": "integration",
            "failure_signature": "",
        },
        {
            "run_id": "fixture",
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 2,
            "ownership_class": "global_baseline",
            "suite_kind": "global_baseline",
            "failure_signature": "pre-existing unrelated global baseline failure",
            "pre_existing": True,
        },
        {
            "run_id": "fixture",
            "command": ".venv/bin/python scripts/check_spec_coverage.py",
            "exit_code": 1,
            "ownership_class": "spec_coverage",
            "suite_kind": "spec_coverage",
            "failure_signature": "pre-existing unrelated spec coverage baseline",
            "pre_existing": True,
        },
    ]


def test_req_report_5771_spec_declares_exact_deliverable_contract() -> None:
    """REQ-REPORT-5771: the OpenSpec names identity, aliases, and fields."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5771") :]

    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5771-EXACT-LOOKUP" in section
    assert "SCENARIO-REPORT-5771-FAIL-CLOSED" in section
    assert "canonical_lookup()" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_real_v514_alias_groups_remain_aliases_not_canonical() -> None:
    """SCENARIO-REPORT-5771-EXACT-LOOKUP: real 5760/5764/5766 collisions are safe."""

    report = mod.build_report(
        REPO,
        tests_run=[{"command": "unit", "exit_code": 0}],
        result_path=REPO / mod.RESULT_RELATIVE_PATH,
    )

    assert report["status"] == "complete"
    assert report["evidence_index_ready_score"] == 1.0
    assert report["next_range_collision_count"] == 0
    assert report["unresolved_canonical_count"] == 0
    for number, canonical_path, alias_path in (
        (
            "5760",
            "results/experiment_5760_selective_exact_feedback_search.json",
            "results/experiment_5760_cegis_refinement_induction_ab.json",
        ),
        (
            "5764",
            "results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json",
            "results/experiment_5764_gemma31b_singleshot_induction_ab.json",
        ),
        (
            "5766",
            "results/experiment_5766_arc_loo_component_interaction_audit.json",
            "results/experiment_5766_gemma31b_cegis_refinement_ab.json",
        ),
    ):
        group = report["same_number_alias_groups"][number]
        task_id = group["canonical"]["task_id"]
        receipt = mod.canonical_lookup(
            report["canonical_task_index"],
            mod.MILESTONE_FROM,
            task_id,
            canonical_path,
        )

        assert receipt["status"] == "resolved"
        assert receipt["resolved_path"] == canonical_path
        assert group["canonical"]["path"] == canonical_path
        assert alias_path in [alias["path"] for alias in group["aliases"]]
        assert group["canonical"]["role"] == "canonical_declared_deliverable"
        assert {alias["role"] for alias in group["aliases"]} == {"same_number_alias"}


def test_mtime_inversion_control_declared_path_wins(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5771-EXACT-LOOKUP: newer aliases cannot become canonical."""

    canonical_rel = "results/experiment_5760_declared.json"
    alias_rel = "results/experiment_5760_newer_alias.json"
    task_id = "exp5760-declared-task"
    _minimal_root(tmp_path, [_task(task_id, canonical_rel)])
    alias = _write_json(
        tmp_path,
        alias_rel,
        {"status": "complete", "honest_verdict": "complete: newer alias"},
    )
    canonical = tmp_path / canonical_rel
    os.utime(canonical, (100.0, 100.0))
    os.utime(alias, (200.0, 200.0))

    report = mod.build_report(tmp_path, result_path=tmp_path / mod.RESULT_RELATIVE_PATH)
    receipt = mod.canonical_lookup(
        report["canonical_task_index"],
        mod.MILESTONE_FROM,
        task_id,
        canonical_rel,
    )

    assert receipt["status"] == "resolved"
    assert receipt["resolved_path"] == canonical_rel
    assert report["same_number_alias_groups"]["5760"]["mtime_selected_path"] == alias_rel
    assert report["mtime_inversion_control"]["alias_newer_than_canonical"] is True
    assert report["mtime_inversion_control"]["canonical_lookup_resolved_path"] == canonical_rel


def test_missing_duplicate_hash_conflict_and_wrapper_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5771-FAIL-CLOSED: malformed identities are diagnostics."""

    wrapped_task_id = {"principle": "wrapper control", "value": "exp5760-wrapped"}
    wrapped_deliverable = {
        "principle": "wrapper control",
        "value": "results/experiment_5760_wrapped.json",
    }
    _minimal_root(tmp_path, [_task(wrapped_task_id, wrapped_deliverable)])
    wrapped_report = mod.build_report(tmp_path, result_path=tmp_path / mod.RESULT_RELATIVE_PATH)
    assert wrapped_report["canonical_task_index"][0]["task_id"] == "exp5760-wrapped"
    assert wrapped_report["negative_control_receipts"]["wrapper_field"]["resolved"] is True

    missing_root = tmp_path / "missing"
    _minimal_root(
        missing_root,
        [
            _task(
                "exp5760-missing",
                "results/experiment_5760_missing.json",
                result="",
            )
        ],
    )
    (missing_root / "results/experiment_5760_missing.json").unlink()
    missing_report = mod.build_report(
        missing_root,
        result_path=missing_root / mod.RESULT_RELATIVE_PATH,
    )
    missing_receipt = mod.canonical_lookup(
        missing_report["canonical_task_index"],
        mod.MILESTONE_FROM,
        "exp5760-missing",
        "results/experiment_5760_missing.json",
    )
    assert missing_report["status"] == "blocked"
    assert missing_report["unresolved_canonical_count"] == 1
    assert missing_receipt["status"] == "unresolved"
    assert missing_receipt["same_number_candidates"] == []

    duplicate_root = tmp_path / "duplicate"
    _minimal_root(
        duplicate_root,
        [
            _task("exp5760-dup", "results/experiment_5760_a.json"),
            _task("exp5760-dup", "results/experiment_5760_b.json"),
        ],
    )
    duplicate_report = mod.build_report(
        duplicate_root,
        result_path=duplicate_root / mod.RESULT_RELATIVE_PATH,
    )
    assert duplicate_report["status"] == "blocked"
    assert duplicate_report["duplicate_task_ids"] == [
        {
            "milestone": mod.MILESTONE_FROM,
            "task_id": "exp5760-dup",
            "declared_deliverables": [
                "results/experiment_5760_a.json",
                "results/experiment_5760_b.json",
            ],
        }
    ]

    conflict_root = tmp_path / "conflict"
    tasks = [
        _task(
            "exp5760-conflict",
            "results/experiment_5760_conflict.json",
            sha256="sha256:aaa",
        )
    ]
    _minimal_root(conflict_root, tasks)
    changed = deepcopy(tasks)
    changed[0]["sha256"] = "sha256:bbb"
    blocks = _history(tasks)
    blocks["milestones"].append(_history(changed)["milestones"][0])
    _write_text(conflict_root, "research-complete.yaml", yaml.safe_dump(blocks))
    conflict_report = mod.build_report(
        conflict_root,
        result_path=conflict_root / mod.RESULT_RELATIVE_PATH,
    )
    assert conflict_report["status"] == "blocked"
    assert conflict_report["conflicting_hashes"] == [
        {
            "identity": [
                mod.MILESTONE_FROM,
                "exp5760-conflict",
                "results/experiment_5760_conflict.json",
            ],
            "declared_hashes": ["sha256:aaa", "sha256:bbb"],
        }
    ]


def test_duplicate_history_blocks_are_readonly_diagnostics(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5771-HISTORY-READONLY: history duplicates are preserved."""

    task_id = "exp5760-duplicate-history"
    deliverable = "results/experiment_5760_duplicate_history.json"
    _minimal_root(tmp_path, [_task(task_id, deliverable)], copies=3)
    before = (tmp_path / "research-complete.yaml").read_bytes()

    report = mod.emit_report(
        tmp_path,
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        tests_run=[{"command": "unit", "exit_code": 0}],
    )
    after = (tmp_path / "research-complete.yaml").read_bytes()

    assert before == after
    assert report["research_complete_modified"] is False
    assert report["history_mutation_count"] == 0
    assert report["duplicate_history_blocks"] == [
        {
            "milestone": mod.MILESTONE_FROM,
            "block_count": 3,
            "unique_block_signature_count": 1,
            "mutation": "preserved_read_only",
        }
    ]


def test_artifact_schema_producer_fields_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5771-FIELD-PRINCIPLES: emitted artifact gates are stable."""

    task_id = "exp5760-schema"
    deliverable = "results/experiment_5760_schema.json"
    _minimal_root(tmp_path, [_task(task_id, deliverable)])
    output = tmp_path / mod.RESULT_RELATIVE_PATH

    report = mod.emit_report(
        tmp_path,
        output_path=output,
        tests_run=[{"command": "unit", "exit_code": 0}],
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written == report
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert set(report).issubset(report["field_principles"])
    assert all(report["field_principles"][field] for field in report)
    assert report["research_complete_hash_before"] == report["research_complete_hash_after"]
    assert report["producer_gate_fields"] == {
        "evidence_index_ready_score": report["evidence_index_ready_score"],
        "next_range_collision_count": report["next_range_collision_count"],
        "unresolved_canonical_count": report["unresolved_canonical_count"],
        "history_mutation_count": report["history_mutation_count"],
    }
    assert all(not isinstance(value, dict) for value in report["producer_gate_fields"].values())
    assert report["conductor_unchanged"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    old = mod.FIELD_PRINCIPLES.pop("status")
    try:
        with pytest.raises(KeyError, match="missing field principles"):
            mod.build_report(tmp_path, result_path=output)
    finally:
        mod.FIELD_PRINCIPLES["status"] = old


def test_defensive_branches_and_cli_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5771: malformed inputs fail closed and CLI emits the receipt."""

    invalid_yaml = _write_text(tmp_path, "invalid.yaml", ":\n")
    yaml_list = _write_text(tmp_path, "list.yaml", "- a\n")
    assert mod.read_yaml_mapping(invalid_yaml)[1]["parsed"] is False
    assert "expected mapping" in mod.read_yaml_mapping(yaml_list)[1]["error"]

    invalid_json = _write_text(tmp_path, "invalid.json", "{")
    json_list = _write_text(tmp_path, "list.json", "[]")
    assert mod.read_json_mapping(invalid_json)[1]["loadable"] is False
    assert "expected mapping" in mod.read_json_mapping(json_list)[1]["error"]

    bad_history_root = tmp_path / "bad-history"
    _write_text(bad_history_root, "research-complete.yaml", yaml.safe_dump({"milestones": "bad"}))
    assert mod._history_blocks(bad_history_root) == []
    assert mod._selected_tasks(bad_history_root) == []

    bad_tasks_root = tmp_path / "bad-tasks"
    _write_text(
        bad_tasks_root,
        "research-complete.yaml",
        yaml.safe_dump({"milestones": [{"id": mod.MILESTONE_FROM, "tasks": "bad"}]}),
    )
    assert mod._selected_tasks(bad_tasks_root) == []
    assert mod.conflicting_hashes(bad_tasks_root) == []

    non_mapping_root = tmp_path / "non-mapping"
    _write_text(
        non_mapping_root,
        "research-complete.yaml",
        yaml.safe_dump({"milestones": [{"id": mod.MILESTONE_FROM, "tasks": ["bad"]}]}),
    )
    assert mod.conflicting_hashes(non_mapping_root) == []
    assert mod.same_number_candidates(non_mapping_root, "not-an-exp") == []
    assert mod.same_number_candidates(non_mapping_root, "exp5760-no-results-dir") == []
    assert mod._artifact_status({}, {"exists": True, "loadable": False}) == "malformed"
    assert (
        mod._artifact_status({"honest_verdict": "blocked: x"}, {"exists": True, "loadable": True})
        == "blocked"
    )

    skip_root = tmp_path / "skip-row"
    _write_text(
        skip_root,
        "research-complete.yaml",
        yaml.safe_dump(
            {
                "milestones": [
                    {
                        "id": mod.MILESTONE_FROM,
                        "tasks": [{"id": "", "deliverable": ""}],
                    }
                ]
            }
        ),
    )
    assert mod.build_canonical_task_index(skip_root) == ([], {})
    assert (
        mod.same_number_alias_groups(
            skip_root,
            [
                {
                    "task_id": "bad",
                    "alias_paths": ["results/experiment_0000_alias.json"],
                    "declared_deliverable": "results/x.json",
                }
            ],
        )
        == {}
    )

    collision_root = tmp_path / "collision"
    _minimal_root(
        collision_root,
        [_task("exp5760-collision", "results/experiment_5760_collision.json")],
    )
    _write_json(collision_root, "results/experiment_5772_unowned_collision.json", {})
    collision_report = mod.build_report(
        collision_root,
        tests_run=[{"command": "unit", "exit_code": 0}],
        result_path=collision_root / mod.RESULT_RELATIVE_PATH,
    )
    assert collision_report["status"] == "blocked"
    assert (
        "next_range_collisions" in collision_report["preconditions_checked"]["failed_preconditions"]
    )

    no_tasks_root = tmp_path / "no-tasks"
    _write_text(no_tasks_root, "research-roadmap.yaml", ":\n")
    _write_text(no_tasks_root, "research-complete.yaml", yaml.safe_dump({"milestones": []}))
    no_tasks_report = mod.build_report(
        no_tasks_root,
        tests_run=[{"command": "unit", "exit_code": 0}],
        result_path=no_tasks_root / mod.RESULT_RELATIVE_PATH,
    )
    assert (
        "active_roadmap_unparseable"
        in no_tasks_report["preconditions_checked"]["failed_preconditions"]
    )
    assert (
        "no_v514_history_tasks" in no_tasks_report["preconditions_checked"]["failed_preconditions"]
    )

    mutated_root = tmp_path / "mutated"
    _minimal_root(mutated_root, [_task("exp5760-mutated", "results/experiment_5760_mutated.json")])
    original_sha = mod.sha256_file
    calls = {"research_complete": 0}

    def fake_sha(path: Path) -> str | None:
        if path.name == "research-complete.yaml":
            calls["research_complete"] += 1
            return f"sha256:fake-{calls['research_complete']}"
        return original_sha(path)

    monkeypatch.setattr(mod, "sha256_file", fake_sha)
    mutated_report = mod.build_report(
        mutated_root,
        tests_run=[{"command": "unit", "exit_code": 0}],
        result_path=mutated_root / mod.RESULT_RELATIVE_PATH,
    )
    assert mutated_report["history_mutation_count"] == 1
    assert "history_mutated" in mutated_report["preconditions_checked"]["failed_preconditions"]

    cli_root = tmp_path / "cli"
    _minimal_root(cli_root, [_task("exp5760-cli", "results/experiment_5760_cli.json")])
    tests_json = _write_json(cli_root, "tests-run.json", [{"command": "unit", "exit_code": 0}])
    output = cli_root / mod.RESULT_RELATIVE_PATH
    assert (
        mod.main(
            [
                "--root",
                str(cli_root),
                "--output",
                str(output),
                "--tests-run-json",
                str(tests_json),
            ]
        )
        == 0
    )
    assert output.exists()
    assert mod._load_tests_run(None) == []
    assert mod._load_tests_run(tests_json) == [{"command": "unit", "exit_code": 0}]
    bad_tests_json = _write_json(cli_root, "bad-tests.json", {"command": "unit"})
    with pytest.raises(ValueError, match="tests-run JSON must be a list"):
        mod._load_tests_run(bad_tests_json)


def test_req_report_5784_spec_declares_terminal_qualification_contract() -> None:
    """REQ-REPORT-5784: OpenSpec separates task-owned readiness from global health."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5784") :]

    assert "SCENARIO-REPORT-5784-TASK-OWNED-READINESS" in section
    assert "SCENARIO-REPORT-5784-TASK-OWNED-BLOCK" in section
    assert "SCENARIO-REPORT-5784-GATE-REPLAY" in section
    assert mod.QUALIFICATION_INFERENCE_SUBSTRATE in section
    for field in mod.QUALIFICATION_REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_terminal_qualification_discloses_global_failures_without_blocking(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5784-TASK-OWNED-READINESS: task-owned checks authorize readiness."""

    _qualification_root(tmp_path)
    output = tmp_path / mod.QUALIFICATION_RESULT_RELATIVE_PATH

    report = mod.emit_terminal_qualification(
        tmp_path,
        output_path=output,
        test_receipts=_passing_qualification_receipts(),
        implementation_hashes_before={"scripts/evidence_index_collision_preflight.py": "before"},
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written == report
    assert report["status"] == "complete"
    assert report["evidence_index_ready_score"] == 1.0
    assert report["next_range_collision_count"] == 0
    assert report["unresolved_canonical_count"] == 0
    assert report["history_mutation_count"] == 0
    assert report["producer_gate_fields"] == {
        "evidence_index_ready_score": 1.0,
        "next_range_collision_count": 0,
        "unresolved_canonical_count": 0,
        "history_mutation_count": 0,
    }
    assert report["task_owned_failures"] == []
    assert {row["ownership_class"] for row in report["focused_test_receipts"]} == {"task_owned"}
    assert {row["ownership_class"] for row in report["integration_test_receipts"]} == {"task_owned"}
    assert report["global_baseline_receipts"][0]["exit_code"] == 2
    assert report["spec_coverage_receipts"][0]["exit_code"] == 1
    assert len(report["pre_existing_global_failures"]) == 2
    assert report["research_complete_modified"] is False
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["bootstrap_skeleton_absent"] is True
    assert report["terminal_finalizer_receipt"]["atomic_replace"] is True
    assert report["terminal_finalizer_receipt"]["reloaded_checksum_match"] is True
    assert all(row["passed"] is True for row in report["gate_replay_receipts"])
    assert mod.payload_checksum(report) == report["reproducibility_checksum"]


def test_terminal_qualification_blocks_task_owned_failures(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5784-TASK-OWNED-BLOCK: task-owned failures block readiness."""

    _qualification_root(tmp_path)
    receipts = _passing_qualification_receipts()
    receipts[1] = receipts[1] | {
        "exit_code": 1,
        "failure_signature": "new integration gate replay failure",
    }

    report = mod.emit_terminal_qualification(
        tmp_path,
        output_path=tmp_path / mod.QUALIFICATION_RESULT_RELATIVE_PATH,
        test_receipts=receipts,
        implementation_hashes_before={},
    )

    assert report["status"] == "blocked"
    assert report["evidence_index_ready_score"] == 0.0
    assert report["task_owned_failures"] == [report["integration_test_receipts"][0]]
    assert report["pre_existing_global_failures"]
    assert report["honest_verdict"].startswith("blocked:")


def test_terminal_qualification_gate_replay_uses_reopened_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5784-GATE-REPLAY: the on-disk artifact is gate authority."""

    _qualification_root(tmp_path)
    output = tmp_path / mod.QUALIFICATION_RESULT_RELATIVE_PATH
    report = mod.emit_terminal_qualification(
        tmp_path,
        output_path=output,
        test_receipts=_passing_qualification_receipts(),
        implementation_hashes_before={},
    )
    replay = mod.replay_qualification_gates(tmp_path, output)

    assert replay == report["gate_replay_receipts"]
    assert all(row["source"] == output.relative_to(tmp_path).as_posix() for row in replay)
    assert all(row["reopened_from_disk"] is True for row in replay)
    assert all(row["bootstrap_skeleton_absent"] is True for row in replay)


def test_terminal_qualification_defensive_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5784: malformed receipts, collisions, and mutation checks fail closed."""

    assert mod.normalize_qualification_receipts([{"command": "bool", "exit_code": True}]) == [
        {
            "command": "bool",
            "exit_code": 1,
            "ownership_class": "global_baseline",
            "suite_kind": "",
            "failure_signature": "",
            "pre_existing": False,
        }
    ]
    assert mod.bootstrap_skeleton_absent({"status": "complete"}) is False

    history_root = tmp_path / "history-owned"
    _write_text(
        history_root,
        "research-complete.yaml",
        yaml.safe_dump(
            {
                "milestones": [
                    "bad",
                    {"tasks": "bad"},
                    {"tasks": ["bad", {"deliverable": "results/experiment_5773_owned.json"}]},
                ]
            }
        ),
    )
    _write_text(history_root, "research-roadmap.yaml", yaml.safe_dump({"tasks": []}))
    _write_json(history_root, "results/experiment_5773_owned.json", {"status": "blocked"})
    assert (
        mod.next_range_collision_scan(
            history_root,
            history_root / mod.RESULT_RELATIVE_PATH,
        )["collision_count"]
        == 0
    )

    scan_root = tmp_path / "range-scan"
    _write_text(
        scan_root,
        "research-roadmap.yaml",
        yaml.safe_dump(
            {
                "tasks": [
                    "bad",
                    {
                        "id": "exp5785-hardness-surface-prospective-fixture",
                        "deliverable": "results/experiment_5785_hardness_surface_fixture.json",
                    },
                ]
            }
        ),
    )
    _write_json(scan_root, "results/experiment_5785_hardness_surface_fixture.json", {})
    _write_text(scan_root, "python/carnot/experiment_5782_existing.py", "# owned\n")
    _write_json(scan_root, "results/experiment_5794_unowned.json", {})
    range_scan = mod.exp5782_5795_collision_scan(
        scan_root,
        mod.QUALIFICATION_RESULT_RELATIVE_PATH,
    )
    assert range_scan["collision_count"] == 1
    assert range_scan["owned_paths"] == [
        {
            "path": "python/carnot/experiment_5782_existing.py",
            "ownership": "owned_implementation_or_test_path",
        },
        {
            "path": "results/experiment_5785_hardness_surface_fixture.json",
            "ownership": "active_roadmap_declared_deliverable",
        },
    ]

    bad_gate_root = tmp_path / "bad-gates"
    _write_text(bad_gate_root, "research-roadmap.yaml", yaml.safe_dump({"tasks": "bad"}))
    assert mod._qualification_gate_task(bad_gate_root) is None
    no_match_root = tmp_path / "no-match-gates"
    _write_text(
        no_match_root,
        "research-roadmap.yaml",
        yaml.safe_dump({"tasks": ["bad", {"id": "exp0000-other"}]}),
    )
    assert mod._qualification_gate_task(no_match_root) is None
    missing_replay = mod.replay_qualification_gates(
        no_match_root,
        no_match_root / mod.QUALIFICATION_RESULT_RELATIVE_PATH,
    )
    assert missing_replay[0]["passed"] is False
    assert missing_replay[0]["reason"] == "missing"

    _qualification_root(no_match_root)
    emitted = mod.emit_terminal_qualification(
        no_match_root,
        output_path=no_match_root / mod.QUALIFICATION_RESULT_RELATIVE_PATH,
        test_receipts=_passing_qualification_receipts(),
        implementation_hashes_before={},
    )
    _write_text(no_match_root, "research-roadmap.yaml", yaml.safe_dump({"tasks": []}))
    no_task_replay = mod.replay_qualification_gates(
        no_match_root,
        no_match_root / mod.QUALIFICATION_RESULT_RELATIVE_PATH,
    )
    assert emitted["status"] == "complete"
    assert no_task_replay[0]["reason"] == "exp5785 gate task not found in active roadmap"

    old = mod.QUALIFICATION_FIELD_PRINCIPLES.pop("status")
    try:
        with pytest.raises(KeyError, match="missing qualification field principles"):
            mod.build_terminal_qualification_report(
                no_match_root,
                test_receipts=_passing_qualification_receipts(),
            )
    finally:
        mod.QUALIFICATION_FIELD_PRINCIPLES["status"] = old

    collision_root = tmp_path / "qualification-collision"
    _qualification_root(collision_root)
    _write_json(collision_root, "results/experiment_5794_unowned.json", {})
    collision_report = mod.build_terminal_qualification_report(
        collision_root,
        test_receipts=_passing_qualification_receipts(),
    )
    assert (
        "exp5782_exp5795_unowned_collisions"
        in collision_report["preconditions_checked"]["failed_preconditions"]
    )

    duplicate_root = tmp_path / "qualification-duplicate"
    _qualification_root(duplicate_root)
    _minimal_root(
        duplicate_root,
        [
            _task("exp5760-dup", "results/experiment_5760_a.json"),
            _task("exp5760-dup", "results/experiment_5760_b.json"),
        ],
    )
    mod.emit_report(
        duplicate_root,
        output_path=duplicate_root / mod.RESULT_RELATIVE_PATH,
        tests_run=[{"command": "prior-focused", "exit_code": 0}],
    )
    duplicate_report = mod.build_terminal_qualification_report(
        duplicate_root,
        test_receipts=_passing_qualification_receipts(),
    )
    assert (
        "ambiguous_exact_identities"
        in duplicate_report["preconditions_checked"]["failed_preconditions"]
    )

    protected_root = tmp_path / "protected-mutation"
    _qualification_root(protected_root)
    original_sha = mod.sha256_file
    calls = {"roadmap": 0, "conductor": 0}

    def fake_sha(path: Path) -> str | None:
        if path.name == "research-roadmap.yaml":
            calls["roadmap"] += 1
            return f"sha256:roadmap-{calls['roadmap']}"
        if path.name == "research_conductor.py":
            calls["conductor"] += 1
            return f"sha256:conductor-{calls['conductor']}"
        return original_sha(path)

    monkeypatch.setattr(mod, "sha256_file", fake_sha)
    protected_report = mod.build_terminal_qualification_report(
        protected_root,
        test_receipts=_passing_qualification_receipts(),
    )
    protected_failed = protected_report["preconditions_checked"]["failed_preconditions"]
    assert "research_roadmap_modified" in protected_failed
    assert "research_conductor_modified" in protected_failed
    monkeypatch.setattr(mod, "sha256_file", original_sha)

    cli_root = tmp_path / "qualification-cli"
    _qualification_root(cli_root)
    receipts_json = _write_json(cli_root, "receipts.json", _passing_qualification_receipts())
    before_json = _write_json(cli_root, "before.json", {"script": "before"})
    assert (
        mod.main(
            [
                "--root",
                str(cli_root),
                "--output",
                str(cli_root / mod.QUALIFICATION_RESULT_RELATIVE_PATH),
                "--qualify-terminal",
                "--qualification-tests-run-json",
                str(receipts_json),
                "--implementation-hashes-before-json",
                str(before_json),
            ]
        )
        == 0
    )
    assert mod._load_json_mapping_file(None) == {}
    bad_map_json = _write_json(cli_root, "bad-map.json", [])
    with pytest.raises(ValueError, match="JSON file must be a mapping"):
        mod._load_json_mapping_file(bad_map_json)

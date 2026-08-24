"""Tests for the Exp6579 V571 terminal-recovery contract.

Spec refs: REQ-REPORT-6579, SCENARIO-REPORT-6579-NO-ARTIFACT,
SCENARIO-REPORT-6579-GATE-SKIPS, SCENARIO-REPORT-6579-AUDIT,
SCENARIO-REPORT-6579-DECOMPOSITION, SCENARIO-REPORT-6579-ATTACKS,
SCENARIO-REPORT-6579-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6579_v572_terminal_recovery_and_decomposition_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """Build one deterministic report so tests inspect the same evidence."""

    return mod.build_report(
        REPO,
        date="20260824",
        duration_s=1.25,
        tests_run=[{"command": "focused fixture", "exit_code": 0, "duration_s": 0.01}],
    )


def test_req_report_6579_spec_declares_terminal_recovery_contract() -> None:
    """REQ-REPORT-6579: the spec names every recovery scenario."""

    text = SPEC.read_text(encoding="utf-8")
    for anchor in (
        "REQ-REPORT-6579",
        "REQ-REPORT-6579-PRECONDITIONS",
        "REQ-REPORT-6579-TIMEOUTS",
        "REQ-REPORT-6579-TERMINALS",
        "REQ-REPORT-6579-DECOMPOSITION",
        "REQ-REPORT-6579-GATES",
        "REQ-REPORT-6579-ATTACKS",
        "REQ-REPORT-6579-REDUCER",
        "REQ-REPORT-6579-ATOMIC",
        "SCENARIO-REPORT-6579-NO-ARTIFACT",
        "SCENARIO-REPORT-6579-GATE-SKIPS",
        "SCENARIO-REPORT-6579-AUDIT",
        "SCENARIO-REPORT-6579-DECOMPOSITION",
        "SCENARIO-REPORT-6579-ATTACKS",
        "SCENARIO-REPORT-6579-ATOMIC",
    ):
        assert anchor in text


def test_scenario_report_6579_no_artifact_replays_three_exact_attempts(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6579-NO-ARTIFACT: timeout rows remain distinct."""

    rows = report["exp6575_timeout_attempt_rows"]
    assert [row["attempt_index"] for row in rows] == [1, 2, 3]
    assert [row["elapsed_s"] for row in rows] == [4801, 4803, 4804]
    assert all(row["terminal_code"] == "hard_wall_clock_cap" for row in rows)
    assert all(row["hard_cap_s"] == 4800 for row in rows)
    assert all(row["agent_backend"] == "codex_cli" for row in rows)
    assert all(row["artifact_exists_after_attempt"] is False for row in rows)
    assert all(row["log_source_sha256"] == rows[0]["log_source_sha256"] for row in rows)
    assert all(row["start_utc_derived"] < row["end_utc_logged"] for row in rows)
    assert mod.timeout_attempt_rows_valid(rows, rows[0]["log_source_sha256"])


def test_scenario_report_6579_gate_skips_and_audit_stay_distinct(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6579-GATE-SKIPS and AUDIT preserve task classes."""

    rows = {row["task_id"]: row for row in report["v571_terminal_rows"]}
    assert set(rows) == set(mod.V571_TASKS)
    assert rows[mod.EXP6575_TASK_ID]["terminal_class"] == "hard_timeout_no_artifact"
    assert "honest_verdict" not in rows[mod.EXP6575_TASK_ID]
    assert rows[mod.EXP6576_TASK_ID]["terminal_class"] == "gate_skip"
    assert rows[mod.EXP6576_TASK_ID]["science_disposition"] == "not_run_gate_skip"
    assert rows[mod.EXP6578_TASK_ID]["terminal_class"] == "gate_skip"
    assert rows[mod.EXP6578_TASK_ID]["science_disposition"] == "not_run_gate_skip"
    audit = rows[mod.EXP6577_TASK_ID]
    assert audit["terminal_class"] == "independent_audit_blocked_diagnosis"
    assert audit["stored_verdict_class"] == "blocked"
    assert audit["stored_ready_score"] == 0.0
    assert audit["stored_first_failed_check"]["observed"] == "missing"
    assert "experiment_6576" in audit["stored_first_failed_check"]["field"]
    assert mod.terminal_rows_valid(list(rows.values()))


def test_scenario_report_6579_decomposition_freezes_one_family_budgets(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6579-DECOMPOSITION: every family task is bounded."""

    contract = report["decomposition_contract"]
    tasks = contract["model_task_rows"]
    assert len(tasks) == 3
    assert {row["task_id"] for row in tasks} == set(mod.MODEL_TASK_FAMILIES)
    assert {tuple(row["mandated_model_families"]) for row in tasks} == {
        (family,) for family in mod.MODEL_TASK_FAMILIES.values()
    }
    for row in tasks:
        assert row["fresh_process_per_task"] is True
        assert row["fresh_context_per_task"] is True
        assert row["max_source_units"] == 3
        assert row["task_timeout_s"] == 4200
        assert row["checkpoint_interval_s"] <= 300
        assert row["cleanup_budget_s"] > 0
        assert row["terminal_output_budget_s"] > 0
        assert row["verified_unload_required"] is True
        assert row["terminal_artifact_count"] == 1
        assert row["cross_family_aggregate_allowed"] is False
    assert mod.decomposition_contract_valid(contract)


def test_req_report_6579_current_roadmap_gate_map_uses_declared_fields(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6579-GATES: owners and consumers use exact field names."""

    rows = report["current_roadmap_gate_contract_rows"]
    assert rows
    assert all(row["passed"] is True for row in rows)
    assert all(row["resolved_roadmap_path"] == "research-roadmap.yaml" for row in rows)
    assert all(row["roadmap_next_exists"] is False for row in rows)
    assert mod.gate_contract_rows_valid(rows)
    owned = {(row["owner_task_id"], row["artifact_field"]) for row in rows}
    assert set(mod.EXPECTED_READINESS_FIELDS.items()) <= owned


def test_scenario_report_6579_attacks_reject_every_false_recovery(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6579-ATTACKS: each mutation fails closed."""

    attacks = report["attack_rows"]
    assert {row["attack_id"] for row in attacks} == set(mod.REQUIRED_ATTACKS)
    assert all(row["passed"] is True for row in attacks)
    assert all(row["candidate_ready_score"] == 0.0 for row in attacks)

    timeouts = deepcopy(report["exp6575_timeout_attempt_rows"])
    timeouts[1]["attempt_index"] = 1
    assert not mod.timeout_attempt_rows_valid(timeouts, timeouts[0]["log_source_sha256"])
    timeouts = deepcopy(report["exp6575_timeout_attempt_rows"])
    timeouts[0]["elapsed_s"] -= 1
    assert not mod.timeout_attempt_rows_valid(timeouts, timeouts[0]["log_source_sha256"])

    terminal = deepcopy(report["v571_terminal_rows"])
    terminal.pop()
    assert not mod.terminal_rows_valid(terminal)
    gates = deepcopy(report["current_roadmap_gate_contract_rows"])
    gates[0]["owner_task_id"] = mod.EXP6578_TASK_ID
    assert not mod.gate_contract_rows_valid(gates)
    contract = deepcopy(report["decomposition_contract"])
    contract["model_task_rows"][0]["mandated_model_families"].append("second-family")
    assert not mod.decomposition_contract_valid(contract)


def test_req_report_6579_ready_reducer_and_principles_are_complete(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6579-REDUCER: readiness and provenance recompute."""

    assert report["v572_decomposition_contract_ready_score"] == 1.0
    assert report["status"] == "complete_v572_terminal_recovery_and_decomposition_contract_ready"
    assert report["verdict_class"] == "null"
    assert report["inference_substrate"] == "v571_terminal_receipt_replay_no_llm"
    assert report["verifier_is_oracle"] is True
    assert report["gate_check_summary"]["passed"] is True
    assert report["gate_check_summary"]["failed_check_count"] == 0
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert all(item["principle"] for item in report["field_provenance"].values())
    assert report["reproducibility_checksum"] == mod.artifact_checksum(report)
    assert mod.validate_report(report) == []


def test_req_report_6579_preconditions_record_resources_and_no_llm(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6579-PRECONDITIONS: local substrate receipts are explicit."""

    checks = report["preconditions_checked"]
    assert checks["planning_date"] == "20260824"
    assert checks["model_inference_invoked"] is False
    assert checks["inference_substrate"] == "v571_terminal_receipt_replay_no_llm"
    assert checks["conductor_log"]["sha256"].startswith("sha256:")
    assert checks["failure_ledger"]["sha256"].startswith("sha256:")
    assert checks["exclusion_manifest"]["sha256"].startswith("sha256:")
    assert checks["python"]["version"]
    assert {row["tool"] for row in checks["tool_versions"]} == {
        "coverage",
        "pytest",
        "ruff",
    }
    assert checks["cpu"]["logical_count"] > 0
    assert checks["ram"]["total_bytes"] > 0
    assert checks["disk"]["free_bytes"] > 0
    assert set(checks["expected_v571_artifacts"]) == set(mod.V571_TASKS)


def test_scenario_report_6579_atomic_writer_and_validator_fail_closed(
    tmp_path: Path, report: dict[str, Any]
) -> None:
    """SCENARIO-REPORT-6579-ATOMIC: one atomic JSON validates."""

    output = tmp_path / "result.json"
    receipt = mod.atomic_write_report(output, report)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["atomic_replace"] is True
    assert receipt["temporary_path_exists_after_replace"] is False
    assert receipt["output_sha256"] == mod.sha256_file(output)
    assert loaded == report
    assert not list(tmp_path.glob(".*.tmp"))

    bad = deepcopy(report)
    bad["verdict_class"] = "positive"
    assert "verdict_class must be null when ready" in mod.validate_report(bad)
    with pytest.raises(ValueError, match="verdict_class must be null when ready"):
        mod.atomic_write_report(tmp_path / "bad.json", bad)

    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:stale"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)


def test_req_report_6579_helpers_reject_missing_receipts_and_bad_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, Any]
) -> None:
    """REQ-REPORT-6579-ATOMIC: defensive input paths do not pass silently."""

    log_text = (REPO / "ops/conductor-log.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="expected three Exp6575 hard-timeout attempts"):
        mod.parse_exp6575_timeout_attempts(log_text.replace("after 4803s", "after missing"), "x")

    missing = deepcopy(report)
    del missing["status"]
    assert "missing required fields: status" in mod.validate_report(missing)
    missing["status"] = report["status"]
    missing["duration_s"] = 0
    assert "duration_s must be positive and finite" in mod.validate_report(missing)

    output = tmp_path / "cli.json"
    monkeypatch.setattr(mod, "REPO_ROOT", REPO)
    assert mod.main(["--date", "20260824", "--output", str(output)]) == 0
    assert output.is_file()
    assert mod.main(["--validate", "--output", str(output)]) == 0
    output.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", "--output", str(output)]) == 1


def test_req_report_6579_timeout_and_terminal_guards_cover_each_receipt(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6579-TIMEOUTS/TERMINALS: every receipt guard fails closed."""

    timeout_rows = report["exp6575_timeout_attempt_rows"]
    log_hash = timeout_rows[0]["log_source_sha256"]
    assert not mod.timeout_attempt_rows_valid(timeout_rows[:2], log_hash)
    timeout_mutations = (
        ("task_id", "wrong-task"),
        ("terminal_code", "success"),
        ("hard_cap_s", 1),
        ("agent_backend", "wrong-backend"),
        ("artifact_exists_after_attempt", True),
        ("log_source_sha256", "sha256:stale"),
        ("row_hash", "sha256:stale"),
    )
    for key, value in timeout_mutations:
        mutated = deepcopy(timeout_rows)
        mutated[0][key] = value
        assert not mod.timeout_attempt_rows_valid(mutated, log_hash)

    terminal_rows = report["v571_terminal_rows"]
    terminal_mutations = (
        (mod.EXP6575_TASK_ID, "terminal_class", "success"),
        (mod.EXP6575_TASK_ID, "artifact_exists", True),
        (mod.EXP6575_TASK_ID, "honest_verdict", "invented"),
        (mod.EXP6576_TASK_ID, "terminal_class", "null"),
        (mod.EXP6576_TASK_ID, "science_disposition", "null"),
        (mod.EXP6578_TASK_ID, "failed_upstream_task_id", "wrong-task"),
        (mod.EXP6577_TASK_ID, "terminal_class", "success"),
        (mod.EXP6577_TASK_ID, "stored_verdict_class", "positive"),
        (mod.EXP6577_TASK_ID, "stored_first_failed_check", None),
    )
    for task_id, key, value in terminal_mutations:
        mutated = deepcopy(terminal_rows)
        next(row for row in mutated if row["task_id"] == task_id)[key] = value
        assert not mod.terminal_rows_valid(mutated)


def test_req_report_6579_contract_and_gate_guards_cover_each_budget(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6579-DECOMPOSITION/GATES: each frozen constraint is enforced."""

    contract = report["decomposition_contract"]
    contract_mutations = (
        lambda value: value.update(model_task_rows=value["model_task_rows"][:2]),
        lambda value: value.update(one_family_per_model_task=False),
        lambda value: value.update(all_family_aggregation_inside_model_task=True),
        lambda value: value["model_task_rows"].__setitem__(0, "not-a-row"),
        lambda value: value["model_task_rows"][0].update(roadmap_task_exists=False),
        lambda value: value["model_task_rows"][0].update(fresh_process_per_task=False),
        lambda value: value["model_task_rows"][0].update(max_source_units=2),
        lambda value: value["model_task_rows"][0].update(verified_unload_required=False),
        lambda value: value["model_task_rows"][0].update(terminal_artifact_count=2),
        lambda value: value["model_task_rows"][0].update(cross_family_aggregate_allowed=True),
        lambda value: value["model_task_rows"][0].update(contingency_budget_s=1),
        lambda value: value["model_task_rows"][0].update(row_hash="sha256:stale"),
    )
    for mutate in contract_mutations:
        mutated = deepcopy(contract)
        mutate(mutated)
        assert not mod.decomposition_contract_valid(mutated)

    gates = report["current_roadmap_gate_contract_rows"]
    gate_mutations = (
        ("passed", False),
        ("owner_task_exists", False),
        ("all_named_consumers_exist", False),
        ("roadmap_milestone", "wrong"),
        ("row_hash", "sha256:stale"),
    )
    for key, value in gate_mutations:
        mutated = deepcopy(gates)
        mutated[0][key] = value
        assert not mod.gate_contract_rows_valid(mutated)


def test_req_report_6579_source_replay_guards_reject_bad_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6579-GATES/AUDIT: malformed checked-in sources cannot pass."""

    log_text = (REPO / "ops/conductor-log.md").read_text(encoding="utf-8")
    fake_timeout = (
        f"| 2026-08-24 00:00 UTC | "
        f"{mod.V571_TASKS[mod.EXP6575_TASK_ID]['title_prefix']} | FAIL | "
        "Hard wall-clock cap after 999s |\n"
    )
    rows = mod.parse_exp6575_timeout_attempts(fake_timeout + log_text, "sha256:test")
    assert [row["elapsed_s"] for row in rows] == [4801, 4803, 4804]

    (tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).write_text(
        "milestones: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lacks milestone"):
        mod._v571_ledger_tasks(tmp_path)
    with pytest.raises(ValueError, match="missing conductor event"):
        mod._first_event([], "missing", "OK")

    valid_ledger = {
        task_id: {"id": task_id, "result": expected["ledger_result"]}
        for task_id, expected in mod.V571_TASKS.items()
    }
    bad_ledger = deepcopy(valid_ledger)
    bad_ledger[mod.EXP6575_TASK_ID]["result"] = "wrong"
    monkeypatch.setattr(mod, "_v571_ledger_tasks", lambda _root: bad_ledger)
    with pytest.raises(ValueError, match="V571 ledger mismatch"):
        mod.build_v571_terminal_rows(tmp_path, "", "sha256:test", [])

    events = [
        {
            "title": mod.V571_TASKS[task_id]["title_prefix"],
            "result": result,
            "timestamp": "2026-08-24 00:00 UTC",
            "line_number": index,
            "source_line_sha256": f"sha256:{index}",
        }
        for index, (task_id, result) in enumerate(
            (
                (mod.EXP6576_TASK_ID, "GATE_BLOCK"),
                (mod.EXP6578_TASK_ID, "GATE_BLOCK"),
                (mod.EXP6577_TASK_ID, "OK"),
            ),
            start=1,
        )
    ]
    monkeypatch.setattr(mod, "_v571_ledger_tasks", lambda _root: valid_ledger)
    monkeypatch.setattr(mod, "_log_events", lambda _text: events)
    audit_path = tmp_path / mod.V571_TASKS[mod.EXP6577_TASK_ID]["artifact"]
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text('{"gate_check_summary": {}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="gate_check_summary is malformed"):
        mod.build_v571_terminal_rows(tmp_path, "", "sha256:test", [])
    audit_path.write_text(
        json.dumps(
            {
                "verdict_class": "blocked",
                "claim_stream_audit_ready_score": 0.0,
                "gate_check_summary": {
                    "checks": [],
                    "first_failure": {"observed": "wrong", "field": "experiment_6576"},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blocked diagnosis does not replay"):
        mod.build_v571_terminal_rows(tmp_path, "", "sha256:test", [])


def test_req_report_6579_environment_and_atomic_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, Any]
) -> None:
    """REQ-REPORT-6579-PRECONDITIONS/ATOMIC: fallback and cleanup paths execute."""

    next_path = tmp_path / mod.NEXT_ROADMAP_RELATIVE_PATH
    next_path.write_text("milestone: 2026.08.572\ntasks: []\n", encoding="utf-8")
    resolved, payload, source = mod._resolve_v572_roadmap(tmp_path)
    assert resolved == next_path
    assert payload["milestone"] == "2026.08.572"
    assert source == "pre_staged_next_roadmap"
    next_path.unlink()
    (tmp_path / mod.ACTIVE_ROADMAP_RELATIVE_PATH).write_text(
        "milestone: wrong\ntasks: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="active roadmap is not V572"):
        mod._resolve_v572_roadmap(tmp_path)

    monkeypatch.setattr(mod.Path, "is_file", lambda _path: False)
    monkeypatch.setattr(mod.platform, "processor", lambda: "")
    assert mod._cpu_model() == "unknown"
    monkeypatch.setattr(
        mod.metadata,
        "version",
        lambda _distribution: (_ for _ in ()).throw(mod.metadata.PackageNotFoundError()),
    )
    assert {row["version"] for row in mod._tool_versions()} == {"unknown"}

    monkeypatch.undo()
    output = tmp_path / "replace-failure.json"
    monkeypatch.setattr(
        mod.os,
        "replace",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        mod.atomic_write_report(output, report)
    assert not list(tmp_path.glob(".*.tmp"))

    missing_output = tmp_path / "missing.json"
    assert mod.main(["--validate", "--output", str(missing_output)]) == 1

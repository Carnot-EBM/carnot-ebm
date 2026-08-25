"""Focused checks for the V572 receipt replay and V573 execution contract.

Spec refs: REQ-REPORT-6585, SCENARIO-REPORT-6585-NO-ARTIFACT,
SCENARIO-REPORT-6585-BLOCKED-REPLAY, SCENARIO-REPORT-6585-RUNTIME-REPLAY,
SCENARIO-REPORT-6585-GATE-CLOSURE, SCENARIO-REPORT-6585-BOUNDED-MODELS,
SCENARIO-REPORT-6585-ATTACKS, SCENARIO-REPORT-6585-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python/carnot/experiment_6585_v573_terminal_recovery_and_execution_contract.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("exp6585_under_test", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
exp = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = exp
MODULE_SPEC.loader.exec_module(exp)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Use the checked-in receipts as immutable input evidence."""

    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def replay(repo_root: Path) -> dict[str, object]:
    """Build the common replay once so tests inspect the same evidence."""

    log_path = repo_root / exp.CONDUCTOR_LOG_RELATIVE_PATH
    log_text = log_path.read_text(encoding="utf-8")
    log_hash = exp.sha256_file(log_path)
    attempts = exp.parse_exp6584_hard_limit_attempts(log_text, log_hash)
    terminals = exp.build_v572_terminal_rows(repo_root, attempts, log_hash)
    budgets = exp.build_execution_budget_contract()
    gates = exp.build_gate_contract_rows(repo_root)
    return {
        "log_hash": log_hash,
        "attempts": attempts,
        "terminals": terminals,
        "budgets": budgets,
        "gates": gates,
    }


def test_req_report_6585_no_artifact_attempts_stay_distinct(
    repo_root: Path, replay: dict[str, object]
) -> None:
    """SCENARIO-REPORT-6585-NO-ARTIFACT keeps all three source rows."""

    attempts = replay["attempts"]
    assert isinstance(attempts, list)
    assert [row["elapsed_s"] for row in attempts] == [4801, 4800, 4801]
    assert [row["attempt_index"] for row in attempts] == [1, 2, 3]
    assert len({row["log_source_line_sha256"] for row in attempts}) == 3
    assert all(row["artifact_exists_after_attempt"] is False for row in attempts)
    assert not (repo_root / exp.V572_TASKS[exp.EXP6584_TASK_ID]["artifact"]).exists()
    assert exp.hard_limit_attempt_rows_valid(attempts, str(replay["log_hash"]))


def test_req_report_6585_attempt_mutations_fail_closed(replay: dict[str, object]) -> None:
    """REQ-REPORT-6585-HARD-LIMITS rejects collapse, stale hash, and invention."""

    attempts = deepcopy(replay["attempts"])
    assert isinstance(attempts, list)
    assert not exp.hard_limit_attempt_rows_valid(attempts[:2], str(replay["log_hash"]))
    attempts[1]["log_source_sha256"] = "sha256:stale"
    assert not exp.hard_limit_attempt_rows_valid(attempts, str(replay["log_hash"]))
    attempts = deepcopy(replay["attempts"])
    attempts[0]["artifact_exists_after_attempt"] = True
    attempts[0]["row_hash"] = exp.row_hash(attempts[0])
    assert not exp.hard_limit_attempt_rows_valid(attempts, str(replay["log_hash"]))


def test_req_report_6585_terminal_rows_preserve_block_and_runtime(
    repo_root: Path, replay: dict[str, object]
) -> None:
    """SCENARIO-REPORT-6585-BLOCKED-REPLAY and RUNTIME-REPLAY stay honest."""

    rows = replay["terminals"]
    assert isinstance(rows, list)
    assert exp.v572_terminal_rows_valid(rows, repo_root)
    by_id = {row["task_id"]: row for row in rows}
    blocked = by_id[exp.EXP6581_TASK_ID]
    assert blocked["terminal_class"] == "blocked_precondition_before_model_load"
    assert blocked["failed_receipt_check"]["field"] == (
        "preconditions_checked.checks.verification_commands"
    )
    assert blocked["failed_receipt_check"]["observed_value"] is False
    assert blocked["model_process_started"] is False
    for task_id in (exp.EXP6582_TASK_ID, exp.EXP6583_TASK_ID):
        row = by_id[task_id]
        assert row["terminal_class"] == "complete_runtime_shard"
        assert row["terminal_row_count"] == 4
        assert row["checkpoint_count"] == 4
        assert row["science_disposition"] == "runtime_evidence_no_quality_verdict"
    missing = by_id[exp.EXP6584_TASK_ID]
    assert missing["terminal_class"] == "hard_limit_no_artifact"
    assert missing["science_disposition"] == "not_run_to_terminal_artifact"


def test_req_report_6585_terminal_hash_and_count_attacks_fail_closed(
    repo_root: Path, replay: dict[str, object]
) -> None:
    """REQ-REPORT-6585-TERMINALS rejects incomplete and stale-source replay."""

    rows = deepcopy(replay["terminals"])
    assert isinstance(rows, list)
    assert not exp.v572_terminal_rows_valid(rows[:-1], repo_root)
    rows[0]["artifact_sha256"] = "sha256:stale"
    rows[0]["row_hash"] = exp.row_hash(rows[0])
    assert not exp.v572_terminal_rows_valid(rows, repo_root)


def test_req_report_6585_execution_budgets_are_one_model_and_bounded(
    replay: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6585-BOUNDED-MODELS freezes both fresh processes."""

    budgets = replay["budgets"]
    assert isinstance(budgets, list)
    assert exp.execution_budget_contract_valid(budgets)
    assert {row["task_id"] for row in budgets} == {
        "exp6588-qwen36-constraint-first-stream",
        "exp6589-gemma4-31b-constraint-first-stream",
    }
    assert all(row["raw_checkpoints_per_completed_unit_min"] == 1 for row in budgets)
    assert all(row["max_model_processes"] == 1 for row in budgets)
    assert all(row["atomic_terminal_output"] is True for row in budgets)


def test_req_report_6585_budget_monolith_and_checkpoint_attacks_fail_closed(
    replay: dict[str, object],
) -> None:
    """REQ-REPORT-6585-BUDGETS rejects two models and lost checkpoints."""

    budgets = deepcopy(replay["budgets"])
    assert isinstance(budgets, list)
    budgets[0]["model_families"].append("unsloth/gemma-4-31B-it-GGUF")
    assert not exp.execution_budget_contract_valid(budgets)
    budgets = deepcopy(replay["budgets"])
    budgets[1]["raw_checkpoints_per_completed_unit_min"] = 0
    assert not exp.execution_budget_contract_valid(budgets)


def test_req_report_6585_gate_fields_close_against_active_roadmap(
    replay: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6585-GATE-CLOSURE matches owner and field spelling."""

    rows = replay["gates"]
    assert isinstance(rows, list)
    assert len(rows) == 4
    assert exp.gate_contract_rows_valid(rows)
    assert {row["consumer_task_id"] for row in rows} == {
        "exp6588-qwen36-constraint-first-stream",
        "exp6589-gemma4-31b-constraint-first-stream",
    }
    assert all(row["upstream_task_exists_in_active_roadmap"] for row in rows)
    assert all(row["field_declared_with_identical_spelling"] for row in rows)


def test_req_report_6585_gate_drift_attacks_fail_closed(replay: dict[str, object]) -> None:
    """REQ-REPORT-6585-GATES rejects outside owners and misspelled fields."""

    rows = deepcopy(replay["gates"])
    assert isinstance(rows, list)
    rows[0]["upstream_task_id"] = "exp9999-outside-roadmap"
    rows[0]["upstream_task_exists_in_active_roadmap"] = False
    rows[0]["row_hash"] = exp.row_hash(rows[0])
    assert not exp.gate_contract_rows_valid(rows)
    rows = deepcopy(replay["gates"])
    rows[1]["artifact_field"] = "v573_execution_contract_ready_scor"
    rows[1]["field_declared_with_identical_spelling"] = False
    rows[1]["row_hash"] = exp.row_hash(rows[1])
    assert not exp.gate_contract_rows_valid(rows)


def test_req_report_6585_attacks_and_ready_reducer(
    repo_root: Path, replay: dict[str, object]
) -> None:
    """SCENARIO-REPORT-6585-ATTACKS proves all eight candidates stay closed."""

    attacks = exp.build_attack_rows(
        repo_root=repo_root,
        terminal_rows=replay["terminals"],
        attempt_rows=replay["attempts"],
        log_hash=str(replay["log_hash"]),
        budget_rows=replay["budgets"],
        gate_rows=replay["gates"],
    )
    assert [row["attack"] for row in attacks] == list(exp.REQUIRED_ATTACKS)
    assert all(row["passed"] and row["observed_ready_score"] == 0.0 for row in attacks)
    protected = {"unchanged": True}
    assert (
        exp.execution_contract_ready_score(
            repo_root,
            replay["terminals"],
            replay["attempts"],
            str(replay["log_hash"]),
            replay["budgets"],
            replay["gates"],
            attacks,
            protected,
        )
        == 1.0
    )
    assert (
        exp.execution_contract_ready_score(
            repo_root,
            replay["terminals"][:-1],
            replay["attempts"],
            str(replay["log_hash"]),
            replay["budgets"],
            replay["gates"],
            attacks,
            protected,
        )
        == 0.0
    )


def test_req_report_6585_preconditions_name_resources_and_no_llm(repo_root: Path) -> None:
    """REQ-REPORT-6585-PRECONDITIONS records the complete replay substrate."""

    protected = exp.hash_protected_files(repo_root)
    preconditions = exp.collect_preconditions(repo_root, protected)
    assert preconditions["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert preconditions["llm_loaded"] is False
    assert len(preconditions["expected_v572_tasks_and_artifacts"]) == 6
    assert preconditions["expected_v572_tasks_and_artifacts"][-1]["exists"] is False
    assert preconditions["conductor_log_sha256"].startswith("sha256:")
    assert preconditions["failure_ledger_sha256"].startswith("sha256:")
    assert preconditions["exclusion_manifest_sha256"].startswith("sha256:")
    assert preconditions["cpu"]["count"] >= 1
    assert preconditions["ram"]["total_kib"] > 0
    assert preconditions["disk"]["total_bytes"] > 0


def test_req_report_6585_atomic_report_checksum_and_validation(
    tmp_path: Path, repo_root: Path, replay: dict[str, object]
) -> None:
    """SCENARIO-REPORT-6585-ATOMIC writes one valid null-class artifact."""

    protected_hashes = exp.hash_protected_files(repo_root)
    protected = exp.protected_files_receipt(protected_hashes, protected_hashes)
    report = exp.build_report(
        repo_root=repo_root,
        run_date="20260825",
        terminal_rows=replay["terminals"],
        attempt_rows=replay["attempts"],
        log_hash=str(replay["log_hash"]),
        budget_rows=replay["budgets"],
        gate_rows=replay["gates"],
        protected=protected,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 1.0}],
        duration_s=2.0,
    )
    assert report["status"] == "complete_v573_execution_contract_ready"
    assert report["honest_verdict"].startswith("complete:")
    assert report["verdict_class"] == "null"
    assert report["v573_execution_contract_ready_score"] == 1.0
    assert report["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True
    assert exp.validate_report(report, repo_root) == []
    output = tmp_path / "artifact.json"
    receipt = exp.atomic_write_report(output, report, repo_root)
    assert receipt["atomic_replace"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert receipt["sha256"] == exp.sha256_file(output)


def test_req_report_6585_validation_rejects_mutated_contract(
    repo_root: Path, replay: dict[str, object]
) -> None:
    """REQ-REPORT-6585-REDUCER rejects class, checksum, and field mutation."""

    protected_hashes = exp.hash_protected_files(repo_root)
    report = exp.build_report(
        repo_root=repo_root,
        run_date="20260825",
        terminal_rows=replay["terminals"],
        attempt_rows=replay["attempts"],
        log_hash=str(replay["log_hash"]),
        budget_rows=replay["budgets"],
        gate_rows=replay["gates"],
        protected=exp.protected_files_receipt(protected_hashes, protected_hashes),
        tests_run=[],
        duration_s=1.0,
    )
    broken = deepcopy(report)
    broken["verdict_class"] = "positive"
    broken["v572_terminal_rows"] = broken["v572_terminal_rows"][:-1]
    broken.pop("status")
    errors = exp.validate_report(broken, repo_root)
    assert "missing_required_field:status" in errors
    assert "ready_contract_verdict_class_must_be_null" in errors
    assert "ready_score_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_report_6585_defensive_inputs_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    replay: dict[str, object],
) -> None:
    """REQ-REPORT-6585 rejects malformed sources and all validator edge cases."""

    log_text = (repo_root / exp.CONDUCTOR_LOG_RELATIVE_PATH).read_text(encoding="utf-8")
    malformed_event = (
        "| 2026-08-24 22:00 UTC | Independent three-family source receipt audit "
        "| FAIL | missing hard-cap detail |\n"
    )
    attempts = exp.parse_exp6584_hard_limit_attempts(
        malformed_event + log_text, str(replay["log_hash"])
    )
    assert len(attempts) == 3
    with pytest.raises(ValueError, match="expected three"):
        exp.parse_exp6584_hard_limit_attempts(malformed_event, str(replay["log_hash"]))

    complete = tmp_path / exp.RESEARCH_COMPLETE_RELATIVE_PATH
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.write_text("milestones:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks milestone"):
        exp._v572_ledger(tmp_path)
    complete.write_text(
        "milestones:\n"
        "- id: 2026.08.572\n"
        "  tasks:\n"
        "  - id: exp6585-test\n"
        "    result: OK\n"
        "- id: 2026.08.573\n"
        "  tasks: []\n",
        encoding="utf-8",
    )
    assert list(exp._v572_ledger(tmp_path)) == ["exp6585-test"]

    non_object = tmp_path / "list.json"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp._load_json(non_object)

    ledger = {
        task_id: {"id": task_id, "result": values["ledger_result"]}
        for task_id, values in exp.V572_TASKS.items()
    }
    with monkeypatch.context() as patcher:
        bad_ledger = deepcopy(ledger)
        bad_ledger[exp.EXP6579_TASK_ID]["result"] = "WRONG"
        patcher.setattr(exp, "_v572_ledger", lambda _root: bad_ledger)
        with pytest.raises(ValueError, match="ledger mismatch"):
            exp.build_v572_terminal_rows(tmp_path, replay["attempts"], str(replay["log_hash"]))
    with monkeypatch.context() as patcher:
        patcher.setattr(exp, "_v572_ledger", lambda _root: ledger)
        patcher.setattr(exp, "_load_json", lambda _path: {})
        fabricated = tmp_path / exp.V572_TASKS[exp.EXP6584_TASK_ID]["artifact"]
        fabricated.parent.mkdir(parents=True, exist_ok=True)
        fabricated.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must remain absent"):
            exp.build_v572_terminal_rows(tmp_path, replay["attempts"], str(replay["log_hash"]))

    terminal_rows = deepcopy(replay["terminals"])
    terminal_rows[0]["row_hash"] = "sha256:stale"
    assert not exp.v572_terminal_rows_valid(terminal_rows, repo_root)
    terminal_rows = deepcopy(replay["terminals"])
    blocked = next(row for row in terminal_rows if row["task_id"] == exp.EXP6581_TASK_ID)
    blocked["model_process_started"] = True
    blocked["row_hash"] = exp.row_hash(blocked)
    assert not exp.v572_terminal_rows_valid(terminal_rows, repo_root)
    terminal_rows = deepcopy(replay["terminals"])
    runtime = next(row for row in terminal_rows if row["task_id"] == exp.EXP6582_TASK_ID)
    runtime["checkpoint_count"] = 3
    runtime["row_hash"] = exp.row_hash(runtime)
    assert not exp.v572_terminal_rows_valid(terminal_rows, repo_root)

    assert not exp.execution_budget_contract_valid(replay["budgets"][:1])
    budgets = deepcopy(replay["budgets"])
    budgets[0]["task_id"] = "exp9999-wrong"
    budgets[0]["row_hash"] = exp.row_hash(budgets[0])
    assert not exp.execution_budget_contract_valid(budgets)
    assert exp._required_artifact_fields("no required field marker") == set()
    assert not exp.gate_contract_rows_valid(replay["gates"][:3])

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "is_file", lambda _path: False)
        patcher.setattr(exp.platform, "processor", lambda: "fallback-cpu")
        assert exp._cpu_model() == "fallback-cpu"
    with monkeypatch.context() as patcher:
        patcher.setattr(
            exp.metadata,
            "version",
            lambda _package: (_ for _ in ()).throw(exp.metadata.PackageNotFoundError),
        )
        assert set(exp._tool_versions().values()) == {exp.platform.python_version(), "missing"}

    protected_hashes = exp.hash_protected_files(repo_root)
    report = exp.build_report(
        repo_root=repo_root,
        run_date="20260825",
        terminal_rows=replay["terminals"],
        attempt_rows=replay["attempts"],
        log_hash=str(replay["log_hash"]),
        budget_rows=replay["budgets"],
        gate_rows=replay["gates"],
        protected=exp.protected_files_receipt(protected_hashes, protected_hashes),
        tests_run=[],
        duration_s=1.0,
    )
    broken = deepcopy(report)
    broken["inference_substrate"] = "wrong"
    broken["verifier_is_oracle"] = False
    broken["honest_verdict"] = "not terminal"
    errors = exp.validate_report(broken, repo_root)
    assert "inference_substrate_mismatch" in errors
    assert "verifier_is_oracle_mismatch" in errors
    assert "terminal_success_prefix_missing" in errors
    with pytest.raises(ValueError, match="inference_substrate_mismatch"):
        exp.atomic_write_report(tmp_path / "invalid.json", broken, repo_root)

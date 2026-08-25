"""Focused tests for complete isolated pytest receipts.

Spec refs: REQ-REPORT-6589, REQ-REPORT-6589-RECEIPT,
REQ-REPORT-6589-FIXTURES, REQ-REPORT-6589-CHECKOUT,
REQ-REPORT-6589-MUTATION, REQ-REPORT-6589-SUITE,
REQ-REPORT-6589-ROWS, REQ-REPORT-6589-TIMEOUT,
REQ-REPORT-6589-VERDICT, REQ-REPORT-6589-ATTACKS,
REQ-REPORT-6589-ATOMIC, SCENARIO-REPORT-6589-RECEIPT,
SCENARIO-REPORT-6589-DISPOSABLE, SCENARIO-REPORT-6589-RED,
SCENARIO-REPORT-6589-TIMEOUT, SCENARIO-REPORT-6589-MUTATION,
SCENARIO-REPORT-6589-ATTACKS, SCENARIO-REPORT-6589-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_6589_isolated_pytest_receipt_remediation as exp


REPO = Path(__file__).resolve().parents[2]


def _complete_command_receipt(checkout: Path, *, exit_code: int = 1) -> dict[str, object]:
    rows = [
        {
            "nodeid": "tests/python/test_sample.py::test_bad",
            "outcome": "failed",
            "phase": "call",
            "longrepr": "assert False",
        },
        {
            "nodeid": "tests/python/test_sample.py::test_skip",
            "outcome": "skipped",
            "phase": "setup",
            "longrepr": "skip",
        },
    ]
    return {
        "command": exp.SUITE_COMMAND_TEXT,
        "argv": list(exp.SUITE_COMMAND),
        "cwd": str(checkout.resolve()),
        "environment": {"PYTHONPATH": str(checkout / "python")},
        "environment_sha256": "sha256:environment",
        "exit_code": exit_code,
        "duration_s": 2.5,
        "stdout": "1 failed, 1 passed, 1 skipped\n",
        "stderr": "",
        "timed_out": False,
        "timeout_s": 3600.0,
        "process_cleanup": {
            "clean": True,
            "signals": [],
            "surviving_owned_pids": [],
            "unrelated_process_signal_count": 0,
        },
        "pytest_receipt_state": "complete",
        "pytest_exit_status": exit_code,
        "collected_count": 3,
        "nodeids_sha256": "sha256:nodeids",
        "terminal_outcome_counts": {
            "passed": 1,
            "failed": 1,
            "errored": 0,
            "skipped": 1,
        },
        "rows": rows,
        "collection_rows": [],
        "family_summaries": [
            {
                "family": "sample",
                "passed": 1,
                "failed": 1,
                "errored": 0,
                "skipped": 1,
            }
        ],
        "receipt_sha256": "sha256:receipt",
    }


def _base_evidence(tmp_path: Path) -> dict[str, object]:
    checkout = tmp_path / "carnot-exp6589" / "checkout"
    checkout.mkdir(parents=True)
    suite = _complete_command_receipt(checkout)
    checkout_receipt = {
        "active_root": str(tmp_path / "active"),
        "checkout_root": str(checkout.resolve()),
        "validated_temporary_root": str(checkout.parent.resolve()),
        "revision": "a" * 40,
        "dirty_content_patch_hash": "sha256:patch",
        "patch_rows": [],
        "dirty_paths": [],
        "overlay_complete": True,
        "mutation_scan_complete": True,
        "changed_tracked_paths": [],
        "cleanup": {"attempted": True, "removed": True, "exit_code": 0},
    }
    active = {
        "unchanged": True,
        "tracked_hashes_before_sha256": "sha256:active",
        "tracked_hashes_after_sha256": "sha256:active",
        "dirty_status_before_sha256": "sha256:dirty",
        "dirty_status_after_sha256": "sha256:dirty",
        "preexisting_dirty_status_preserved": True,
    }
    protected = {
        "unchanged": True,
        "before": {path: "sha256:same" for path in exp.PROTECTED_PATHS},
        "after": {path: "sha256:same" for path in exp.PROTECTED_PATHS},
    }
    return {
        "suite": suite,
        "checkout": checkout_receipt,
        "active": active,
        "protected": protected,
    }


def test_req_report_6589_spec_and_constants_are_exact() -> None:
    """REQ-REPORT-6589 binds the mandated command and artifact substrate."""

    spec = (REPO / "openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-6589-RECEIPT" in spec
    assert "SCENARIO-REPORT-6589-ATOMIC" in spec
    assert exp.SUITE_COMMAND_TEXT == (
        ".venv/bin/python -m pytest tests/python --no-cov -o addopts= -n 0"
    )
    assert exp.INFERENCE_SUBSTRATE == "isolated_pytest_receipt_repair_no_llm"
    assert exp.RESULT_RELATIVE_PATH == Path(
        "results/experiment_6589_isolated_pytest_receipt_remediation.json"
    )


def test_req_report_6589_exp6586_failure_replay_is_source_bound() -> None:
    """REQ-REPORT-6589-REPLAY retains the missing receipt and adversarial flag."""

    replay = exp.exp6586_failure_replay(REPO)
    assert replay["artifact_sha256"] == (
        "sha256:74a2f44f868943d64b4c491222f66783c8759314c6d2c1f8d0450382c2b5ef29"
    )
    assert replay["honest_verdict"] == "blocked_isolated_environment: pytest_receipt"
    assert replay["failed_receipt_fields"] == [
        "collection_receipt",
        "disposable_checkout_receipt",
        "rows",
        "suite_command_receipt",
    ]
    assert replay["row_count"] == 0
    assert replay["flagged_adversarial"] is True
    assert replay["adversarial_disposition"][0]["kind"] == "NONTERMINAL_DECLARED_ARTIFACT"


def test_req_report_6589_each_receipt_field_has_positive_and_negative_fixture(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6589-RECEIPT exercises each required field both ways."""

    receipt = _complete_command_receipt(tmp_path)
    rows = exp.focused_receipt_fixture_rows(receipt, checkout_root=tmp_path)
    by_field: dict[str, set[str]] = {}
    for row in rows:
        by_field.setdefault(str(row["field"]), set()).add(str(row["polarity"]))
    assert set(by_field) == set(exp.REQUIRED_COMMAND_RECEIPT_FIELDS)
    assert all(polarities == {"positive", "negative"} for polarities in by_field.values())
    assert all(row["passed"] is True for row in rows)

    assert (
        exp.validate_command_receipt(
            receipt, expected_command=exp.SUITE_COMMAND_TEXT, checkout_root=tmp_path
        )
        == []
    )
    for field in exp.REQUIRED_COMMAND_RECEIPT_FIELDS:
        broken = deepcopy(receipt)
        broken.pop(field)
        errors = exp.validate_command_receipt(
            broken, expected_command=exp.SUITE_COMMAND_TEXT, checkout_root=tmp_path
        )
        assert f"missing_command_receipt_field:{field}" in errors


def test_req_report_6589_missing_sidecar_keeps_raw_streams(tmp_path: Path) -> None:
    """REQ-REPORT-6589-RECEIPT reproduces Exp6586 without discarding command data."""

    raw = {
        "command": "focused pytest",
        "argv": [sys.executable, "-m", "pytest"],
        "cwd": str(tmp_path.resolve()),
        "exit_code": 3,
        "duration_s": 0.25,
        "stdout": "raw standard output",
        "stderr": "plugin import failed",
        "timed_out": False,
        "timeout_s": 5.0,
        "process_cleanup": {
            "clean": True,
            "signals": [],
            "surviving_owned_pids": [],
            "unrelated_process_signal_count": 0,
        },
    }
    merged = exp.merge_pytest_receipt(
        raw,
        tmp_path / "missing-sidecar.json",
        environment={"PYTHONPATH": "fixture"},
    )
    assert merged["pytest_receipt_state"] == "missing"
    assert merged["stdout"] == "raw standard output"
    assert merged["stderr"] == "plugin import failed"
    assert "pytest_sidecar_missing" in merged["receipt_errors"]
    assert merged["receipt_sha256"].startswith("sha256:")


def test_req_report_6589_plugin_writes_counts_rows_and_atomic_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6589-ROWS preserves pass, fail, error, skip, and collection."""

    sidecar = tmp_path / "pytest-sidecar.json"
    monkeypatch.setenv(exp.PLUGIN_RECEIPT_ENV, str(sidecar))
    exp.pytest_sessionstart(SimpleNamespace())
    exp.pytest_collection_finish(
        SimpleNamespace(
            items=[SimpleNamespace(nodeid=f"sample::test_{index}") for index in range(4)]
        )
    )
    exp.pytest_collectreport(
        SimpleNamespace(failed=True, nodeid="bad_collection.py", longrepr="import error")
    )
    for outcome, phase in (("passed", "call"), ("failed", "call"), ("skipped", "setup")):
        exp.pytest_runtest_logreport(
            SimpleNamespace(
                passed=outcome == "passed",
                failed=outcome == "failed",
                skipped=outcome == "skipped",
                nodeid=f"sample::test_{outcome}",
                when=phase,
                longrepr=f"{outcome} detail",
                wasxfail=None,
            )
        )
    exp.pytest_sessionfinish(SimpleNamespace(), 1)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["collected_count"] == 4
    assert payload["terminal_outcome_counts"] == {
        "errored": 1,
        "failed": 1,
        "passed": 1,
        "skipped": 1,
    }
    assert {row["outcome"] for row in payload["rows"]} == {
        "errored",
        "failed",
        "skipped",
    }
    assert not list(tmp_path.glob(".pytest-sidecar.json.*.tmp"))


def test_req_report_6589_red_green_and_timeout_reduce_honestly(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6589-RED separates suite truth from receipt readiness."""

    evidence = _base_evidence(tmp_path)
    red = exp.reduce_suite_truth(
        suite=evidence["suite"],
        checkout=evidence["checkout"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        focused_contract_passed=True,
    )
    assert red["state"] == "measured_red"
    assert red["complete"] is True
    assert red["ready_score"] == 1.0
    assert red["verdict_class"] == "null"

    green_suite = deepcopy(evidence["suite"])
    green_suite.update(
        {
            "exit_code": 0,
            "pytest_exit_status": 0,
            "rows": [],
            "terminal_outcome_counts": {
                "passed": 3,
                "failed": 0,
                "errored": 0,
                "skipped": 0,
            },
        }
    )
    green = exp.reduce_suite_truth(
        suite=green_suite,
        checkout=evidence["checkout"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        focused_contract_passed=True,
    )
    assert green["state"] == "measured_green"

    unjustified = deepcopy(green_suite)
    unjustified["terminal_outcome_counts"]["passed"] = 0
    refused = exp.reduce_suite_truth(
        suite=unjustified,
        checkout=evidence["checkout"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        focused_contract_passed=True,
    )
    assert refused["state"] == "receipt_block"

    timed = deepcopy(green_suite)
    timed["timed_out"] = True
    timed["exit_code"] = -15
    timed["rows"] = [exp.timeout_row(10.0)]
    timeout = exp.reduce_suite_truth(
        suite=timed,
        checkout=evidence["checkout"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        focused_contract_passed=True,
    )
    assert timeout["state"] == "timeout"
    assert timeout["complete"] is False
    assert timeout["ready_score"] == 0.0


def test_req_report_6589_timeout_cleans_only_owned_processes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6589-TIMEOUT retains streams and owned cleanup."""

    buffered_env = os.environ.copy()
    buffered_env.pop("PYTHONUNBUFFERED", None)
    receipt = exp.run_owned_command(
        [
            sys.executable,
            "-c",
            "import sys,time; print('before timeout'); print('err', file=sys.stderr); time.sleep(30)",
        ],
        cwd=tmp_path,
        env=buffered_env,
        timeout_s=0.05,
        display_command="timeout fixture",
        cleanup_grace_s=0.05,
    )
    assert receipt["timed_out"] is True
    assert "before timeout" in receipt["stdout"]
    assert "err" in receipt["stderr"]
    assert receipt["process_cleanup"]["clean"] is True
    assert receipt["process_cleanup"]["surviving_owned_pids"] == []
    assert receipt["process_cleanup"]["unrelated_process_signal_count"] == 0
    assert receipt["process_cleanup"]["signals"][0]["signal"] == "SIGTERM"


def test_req_report_6589_overlay_and_mutation_rows_keep_hashes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6589-MUTATION binds dirty content and tracked writes."""

    active = tmp_path / "active"
    active.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=active, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=active, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=active, check=True)
    (active / "tracked.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=active, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=active, check=True)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", str(active), str(checkout)], check=True)

    (active / "tracked.txt").write_text("active change\n", encoding="utf-8")
    (active / "new.txt").write_text("new\n", encoding="utf-8")
    dirty = exp.active_dirty_paths(active)
    patch_rows = exp.apply_content_overlay(active, checkout, dirty)
    assert exp.overlay_is_complete(dirty, patch_rows, active, checkout)
    assert exp.sha256_json(patch_rows).startswith("sha256:")

    before = exp.snapshot_tracked_files(checkout)
    (checkout / "tracked.txt").write_text("suite write\n", encoding="utf-8")
    after = exp.snapshot_tracked_files(checkout)
    rows = exp.tracked_mutation_rows(before, after, observed_paths=["tracked.txt"])
    assert rows == [
        {
            "path": "tracked.txt",
            "before_hash": exp.hash_path(active / "tracked.txt"),
            "after_hash": exp.hash_path(checkout / "tracked.txt"),
            "before_exists": True,
            "after_exists": True,
            "observed_write_attempt": True,
            "content_changed": True,
        }
    ]


def test_req_report_6589_report_attacks_and_atomic_output(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6589-ATTACKS and ATOMIC reject false terminal claims."""

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        exp6586_replay={"artifact_sha256": "sha256:source"},
        focused_rows=exp.focused_receipt_fixture_rows(
            evidence["suite"], checkout_root=Path(evidence["suite"]["cwd"])
        ),
        preconditions={"focused_contract_passed": True},
        checkout=evidence["checkout"],
        suite=evidence["suite"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[{"command": "focused", "exit_code": 0}],
        duration_s=3.0,
    )
    assert report["status"] == "measured_red"
    assert report["honest_verdict"].startswith("complete:")
    assert report["verdict_class"] == "null"
    assert report["pytest_receipt_remediation_ready_score"] == 1.0
    assert [row["attack"] for row in report["attack_rows"]] == list(exp.REQUIRED_ATTACKS)
    assert exp.validate_report(report) == []

    for field, value in (
        ("rows", []),
        ("suite_command_receipt", {**evidence["suite"], "stdout": None}),
        ("mutation_rows", [{"path": "hidden.txt"}]),
        ("active_worktree_unchanged", {**evidence["active"], "unchanged": False}),
        ("attack_rows", []),
    ):
        broken = deepcopy(report)
        broken[field] = value
        broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
        assert exp.validate_report(broken), field

    target = tmp_path / "terminal.json"
    write = exp.atomic_write_report(target, report)
    assert write["atomic_replace"] is True
    assert write["file_fsync"] is True
    assert write["directory_fsync"] is True
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert not list(tmp_path.glob(".terminal.json.*.tmp"))


def test_req_report_6589_focused_failure_blocks_suite_launch(tmp_path: Path) -> None:
    """REQ-REPORT-6589-FIXTURES writes a terminal block before suite launch."""

    blocked = exp.blocked_report(
        run_date="20260825",
        status="focused_fixture_failure",
        failed_check="receipt_field:stderr",
        observed_value=None,
        exp6586_replay={"artifact_sha256": "sha256:source"},
        focused_rows=[{"field": "stderr", "polarity": "positive", "passed": False}],
        preconditions={"focused_contract_passed": False},
        duration_s=0.5,
    )
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["suite_truth_baseline"]["state"] == "not_run"
    assert blocked["suite_command_receipt"] == {}
    assert blocked["pytest_receipt_remediation_ready_score"] == 0.0
    assert blocked["gate_check_summary"][0]["observed_value"] is None
    assert exp.validate_report(blocked) == []


def test_req_report_6589_plugin_no_output_without_owned_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6589-ATOMIC refuses an implicit sidecar destination."""

    monkeypatch.delenv(exp.PLUGIN_RECEIPT_ENV, raising=False)
    exp.pytest_sessionstart(SimpleNamespace())
    exp.pytest_sessionfinish(SimpleNamespace(), 0)


@pytest.mark.parametrize("focused_passed", [True, False])
def test_req_report_6589_orchestrator_gates_one_disposable_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, focused_passed: bool
) -> None:
    """REQ-REPORT-6589-CHECKOUT starts the suite only after focused validation."""

    active = tmp_path / ("active-pass" if focused_passed else "active-block")
    active.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=active, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=active, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=active, check=True)
    (active / "research-roadmap.yaml").write_text("milestone: fixture\n", encoding="utf-8")
    conductor = active / "scripts/research_conductor.py"
    conductor.parent.mkdir()
    conductor.write_text("# protected\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=active, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=active, check=True)

    monkeypatch.setattr(
        exp,
        "exp6586_failure_replay",
        lambda _root: {"artifact_sha256": "sha256:source", "failed_check": "pytest_receipt"},
    )

    def fake_focused(checkout: Path, temporary_root: Path) -> dict[str, object]:
        receipt = _complete_command_receipt(checkout)
        rows = exp.focused_receipt_fixture_rows(receipt, checkout_root=checkout)
        if not focused_passed:
            rows[0]["passed"] = False
        return {
            "passed": focused_passed,
            "failed_check": None if focused_passed else "focused_receipt_fixture:command",
            "observed_value": None if focused_passed else False,
            "rows": rows,
            "tests_run": [
                {"command": "focused receipt fixture", "exit_code": 0 if focused_passed else 1}
            ],
            "temporary_root": str(temporary_root),
        }

    suite_calls: list[Path] = []

    def fake_suite(active_root: Path, checkout: Path, temporary_root: Path) -> dict[str, object]:
        del active_root, temporary_root
        suite_calls.append(checkout)
        return _complete_command_receipt(checkout)

    monkeypatch.setattr(exp, "run_focused_contract", fake_focused)
    monkeypatch.setattr(exp, "run_suite_measurement", fake_suite)
    report = exp.run_experiment(active, "20260825")
    if focused_passed:
        assert report["status"] == "measured_red"
        assert report["pytest_receipt_remediation_ready_score"] == 1.0
        assert len(suite_calls) == 1
        assert report["disposable_checkout_receipt"]["cleanup"]["removed"] is True
    else:
        assert report["status"] == "focused_fixture_failure"
        assert report["suite_command_receipt"] == {}
        assert suite_calls == []
    assert exp.validate_report(report) == []
    assert (active / exp.RESULT_RELATIVE_PATH).is_file()


def test_req_report_6589_main_validates_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6589-ATOMIC exposes a read-only checksum validation command."""

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        exp6586_replay={"artifact_sha256": "sha256:source"},
        focused_rows=exp.focused_receipt_fixture_rows(
            evidence["suite"], checkout_root=Path(evidence["suite"]["cwd"])
        ),
        preconditions={"focused_contract_passed": True},
        checkout=evidence["checkout"],
        suite=evidence["suite"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    target = tmp_path / exp.RESULT_RELATIVE_PATH
    exp.atomic_write_report(target, report)
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    assert exp.main(["--validate"]) == 0
    assert "valid:" in capsys.readouterr().out

    broken = deepcopy(report)
    broken["reproducibility_checksum"] = "sha256:bad"
    target.write_text(json.dumps(broken), encoding="utf-8")
    assert exp.main(["--validate"]) == 1
    assert "reproducibility_checksum_mismatch" in capsys.readouterr().out


def test_req_report_6589_merge_rejects_invalid_sidecars(tmp_path: Path) -> None:
    """REQ-REPORT-6589-RECEIPT distinguishes invalid JSON and invalid schemas."""

    raw = {
        "command": "fixture",
        "argv": ["pytest"],
        "cwd": str(tmp_path),
        "exit_code": 2,
        "duration_s": 0.1,
        "stdout": "out",
        "stderr": "err",
        "timed_out": False,
        "timeout_s": 1.0,
        "process_cleanup": {
            "clean": True,
            "surviving_owned_pids": [],
            "unrelated_process_signal_count": 0,
        },
    }
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    invalid = exp.merge_pytest_receipt(raw, invalid_json, environment={})
    assert invalid["pytest_receipt_state"] == "invalid"
    assert invalid["receipt_errors"] == ["pytest_sidecar_invalid:JSONDecodeError"]

    invalid_schema = tmp_path / "list.json"
    invalid_schema.write_text("[]", encoding="utf-8")
    schema = exp.merge_pytest_receipt(raw, invalid_schema, environment={})
    assert schema["pytest_receipt_state"] == "invalid"
    assert schema["receipt_errors"] == ["pytest_sidecar_schema:not_object"]


def test_req_report_6589_command_validator_names_all_malformed_values(tmp_path: Path) -> None:
    """REQ-REPORT-6589-ATTACKS covers each non-missing receipt refusal."""

    receipt = _complete_command_receipt(tmp_path)
    malformed = deepcopy(receipt)
    malformed.update(
        {
            "command": "wrong",
            "argv": [],
            "cwd": "/active",
            "environment": None,
            "environment_sha256": None,
            "duration_s": -1,
            "stdout": None,
            "stderr": None,
            "timed_out": None,
            "timeout_s": 0,
            "process_cleanup": None,
            "pytest_receipt_state": "missing",
            "collected_count": -1,
            "nodeids_sha256": None,
            "terminal_outcome_counts": None,
            "rows": None,
            "collection_rows": None,
            "family_summaries": None,
            "pytest_exit_status": 0,
            "receipt_sha256": None,
        }
    )
    errors = exp.validate_command_receipt(
        malformed, expected_command=exp.SUITE_COMMAND_TEXT, checkout_root=tmp_path
    )
    assert {
        "argv_missing",
        "cleanup_receipt_missing",
        "collected_count_invalid",
        "collection_rows_invalid",
        "command_mismatch",
        "cwd_not_disposable_checkout",
        "duration_invalid",
        "environment_receipt_invalid",
        "family_summaries_invalid",
        "nodeids_hash_invalid",
        "outcome_counts_invalid",
        "pytest_exit_mismatch",
        "pytest_sidecar_incomplete",
        "receipt_hash_missing",
        "rows_invalid",
        "stderr_missing",
        "stdout_missing",
        "timeout_budget_invalid",
        "timeout_state_missing",
    }.issubset(errors)

    inconsistent = _complete_command_receipt(tmp_path)
    inconsistent["process_cleanup"] = {
        "clean": False,
        "surviving_owned_pids": [99],
        "unrelated_process_signal_count": 1,
    }
    inconsistent["collected_count"] = 99
    inconsistent["rows"] = []
    inconsistent["receipt_sha256"] = "bad"
    errors = exp.validate_command_receipt(
        inconsistent, expected_command=exp.SUITE_COMMAND_TEXT, checkout_root=tmp_path
    )
    assert {
        "exception_rows_incomplete",
        "fabricated_collection_count",
        "owned_process_leak",
        "receipt_hash_missing",
        "unrelated_process_signaled",
        "zero_rows_without_raw_justification",
    }.issubset(errors)


def test_req_report_6589_builds_green_timeout_and_receipt_block_reports(tmp_path: Path) -> None:
    """REQ-REPORT-6589-VERDICT covers all non-isolation terminal reducers."""

    evidence = _base_evidence(tmp_path)
    focused = exp.focused_receipt_fixture_rows(
        evidence["suite"], checkout_root=Path(evidence["suite"]["cwd"])
    )
    green_suite = deepcopy(evidence["suite"])
    green_suite.update(
        {
            "exit_code": 0,
            "pytest_exit_status": 0,
            "rows": [],
            "terminal_outcome_counts": {
                "passed": 3,
                "failed": 0,
                "errored": 0,
                "skipped": 0,
            },
        }
    )
    green = exp.build_report(
        run_date="20260825",
        exp6586_replay={},
        focused_rows=focused,
        preconditions={"focused_contract_passed": True},
        checkout=evidence["checkout"],
        suite=green_suite,
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    assert green["status"] == "measured_green"
    assert exp.validate_report(green) == []

    timed_suite = deepcopy(green_suite)
    timed_suite.update(
        {
            "timed_out": True,
            "exit_code": -15,
            "pytest_receipt_state": "missing",
            "rows": [exp.timeout_row(10.0)],
        }
    )
    timed = exp.build_report(
        run_date="20260825",
        exp6586_replay={},
        focused_rows=focused,
        preconditions={"focused_contract_passed": True},
        checkout=evidence["checkout"],
        suite=timed_suite,
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    assert timed["status"] == "timeout"
    assert timed["honest_verdict"].startswith("timeout:")

    blocked = exp.build_report(
        run_date="20260825",
        exp6586_replay={},
        focused_rows=focused,
        preconditions={"focused_contract_passed": False},
        checkout=evidence["checkout"],
        suite=evidence["suite"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    assert blocked["status"] == "receipt_block"
    assert blocked["honest_verdict"].startswith("blocked_pytest_receipt:")


def test_req_report_6589_report_validator_and_atomic_failure_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6589-ATOMIC rejects corrupt blocks and cleans failed temp writes."""

    blocked = exp.blocked_report(
        run_date="20260825",
        status="focused_fixture_failure",
        failed_check="x",
        observed_value=False,
        exp6586_replay={},
        focused_rows=[],
        preconditions={},
        duration_s=0.1,
    )
    corrupt = deepcopy(blocked)
    corrupt.update(
        {
            "inference_substrate": "wrong",
            "verifier_is_oracle": False,
            "verdict_class": "null",
            "honest_verdict": "wrong",
            "gate_check_summary": [],
            "suite_command_receipt": {"ran": True},
        }
    )
    corrupt["reproducibility_checksum"] = exp.artifact_checksum(corrupt)
    errors = exp.validate_report(corrupt)
    assert {
        "blocked_failed_check_missing",
        "blocked_suite_must_not_run",
        "blocked_verdict_class_mismatch",
        "blocked_verdict_prefix_missing",
        "inference_substrate_mismatch",
        "verifier_is_oracle_mismatch",
    }.issubset(errors)

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        exp6586_replay={},
        focused_rows=exp.focused_receipt_fixture_rows(
            evidence["suite"], checkout_root=Path(evidence["suite"]["cwd"])
        ),
        preconditions={"focused_contract_passed": True},
        checkout=evidence["checkout"],
        suite=evidence["suite"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    invalid = deepcopy(report)
    invalid["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        exp.atomic_write_report(tmp_path / "invalid.json", invalid)

    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("fail")))
    with pytest.raises(OSError, match="fail"):
        exp.atomic_write_report(tmp_path / "replace-fail.json", report)
    assert not list(tmp_path.glob(".replace-fail.json.*.tmp"))


def test_req_report_6589_runs_the_actual_smallest_receipt_fixture(tmp_path: Path) -> None:
    """REQ-REPORT-6589-FIXTURES validates the live plugin before any suite run."""

    checkout = REPO
    temporary_root = tmp_path / "focused-owned"
    temporary_root.mkdir()
    result = exp.run_focused_contract(checkout, temporary_root)
    assert result["passed"] is True, result["observed_value"]
    assert result["receipt"]["pytest_receipt_state"] == "complete"
    assert result["receipt"]["terminal_outcome_counts"] == {
        "errored": 0,
        "failed": 1,
        "passed": 1,
        "skipped": 1,
    }


@pytest.mark.parametrize(
    ("failure", "expected_check"),
    [
        ("receipt", "focused_command_receipt"),
        ("exit", "focused_expected_red_exit"),
        ("fields", "focused_field_fixtures"),
        ("incident", "exp6586_missing_sidecar_replay"),
        ("mutation", "focused_mutation_serialization"),
    ],
)
def test_req_report_6589_focused_contract_names_each_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_check: str,
) -> None:
    """REQ-REPORT-6589-FIXTURES reports the first exact failed fixture."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    root = tmp_path / f"owned-{failure}"
    root.mkdir()
    raw = _complete_command_receipt(checkout)
    raw["exit_code"] = 0 if failure == "exit" else 1
    monkeypatch.setattr(exp, "run_owned_command", lambda *_args, **_kwargs: raw)
    merged = _complete_command_receipt(checkout)
    merged["command"] = "focused"
    missing = {**merged, "pytest_receipt_state": "missing", "receipt_errors": []}
    merge_calls = 0

    def fake_merge(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal merge_calls
        merge_calls += 1
        return merged if merge_calls == 1 else missing

    monkeypatch.setattr(exp, "merge_pytest_receipt", fake_merge)
    monkeypatch.setattr(
        exp,
        "validate_command_receipt",
        lambda *_args, **_kwargs: ["bad"] if failure == "receipt" else [],
    )
    base_rows = [{"field": "x", "polarity": "positive", "passed": failure != "fields"}]
    monkeypatch.setattr(exp, "focused_receipt_fixture_rows", lambda *_args, **_kwargs: base_rows)
    if failure == "incident":
        missing["receipt_errors"] = []
    else:
        missing["receipt_errors"] = ["pytest_sidecar_missing"]
    if failure == "mutation":
        monkeypatch.setattr(exp, "tracked_mutation_rows", lambda *_args, **_kwargs: [])
    result = exp.run_focused_contract(checkout, root)
    assert result["passed"] is False
    assert result["failed_check"] == expected_check


@pytest.mark.parametrize("timed_out", [False, True])
def test_req_report_6589_suite_measurement_merges_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timed_out: bool
) -> None:
    """REQ-REPORT-6589-SUITE keeps normal and timeout command receipts."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    raw = _complete_command_receipt(checkout)
    raw["timed_out"] = timed_out
    monkeypatch.setattr(exp, "run_owned_command", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(
        exp,
        "merge_pytest_receipt",
        lambda *_args, **_kwargs: deepcopy(raw),
    )
    receipt = exp.run_suite_measurement(REPO, checkout, tmp_path / "owned")
    assert receipt["mutation_run_id"].startswith("exp6589-suite-")
    if timed_out:
        assert receipt["rows"][0]["outcome"] == "timed_out"


@pytest.mark.parametrize("failure", ["add", "revision", "overlay"])
def test_req_report_6589_isolation_failures_write_terminal_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """REQ-REPORT-6589-CHECKOUT closes each pre-suite isolation failure."""

    active = tmp_path / f"active-{failure}"
    active.mkdir()
    (active / "research-roadmap.yaml").write_text("roadmap", encoding="utf-8")
    conductor = active / "scripts/research_conductor.py"
    conductor.parent.mkdir()
    conductor.write_text("conductor", encoding="utf-8")
    monkeypatch.setattr(exp, "exp6586_failure_replay", lambda _root: {})
    monkeypatch.setattr(
        exp, "dirty_status_receipt", lambda _root: {"sha256": "dirty", "records": []}
    )
    monkeypatch.setattr(exp, "snapshot_tracked_files", lambda _root: {})
    monkeypatch.setattr(exp, "validate_temporary_root", lambda path, _active: path)
    revisions = iter(["a" * 40, "b" * 40] if failure == "revision" else ["a" * 40, "a" * 40])
    monkeypatch.setattr(exp, "git_revision", lambda _root: next(revisions))
    monkeypatch.setattr(exp, "active_dirty_paths", lambda _root: [])
    monkeypatch.setattr(exp, "apply_content_overlay", lambda *_args: [])
    monkeypatch.setattr(exp, "overlay_is_complete", lambda *_args: failure != "overlay")
    monkeypatch.setattr(exp, "operator_curated_snapshot", lambda _root: {})

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        if "add" in args:
            if failure == "add":
                return SimpleNamespace(returncode=1, stdout="", stderr="add failed")
            Path(args[-2] if args[-1] == "a" * 40 else args[-1]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        checkout = Path(args[-1])
        if checkout.exists():
            __import__("shutil").rmtree(checkout)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(exp.subprocess, "run", fake_run)
    report = exp.run_experiment(active, "20260825")
    assert report["status"] == "isolated_environment_block"
    assert report["pytest_receipt_remediation_ready_score"] == 0.0


def test_req_report_6589_main_runs_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6589-ATOMIC exposes the required experiment entry point."""

    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _root, _date: {"status": "measured_red"},
    )
    assert exp.main(["--date", "20260825"]) == 0
    assert "measured_red" in capsys.readouterr().out


def test_req_report_6589_report_validator_covers_structural_attacks(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6589-ATTACKS exercises each report-level refusal."""

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        exp6586_replay={},
        focused_rows=exp.focused_receipt_fixture_rows(
            evidence["suite"], checkout_root=Path(evidence["suite"]["cwd"])
        ),
        preconditions={"focused_contract_passed": True},
        checkout=evidence["checkout"],
        suite=evidence["suite"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    structural = deepcopy(report)
    structural["suite_command_receipt"]["cwd"] = "/wrong"
    structural["disposable_checkout_receipt"].update(
        {
            "active_root": structural["disposable_checkout_receipt"]["checkout_root"],
            "dirty_paths": ["omitted.txt"],
        }
    )
    structural["protected_files_unchanged"]["unchanged"] = False
    structural["reproducibility_checksum"] = exp.artifact_checksum(structural)
    assert {
        "active_root_execution",
        "dirty_overlay_incomplete",
        "protected_file_drift",
        "suite_cwd_not_checkout",
    }.issubset(exp.validate_report(structural))

    bad_prefix = deepcopy(report)
    bad_prefix["honest_verdict"] = "measured without terminal prefix"
    bad_prefix["reproducibility_checksum"] = exp.artifact_checksum(bad_prefix)
    assert "terminal_prefix_missing" in exp.validate_report(bad_prefix)

    false_green = deepcopy(report)
    false_green["status"] = "measured_green"
    false_green["reproducibility_checksum"] = exp.artifact_checksum(false_green)
    assert "false_green" in exp.validate_report(false_green)

    false_timeout = deepcopy(report)
    false_timeout["status"] = "timeout"
    false_timeout["pytest_receipt_remediation_ready_score"] = 1.0
    false_timeout["reproducibility_checksum"] = exp.artifact_checksum(false_timeout)
    assert "timeout_called_complete" in exp.validate_report(false_timeout)


def test_req_report_6589_environment_keeps_safe_external_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6589-RECEIPT keeps external paths but removes parent controls."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    owned = tmp_path / "owned"
    owned.mkdir()
    monkeypatch.setenv("PYTHONPATH", "/tmp/safe-external-pythonpath")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--bad-parent-option")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("COVERAGE_FILE", "/tmp/parent.coverage")
    env, public = exp._effective_environment(checkout, owned, owned / "sidecar.json", "fixture-run")
    assert "/tmp/safe-external-pythonpath" in public["PYTHONPATH"]
    assert env["PYTHONUNBUFFERED"] == public["PYTHONUNBUFFERED"] == "1"
    assert "PYTEST_ADDOPTS" not in env
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in env
    assert "COVERAGE_FILE" not in env


def test_req_report_6589_terminal_validation_failure_keeps_available_suite_evidence(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6589-ATOMIC converts final validation failure into an artifact."""

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        exp6586_replay={},
        focused_rows=exp.focused_receipt_fixture_rows(
            evidence["suite"], checkout_root=Path(evidence["suite"]["cwd"])
        ),
        preconditions={"focused_contract_passed": True},
        checkout=evidence["checkout"],
        suite=evidence["suite"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    blocked = exp.terminal_validation_block(report, ["fabricated_collection_count"])
    assert blocked["status"] == "receipt_validation_block"
    assert blocked["suite_command_receipt"] == report["suite_command_receipt"]
    assert blocked["rows"] == report["rows"]
    assert blocked["pytest_receipt_remediation_ready_score"] == 0.0
    assert blocked["gate_check_summary"][0]["observed_value"] == ["fabricated_collection_count"]
    assert exp.validate_report(blocked) == []


def test_req_report_6589_lost_attempt_recovery_never_invents_raw_fields(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6589-RECEIPT names evidence lost by the failed atomic writer."""

    active = {
        "unchanged": True,
        "tracked_hashes_before_sha256": "sha256:same",
        "tracked_hashes_after_sha256": "sha256:same",
        "dirty_status_before_sha256": "sha256:dirty",
        "dirty_status_after_sha256": "sha256:dirty",
        "preexisting_dirty_status_preserved": True,
    }
    protected = {
        "unchanged": True,
        "before": {path: "sha256:same" for path in exp.PROTECTED_PATHS},
        "after": {path: "sha256:same" for path in exp.PROTECTED_PATHS},
    }
    report = exp.lost_attempt_recovery_report(
        run_date="20260825",
        exp6586_replay={"artifact_sha256": "sha256:source"},
        focused_rows=[{"field": "command", "polarity": "positive", "passed": True}],
        preconditions={"focused_contract_passed": True},
        active_unchanged=active,
        protected=protected,
        tests_run=[],
        duration_s=0.5,
    )
    assert report["status"] == "receipt_validation_block"
    assert report["suite_command_receipt"]["launched"] is True
    assert report["suite_command_receipt"]["raw_receipt_recoverable"] is False
    assert "stdout" in report["suite_command_receipt"]["lost_fields"]
    assert report["disposable_checkout_receipt"]["cleanup"]["removed"] is True
    assert report["suite_truth_baseline"]["state"] == "receipt_validation_block"
    assert exp.validate_report(report) == []


def test_req_report_6589_main_recovery_mode_does_not_launch_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6589-ATOMIC exposes a no-suite recovery for the failed attempt."""

    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    calls: list[tuple[Path, str]] = []

    def fake_recovery(root: Path, run_date: str) -> dict[str, object]:
        calls.append((root, run_date))
        return {"status": "receipt_validation_block"}

    monkeypatch.setattr(exp, "write_failed_attempt_recovery", fake_recovery)
    assert exp.main(["--date", "20260825", "--recover-terminal-validation-block"]) == 0
    assert calls == [(tmp_path, "20260825")]
    assert "receipt_validation_block" in capsys.readouterr().out


def test_req_report_6589_atomic_fallback_writes_terminal_validation_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6589-ATOMIC writes a block when final validation rejects."""

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        exp6586_replay={},
        focused_rows=exp.focused_receipt_fixture_rows(
            evidence["suite"], checkout_root=Path(evidence["suite"]["cwd"])
        ),
        preconditions={"focused_contract_passed": True},
        checkout=evidence["checkout"],
        suite=evidence["suite"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    writes: list[dict[str, object]] = []

    def fake_write(_path: Path, payload: dict[str, object]) -> dict[str, object]:
        writes.append(payload)
        if len(writes) == 1:
            raise ValueError("fabricated_collection_count")
        return {"atomic_replace": True}

    monkeypatch.setattr(exp, "atomic_write_report", fake_write)
    terminal = exp.write_report_with_terminal_fallback(tmp_path / "result.json", report)
    assert len(writes) == 2
    assert terminal["status"] == "receipt_validation_block"
    assert terminal["terminal_validation_failure"]["raw_suite_receipt_recoverable"] is True

    no_suite = exp.blocked_report(
        run_date="20260825",
        status="isolated_environment_block",
        failed_check="git",
        observed_value=False,
        exp6586_replay={},
        focused_rows=[],
        preconditions={},
        duration_s=0.1,
    )
    writes.clear()
    with pytest.raises(ValueError, match="fabricated_collection_count"):
        exp.write_report_with_terminal_fallback(tmp_path / "blocked.json", no_suite)


def test_req_report_6589_recovery_writer_records_resources_without_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6589-RECEIPT writes the one-attempt recovery artifact."""

    active = tmp_path / "active"
    active.mkdir()
    monkeypatch.setattr(exp, "exp6586_failure_replay", lambda _root: {})
    monkeypatch.setattr(
        exp, "dirty_status_receipt", lambda _root: {"sha256": "sha256:dirty", "records": []}
    )
    monkeypatch.setattr(exp, "snapshot_tracked_files", lambda _root: {})
    monkeypatch.setattr(
        exp,
        "_protected_hashes",
        lambda _root: {path: "sha256:same" for path in exp.PROTECTED_PATHS},
    )
    monkeypatch.setattr(exp, "validate_temporary_root", lambda path, _active: path)
    monkeypatch.setattr(
        exp,
        "_resource_preconditions",
        lambda active_root, *_args: {
            "active_root": str(active_root),
            "git_revision": "a" * 40,
        },
    )
    monkeypatch.setattr(
        exp,
        "run_focused_contract",
        lambda *_args: {
            "passed": True,
            "rows": [{"field": "command", "polarity": "positive", "passed": True}],
            "tests_run": [{"command": "focused", "exit_code": 0}],
        },
    )
    written: list[dict[str, object]] = []

    def fake_atomic(_path: Path, report: dict[str, object]) -> dict[str, object]:
        assert exp.validate_report(report) == []
        written.append(report)
        return {"atomic_replace": True}

    monkeypatch.setattr(exp, "atomic_write_report", fake_atomic)
    report = exp.write_failed_attempt_recovery(active, "20260825")
    assert written == [report]
    assert report["preconditions_checked"]["recovery_did_not_launch_suite"] is True
    assert report["preconditions_checked"]["failed_attempt_count"] == 1
    assert report["active_worktree_unchanged"]["unchanged"] is True


def test_req_report_6589_validation_block_requires_an_attempt() -> None:
    """REQ-REPORT-6589-ATOMIC rejects a validation block with no suite attempt."""

    report = exp.blocked_report(
        run_date="20260825",
        status="receipt_validation_block",
        failed_check="terminal_report_validation",
        observed_value=[],
        exp6586_replay={},
        focused_rows=[],
        preconditions={},
        duration_s=0.1,
    )
    assert "validation_block_suite_attempt_missing" in exp.validate_report(report)
